"""Body-owned routing from semantic intent to deterministic Luxo animation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import re
from typing import TypeAlias

from core.animation.lookat import LookAtTarget
from core.animation.runtime import (
    FIXED_DT,
    PUNCTUATION_BLINK_DURATION_S,
    AnimationDiscontinuityError,
    AnimationRuntime,
    AnimationSample,
    InspectionMotion,
)
from core.blackboard import BlackboardSnapshot
from core.brain.schema import (
    Action,
    ActionOp,
    GestureName,
    LightPattern,
    LightPreset,
    PostureName,
    SfxName,
)
from core.fsm import BehaviorState, Transition


NOTICE_FREEZE_S = 0.150
# The browser's face-centroid elevation still leaves the shade visually low at
# the neutral camera position. Positive head pitch is down after the URDF's pi
# flip, so this body-owned calibration must be subtracted from tracked targets.
FACE_EYE_LEVEL_RAISE_RAD = 0.20
# Hand landmarks sit visually below the object being presented once the
# browser camera ray is mapped onto the emitter.  Keep the live hand centroid
# as the target and raise only that path by a further 25 percent; face aim is
# already calibrated correctly.
HAND_TARGET_RAISE_RAD = FACE_EYE_LEVEL_RAISE_RAD * 1.25
THINKING_DIP_RAD = math.radians(2.0)
SCAN_DEFAULT_ARC_RAD = 1.20
SCAN_DEFAULT_SPEED_RAD_S = 0.80
SCAN_MIN_ARC_RAD = 0.30
SCAN_MAX_ARC_RAD = 1.50
SCAN_MIN_SPEED_RAD_S = 0.30
SCAN_MAX_SPEED_RAD_S = 1.20
SCAN_ELEVATION_RAD = 0.15

_OBSERVATION_ID = re.compile(r"obs_[1-9][0-9]*\Z")
_STEP_TOLERANCE_S = 1.0e-6
_DEFAULT_LIGHT_PATTERN = {
    LightPreset.WARM_IDLE: LightPattern.STEADY,
    LightPreset.WARM_BRIGHT: LightPattern.STEADY,
    LightPreset.COOL_DIM: LightPattern.STEADY,
    LightPreset.CURIOUS_FOCUS: LightPattern.STEADY,
    LightPreset.THINKING_PULSE: LightPattern.PULSE,
    LightPreset.EXCITED_FLASH: LightPattern.FLICKER,
    LightPreset.SAD_FADE: LightPattern.STEADY,
}

TargetResolver = Callable[[str, BlackboardSnapshot], LookAtTarget | None]


@dataclass(frozen=True, slots=True)
class SfxCue:
    name: SfxName

    def __post_init__(self) -> None:
        if not isinstance(self.name, SfxName):
            raise TypeError("name must be an SfxName")


@dataclass(frozen=True, slots=True)
class CaptureRequest:
    request_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not _OBSERVATION_ID.fullmatch(
            self.request_id
        ):
            raise ValueError("request_id must be obs_<positive integer>")


DirectorEffect: TypeAlias = SfxCue | CaptureRequest


@dataclass(frozen=True, slots=True)
class _Scan:
    arc_rad: float
    speed_rad_s: float
    started_at: float | None = None


class AnimationDirector:
    """Translate closed intent into body-authored runtime commands and facts."""

    def __init__(
        self,
        runtime: AnimationRuntime,
        target_resolver: TargetResolver,
    ) -> None:
        if not isinstance(runtime, AnimationRuntime):
            raise TypeError("runtime must be an AnimationRuntime")
        if not callable(target_resolver):
            raise TypeError("target_resolver must be callable")
        self._runtime = runtime
        self._target_resolver = target_resolver
        self._effects: list[DirectorEffect] = []
        self._desired_target: str | None = None
        self._scan: _Scan | None = None
        self._pending_capture_id: str | None = None
        self._notice_due: float | None = None
        self._droop_due: float | None = None
        self._thinking_dip = False
        self._inspection_motion = InspectionMotion.OFF
        self._nonblink_light = (LightPreset.WARM_IDLE, LightPattern.STEADY)
        self._blink_restore: tuple[LightPreset, LightPattern] | None = None
        self._blink_started_at: float | None = None
        self._last_transition: Transition | None = None
        self._last_tick = runtime.last_timestamp

    @property
    def pending_effects(self) -> tuple[DirectorEffect, ...]:
        return tuple(self._effects)

    def apply_action(
        self,
        action: Action,
        *,
        observation_id: str | None = None,
    ) -> None:
        """Route one validated semantic action without accepting joint values."""

        if not isinstance(action, Action):
            raise TypeError("action must be a validated Action")
        if action.op is ActionOp.OBSERVE:
            capture = CaptureRequest(observation_id)  # type: ignore[arg-type]
            if self._pending_capture_id is not None:
                raise ValueError("an observation capture is already pending")
            if self._scan is None:
                self._effects.append(capture)
            else:
                self._pending_capture_id = capture.request_id
            return
        if observation_id is not None:
            raise ValueError("observation_id is valid only for observe")

        if action.op is ActionOp.GESTURE:
            self._runtime.start_gesture(action.name)  # type: ignore[arg-type]
        elif action.op is ActionOp.POSTURE:
            self._runtime.start_posture(action.name)  # type: ignore[arg-type]
        elif action.op is ActionOp.LIGHT:
            self._set_light(action.preset, action.pattern)  # type: ignore[arg-type]
        elif action.op is ActionOp.SFX:
            self._effects.append(SfxCue(action.name))  # type: ignore[arg-type]
        elif action.op is ActionOp.LOOK_AT:
            self._desired_target = action.target
        elif action.op is ActionOp.SCAN:
            self._scan = _scan_from_action(action)
        # WAIT is deliberately PlanExecutor-owned and creates no director timer.

    def apply_transition(self, transition: Transition) -> bool:
        """Apply a state beat once, returning false for an exact replay."""

        _validate_transition(transition)
        if transition == self._last_transition:
            return False
        if (
            self._last_transition is not None
            and transition.t < self._last_transition.t
        ):
            raise ValueError("transition time must be monotonic")
        self._last_transition = transition
        state = transition.current
        self._thinking_dip = state is BehaviorState.THINKING
        if (
            self._inspection_motion is not InspectionMotion.OFF
            and state is not BehaviorState.INSPECTING
        ):
            self._runtime.set_inspection_motion(InspectionMotion.OFF)
            self._inspection_motion = InspectionMotion.OFF

        if state is BehaviorState.DORMANT:
            self._notice_due = None
            self._droop_due = None
            self._cancel_observation()
            self._desired_target = None
            self._runtime.clear_look_at_target()
            self._runtime.start_posture(PostureName.REST)
            self._set_light(LightPreset.WARM_IDLE)
        elif state is BehaviorState.NOTICING:
            # First turn the body toward the detected viewer. The look-at lock
            # is deliberately delayed until the anticipation beat completes,
            # so engagement reads as rotate, then make direct eye contact.
            self._desired_target = None
            self._runtime.clear_look_at_target()
            self._runtime.start_gesture(GestureName.REGARD)
            self._notice_due = transition.t + NOTICE_FREEZE_S
        elif state is BehaviorState.ENGAGED:
            self._desired_target = "person"
            if transition.previous is BehaviorState.DISENGAGING:
                self._droop_due = None
                self._runtime.start_gesture(GestureName.PERK_UP)
                self._set_light(LightPreset.EXCITED_FLASH)
                self._effects.append(SfxCue(SfxName.BOING))
            else:
                self._set_light(LightPreset.WARM_BRIGHT)
        elif state is BehaviorState.LISTENING:
            self._desired_target = "person"
            self._set_light(LightPreset.WARM_BRIGHT)
        elif state is BehaviorState.THINKING:
            # Owner-directed performance tuning overrides PRD 7.2's glance
            # away: Luxo keeps eye contact and shows thought mainly in light.
            self._desired_target = "person"
            self._set_light(LightPreset.THINKING_PULSE)
            self._effects.append(SfxCue(SfxName.HMM))
        elif state is BehaviorState.SPEAKING:
            self._desired_target = "person"
            self._set_light(LightPreset.WARM_BRIGHT)
        elif state is BehaviorState.INSPECTING:
            # Hold a whole-body reach for the entire camera/model round trip.
            # Keep any explicit target selected by the plan; a direct observe
            # defaults to the live person/hand ray rather than a fixed pose.
            if self._desired_target is None:
                self._desired_target = "person"
            self._runtime.set_inspection_motion(InspectionMotion.REACH)
            self._inspection_motion = InspectionMotion.REACH
            self._set_light(LightPreset.CURIOUS_FOCUS)
        elif state is BehaviorState.DISENGAGING:
            self._notice_due = None
            self._cancel_observation()
            self._desired_target = None
            self._runtime.clear_look_at_target()
            self._set_light(LightPreset.COOL_DIM)
            self._droop_due = transition.t + FIXED_DT
        return True

    def set_inspection_motion(self, motion: InspectionMotion) -> bool:
        """Apply fresh observation presentation evidence to INSPECTING only."""

        if not isinstance(motion, InspectionMotion):
            raise TypeError("motion must be an InspectionMotion")
        inspecting = (
            self._last_transition is not None
            and self._last_transition.current is BehaviorState.INSPECTING
        )
        if motion is not InspectionMotion.OFF and not inspecting:
            return False
        if motion is self._inspection_motion:
            return True
        self._runtime.set_inspection_motion(motion)
        self._inspection_motion = motion
        return True

    def tick(self, snapshot: BlackboardSnapshot, now: float) -> AnimationSample:
        """Advance scheduled body beats and exactly one runtime fixed step."""

        if not isinstance(snapshot, BlackboardSnapshot):
            raise TypeError("snapshot must be a BlackboardSnapshot")
        instant = _finite_time(now)
        self._validate_tick_time(instant)
        self._apply_due_beats(instant)
        self._update_blink(instant)
        target = self._target_for_tick(snapshot, instant)
        if target is None:
            self._runtime.clear_look_at_target()
        else:
            self._runtime.set_look_at_target(target, instant)
        sample = self._runtime.tick(snapshot, instant)
        self._last_tick = instant
        return sample

    def drain_effects(self) -> tuple[DirectorEffect, ...]:
        effects = tuple(self._effects)
        self._effects.clear()
        return effects

    def reset(self) -> None:
        self._runtime.reset()
        self._effects.clear()
        self._desired_target = None
        self._scan = None
        self._pending_capture_id = None
        self._notice_due = None
        self._droop_due = None
        self._thinking_dip = False
        self._inspection_motion = InspectionMotion.OFF
        self._nonblink_light = (LightPreset.WARM_IDLE, LightPattern.STEADY)
        self._blink_restore = None
        self._blink_started_at = None
        self._last_transition = None
        self._last_tick = None

    def _set_light(
        self, preset: LightPreset, pattern: LightPattern | None = None
    ) -> None:
        selected = pattern or _DEFAULT_LIGHT_PATTERN[preset]
        if selected is LightPattern.BLINK:
            self._blink_restore = self._nonblink_light
            if self._nonblink_light[0] is not preset:
                self._blink_restore = (preset, _DEFAULT_LIGHT_PATTERN[preset])
            self._blink_started_at = None
        else:
            self._nonblink_light = (preset, selected)
            self._blink_restore = None
            self._blink_started_at = None
            self._runtime.cancel_punctuation_blink()
        self._runtime.set_light(preset, selected)
        if selected is LightPattern.BLINK:
            self._runtime.start_punctuation_blink()

    def _update_blink(self, now: float) -> None:
        if self._blink_restore is None:
            return
        if self._blink_started_at is None:
            self._blink_started_at = now
            return
        if now - self._blink_started_at >= PUNCTUATION_BLINK_DURATION_S:
            self._runtime.set_light(*self._blink_restore)
            self._nonblink_light = self._blink_restore
            self._blink_restore = None
            self._blink_started_at = None

    def _apply_due_beats(self, now: float) -> None:
        if self._notice_due is not None and now >= self._notice_due:
            self._notice_due = None
            self._desired_target = "person"
            self._set_light(LightPreset.WARM_BRIGHT)
            self._effects.append(SfxCue(SfxName.CHIRP_UP))
        if self._droop_due is not None and now >= self._droop_due:
            self._droop_due = None
            self._runtime.start_gesture(GestureName.DROOP)

    def _cancel_observation(self) -> None:
        self._scan = None
        self._pending_capture_id = None
        self._effects[:] = [
            effect for effect in self._effects if not isinstance(effect, CaptureRequest)
        ]

    def _target_for_tick(
        self,
        snapshot: BlackboardSnapshot,
        now: float,
    ) -> LookAtTarget | None:
        if self._scan is not None:
            scan = self._scan
            if scan.started_at is None:
                scan = _Scan(scan.arc_rad, scan.speed_rad_s, now)
                self._scan = scan
            elapsed = now - scan.started_at
            duration = abs(scan.arc_rad) / scan.speed_rad_s
            if elapsed >= duration:
                self._scan = None
                if self._pending_capture_id is not None:
                    self._effects.append(CaptureRequest(self._pending_capture_id))
                    self._pending_capture_id = None
            else:
                phase = elapsed / duration
                return LookAtTarget(
                    "scene",
                    -scan.arc_rad / 2.0 + scan.arc_rad * phase,
                    SCAN_ELEVATION_RAD,
                )
        if self._desired_target is None:
            return None
        target = self._target_resolver(self._desired_target, snapshot)
        if target is None:
            return None
        if not isinstance(target, LookAtTarget):
            raise TypeError("target_resolver must return LookAtTarget or None")
        if target.target != self._desired_target:
            raise ValueError("resolved target name must match the semantic target")
        return self._adjust_person_elevation(target, snapshot)

    def _adjust_person_elevation(
        self,
        target: LookAtTarget,
        snapshot: BlackboardSnapshot,
    ) -> LookAtTarget:
        """Calibrate face and hand measurements without flattening their motion."""

        if target.target != "person":
            return target
        gaze = snapshot.gaze
        tracking_hand = gaze.hands_present and gaze.hand_conf >= 0.5
        elevation = target.elevation_rad
        # Positive head pitch looks down after the URDF's pi frame flip. Hands
        # need a slightly stronger camera-to-emitter bias than faces;
        # subtracting a constant preserves their measured vertical movement
        # instead of replacing the centroid with a fixed low pose.
        calibration = (
            HAND_TARGET_RAISE_RAD if tracking_hand else FACE_EYE_LEVEL_RAISE_RAD
        )
        elevation -= calibration
        if self._thinking_dip:
            elevation += THINKING_DIP_RAD
        return LookAtTarget(target.target, target.azimuth_rad, elevation)

    def _validate_tick_time(self, now: float) -> None:
        if now < 0.0:
            raise ValueError("now must be non-negative")
        if self._last_tick is None:
            return
        delta = now - self._last_tick
        if delta <= 0.0:
            raise ValueError("now must increase monotonically")
        if not math.isclose(
            delta, FIXED_DT, rel_tol=0.0, abs_tol=_STEP_TOLERANCE_S
        ):
            raise AnimationDiscontinuityError("director tick must advance by 1/120 s")


def _scan_from_action(action: Action) -> _Scan:
    raw_arc = SCAN_DEFAULT_ARC_RAD if action.arc is None else action.arc
    raw_speed = SCAN_DEFAULT_SPEED_RAD_S if action.speed is None else action.speed
    direction = -1.0 if raw_arc < 0.0 else 1.0
    arc = direction * min(max(abs(raw_arc), SCAN_MIN_ARC_RAD), SCAN_MAX_ARC_RAD)
    speed = min(max(raw_speed, SCAN_MIN_SPEED_RAD_S), SCAN_MAX_SPEED_RAD_S)
    return _Scan(arc, speed)


def _finite_time(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("time must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("time must be a finite number")
    return parsed


def _validate_transition(transition: object) -> None:
    if not isinstance(transition, Transition):
        raise TypeError("transition must be a Transition")
    if _finite_time(transition.t) < 0.0:
        raise ValueError("transition time must be non-negative")
    if not isinstance(transition.previous, BehaviorState) or not isinstance(
        transition.current, BehaviorState
    ):
        raise TypeError("transition states must be BehaviorState values")
    if not isinstance(transition.reason, str) or not transition.reason:
        raise ValueError("transition reason must be non-empty")


__all__ = [
    "AnimationDirector", "CaptureRequest", "DirectorEffect", "NOTICE_FREEZE_S",
    "SCAN_DEFAULT_ARC_RAD", "SCAN_DEFAULT_SPEED_RAD_S", "SCAN_MAX_ARC_RAD",
    "SCAN_MAX_SPEED_RAD_S", "SCAN_MIN_ARC_RAD", "SCAN_MIN_SPEED_RAD_S",
    "SfxCue", "TargetResolver",
]
