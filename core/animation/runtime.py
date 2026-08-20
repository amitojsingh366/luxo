"""Deterministic 120 Hz coordinator for Luxo's semantic motion layers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re

from core.animation import JointVector
from core.animation.gestures import GestureController, SearchingController
from core.animation.layers import IdleLayer, LayerMixer, LayerSample, LightCommand
from core.animation.lookat import LookAtSolver, LookAtTarget
from core.animation.output_stage import OutputStage
from core.animation.poses import PoseLibrary
from core.animation.springs import SpringBank
from core.blackboard import BlackboardSnapshot, ClampCounts
from core.brain.schema import GestureName, LightPattern, LightPreset, PostureName


TICK_HZ = 120
FIXED_DT = 1.0 / TICK_HZ
LOOK_TARGET_MAX_AGE_S = 1.0
SPEECH_BOB_MAX_RAD = 0.025
SPEECH_LIGHT_PULSE_GAIN = 0.18
SPEECH_BLOOM_PULSE_GAIN = 0.10
PUNCTUATION_BLINK_DURATION_S = 0.090
PUNCTUATION_BLINK_HEAD_BOB_RAD = -0.020

# At contemporary epoch values a binary64 timestamp has an approximately
# 0.24 microsecond ULP. One microsecond admits accumulated 1/120 calculations
# without admitting any meaningful cadence discontinuity.
_FIXED_STEP_TOLERANCE_S = 1.0e-6
_OBJECT_TARGET = re.compile(r"obj:[a-z0-9][a-z0-9_-]{0,63}\Z")
_LIGHT_PRESETS = {
    LightPreset.WARM_IDLE: (0.55, 2700),
    LightPreset.WARM_BRIGHT: (1.00, 2900),
    LightPreset.EXCITED_FLASH: (1.15, 3100),
    LightPreset.CURIOUS_FOCUS: (0.95, 3400),
    LightPreset.THINKING_PULSE: (0.70, 2800),
    LightPreset.COOL_DIM: (0.35, 4200),
    LightPreset.SAD_FADE: (0.20, 4500),
}


class AnimationDiscontinuityError(ValueError):
    """Raised when a tick is not exactly one fixed step after its predecessor."""


class InspectionMotion(str, Enum):
    """Body-owned presentation within the closed INSPECTING FSM state."""

    OFF = "off"
    REACH = "inspecting"
    SEARCHING = "searching"


@dataclass(frozen=True, slots=True)
class LookAtFact:
    """Fresh local geometry fact; it is not a model-authored action."""

    target: LookAtTarget
    observed_at: float

    def __post_init__(self) -> None:
        if not isinstance(self.target, LookAtTarget):
            raise TypeError("target must be a LookAtTarget")
        if not isinstance(self.target.target, str):
            raise TypeError("target name must be a string")
        if self.target.target not in {"person", "scene"} and not _OBJECT_TARGET.fullmatch(
            self.target.target
        ):
            raise ValueError("target must be person, scene, or obj:<safe-id>")
        _finite_number("azimuth_rad", self.target.azimuth_rad)
        _finite_number("elevation_rad", self.target.elevation_rad)
        observed_at = _finite_number("observed_at", self.observed_at)
        if observed_at < 0.0:
            raise ValueError("observed_at must be non-negative")


@dataclass(frozen=True, slots=True)
class AnimationSample:
    """Immutable safe output and semantic facts from one animation tick."""

    timestamp: float
    joints: JointVector
    velocities: JointVector
    light: LightCommand
    clamps: ClampCounts
    active_motion: GestureName | PostureName | InspectionMotion | None
    look_target: str | None
    requested_posture: PostureName | None
    speaking: bool
    punctuation_blink: bool


class AnimationRuntime:
    """Compose semantic layers and emit one constrained sample per fixed tick."""

    def __init__(self, poses: PoseLibrary, *, idle_seed: int = 0) -> None:
        if not isinstance(poses, PoseLibrary):
            raise TypeError("poses must be a PoseLibrary")
        if isinstance(idle_seed, bool) or not isinstance(idle_seed, int):
            raise TypeError("idle_seed must be an integer")

        self._poses = poses
        self._idle = IdleLayer(seed=idle_seed)
        self._mixer = LayerMixer()
        self._look_solver = LookAtSolver()
        self._gestures = GestureController(poses)
        self._inspection = GestureController(poses)
        self._searching = SearchingController()
        self._output = OutputStage(SpringBank())

        self._last_now: float | None = None
        self._look_fact: LookAtFact | None = None
        self._pending_motion: GestureName | PostureName | None = None
        self._pending_cancel = False
        self._pending_inspection: InspectionMotion | None = None
        self._posture_name: PostureName | None = PostureName.REST
        self._posture_owned_by_look = False
        self._speech_amplitude = 0.0
        self._speaking = False
        self._blink_pending = False
        self._blink_started_at: float | None = None
        self._light_preset = LightPreset.WARM_IDLE
        self._light_pattern = LightPattern.STEADY
        self._output.reset(self._poses.home)

    @property
    def last_timestamp(self) -> float | None:
        return self._last_now

    @property
    def gesture_in_progress(self) -> bool:
        """Whether a semantic gesture is queued or still animating.

        Postures are intentionally excluded: they are held state, whereas a
        gesture is a finite action whose authored release must finish before
        the behavior plan can advance or announce itself drained.
        """

        return isinstance(self._pending_motion, GestureName) or isinstance(
            self._gestures.active, GestureName
        )

    def start_gesture(self, gesture: GestureName) -> None:
        """Queue a closed-vocabulary gesture for the next fixed tick."""

        if not isinstance(gesture, GestureName):
            raise TypeError("gesture must be a GestureName")
        self._pending_motion = gesture
        self._pending_cancel = False

    def start_posture(self, posture: PostureName) -> None:
        """Queue a closed-vocabulary posture for the next fixed tick."""

        if not isinstance(posture, PostureName):
            raise TypeError("posture must be a PostureName")
        self._pending_motion = posture
        self._pending_cancel = False

    def cancel_motion(self) -> None:
        """Queue a smooth cancellation for the next fixed tick."""

        self._pending_motion = None
        self._pending_cancel = True

    def set_inspection_reach(self, active: bool) -> None:
        """Enter or leave the held, gaze-preserving inspection reach."""

        if not isinstance(active, bool):
            raise TypeError("active must be a boolean")
        self._pending_inspection = (
            InspectionMotion.REACH if active else InspectionMotion.OFF
        )

    def set_inspection_motion(self, motion: InspectionMotion) -> None:
        """Select a body-owned inspection presentation for the next tick."""

        if not isinstance(motion, InspectionMotion):
            raise TypeError("motion must be an InspectionMotion")
        self._pending_inspection = motion

    def set_look_at_target(self, target: LookAtTarget, observed_at: float) -> None:
        """Set a timestamped local target fact without accepting model angles."""

        self._look_fact = LookAtFact(target=target, observed_at=observed_at)

    def clear_look_at_target(self) -> None:
        self._look_fact = None
        self._look_solver.reset()

    def set_light(self, preset: LightPreset, pattern: LightPattern) -> None:
        """Set closed light facts independently of joint motion."""

        if not isinstance(preset, LightPreset):
            raise TypeError("preset must be a LightPreset")
        if not isinstance(pattern, LightPattern):
            raise TypeError("pattern must be a LightPattern")
        self._light_preset = preset
        self._light_pattern = pattern

    def set_speech_amplitude(self, amplitude: float) -> None:
        """Set a local envelope sample, clamped to the normalized range."""

        value = _finite_number("amplitude", amplitude)
        self._speech_amplitude = min(1.0, max(0.0, value))
        self._speaking = True

    def clear_speech_amplitude(self) -> None:
        self._speech_amplitude = 0.0
        self._speaking = False

    def start_punctuation_blink(self) -> None:
        """Queue the body-owned 90 ms punctuation bob for the next tick."""

        self._blink_pending = True

    def cancel_punctuation_blink(self) -> None:
        """Cancel a pending or active punctuation bob."""
        self._blink_pending = False
        self._blink_started_at = None

    def reset(self) -> None:
        """Return all temporal state to the canonical rest boundary."""

        self._last_now = None
        self._look_fact = None
        self._pending_motion = None
        self._pending_cancel = False
        self._pending_inspection = None
        self._posture_name = PostureName.REST
        self._posture_owned_by_look = False
        self._speech_amplitude = 0.0
        self._speaking = False
        self._blink_pending = False
        self._blink_started_at = None
        self._light_preset = LightPreset.WARM_IDLE
        self._light_pattern = LightPattern.STEADY
        self._look_solver.reset()
        self._gestures = GestureController(self._poses)
        self._inspection = GestureController(self._poses)
        self._searching.reset()
        self._output.reset(self._poses.home)

    def tick(self, snapshot: BlackboardSnapshot, now: float) -> AnimationSample:
        """Advance exactly one monotonic 120 Hz sample without blocking or I/O."""

        if not isinstance(snapshot, BlackboardSnapshot):
            raise TypeError("snapshot must be a BlackboardSnapshot")
        timestamp = _finite_number("now", now)
        if timestamp < 0.0:
            raise ValueError("now must be non-negative")
        self._validate_step(timestamp)
        punctuation_blink = self._punctuation_blink_active(timestamp)

        self._apply_pending_motion(timestamp)
        self._apply_pending_inspection(timestamp)
        home = LayerSample(joints=self._poses.home)
        idle = self._idle.sample(snapshot, timestamp)
        inspection_sample = self._inspection.sample(timestamp)
        inspection = LayerSample(joints=inspection_sample.offsets)
        # Inspection changes both upstream pitch joints while the camera/model
        # round trip is in flight. Include that live reach sample in the
        # kinematic foundation so gaze keeps the emitter on the target through
        # anticipation, staggered arrivals, and the held owner silhouette.
        pre_gaze = self._mixer.sum((home, idle, inspection))

        gaze = LayerSample()
        requested_posture: PostureName | None = None
        look_target: str | None = None
        look_fact = self._fresh_look_fact(timestamp)
        if look_fact is not None:
            solution = self._look_solver.solve(
                look_fact.target, current=pre_gaze.joints, dt=FIXED_DT
            )
            gaze = LayerSample(joints=solution.offsets)
            requested_posture = solution.requested_posture
            if self._inspection.active or self._searching.active:
                # Inspection/search already own the whole-body silhouette.
                # A simultaneous look-at crane or stoop would compound those
                # offsets, overextend the reach, and could remain after the
                # observation when a face target sits near a pitch limit.
                requested_posture = None
            look_target = look_fact.target.target
        if requested_posture is None:
            self._release_look_posture(timestamp)
        else:
            self._handoff_look_posture(requested_posture, timestamp)

        gesture_sample = self._gestures.sample(timestamp)
        gesture = LayerSample(joints=gesture_sample.offsets)
        searching_sample = self._searching.sample(timestamp)
        searching = LayerSample(joints=searching_sample.offsets)
        light = LayerSample(light=self._light_command())
        speech = LayerSample(
            joints=JointVector(
                head_pitch=(
                    self._speech_amplitude * SPEECH_BOB_MAX_RAD
                    if self._speaking
                    else 0.0
                )
                + (
                    PUNCTUATION_BLINK_HEAD_BOB_RAD
                    if punctuation_blink
                    else 0.0
                )
            )
        )

        summed = self._mixer.sum(
            (home, idle, gaze, gesture, inspection, searching, light, speech)
        )
        output = self._output.emit(summed.joints, FIXED_DT)
        self._last_now = timestamp
        if summed.light is None:  # Defensive: the fixed light layer is never empty.
            raise RuntimeError("animation light layer was empty")
        return AnimationSample(
            timestamp=timestamp,
            joints=output.positions,
            velocities=output.velocities,
            light=summed.light,
            clamps=output.clamps,
            # Inspection presentations are physical overlays and remain the
            # state cue even while a prior posture settles underneath them.
            active_motion=(
                InspectionMotion.SEARCHING
                if self._searching.active
                else self._inspection.active or self._gestures.active
            ),
            look_target=look_target,
            requested_posture=requested_posture,
            speaking=self._speaking,
            punctuation_blink=punctuation_blink,
        )

    def _validate_step(self, now: float) -> None:
        if self._last_now is None:
            return
        delta = now - self._last_now
        if delta <= 0.0:
            raise ValueError("now must increase monotonically")
        if not math.isclose(
            delta, FIXED_DT, rel_tol=0.0, abs_tol=_FIXED_STEP_TOLERANCE_S
        ):
            raise AnimationDiscontinuityError(
                f"tick delta must be {FIXED_DT:.12f} seconds; reset after a discontinuity"
            )

    def _apply_pending_motion(self, now: float) -> None:
        if self._pending_cancel:
            active = self._gestures.active
            self._gestures.cancel(now)
            if isinstance(active, PostureName):
                self._posture_name = None
                self._posture_owned_by_look = False
        elif self._pending_motion is not None:
            self._gestures.start(self._pending_motion, now)
            if isinstance(self._pending_motion, PostureName):
                self._posture_name = self._pending_motion
                self._posture_owned_by_look = False
            elif self._posture_owned_by_look:
                self._posture_name = None
        self._pending_motion = None
        self._pending_cancel = False

    def _apply_pending_inspection(self, now: float) -> None:
        motion = self._pending_inspection
        if motion is None:
            return
        if motion is InspectionMotion.REACH:
            self._searching.cancel(now)
            self._inspection.start(GestureName.LEAN_IN, now, hold=True)
        elif motion is InspectionMotion.SEARCHING:
            self._inspection.cancel(now)
            self._searching.start(now)
        else:
            self._inspection.cancel(now)
            self._searching.cancel(now)
        self._pending_inspection = None

    def _fresh_look_fact(self, now: float) -> LookAtFact | None:
        fact = self._look_fact
        if fact is None:
            return None
        age = now - fact.observed_at
        if age < 0.0 or age > LOOK_TARGET_MAX_AGE_S:
            self._look_fact = None
            self._look_solver.reset()
            return None
        return fact

    def _punctuation_blink_active(self, now: float) -> bool:
        if self._blink_pending:
            self._blink_pending = False
            self._blink_started_at = now
        if self._blink_started_at is None:
            return False
        if now - self._blink_started_at < PUNCTUATION_BLINK_DURATION_S:
            return True
        self._blink_started_at = None
        return False

    def _handoff_look_posture(
        self, requested: PostureName | None, now: float
    ) -> None:
        if requested is None or requested is self._posture_name:
            return
        if self._gestures.active is not None:
            return
        self._gestures.start(requested, now)
        self._posture_name = requested
        self._posture_owned_by_look = True

    def _release_look_posture(self, now: float) -> None:
        if not self._posture_owned_by_look:
            return
        active = self._gestures.active
        if isinstance(active, GestureName):
            return
        if active is not None and active is not self._posture_name:
            return
        self._gestures.start(PostureName.REST, now)
        self._posture_name = PostureName.REST
        self._posture_owned_by_look = False

    def _light_command(self) -> LightCommand:
        intensity, color_k = _LIGHT_PRESETS[self._light_preset]
        speech_level = self._speech_amplitude if self._speaking else 0.0
        return LightCommand(
            intensity=intensity * (1.0 + speech_level * SPEECH_LIGHT_PULSE_GAIN),
            color_k=color_k,
            pattern=self._light_pattern.value,
            bloom=0.60 + speech_level * SPEECH_BLOOM_PULSE_GAIN,
        )


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


__all__ = [
    "AnimationDiscontinuityError",
    "AnimationRuntime",
    "AnimationSample",
    "FIXED_DT",
    "InspectionMotion",
    "LOOK_TARGET_MAX_AGE_S",
    "LookAtFact",
    "PUNCTUATION_BLINK_DURATION_S",
    "PUNCTUATION_BLINK_HEAD_BOB_RAD",
    "SPEECH_BOB_MAX_RAD",
    "SPEECH_BLOOM_PULSE_GAIN",
    "SPEECH_LIGHT_PULSE_GAIN",
    "TICK_HZ",
]
