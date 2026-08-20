"""Body-authored additive gestures and canonical posture transitions.

The URDF's pi flip makes world -x Luxo's front and positive head pitch look
down. The body has no roll joint, so curiosity is authored as a base-oriented
whole-body crane rather than a synthetic head tilt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping

from ..brain.schema import GestureName, PostureName
from . import JOINT_NAMES, JointName, JointVector
from .poses import PoseLibrary


ANTICIPATION_SECONDS: Final = 0.120
ARRIVAL_STAGGER_SECONDS: Final = 0.060
MOVE_SECONDS: Final = 0.240
STRONG_HOLD_SECONDS: Final = 0.850
BLEND_OUT_SECONDS: Final = 0.220
CANCEL_SECONDS: Final = 0.180
ANTICIPATION_SCALE: Final = 0.16
MASS_ORDER: Final = JOINT_NAMES

# Owner-authored URDF-editor anchors, adapted into a gaze-relative inspection
# layer.  The editor's base endpoint was the hard limit (+2.6 rad) and its
# neck/head values aimed at one fixed camera, so those absolute angles do not
# belong in live animation.  Shoulder and elbow define the silhouette exactly.
# A quarter of the editor's base turn gives the body a readable twist without
# consuming the live solver's yaw range; neck cancels it so emitter azimuth is
# unchanged. After the pi flip, emitter elevation is head minus shoulder and
# elbow, so the head follows the arm-pitch delta with the same sign.
_REFERENCE_ENGAGED_SHOULDER: Final = 0.654
_REFERENCE_ENGAGED_ELBOW: Final = -1.265
_REFERENCE_ENGAGED_BASE: Final = 2.600
_REFERENCE_INSPECTING_BASE: Final = 1.7836
_REFERENCE_INSPECTING_SHOULDER: Final = -0.1686
_REFERENCE_INSPECTING_ELBOW: Final = -0.57425
_INSPECTION_TWIST_SCALE: Final = 0.25

_INSPECTION_BASE_OFFSET: Final = (
    _REFERENCE_INSPECTING_BASE - _REFERENCE_ENGAGED_BASE
) * _INSPECTION_TWIST_SCALE
_INSPECTION_SHOULDER_OFFSET: Final = (
    _REFERENCE_INSPECTING_SHOULDER - _REFERENCE_ENGAGED_SHOULDER
)
_INSPECTION_ELBOW_OFFSET: Final = (
    _REFERENCE_INSPECTING_ELBOW - _REFERENCE_ENGAGED_ELBOW
)
_INSPECTION_NECK_OFFSET: Final = -_INSPECTION_BASE_OFFSET
_INSPECTION_HEAD_OFFSET: Final = (
    _INSPECTION_SHOULDER_OFFSET + _INSPECTION_ELBOW_OFFSET
)

# A missing subject must read as looking around, not as a negative head shake.
# This body-owned layer leaves the engaged arm silhouette alone and searches
# with the two real yaw joints while aiming below the live face-height target.
# Positive head pitch looks down after the URDF's pi flip.  Keeping the bias in
# this additive layer makes it search the nearby work surface without changing
# ordinary engaged gaze or the arm-owned inspection reach.
_SEARCH_DOWNWARD_BIAS_RAD: Final = 0.22
_SEARCH_RIGHT: Final = JointVector(
    base_yaw=0.16,
    neck_yaw=0.24,
    head_pitch=_SEARCH_DOWNWARD_BIAS_RAD - 0.03,
)
_SEARCH_LEFT: Final = JointVector(
    base_yaw=-0.16,
    neck_yaw=-0.24,
    head_pitch=_SEARCH_DOWNWARD_BIAS_RAD + 0.03,
)
SEARCH_HOLD_SECONDS: Final = 0.35


@dataclass(frozen=True, slots=True)
class GestureKeyframe:
    """One authored additive pose and its hold before the next move."""

    offsets: JointVector
    hold_seconds: float


@dataclass(frozen=True, slots=True)
class GestureDefinition:
    keyframes: tuple[GestureKeyframe, ...]


def _frame(
    base: float,
    shoulder: float,
    elbow: float,
    neck: float,
    head: float,
    hold: float,
) -> GestureKeyframe:
    return GestureKeyframe(JointVector(base, shoulder, elbow, neck, head), hold)


# These modest offsets express an eager, puppyish read without approaching the
# URDF's soft limits. Multi-beat gestures anticipate each authored target.
GESTURE_DEFINITIONS: Final[Mapping[GestureName, GestureDefinition]] = (
    MappingProxyType(
        {
            GestureName.PERK_UP: GestureDefinition(
                (_frame(-0.015, -0.18, 0.14, 0.025, -0.18, STRONG_HOLD_SECONDS),)
            ),
            GestureName.NOD: GestureDefinition(
                (
                    _frame(0.008, 0.035, -0.040, -0.010, 0.18, 0.090),
                    _frame(-0.006, -0.025, 0.030, 0.008, -0.08, STRONG_HOLD_SECONDS),
                )
            ),
            GestureName.DOUBLE_TAKE: GestureDefinition(
                (
                    _frame(-0.045, 0.025, -0.025, -0.13, 0.035, 0.080),
                    _frame(0.055, -0.11, 0.09, 0.18, -0.11, STRONG_HOLD_SECONDS),
                )
            ),
            GestureName.RECOIL: GestureDefinition(
                (_frame(-0.035, 0.16, -0.20, -0.055, 0.13, STRONG_HOLD_SECONDS),)
            ),
            GestureName.LEAN_IN: GestureDefinition(
                # Reach toward the owner's inspecting silhouette without
                # disturbing emitter aim. The live gaze layer keeps owning the
                # shade while these offsets are held for the observation.
                (
                    _frame(
                        _INSPECTION_BASE_OFFSET,
                        _INSPECTION_SHOULDER_OFFSET,
                        _INSPECTION_ELBOW_OFFSET,
                        _INSPECTION_NECK_OFFSET,
                        _INSPECTION_HEAD_OFFSET,
                        STRONG_HOLD_SECONDS,
                    ),
                )
            ),
            GestureName.BOUNCE: GestureDefinition(
                (
                    _frame(-0.010, 0.085, -0.11, 0.015, 0.055, 0.070),
                    _frame(0.012, -0.14, 0.12, -0.018, -0.13, STRONG_HOLD_SECONDS),
                )
            ),
            GestureName.SHAKE_NO: GestureDefinition(
                (
                    _frame(-0.035, 0.025, -0.030, -0.18, 0.025, 0.070),
                    _frame(0.040, -0.020, 0.025, 0.20, -0.020, 0.070),
                    _frame(-0.025, 0.015, -0.018, -0.13, 0.015, STRONG_HOLD_SECONDS),
                )
            ),
            GestureName.SETTLE: GestureDefinition(
                (_frame(0.010, 0.065, -0.085, -0.012, 0.070, STRONG_HOLD_SECONDS),)
            ),
            # The light cue belongs elsewhere. Motion is deliberately staged
            # head -> arm -> base, each stage internally mass-staggered.
            GestureName.DROOP: GestureDefinition(
                (
                    _frame(0.003, 0.006, -0.007, 0.005, 0.17, 0.080),
                    _frame(0.006, 0.15, -0.24, 0.009, 0.19, 0.080),
                    _frame(0.045, 0.18, -0.30, 0.015, 0.21, STRONG_HOLD_SECONDS),
                )
            ),
            # Neck/head only refine aim; they do not counterfeit a roll axis.
            GestureName.REGARD: GestureDefinition(
                (_frame(0.070, -0.19, 0.16, 0.025, -0.085, STRONG_HOLD_SECONDS),)
            ),
        }
    )
)


@dataclass(frozen=True, slots=True)
class JointArrival:
    beat: int
    joint: JointName
    seconds: float


@dataclass(frozen=True, slots=True)
class MotionTiming:
    """Relative timing metadata for verification and later orchestration."""

    anticipation_windows: tuple[tuple[float, float], ...]
    arrivals: tuple[JointArrival, ...]
    strong_hold_start: float
    strong_hold_end: float
    duration: float


@dataclass(frozen=True, slots=True)
class GestureSample:
    """One additive layer sample; complete means no motion remains active."""

    offsets: JointVector
    complete: bool


@dataclass(frozen=True, slots=True)
class _Point:
    seconds: float
    value: float


@dataclass(frozen=True, slots=True)
class _Timeline:
    tracks: tuple[tuple[_Point, ...], ...]
    timing: MotionTiming
    terminal: JointVector

    def at(self, seconds: float) -> JointVector:
        return JointVector(*(_track_value(track, seconds) for track in self.tracks))


class UnknownMotionError(ValueError):
    """A requested name is outside the closed gesture/posture vocabulary."""


class GestureController:
    """Deterministically sample one active body-authored motion at a time."""

    __slots__ = (
        "_active",
        "_cancel_target",
        "_hold_terminal",
        "_poses",
        "_settled",
        "_started_at",
        "_timeline",
    )

    def __init__(self, poses: PoseLibrary) -> None:
        if not isinstance(poses, PoseLibrary):
            raise TypeError("poses must be a PoseLibrary")
        self._poses = poses
        self._settled = JointVector()
        self._active: GestureName | PostureName | None = None
        self._timeline: _Timeline | None = None
        self._started_at = 0.0
        self._cancel_target = JointVector()
        self._hold_terminal = False

    @property
    def active(self) -> GestureName | PostureName | None:
        return self._active

    @property
    def timing(self) -> MotionTiming | None:
        return self._timeline.timing if self._timeline is not None else None

    def start(
        self,
        name: GestureName | PostureName | str,
        now: float,
        *,
        hold: bool = False,
    ) -> None:
        """Start a closed-vocabulary motion without discontinuity."""

        checked_now = _time(now)
        if not isinstance(hold, bool):
            raise TypeError("hold must be a boolean")
        current = self._capture(checked_now)
        motion = _motion_name(name)
        if hold and not isinstance(motion, GestureName):
            raise ValueError("only gestures can request an indefinite hold")
        previous_anchor = self._settled
        if self._active is not None:
            previous_anchor = current
            self._settled = current

        if isinstance(motion, GestureName):
            definition = GESTURE_DEFINITIONS[motion]
            frames = tuple(
                GestureKeyframe(_add(current, key.offsets), key.hold_seconds)
                for key in definition.keyframes
            )
            terminal = frames[-1].offsets if hold else current
            self._timeline = _compile(
                current,
                frames,
                terminal,
                release=not hold,
            )
            self._cancel_target = current
            self._hold_terminal = hold
        else:
            target = _subtract(self._poses.pose(motion.value), self._poses.home)
            frames = (GestureKeyframe(target, STRONG_HOLD_SECONDS),)
            self._timeline = _compile(current, frames, target, release=False)
            self._cancel_target = previous_anchor
            self._hold_terminal = False

        self._active = motion
        self._started_at = checked_now

    def cancel(self, now: float) -> None:
        """Ease an active move back to its prior anchor; never snap."""

        checked_now = _time(now)
        if self._active is None:
            return
        current = self._capture(checked_now)
        if self._active is None:
            return
        self._timeline = _release(current, self._cancel_target)
        self._started_at = checked_now
        self._hold_terminal = False

    def sample(self, now: float) -> GestureSample:
        checked_now = _time(now)
        offsets = self._capture(checked_now)
        return GestureSample(offsets, self._active is None)

    def _capture(self, now: float) -> JointVector:
        if self._timeline is None or self._active is None:
            return self._settled
        elapsed = max(0.0, now - self._started_at)
        if elapsed + 1e-12 < self._timeline.timing.duration:
            return self._timeline.at(elapsed)
        if self._hold_terminal:
            return self._timeline.terminal
        self._settled = self._timeline.terminal
        self._active = None
        self._timeline = None
        return self._settled


class SearchingController:
    """Loop a curious engaged-pose search until the observation resolves.

    The first move uses the same anticipation and mass-ordered arrival compiler
    as every authored gesture.  A right-left-right cycle then loops from two
    identical endpoint poses, so even a long cloud round trip has no seam.
    Cancellation blends every participating joint back to zero.
    """

    __slots__ = ("_active", "_canceling", "_started_at", "_timeline")

    def __init__(self) -> None:
        self._active = False
        self._canceling = False
        self._started_at = 0.0
        self._timeline: _Timeline | None = None

    @property
    def active(self) -> bool:
        return self._active

    def start(self, now: float) -> None:
        checked_now = _time(now)
        if self._active and not self._canceling:
            return
        current = self._sample_offset(checked_now)
        frames = tuple(
            GestureKeyframe(target, SEARCH_HOLD_SECONDS)
            for target in (_SEARCH_RIGHT, _SEARCH_LEFT, _SEARCH_RIGHT)
        )
        self._timeline = _compile(
            current,
            frames,
            _SEARCH_RIGHT,
            release=False,
        )
        self._started_at = checked_now
        self._active = True
        self._canceling = False

    def cancel(self, now: float) -> None:
        checked_now = _time(now)
        if not self._active:
            return
        current = self._sample_offset(checked_now)
        self._timeline = _release(current, JointVector())
        self._started_at = checked_now
        self._canceling = True

    def reset(self) -> None:
        self._active = False
        self._canceling = False
        self._started_at = 0.0
        self._timeline = None

    def sample(self, now: float) -> GestureSample:
        checked_now = _time(now)
        offsets = self._sample_offset(checked_now)
        return GestureSample(offsets, not self._active)

    def _sample_offset(self, now: float) -> JointVector:
        timeline = self._timeline
        if not self._active or timeline is None:
            return JointVector()
        elapsed = max(0.0, now - self._started_at)
        duration = timeline.timing.duration
        if self._canceling:
            if elapsed + 1e-12 >= duration:
                self.reset()
                return JointVector()
            return timeline.at(elapsed)

        # Beat zero ends on the right-hand pose. Beats one and two return from
        # right -> left -> right, making that suffix a continuous loop.
        first_beat_end = (
            ANTICIPATION_SECONDS
            + MOVE_SECONDS
            + (len(MASS_ORDER) - 1) * ARRIVAL_STAGGER_SECONDS
            + SEARCH_HOLD_SECONDS
        )
        if elapsed > duration:
            loop_duration = duration - first_beat_end
            elapsed = first_beat_end + (elapsed - first_beat_end) % loop_duration
        return timeline.at(elapsed)


def _compile(
    initial: JointVector,
    frames: tuple[GestureKeyframe, ...],
    terminal: JointVector,
    *,
    release: bool = True,
) -> _Timeline:
    tracks: list[list[_Point]] = [[_Point(0.0, value)] for value in _values(initial)]
    windows: list[tuple[float, float]] = []
    arrivals: list[JointArrival] = []
    previous = initial
    beat_start = 0.0
    strong_start = 0.0
    strong_end = 0.0

    for beat, frame in enumerate(frames):
        anticipation_end = beat_start + ANTICIPATION_SECONDS
        windows.append((beat_start, anticipation_end))
        anticipated_values = _values(_away(previous, frame.offsets))
        frame_values = _values(frame.offsets)
        for index, track in enumerate(tracks):
            track.append(_Point(anticipation_end, anticipated_values[index]))
            arrival = anticipation_end + MOVE_SECONDS + index * ARRIVAL_STAGGER_SECONDS
            track.append(_Point(arrival, frame_values[index]))
            arrivals.append(JointArrival(beat, MASS_ORDER[index], arrival))
        last_stagger = (len(MASS_ORDER) - 1) * ARRIVAL_STAGGER_SECONDS
        strong_start = anticipation_end + MOVE_SECONDS + last_stagger
        strong_end = strong_start + frame.hold_seconds
        for index, track in enumerate(tracks):
            track.append(_Point(strong_end, frame_values[index]))
        previous = frame.offsets
        beat_start = strong_end

    duration = strong_end
    if release:
        for index, track in enumerate(tracks):
            end = strong_end + BLEND_OUT_SECONDS + index * ARRIVAL_STAGGER_SECONDS
            track.append(_Point(end, _values(terminal)[index]))
        duration = strong_end + BLEND_OUT_SECONDS + last_stagger

    timing = MotionTiming(
        tuple(windows), tuple(arrivals), strong_start, strong_end, duration
    )
    return _Timeline(tuple(tuple(track) for track in tracks), timing, terminal)


def _release(initial: JointVector, terminal: JointVector) -> _Timeline:
    tracks = tuple(
        (
            _Point(0.0, start),
            _Point(CANCEL_SECONDS + index * ARRIVAL_STAGGER_SECONDS, end),
        )
        for index, (start, end) in enumerate(
            zip(_values(initial), _values(terminal), strict=True)
        )
    )
    duration = CANCEL_SECONDS + 4 * ARRIVAL_STAGGER_SECONDS
    timing = MotionTiming((), (), 0.0, 0.0, duration)
    return _Timeline(tracks, timing, terminal)


def _track_value(track: tuple[_Point, ...], seconds: float) -> float:
    if seconds <= track[0].seconds:
        return track[0].value
    for left, right in zip(track, track[1:]):
        if seconds <= right.seconds:
            span = right.seconds - left.seconds
            if span <= 0.0:
                return right.value
            phase = (seconds - left.seconds) / span
            smooth = phase**3 * (phase * (phase * 6.0 - 15.0) + 10.0)
            return left.value + (right.value - left.value) * smooth
    return track[-1].value


def _motion_name(name: GestureName | PostureName | str) -> GestureName | PostureName:
    try:
        return GestureName(name)
    except (TypeError, ValueError):
        try:
            return PostureName(name)
        except (TypeError, ValueError) as error:
            raise UnknownMotionError(f"unknown gesture or posture: {name!r}") from error


def _time(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("now must be a finite number")
    return float(value)


def _values(vector: JointVector) -> tuple[float, ...]:
    return tuple(getattr(vector, joint) for joint in JOINT_NAMES)


def _add(left: JointVector, right: JointVector) -> JointVector:
    return JointVector(
        *(a + b for a, b in zip(_values(left), _values(right), strict=True))
    )


def _subtract(left: JointVector, right: JointVector) -> JointVector:
    return JointVector(
        *(a - b for a, b in zip(_values(left), _values(right), strict=True))
    )


def _away(start: JointVector, target: JointVector) -> JointVector:
    return JointVector(
        *(
            a - ANTICIPATION_SCALE * (b - a)
            for a, b in zip(_values(start), _values(target), strict=True)
        )
    )


__all__ = [
    "ANTICIPATION_SECONDS",
    "ARRIVAL_STAGGER_SECONDS",
    "GESTURE_DEFINITIONS",
    "MASS_ORDER",
    "STRONG_HOLD_SECONDS",
    "GestureController",
    "GestureDefinition",
    "GestureKeyframe",
    "GestureSample",
    "JointArrival",
    "MotionTiming",
    "SEARCH_HOLD_SECONDS",
    "SearchingController",
    "UnknownMotionError",
]
