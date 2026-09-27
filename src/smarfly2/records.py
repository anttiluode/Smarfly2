from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class VisibleFeatures:
    brightness: float
    brightness_lr: float
    motion: float
    motion_lr: float
    contrast: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.brightness, self.brightness_lr, self.motion, self.motion_lr, self.contrast],
            dtype=float,
        )


@dataclass(frozen=True)
class VisibleFlyState:
    x: float
    y: float
    heading: float
    vx: float
    vy: float

    def as_array(self) -> np.ndarray:
        return np.asarray([self.x, self.y, self.heading, self.vx, self.vy], dtype=float)


@dataclass(frozen=True)
class HiddenFlyState:
    fast_trace: float
    slow_context: float
    adaptation: float

    def as_array(self) -> np.ndarray:
        return np.asarray([self.fast_trace, self.slow_context, self.adaptation], dtype=float)


@dataclass(frozen=True)
class FrameRecord:
    t: float
    dt: float
    visible: VisibleFlyState
    features: VisibleFeatures
    hidden: HiddenFlyState

    def observer_vector(self) -> np.ndarray:
        """Visible-only observer input; hidden state is intentionally excluded."""
        return np.concatenate(
            [
                self.visible.as_array(),
                self.features.as_array(),
                np.asarray([self.t, self.dt], dtype=float),
            ]
        )
