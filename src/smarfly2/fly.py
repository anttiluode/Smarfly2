from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .records import HiddenFlyState, VisibleFeatures, VisibleFlyState


@dataclass(frozen=True)
class FlyParams:
    fast_tau: float = 0.25
    slow_tau: float = 3.0
    adaptation_tau: float = 0.8
    base_speed: float = 42.0
    motion_gain: float = 28.0
    adaptation_cost: float = 8.0
    visual_turn_gain: float = 2.2
    fast_gain: float = 1.4
    slow_gain: float = 0.9
    adaptation_gain: float = 1.1
    turn_limit: float = 4.0
    process_noise: float = 0.025


class Fly:
    def __init__(
        self,
        width: int,
        height: int,
        seed: int = 0,
        params: FlyParams | None = None,
    ):
        self.width = int(width)
        self.height = int(height)
        self.params = params or FlyParams()
        self.rng = np.random.default_rng(seed)
        self.x = self.width / 2.0
        self.y = self.height / 2.0
        self.heading = float(self.rng.uniform(0.0, 2.0 * math.pi))
        self.vx = 0.0
        self.vy = 0.0
        self.fast_trace = 0.0
        self.slow_context = 0.0
        self.adaptation = 0.0

    @staticmethod
    def _alpha(dt: float, tau: float) -> float:
        return 1.0 - math.exp(-max(dt, 0.0) / tau)

    def step(
        self, features: VisibleFeatures, dt: float
    ) -> tuple[VisibleFlyState, HiddenFlyState]:
        if dt <= 0:
            raise ValueError("dt must be positive")
        p = self.params
        sensory = math.tanh(features.brightness_lr + 0.7 * features.motion_lr)

        af = self._alpha(dt, p.fast_tau)
        aslow = self._alpha(dt, p.slow_tau)
        self.fast_trace += af * (sensory - self.fast_trace)
        self.slow_context += aslow * (self.fast_trace - self.slow_context)

        turn_drive = (
            p.visual_turn_gain * sensory
            + p.fast_gain * self.fast_trace
            + p.slow_gain * self.slow_context
            - p.adaptation_gain * self.adaptation
        )
        turn_drive += p.process_noise * float(self.rng.normal())
        turn_rate = float(np.clip(turn_drive, -p.turn_limit, p.turn_limit))

        aa = self._alpha(dt, p.adaptation_tau)
        target_adaptation = math.tanh(turn_rate / max(p.turn_limit, 1e-9))
        self.adaptation += aa * (target_adaptation - self.adaptation)

        self.fast_trace = float(np.clip(self.fast_trace, -1.0, 1.0))
        self.slow_context = float(np.clip(self.slow_context, -1.0, 1.0))
        self.adaptation = float(np.clip(self.adaptation, -1.0, 1.0))

        self.heading = (self.heading + turn_rate * dt) % (2.0 * math.pi)
        speed = p.base_speed + p.motion_gain * max(0.0, features.motion)
        speed -= p.adaptation_cost * abs(self.adaptation)
        speed = max(0.0, speed)
        self.vx = speed * math.cos(self.heading)
        self.vy = speed * math.sin(self.heading)
        self.x = (self.x + self.vx * dt) % self.width
        self.y = (self.y + self.vy * dt) % self.height

        visible = VisibleFlyState(self.x, self.y, self.heading, self.vx, self.vy)
        hidden = HiddenFlyState(self.fast_trace, self.slow_context, self.adaptation)
        return visible, hidden
