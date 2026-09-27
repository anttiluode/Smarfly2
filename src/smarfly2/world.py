from __future__ import annotations

import math
import cv2
import numpy as np


class SyntheticWorld:
    """Deterministic moving geometry used only for tests/headless smoke runs."""

    def __init__(self, width: int, height: int, seed: int = 0):
        self.width = int(width)
        self.height = int(height)
        rng = np.random.default_rng(seed)
        self.phase = float(rng.uniform(0, 2 * math.pi))
        self.phase2 = float(rng.uniform(0, 2 * math.pi))
        self.base = int(rng.integers(18, 42))
        self.circle_radius = max(4, int(min(width, height) * 0.08))

    def frame(self, t: int) -> np.ndarray:
        illum = self.base + int(15 * math.sin(0.035 * t + self.phase))
        frame = np.full((self.height, self.width, 3), np.clip(illum, 0, 255), dtype=np.uint8)

        cx = int((0.5 + 0.36 * math.sin(0.055 * t + self.phase)) * (self.width - 1))
        cy = int((0.5 + 0.30 * math.cos(0.043 * t + self.phase2)) * (self.height - 1))
        cv2.circle(frame, (cx, cy), self.circle_radius, (220, 220, 220), -1)

        rw = max(6, self.width // 7)
        rh = max(5, self.height // 9)
        rx = int((0.5 + 0.42 * math.sin(0.031 * t + self.phase2)) * max(1, self.width - rw - 1))
        ry = int((0.5 + 0.35 * math.sin(0.047 * t + self.phase)) * max(1, self.height - rh - 1))
        cv2.rectangle(frame, (rx, ry), (min(self.width - 1, rx + rw), min(self.height - 1, ry + rh)), (90, 170, 245), -1)

        stripe_x = (3 * t + int(11 * self.phase)) % self.width
        cv2.line(frame, (stripe_x, 0), (stripe_x, self.height - 1), (60, 60, 100), 2)
        return frame
