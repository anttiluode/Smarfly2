from __future__ import annotations

import math
import cv2
import numpy as np

from .records import VisibleFeatures, VisibleFlyState


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.uint8, copy=False)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _cone_masks(shape: tuple[int, int], fly: VisibleFlyState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    length = max(8.0, 0.45 * min(h, w))
    half_angle = math.pi / 4.0
    p0 = np.array([fly.x, fly.y], dtype=float)
    p1 = p0 + length * np.array([math.cos(fly.heading - half_angle), math.sin(fly.heading - half_angle)])
    p2 = p0 + length * np.array([math.cos(fly.heading + half_angle), math.sin(fly.heading + half_angle)])
    pts = np.round(np.stack([p0, p1, p2])).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 1)

    yy, xx = np.indices((h, w), dtype=float)
    left_axis = np.array([-math.sin(fly.heading), math.cos(fly.heading)])
    lateral = (xx - fly.x) * left_axis[0] + (yy - fly.y) * left_axis[1]
    left = ((mask == 1) & (lateral >= 0)).astype(np.uint8)
    right = ((mask == 1) & (lateral < 0)).astype(np.uint8)
    return mask, left, right


def _mean_where(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask.astype(bool)]
    return float(selected.mean()) if selected.size else 0.0


def extract_local_features(
    frame: np.ndarray,
    fly: VisibleFlyState,
    previous_gray: np.ndarray | None,
) -> VisibleFeatures:
    gray = _gray(frame).astype(np.float32)
    mask, left, right = _cone_masks(gray.shape, fly)
    brightness = _mean_where(gray, mask) / 255.0
    left_b = _mean_where(gray, left) / 255.0
    right_b = _mean_where(gray, right) / 255.0
    brightness_lr = left_b - right_b

    values = gray[mask.astype(bool)]
    contrast = float(values.std() / 128.0) if values.size else 0.0

    if previous_gray is None:
        motion = 0.0
        motion_lr = 0.0
    else:
        prev = _gray(previous_gray).astype(np.float32)
        if prev.shape != gray.shape:
            prev = cv2.resize(prev, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)
        diff = np.abs(gray - prev)
        motion = _mean_where(diff, mask) / 255.0
        motion_lr = _mean_where(diff, left) / 255.0 - _mean_where(diff, right) / 255.0

    return VisibleFeatures(
        brightness=float(np.clip(brightness, 0.0, 1.0)),
        brightness_lr=float(np.clip(brightness_lr, -1.0, 1.0)),
        motion=float(np.clip(motion, 0.0, 1.0)),
        motion_lr=float(np.clip(motion_lr, -1.0, 1.0)),
        contrast=float(max(0.0, contrast)),
    )
