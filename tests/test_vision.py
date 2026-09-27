import math
import numpy as np

from smarfly2.records import VisibleFlyState
from smarfly2.vision import extract_local_features


def fly():
    return VisibleFlyState(x=32, y=48, heading=-math.pi/2, vx=0, vy=0)


def test_brightness_and_contrast_follow_local_image_content():
    dark = np.full((64, 64, 3), 20, np.uint8)
    bright = np.full((64, 64, 3), 220, np.uint8)
    checker = dark.copy()
    checker[::2, ::2] = 240
    checker[1::2, 1::2] = 240
    f_dark = extract_local_features(dark, fly(), None)
    f_bright = extract_local_features(bright, fly(), None)
    f_checker = extract_local_features(checker, fly(), None)
    assert f_bright.brightness > f_dark.brightness
    assert f_checker.contrast > f_dark.contrast


def test_left_right_asymmetry_changes_sign_when_lateral_brightness_swaps():
    a = np.zeros((64, 64, 3), np.uint8)
    b = np.zeros_like(a)
    a[:48, :32] = 255
    b[:48, 32:] = 255
    fa = extract_local_features(a, fly(), None)
    fb = extract_local_features(b, fly(), None)
    assert fa.brightness_lr * fb.brightness_lr < 0
    assert abs(fa.brightness_lr) > 0.05
    assert abs(fb.brightness_lr) > 0.05


def test_motion_features_detect_change_and_lateral_sign():
    prev = np.zeros((64, 64), np.uint8)
    left = np.zeros((64, 64, 3), np.uint8)
    right = np.zeros_like(left)
    left[8:32, 8:30] = 255
    right[8:32, 34:56] = 255
    fl = extract_local_features(left, fly(), prev)
    fr = extract_local_features(right, fly(), prev)
    assert fl.motion > 0
    assert fr.motion > 0
    assert fl.motion_lr * fr.motion_lr < 0
