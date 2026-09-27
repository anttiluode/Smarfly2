"""Smarfly2 artificial ethology observer bench."""

from .fly import Fly, FlyParams
from .records import FrameRecord, HiddenFlyState, VisibleFeatures, VisibleFlyState

__all__ = [
    "Fly",
    "FlyParams",
    "FrameRecord",
    "HiddenFlyState",
    "VisibleFeatures",
    "VisibleFlyState",
]
