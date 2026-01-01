"""Core dataclasses for the golf swing pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np


@dataclass
class PoseSequence:
    """A pose landmark sequence across time.

    data: (T, 33, 4) with [x, y, z, visibility]
    interpolation_mask: (T, 33) True where a joint was interpolated (synthetic)
    valid_mask: (T,) True where at least one pose was detected (or considered usable)
    """
    data: np.ndarray
    frame_times: np.ndarray
    fps: float
    interpolation_mask: np.ndarray
    valid_mask: np.ndarray


@dataclass
class SwingWindow:
    """Swing window [start_frame, end_frame) in the resampled video timeline."""
    start_frame: int
    end_frame: int
    confidence: float
    method: str = "wrist_velocity"

    @property
    def num_frames(self) -> int:
        return max(0, self.end_frame - self.start_frame)


@dataclass
class QualityMetrics:
    """Quality indicators for a processed pose sequence."""
    valid_ratio: float
    mean_visibility: float
    keypoint_visibility: Dict[str, float]
    longest_dropout: int
    low_quality: bool


@dataclass
class SamplePayload:
    """Metadata for a processed sample saved to disk."""
    video_id: str
    env: str
    band: str
    sequence_path: Path
    interpolation_mask_path: Path
    metadata: Dict[str, float] = field(default_factory=dict)
