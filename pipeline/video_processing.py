"""Video utilities: resampling and swing window detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np

import config
from pipeline.types import PoseSequence, SwingWindow


@dataclass
class VideoClip:
    frames: List[np.ndarray]  # BGR frames
    fps: float                # target fps after resampling
    original_fps: float       # source fps

    @property
    def duration(self) -> float:
        return len(self.frames) / self.fps if self.frames else 0.0


class VideoProcessor:
    """Video I/O and swing window detection.

    CRITICAL CONTRACT:
    - detect_swing_window() MUST be called on a PoseSequence that is NOT spatial-normalized
      and NOT temporally resampled to N_FRAMES.
    - It should use raw (or minimally interpolated/smoothed) MediaPipe coordinates, in the
      same timeline as the resampled video frames.
    """

    def __init__(self, target_fps: int = config.TARGET_FPS) -> None:
        self.target_fps = target_fps
        self.padding_margin = config.PADDING_MARGIN_FRAMES

    def load_and_resample(self, video_path: Path) -> VideoClip:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS) or float(self.target_fps)
        orig_dt = 1.0 / max(orig_fps, 1e-3)
        target_dt = 1.0 / self.target_fps

        next_sample_time = 0.0
        timestamp = 0.0
        frames: List[np.ndarray] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if timestamp + 1e-6 >= next_sample_time:
                frames.append(frame)
                next_sample_time += target_dt
            timestamp += orig_dt

        cap.release()
        if not frames:
            raise RuntimeError(f"Video has no frames after resampling: {video_path}")

        return VideoClip(frames=frames, fps=float(self.target_fps), original_fps=float(orig_fps))

    def detect_swing_window(self, pose_sequence: PoseSequence) -> SwingWindow:
        """Detect swing window using wrist speed on RAW/MINIMALLY-PROCESSED poses.

        Input pose_sequence should be aligned with resampled video frames, and should not be:
        - spatially normalized (hip-centered / rotated / scaled)
        - temporally resampled to N_FRAMES
        """
        data = pose_sequence.data
        vis = data[:, :, 3]
        fps = pose_sequence.fps
        wrist_indices = (15, 16)
        interp_mask = getattr(pose_sequence, "interpolation_mask", None)
        if interp_mask is None or interp_mask.shape != vis.shape:
            interp_mask = np.zeros_like(vis, dtype=bool)
        else:
            interp_mask = interp_mask.astype(bool, copy=False)

        speeds = np.zeros(len(data), dtype=np.float32)
        for idx in wrist_indices:
            coords = data[:, idx, :3]
            visibility = vis[:, idx]
            joint_interp = interp_mask[:, idx]

            diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
            diffs = np.insert(diffs, 0, 0.0)
            diffs *= fps

            invalid = (visibility < config.THRESHOLDS["pose_visibility"]) | joint_interp
            diffs[invalid] = 0.0  # Ignore interpolated wrists to avoid synthetic speed spikes
            speeds = np.maximum(speeds, diffs)

        max_speed = float(np.max(speeds))
        if max_speed <= 0.0:
            return self._torso_based_window(pose_sequence)

        low_th = 0.1 * max_speed
        peak = int(np.argmax(speeds))

        start = 0
        for i in range(peak, -1, -1):
            if speeds[i] < low_th:
                start = i
                break

        end = len(speeds) - 1
        for i in range(peak, len(speeds)):
            if speeds[i] < low_th:
                end = i
                break

        end = max(end, peak + int(0.5 * fps))

        start = max(0, start - self.padding_margin)
        end = min(len(speeds), end + self.padding_margin)  # end is exclusive

        if end - start < config.N_FRAMES // 2:
            end = min(len(speeds), start + config.N_FRAMES)

        confidence = min(1.0, max_speed)
        return SwingWindow(start_frame=start, end_frame=end, confidence=confidence, method="wrist_velocity")

    def _torso_based_window(self, pose_sequence: PoseSequence) -> SwingWindow:
        data = pose_sequence.data
        fps = pose_sequence.fps

        left_shoulder = data[:, 11, :2]
        right_shoulder = data[:, 12, :2]
        vec = right_shoulder - left_shoulder

        angles = np.degrees(np.arctan2(vec[:, 1], vec[:, 0]))
        angle_speed = np.abs(np.diff(angles, prepend=angles[0])) * fps

        peak = int(np.argmax(angle_speed))
        mx = float(np.max(angle_speed))
        if mx <= 0:
            # fallback: take the middle portion
            start = max(0, peak - int(0.5 * fps))
            end = min(len(data), peak + int(0.8 * fps))
            return SwingWindow(start_frame=start, end_frame=end, confidence=0.1, method="torso_angle")

        threshold = 0.25 * mx
        start = max(0, peak - int(0.5 * fps))
        end = min(len(data), peak + int(0.8 * fps))

        for i in range(peak, -1, -1):
            if angle_speed[i] < threshold:
                start = i
                break
        for i in range(peak, len(angle_speed)):
            if angle_speed[i] < threshold:
                end = i
                break

        start = max(0, start - self.padding_margin)
        end = min(len(data), end + self.padding_margin)

        return SwingWindow(start_frame=start, end_frame=end, confidence=min(1.0, mx), method="torso_angle")

    @staticmethod
    def trim_frames(frames: List[np.ndarray], swing_window: SwingWindow) -> List[np.ndarray]:
        return frames[swing_window.start_frame : swing_window.end_frame]
