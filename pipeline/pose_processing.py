"""Pose extraction, interpolation tracking, and normalization utilities.

Key design:
- We expose TWO preparation paths:
  (A) prepare_for_window_detection(): minimal processing for swing window detection
  (B) prepare_sequence(): full processing for feature extraction/training
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from mediapipe import Image as MPImage
from mediapipe import ImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import config
from pipeline.types import PoseSequence, QualityMetrics


class PoseProcessor:
    def __init__(self) -> None:
        self.landmarker = None

    def reset(self) -> None:
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def _get_landmarker(self):
        if self.landmarker is not None:
            return self.landmarker

        if not config.POSE_MODEL_PATH.exists():
            raise RuntimeError(
                f"Pose model not found at {config.POSE_MODEL_PATH}. "
                "Please download pose_landmarker_full.task manually."
            )

        base_options = mp_python.BaseOptions(model_asset_path=str(config.POSE_MODEL_PATH))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_segmentation_masks=False,
            min_tracking_confidence=0.5,
            min_pose_detection_confidence=0.5,
            num_poses=1,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        return self.landmarker

    def extract_sequence(self, frames: list[np.ndarray], fps: float) -> PoseSequence:
        num_frames = len(frames)
        pose_array = np.zeros((num_frames, 33, 4), dtype=np.float32)
        interpolation_mask = np.zeros((num_frames, 33), dtype=bool)
        valid_mask = np.zeros(num_frames, dtype=bool)

        for idx, frame in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)

            timestamp_us = int(idx * 1_000_000 / max(fps, 1e-3))
            result = self._get_landmarker().detect_for_video(mp_image, timestamp_us)

            if not result.pose_landmarks:
                continue

            landmarks = result.pose_landmarks[0]
            for j, lm in enumerate(landmarks):
                pose_array[idx, j, 0] = lm.x
                pose_array[idx, j, 1] = lm.y
                pose_array[idx, j, 2] = lm.z
                pose_array[idx, j, 3] = lm.visibility
            valid_mask[idx] = True

        frame_times = np.arange(num_frames, dtype=np.float32) / max(fps, 1e-3)
        return PoseSequence(
            data=pose_array,
            frame_times=frame_times,
            fps=float(fps),
            interpolation_mask=interpolation_mask,
            valid_mask=valid_mask,
        )

    # ---------- PATH A: for swing window detection ----------
    def prepare_for_window_detection(self, sequence: PoseSequence) -> PoseSequence:
        """Minimal processing to stabilize window detection.

        - Interpolate missing points (track interpolation_mask)
        - Optional light smoothing
        - NO spatial normalization
        - NO handedness mirroring
        - NO temporal resampling
        """
        smoothed = self._smooth(sequence.data)
        interp_mask = np.zeros((smoothed.shape[0], smoothed.shape[1]), dtype=bool)

        valid_mask = np.max(smoothed[:, :, 3], axis=1) >= config.THRESHOLDS["pose_visibility"]
        return PoseSequence(
            data=smoothed,
            frame_times=sequence.frame_times,
            fps=sequence.fps,
            interpolation_mask=interp_mask,
            valid_mask=valid_mask,
        )

    # ---------- PATH B: for training features ----------
    def prepare_sequence(
        self,
        sequence: PoseSequence,
        target_frames: int = config.N_FRAMES,
    ) -> Tuple[PoseSequence, QualityMetrics]:
        """Full processing for training.

        Order:
        1) interpolate (+mask)
        2) smooth
        3) spatial normalize
        4) no handedness mirroring (disabled)
        5) temporal resample to target_frames
        6) resample interpolation mask
        """
        padded, interp_mask = self._interpolate(sequence.data)
        smoothed = self._smooth(padded)
        normalized = self._spatial_normalize(smoothed)
        # Handedness normalization disabled by design (do nothing)
        resampled = self._temporal_resample(normalized, target_frames)
        mask_resampled = self._resample_mask(interp_mask, target_frames)

        valid_mask = np.max(resampled[:, :, 3], axis=1) >= config.THRESHOLDS["pose_visibility"]
        frame_times = np.linspace(0, (target_frames - 1) / sequence.fps, target_frames).astype(np.float32)

        quality = self._quality_metrics(resampled, mask_resampled)
        processed = PoseSequence(
            data=resampled,
            frame_times=frame_times,
            fps=sequence.fps,
            interpolation_mask=mask_resampled,
            valid_mask=valid_mask,
        )
        return processed, quality

    def _interpolate(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pose = data.copy()
        vis = pose[:, :, 3]
        ok = vis >= config.THRESHOLDS["pose_visibility"]

        interp_mask = np.zeros_like(ok, dtype=bool)

        for joint in range(pose.shape[1]):
            valid_idx = np.where(ok[:, joint])[0]
            missing_idx = np.where(~ok[:, joint])[0]
            if valid_idx.size < 2:
                if missing_idx.size > 0:
                    interp_mask[missing_idx, joint] = True  # Flag missing-hard joints for downstream exclusion
                    pose[missing_idx, joint, 3] = 0.0
                continue
            if missing_idx.size == 0:
                continue
            for dim in range(3):
                series = pose[:, joint, dim]
                series[missing_idx] = np.interp(missing_idx, valid_idx, series[valid_idx])
                pose[:, joint, dim] = series

            pose[missing_idx, joint, 3] = config.THRESHOLDS["pose_visibility"]
            interp_mask[missing_idx, joint] = True

        return pose, interp_mask

    def _smooth(self, data: np.ndarray) -> np.ndarray:
        try:
            from scipy.signal import savgol_filter

            window = min(11, data.shape[0] - (1 - data.shape[0] % 2))
            if window < 5:
                return data
            poly = 3 if window >= 5 else 1
            return savgol_filter(data, window_length=window, polyorder=poly, axis=0)
        except Exception:
            kernel = np.ones((5,), dtype=np.float32) / 5.0
            smoothed = data.copy()
            for joint in range(data.shape[1]):
                for dim in range(data.shape[2]):
                    smoothed[:, joint, dim] = np.convolve(data[:, joint, dim], kernel, mode="same")
            return smoothed

    def _spatial_normalize(self, data: np.ndarray) -> np.ndarray:
        normalized = data.copy()
        for t in range(normalized.shape[0]):
            frame = normalized[t]

            left_hip = frame[23, :3]
            right_hip = frame[24, :3]
            mid_hip = (left_hip + right_hip) / 2
            frame[:, :3] -= mid_hip

            hip_width = np.linalg.norm(right_hip - left_hip)
            if hip_width < 1e-4:
                shoulder_width = np.linalg.norm(frame[12, :3] - frame[11, :3])
                hip_width = shoulder_width if shoulder_width > 1e-4 else 1.0
            frame[:, :3] /= hip_width

            hip_vec = frame[24, :2] - frame[23, :2]
            hip_angle = np.arctan2(hip_vec[1], hip_vec[0])
            cos_a, sin_a = np.cos(-hip_angle), np.sin(-hip_angle)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
            frame[:, :2] = frame[:, :2] @ rot.T

            mid_shoulder = (frame[11, :2] + frame[12, :2]) / 2
            if mid_shoulder[1] < 0:
                frame[:, 1] *= -1
                frame[:, 2] *= -1

        return normalized

    def _temporal_resample(self, data: np.ndarray, target_frames: int) -> np.ndarray:
        num_frames = data.shape[0]
        if num_frames == target_frames:
            return data

        src_x = np.linspace(0, 1, num_frames)
        dst_x = np.linspace(0, 1, target_frames)

        out = np.zeros((target_frames, data.shape[1], data.shape[2]), dtype=np.float32)
        for joint in range(data.shape[1]):
            for dim in range(data.shape[2]):
                out[:, joint, dim] = np.interp(dst_x, src_x, data[:, joint, dim])
        return out

    def _resample_mask(self, mask: np.ndarray, target_frames: int) -> np.ndarray:
        num_frames = mask.shape[0]
        if num_frames == target_frames:
            return mask
        src_idx = np.linspace(0, num_frames - 1, target_frames).astype(int)
        return mask[src_idx]

    def _quality_metrics(self, data: np.ndarray, interp_mask: np.ndarray) -> QualityMetrics:
        vis = data[:, :, 3]
        valid_mask = np.max(vis, axis=1) >= config.THRESHOLDS["pose_visibility"]
        valid_ratio = float(np.mean(valid_mask))

        keypoint_visibility = {}
        for name, (l_idx, r_idx) in config.KEY_JOINTS.items():
            keypoint_visibility[name] = float(np.mean(vis[:, [l_idx, r_idx]]))

        mean_visibility = float(np.mean(list(keypoint_visibility.values())))
        longest_dropout = self._longest_dropout(valid_mask)

        low_quality = (
            valid_ratio < config.THRESHOLDS["valid_ratio"]
            or mean_visibility < config.THRESHOLDS["mean_visibility"]
        )

        return QualityMetrics(
            valid_ratio=valid_ratio,
            mean_visibility=mean_visibility,
            keypoint_visibility=keypoint_visibility,
            longest_dropout=longest_dropout,
            low_quality=low_quality,
        )

    @staticmethod
    def _longest_dropout(valid_mask: np.ndarray) -> int:
        longest = 0
        cur = 0
        for ok in valid_mask:
            if not ok:
                cur += 1
                longest = max(longest, cur)
            else:
                cur = 0
        return longest
