"""Pose extraction, interpolation, and normalization utilities."""
from __future__ import annotations

from pathlib import Path
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
    """Xử lý pose extraction và normalization.
    
    Class này thực hiện:
    1. Trích xuất pose landmarks bằng MediaPipe
    2. Interpolation cho các frame bị thiếu
    3. Smoothing bằng Savitzky-Golay filter
    4. Spatial normalization (hip-centered, scaled, aligned)
    5. Handedness normalization (mirror left-handed swings)
    6. Temporal resampling về N_FRAMES cố định
    """
    def __init__(self) -> None:
        self.landmarker = None  # Lazy initialization

    def reset(self) -> None:
        """Reset landmarker để xử lý video mới.
        
        MediaPipe VIDEO mode yêu cầu timestamps tăng dần (monotonic).
        
        MediaPipe VIDEO mode requires monotonically increasing timestamps.
        Call this before processing a new video or re-extracting on trimmed frames.
        """
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def _get_landmarker(self):
        if self.landmarker is not None:
            return self.landmarker

        # đảm bảo model tồn tại
        if not config.POSE_MODEL_PATH.exists():
            raise RuntimeError(
                f"Pose model not found at {config.POSE_MODEL_PATH}. "
                "Please download pose_landmarker_full.task manually."
            )

        base_options = mp_python.BaseOptions(
            model_asset_path=str(config.POSE_MODEL_PATH)
        )
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
            timestamp = int(idx * 1_000_000 / max(fps, 1e-3))
            landmarker = self._get_landmarker()
            result = landmarker.detect_for_video(mp_image, timestamp)
            if not result.pose_landmarks:
                continue
            landmarks = result.pose_landmarks[0]
            for j, lm in enumerate(landmarks):
                pose_array[idx, j, 0] = lm.x
                pose_array[idx, j, 1] = lm.y
                pose_array[idx, j, 2] = lm.z
                pose_array[idx, j, 3] = lm.visibility
            valid_mask[idx] = True

        frame_times = np.arange(num_frames) / max(fps, 1e-3)
        return PoseSequence(
            data=pose_array,
            frame_times=frame_times,
            fps=fps,
            interpolation_mask=interpolation_mask,
            valid_mask=valid_mask,
        )

    def prepare_sequence(self, sequence: PoseSequence, target_frames: int = config.N_FRAMES) -> Tuple[PoseSequence, QualityMetrics]:
        padded, interp_mask = self._interpolate(sequence.data)
        smoothed = self._smooth(padded)
        normalized = self._spatial_normalize(smoothed)
        mirrored = self._normalize_handedness(normalized)
        resampled = self._temporal_resample(mirrored, target_frames)
        mask_resampled = self._resample_mask(interp_mask, target_frames)
        valid_mask = np.max(resampled[:, :, 3], axis=1) >= config.THRESHOLDS["pose_visibility"]
        frame_times = np.linspace(0, (target_frames - 1) / sequence.fps, target_frames)
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
        mask = vis >= config.THRESHOLDS["pose_visibility"]
        interp_mask = np.zeros_like(mask, dtype=bool)
        num_frames = pose.shape[0]
        for joint in range(pose.shape[1]):
            valid_idx = np.where(mask[:, joint])[0]
            if valid_idx.size < 2:
                continue
            missing_idx = np.where(~mask[:, joint])[0]
            for dim in range(3):
                series = pose[:, joint, dim]
                interp_values = np.interp(missing_idx, valid_idx, series[valid_idx])
                series[missing_idx] = interp_values
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
            smoothed = savgol_filter(data, window_length=window, polyorder=poly, axis=0)
            return smoothed
        except Exception:
            kernel = np.ones((5,), dtype=np.float32) / 5.0
            smoothed = data.copy()
            for joint in range(data.shape[1]):
                for dim in range(data.shape[2]):
                    smoothed[:, joint, dim] = np.convolve(
                        data[:, joint, dim], kernel, mode="same"
                    )
            return smoothed

    def _spatial_normalize(self, data: np.ndarray) -> np.ndarray:
        """Chuẩn hóa không gian pose theo từng frame.
        
        Các bước:
        1. Translation: Dịch chuyển mid_hip về gốc tọa độ (0, 0, 0)
        2. Scaling: Chia cho hip_width để normalize kích thước cơ thể
        3. Rotation: Xoay để hip line song song với trục X
        4. Flip: Đảm bảo vai hướng lên (+Y)
        
        Args:
            data: Pose array shape (T, 33, 4)
            
        Returns:
            normalized: Pose đã chuẩn hóa, shape (T, 33, 4)
        """
        normalized = data.copy()
        for t in range(normalized.shape[0]):
            frame = normalized[t]
            
            # Bước 1: Translation - dịch mid_hip về origin
            left_hip = frame[23, :3]
            right_hip = frame[24, :3]
            mid_hip = (left_hip + right_hip) / 2
            frame[:, :3] -= mid_hip
            
            # Bước 2: Scaling - normalize theo hip width
            hip_width = np.linalg.norm(right_hip - left_hip)
            if hip_width < 1e-4:  # Fallback nếu hips quá gần nhau
                shoulder_width = np.linalg.norm(frame[12, :3] - frame[11, :3])
                hip_width = shoulder_width if shoulder_width > 1e-4 else 1.0
            frame[:, :3] /= hip_width
            
            # Bước 3: Rotation - align hip line với trục X
            hip_vec = frame[24, :2] - frame[23, :2]
            hip_angle = np.arctan2(hip_vec[1], hip_vec[0])
            cos_a, sin_a = np.cos(-hip_angle), np.sin(-hip_angle)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            frame[:, :2] = frame[:, :2] @ rot.T
            
            # Bước 4: Flip - đảm bảo shoulders hướng lên (+Y)
            mid_shoulder = (frame[11, :2] + frame[12, :2]) / 2
            if mid_shoulder[1] < 0:
                frame[:, 1] *= -1  # Flip Y
                frame[:, 2] *= -1  # Flip Z để giữ handedness
        return normalized

    def _normalize_handedness(self, data: np.ndarray) -> np.ndarray:
        left_wrist = np.mean(data[:, 15, 0])
        right_wrist = np.mean(data[:, 16, 0])
        mirrored = data.copy()
        if right_wrist > left_wrist:
            mirrored[:, :, 0] *= -1
        return mirrored

    def _temporal_resample(self, data: np.ndarray, target_frames: int) -> np.ndarray:
        num_frames = data.shape[0]
        if num_frames == target_frames:
            return data
        src_x = np.linspace(0, 1, num_frames)
        dst_x = np.linspace(0, 1, target_frames)
        resampled = np.zeros((target_frames, data.shape[1], data.shape[2]), dtype=np.float32)
        for joint in range(data.shape[1]):
            for dim in range(data.shape[2]):
                resampled[:, joint, dim] = np.interp(dst_x, src_x, data[:, joint, dim])
        return resampled

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
            key_vis = np.mean(vis[:, [l_idx, r_idx]])
            keypoint_visibility[name] = float(key_vis)
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
        current = 0
        for is_valid in valid_mask:
            if not is_valid:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest
