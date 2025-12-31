"""Feature extraction, augmentation, and dataset export helpers.

Module này thực hiện bước cuối của pipeline:
1. Trích xuất 16 features cho mỗi keypoint
2. Augmentation (Gaussian, time-warp, jitter, dropout)
3. Lưu samples dưới dạng .npz
4. Tính z-score normalization
5. Export metadata và splits
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import config
from pipeline.types import PoseSequence, QualityMetrics, SamplePayload

# Các triplets (a, b, c) để tính góc khớp tại điểm b
# Ví dụ: elbow angle = góc giữa shoulder-elbow-wrist
ANGLE_TRIPLETS = {
    13: (11, 13, 15),  # left elbow: shoulder(11) - elbow(13) - wrist(15)
    14: (12, 14, 16),  # right elbow
    15: (13, 15, 17),  # left wrist angle: elbow - wrist - index finger
    16: (14, 16, 18),  # right wrist angle
    25: (23, 25, 27),  # left knee: hip - knee - ankle
    26: (24, 26, 28),  # right knee
}


@dataclass
class RunningFeatureStats:
    """Tính toán mean và std theo thuật toán Welford (online algorithm).
    
    Thuật toán này cho phép tính mean và variance một cách hiệu quả
    mà không cần lưu trữ toàn bộ dữ liệu trong bộ nhớ.
    
    Attributes:
        feature_dim: Số chiều của feature vector
        count: Số lượng vector đã xử lý
        mean: Mean tích lũy theo từng chiều
        m2: Tổng bình phương độ lệch (dùng để tính variance)
    """
    feature_dim: int
    count: int = 0
    mean: np.ndarray = field(init=False)
    m2: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        # Khởi tạo mean và m2 với giá trị 0
        self.mean = np.zeros(self.feature_dim, dtype=np.float64)
        self.m2 = np.zeros(self.feature_dim, dtype=np.float64)

    def update(self, feature_block: np.ndarray) -> None:
        """Cập nhật thống kê với batch features mới.
        
        Sử dụng thuật toán Welford để tính mean và variance online:
        - delta = x - mean_old
        - mean_new = mean_old + delta / n
        - delta2 = x - mean_new
        - M2 = M2 + delta * delta2
        
        Args:
            feature_block: Array shape (T, 33, feature_dim) hoặc (N, feature_dim)
        """
        flat = feature_block.reshape(-1, self.feature_dim)
        for vector in flat:
            self.count += 1
            delta = vector - self.mean
            self.mean += delta / self.count
            delta2 = vector - self.mean
            self.m2 += delta * delta2

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Hoàn tất tính toán và trả về mean, std.
        
        Returns:
            Tuple[mean, std] - cả hai đều shape (feature_dim,)
            Nếu count < 2, std = 1 (tránh chia cho 0)
        """
        if self.count < 2:
            # Không đủ dữ liệu để tính variance, dùng std = 1
            std = np.ones_like(self.mean)
        else:
            # Variance = M2 / (n-1) - sử dụng Bessel's correction
            variance = self.m2 / (self.count - 1)
            std = np.sqrt(np.maximum(variance, 1e-8))  # Clip để tránh sqrt âm
        return self.mean.astype(np.float32), std.astype(np.float32)


class FeatureEngineer:
    """Trích xuất features từ pose sequence và quản lý augmentation.
    
    Class này thực hiện:
    1. Tính toán features từ keypoints (velocity, acceleration, angles, metrics)
    2. Lưu trữ samples dưới dạng .npz
    3. Tích lũy thống kê để tính z-score normalization
    4. Augmentation (Gaussian noise, time warp, jitter, dropout)
    
    IMPORTANT: Data Leakage Prevention
    - save_sample() DOES NOT update scaler stats
    - Scaler must be fit AFTER splits are assigned
    - Use fit_scaler_from_saved_sequences() to fit on train-only, non-augmented samples
    - This ensures val/test data does NOT influence normalization parameters
    """
    def __init__(self) -> None:
        self.stats = RunningFeatureStats(config.POSE_FEATURE_DIM)
        self.samples: List[SamplePayload] = []  # Danh sách các sample đã lưu
        self.sample_records: List[Dict[str, object]] = []  # Metadata records

    def compute_features(self, pose_sequence: PoseSequence) -> np.ndarray:
        """Tính toán 16 features cho mỗi keypoint tại mỗi frame.
        
        Features bao gồm:
        - coords (x, y, z): 3 chiều - tọa độ đã normalize
        - visibility: 1 chiều - độ tin cậy của keypoint
        - velocity (vx, vy, vz): 3 chiều - vận tốc (gradient bậc 1)
        - acceleration (ax, ay, az): 3 chiều - gia tốc (gradient bậc 2)
        - speed_mag: 1 chiều - độ lớn vận tốc
        - accel_mag: 1 chiều - độ lớn gia tốc
        - joint_angles: 1 chiều - góc khớp (elbow, knee, wrist)
        - global_metrics: 3 chiều - X-factor, hip-shoulder separation, max wrist speed
        
        Args:
            pose_sequence: PoseSequence đã qua normalize (T, 33, 4)
            
        Returns:
            features: Array shape (T, 33, 16)
        """
        # Tách tọa độ và visibility từ pose data
        coords = pose_sequence.data[:, :, :3]  # (T, 33, 3) - x, y, z
        visibility = pose_sequence.data[:, :, 3:4]  # (T, 33, 1)
        
        # Tính delta time giữa các frame
        dt = 1.0 / max(pose_sequence.fps, 1e-3)
        
        # Tính vận tốc và gia tốc bằng gradient
        # edge_order=2 cho kết quả chính xác hơn ở biên
        edge_order = 2 if coords.shape[0] > 2 else 1
        velocity = np.gradient(coords, axis=0, edge_order=edge_order) / dt  # (T, 33, 3)
        acceleration = np.gradient(velocity, axis=0, edge_order=edge_order) / dt  # (T, 33, 3)
        
        # Độ lớn vận tốc và gia tốc
        speed_mag = np.linalg.norm(velocity, axis=-1, keepdims=True)  # (T, 33, 1)
        accel_mag = np.linalg.norm(acceleration, axis=-1, keepdims=True)  # (T, 33, 1)
        
        # Tính góc khớp cho elbow, knee, wrist (sử dụng 3 điểm liên tiếp)
        joint_angles = np.zeros_like(visibility)  # (T, 33, 1)
        for joint_idx, (a, b, c) in ANGLE_TRIPLETS.items():
            # Góc tại điểm b, tạo bởi vector ba và bc
            joint_angles[:, joint_idx, 0] = self._calc_angle(coords[:, a], coords[:, b], coords[:, c])
        
        # Global metrics - need to be (T, 1) then broadcast to (T, 33, 1)
        shoulder_vec = coords[:, 12, :2] - coords[:, 11, :2]  # (T, 2)
        hip_vec = coords[:, 24, :2] - coords[:, 23, :2]  # (T, 2)
        shoulder_angle = np.degrees(np.arctan2(shoulder_vec[:, 1], shoulder_vec[:, 0]))  # (T,)
        hip_angle = np.degrees(np.arctan2(hip_vec[:, 1], hip_vec[:, 0]))  # (T,)
        x_factor = self._shortest_angle_diff(shoulder_angle, hip_angle)  # (T,)
        
        mid_shoulder = (coords[:, 11, :3] + coords[:, 12, :3]) / 2  # (T, 3)
        mid_hip = (coords[:, 23, :3] + coords[:, 24, :3]) / 2  # (T, 3)
        hip_shoulder_sep = np.linalg.norm(mid_shoulder - mid_hip, axis=-1)  # (T,)
        
        # Wrist speed: take max of left/right wrist speeds - need to squeeze properly
        wrist_speeds = speed_mag[:, [15, 16], 0]  # (T, 2) - remove last dim
        wrist_speed = np.max(wrist_speeds, axis=1)  # (T,)
        
        # Stack global metrics: all are (T,) -> stack to (T, 3) -> expand to (T, 1, 3) -> repeat to (T, 33, 3)
        global_metrics = np.stack([x_factor, hip_shoulder_sep, wrist_speed], axis=-1)  # (T, 3)
        global_metrics = np.repeat(global_metrics[:, None, :], coords.shape[1], axis=1)  # (T, 33, 3)
        
        features = np.concatenate(
            [
                coords,          # (T, 33, 3)
                visibility,      # (T, 33, 1)
                velocity,        # (T, 33, 3)
                acceleration,    # (T, 33, 3)
                speed_mag,       # (T, 33, 1)
                accel_mag,       # (T, 33, 1)
                joint_angles,    # (T, 33, 1)
                global_metrics,  # (T, 33, 3)
            ],
            axis=-1,
        )
        # Total: 3+1+3+3+1+1+1+3 = 16 features per joint
        return features.astype(np.float32)

    @staticmethod
    def _calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Tính góc tại điểm b giữa 3 điểm a-b-c.
        
        Sử dụng công thức dot product:
        cos(θ) = (ba · bc) / (|ba| * |bc|)
        
        Args:
            a, b, c: Arrays shape (..., 3) - tọa độ 3D của 3 điểm
            
        Returns:
            Góc tính bằng độ (degrees), shape (...)
        """
        ba = a - b  # Vector từ b tới a
        bc = c - b  # Vector từ b tới c
        
        # Tính độ dài vector
        norm_ba = np.linalg.norm(ba, axis=-1)
        norm_bc = np.linalg.norm(bc, axis=-1)
        
        # Tính dot product
        dot = np.sum(ba * bc, axis=-1)
        
        # Tránh chia cho 0
        denom = np.maximum(norm_ba * norm_bc, 1e-6)
        
        # Tính cos và clip về [-1, 1] để tránh lỗi số học
        cos_angle = np.clip(dot / denom, -1.0, 1.0)
        
        # Chuyển sang độ
        return np.degrees(np.arccos(cos_angle))

    @staticmethod
    def _shortest_angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Tính hiệu góc ngắn nhất giữa 2 góc (xử lý wrap-around).
        
        Ví dụ: diff(350°, 10°) = 20° (không phải 340°)
        
        Args:
            a, b: Góc tính bằng độ
            
        Returns:
            Hiệu góc nhỏ nhất (luôn dương)
        """
        diff = (b - a + 180) % 360 - 180
        return np.abs(diff)

    def save_sample(
        self,
        video_id: str,
        env: str,
        band: str,
        features: np.ndarray,
        interpolation_mask: np.ndarray,
        quality: QualityMetrics,
        metadata: Dict[str, float],
        augmented_suffix: str = "",
    ) -> SamplePayload:
        """Lưu một sample (features + metadata) vào disk dưới dạng .npz.
        
        Mỗi sample được lưu thành 2 files:
        1. sequences/{name}.npz: Chứa features (X), label (y), và metadata
        2. interp_masks/{name}.npz: Chứa interpolation mask
        
        Sample name format: {video_id}__{aug}__{env}__{band}
        Ví dụ: "Backside-8748-3__gaussian__indoor__8_10.npz"
        
        Args:
            video_id: ID của video gốc
            env: Environment (indoor/outdoor)
            band: Handicap band (target label)
            features: Feature array shape (100, 33, 16)
            interpolation_mask: Bool mask shape (100, 33)
            quality: QualityMetrics của sample
            metadata: Dict chứa thông tin bổ sung
            augmented_suffix: Tên augmentation strategy ("gaussian", "jitter", etc.)
            
        Returns:
            SamplePayload chứa metadata và paths
        """
        # Tạo tên file với suffix nếu là augmented
        suffix = f"__{augmented_suffix}" if augmented_suffix else ""
        sample_name = f"{video_id}{suffix}__{env}__{band}"
        
        # Đường dẫn lưu files
        sequence_path = config.OUTPUT_ROOT / config.FINAL_SEQUENCE_SUBDIR / f"{sample_name}.npz"
        mask_path = config.OUTPUT_ROOT / config.INTERPOLATION_MASK_SUBDIR / f"{sample_name}.npz"
        
        # Lưu features và metadata
        np.savez_compressed(
            sequence_path,
            X=features.astype(np.float32),  # Features: (100, 33, 16)
            y=np.array(band),               # Label: handicap band
            env=np.array(env),              # Environment
            video_id=np.array(video_id),    # Video ID
            low_quality=np.array(quality.low_quality),  # Quality flag
            quality=json.dumps(             # Quality metrics as JSON string
                {
                    "valid_ratio": quality.valid_ratio,
                    "mean_visibility": quality.mean_visibility,
                    "longest_dropout": quality.longest_dropout,
                }
            ),
        )
        np.savez_compressed(mask_path, interpolation_mask=interpolation_mask.astype(np.bool_))
        payload = SamplePayload(
            video_id=video_id + suffix,
            env=env,
            band=band,
            sequence_path=sequence_path,
            interpolation_mask_path=mask_path,
            metadata=metadata,
        )
        self.samples.append(payload)
        self.sample_records.append(
            {
                "video_id": video_id,
                "augmented": bool(augmented_suffix),
                "env": env,
                "band": band,
                "sequence_path": str(sequence_path.relative_to(config.BASE_DIR)),
                "low_quality": quality.low_quality,
                "valid_ratio": quality.valid_ratio,
                "mean_visibility": quality.mean_visibility,
                "longest_dropout": quality.longest_dropout,
            }
        )
        # DO NOT update stats here - scaler must be fit AFTER splits are known
        # to avoid data leakage (val/test should not influence normalization)
        return payload

    def augment_sequence(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        aug = {}
        aug["gaussian"] = self._gaussian_noise(data)
        aug["timewarp"] = self._time_warp(data)
        aug["jitter"] = self._temporal_jitter(data)
        aug["dropout"] = self._dropout_interpolate(data)
        return aug

    def fit_scaler_from_saved_sequences(self, exclude_low_quality: bool = True) -> int:
        """Fit scaler from saved .npz files using ONLY train split, non-augmented samples.
        
        This must be called AFTER export_splits() so that sample_records have 'split' field.
        
        WHY: Fitting scaler on train-only prevents data leakage. If we include val/test
        samples in computing mean/std, we leak information about the validation/test
        distributions into the training process, making evaluation optimistically biased.
        
        Args:
            exclude_low_quality: If True, exclude low_quality samples from scaler fitting
            
        Returns:
            Number of samples used for fitting (for logging/verification)
        """
        # Reset stats to start fresh
        self.stats = RunningFeatureStats(config.POSE_FEATURE_DIM)
        
        count = 0
        for rec in self.sample_records:
            # Filter: train only, non-augmented only
            if rec.get("split") != "train":
                continue
            if rec.get("augmented", False):
                continue
            if exclude_low_quality and rec.get("low_quality", False):
                continue
            
            # Load saved sequence
            seq_path = config.BASE_DIR / Path(str(rec["sequence_path"]))
            if not seq_path.exists():
                continue
            
            data = np.load(seq_path, allow_pickle=True)
            X = data["X"]  # Shape: (100, 33, 16)
            
            # Update running statistics
            self.stats.update(X)
            count += 1
        
        return count

    def finalize_scaler(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute final mean/std and save to JSON.
        
        Must be called AFTER fit_scaler_from_saved_sequences().
        """
        mean, std = self.stats.finalize()
        scaler_path = config.OUTPUT_ROOT / config.FINAL_FEATURE_SUBDIR / config.SCALER_FILENAME
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_path, "w", encoding="utf-8") as f:
            json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)
        return mean, std

    def normalize_saved_sequences(self, mean: np.ndarray, std: np.ndarray) -> None:
        reshaped_mean = mean.reshape(1, 1, -1)
        reshaped_std = std.reshape(1, 1, -1)
        for npz_path in (config.OUTPUT_ROOT / config.FINAL_SEQUENCE_SUBDIR).glob("*.npz"):
            data = np.load(npz_path, allow_pickle=True)
            X = data["X"]
            normalized = (X - reshaped_mean) / reshaped_std
            payload = {key: data[key] for key in data.files if key != "X"}
            np.savez_compressed(npz_path, X=normalized.astype(np.float32), **payload)

    def export_metadata(self) -> Path:
        import pandas as pd
        import time

        df = pd.DataFrame(self.sample_records)
        metadata_path = config.OUTPUT_ROOT / config.FINAL_METADATA_SUBDIR / "processed_metadata.csv"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to save, with retry logic for locked files
        for attempt in range(3):
            try:
                df.to_csv(metadata_path, index=False)
                return metadata_path
            except PermissionError:
                if attempt < 2:
                    print(f"File locked, retrying in 1s... (attempt {attempt + 1}/3)")
                    time.sleep(1)
                else:
                    # Try alternative filename
                    alt_path = metadata_path.with_name("processed_metadata_new.csv")
                    df.to_csv(alt_path, index=False)
                    print(f"Warning: Could not write to {metadata_path.name}, saved to {alt_path.name} instead.")
                    return alt_path
        return metadata_path

    def export_splits(self) -> Path:
        rng = random.Random(config.RANDOM_SEED)
        base_video_ids = sorted({
            str(rec["video_id"])
            for rec in self.sample_records
            if bool(rec["augmented"]) is False
        })
        rng.shuffle(base_video_ids)
        total = len(base_video_ids)
        train_cut = int(total * config.SPLIT_RATIOS["train"])
        val_cut = train_cut + int(total * config.SPLIT_RATIOS["val"])
        split_map: Dict[str, str] = {}
        for idx, vid in enumerate(base_video_ids):
            if idx < train_cut:
                split_map[vid] = "train"
            elif idx < val_cut:
                split_map[vid] = "val"
            else:
                split_map[vid] = "test"
        for rec in self.sample_records:
            video_id = str(rec["video_id"])
            rec["split"] = split_map.get(video_id, "train")
        split_path = config.OUTPUT_ROOT / config.FINAL_SPLIT_SUBDIR / "dataset_splits.json"
        split_path.write_text(json.dumps(split_map, indent=2), encoding="utf-8")
        return split_path

    @staticmethod
    def _gaussian_noise(data: np.ndarray) -> np.ndarray:
        noisy = data.copy()
        noise = np.random.normal(0, config.AUGMENTATION_CONFIG["gaussian_std"], size=data[:, :, :3].shape)
        noisy[:, :, :3] += noise
        return noisy

    @staticmethod
    def _time_warp(data: np.ndarray) -> np.ndarray:
        warp_pct = config.AUGMENTATION_CONFIG["time_warp_pct"]
        factor = 1 + np.random.uniform(-warp_pct, warp_pct)
        num_frames = data.shape[0]
        new_len = int(num_frames * factor)
        new_len = max(10, new_len)
        src_x = np.linspace(0, 1, num_frames)
        dst_x = np.linspace(0, 1, new_len)
        warped = np.zeros((new_len, data.shape[1], data.shape[2]), dtype=data.dtype)
        for joint in range(data.shape[1]):
            for dim in range(data.shape[2]):
                warped[:, joint, dim] = np.interp(dst_x, src_x, data[:, joint, dim])
        final = np.zeros_like(data)
        resample_x = np.linspace(0, 1, final.shape[0])
        warped_x = np.linspace(0, 1, warped.shape[0])
        for joint in range(data.shape[1]):
            for dim in range(data.shape[2]):
                final[:, joint, dim] = np.interp(resample_x, warped_x, warped[:, joint, dim])
        return final

    @staticmethod
    def _temporal_jitter(data: np.ndarray) -> np.ndarray:
        max_jitter = int(config.AUGMENTATION_CONFIG["temporal_jitter_frames"])
        jitter = np.random.randint(-max_jitter, max_jitter + 1)
        if jitter == 0:
            return data
        if jitter > 0:
            padded = np.concatenate([np.repeat(data[[0]], jitter, axis=0), data[:-jitter]], axis=0)
        else:
            jitter = abs(jitter)
            padded = np.concatenate([data[jitter:], np.repeat(data[[-1]], jitter, axis=0)], axis=0)
        return padded

    @staticmethod
    def _dropout_interpolate(data: np.ndarray) -> np.ndarray:
        dropout_prob = config.AUGMENTATION_CONFIG["dropout_prob"]
        dropped = data.copy()
        mask = np.random.rand(*data[:, :, 3].shape) < dropout_prob
        dropped[:, :, 3][mask] = 0.0
        for joint in range(data.shape[1]):
            valid_idx = np.where(~mask[:, joint])[0]
            if valid_idx.size < 2:
                continue
            missing_idx = np.where(mask[:, joint])[0]
            for dim in range(3):
                dropped[:, joint, dim][missing_idx] = np.interp(
                    missing_idx, valid_idx, dropped[:, joint, dim][valid_idx]
                )
        return dropped
