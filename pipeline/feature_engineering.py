"""Feature extraction, missingness dropout, and dataset export helpers.

Contract:
- Joint features: X_joint (T, 33, 13)
- Global features: X_global (T, 3)
- Label: y (int64), plus band_str for reference
- Interpolation mask saved in a parallel folder, same stem as sequence file.

Key fixes:
- Temporal leakage removed: causal backward differences
- Global metrics are NOT broadcast across joints
- Only augmentation supported: missingness dropout (no interpolation smoothing)
- Scaler fit: train-only, non-augmented, vectorized exclusion of interpolated joints
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


ANGLE_TRIPLETS = {
    13: (11, 13, 15),
    14: (12, 14, 16),
    15: (13, 15, 17),
    16: (14, 16, 18),
    25: (23, 25, 27),
    26: (24, 26, 28),
}


@dataclass
class RunningFeatureStats:
    feature_dim: int
    count: int = 0
    mean: np.ndarray = field(init=False)
    m2: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.mean = np.zeros(self.feature_dim, dtype=np.float64)
        self.m2 = np.zeros(self.feature_dim, dtype=np.float64)

    def update(self, feature_block: np.ndarray) -> None:
        """Update with (N, D) or (..., D)."""
        flat = feature_block.reshape(-1, self.feature_dim).astype(np.float64, copy=False)
        for vec in flat:
            self.count += 1
            delta = vec - self.mean
            self.mean += delta / self.count
            delta2 = vec - self.mean
            self.m2 += delta * delta2

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.count < 2:
            std = np.ones_like(self.mean)
        else:
            variance = self.m2 / (self.count - 1)
            std = np.sqrt(np.maximum(variance, 1e-8))
        return self.mean.astype(np.float32), std.astype(np.float32)


class FeatureEngineer:
    def __init__(self) -> None:
        self.stats_joint = RunningFeatureStats(config.POSE_JOINT_FEATURE_DIM)   # 13
        self.stats_global = RunningFeatureStats(config.POSE_GLOBAL_FEATURE_DIM) # 3
        self.samples: List[SamplePayload] = []
        self.sample_records: List[Dict[str, object]] = []
        
    # ------------------- FEATURES -------------------
    def compute_features(self, pose_sequence: PoseSequence) -> Tuple[np.ndarray, np.ndarray]:
        coords = pose_sequence.data[:, :, :3]
        visibility = pose_sequence.data[:, :, 3:4]
        T = coords.shape[0]
        dt = 1.0 / max(pose_sequence.fps, 1e-3)

        velocity = np.zeros_like(coords)
        if T > 1:
            velocity[1:] = (coords[1:] - coords[:-1]) / dt
            velocity[0] = 0.0  # safer than copying frame 1

        acceleration = np.zeros_like(velocity)
        if T > 1:
            acceleration[1:] = (velocity[1:] - velocity[:-1]) / dt
            acceleration[0] = 0.0

        speed_mag = np.linalg.norm(velocity, axis=-1, keepdims=True)
        accel_mag = np.linalg.norm(acceleration, axis=-1, keepdims=True)

        joint_angles = np.zeros_like(visibility)
        for joint_idx, (a, b, c) in ANGLE_TRIPLETS.items():
            joint_angles[:, joint_idx, 0] = self._calc_angle(coords[:, a], coords[:, b], coords[:, c])

        joint_features = np.concatenate(
            [coords, visibility, velocity, acceleration, speed_mag, accel_mag, joint_angles],
            axis=-1,
        ).astype(np.float32)  # (T,33,13)

        # global (T,3)
        shoulder_vec = coords[:, 12, :2] - coords[:, 11, :2]
        hip_vec = coords[:, 24, :2] - coords[:, 23, :2]
        shoulder_angle = np.degrees(np.arctan2(shoulder_vec[:, 1], shoulder_vec[:, 0]))
        hip_angle = np.degrees(np.arctan2(hip_vec[:, 1], hip_vec[:, 0]))
        x_factor = self._shortest_angle_diff(shoulder_angle, hip_angle)

        mid_shoulder = (coords[:, 11, :3] + coords[:, 12, :3]) / 2
        mid_hip = (coords[:, 23, :3] + coords[:, 24, :3]) / 2
        hip_shoulder_sep = np.linalg.norm(mid_shoulder - mid_hip, axis=-1)

        wrist_speeds = speed_mag[:, [15, 16], 0]
        max_wrist_speed = np.max(wrist_speeds, axis=1)

        global_features = np.stack([x_factor, hip_shoulder_sep, max_wrist_speed], axis=-1).astype(np.float32)
        return joint_features, global_features

    @staticmethod
    def _calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        ba = a - b
        bc = c - b
        norm_ba = np.linalg.norm(ba, axis=-1)
        norm_bc = np.linalg.norm(bc, axis=-1)
        dot = np.sum(ba * bc, axis=-1)
        denom = np.maximum(norm_ba * norm_bc, 1e-6)
        cos_angle = np.clip(dot / denom, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    @staticmethod
    def _shortest_angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = (b - a + 180) % 360 - 180
        return np.abs(diff)

    # ------------------- SAVE -------------------
    def save_sample(
        self,
        video_id: str,
        env: str,
        band: str,
        joint_features: np.ndarray,
        global_features: np.ndarray,
        interpolation_mask: np.ndarray,
        quality: QualityMetrics,
        metadata: Dict[str, float],
        augmented_suffix: str = "",
    ) -> SamplePayload:
        if env not in config.ENVIRONMENTS:
            raise ValueError(f"Unknown env: {env}. Valid: {list(config.ENVIRONMENTS)}")

        if band not in config.BAND_TO_ID:
            raise ValueError(f"Unknown band: {band}. Valid: {list(config.BAND_TO_ID.keys())}")
        band_id = int(config.BAND_TO_ID[band])

        interpolation_ratio = float(np.mean(interpolation_mask))

        suffix = f"__{augmented_suffix}" if augmented_suffix else ""
        sample_name = f"{video_id}{suffix}__{env}__{band}"

        seq_dir = config.OUTPUT_ROOT / config.FINAL_SEQUENCE_SUBDIR
        mask_dir = config.OUTPUT_ROOT / config.INTERPOLATION_MASK_SUBDIR
        seq_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        sequence_path = seq_dir / f"{sample_name}.npz"
        mask_path = mask_dir / f"{sample_name}.npz"

        np.savez_compressed(
            sequence_path,
            X_joint=joint_features.astype(np.float32),
            X_global=global_features.astype(np.float32),
            y=np.array(band_id, dtype=np.int64),
            band_str=np.array(band, dtype=np.str_),
            env=np.array(env, dtype=np.str_),
            video_id=np.array(video_id, dtype=np.str_),
            low_quality=np.array(bool(quality.low_quality)),
            interpolation_ratio=np.array(interpolation_ratio, dtype=np.float32),
            quality_json=np.array(
                json.dumps(
                    {
                        "valid_ratio": quality.valid_ratio,
                        "mean_visibility": quality.mean_visibility,
                        "longest_dropout": quality.longest_dropout,
                    }
                ),
                dtype=np.str_,
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
                "band_id": band_id,
                "sequence_path": str(sequence_path.relative_to(config.BASE_DIR)),
                "interpolation_mask_path": str(mask_path.relative_to(config.BASE_DIR)),
                "low_quality": bool(quality.low_quality),
                "valid_ratio": float(quality.valid_ratio),
                "mean_visibility": float(quality.mean_visibility),
                "longest_dropout": int(quality.longest_dropout),
                "interpolation_ratio": float(interpolation_ratio),
            }
        )
        return payload

    # ------------------- AUGMENTATION: missingness only -------------------
    def augment_missingness_dropout(
        self,
        joint_features: np.ndarray,
        global_features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Missingness modeling dropout.

        - Works on joint_features (which include coords + visibility).
        - For dropped (t,j):
          * visibility := 0
          * xyz := 0  (remove signal; do NOT interpolate)
        - Global features unchanged (optional: you can also zero them if you want stronger corruption).
        """
        p = float(config.AUGMENTATION_CONFIG.get("dropout_prob", 0.0))
        if p <= 0:
            return joint_features, global_features

        out = joint_features.copy()
        # joint_features layout: [x,y,z, vis, vx,vy,vz, ax,ay,az, speed, accel, angle]
        vis = out[:, :, 3]
        mask = (np.random.rand(*vis.shape) < p)

        out[:, :, 3][mask] = 0.0      # visibility
        out[:, :, :3][mask] = 0.0     # remove xyz
        out[:, :, 4:][mask] = 0.0     # remove derived kinematics too (critical)

        g = global_features.copy()

        # joints used by globals:
        # x_factor uses shoulders (11,12) + hips (23,24)
        # hip_shoulder_sep uses shoulders (11,12) + hips (23,24)
        # max_wrist_speed uses wrists (15,16)
        shoulders_ok = (out[:, 11, 3] > 0) & (out[:, 12, 3] > 0)
        hips_ok = (out[:, 23, 3] > 0) & (out[:, 24, 3] > 0)
        wrists_ok = (out[:, 15, 3] > 0) | (out[:, 16, 3] > 0)

        # if missing required joints => zero corresponding global signal
        g[:, 0] = np.where(shoulders_ok & hips_ok, g[:, 0], 0.0)  # x_factor
        g[:, 1] = np.where(shoulders_ok & hips_ok, g[:, 1], 0.0)  # hip_shoulder_sep
        g[:, 2] = np.where(wrists_ok, g[:, 2], 0.0)               # max_wrist_speed

        return out, g

    # ------------------- SCALER -------------------
    def fit_scaler_from_saved_sequences(
        self,
        exclude_low_quality: bool = True,
        interpolation_threshold: float = 0.3,
    ) -> int:
        """Fit joint/global-feature scalers using train-only, non-augmented samples.

        Vectorized exclusion:
        - Joint stats: exclude interpolated (t,j) rows via interpolation_mask.
        - Global stats: exclude timesteps where key joints used by global features were interpolated.
        """
        # Hard guard against leakage by wrong call order
        if any("split" not in r for r in self.sample_records):
            raise RuntimeError(
                "Missing 'split' in sample_records. Call export_splits() before fitting scaler to avoid leakage."
            )

        self.stats_joint = RunningFeatureStats(config.POSE_JOINT_FEATURE_DIM)    # 13
        self.stats_global = RunningFeatureStats(config.POSE_GLOBAL_FEATURE_DIM)  # 3

        used = 0

        # key joints that drive global features:
        # shoulders (11,12), hips (23,24) for x_factor + hip_shoulder_sep
        # wrists (15,16) for max_wrist_speed
        global_key = np.array([11, 12, 23, 24, 15, 16], dtype=int)

        for rec in self.sample_records:
            if rec.get("split") != "train":
                continue
            if rec.get("augmented", False):
                continue
            if exclude_low_quality and rec.get("low_quality", False):
                continue
            if float(rec.get("interpolation_ratio", 0.0)) > interpolation_threshold:
                continue

            seq_path = config.BASE_DIR / Path(str(rec["sequence_path"]))
            if not seq_path.exists():
                continue

            data = np.load(seq_path, allow_pickle=True)
            X_joint = data.get("X_joint")
            X_global = data.get("X_global")
            if X_joint is None:
                continue

            # Get mask path robustly: prefer recorded path
            mask_path_str = rec.get("interpolation_mask_path")
            if mask_path_str:
                mask_path = config.BASE_DIR / Path(str(mask_path_str))
            else:
                mask_path = self._infer_mask_path_from_sequence_path(seq_path)

            if mask_path.exists():
                m = np.load(mask_path, allow_pickle=True)
                interp_mask = m["interpolation_mask"].astype(bool)  # (T,33)

                # ---- Joint stats: exclude interpolated joints ----
                # X_joint: (T,33,13), interp_mask: (T,33)
                valid_joint = X_joint[~interp_mask]  # (K,13)
                if valid_joint.size > 0:
                    self.stats_joint.update(valid_joint)

                # ---- Global stats: exclude timesteps where key joints were interpolated ----
                if X_global is not None:
                    bad_t = np.any(interp_mask[:, global_key], axis=1)  # (T,)
                    good_global = X_global[~bad_t]
                    if good_global.size > 0:
                        self.stats_global.update(good_global)
            else:
                # No mask: include everything
                self.stats_joint.update(X_joint)
                if X_global is not None:
                    self.stats_global.update(X_global)

            used += 1

        return used


    @staticmethod
    def _infer_mask_path_from_sequence_path(seq_path: Path) -> Path:
        # seq_path .../sequences/<name>.npz -> .../interp_masks/<name>.npz
        parts = list(seq_path.parts)
        if config.FINAL_SEQUENCE_SUBDIR in parts:
            idx = parts.index(config.FINAL_SEQUENCE_SUBDIR)
            parts[idx] = config.INTERPOLATION_MASK_SUBDIR
            return Path(*parts)
        # fallback: same folder
        return seq_path.parent / seq_path.name

    def finalize_scaler(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        joint_mean, joint_std = self.stats_joint.finalize()
        global_mean, global_std = self.stats_global.finalize()

        out_dir = config.OUTPUT_ROOT / config.FINAL_FEATURE_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = out_dir / config.SCALER_FILENAME
        with open(scaler_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "joint_mean": joint_mean.tolist(),
                    "joint_std": joint_std.tolist(),
                    "global_mean": global_mean.tolist(),
                    "global_std": global_std.tolist(),
                },
                f,
                indent=2,
            )
        return (joint_mean, joint_std, global_mean, global_std)


    def normalize_saved_sequences(self, joint_mean, joint_std, global_mean, global_std):
        jm = joint_mean.reshape(1, 1, -1)
        js = joint_std.reshape(1, 1, -1)
        gm = global_mean.reshape(1, -1)
        gs = global_std.reshape(1, -1)

        seq_dir = config.OUTPUT_ROOT / config.FINAL_SEQUENCE_SUBDIR
        for npz_path in seq_dir.glob("*.npz"):
            data = np.load(npz_path, allow_pickle=True)
            X_joint = data.get("X_joint")
            X_global = data.get("X_global")
            if X_joint is None:
                continue

            normalized_joint = (X_joint - jm) / js
            payload = {k: data[k] for k in data.files if k not in ("X_joint", "X_global")}

            if X_global is not None:
                normalized_global = (X_global - gm) / gs
                np.savez_compressed(
                    npz_path,
                    X_joint=normalized_joint.astype(np.float32),
                    X_global=normalized_global.astype(np.float32),
                    **payload,
                )
            else:
                np.savez_compressed(
                    npz_path,
                    X_joint=normalized_joint.astype(np.float32),
                    **payload,
                )

    # ------------------- EXPORTS -------------------
    def export_metadata(self) -> Path:
        import pandas as pd

        df = pd.DataFrame(self.sample_records)
        out_dir = config.OUTPUT_ROOT / config.FINAL_METADATA_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "processed_metadata.csv"
        df.to_csv(path, index=False)
        return path

    def export_splits(self) -> Path:
        rng = random.Random(config.RANDOM_SEED)
        base_video_ids = sorted({str(r["video_id"]) for r in self.sample_records if not bool(r.get("augmented", False))})
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
            assigned = split_map.get(str(rec["video_id"]), "train")
            if rec.get("augmented", False):
                rec["split"] = "train"  # Augmented variants must stay in train split
            else:
                rec["split"] = assigned

        out_dir = config.OUTPUT_ROOT / config.FINAL_SPLIT_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        split_path = out_dir / "dataset_splits.json"
        split_path.write_text(json.dumps(split_map, indent=2), encoding="utf-8")
        return split_path