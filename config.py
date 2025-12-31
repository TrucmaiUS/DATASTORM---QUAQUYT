"""Centralized configuration for the golf swing sequence pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "Public Test"
OUTPUT_ROOT = BASE_DIR / "processed_videos"
MODELS_DIR = BASE_DIR / "models"
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_full.task"

VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi")
ENVIRONMENTS: Tuple[str, ...] = ("indoor", "outdoor")
BANDS: Tuple[str, ...] = ("1_2", "2_4", "4_6", "6_8", "8_10")

ENVIRONMENT_FOLDER_MAP: Dict[str, str] = {
    "Trong nhà - Indoor": "indoor",
    "Ngoài trời - Outdoor": "outdoor",
    "Indoor": "indoor",
    "Outdoor": "outdoor",
}

BAND_FOLDER_MAP: Dict[str, str] = {
    "Band 1-2": "1_2",
    "Band 2-4": "2_4",
    "Band 4-6": "4_6",
    "Band 6-8": "6_8",
    "Band 8-10": "8_10",
}

# Video / sequence parameters
TARGET_FPS: int = 30
N_FRAMES: int = 100
STRIDE: int = 1
PADDING_MARGIN_FRAMES: int = 5

# Thresholds for detectors and quality gates
THRESHOLDS: Dict[str, float] = {
    "yolo_conf": 0.35,
    "pose_visibility": 0.5,
    "valid_ratio": 0.85,
    "mean_visibility": 0.6,
}

# Pose-specific constants
POSE_FEATURE_DIM: int = 16  # x,y,z,vis + velocity + acceleration + angles + metrics
KEY_JOINTS = {
    "hips": (23, 24),
    "shoulders": (11, 12),
    "elbows": (13, 14),
    "wrists": (15, 16),
    "knees": (25, 26),
    "ankles": (27, 28),
}

AUGMENTATION_CONFIG: Dict[str, float] = {
    "gaussian_std": 0.01,
    "time_warp_pct": 0.05,
    "temporal_jitter_frames": 2,
    "dropout_prob": 0.05,
}

SPLIT_RATIOS: Dict[str, float] = {"train": 0.7, "val": 0.15, "test": 0.15}
RANDOM_SEED: int = 42

FINAL_SEQUENCE_SUBDIR = "sequences"
FINAL_METADATA_SUBDIR = "metadata"
FINAL_FEATURE_SUBDIR = "features"
FINAL_SPLIT_SUBDIR = "splits"
SCALER_FILENAME = "feature_scaler.json"
INTERPOLATION_MASK_SUBDIR = "interp_masks"


def ensure_directories() -> None:
    """Create output folders required by the pipeline."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / FINAL_SEQUENCE_SUBDIR).mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / FINAL_METADATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / FINAL_FEATURE_SUBDIR).mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / FINAL_SPLIT_SUBDIR).mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / INTERPOLATION_MASK_SUBDIR).mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


ensure_directories()
