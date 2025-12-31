"""Metadata helpers for the golf swing dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import pandas as pd

import config


@dataclass
class VideoMetadata:
    video_id: str
    env: str
    band: str
    file_path: Path
    fps: float
    duration: float
    frame_count: int
    resolution: str
    swing_start_frame: int = -1
    swing_end_frame: int = -1


def _infer_env_band(path_parts: Iterable[str]) -> Tuple[str, str]:
    env = "unknown"
    band = "unknown"
    for part in path_parts:
        if part in config.ENVIRONMENT_FOLDER_MAP:
            env = config.ENVIRONMENT_FOLDER_MAP[part]
        if part in config.BAND_FOLDER_MAP:
            band = config.BAND_FOLDER_MAP[part]
    return env, band


def _read_video_stats(video_path: Path) -> Tuple[float, float, int, str]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or config.TARGET_FPS
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()
    resolution = f"{width}x{height}"
    return fps, duration, frame_count, resolution


def scan_dataset(data_root: Path | None = None) -> List[VideoMetadata]:
    """Quét toàn bộ dataset và thu thập metadata cho mỗi video.
    
    Quy trình:
    1. Duyệt đệ quy tất cả file trong data_root
    2. Lọc các file video (theo VIDEO_EXTENSIONS)
    3. Suy luận env và band từ cấu trúc thư mục
    4. Đọc thông tin video (fps, duration, frame_count, resolution)
    5. Trả về danh sách VideoMetadata
    
    Args:
        data_root: Thư mục gốc chứa dataset (mặc định: config.DATA_ROOT)
        
    Returns:
        List[VideoMetadata] - danh sách metadata của tất cả videos
    """
    root = data_root or config.DATA_ROOT
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    records: List[VideoMetadata] = []
    # Duyệt đệ quy tất cả file
    for video_path in root.rglob("*"):
        # Chỉ xử lý video files
        if not video_path.is_file() or video_path.suffix.lower() not in config.VIDEO_EXTENSIONS:
            continue
        # Suy luận env và band từ đường dẫn
        rel_parts = video_path.relative_to(root).parts
        env, band = _infer_env_band(rel_parts)
        # Đọc thông tin video
        fps, duration, frame_count, resolution = _read_video_stats(video_path)
        video_id = video_path.stem
        records.append(
            VideoMetadata(
                video_id=video_id,
                env=env,
                band=band,
                file_path=video_path,
                fps=fps,
                duration=duration,
                frame_count=frame_count,
                resolution=resolution,
            )
        )
    return records


def build_metadata_csv(output_csv: Path | None = None) -> pd.DataFrame:
    """Generate metadata.csv with standardized columns."""
    output_csv = output_csv or (config.BASE_DIR / "metadata.csv")
    records = scan_dataset()
    rows: List[Dict[str, object]] = []
    for item in records:
        rows.append(
            {
                "video_id": item.video_id,
                "env": item.env,
                "band": item.band,
                "file_path": str(item.file_path.relative_to(config.BASE_DIR)),
                "fps": round(item.fps, 3),
                "duration": round(item.duration, 3),
                "frame_count": item.frame_count,
                "resolution": item.resolution,
                "swing_start_frame": item.swing_start_frame,
                "swing_end_frame": item.swing_end_frame,
            }
        )
    df = pd.DataFrame(rows)
    df.sort_values(["env", "band", "video_id"], inplace=True)
    df.to_csv(output_csv, index=False)
    return df
