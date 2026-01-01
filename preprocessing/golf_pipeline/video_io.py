from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np

from .augmentation import DataAugmentor


def extract_frames_from_video(
    video_path: str | Path,
    sample_every: int = 1,
    max_frames: int | None = None,
) -> tuple[List[np.ndarray], float]:
    """Decode frames from a video with optional stride sampling."""
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames: List[np.ndarray] = []
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % max(sample_every, 1) == 0:
                frames.append(frame)
                if max_frames and len(frames) >= max_frames:
                    break
            frame_idx += 1
    finally:
        cap.release()
    return frames, float(fps)


def export_augmented_videos(
    frames: Sequence[np.ndarray],
    augmentor: DataAugmentor,
    video_count: int,
    output_dir: str | Path = "augmented_videos",
    fps: int = 30,
    base_name: str | None = None,
) -> List[Path]:
    """Persist multiple augmented copies of a frame stack."""
    if not frames:
        raise ValueError("No frames were provided to export.")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    label = (base_name or "augmented").rstrip("_")
    written: List[Path] = []
    for idx in range(max(video_count, 1)):
        video_name = output_path / f"{label}_no{idx:02d}.mp4"
        writer = cv2.VideoWriter(str(video_name), fourcc, fps, (width, height))
        try:
            video_frames = frames if idx == 0 else augmentor.augment_sequence_consistently(list(frames))
            for frame in video_frames:
                writer.write(frame)
        finally:
            writer.release()
        written.append(video_name)
    return written


def list_videos(root: Path, patterns: Iterable[str] = ("*.mp4", "*.mov")) -> List[Path]:
    """Return sorted video paths for UI dropdowns."""
    files: List[Path] = []
    if not root.exists():
        return files
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    return sorted(files)
