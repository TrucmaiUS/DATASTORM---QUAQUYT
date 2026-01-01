"""
Advanced video augmentation effects for golf pose dataset.
Includes spatial, temporal, and lighting augmentations.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np


class VideoAugmentor:
    """
    Comprehensive video augmentation system that applies effects consistently
    across all frames in a video sequence.
    """
    
    def __init__(self):
        """Initialize augmentation pipeline with all available effects."""
        # Albumentations pipeline for spatial/color augmentations
        self.spatial_transform = A.ReplayCompose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.GaussNoise(var_limit=(3.0, 20.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomScale(scale_limit=0.2, p=0.5),
        ])
    
    def augment_sequence_consistently(
        self,
        frames: List[np.ndarray],
        apply_time_warp: bool = False,
        apply_flicker: bool = False,
        time_warp_params: dict = None,
        flicker_params: dict = None,
        seed: int = None
    ) -> List[np.ndarray]:
        """
        Apply augmentations consistently across all frames in a video sequence.
        
        This ensures that the same spatial transformation (rotation, flip, etc.)
        is applied to every frame, maintaining temporal consistency.
        
        Args:
            frames: List of video frames
            apply_time_warp: Whether to apply temporal distortion
            apply_flicker: Whether to apply lighting flicker
            time_warp_params: Parameters for time warp effect
            flicker_params: Parameters for flicker effect
            seed: Random seed for reproducibility
            
        Returns:
            Augmented video frames
        """
        if not frames:
            return []
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Get target dimensions from first frame
        target_h, target_w = frames[0].shape[:2]
        
        # Apply time warp first (temporal reordering)
        augmented = frames
        if apply_time_warp:
            tw_params = time_warp_params or {'warp_strength': 0.3, 'num_control_points': 5}
            augmented = time_warp_augmentation(augmented, seed=seed, **tw_params)
        
        # Apply spatial/color augmentations consistently across all frames
        # Sample ONE transformation and apply it to all frames
        replay = self.spatial_transform(image=augmented[0])
        first_frame = replay['image']
        
        # Resize if needed
        if first_frame.shape[:2] != (target_h, target_w):
            first_frame = cv2.resize(first_frame, (target_w, target_h))
        
        result_frames = [first_frame]
        
        # Apply the SAME transformation to all remaining frames
        for frame in augmented[1:]:
            result = A.ReplayCompose.replay(replay['replay'], image=frame)
            aug_frame = result['image']
            
            # Resize if needed
            if aug_frame.shape[:2] != (target_h, target_w):
                aug_frame = cv2.resize(aug_frame, (target_w, target_h))
            
            result_frames.append(aug_frame)
        
        # Apply flicker last (per-frame brightness variation)
        if apply_flicker:
            fl_params = flicker_params or {'flicker_strength': 0.3, 'flicker_frequency': 0.1, 'smooth': True}
            result_frames = flicker_augmentation(result_frames, seed=seed, **fl_params)
        
        return result_frames


def read_video(video_path: str | Path) -> Tuple[List[np.ndarray], float, Tuple[int, int]]:
    """
    Read video and return frames, fps, and resolution.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (frames, fps, (width, height))
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames, fps, (width, height)


def write_video(frames: List[np.ndarray], output_path: Path | str, fps: float) -> None:
    """
    Write frames to video file.
    
    Args:
        frames: List of video frames
        output_path: Path to output video
        fps: Frames per second
    """
    if not frames:
        print("Warning: No frames to write!")
        return
    
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    for frame in frames:
        writer.write(frame)
    
    writer.release()


def time_warp_augmentation(
    frames: List[np.ndarray],
    warp_strength: float = 0.3,
    num_control_points: int = 5,
    seed: int = None
) -> List[np.ndarray]:
    """
    Apply time warping to video frames.
    
    Time warp creates temporal distortion by varying playback speed throughout 
    the video. This simulates speed variations, non-uniform temporal sampling,
    and camera frame rate inconsistencies.
    
    Args:
        frames: List of video frames
        warp_strength: Strength of warping (0-1). Higher = more distortion
        num_control_points: Number of control points for smooth warping curve
        seed: Random seed for reproducibility
    
    Returns:
        Time-warped frames
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_frames = len(frames)
    if num_frames < 2:
        return frames
    
    # Generate smooth warping curve using control points
    control_indices = np.linspace(0, num_frames - 1, num_control_points)
    control_values = np.random.uniform(
        -warp_strength,
        warp_strength,
        num_control_points
    )
    
    # Interpolate to get smooth warp for each frame
    warp_curve = np.interp(
        np.arange(num_frames),
        control_indices,
        control_values
    )
    
    # Create warped frame indices
    original_indices = np.arange(num_frames)
    warped_indices = original_indices + warp_curve * num_frames
    
    # Normalize to valid range [0, num_frames-1]
    warped_indices = np.clip(warped_indices, 0, num_frames - 1)
    
    # Create output with same number of frames
    output_indices = np.linspace(0, num_frames - 1, num_frames)
    final_indices = np.interp(output_indices, warped_indices, original_indices)
    final_indices = np.clip(final_indices, 0, num_frames - 1).astype(int)
    
    # Sample frames according to warped timeline
    warped_frames = [frames[idx] for idx in final_indices]
    
    return warped_frames


def flicker_augmentation(
    frames: List[np.ndarray],
    flicker_strength: float = 0.3,
    flicker_frequency: float = 0.1,
    smooth: bool = True,
    seed: int = None
) -> List[np.ndarray]:
    """
    Apply flicker effect to video frames.
    
    Flicker simulates lighting variations by randomly adjusting brightness 
    across frames. This mimics unstable lighting conditions, auto-exposure 
    adjustments, shadow movements, and environmental lighting changes.
    
    Args:
        frames: List of video frames
        flicker_strength: Maximum brightness change (0-1). 0.3 = ±30% brightness
        flicker_frequency: Probability of flicker change per frame (0-1)
        smooth: If True, apply gradual transitions; if False, apply abrupt changes
        seed: Random seed for reproducibility
    
    Returns:
        Frames with flicker effect
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    num_frames = len(frames)
    if num_frames == 0:
        return frames
    
    if smooth:
        # Generate smooth flicker pattern using sinusoidal components
        x = np.linspace(0, num_frames * flicker_frequency, num_frames)
        
        # Combine multiple frequency components for natural variation
        flicker_pattern = (
            np.sin(x * 2 * np.pi) * 0.5 +
            np.sin(x * 5 * np.pi) * 0.3 +
            np.sin(x * 13 * np.pi) * 0.2
        )
        
        # Normalize and scale
        flicker_pattern = flicker_pattern / np.max(np.abs(flicker_pattern))
        brightness_factors = 1.0 + flicker_pattern * flicker_strength
    else:
        # Abrupt random flicker
        brightness_factors = np.ones(num_frames)
        for i in range(num_frames):
            if random.random() < flicker_frequency:
                brightness_factors[i] = 1.0 + random.uniform(-flicker_strength, flicker_strength)
    
    # Apply brightness adjustment to each frame
    flickered_frames = []
    for frame, factor in zip(frames, brightness_factors):
        # Convert to float for precision
        adjusted = frame.astype(np.float32) * factor
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        flickered_frames.append(adjusted)
    
    return flickered_frames


def generate_multiple_augmented_videos(
    frames: List[np.ndarray],
    num_videos: int,
    augmentor: VideoAugmentor,
    include_original: bool = True,
    apply_time_warp: bool = True,
    apply_flicker: bool = True,
    seed: int = None
) -> List[List[np.ndarray]]:
    """
    Generate multiple augmented versions of a video with random effects.
    
    Each augmented video gets a different random transformation, but the
    transformation is applied consistently across all frames in that video.
    
    Args:
        frames: Original video frames
        num_videos: Number of augmented videos to create
        augmentor: VideoAugmentor instance
        include_original: Whether to include original as first video
        apply_time_warp: Whether to apply time warp effect
        apply_flicker: Whether to apply flicker effect
        seed: Base random seed for reproducibility
        
    Returns:
        List of augmented video frame sequences
    """
    if not frames:
        return []
    
    augmented_videos = []
    
    # Include original if requested
    if include_original:
        augmented_videos.append(frames)
        start_idx = 1
    else:
        start_idx = 0
    
    # Generate augmented versions with different random seeds
    for i in range(start_idx, num_videos):
        video_seed = (seed + i) if seed is not None else None
        
        # Randomly decide whether to apply temporal effects for variety
        use_time_warp = apply_time_warp and (random.random() > 0.5)
        use_flicker = apply_flicker and (random.random() > 0.5)
        
        augmented = augmentor.augment_sequence_consistently(
            frames,
            apply_time_warp=use_time_warp,
            apply_flicker=use_flicker,
            seed=video_seed
        )
        augmented_videos.append(augmented)
    
    return augmented_videos


def process_video_file(
    video_path: Path,
    output_dir: Path,
    num_augmented: int = 5,
    include_original: bool = True,
    apply_time_warp: bool = True,
    apply_flicker: bool = True,
    seed: int = None
) -> List[Path]:
    """
    Process a single video file and create multiple augmented versions.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save augmented videos
        num_augmented: Number of augmented videos to create
        include_original: Whether to save original as first video
        apply_time_warp: Whether to apply time warp effect
        apply_flicker: Whether to apply flicker effect
        seed: Random seed for reproducibility
    
    Returns:
        List of output video paths
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Read video
    frames, fps, resolution = read_video(video_path)
    video_stem = video_path.stem
    
    # Create augmentor
    augmentor = VideoAugmentor()
    
    # Generate augmented videos
    augmented_videos = generate_multiple_augmented_videos(
        frames,
        num_videos=num_augmented,
        augmentor=augmentor,
        include_original=include_original,
        apply_time_warp=apply_time_warp,
        apply_flicker=apply_flicker,
        seed=seed
    )
    
    # Save all augmented videos
    output_paths = []
    for idx, aug_frames in enumerate(augmented_videos):
        suffix = 'original' if (idx == 0 and include_original) else f'aug{idx:02d}'
        output_path = output_dir / f"{video_stem}_{suffix}.mp4"
        write_video(aug_frames, output_path, fps)
        output_paths.append(output_path)
    
    return output_paths


def process_folder(
    input_folder: Path | str,
    output_folder: Path | str,
    num_augmented: int = 5,
    include_original: bool = True,
    apply_time_warp: bool = True,
    apply_flicker: bool = True,
    recursive: bool = True,
    video_extensions: List[str] = None,
    seed: int = None
) -> dict:
    """
    Process all videos in a folder, creating multiple augmented versions of each.
    
    Args:
        input_folder: Path to input folder containing videos
        output_folder: Path to output folder for augmented videos
        num_augmented: Number of augmented videos to create per source video
        include_original: Whether to save original as first video
        apply_time_warp: Whether to apply time warp effect
        apply_flicker: Whether to apply flicker effect
        recursive: Whether to process subfolders recursively
        video_extensions: List of video file extensions to process (default: .mp4, .avi, .mov)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary mapping input paths to output paths
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    
    if video_extensions is None:
        video_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
    
    # Find all video files
    if recursive:
        video_files = [f for f in input_folder.rglob('*') if f.suffix in video_extensions]
    else:
        video_files = [f for f in input_folder.glob('*') if f.suffix in video_extensions]
    
    if not video_files:
        print(f"No video files found in {input_folder}")
        return {}
    
    print(f"Found {len(video_files)} video files to process")
    print(f"Will create {num_augmented} augmented videos per source video")
    print(f"Total videos to generate: {len(video_files) * num_augmented}")
    
    # Process each video
    results = {}
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_path.name}")
        
        # Preserve folder structure in output
        relative_path = video_path.relative_to(input_folder)
        video_output_dir = output_folder / relative_path.parent / video_path.stem
        
        try:
            output_paths = process_video_file(
                video_path,
                video_output_dir,
                num_augmented=num_augmented,
                include_original=include_original,
                apply_time_warp=apply_time_warp,
                apply_flicker=apply_flicker,
                seed=seed
            )
            results[str(video_path)] = [str(p) for p in output_paths]
            print(f"  ✓ Created {len(output_paths)} augmented version(s)")
        except Exception as e:
            print(f"  ✗ Error processing {video_path.name}: {e}")
            results[str(video_path)] = []
    
    total_generated = sum(len(v) for v in results.values())
    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"Source videos processed: {len([r for r in results.values() if r])}/{len(video_files)}")
    print(f"Total augmented videos created: {total_generated}")
    print(f"Failed: {len([r for r in results.values() if not r])} videos")
    print(f"Output folder: {output_folder}")
    print('='*60)
    
    return results
