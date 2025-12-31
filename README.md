# Golf Swing Sequence Pipeline

This workspace implements a reproducible pipeline for golf swing scoring based on MediaPipe Pose sequences. The new workflow replaces the legacy `Balanced_Dataset` routine with a documented **processed_videos** pipeline that operates directly on the raw `Public Test` footage.

## Key Components

1. **Config-first design** – `config.py` defines roots, classes, FPS, frame counts, stride, padding, thresholds, augmentation limits, and output folders. All notebooks/scripts import from this single source.
2. **Metadata standardization** – `pipeline/metadata_utils.py` scans the folder tree, normalizes labels (indoor/outdoor and band ranges), records fps/duration/resolution, and tracks swing windows.
3. **Video cleaning** – `pipeline/video_processing.py` resamples every clip to the target FPS, determines a single swing window using wrist velocity fallback torso angle changes, and stores start/end frames.
4. **Pose robustness** – `pipeline/pose_processing.py` relies solely on the MediaPipe Tasks Pose Landmarker, performs interpolation + Savitzky–Golay smoothing, spatial normalization (translate mid-hip to origin, scale by hip width, align axes, mirror handedness), and temporal resampling to `N_FRAMES`.
5. **Feature engineering & augmentation** – `pipeline/feature_engineering.py` produces `[T, J, F]` tensors (raw positions, velocity, acceleration, angles, golf metrics), applies pose-safe augmentations, z-score normalizes features, saves scaler parameters, and exports dataset splits.
6. **Quality controls** – Each sequence stores pose coverage (% valid frames), mean key-joint visibility, dropout streaks, and a `low_quality` flag guided by config thresholds.

## Outputs

- `metadata.csv` – raw + trimmed metadata with swing boundaries and quality stats.
- `processed_videos/sequences/*.npz` – normalized tensors with metadata fields (env, band, video_id, quality metrics) alongside interpolation masks.
- `processed_videos/metadata/processed_metadata.csv` – per-sample summary including split assignment.
- `processed_videos/features/feature_scaler.json` – means and stds for z-score normalization.
- `processed_videos/splits/dataset_splits.json` – deterministic train/val/test split by `video_id`.

## Running the Pipeline

1. Install dependencies (OpenCV, MediaPipe, pandas, numpy, seaborn, matplotlib, SciPy recommended).
2. Open `eda_reorganized-1.ipynb` and execute cells sequentially. Each section includes Markdown that explains *what* and *why* with respect to golf swing scoring.
3. Inspect logs for warnings about low-quality clips; problematic videos remain in the dataset with flags so downstream models can filter or reweight them.

## Notes

- `processed_videos` now supersedes the historical `Balanced_Dataset`. Class balance is reported in metadata so augmentation strategies can be tuned with evidence.
- The pipeline keeps the native directory layout (environment/band) to simplify auditing and manual review.
- CSV outputs are meant for debugging/EDA only; models should consume the `.npz` tensors described in Task H.
