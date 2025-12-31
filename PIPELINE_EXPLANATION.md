# Golf Swing Analysis Pipeline - Giải Thích Chi Tiết

## 📋 Mục Lục
1. [Tổng Quan Pipeline](#tổng-quan-pipeline)
2. [Cấu Trúc Dữ Liệu](#cấu-trúc-dữ-liệu)
3. [Task A: Cấu Hình & Metadata](#task-a-cấu-hình--metadata)
4. [Task B: Xử Lý Video](#task-b-xử-lý-video)
5. [Task C: Phát Hiện Swing Window](#task-c-phát-hiện-swing-window)
6. [Task D: Trích Xuất Pose](#task-d-trích-xuất-pose)
7. [Task E: Chuẩn Hóa Không Gian & Thời Gian](#task-e-chuẩn-hóa-không-gian--thời-gian)
8. [Task F: Feature Engineering](#task-f-feature-engineering)
9. [Task G: Augmentation](#task-g-augmentation)
10. [Task H: Dataset Export](#task-h-dataset-export)
11. [Workflow Tổng Thể](#workflow-tổng-thể)

---

## Tổng Quan Pipeline

Pipeline này xử lý video golf swing từ raw video → pose sequences → features → dataset sẵn sàng cho training.

### Luồng Xử Lý Chính

```
Video MP4 (raw)
    ↓
[Task A] Scan metadata (fps, duration, environment, band)
    ↓
[Task B] Load & Resample video → 30 FPS
    ↓
[Task C] Detect swing window (wrist velocity)
    ↓
[Task D] Extract pose landmarks (MediaPipe)
    ↓
[Task E] Normalize (spatial + temporal) → 100 frames
    ↓
[Task F] Extract 16 features per keypoint
    ↓
[Task G] Augmentation (4 strategies)
    ↓
[Task H] Save .npz + z-score normalize + splits
    ↓
Final Dataset: sequences/*.npz + metadata.csv
```

### Đầu Vào & Đầu Ra

**Đầu vào:**
- Video files: `.mp4`, `.mov`, `.mkv`, `.avi`
- Cấu trúc thư mục: `Public Test/{Indoor|Outdoor}/{Band X-Y}/{video_id}.mp4`

**Đầu ra:**
- `processed_videos/sequences/*.npz`: Features đã normalize (100, 33, 16)
- `processed_videos/interp_masks/*.npz`: Interpolation masks
- `processed_videos/metadata/processed_metadata.csv`: Metadata chi tiết
- `processed_videos/splits/dataset_splits.json`: Train/val/test splits
- `processed_videos/features/feature_scaler.json`: Z-score scaler (mean, std)
- `metadata.csv`: Video metadata gốc

---

## Cấu Trúc Dữ Liệu

### 1. PoseSequence

Dataclass đại diện cho một chuỗi pose qua thời gian:

```python
@dataclass
class PoseSequence:
    data: np.ndarray           # Shape: (T, 33, 4)
                               # T = số frames, 33 = keypoints, 4 = [x,y,z,vis]
    frame_times: np.ndarray    # Shape: (T,) - timestamps tính bằng giây
    fps: float                 # Frame rate của video
    interpolation_mask: np.ndarray  # Shape: (T, 33) - True = interpolated
    valid_mask: np.ndarray     # Shape: (T,) - True = frame có pose
```

**33 Keypoints của MediaPipe Pose:**
```
0-10:   Head (nose, eyes, ears, mouth)
11-16:  Arms (shoulders, elbows, wrists)
17-22:  Hands (fingers)
23-28:  Legs (hips, knees, ankles)
29-32:  Feet (heel, foot index)
```

### 2. SwingWindow

Xác định khoảng thời gian của swing:

```python
@dataclass
class SwingWindow:
    start_frame: int      # Frame bắt đầu swing
    end_frame: int        # Frame kết thúc swing
    confidence: float     # Độ tin cậy detection (0-1)
    method: str           # "wrist_velocity" hoặc "torso_angle"
```

### 3. QualityMetrics

Đánh giá chất lượng pose sequence:

```python
@dataclass
class QualityMetrics:
    valid_ratio: float              # Tỉ lệ frames hợp lệ (0-1)
    mean_visibility: float          # Visibility trung bình của key joints
    keypoint_visibility: Dict       # Visibility từng nhóm keypoint
    longest_dropout: int            # Số frames liên tiếp bị thiếu
    low_quality: bool               # True = không nên augment
```

---

## Task A: Cấu Hình & Metadata

### 1. Cấu Hình Tập Trung (config.py)

Tất cả các tham số được định nghĩa trong `config.py`:

```python
# Đường dẫn
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "Public Test"
OUTPUT_ROOT = BASE_DIR / "processed_videos"
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_full.task"

# Video parameters
TARGET_FPS: int = 30          # Resample tất cả video về 30 FPS
N_FRAMES: int = 100           # Số frames sau khi temporal resample
PADDING_MARGIN_FRAMES: int = 5  # Padding cho swing window

# Thresholds
THRESHOLDS = {
    "pose_visibility": 0.5,    # Keypoint có visibility >= 0.5 mới hợp lệ
    "valid_ratio": 0.85,       # Ít nhất 85% frames phải hợp lệ
    "mean_visibility": 0.6,    # Visibility trung bình >= 0.6
}

# Features
POSE_FEATURE_DIM: int = 16    # Số features per keypoint

# Augmentation
AUGMENTATION_CONFIG = {
    "gaussian_std": 0.01,          # Std của Gaussian noise
    "time_warp_pct": 0.05,         # ±5% time warp
    "temporal_jitter_frames": 2,   # ±2 frames jitter
    "dropout_prob": 0.05,          # 5% dropout probability
}

# Dataset splits
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
```

### 2. Mapping Folders → Labels

```python
ENVIRONMENT_FOLDER_MAP = {
    "Trong nhà - Indoor": "indoor",
    "Ngoài trời - Outdoor": "outdoor",
    "Indoor": "indoor",
    "Outdoor": "outdoor",
}

BAND_FOLDER_MAP = {
    "Band 1-2": "1_2",
    "Band 2-4": "2_4",
    "Band 4-6": "4_6",
    "Band 6-8": "6_8",
    "Band 8-10": "8_10",
}
```

### 3. Scan Dataset & Build Metadata

File: `pipeline/metadata_utils.py`

```python
def scan_dataset(data_root: Path) -> List[VideoMetadata]:
    """Quét đệ quy tất cả video và thu thập metadata.
    
    Quy trình:
    1. Duyệt tất cả file trong data_root
    2. Lọc video files (theo VIDEO_EXTENSIONS)
    3. Suy luận env và band từ cấu trúc thư mục
    4. Đọc thông tin video (fps, duration, frame_count, resolution)
    """
    records = []
    for video_path in root.rglob("*"):
        if not video_path.is_file() or video_path.suffix.lower() not in config.VIDEO_EXTENSIONS:
            continue
        
        # Suy luận env và band từ đường dẫn
        rel_parts = video_path.relative_to(root).parts
        env, band = _infer_env_band(rel_parts)
        
        # Đọc thông tin video
        fps, duration, frame_count, resolution = _read_video_stats(video_path)
        
        records.append(VideoMetadata(
            video_id=video_path.stem,
            env=env,
            band=band,
            file_path=video_path,
            fps=fps,
            duration=duration,
            frame_count=frame_count,
            resolution=resolution,
        ))
    return records
```

**Kết quả:** `metadata.csv` với các cột:
```
video_id, env, band, file_path, fps, duration, frame_count, resolution, 
swing_start_frame, swing_end_frame, low_quality, ...
```

---

## Task B: Xử Lý Video

File: `pipeline/video_processing.py`

### Load & Resample Video

```python
def load_and_resample(self, video_path: Path) -> VideoClip:
    """Load video và resample về target FPS.
    
    Phương pháp:
    - Đọc frame theo thời gian thực của video
    - Sample frame khi timestamp >= next_sample_time
    - Đảm bảo khoảng cách đều đặn giữa các frame
    """
    cap = cv2.VideoCapture(str(video_path))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or float(self.target_fps)
    orig_dt = 1.0 / max(orig_fps, 1e-3)
    target_dt = 1.0 / self.target_fps
    
    next_sample_time = 0.0
    frames = []
    timestamp = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample frame khi timestamp đạt next_sample_time
        if timestamp + 1e-6 >= next_sample_time:
            frames.append(frame)
            next_sample_time += target_dt
        
        timestamp += orig_dt
    
    cap.release()
    return VideoClip(frames=frames, fps=self.target_fps, original_fps=orig_fps)
```

**Ví dụ:**
- Video gốc: 60 FPS, 180 frames → 3 giây
- Sau resample: 30 FPS, 90 frames → vẫn 3 giây

---

## Task C: Phát Hiện Swing Window

File: `pipeline/video_processing.py`

### Method 1: Wrist Velocity (Primary)

```python
def detect_swing_window(self, pose_sequence: PoseSequence) -> SwingWindow:
    """Phát hiện swing window dựa trên wrist velocity.
    
    Thuật toán:
    1. Tính vận tốc cổ tay trái và phải tại mỗi frame
    2. Lấy max của 2 cổ tay
    3. Tìm peak (điểm có vận tốc cao nhất)
    4. Mở rộng về 2 phía từ peak cho đến khi tốc độ < threshold
    5. Thêm padding margin
    """
    data = pose_sequence.data
    vis = data[:, :, 3]
    wrist_indices = (15, 16)  # Left wrist (15), Right wrist (16)
    fps = pose_sequence.fps
    
    # Tính wrist speed tại mỗi frame
    speeds = np.zeros(len(data))
    for idx in wrist_indices:
        coords = data[:, idx, :3]
        visibility = vis[:, idx]
        
        # Tính khoảng cách Euclidean giữa các frame liên tiếp
        diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        diffs = np.insert(diffs, 0, 0.0)  # Frame đầu tiên: speed = 0
        diffs *= fps  # Chuyển sang units/second
        
        # Bỏ qua frames có visibility thấp
        diffs[visibility < config.THRESHOLDS["pose_visibility"]] = 0.0
        
        # Lấy max của left và right wrist
        speeds = np.maximum(speeds, diffs)
    
    max_speed = float(np.max(speeds))
    if max_speed == 0:
        return self._torso_based_window(pose_sequence)  # Fallback
    
    # Thiết lập ngưỡng
    low_th = 0.1 * max_speed
    peak = int(np.argmax(speeds))
    
    # Tìm điểm bắt đầu: quét ngược từ peak
    start = peak
    for i in range(peak, -1, -1):
        if speeds[i] < low_th:
            start = i
            break
    
    # Tìm điểm kết thúc: quét xuôi từ peak
    end = peak
    for i in range(peak, len(speeds)):
        if speeds[i] < low_th:
            end = i
            break
    
    # Đảm bảo có ít nhất 0.5s sau peak (follow-through)
    end = max(end, peak + int(0.5 * fps))
    
    # Thêm padding
    start = max(0, start - self.padding_margin)
    end = min(len(speeds) - 1, end + self.padding_margin)
    
    return SwingWindow(start_frame=start, end_frame=end, 
                      confidence=min(1.0, max_speed), method="wrist_velocity")
```

### Method 2: Torso Angle (Fallback)

Khi không phát hiện được chuyển động cổ tay (max_speed = 0):

```python
def _torso_based_window(self, pose_sequence: PoseSequence) -> SwingWindow:
    """Fallback method dựa trên góc xoay vai."""
    data = pose_sequence.data
    fps = pose_sequence.fps
    
    # Lấy tọa độ 2D của vai trái và vai phải
    left_shoulder = data[:, 11, :2]
    right_shoulder = data[:, 12, :2]
    
    # Vector nối 2 vai
    shoulder_vec = right_shoulder - left_shoulder
    
    # Tính góc của vector (độ)
    angles = np.degrees(np.arctan2(shoulder_vec[:, 1], shoulder_vec[:, 0]))
    
    # Tính tốc độ góc (degrees/second)
    angle_speed = np.abs(np.diff(angles, prepend=angles[0])) * fps
    
    # Tìm peak và mở rộng
    peak = int(np.argmax(angle_speed))
    threshold = 0.25 * float(np.max(angle_speed))
    
    # Ước lượng window: ~0.5s trước peak, ~0.8s sau peak
    start = max(0, peak - int(0.5 * fps))
    end = min(len(angles) - 1, peak + int(0.8 * fps))
    
    # Tinh chỉnh bằng threshold...
    
    return SwingWindow(start_frame=start, end_frame=end, 
                      confidence=float(np.max(angle_speed)), method="torso_angle")
```

---

## Task D: Trích Xuất Pose

File: `pipeline/pose_processing.py`

### Extract Pose Landmarks với MediaPipe

```python
def extract_sequence(self, frames: list[np.ndarray], fps: float) -> PoseSequence:
    """Trích xuất pose từ mỗi frame.
    
    MediaPipe VIDEO mode yêu cầu:
    - Timestamps phải tăng dần (monotonic)
    - Đơn vị: microseconds
    """
    num_frames = len(frames)
    pose_array = np.zeros((num_frames, 33, 4), dtype=np.float32)
    interpolation_mask = np.zeros((num_frames, 33), dtype=bool)
    valid_mask = np.zeros(num_frames, dtype=bool)
    
    for idx, frame in enumerate(frames):
        # Convert BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)
        
        # Tính timestamp (microseconds)
        timestamp = int(idx * 1_000_000 / max(fps, 1e-3))
        
        # Detect pose
        landmarker = self._get_landmarker()
        result = landmarker.detect_for_video(mp_image, timestamp)
        
        if not result.pose_landmarks:
            continue  # Frame này không có pose
        
        # Lấy 33 keypoints
        landmarks = result.pose_landmarks[0]
        for j, lm in enumerate(landmarks):
            pose_array[idx, j, :] = [lm.x, lm.y, lm.z, lm.visibility]
        
        valid_mask[idx] = True
    
    frame_times = np.arange(num_frames) / max(fps, 1e-3)
    return PoseSequence(
        data=pose_array,
        frame_times=frame_times,
        fps=fps,
        interpolation_mask=interpolation_mask,
        valid_mask=valid_mask,
    )
```

### Interpolation cho Missing Keypoints

```python
def _interpolate(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Nội suy các keypoints bị thiếu.
    
    Keypoint bị thiếu khi visibility < threshold.
    Phương pháp: Linear interpolation giữa 2 keypoints hợp lệ gần nhất.
    """
    pose = data.copy()
    vis = pose[:, :, 3]
    mask = vis >= config.THRESHOLDS["pose_visibility"]
    interp_mask = np.zeros_like(mask, dtype=bool)
    
    for joint in range(pose.shape[1]):  # 33 keypoints
        valid_idx = np.where(mask[:, joint])[0]
        if valid_idx.size < 2:
            continue  # Không đủ điểm để interpolate
        
        missing_idx = np.where(~mask[:, joint])[0]
        
        # Interpolate từng chiều (x, y, z)
        for dim in range(3):
            pose[missing_idx, joint, dim] = np.interp(
                missing_idx, valid_idx, pose[valid_idx, joint, dim]
            )
        
        # Đặt visibility = threshold cho interpolated points
        pose[missing_idx, joint, 3] = config.THRESHOLDS["pose_visibility"]
        interp_mask[missing_idx, joint] = True
    
    return pose, interp_mask
```

### Smoothing với Savitzky-Golay Filter

```python
def _smooth(self, data: np.ndarray) -> np.ndarray:
    """Làm mượt pose sequence bằng Savitzky-Golay filter.
    
    Loại bỏ jitter và noise từ detection.
    Fallback: Moving average nếu không có scipy.
    """
    from scipy.signal import savgol_filter
    
    # Window size phải lẻ và < số frames
    window = min(11, data.shape[0] - (1 - data.shape[0] % 2))
    if window < 5:
        return data
    
    poly = 3 if window >= 5 else 1
    smoothed = savgol_filter(data, window_length=window, polyorder=poly, axis=0)
    return smoothed
```

---

## Task E: Chuẩn Hóa Không Gian & Thời Gian

File: `pipeline/pose_processing.py`

### Spatial Normalization

Mục tiêu: Loại bỏ ảnh hưởng của vị trí camera, khoảng cách, và góc quay.

```python
def _spatial_normalize(self, data: np.ndarray) -> np.ndarray:
    """Chuẩn hóa không gian pose theo từng frame.
    
    Các bước:
    1. Translation: Dịch chuyển mid_hip về gốc tọa độ (0, 0, 0)
    2. Scaling: Chia cho hip_width để normalize kích thước cơ thể
    3. Rotation: Xoay để hip line song song với trục X
    4. Flip: Đảm bảo vai hướng lên (+Y)
    """
    normalized = data.copy()
    
    for t in range(normalized.shape[0]):
        frame = normalized[t]
        
        # Bước 1: Translation - dịch mid_hip về origin
        left_hip = frame[23, :3]   # Keypoint 23
        right_hip = frame[24, :3]  # Keypoint 24
        mid_hip = (left_hip + right_hip) / 2
        frame[:, :3] -= mid_hip
        
        # Bước 2: Scaling - normalize theo hip width
        hip_width = np.linalg.norm(right_hip - left_hip)
        if hip_width < 1e-4:
            continue  # Skip nếu hip width quá nhỏ
        frame[:, :3] /= hip_width
        
        # Bước 3: Rotation - align hip line với trục X
        hip_vec = frame[24, :2] - frame[23, :2]  # 2D vector
        hip_angle = np.arctan2(hip_vec[1], hip_vec[0])
        
        # Rotation matrix
        cos_a, sin_a = np.cos(-hip_angle), np.sin(-hip_angle)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        frame[:, :2] = frame[:, :2] @ rot.T
        
        # Bước 4: Flip - đảm bảo shoulders hướng lên (+Y)
        mid_shoulder = (frame[11, :2] + frame[12, :2]) / 2
        if mid_shoulder[1] < 0:
            frame[:, 1] *= -1  # Flip Y axis
    
    return normalized
```

**Minh họa:**
```
Before normalization:          After normalization:
    * (shoulder)                   * (shoulder) ← Y+
   /|\                            /|\
  / | \                          / | \
 *  *  *                        *  *  * 
    |                              |
  *(hip) ← camera                *(0,0) ← origin
                                   |
                              ←────┴────→ X axis (hip line)
```

### Handedness Normalization

Đảm bảo tất cả swing đều là right-handed (hoặc left-handed):

```python
def _normalize_handedness(self, data: np.ndarray) -> np.ndarray:
    """Mirror left-handed swings thành right-handed.
    
    Phát hiện: Left-handed khi right wrist nằm ở phía X+ (phía phải)
    """
    # Tính vị trí X trung bình của 2 cổ tay
    left_wrist = np.mean(data[:, 15, 0])   # Keypoint 15
    right_wrist = np.mean(data[:, 16, 0])  # Keypoint 16
    
    mirrored = data.copy()
    
    # Nếu right wrist > left wrist → left-handed → mirror
    if right_wrist > left_wrist:
        mirrored[:, :, 0] *= -1  # Flip X axis
    
    return mirrored
```

### Temporal Resampling

Chuẩn hóa tất cả sequences về cùng số frames:

```python
def _temporal_resample(self, data: np.ndarray, target_frames: int) -> np.ndarray:
    """Resample pose sequence về số frames cố định.
    
    Phương pháp: Linear interpolation
    """
    num_frames = data.shape[0]
    if num_frames == target_frames:
        return data
    
    # Tạo mapping từ original → target frames
    src_x = np.linspace(0, 1, num_frames)
    dst_x = np.linspace(0, 1, target_frames)
    
    resampled = np.zeros((target_frames, data.shape[1], data.shape[2]), dtype=np.float32)
    
    # Interpolate từng keypoint, từng chiều
    for joint in range(data.shape[1]):      # 33 keypoints
        for dim in range(data.shape[2]):    # 4 chiều (x,y,z,vis)
            resampled[:, joint, dim] = np.interp(dst_x, src_x, data[:, joint, dim])
    
    return resampled
```

**Ví dụ:**
- Frame ban đầu: 75 frames (2.5s @ 30fps)
- Sau resample: 100 frames (vẫn đại diện cho 2.5s)

### Quality Metrics

```python
def _quality_metrics(self, data: np.ndarray, interp_mask: np.ndarray) -> QualityMetrics:
    """Đánh giá chất lượng của pose sequence."""
    vis = data[:, :, 3]
    
    # 1. Valid ratio: Tỉ lệ frames có ít nhất 1 keypoint visible
    valid_mask = np.max(vis, axis=1) >= config.THRESHOLDS["pose_visibility"]
    valid_ratio = float(np.mean(valid_mask))
    
    # 2. Keypoint visibility: Visibility của từng nhóm keypoint quan trọng
    keypoint_visibility = {}
    for name, (l_idx, r_idx) in config.KEY_JOINTS.items():
        key_vis = np.mean(vis[:, [l_idx, r_idx]])
        keypoint_visibility[name] = float(key_vis)
    
    mean_visibility = float(np.mean(list(keypoint_visibility.values())))
    
    # 3. Longest dropout: Độ dài chuỗi frames bị thiếu dài nhất
    longest_dropout = self._longest_dropout(valid_mask)
    
    # 4. Low quality flag
    low_quality = (
        valid_ratio < config.THRESHOLDS["valid_ratio"] or
        mean_visibility < config.THRESHOLDS["mean_visibility"]
    )
    
    return QualityMetrics(
        valid_ratio=valid_ratio,
        mean_visibility=mean_visibility,
        keypoint_visibility=keypoint_visibility,
        longest_dropout=longest_dropout,
        low_quality=low_quality,
    )
```

---

## Task F: Feature Engineering

File: `pipeline/feature_engineering.py`

### 16 Features per Keypoint

Mỗi keypoint (33 keypoints) tại mỗi frame (100 frames) có 16 features:

```python
def compute_features(self, pose_sequence: PoseSequence) -> np.ndarray:
    """Tính toán 16 features cho mỗi keypoint tại mỗi frame.
    
    Features (16 chiều):
    ┌─────────────────────────────────────────────────────────┐
    │ 1-3:   coords (x, y, z)        - Tọa độ đã normalize   │
    │ 4:     visibility              - Độ tin cậy (0-1)      │
    │ 5-7:   velocity (vx, vy, vz)   - Gradient bậc 1        │
    │ 8-10:  acceleration (ax,ay,az) - Gradient bậc 2        │
    │ 11:    speed_mag               - |velocity|            │
    │ 12:    accel_mag               - |acceleration|        │
    │ 13:    joint_angles            - Góc khớp (elbow,knee) │
    │ 14:    x_factor                - Hip-shoulder rotation │
    │ 15:    hip_shoulder_sep        - Khoảng cách hips-shoulders │
    │ 16:    max_wrist_speed         - Max wrist speed       │
    └─────────────────────────────────────────────────────────┘
    """
    # Tách tọa độ và visibility
    coords = pose_sequence.data[:, :, :3]  # (T, 33, 3)
    visibility = pose_sequence.data[:, :, 3:4]  # (T, 33, 1)
    
    # Tính delta time giữa các frame
    dt = 1.0 / max(pose_sequence.fps, 1e-3)
    
    # 1. Velocity: gradient bậc 1
    velocity = np.gradient(coords, axis=0, edge_order=2) / dt  # (T, 33, 3)
    
    # 2. Acceleration: gradient bậc 2
    acceleration = np.gradient(velocity, axis=0, edge_order=2) / dt  # (T, 33, 3)
    
    # 3. Speed magnitude
    speed_mag = np.linalg.norm(velocity, axis=-1, keepdims=True)  # (T, 33, 1)
    
    # 4. Acceleration magnitude
    accel_mag = np.linalg.norm(acceleration, axis=-1, keepdims=True)  # (T, 33, 1)
    
    # 5. Joint angles (elbow, knee, wrist)
    joint_angles = np.zeros_like(visibility)  # (T, 33, 1)
    for joint_idx, (a, b, c) in ANGLE_TRIPLETS.items():
        angles = self._calc_angle(
            coords[:, a, :], coords[:, b, :], coords[:, c, :]
        )
        joint_angles[:, joint_idx, 0] = angles
    
    # 6. Global metrics (replicated across all keypoints)
    # X-factor: Hiệu góc giữa shoulder line và hip line
    shoulder_vec = coords[:, 12, :2] - coords[:, 11, :2]  # (T, 2)
    hip_vec = coords[:, 24, :2] - coords[:, 23, :2]  # (T, 2)
    shoulder_angle = np.degrees(np.arctan2(shoulder_vec[:, 1], shoulder_vec[:, 0]))
    hip_angle = np.degrees(np.arctan2(hip_vec[:, 1], hip_vec[:, 0]))
    x_factor = self._shortest_angle_diff(shoulder_angle, hip_angle)  # (T,)
    
    # Hip-shoulder separation
    mid_shoulder = (coords[:, 11, :3] + coords[:, 12, :3]) / 2  # (T, 3)
    mid_hip = (coords[:, 23, :3] + coords[:, 24, :3]) / 2  # (T, 3)
    hip_shoulder_sep = np.linalg.norm(mid_shoulder - mid_hip, axis=-1)  # (T,)
    
    # Max wrist speed
    wrist_speeds = speed_mag[:, [15, 16], 0]  # (T, 2)
    wrist_speed = np.max(wrist_speeds, axis=1)  # (T,)
    
    # Stack global metrics và replicate cho 33 keypoints
    global_metrics = np.stack([x_factor, hip_shoulder_sep, wrist_speed], axis=-1)  # (T, 3)
    global_metrics = np.repeat(global_metrics[:, None, :], 33, axis=1)  # (T, 33, 3)
    
    # Concatenate tất cả features
    features = np.concatenate([
        coords,          # 3 chiều
        visibility,      # 1 chiều
        velocity,        # 3 chiều
        acceleration,    # 3 chiều
        speed_mag,       # 1 chiều
        accel_mag,       # 1 chiều
        joint_angles,    # 1 chiều
        global_metrics,  # 3 chiều
    ], axis=-1)  # Total: 3+1+3+3+1+1+1+3 = 16 chiều
    
    return features.astype(np.float32)  # Shape: (100, 33, 16)
```

**Output Shape:** `(100, 33, 16)` = 100 frames × 33 keypoints × 16 features = **52,800 values**

---

## Task G: Augmentation

File: `pipeline/feature_engineering.py`

Chỉ augment các samples **chất lượng cao** (`low_quality=False`).

### 1. Gaussian Noise

```python
def _gaussian_noise(self, data: np.ndarray) -> np.ndarray:
    """Thêm Gaussian noise vào tọa độ (x, y, z).
    
    Không thêm noise vào visibility.
    """
    aug = data.copy()
    std = config.AUGMENTATION_CONFIG["gaussian_std"]
    
    # Chỉ thêm noise vào 3 chiều đầu (x, y, z)
    noise = np.random.normal(0, std, size=aug[:, :, :3].shape)
    aug[:, :, :3] += noise
    
    return aug.astype(np.float32)
```

### 2. Time Warp

```python
def _time_warp(self, data: np.ndarray) -> np.ndarray:
    """Biến dạng thời gian: Tăng/giảm tốc độ cục bộ.
    
    Giữ nguyên số frames nhưng thay đổi temporal distribution.
    """
    pct = config.AUGMENTATION_CONFIG["time_warp_pct"]  # 0.05 = ±5%
    num_frames = data.shape[0]
    
    # Tạo warp factors ngẫu nhiên
    # warp_factor = 1.0 ± pct → tốc độ tăng/giảm 5%
    warp_factors = 1.0 + np.random.uniform(-pct, pct, size=num_frames)
    
    # Tích lũy để tạo new time indices
    cum_warp = np.cumsum(warp_factors)
    cum_warp = cum_warp / cum_warp[-1] * (num_frames - 1)  # Normalize về [0, num_frames-1]
    
    # Interpolate tất cả keypoints theo new time indices
    warped = np.zeros_like(data)
    for joint in range(data.shape[1]):
        for dim in range(data.shape[2]):
            warped[:, joint, dim] = np.interp(
                cum_warp, np.arange(num_frames), data[:, joint, dim]
            )
    
    return warped.astype(np.float32)
```

### 3. Temporal Jitter

```python
def _temporal_jitter(self, data: np.ndarray) -> np.ndarray:
    """Dịch chuyển ngẫu nhiên theo thời gian.
    
    Shift từng keypoint ±N frames một cách ngẫu nhiên.
    """
    jittered = data.copy()
    max_jitter = config.AUGMENTATION_CONFIG["temporal_jitter_frames"]  # ±2 frames
    
    for joint in range(data.shape[1]):
        # Random shift cho keypoint này
        shift = np.random.randint(-max_jitter, max_jitter + 1)
        if shift == 0:
            continue
        
        # Roll array (circular shift)
        jittered[:, joint, :] = np.roll(data[:, joint, :], shift, axis=0)
    
    return jittered.astype(np.float32)
```

### 4. Dropout & Re-interpolate

```python
def _dropout_interpolate(self, data: np.ndarray) -> np.ndarray:
    """Dropout ngẫu nhiên một số keypoints và interpolate lại.
    
    Simulate occlusion và missing detections.
    """
    aug = data.copy()
    prob = config.AUGMENTATION_CONFIG["dropout_prob"]  # 0.05 = 5%
    
    # Random dropout mask
    dropout_mask = np.random.rand(*aug[:, :, 0].shape) < prob  # (T, 33)
    
    # Set visibility = 0 cho dropped keypoints
    aug[dropout_mask, 3] = 0.0
    
    # Re-interpolate như trong _interpolate()
    for joint in range(aug.shape[1]):
        valid_idx = np.where(~dropout_mask[:, joint])[0]
        if valid_idx.size < 2:
            continue
        
        missing_idx = np.where(dropout_mask[:, joint])[0]
        for dim in range(3):  # x, y, z
            aug[missing_idx, joint, dim] = np.interp(
                missing_idx, valid_idx, aug[valid_idx, joint, dim]
            )
    
    return aug.astype(np.float32)
```

### Augmentation Pipeline

```python
def augment_sequence(self, data: np.ndarray) -> Dict[str, np.ndarray]:
    """Tạo 4 phiên bản augmented."""
    aug = {}
    aug["gaussian"] = self._gaussian_noise(data)
    aug["timewarp"] = self._time_warp(data)
    aug["jitter"] = self._temporal_jitter(data)
    aug["dropout"] = self._dropout_interpolate(data)
    return aug
```

**Kết quả:** Mỗi sample chất lượng cao → 1 original + 4 augmented = **5 samples**

---

## Task H: Dataset Export

File: `pipeline/feature_engineering.py`

### 1. Save Samples (.npz)

```python
def save_sample(
    self, video_id, env, band, features, 
    interpolation_mask, quality, metadata, augmented_suffix=""
) -> SamplePayload:
    """Lưu sample vào 2 files .npz."""
    # Tạo tên file
    suffix = f"__{augmented_suffix}" if augmented_suffix else ""
    sample_name = f"{video_id}{suffix}__{env}__{band}"
    
    # File 1: Features + metadata
    sequence_path = config.OUTPUT_ROOT / config.FINAL_SEQUENCE_SUBDIR / f"{sample_name}.npz"
    np.savez_compressed(
        sequence_path,
        X=features.astype(np.float32),  # (100, 33, 16)
        y=np.array(band),               # Label
        env=np.array(env),
        video_id=np.array(video_id),
        low_quality=np.array(quality.low_quality),
        quality=json.dumps(quality.__dict__),
    )
    
    # File 2: Interpolation mask
    mask_path = config.OUTPUT_ROOT / config.INTERPOLATION_MASK_SUBDIR / f"{sample_name}.npz"
    np.savez_compressed(mask_path, interpolation_mask=interpolation_mask)
    
    # Update thống kê cho scaler
    self.stats.update(features)
    
    return SamplePayload(...)
```

### 2. Z-Score Normalization

**⚠️ CRITICAL: Preventing Data Leakage**

Scaler must be fit ONLY on training data to prevent data leakage:
- ❌ **WRONG**: Fit scaler during `save_sample()` → includes val/test/augmented samples
- ✅ **CORRECT**: Fit scaler AFTER splits, using only train + non-augmented samples

**Why this matters:**
If validation/test samples influence the mean/std computation, we leak information about the test distribution into the normalization. This makes evaluation optimistically biased because the model is trained on features normalized using statistics that included test data.

**Correct Pipeline Order:**
1. Save all samples (base + augmented) to disk - NO scaler update during saving
2. Assign train/val/test splits by video_id
3. **Fit scaler**: Read saved .npz files, compute mean/std from train-only, non-augmented
4. Normalize ALL sequences using the train-only scaler
5. Export metadata and splits

#### Bước 1: Tính Mean & Std (Welford's Algorithm)

```python
@dataclass
class RunningFeatureStats:
    """Online algorithm để tính mean và std hiệu quả."""
    
    def update(self, feature_block: np.ndarray):
        """Cập nhật với batch mới.
        
        Welford's algorithm:
        - delta = x - mean_old
        - mean_new = mean_old + delta / n
        - M2 = M2 + delta * (x - mean_new)
        """
        flat = feature_block.reshape(-1, self.feature_dim)  # (T*33, 16)
        for vector in flat:
            self.count += 1
            delta = vector - self.mean
            self.mean += delta / self.count
            delta2 = vector - self.mean
            self.m2 += delta * delta2
    
    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Trả về mean, std."""
        if self.count < 2:
            std = np.ones_like(self.mean)
        else:
            variance = self.m2 / (self.count - 1)  # Bessel's correction
            std = np.sqrt(variance)
            std = np.maximum(std, 1e-6)  # Tránh chia cho 0
        return self.mean, std
```

#### Bước 2: Fit Scaler ONLY on Train Split

**NEW METHOD (Added to prevent data leakage):**

```python
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
```

#### Bước 3: Finalize và Lưu Scaler

```python
def finalize_scaler(self) -> Tuple[np.ndarray, np.ndarray]:
    """Lưu scaler vào JSON."""
    mean, std = self.stats.finalize()  # Shape: (16,)
    
    scaler_path = config.OUTPUT_ROOT / config.FINAL_FEATURE_SUBDIR / "feature_scaler.json"
    with open(scaler_path, "w") as f:
        json.dump({
            "mean": mean.tolist(),
            "std": std.tolist()
        }, f, indent=2)
    
    return mean, std
```

#### Bước 4: Normalize All Files

```python
def normalize_saved_sequences(self, mean: np.ndarray, std: np.ndarray):
    """Normalize tất cả .npz files.
    
    X_norm = (X - mean) / std
    """
    # Reshape để broadcast: (16,) → (1, 1, 16)
    reshaped_mean = mean.reshape(1, 1, -1)
    reshaped_std = std.reshape(1, 1, -1)
    
    for npz_path in (config.OUTPUT_ROOT / config.FINAL_SEQUENCE_SUBDIR).glob("*.npz"):
        # Load
        data = np.load(npz_path)
        X = data["X"]  # (100, 33, 16)
        
        # Normalize
        X_norm = (X - reshaped_mean) / reshaped_std
        
        # Save (overwrite)
        np.savez_compressed(
            npz_path,
            X=X_norm.astype(np.float32),
            y=data["y"],
            env=data["env"],
            video_id=data["video_id"],
            low_quality=data["low_quality"],
            quality=data["quality"],
        )
```

### 3. Dataset Splits

```python
def export_splits(self) -> Path:
    """Chia dataset thành train/val/test theo video_id.
    
    Đảm bảo:
    - Cùng video_id (bao gồm augmented versions) thuộc cùng split
    - Tỉ lệ: 70% train, 15% val, 15% test
    """
    # Lấy danh sách unique video_ids (không tính augmented suffix)
    unique_ids = list(set([
        sample.video_id.split("__")[0] for sample in self.samples
    ]))
    
    # Shuffle
    random.seed(config.RANDOM_SEED)
    random.shuffle(unique_ids)
    
    # Chia tỉ lệ
    n = len(unique_ids)
    train_end = int(n * config.SPLIT_RATIOS["train"])
    val_end = train_end + int(n * config.SPLIT_RATIOS["val"])
    
    train_ids = unique_ids[:train_end]
    val_ids = unique_ids[train_end:val_end]
    test_ids = unique_ids[val_end:]
    
    # Tạo mapping: video_id → split
    split_map = {}
    for vid in train_ids:
        split_map[vid] = "train"
    for vid in val_ids:
        split_map[vid] = "val"
    for vid in test_ids:
        split_map[vid] = "test"
    
    # Lưu JSON
    split_path = config.OUTPUT_ROOT / config.FINAL_SPLIT_SUBDIR / "dataset_splits.json"
    with open(split_path, "w") as f:
        json.dump(split_map, f, indent=2)
    
    return split_path
```

### 4. Export Metadata CSV

```python
def export_metadata(self) -> Path:
    """Export processed_metadata.csv."""
    df = pd.DataFrame(self.sample_records)
    # Columns: video_id, augmented, env, band, sequence_path, 
    #          low_quality, valid_ratio, mean_visibility, longest_dropout
    
    metadata_path = config.OUTPUT_ROOT / config.FINAL_METADATA_SUBDIR / "processed_metadata.csv"
    df.to_csv(metadata_path, index=False)
    return metadata_path
```

---

## Workflow Tổng Thể

### Trong Notebook (eda_reorganized-1.ipynb)

```python
# --- Khởi tạo processors ---
video_processor = VideoProcessor()
pose_processor = PoseProcessor()
feature_engineer = FeatureEngineer()

# --- Master pipeline loop ---
for row in metadata_df.itertuples():
    video_path = row.file_path
    
    # 1. Load & Resample video → 30 FPS
    clip = video_processor.load_and_resample(video_path)
    
    # 2. Extract pose từ full clip
    pose_processor.reset()  # Reset landmarker
    raw_sequence = pose_processor.extract_sequence(clip.frames, clip.fps)
    
    # 3. Detect swing window
    swing_window = video_processor.detect_swing_window(raw_sequence)
    
    # 4. Trim frames theo swing window
    start_frame = max(0, swing_window.start_frame - PADDING_MARGIN)
    end_frame = min(len(clip.frames), swing_window.end_frame + PADDING_MARGIN)
    trimmed_frames = clip.frames[start_frame:end_frame]
    
    # 5. Re-extract pose trên trimmed frames
    pose_processor.reset()
    raw_trimmed_sequence = pose_processor.extract_sequence(trimmed_frames, clip.fps)
    
    # 6. Prepare sequence (normalize + resample → 100 frames)
    processed_sequence, quality = pose_processor.prepare_sequence(raw_trimmed_sequence)
    
    # 7. Extract features (16 features per keypoint)
    features = feature_engineer.compute_features(processed_sequence)
    
    # 8. Save original sample (NO scaler update here!)
    payload = feature_engineer.save_sample(
        video_id=row.video_id,
        env=row.env,
        band=row.band,
        features=features,
        interpolation_mask=processed_sequence.interpolation_mask,
        quality=quality,
        metadata={"trimmed_duration_s": trimmed_duration},
    )
    
    # 9. Augmentation (nếu chất lượng cao)
    if not quality.low_quality:
        augmentations = feature_engineer.augment_sequence(processed_sequence.data)
        for aug_name, aug_data in augmentations.items():
            aug_sequence = PoseSequence(...)
            aug_features = feature_engineer.compute_features(aug_sequence)
            # Save augmented (NO scaler update here either!)
            feature_engineer.save_sample(
                video_id=row.video_id,
                env=row.env,
                band=row.band,
                features=aug_features,
                interpolation_mask=aug_sequence.interpolation_mask,
                quality=quality,
                metadata={"augmented": True, "strategy": aug_name},
                augmented_suffix=aug_name,
            )

# --- Finalize (CORRECT ORDER to prevent data leakage) ---
# 10. Assign splits FIRST
split_path = feature_engineer.export_splits()

# 11. Fit scaler on TRAIN ONLY, NON-AUGMENTED
scaler_fit_count = feature_engineer.fit_scaler_from_saved_sequences(exclude_low_quality=True)
print(f"Scaler fit on {scaler_fit_count} train samples (excludes val/test/augmented)")

# 12. Finalize scaler (compute mean/std and save JSON)
mean, std = feature_engineer.finalize_scaler()

# 13. Normalize ALL sequences using train-only scaler
feature_engineer.normalize_saved_sequences(mean, std)

# 14. Export metadata
processed_metadata_path = feature_engineer.export_metadata()
```

**⚠️ Key Change from Original Implementation:**
- **OLD (WRONG)**: `save_sample()` called `self.stats.update(features)` → included val/test/augmented
- **NEW (CORRECT)**: Scaler update removed from `save_sample()`, done separately via `fit_scaler_from_saved_sequences()` after splits are assigned
    features = feature_engineer.compute_features(processed_sequence)
    
    # 8. Save original sample
    payload = feature_engineer.save_sample(
        video_id=row.video_id,
        env=row.env,
        band=row.band,
        features=features,
        interpolation_mask=processed_sequence.interpolation_mask,
        quality=quality,
        metadata={"trimmed_duration_s": trimmed_duration},
    )
    
    # 9. Augmentation (nếu chất lượng cao)
    if not quality.low_quality:
        augmentations = feature_engineer.augment_sequence(processed_sequence.data)
        for aug_name, aug_data in augmentations.items():
            # Tạo PoseSequence mới
            aug_sequence = PoseSequence(
                data=aug_data,
                frame_times=processed_sequence.frame_times,
                fps=processed_sequence.fps,
                interpolation_mask=np.zeros_like(processed_sequence.interpolation_mask),
                valid_mask=processed_sequence.valid_mask,
            )
            # Extract features và save
            aug_features = feature_engineer.compute_features(aug_sequence)
            feature_engineer.save_sample(
                video_id=row.video_id,
                env=row.env,
                band=row.band,
                features=aug_features,
                interpolation_mask=aug_sequence.interpolation_mask,
                quality=quality,
                metadata={"augmented": True, "strategy": aug_name},
                augmented_suffix=aug_name,
            )

# --- Finalize ---
# 10. Tính z-score scaler
mean, std = feature_engineer.finalize_scaler()

# 11. Normalize tất cả .npz files
feature_engineer.normalize_saved_sequences(mean, std)

# 12. Export splits
split_path = feature_engineer.export_splits()

# 13. Export metadata
metadata_path = feature_engineer.export_metadata()
```

### Output Files

```
processed_videos/
├── sequences/
│   ├── Backside-8748-3__indoor__8_10.npz          # Original
│   ├── Backside-8748-3__gaussian__indoor__8_10.npz # Augmented
│   ├── Backside-8748-3__timewarp__indoor__8_10.npz
│   ├── Backside-8748-3__jitter__indoor__8_10.npz
│   └── Backside-8748-3__dropout__indoor__8_10.npz
├── interp_masks/
│   └── Backside-8748-3__indoor__8_10.npz
├── metadata/
│   └── processed_metadata.csv
├── features/
│   └── feature_scaler.json
└── splits/
    └── dataset_splits.json
```

### Loading Data for Training

```python
# Load một sample
data = np.load("processed_videos/sequences/Backside-8748-3__indoor__8_10.npz")
X = data["X"]           # Shape: (100, 33, 16) - normalized features
y = str(data["y"])      # "8_10" - handicap band
env = str(data["env"])  # "indoor"

# Load scaler
with open("processed_videos/features/feature_scaler.json") as f:
    scaler = json.load(f)
    mean = np.array(scaler["mean"])  # Shape: (16,)
    std = np.array(scaler["std"])    # Shape: (16,)

# Load splits
with open("processed_videos/splits/dataset_splits.json") as f:
    splits = json.load(f)
    split = splits["Backside-8748-3"]  # "train" / "val" / "test"
```

---

## Tổng Kết

### Các Thay Đổi Quan Trọng (Bug Fixes)

1. **Swing Window Bug**: Trim frames TRƯỚC khi re-extract pose (không phải chỉ trim pose)
2. **Duration Bug**: Tính `trimmed_duration` từ `raw_trimmed_sequence` TRƯỚC khi resample
3. **Augmentation Mask Bug**: Mỗi augmented sample có mask riêng (all-False)
4. **Short Sequence Bug**: Raise error nếu trimmed sequence < 3 frames
5. **Landmarker Reset**: Call `pose_processor.reset()` trước mỗi lần extract

### Các Con Số Quan Trọng

- **Input**: Raw video (variable fps, variable duration)
- **After resample**: 30 FPS (config.TARGET_FPS)
- **After trim**: Variable frames (swing window + padding)
- **After temporal resample**: 100 frames (config.N_FRAMES)
- **Keypoints**: 33 (MediaPipe Pose)
- **Features per keypoint**: 16
- **Final shape**: `(100, 33, 16)` = **52,800 values per sample**
- **Augmentation ratio**: 1 original + 4 augmented = **5× data**

### Performance Metrics

- **Processing time**: ~2-5 seconds per video (depending on length)
- **Disk usage**: ~100 KB per .npz file (compressed)
- **Memory**: ~50 MB per video during processing

### Best Practices

1. **Luôn reset landmarker** trước khi extract pose mới
2. **Check quality metrics** trước khi augment
3. **Normalize toàn bộ dataset** sau khi tính scaler
4. **Chia splits theo video_id** (không phải theo sample)
5. **Track interpolation mask** để biết keypoints nào là estimated

---

## Appendix: MediaPipe Pose Keypoints

```
Keypoint Index Map (33 points):

HEAD (0-10):
0:  nose
1:  left_eye_inner      2:  left_eye         3:  left_eye_outer
4:  right_eye_inner     5:  right_eye        6:  right_eye_outer
7:  left_ear            8:  right_ear
9:  mouth_left          10: mouth_right

ARMS (11-22):
11: left_shoulder       12: right_shoulder
13: left_elbow          14: right_elbow
15: left_wrist          16: right_wrist
17: left_pinky          18: right_pinky
19: left_index          20: right_index
21: left_thumb          22: right_thumb

LEGS (23-32):
23: left_hip            24: right_hip
25: left_knee           26: right_knee
27: left_ankle          28: right_ankle
29: left_heel           30: right_heel
31: left_foot_index     32: right_foot_index
```

---

**Tác giả:** Pipeline Documentation  
**Ngày tạo:** 2025-01-XX  
**Version:** 1.0
