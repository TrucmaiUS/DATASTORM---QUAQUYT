"""Video utilities: resampling and swing window detection.

Module này xử lý các tác vụ liên quan đến video:
- Load và resample video về FPS chuẩn
- Phát hiện swing window tự động
- Trim frames theo cửa sổ đã phát hiện
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

import config
from pipeline.types import PoseSequence, SwingWindow


@dataclass
class VideoClip:
    """Container cho video frames đã được load và resample.
    
    Attributes:
        frames: List các frame dưới dạng numpy arrays (BGR format)
        fps: Frame rate sau khi resample (thường là TARGET_FPS=30)
        original_fps: Frame rate của video gốc
    """
    frames: List[np.ndarray]
    fps: float
    original_fps: float

    @property
    def duration(self) -> float:
        """Tính độ dài video (giây) dựa trên số frames và FPS."""
        return len(self.frames) / self.fps if self.frames else 0.0


class VideoProcessor:
    """Xử lý video I/O và phát hiện swing window.
    
    Class này thực hiện:
    1. Load và resample video về target_fps
    2. Phát hiện swing window dựa trên wrist velocity
    3. Fallback sang torso angle nếu wrist detection thất bại
    """

    def __init__(self, target_fps: int = config.TARGET_FPS) -> None:
        self.target_fps = target_fps
        self.padding_margin = config.PADDING_MARGIN_FRAMES

    def load_and_resample(self, video_path: Path) -> VideoClip:
        """Load video và resample về target FPS.
        
        Phương pháp:
        - Đọc frame theo thời gian thực của video
        - Sample frame khi timestamp >= next_sample_time
        - Đảm bảo khoảng cách đều đặn giữa các frame
        
        Args:
            video_path: Đường dẫn tới file video
            
        Returns:
            VideoClip với frames đã resample
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS) or float(self.target_fps)
        orig_dt = 1.0 / max(orig_fps, 1e-3)
        target_dt = 1.0 / self.target_fps
        next_sample_time = 0.0
        frames: List[np.ndarray] = []
        timestamp = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if timestamp + 1e-6 >= next_sample_time:
                frames.append(frame)
                next_sample_time += target_dt
            timestamp += orig_dt

        cap.release()
        if not frames:
            raise RuntimeError(f"Video has no frames after resampling: {video_path}")

        return VideoClip(frames=frames, fps=self.target_fps, original_fps=orig_fps)

    def detect_swing_window(self, pose_sequence: PoseSequence) -> SwingWindow:
        """Phát hiện swing window dựa trên wrist velocity.
        
        Thuật toán:
        1. Tính vận tốc cổ tay trái và phải tại mỗi frame
        2. Lấy max của 2 cổ tay
        3. Tìm peak (điểm có vận tốc cao nhất)
        4. Mở rộng về 2 phía từ peak cho đến khi tốc độ < threshold
        5. Thêm padding margin
        
        Nếu không phát hiện được chuyển động cổ tay (max_speed = 0),
        sử dụng fallback: phát hiện dựa trên góc xoay vai.
        
        Args:
            pose_sequence: PoseSequence đã normalize
            
        Returns:
            SwingWindow với start_frame, end_frame, confidence
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
            # diff[t] = distance(frame[t], frame[t-1])
            diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
            diffs = np.insert(diffs, 0, 0.0)  # Frame đầu tiên: speed = 0
            diffs *= fps  # Chuyển sang đơn vị units/second
            
            # Bỏ qua các frame có visibility thấp (không tin cậy)
            diffs[visibility < config.THRESHOLDS["pose_visibility"]] = 0.0
            
            # Lấy max của left và right wrist → wrist speed tổng hợp
            speeds = np.maximum(speeds, diffs)

        max_speed = float(np.max(speeds))
        
        # Nếu không phát hiện được chuyển động cổ tay → fallback
        if max_speed == 0:
            return self._torso_based_window(pose_sequence)

        # Thiết lập ngưỡng tốc độ
        high_th = 0.3 * max_speed  # Tốc độ cao (không dùng trong code hiện tại)
        low_th = 0.1 * max_speed   # Tốc độ thấp - coi như bắt đầu/kết thúc swing
        peak = int(np.argmax(speeds))  # Frame có tốc độ cao nhất

        # Tìm điểm bắt đầu: quét ngược từ peak cho đến khi speed < low_th
        start = peak
        for i in range(peak, -1, -1):
            if speeds[i] < low_th:
                start = i
                break

        # Tìm điểm kết thúc: quét xuôi từ peak cho đến khi speed < low_th
        end = peak
        for i in range(peak, len(speeds)):
            if speeds[i] < low_th:
                end = i
                break
        # Đảm bảo có ít nhất 0.5s sau peak (follow-through)
        end = max(end, peak + int(0.5 * fps))

        # Thêm padding margin để không mất thông tin đầu/cuối swing
        start = max(0, start - self.padding_margin)
        end = min(len(speeds) - 1, end + self.padding_margin)

        # Đảm bảo window đủ dài (ít nhất N_FRAMES/2)
        if end - start < config.N_FRAMES // 2:
            end = min(len(speeds) - 1, start + config.N_FRAMES)

        confidence = min(1.0, max_speed)
        return SwingWindow(start_frame=start, end_frame=end, confidence=confidence, method="wrist_velocity")

    def _torso_based_window(self, pose_sequence: PoseSequence) -> SwingWindow:
        """Phát hiện swing window dựa trên góc xoay vai (fallback method).
        
        Được sử dụng khi:
        - Wrist velocity = 0 (không phát hiện được chuyển động cổ tay)
        - Pose bị occlusion ở tay
        - Video chất lượng thấp
        
        Thuật toán:
        1. Tính vector nối 2 vai (shoulder line)
        2. Tính góc của vector này so với trục X
        3. Tính tốc độ góc (angle_speed) = derivative của góc
        4. Tìm peak và mở rộng về 2 phía
        
        Args:
            pose_sequence: PoseSequence đã normalize
            
        Returns:
            SwingWindow với method="torso_angle"
        """
        data = pose_sequence.data
        fps = pose_sequence.fps
        
        # Lấy tọa độ 2D của vai trái và vai phải
        left_shoulder = data[:, 11, :2]
        right_shoulder = data[:, 12, :2]
        
        # Vector nối 2 vai
        hip_vector = right_shoulder - left_shoulder
        
        # Tính góc của vector (độ)
        angles = np.degrees(np.arctan2(hip_vector[:, 1], hip_vector[:, 0]))
        
        # Tính tốc độ góc (degrees/second)
        angle_speed = np.abs(np.diff(angles, prepend=angles[0])) * fps
        
        # Tìm peak và thiết lập ngưỡng
        peak = int(np.argmax(angle_speed))
        threshold = 0.25 * float(np.max(angle_speed))
        
        # Ước lượng window dựa trên kinh nghiệm:
        # - Backswing: ~0.5s trước peak
        # - Follow-through: ~0.8s sau peak
        start = max(0, peak - int(0.5 * fps))
        end = min(len(angles) - 1, peak + int(0.8 * fps))
        
        # Tinh chỉnh bằng threshold
        for i in range(peak, -1, -1):
            if angle_speed[i] < threshold:
                start = i
                break
        for i in range(peak, len(angle_speed)):
            if angle_speed[i] < threshold:
                end = i
                break
        
        # Thêm padding
        start = max(0, start - self.padding_margin)
        end = min(len(angles) - 1, end + self.padding_margin)
        
        return SwingWindow(
            start_frame=start, 
            end_frame=end, 
            confidence=float(np.max(angle_speed)), 
            method="torso_angle"
        )

    @staticmethod
    def trim_frames(frames: List[np.ndarray], swing_window: SwingWindow) -> List[np.ndarray]:
        """Cắt frames theo swing window đã phát hiện.
        
        Args:
            frames: List tất cả các frames của video
            swing_window: SwingWindow chứa start_frame và end_frame
            
        Returns:
            List frames đã được cắt [start_frame:end_frame]
        """
        return frames[swing_window.start_frame : swing_window.end_frame]
