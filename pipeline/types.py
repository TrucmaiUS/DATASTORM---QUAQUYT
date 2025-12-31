"""Các dataclass chung cho pipeline xử lý golf swing.

Module này định nghĩa các cấu trúc dữ liệu cốt lõi được sử dụng
xuyên suốt pipeline từ video → pose → features → dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass
class PoseSequence:
    """Đại diện cho một chuỗi pose (skeleton) qua thời gian.
    
    Đây là cấu trúc dữ liệu chính cho pose landmarks được trích xuất
    từ MediaPipe. Mỗi frame có 33 keypoints, mỗi keypoint có 4 giá trị.
    
    Attributes:
        data: Array chứa pose data, shape (T, 33, 4)
            - T: số lượng frames
            - 33: số keypoints (MediaPipe pose landmarks)
            - 4: [x, y, z, visibility]
                * x, y, z: tọa độ không gian đã normalize
                * visibility: độ tin cậy của detection (0-1)
        
        frame_times: Timestamp của mỗi frame tính bằng giây, shape (T,)
            Dùng để track thời gian thực của video
        
        fps: Frame rate của video gốc (frames per second)
            Quan trọng để tính velocity và acceleration chính xác
        
        interpolation_mask: Mask đánh dấu keypoints đã được interpolate, shape (T, 33)
            True = keypoint này bị thiếu ở frame gốc và đã được nội suy
            False = keypoint có sẵn từ detection
        
        valid_mask: Mask đánh dấu frames hợp lệ, shape (T,)
            True = frame này có ít nhất 1 pose được detect
            False = frame không có pose nào
    """
    data: np.ndarray  # [T, 33, 4]
    frame_times: np.ndarray  # seconds
    fps: float
    interpolation_mask: np.ndarray  # [T, 33] bool where interpolated
    valid_mask: np.ndarray  # [T] bool


@dataclass
class SwingWindow:
    """Định nghĩa cửa sổ thời gian của một golf swing.
    
    Swing window xác định khoảng thời gian (bắt đầu → kết thúc) chứa
    chuyển động swing chính. Được phát hiện tự động dựa trên:
    1. Wrist velocity (ưu tiên)
    2. Torso rotation (fallback)
    
    Attributes:
        start_frame: Index của frame bắt đầu swing (inclusive)
        end_frame: Index của frame kết thúc swing (exclusive)
        confidence: Độ tin cậy của detection (0-1)
            Càng cao = chuyển động swing càng rõ ràng
        method: Phương pháp phát hiện được sử dụng
            "wrist_velocity" hoặc "torso_angle"
    """
    start_frame: int
    end_frame: int
    confidence: float
    method: str = "wrist_velocity"

    @property
    def num_frames(self) -> int:
        """Tính số lượng frames trong swing window."""
        return max(0, self.end_frame - self.start_frame)


@dataclass
class QualityMetrics:
    """Các chỉ số đánh giá chất lượng của pose sequence.
    
    Dùng để filter các sample kém chất lượng (occlusion, motion blur, etc.)
    và quyết định có nên augment sample hay không.
    
    Attributes:
        valid_ratio: Tỉ lệ frames có pose hợp lệ (0-1)
            Ví dụ: 0.85 = 85% frames có ít nhất 1 keypoint visible
        
        mean_visibility: Visibility trung bình của các keypoints quan trọng
            Được tính từ KEY_JOINTS (hips, shoulders, elbows, wrists, etc.)
            Cao = các điểm quan trọng được nhìn thấy rõ
        
        keypoint_visibility: Dict chứa visibility riêng của từng nhóm keypoint
            Ví dụ: {"hips": 0.95, "wrists": 0.72, ...}
        
        longest_dropout: Độ dài (frames) của khoảng thiếu dữ liệu dài nhất
            Ví dụ: 5 = có 5 frames liên tiếp không detect được pose
        
        low_quality: Flag đánh dấu sample có chất lượng thấp (True/False)
            Được quyết định dựa trên THRESHOLDS trong config.py
            True = không nên dùng cho augmentation
    """
    valid_ratio: float
    mean_visibility: float
    keypoint_visibility: Dict[str, float]
    longest_dropout: int
    low_quality: bool


@dataclass
class SamplePayload:
    """Metadata của một sample đã xử lý và lưu vào disk.
    
    Mỗi sample là một golf swing đã qua toàn bộ pipeline và được
    lưu dưới dạng .npz files. Payload này track các thông tin cần thiết
    để load lại data và metadata cho training.
    
    Attributes:
        video_id: ID của video gốc
            Ví dụ: "Backside-8748-3" hoặc "Backside-8748-3__gaussian" (augmented)
        
        env: Môi trường swing (indoor/outdoor)
            Được suy ra từ cấu trúc thư mục hoặc metadata
        
        band: Mức độ khó (handicap band)
            Ví dụ: "1_2", "8_10"
            Đây là target label cho classification
        
        sequence_path: Đường dẫn tới file .npz chứa features
            Format: processed_videos/sequences/{video_id}__{env}__{band}.npz
            Chứa: X (features), y (band), env, video_id, quality metrics
        
        interpolation_mask_path: Đường dẫn tới file .npz chứa interpolation mask
            Dùng để track keypoints nào đã bị interpolate
            Hữu ích cho debugging và weighted loss
        
        metadata: Dict chứa các thông tin bổ sung
            Ví dụ: {"trimmed_duration_s": 2.5, "augmented": True, "strategy": "gaussian"}
    """
    video_id: str
    env: str
    band: str
    sequence_path: Path
    interpolation_mask_path: Path
    metadata: Dict[str, float] = field(default_factory=dict)
