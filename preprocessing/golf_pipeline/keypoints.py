from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import urllib.request


@dataclass
class Detection:
    bbox: List[int]
    confidence: float


class GolferDetector:
    """YOLOv8-based golfer detector used to focus skeleton extraction."""

    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        from ultralytics import YOLO  # Lazy import keeps startup light

        self.model = YOLO(model_path)

    def detect_golfers(self, image: np.ndarray, classes: Sequence[int] | None = (0,), conf: float = 0.25) -> List[Detection]:
        results = self.model(image, classes=list(classes) if classes is not None else None, conf=conf)
        detections: List[Detection] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append(
                    Detection(
                        bbox=[int(x1), int(y1), int(x2), int(y2)],
                        confidence=float(box.conf[0].cpu().numpy()),
                    )
                )
        return detections


class SkeletonDetector:
    """MediaPipe Pose detector with GolfPose-style keypoint format."""

    # COCO 17 keypoints format (compatible with GolfPose)
    COCO_KEYPOINT_NAMES = {
        "nose": 0,
        "left_eye": 2,
        "right_eye": 5,
        "left_ear": 7,
        "right_ear": 8,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
    }
    
    # Extended keypoints for full body tracking
    KEYPOINT_NAMES = {
        **COCO_KEYPOINT_NAMES,
        "left_heel": 29,
        "right_heel": 30,
        "left_foot_index": 31,
        "right_foot_index": 32,
    }

    def __init__(self, use_coco_format: bool = False) -> None:
        """
        Initialize MediaPipe Pose detector.
        
        Args:
            use_coco_format: If True, only return 17 COCO keypoints (GolfPose compatible)
                           If False, return all 22 keypoints (full MediaPipe)
        """
        self.use_coco_format = use_coco_format
        
        # Download model if not exists
        model_path = Path("pose_landmarker_heavy.task")
        if not model_path.exists():
            print("Downloading MediaPipe Pose model...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded successfully.")
        
        # Create pose landmarker for video
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            num_poses=1
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def detect_skeleton(self, image: np.ndarray, timestamp_ms: int) -> tuple[Optional[Dict[str, Dict[str, float]]], Optional[list]]:
        """Detect skeleton in video frame.
        
        Args:
            image: BGR image from OpenCV
            timestamp_ms: Frame timestamp in milliseconds (required for VIDEO mode)
            
        Returns:
            landmarks: Dict of keypoint names to coordinates
            pose_landmarks: Raw MediaPipe landmarks for visualization
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect pose landmarks
        results = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None, None
        
        h, w = image.shape[:2]
        pose_landmarks = results.pose_landmarks[0]  # Take first person
        
        # Select keypoint set based on format
        keypoint_dict = self.COCO_KEYPOINT_NAMES if self.use_coco_format else self.KEYPOINT_NAMES
        
        landmarks: Dict[str, Dict[str, float]] = {}
        for name, idx in keypoint_dict.items():
            landmark = pose_landmarks[idx]
            landmarks[name] = {
                "x": landmark.x * w,
                "y": landmark.y * h,
                "z": landmark.z,
                "visibility": landmark.visibility,
            }
        return landmarks, pose_landmarks
    
    def get_golf_specific_features(self, landmarks: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Extract golf-specific features from keypoints (inspired by GolfPose analysis).
        
        Features include:
        - Shoulder rotation angle
        - Hip rotation angle  
        - Spine angle
        - Arm extension
        - Weight distribution
        
        Returns:
            Dictionary of golf-specific metrics
        """
        features = {}
        
        try:
            # Shoulder rotation (XY plane)
            if "left_shoulder" in landmarks and "right_shoulder" in landmarks:
                ls = landmarks["left_shoulder"]
                rs = landmarks["right_shoulder"]
                shoulder_angle = np.degrees(np.arctan2(rs["y"] - ls["y"], rs["x"] - ls["x"]))
                features["shoulder_rotation"] = float(shoulder_angle)
                features["shoulder_width"] = float(np.hypot(rs["x"] - ls["x"], rs["y"] - ls["y"]))
            
            # Hip rotation (XY plane)
            if "left_hip" in landmarks and "right_hip" in landmarks:
                lh = landmarks["left_hip"]
                rh = landmarks["right_hip"]
                hip_angle = np.degrees(np.arctan2(rh["y"] - lh["y"], rh["x"] - lh["x"]))
                features["hip_rotation"] = float(hip_angle)
                features["hip_width"] = float(np.hypot(rh["x"] - lh["x"], rh["y"] - lh["y"]))
            
            # Spine angle (vertical alignment)
            if "left_shoulder" in landmarks and "left_hip" in landmarks:
                ls = landmarks["left_shoulder"]
                lh = landmarks["left_hip"]
                spine_angle = np.degrees(np.arctan2(lh["x"] - ls["x"], lh["y"] - ls["y"]))
                features["spine_angle"] = float(spine_angle)
            
            # Left arm extension (for swing analysis)
            if all(k in landmarks for k in ["left_shoulder", "left_elbow", "left_wrist"]):
                ls = landmarks["left_shoulder"]
                le = landmarks["left_elbow"]
                lw = landmarks["left_wrist"]
                
                upper_arm = np.hypot(le["x"] - ls["x"], le["y"] - ls["y"])
                forearm = np.hypot(lw["x"] - le["x"], lw["y"] - le["y"])
                full_arm = np.hypot(lw["x"] - ls["x"], lw["y"] - ls["y"])
                
                # Elbow angle using law of cosines
                if upper_arm > 0 and forearm > 0:
                    cos_angle = (upper_arm**2 + forearm**2 - full_arm**2) / (2 * upper_arm * forearm)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    elbow_angle = np.degrees(np.arccos(cos_angle))
                    features["left_elbow_angle"] = float(elbow_angle)
                    features["left_arm_extension"] = float(full_arm / (upper_arm + forearm))
            
            # Right arm extension
            if all(k in landmarks for k in ["right_shoulder", "right_elbow", "right_wrist"]):
                rs = landmarks["right_shoulder"]
                re = landmarks["right_elbow"]
                rw = landmarks["right_wrist"]
                
                upper_arm = np.hypot(re["x"] - rs["x"], re["y"] - rs["y"])
                forearm = np.hypot(rw["x"] - re["x"], rw["y"] - re["y"])
                full_arm = np.hypot(rw["x"] - rs["x"], rw["y"] - rs["y"])
                
                if upper_arm > 0 and forearm > 0:
                    cos_angle = (upper_arm**2 + forearm**2 - full_arm**2) / (2 * upper_arm * forearm)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    elbow_angle = np.degrees(np.arccos(cos_angle))
                    features["right_elbow_angle"] = float(elbow_angle)
                    features["right_arm_extension"] = float(full_arm / (upper_arm + forearm))
            
            # Weight distribution (based on hip-ankle alignment)
            if all(k in landmarks for k in ["left_hip", "right_hip", "left_ankle", "right_ankle"]):
                lh = landmarks["left_hip"]
                rh = landmarks["right_hip"]
                la = landmarks["left_ankle"]
                ra = landmarks["right_ankle"]
                
                # Center of hips vs center of ankles (indicates weight shift)
                hip_center_x = (lh["x"] + rh["x"]) / 2
                ankle_center_x = (la["x"] + ra["x"]) / 2
                weight_shift = hip_center_x - ankle_center_x
                features["weight_shift_x"] = float(weight_shift)
            
            # Knee bend angles
            if all(k in landmarks for k in ["left_hip", "left_knee", "left_ankle"]):
                lh = landmarks["left_hip"]
                lk = landmarks["left_knee"]
                la = landmarks["left_ankle"]
                
                upper_leg = np.hypot(lk["x"] - lh["x"], lk["y"] - lh["y"])
                lower_leg = np.hypot(la["x"] - lk["x"], la["y"] - lk["y"])
                full_leg = np.hypot(la["x"] - lh["x"], la["y"] - lh["y"])
                
                if upper_leg > 0 and lower_leg > 0:
                    cos_angle = (upper_leg**2 + lower_leg**2 - full_leg**2) / (2 * upper_leg * lower_leg)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    knee_angle = np.degrees(np.arccos(cos_angle))
                    features["left_knee_angle"] = float(knee_angle)
            
            if all(k in landmarks for k in ["right_hip", "right_knee", "right_ankle"]):
                rh = landmarks["right_hip"]
                rk = landmarks["right_knee"]
                ra = landmarks["right_ankle"]
                
                upper_leg = np.hypot(rk["x"] - rh["x"], rk["y"] - rh["y"])
                lower_leg = np.hypot(ra["x"] - rk["x"], ra["y"] - rk["y"])
                full_leg = np.hypot(ra["x"] - rh["x"], ra["y"] - rh["y"])
                
                if upper_leg > 0 and lower_leg > 0:
                    cos_angle = (upper_leg**2 + lower_leg**2 - full_leg**2) / (2 * upper_leg * lower_leg)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    knee_angle = np.degrees(np.arccos(cos_angle))
                    features["right_knee_angle"] = float(knee_angle)
                    
        except Exception as e:
            print(f"Warning: Could not compute some golf features: {e}")
        
        return features

    def visualize(self, image: np.ndarray, pose_landmarks, landmarks: Optional[Dict] = None) -> np.ndarray:
        """Draw pose landmarks on image with golf-specific connections."""
        vis_image = image.copy()
        
        if pose_landmarks is None and landmarks is None:
            return vis_image
        
        # Define skeleton connections (GolfPose style)
        connections = [
            # Torso
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            # Arms
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            # Legs
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
            # Head
            ('nose', 'left_eye'),
            ('nose', 'right_eye'),
            ('left_eye', 'left_ear'),
            ('right_eye', 'right_ear'),
        ]
        
        if landmarks:
            # Draw using landmark dict
            # Draw connections
            for start, end in connections:
                if start in landmarks and end in landmarks:
                    start_pt = (int(landmarks[start]['x']), int(landmarks[start]['y']))
                    end_pt = (int(landmarks[end]['x']), int(landmarks[end]['y']))
                    
                    # Color coding
                    if 'left' in start or 'left' in end:
                        color = (255, 0, 0)  # Blue for left
                    elif 'right' in start or 'right' in end:
                        color = (0, 255, 0)  # Green for right
                    else:
                        color = (255, 255, 0)  # Cyan for center
                    
                    cv2.line(vis_image, start_pt, end_pt, color, 2)
            
            # Draw keypoints
            for name, coords in landmarks.items():
                x, y = int(coords['x']), int(coords['y'])
                visibility = coords['visibility']
                
                # Color based on visibility
                if visibility > 0.7:
                    color = (0, 255, 255)  # Yellow - high confidence
                elif visibility > 0.4:
                    color = (0, 165, 255)  # Orange - medium
                else:
                    color = (0, 0, 255)  # Red - low
                
                cv2.circle(vis_image, (x, y), 5, color, -1)
                cv2.circle(vis_image, (x, y), 6, (255, 255, 255), 1)  # White outline
        
        elif pose_landmarks:
            # Draw using raw MediaPipe landmarks
            h, w = vis_image.shape[:2]
            for landmark in pose_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(vis_image, (x, y), 5, (0, 255, 0), -1)
        
        return vis_image


class GolfClubDetector:
    """Rule-based golf club detector leveraging wrist landmarks."""

    def detect_club(self, image: np.ndarray, landmarks: Dict[str, Dict[str, float]]) -> Optional[Dict[str, float]]:
        left_wrist = landmarks.get("left_wrist")
        right_wrist = landmarks.get("right_wrist")
        if not left_wrist or not right_wrist:
            return None
        use_left = left_wrist["y"] > right_wrist["y"]
        grip_point = (
            int(left_wrist["x"]) if use_left else int(right_wrist["x"]),
            int(left_wrist["y"]) if use_left else int(right_wrist["y"]),
        )
        roi_size = 200
        x1 = max(0, grip_point[0] - roi_size)
        y1 = max(0, grip_point[1] - roi_size)
        x2 = min(image.shape[1], grip_point[0] + roi_size)
        y2 = min(image.shape[0], grip_point[1] + roi_size)
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
        if lines is None:
            return None
        best_line = None
        max_length = 0.0
        for line in lines:
            x1_l, y1_l, x2_l, y2_l = line[0]
            x1_l += x1
            y1_l += y1
            x2_l += x1
            y2_l += y1
            length = float(np.hypot(x2_l - x1_l, y2_l - y1_l))
            dist_to_grip = min(
                np.hypot(x1_l - grip_point[0], y1_l - grip_point[1]),
                np.hypot(x2_l - grip_point[0], y2_l - grip_point[1]),
            )
            if dist_to_grip < 100 and length > max_length:
                max_length = length
                best_line = (x1_l, y1_l, x2_l, y2_l)
        if not best_line:
            return None
        angle = float(np.degrees(np.arctan2(best_line[3] - best_line[1], best_line[2] - best_line[0])))
        return {
            "x1": best_line[0],
            "y1": best_line[1],
            "x2": best_line[2],
            "y2": best_line[3],
            "angle_degrees": angle,
            "length_px": max_length,
            "grip_x": grip_point[0],
            "grip_y": grip_point[1],
        }


class KeypointCSVExporter:
    """Collect frame-level metadata and persist it as CSV."""

    def __init__(self, output_path: Path | str) -> None:
        self.output_path = Path(output_path)
        self.records: List[Dict[str, float]] = []

    def add_record(
        self,
        frame_idx: int,
        timestamp_ms: float,
        landmarks: Dict[str, Dict[str, float]],
        detections: Sequence[Detection] | None = None,
        club_info: Optional[Dict[str, float]] = None,
        golf_features: Optional[Dict[str, float]] = None,
    ) -> None:
        record: Dict[str, float] = {
            "frame_idx": frame_idx,
            "timestamp_ms": timestamp_ms,
            "golfer_count": float(len(detections) if detections else 0),
        }
        if detections:
            best = max(detections, key=lambda det: det.confidence)
            record.update(
                {
                    "bbox_x1": float(best.bbox[0]),
                    "bbox_y1": float(best.bbox[1]),
                    "bbox_x2": float(best.bbox[2]),
                    "bbox_y2": float(best.bbox[3]),
                    "bbox_confidence": float(best.confidence),
                }
            )
        if club_info:
            record.update(
                {
                    "club_x1": float(club_info["x1"]),
                    "club_y1": float(club_info["y1"]),
                    "club_x2": float(club_info["x2"]),
                    "club_y2": float(club_info["y2"]),
                    "club_angle_degrees": float(club_info["angle_degrees"]),
                    "club_length_px": float(club_info["length_px"]),
                    "club_grip_x": float(club_info["grip_x"]),
                    "club_grip_y": float(club_info["grip_y"]),
                }
            )
        # Add golf-specific features
        if golf_features:
            for key, value in golf_features.items():
                record[f"golf_{key}"] = float(value)
        
        for name, coords in landmarks.items():
            record[f"{name}_x"] = float(coords["x"])
            record[f"{name}_y"] = float(coords["y"])
            record[f"{name}_visibility"] = float(coords["visibility"])
        self.records.append(record)

    def save(self) -> Path:
        if not self.records:
            raise ValueError("No records to save. Ensure add_record was called.")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(self.records)
        frame.sort_values("frame_idx", inplace=True)
        frame.to_csv(self.output_path, index=False)
        return self.output_path


def export_keypoints_to_csv(
    video_path: Path | str,
    output_csv: Path | str,
    frame_skip: int = 1,
    detector: Optional[GolferDetector] = None,
    skeleton_detector: Optional[SkeletonDetector] = None,
    club_detector: Optional[GolfClubDetector] = None,
    extract_golf_features: bool = True,
) -> Path:
    """
    Decode a video, extract skeleton keypoints with golf-specific features, and write them to CSV.
    
    Args:
        video_path: Path to input video
        output_csv: Path to output CSV
        frame_skip: Process every Nth frame
        detector: Optional golfer detector for bounding boxes
        skeleton_detector: Skeleton detector (MediaPipe-based)
        club_detector: Optional golf club detector
        extract_golf_features: If True, compute golf-specific biomechanical features
    
    Returns:
        Path to saved CSV file
    """
    if frame_skip < 1:
        raise ValueError("frame_skip must be >= 1.")
    skeleton_detector = skeleton_detector or SkeletonDetector()
    exporter = KeypointCSVExporter(output_csv)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video at {video_path}.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            detections = detector.detect_golfers(frame) if detector else []
            
            # Calculate timestamp in milliseconds for MediaPipe video mode
            timestamp_ms = int((frame_idx / fps) * 1000.0)
            landmarks, _ = skeleton_detector.detect_skeleton(frame, timestamp_ms)
            
            if not landmarks:
                frame_idx += 1
                continue
            
            # Extract golf-specific features
            golf_features = None
            if extract_golf_features:
                golf_features = skeleton_detector.get_golf_specific_features(landmarks)
            
            club_info = club_detector.detect_club(frame, landmarks) if club_detector else None
            exporter.add_record(frame_idx, float(timestamp_ms), landmarks, detections, club_info, golf_features)
            frame_idx += 1
    finally:
        cap.release()
    return exporter.save()


def export_video_with_keypoints(
    video_path: Path | str,
    output_video: Path | str,
    frame_skip: int = 1,
    detector: Optional[GolferDetector] = None,
    skeleton_detector: Optional[SkeletonDetector] = None,
    club_detector: Optional[GolfClubDetector] = None,
    show_bbox: bool = True,
    show_club: bool = True,
) -> Path:
    """Process video and overlay pose landmarks, bounding boxes, and club detection."""
    if frame_skip < 1:
        raise ValueError("frame_skip must be >= 1.")
    skeleton_detector = skeleton_detector or SkeletonDetector()
    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video at {video_path}.")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            vis_frame = frame.copy()
            
            if frame_idx % frame_skip == 0:
                # Detect golfer bounding box
                detections = detector.detect_golfers(frame) if detector else []
                if detections and show_bbox:
                    best = max(detections, key=lambda det: det.confidence)
                    x1, y1, x2, y2 = best.bbox
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_frame, f'Conf: {best.confidence:.2f}', 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Detect skeleton
                timestamp_ms = int((frame_idx / fps) * 1000.0)
                landmarks, pose_landmarks = skeleton_detector.detect_skeleton(frame, timestamp_ms)
                
                if landmarks:
                    # Draw pose connections
                    connections = [
                        ('left_shoulder', 'right_shoulder'),
                        ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
                        ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
                        ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
                        ('left_hip', 'right_hip'),
                        ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
                        ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
                    ]
                    
                    # Draw connections
                    for start, end in connections:
                        if start in landmarks and end in landmarks:
                            start_pt = (int(landmarks[start]['x']), int(landmarks[start]['y']))
                            end_pt = (int(landmarks[end]['x']), int(landmarks[end]['y']))
                            cv2.line(vis_frame, start_pt, end_pt, (255, 0, 0), 2)
                    
                    # Draw keypoints
                    for name, coords in landmarks.items():
                        x, y = int(coords['x']), int(coords['y'])
                        visibility = coords['visibility']
                        color = (0, 255, 255) if visibility > 0.5 else (0, 128, 255)
                        cv2.circle(vis_frame, (x, y), 4, color, -1)
                    
                    # Detect and draw golf club
                    if club_detector and show_club:
                        club_info = club_detector.detect_club(frame, landmarks)
                        if club_info:
                            x1, y1 = int(club_info['x1']), int(club_info['y1'])
                            x2, y2 = int(club_info['x2']), int(club_info['y2'])
                            grip_x, grip_y = int(club_info['grip_x']), int(club_info['grip_y'])
                            
                            # Draw club line
                            cv2.line(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            # Draw grip point
                            cv2.circle(vis_frame, (grip_x, grip_y), 8, (255, 0, 255), -1)
                            # Show angle
                            angle_text = f"Angle: {club_info['angle_degrees']:.1f}°"
                            cv2.putText(vis_frame, angle_text, (grip_x + 10, grip_y - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Add frame number
            cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(vis_frame)
            frame_idx += 1
            
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx} frames...", end='\r')
    finally:
        cap.release()
        out.release()
        print(f"\nVideo processing complete: {frame_idx} frames")
    
    return output_path
