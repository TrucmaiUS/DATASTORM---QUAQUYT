# Golfer Detection and Analysis Pipeline

A complete pipeline for detecting golfers, extracting skeleton keypoints (including golf club detection), data augmentation, and preprocessing.

---

## Table of Contents
1. [Setup and Dependencies](#setup-and-dependencies)
2. [Part 1: Golfer Detection with Bounding Box](#part-1-golfer-detection-with-bounding-box)
3. [Part 2: Skeleton Detection](#part-2-skeleton-detection)
4. [Part 3: Golf Club Detection](#part-3-golf-club-detection)
5. [Part 4: Export to CSV](#part-4-export-to-csv)
6. [Part 5: Data Augmentation](#part-5-data-augmentation)
7. [Part 6: Preprocessing - Style Matching](#part-6-preprocessing-style-matching)
8. [Complete Pipeline](#complete-pipeline)

---

## Setup and Dependencies

```python
# Install required packages
!pip install opencv-python mediapipe numpy pandas matplotlib pillow ultralytics scikit-image

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
```

---

## Part 1: Golfer Detection with Bounding Box

### Explanation
We'll use YOLOv8 or a pre-trained person detector to identify golfers in the frame and draw bounding boxes around them.

```python
from ultralytics import YOLO

class GolferDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the golfer detector with YOLOv8
        Args:
            model_path: Path to YOLO model (default uses nano model)
        """
        self.model = YOLO(model_path)
        
    def detect_golfer(self, image):
        """
        Detect golfers (persons) in the image
        Returns: List of bounding boxes [x1, y1, x2, y2, confidence]
        """
        results = self.model(image, classes=[0])  # Class 0 = person
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(confidence)
                })
        
        return detections
    
    def visualize_detections(self, image, detections):
        """
        Draw bounding boxes on the image
        """
        vis_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add confidence label
            label = f"Golfer: {conf:.2f}"
            cv2.putText(vis_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis_image

# Example Usage
detector = GolferDetector()
image = cv2.imread('golfer_image.jpg')
detections = detector.detect_golfer(image)
vis_image = detector.visualize_detections(image, detections)

# Display
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
plt.title('Golfer Detection with Bounding Box')
plt.axis('off')
plt.show()

print(f"Detected {len(detections)} golfer(s)")
for i, det in enumerate(detections):
    print(f"Golfer {i+1}: Confidence = {det['confidence']:.2f}, BBox = {det['bbox']}")
```

### Visualization Output
- Green bounding box around detected golfer
- Confidence score displayed above the box

---

## Part 2: Skeleton Detection

### Explanation
Using MediaPipe Pose to detect 33 key body landmarks including:
- **Head**: Nose, eyes, ears
- **Shoulders**: Left and right shoulder
- **Arms/Hands**: Elbows, wrists
- **Torso**: Hips
- **Legs/Feet**: Knees, ankles, heels, foot index

```python
class SkeletonDetector:
    def __init__(self):
        """
        Initialize MediaPipe Pose detector
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def detect_skeleton(self, image):
        """
        Detect skeleton keypoints
        Returns: Dictionary with landmark coordinates
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return None
        
        h, w = image.shape[:2]
        landmarks = {}
        
        # Define key body parts
        keypoint_names = {
            'nose': 0, 'left_eye': 2, 'right_eye': 5,
            'left_ear': 7, 'right_ear': 8,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_heel': 29, 'right_heel': 30,
            'left_foot_index': 31, 'right_foot_index': 32
        }
        
        for name, idx in keypoint_names.items():
            landmark = results.pose_landmarks.landmark[idx]
            landmarks[name] = {
                'x': landmark.x * w,
                'y': landmark.y * h,
                'z': landmark.z,
                'visibility': landmark.visibility
            }
        
        return landmarks, results.pose_landmarks
    
    def visualize_skeleton(self, image, pose_landmarks):
        """
        Draw skeleton on the image
        """
        vis_image = image.copy()
        
        if pose_landmarks:
            self.mp_drawing.draw_landmarks(
                vis_image,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        return vis_image

# Example Usage
skeleton_detector = SkeletonDetector()
image = cv2.imread('golfer_image.jpg')
landmarks, pose_landmarks = skeleton_detector.detect_skeleton(image)
vis_image = skeleton_detector.visualize_skeleton(image, pose_landmarks)

# Display
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
plt.title('Skeleton Detection - Body Keypoints')
plt.axis('off')
plt.show()

# Print key landmarks
if landmarks:
    print("Key Body Parts:")
    for part in ['nose', 'left_shoulder', 'right_shoulder', 'left_wrist', 
                 'right_wrist', 'left_ankle', 'right_ankle']:
        if part in landmarks:
            print(f"{part}: x={landmarks[part]['x']:.1f}, y={landmarks[part]['y']:.1f}, "
                  f"visibility={landmarks[part]['visibility']:.2f}")
```

### Visualization Output
- Colored skeleton overlay on the golfer
- 33 keypoints connected with lines showing body structure

---

## Part 3: Golf Club Detection

### Explanation
Golf club detection combines edge detection, line detection (Hough Transform), and geometric analysis based on wrist position to identify the club.

```python
class GolfClubDetector:
    def __init__(self):
        """
        Initialize golf club detector
        """
        pass
    
    def detect_club(self, image, landmarks):
        """
        Detect golf club using edge detection and geometric analysis
        Args:
            image: Input image
            landmarks: Skeleton landmarks (especially wrists)
        Returns: Club line coordinates and angle
        """
        if not landmarks:
            return None
        
        # Get wrist positions
        left_wrist = landmarks.get('left_wrist')
        right_wrist = landmarks.get('right_wrist')
        
        if not left_wrist or not right_wrist:
            return None
        
        # Determine dominant hand (lower wrist typically holds club)
        if left_wrist['y'] > right_wrist['y']:
            grip_point = (int(left_wrist['x']), int(left_wrist['y']))
        else:
            grip_point = (int(right_wrist['x']), int(right_wrist['y']))
        
        # Create region of interest around grip point
        roi_size = 
        x1 = max(0, grip_point[0] - roi_size)
        y1 = max(0, grip_point[1] - roi_size)
        x2 = min(image.shape[1], grip_point[0] + roi_size)
        y2 = min(image.shape[0], grip_point[1] + roi_size)
        
        roi = image[y1:y2, x1:x2]
        
        # Edge detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Hough Line Transform to detect straight lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return None
        
        # Find the longest line near grip point (likely the club shaft)
        best_line = None
        max_length = 0
        
        for line in lines:
            x1_l, y1_l, x2_l, y2_l = line[0]
            # Convert back to original image coordinates
            x1_l += x1
            y1_l += y1
            x2_l += x1
            y2_l += y1
            
            # Calculate distance to grip point
            dist_to_grip = min(
                np.sqrt((x1_l - grip_point[0])**2 + (y1_l - grip_point[1])**2),
                np.sqrt((x2_l - grip_point[0])**2 + (y2_l - grip_point[1])**2)
            )
            
            # Calculate line length
            length = np.sqrt((x2_l - x1_l)**2 + (y2_l - y1_l)**2)
            
            # Select line that's long and close to grip
            if dist_to_grip < 100 and length > max_length:
                max_length = length
                best_line = (x1_l, y1_l, x2_l, y2_l)
        
        if best_line:
            # Calculate club angle
            x1_c, y1_c, x2_c, y2_c = best_line
            angle = np.degrees(np.arctan2(y2_c - y1_c, x2_c - x1_c))
            
            return {
                'line': best_line,
                'angle': angle,
                'length': max_length,
                'grip_point': grip_point
            }
        
        return None
    
    def visualize_club(self, image, club_info):
        """
        Draw the detected golf club on the image
        """
        vis_image = image.copy()
        
        if club_info:
            x1, y1, x2, y2 = club_info['line']
            
            # Draw club shaft
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Draw grip point
            cv2.circle(vis_image, club_info['grip_point'], 8, (255, 0, 0), -1)
            
            # Add angle text
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            text = f"Club Angle: {club_info['angle']:.1f}°"
            cv2.putText(vis_image, text, (mid_x, mid_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return vis_image

# Example Usage
club_detector = GolfClubDetector()
image = cv2.imread('golfer_image.jpg')

# First detect skeleton to get wrist positions
landmarks, pose_landmarks = skeleton_detector.detect_skeleton(image)

# Then detect club
club_info = club_detector.detect_club(image, landmarks)
vis_image = club_detector.visualize_club(image, club_info)

# Display
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
plt.title('Golf Club Detection')
plt.axis('off')
plt.show()

if club_info:
    print(f"Club detected: Angle = {club_info['angle']:.1f}°, Length = {club_info['length']:.1f}px")
else:
    print("No club detected")
```

### Visualization Output
- Red line showing detected club shaft
- Blue circle at grip point
- Angle measurement displayed

---

## Part 4: Export to CSV

### Explanation
Export all detected information (bounding boxes, skeleton keypoints, club data) to a structured CSV file for further analysis.

```python
class DataExporter:
    def __init__(self, output_dir='output'):
        """
        Initialize data exporter
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.data_records = []
        
    def add_frame_data(self, frame_id, timestamp, detections, landmarks, club_info, image_path=None):
        """
        Add data for a single frame
        """
        record = {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'image_path': image_path
        }
        
        # Add bounding box data
        if detections and len(detections) > 0:
            bbox = detections[0]['bbox']
            record['bbox_x1'] = bbox[0]
            record['bbox_y1'] = bbox[1]
            record['bbox_x2'] = bbox[2]
            record['bbox_y2'] = bbox[3]
            record['detection_confidence'] = detections[0]['confidence']
        
        # Add skeleton landmarks
        if landmarks:
            for part_name, coords in landmarks.items():
                record[f'{part_name}_x'] = coords['x']
                record[f'{part_name}_y'] = coords['y']
                record[f'{part_name}_z'] = coords['z']
                record[f'{part_name}_visibility'] = coords['visibility']
        
        # Add club data
        if club_info:
            record['club_angle'] = club_info['angle']
            record['club_length'] = club_info['length']
            record['club_x1'] = club_info['line'][0]
            record['club_y1'] = club_info['line'][1]
            record['club_x2'] = club_info['line'][2]
            record['club_y2'] = club_info['line'][3]
            record['grip_x'] = club_info['grip_point'][0]
            record['grip_y'] = club_info['grip_point'][1]
        
        self.data_records.append(record)
        
    def export_to_csv(self, filename='golfer_data.csv'):
        """
        Export all collected data to CSV
        """
        if not self.data_records:
            print("No data to export")
            return
        
        df = pd.DataFrame(self.data_records)
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        
        print(f"Exported {len(self.data_records)} records to {output_path}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        return df
    
    def export_summary_stats(self, filename='summary_stats.json'):
        """
        Export summary statistics
        """
        if not self.data_records:
            return
        
        df = pd.DataFrame(self.data_records)
        
        stats = {
            'total_frames': len(df),
            'frames_with_detection': df['bbox_x1'].notna().sum(),
            'frames_with_skeleton': df.filter(like='_x').notna().any(axis=1).sum(),
            'frames_with_club': df['club_angle'].notna().sum() if 'club_angle' in df else 0,
            'avg_detection_confidence': df['detection_confidence'].mean() if 'detection_confidence' in df else None,
            'avg_club_angle': df['club_angle'].mean() if 'club_angle' in df else None,
            'club_angle_range': {
                'min': df['club_angle'].min() if 'club_angle' in df else None,
                'max': df['club_angle'].max() if 'club_angle' in df else None
            }
        }
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Summary statistics saved to {output_path}")
        return stats

# Example Usage
exporter = DataExporter(output_dir='output')

# Process multiple frames
for frame_id in range(100):  # Example: 100 frames
    # Load image
    image = cv2.imread(f'frame_{frame_id}.jpg')
    timestamp = datetime.now().isoformat()
    
    # Detect
    detections = detector.detect_golfer(image)
    landmarks, pose_landmarks = skeleton_detector.detect_skeleton(image)
    club_info = club_detector.detect_club(image, landmarks) if landmarks else None
    
    # Add to exporter
    exporter.add_frame_data(
        frame_id=frame_id,
        timestamp=timestamp,
        detections=detections,
        landmarks=landmarks,
        club_info=club_info,
        image_path=f'frame_{frame_id}.jpg'
    )

# Export
df = exporter.export_to_csv('golfer_analysis.csv')
stats = exporter.export_summary_stats('analysis_summary.json')
```

### CSV Output Structure
```
frame_id, timestamp, bbox_x1, bbox_y1, bbox_x2, bbox_y2, detection_confidence,
nose_x, nose_y, nose_z, nose_visibility,
left_shoulder_x, left_shoulder_y, left_shoulder_z, left_shoulder_visibility,
right_shoulder_x, right_shoulder_y, right_shoulder_z, right_shoulder_visibility,
left_wrist_x, left_wrist_y, left_wrist_z, left_wrist_visibility,
... (all keypoints)
club_angle, club_length, club_x1, club_y1, club_x2, club_y2, grip_x, grip_y
```

---

## Part 5: Data Augmentation

### Explanation
Expand training dataset by applying various transformations: brightness adjustment, contrast, rotation, zoom, flip, noise, etc.

```python
from PIL import Image, ImageEnhance
import albumentations as A

class DataAugmentor:
    def __init__(self):
        """
        Initialize data augmentation pipeline
        """
        # Define augmentation pipeline using albumentations
        self.transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomScale(scale_limit=0.2, p=0.5),
        ])
        
    def augment_brightness(self, image, factor=1.5):
        """
        Adjust brightness
        Args:
            factor: >1 brightens, <1 darkens (0.5 to 2.0 recommended)
        """
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = enhancer.enhance(factor)
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    def augment_contrast(self, image, factor=1.5):
        """
        Adjust contrast
        Args:
            factor: >1 increases, <1 decreases (0.5 to 2.0 recommended)
        """
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(factor)
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    def augment_zoom(self, image, zoom_factor=1.2):
        """
        Zoom in/out on image
        Args:
            zoom_factor: >1 zooms in, <1 zooms out
        """
        h, w = image.shape[:2]
        new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
        
        # Calculate crop coordinates (center crop)
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        
        cropped = image[top:top+new_h, left:left+new_w]
        zoomed = cv2.resize(cropped, (w, h))
        
        return zoomed
    
    def augment_rotation(self, image, angle=10):
        """
        Rotate image
        Args:
            angle: Rotation angle in degrees
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h))
        return rotated
    
    def augment_flip(self, image, flip_code=1):
        """
        Flip image
        Args:
            flip_code: 1=horizontal, 0=vertical, -1=both
        """
        return cv2.flip(image, flip_code)
    
    def augment_noise(self, image, noise_level=25):
        """
        Add Gaussian noise
        """
        noise = np.random.normal(0, noise_level, image.shape).astype(np.uint8)
        noisy = cv2.add(image, noise)
        return noisy
    
    def augment_all_variations(self, image, output_dir='augmented'):
        """
        Generate all augmentation variations and save them
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        augmentations = {
            'original': image,
            'brighten': self.augment_brightness(image, 1.5),
            'darken': self.augment_brightness(image, 0.6),
            'high_contrast': self.augment_contrast(image, 1.8),
            'low_contrast': self.augment_contrast(image, 0.6),
            'zoom_in': self.augment_zoom(image, 1.3),
            'zoom_out': self.augment_zoom(image, 0.8),
            'rotate_left': self.augment_rotation(image, -15),
            'rotate_right': self.augment_rotation(image, 15),
            'flip_horizontal': self.augment_flip(image, 1),
            'noise': self.augment_noise(image, 30)
        }
        
        # Visualize all augmentations
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.ravel()
        
        for idx, (name, aug_image) in enumerate(augmentations.items()):
            if idx < len(axes):
                axes[idx].imshow(cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB))
                axes[idx].set_title(name.replace('_', ' ').title())
                axes[idx].axis('off')
                
                # Save augmented image
                cv2.imwrite(str(output_path / f'{name}.jpg'), aug_image)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Generated {len(augmentations)} augmented variations")
        return augmentations
    
    def augment_batch(self, image_list, augmentations_per_image=5):
        """
        Apply random augmentations to a batch of images
        """
        augmented_batch = []
        
        for img in image_list:
            augmented_batch.append(img)  # Original
            
            for _ in range(augmentations_per_image):
                # Apply random augmentation using albumentations
                augmented = self.transform(image=img)['image']
                augmented_batch.append(augmented)
        
        print(f"Original batch: {len(image_list)} images")
        print(f"Augmented batch: {len(augmented_batch)} images")
        print(f"Expansion factor: {len(augmented_batch) / len(image_list):.1f}x")
        
        return augmented_batch

# Example Usage
augmentor = DataAugmentor()

# Load sample image
image = cv2.imread('golfer_image.jpg')

# Generate all variations
augmented_images = augmentor.augment_all_variations(image, output_dir='augmented_data')

# Or augment a batch
image_batch = [cv2.imread(f'image_{i}.jpg') for i in range(10)]
augmented_batch = augmentor.augment_batch(image_batch, augmentations_per_image=5)
```

### Visualization Output
- Grid showing all augmentation variations side by side
- Each variation labeled with its transformation type
- Before/after comparison

---

## Part 6: Preprocessing - Style Matching

### Explanation
Normalize images to match the brightness, contrast, and color profile of training data. This ensures consistency across datasets.

```python
class StylePreprocessor:
    def __init__(self, reference_image=None):
        """
        Initialize style preprocessor with a reference image
        Args:
            reference_image: The target style image to match
        """
        self.reference_image = reference_image
        self.reference_stats = None
        
        if reference_image is not None:
            self.compute_reference_stats(reference_image)
    
    def compute_reference_stats(self, image):
        """
        Compute color statistics of reference image
        """
        # Convert to LAB color space for better color matching
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        self.reference_stats = {
            'mean_l': np.mean(lab[:, :, 0]),
            'std_l': np.std(lab[:, :, 0]),
            'mean_a': np.mean(lab[:, :, 1]),
            'std_a': np.std(lab[:, :, 1]),
            'mean_b': np.mean(lab[:, :, 2]),
            'std_b': np.std(lab[:, :, 2])
        }
        
        return self.reference_stats
    
    def match_color_style(self, image):
        """
        Transfer color style from reference image to input image
        """
        if self.reference_stats is None:
            print("No reference image set. Using default preprocessing.")
            return self.normalize_basic(image)
        
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Compute input statistics
        input_mean_l, input_std_l = np.mean(lab[:, :, 0]), np.std(lab[:, :, 0])
        input_mean_a, input_std_a = np.mean(lab[:, :, 1]), np.std(lab[:, :, 1])
        input_mean_b, input_std_b = np.mean(lab[:, :, 2]), np.std(lab[:, :, 2])
        
        # Match statistics to reference
        lab[:, :, 0] = ((lab[:, :, 0] - input_mean_l) * 
                        (self.reference_stats['std_l'] / input_std_l) + 
                        self.reference_stats['mean_l'])
        
        lab[:, :, 1] = ((lab[:, :, 1] - input_mean_a) * 
                        (self.reference_stats['std_a'] / input_std_a) + 
                        self.reference_stats['mean_a'])
        
        lab[:, :, 2] = ((lab[:, :, 2] - input_mean_b) * 
                        (self.reference_stats['std_b'] / input_std_b) + 
                        self.reference_stats['mean_b'])
        
        # Clip and convert back
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def normalize_basic(self, image, target_mean=128, target_std=50):
        """
        Basic normalization to standard brightness/contrast
        """
        # Convert to float
        img_float = image.astype(np.float32)
        
        # Normalize each channel
        for i in range(3):
            channel = img_float[:, :, i]
            channel_mean = np.mean(channel)
            channel_std = np.std(channel)
            
            # Standardize
            if channel_std > 0:
                img_float[:, :, i] = ((channel - channel_mean) / channel_std) * target_std + target_mean
        
        # Clip and convert back
        result = np.clip(img_float, 0, 255).astype(np.uint8)
        return result
    
    def apply_clahe(self, image, clip_limit=2.0, tile_size=(8, 8)):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Improves local contrast
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return result
    
    def preprocess_for_training(self, image, apply_clahe=True):
        """
        Complete preprocessing pipeline for training data
        """
        # Step 1: Match color style to reference
        processed = self.match_color_style(image)
        
        # Step 2: Apply CLAHE for better contrast
        if apply_clahe:
            processed = self.apply_clahe(processed)
        
        return processed
    
    def visualize_preprocessing(self, image):
        """
        Visualize preprocessing steps
        """
        steps = {
            'Original': image,
            'Color Matched': self.match_color_style(image),
            'Basic Normalized': self.normalize_basic(image),
            'CLAHE Applied': self.apply_clahe(image),
            'Full Pipeline': self.preprocess_for_training(image)
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, (name, img) in enumerate(steps.items()):
            if idx < len(axes):
                axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                axes[idx].set_title(name)
                axes[idx].axis('off')
                
                # Show histogram
                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                axes[idx].plot(hist, color='gray', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return steps

# Example Usage
# Load reference image (representative of training data style)
reference_image = cv2.imread('training_sample.jpg')
preprocessor = StylePreprocessor(reference_image)

# Preprocess a new image
image = cv2.imread('new_golfer_image.jpg')
preprocessed = preprocessor.preprocess_for_training(image)

# Visualize
steps = preprocessor.visualize_preprocessing(image)

# Save preprocessed image
cv2.imwrite('preprocessed_output.jpg', preprocessed)

print("Preprocessing complete!")
print(f"Reference stats: {preprocessor.reference_stats}")
```

### Visualization Output
- Side-by-side comparison of preprocessing steps
- Histogram overlay showing brightness distribution changes
- Before/after color profile matching

---

## Complete Pipeline

### Explanation
Integrate all components into a single end-to-end pipeline for processing images or videos.

```python
class GolferAnalysisPipeline:
    def __init__(self, reference_image=None, output_dir='pipeline_output'):
        """
        Complete pipeline for golfer analysis
        """
        self.detector = GolferDetector()
        self.skeleton_detector = SkeletonDetector()
        self.club_detector = GolfClubDetector()
        self.augmentor = DataAugmentor()
        self.preprocessor = StylePreprocessor(reference_image)
        self.exporter = DataExporter(output_dir)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def process_image(self, image_path, frame_id=0, save_visualization=True):
        """
        Process a single image through the complete pipeline
        """
        # Load and preprocess
        image = cv2.imread(str(image_path))
        preprocessed = self.preprocessor.preprocess_for_training(image)
        
        # Detection
        detections = self.detector.detect_golfer(preprocessed)
        landmarks, pose_landmarks = self.skeleton_detector.detect_skeleton(preprocessed)
        club_info = self.club_detector.detect_club(preprocessed, landmarks) if landmarks else None
        
        # Visualization
        if save_visualization:
            vis_image = preprocessed.copy()
            vis_image = self.detector.visualize_detections(vis_image, detections)
            vis_image = self.skeleton_detector.visualize_skeleton(vis_image, pose_landmarks)
            vis_image = self.club_detector.visualize_club(vis_image, club_info)
            
            output_path = self.output_dir / f'frame_{frame_id:04d}_analyzed.jpg'
            cv2.imwrite(str(output_path), vis_image)
        
        # Export data
        self.exporter.add_frame_data(
            frame_id=frame_id,
            timestamp=datetime.now().isoformat(),
            detections=detections,
            landmarks=landmarks,
            club_info=club_info,
            image_path=str(image_path)
        )
        
        return {
            'detections': detections,
            'landmarks': landmarks,
            'club_info': club_info,
            'preprocessed': preprocessed
        }
    
    def process_video(self, video_path, sample_rate=1):
        """
        Process video through the pipeline
        Args:
            video_path: Path to video file
            sample_rate: Process every Nth frame (1 = all frames)
        """
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Processing video: {frame_count} frames at {fps} FPS")
        print(f"Sampling every {sample_rate} frame(s)")
        
        frame_id = 0
        processed_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_id % sample_rate == 0:
                # Save frame temporarily
                temp_path = self.output_dir / f'temp_frame_{frame_id}.jpg'
                cv2.imwrite(str(temp_path), frame)
                
                # Process
                self.process_image(temp_path, frame_id, save_visualization=True)
                processed_count += 1
                
                # Clean up temp file
                temp_path.unlink()
                
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} frames...")
            
            frame_id += 1
        
        cap.release()
        print(f"Video processing complete! Processed {processed_count} frames")
        
        # Export results
        df = self.exporter.export_to_csv('video_analysis.csv')
        stats = self.exporter.export_summary_stats('video_summary.json')
        
        return df, stats
    
    def process_batch_with_augmentation(self, image_paths, augment=True, augmentations_per_image=5):
        """
        Process batch of images with optional augmentation
        """
        print(f"Processing batch of {len(image_paths)} images...")
        
        all_images = []
        for path in image_paths:
            img = cv2.imread(str(path))
            all_images.append((path, img))
        
        # Augment if requested
        if augment:
            print(f"Generating {augmentations_per_image} augmentations per image...")
            augmented_images = []
            
            for path, img in all_images:
                augmented_images.append((path, img))  # Original
                
                for aug_id in range(augmentations_per_image):
                    aug_img = self.augmentor.transform(image=img)['image']
                    aug_path = f"{path.stem}_aug_{aug_id}{path.suffix}"
                    augmented_images.append((aug_path, aug_img))
            
            all_images = augmented_images
            print(f"Total images after augmentation: {len(all_images)}")
        
        # Process all images
        for idx, (path, img) in enumerate(all_images):
            # Save temp image
            temp_path = self.output_dir / f'temp_{idx}.jpg'
            cv2.imwrite(str(temp_path), img)
            
            # Process
            self.process_image(temp_path, frame_id=idx, save_visualization=True)
            
            # Clean up
            temp_path.unlink()
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(all_images)} images...")
        
        # Export
        df = self.exporter.export_to_csv('batch_analysis.csv')
        stats = self.exporter.export_summary_stats('batch_summary.json')
        
        return df, stats

# Example Usage - Complete Pipeline

# Initialize pipeline with reference image for style matching
reference_img = cv2.imread('training_reference.jpg')
pipeline = GolferAnalysisPipeline(reference_image=reference_img, output_dir='final_output')

# Option 1: Process single image
result = pipeline.process_image('golfer_swing.jpg', frame_id=0)
print(f"Detected: {len(result['detections'])} golfer(s)")

# Option 2: Process video
df_video, stats_video = pipeline.process_video('golf_swing_video.mp4', sample_rate=5)

# Option 3: Process batch with augmentation
image_list = [Path('dataset') / f'image_{i}.jpg' for i in range(20)]
df_batch, stats_batch = pipeline.process_batch_with_augmentation(
    image_list, 
    augment=True, 
    augmentations_per_image=5
)

print("\n=== Pipeline Complete ===")
print(f"Total frames processed: {len(df_batch)}")
print(f"Output saved to: {pipeline.output_dir}")
```

### Complete Pipeline Visualization

```python
# Create comprehensive visualization report
def create_analysis_report(pipeline_output_dir):
    """
    Generate a visual report of all pipeline outputs
    """
    output_path = Path(pipeline_output_dir)
    
    # Load all analyzed images
    analyzed_images = sorted(output_path.glob('frame_*_analyzed.jpg'))
    
    if len(analyzed_images) == 0:
        print("No analyzed images found")
        return
    
    # Display grid of results
    n_images = min(12, len(analyzed_images))
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.ravel()
    
    for idx in range(n_images):
        img = cv2.imread(str(analyzed_images[idx]))
        axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(f'Frame {idx}')
        axes[idx].axis('off')
    
    plt.suptitle('Golfer Analysis Pipeline - Sample Results', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path / 'analysis_report.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Load and display CSV statistics
    csv_files = list(output_path.glob('*.csv'))
    if csv_files:
        df = pd.read_csv(csv_files[0])
        
        print("\n=== Analysis Statistics ===")
        print(f"Total frames: {len(df)}")
        print(f"Frames with detection: {df['bbox_x1'].notna().sum()}")
        
        if 'club_angle' in df.columns:
            club_data = df['club_angle'].dropna()
            print(f"\nClub Angle Statistics:")
            print(f"  Mean: {club_data.mean():.1f}°")
            print(f"  Std: {club_data.std():.1f}°")
            print(f"  Range: [{club_data.min():.1f}°, {club_data.max():.1f}°]")
            
            # Plot club angle over time
            plt.figure(figsize=(12, 6))
            plt.plot(club_data.index, club_data.values, marker='o', linewidth=2)
            plt.xlabel('Frame')
            plt.ylabel('Club Angle (degrees)')
            plt.title('Golf Club Angle Trajectory')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_path / 'club_angle_trajectory.png', dpi=150, bbox_inches='tight')
            plt.show()

# Generate report
create_analysis_report('final_output')
```

---

## Summary

### Pipeline Components

1. **Golfer Detection**: YOLOv8-based person detection with bounding boxes
2. **Skeleton Detection**: MediaPipe Pose for 33 body keypoints
3. **Club Detection**: Edge detection + Hough Transform for club tracking
4. **Data Export**: CSV export with all metrics
5. **Data Augmentation**: 10+ augmentation techniques for dataset expansion
6. **Style Preprocessing**: Color/contrast normalization to match training data

### Output Files

- `golfer_analysis.csv`: Complete frame-by-frame data
- `analysis_summary.json`: Summary statistics
- `frame_XXXX_analyzed.jpg`: Visualized detections
- `club_angle_trajectory.png`: Swing analysis chart
- `analysis_report.png`: Grid of results

### Key Features

✅ Detect multiple golfers in frame  
✅ Track 33 skeleton keypoints  
✅ Detect and measure golf club angle  
✅ Export structured CSV data  
✅ 5-10x dataset expansion via augmentation  
✅ Style normalization for consistent preprocessing  
✅ Full visualization at each step  
✅ Video processing support  
✅ Batch processing with augmentation  

### Usage Recommendations

- Use sample_rate=5 for videos to balance speed and detail
- Apply 3-5 augmentations per image for optimal dataset expansion
- Use reference image from your actual training set for best style matching
- CLAHE preprocessing improves detection in varied lighting conditions
- Export CSVs can be directly used for time-series analysis or ML training

