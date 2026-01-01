from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


PreviewPair = Tuple[np.ndarray, np.ndarray]


class StylePreprocessor:
    """Color-style normalization utilities for golfer imagery."""

    def __init__(self, reference_image: Optional[np.ndarray] = None) -> None:
        self.reference_image = reference_image
        self.reference_stats: Optional[Dict[str, float]] = None
        if reference_image is not None:
            self.compute_reference_stats(reference_image)

    @staticmethod
    def _calculate_stats(lab_image: np.ndarray) -> Dict[str, float]:
        return {
            "mean_l": float(np.mean(lab_image[:, :, 0])),
            "std_l": float(np.std(lab_image[:, :, 0])),
            "mean_a": float(np.mean(lab_image[:, :, 1])),
            "std_a": float(np.std(lab_image[:, :, 1])),
            "mean_b": float(np.mean(lab_image[:, :, 2])),
            "std_b": float(np.std(lab_image[:, :, 2])),
        }

    def compute_reference_stats(self, image: np.ndarray) -> Dict[str, float]:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        self.reference_stats = self._calculate_stats(lab)
        return self.reference_stats

    def match_color_style(self, image: np.ndarray) -> np.ndarray:
        if self.reference_stats is None:
            return self.normalize_basic(image)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        input_stats = self._calculate_stats(lab)

        def _match_channel(channel: np.ndarray, key_mean: str, key_std: str) -> np.ndarray:
            input_mean = input_stats[key_mean]
            input_std = input_stats[key_std]
            ref_mean = self.reference_stats[key_mean]
            ref_std = self.reference_stats[key_std]
            safe_std = input_std if input_std > 1e-6 else 1.0
            return (channel - input_mean) * (ref_std / safe_std) + ref_mean

        lab[:, :, 0] = _match_channel(lab[:, :, 0], "mean_l", "std_l")
        lab[:, :, 1] = _match_channel(lab[:, :, 1], "mean_a", "std_a")
        lab[:, :, 2] = _match_channel(lab[:, :, 2], "mean_b", "std_b")
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def normalize_basic(image: np.ndarray, target_mean: float = 128, target_std: float = 50) -> np.ndarray:
        img_float = image.astype(np.float32)
        for idx in range(3):
            channel = img_float[:, :, idx]
            mean = float(np.mean(channel))
            std = float(np.std(channel))
            if std > 1e-6:
                img_float[:, :, idx] = ((channel - mean) / std) * target_std + target_mean
        return np.clip(img_float, 0, 255).astype(np.uint8)

    def apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0, tile_size: tuple[int, int] = (8, 8)) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def boost_contrast(image: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    @staticmethod
    def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        gamma = max(gamma, 1e-6)
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    @staticmethod
    def sharpen_image(image: np.ndarray, strength: float = 1.0, method: str = "unsharp") -> np.ndarray:
        """Apply sharpening to enhance edges and details.
        
        Args:
            image: Input BGR image
            strength: Sharpening strength (0.0 to 3.0, default 1.0)
            method: Sharpening method - "unsharp", "laplacian", or "kernel"
        
        Returns:
            Sharpened image
        """
        if method == "unsharp":
            # Unsharp masking: sharp = original + (original - blurred) * strength
            blurred = cv2.GaussianBlur(image, (0, 0), 3)
            sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
            return np.clip(sharpened, 0, 255).astype(np.uint8)
        
        elif method == "laplacian":
            # Laplacian edge enhancement
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = cv2.convertScaleAbs(laplacian)
            laplacian_3ch = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
            sharpened = cv2.addWeighted(image, 1.0, laplacian_3ch, strength * 0.3, 0)
            return np.clip(sharpened, 0, 255).astype(np.uint8)
        
        elif method == "kernel":
            # Traditional sharpening kernel
            kernel_strength = max(0.1, min(strength, 3.0))
            kernel = np.array([[-1, -1, -1],
                             [-1, 9 + kernel_strength, -1],
                             [-1, -1, -1]]) / (1 + kernel_strength)
            sharpened = cv2.filter2D(image, -1, kernel)
            return np.clip(sharpened, 0, 255).astype(np.uint8)
        
        else:
            raise ValueError(f"Unknown sharpening method: {method}")

    @staticmethod
    def enhance_edges(image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Enhance edges while preserving overall image quality.
        
        Args:
            image: Input BGR image
            alpha: Edge enhancement strength (0.0 to 1.0)
        
        Returns:
            Edge-enhanced image
        """
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges using Sobel
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = cv2.convertScaleAbs(sobel)
        
        # Convert edge map to 3 channels
        edges_3ch = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
        
        # Blend with original
        enhanced = cv2.addWeighted(image, 1.0, edges_3ch, alpha, 0)
        return np.clip(enhanced, 0, 255).astype(np.uint8)

    @staticmethod
    def analyze_image_properties(image: np.ndarray) -> Dict[str, float]:
        """Analyze image properties to determine optimal preprocessing.
        
        Args:
            image: Input BGR image
        
        Returns:
            Dictionary with image statistics:
            - brightness: Average brightness (0-255)
            - contrast: Contrast measure (0-255)
            - noise_level: Estimated noise level (0-100)
            - sharpness: Sharpness measure (0-100)
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Brightness: mean pixel value
        brightness = float(np.mean(gray))
        
        # Contrast: standard deviation
        contrast = float(np.std(gray))
        
        # Noise estimation using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_level = min(100.0, float(np.var(laplacian)) / 10.0)
        
        # Sharpness using gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        sharpness = min(100.0, float(np.mean(gradient_mag)))
        
        return {
            "brightness": brightness,
            "contrast": contrast,
            "noise_level": noise_level,
            "sharpness": sharpness,
        }

    def adaptive_preprocess(self, image: np.ndarray, target_quality: str = "high") -> np.ndarray:
        """Automatically adjust preprocessing based on image properties.
        
        Args:
            image: Input BGR image
            target_quality: "high", "medium", or "fast" - quality vs speed tradeoff
        
        Returns:
            Adaptively preprocessed image
        """
        props = self.analyze_image_properties(image)
        
        # Start with the image
        processed = image.copy()
        
        # Apply denoising if noisy
        if props["noise_level"] > 30:
            if target_quality == "high":
                processed = self.denoise(processed, method="fastNlMeans", h=10)
            elif target_quality == "medium":
                processed = self.denoise(processed, method="bilateral")
            else:  # fast
                processed = self.denoise(processed, method="gaussian")
        
        # Adjust brightness if too dark or too bright
        if props["brightness"] < 100:  # Dark image
            gamma = 1.3  # Brighten
            processed = self.adjust_gamma(processed, gamma=gamma)
        elif props["brightness"] > 170:  # Bright image
            gamma = 0.8  # Darken slightly
            processed = self.adjust_gamma(processed, gamma=gamma)
        
        # Enhance contrast if low
        if props["contrast"] < 40:
            processed = self.apply_clahe(processed, clip_limit=3.0)
        else:
            processed = self.apply_clahe(processed, clip_limit=2.0)
        
        # Apply color normalization
        processed = self.match_color_style(processed)
        
        # Sharpen if image is blurry
        if props["sharpness"] < 30:
            processed = self.sharpen_image(processed, strength=1.5, method="unsharp")
        elif props["sharpness"] < 50:
            processed = self.sharpen_image(processed, strength=1.0, method="unsharp")
        
        # Enhance edges for pose detection
        if target_quality in ["high", "medium"]:
            processed = self.enhance_edges(processed, alpha=0.3)
        
        # Final contrast boost
        processed = self.boost_contrast(processed, alpha=1.2, beta=10.0)
        
        return processed

    @staticmethod
    def denoise(image: np.ndarray, method: str = "fastNlMeans", h: int = 10, template_size: int = 7, search_size: int = 21) -> np.ndarray:
        """Apply denoising to reduce image noise.
        
        Args:
            image: Input BGR image
            method: Denoising method - "fastNlMeans", "bilateral", or "gaussian"
            h: Filter strength for fastNlMeans (higher = more denoising)
            template_size: Size of template patch for fastNlMeans (should be odd)
            search_size: Size of search window for fastNlMeans (should be odd)
        
        Returns:
            Denoised image
        """
        if method == "fastNlMeans":
            return cv2.fastNlMeansDenoisingColored(image, None, h, h, template_size, search_size)
        elif method == "bilateral":
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == "gaussian":
            return cv2.GaussianBlur(image, (5, 5), 0)
        else:
            raise ValueError(f"Unknown denoising method: {method}")

    @staticmethod
    def detect_club_region(image: np.ndarray, detector=None) -> Optional[Tuple[int, int, int, int]]:
        """Detect the region where the golf club is likely located.
        
        Args:
            image: Input BGR image
            detector: Optional YOLO detector for club detection
        
        Returns:
            Bounding box (x1, y1, x2, y2) or None if not detected
        """
        # Use detector (e.g., YOLO for club detection)
        # COCO classes for sports: 32=sports ball, 36=baseball bat, 37=baseball glove,
        # 38=skateboard, 39=surfboard, 40=tennis racket
        # For golf clubs, tennis racket (40) might work due to similar elongated shape
        if detector is not None:
            results = detector(image, classes=[36], conf=0.25)  # class 36 is baseball bat (similar to golf club)
            if results and len(results) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    return (int(x1), int(y1), int(x2), int(y2))
        return None
    
    @staticmethod
    def detect_club_from_hands(image: np.ndarray, hand_keypoints: Dict[str, Dict[str, float]], 
                                region_expand: int = 100) -> Optional[Tuple[float, float, float, float]]:
        """Detect golf club using hand joint regions, edge detection, and line approximation.
        
        This method:
        1. Creates a region around hand keypoints (wrists)
        2. Applies edge detection (Canny)
        3. Uses Hough Line Transform to find straight lines
        4. Filters and returns the most likely club line
        
        Args:
            image: Input BGR image
            hand_keypoints: Dictionary with hand keypoints (must include 'left_wrist' and/or 'right_wrist')
                          Each keypoint is a dict with keys: 'x', 'y', 'visibility'
            region_expand: Pixels to expand around hand region for club detection
        
        Returns:
            Tuple of (x1, y1, x2, y2) representing the club line endpoints in normalized coords (0-1),
            or None if no club detected
        """
        h, w = image.shape[:2]
        
        # Extract hand positions
        hands = []
        for hand_name in ['left_wrist', 'right_wrist']:
            if hand_name in hand_keypoints and hand_keypoints[hand_name]['visibility'] > 0.5:
                x = int(hand_keypoints[hand_name]['x'] * w)
                y = int(hand_keypoints[hand_name]['y'] * h)
                hands.append((x, y))
        
        if not hands:
            return None
        
        # Calculate region of interest around hands
        xs = [h[0] for h in hands]
        ys = [h[1] for h in hands]
        
        x_min = max(0, min(xs) - region_expand)
        x_max = min(w, max(xs) + region_expand)
        y_min = max(0, min(ys) - region_expand)
        y_max = min(h, max(ys) + region_expand)
        
        if x_max <= x_min or y_max <= y_min:
            return None
        
        # Extract ROI
        roi = image[y_min:y_max, x_min:x_max]
        
        # Convert to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        
        # Edge detection using Canny
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Hough Line Transform to detect lines
        # Adjusted for golf club: two parallel lines with gaps
        lines = cv2.HoughLinesP(
            edges,
            rho=1,              # Distance resolution in pixels
            theta=np.pi/180,    # Angle resolution in radians
            threshold=30,       # Reduced threshold to detect more segments
            minLineLength=20,   # Shorter segments allowed
            maxLineGap=30       # Larger gaps allowed (club can be discontinuous)
        )
        
        if lines is None or len(lines) == 0:
            return None
        
        # Find lines that originate from wrist positions
        # Club extends away from wrist for ~50px
        club_candidates = []
        
        for line in lines:
            x1_roi, y1_roi, x2_roi, y2_roi = line[0]
            
            # Calculate line properties
            length = np.sqrt((x2_roi - x1_roi)**2 + (y2_roi - y1_roi)**2)
            
            # Calculate angle
            angle = np.arctan2(y2_roi - y1_roi, x2_roi - x1_roi)
            angle_deg = np.degrees(angle)
            
            # Convert ROI coords to absolute frame coords
            x1_abs = x_min + x1_roi
            y1_abs = y_min + y1_roi
            x2_abs = x_min + x2_roi
            y2_abs = y_min + y2_roi
            
            # Check if line is near any wrist position (within 50px)
            min_dist_to_wrist = float('inf')
            for wx, wy in hands:
                # Distance from line endpoints to wrist
                dist1 = np.sqrt((x1_abs - wx)**2 + (y1_abs - wy)**2)
                dist2 = np.sqrt((x2_abs - wx)**2 + (y2_abs - wy)**2)
                min_dist_to_wrist = min(min_dist_to_wrist, dist1, dist2)
            
            # Filter criteria:
            # - Line should be reasonably long (at least 15 pixels)
            # - At least one endpoint should be within 70px of a wrist
            # - Not too horizontal (prefer more vertical/diagonal lines)
            if length >= 15 and min_dist_to_wrist <= 70:
                # Score based on length and proximity to wrist
                score = length * (1.0 / (1.0 + min_dist_to_wrist / 50.0))
                club_candidates.append({
                    'x1': x1_roi, 'y1': y1_roi, 'x2': x2_roi, 'y2': y2_roi,
                    'length': length, 'angle': angle, 'angle_deg': angle_deg,
                    'score': score, 'dist_to_wrist': min_dist_to_wrist
                })
        
        if not club_candidates:
            return None
        
        # Sort by score and get best candidates
        club_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Try to find two parallel lines (club edges)
        # Take top candidate and look for a parallel partner
        best_line = club_candidates[0]
        parallel_line = None
        
        for candidate in club_candidates[1:]:
            # Check if angles are similar (parallel)
            angle_diff = abs(best_line['angle_deg'] - candidate['angle_deg'])
            if angle_diff < 15 or angle_diff > 165:  # Parallel or anti-parallel
                # Check if they're close to each other (club width ~5-15px)
                # Calculate perpendicular distance between lines
                dx = best_line['x2'] - best_line['x1']
                dy = best_line['y2'] - best_line['y1']
                line_len = np.sqrt(dx**2 + dy**2)
                if line_len > 0:
                    # Distance from candidate point to best line
                    dist = abs(dy * candidate['x1'] - dx * candidate['y1'] + 
                              best_line['x2'] * best_line['y1'] - best_line['y2'] * best_line['x1']) / line_len
                    if 3 <= dist <= 25:  # Club width range
                        parallel_line = candidate
                        break
        
        # If we found parallel lines, use both to define club
        if parallel_line is not None:
            # Average the two lines to get club center
            x1_roi = int((best_line['x1'] + parallel_line['x1']) / 2)
            y1_roi = int((best_line['y1'] + parallel_line['y1']) / 2)
            x2_roi = int((best_line['x2'] + parallel_line['x2']) / 2)
            y2_roi = int((best_line['y2'] + parallel_line['y2']) / 2)
            best_line = {'x1': x1_roi, 'y1': y1_roi, 'x2': x2_roi, 'y2': y2_roi}
        
        # Convert back to absolute coordinates and normalize
        x1_abs = (x_min + best_line['x1']) / w
        y1_abs = (y_min + best_line['y1']) / h
        x2_abs = (x_min + best_line['x2']) / w
        y2_abs = (y_min + best_line['y2']) / h
        
        return (x1_abs, y1_abs, x2_abs, y2_abs)

    @staticmethod
    def detect_club_edges(image: np.ndarray, club_bbox: Optional[Tuple[int, int, int, int]] = None,
                         canny_low: int = 50, canny_high: int = 150) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """Detect edges in the club region for enhanced club visibility.
        
        Args:
            image: Input BGR image
            club_bbox: Optional pre-detected bounding box (x1, y1, x2, y2)
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
        
        Returns:
            Tuple of (edge_image, club_bbox)
            - edge_image: Full-size image with club edges highlighted
            - club_bbox: Detected or provided bounding box
        """
        # Detect club region if not provided
        if club_bbox is None:
            club_bbox = StylePreprocessor.detect_club_region(image)
        
        if club_bbox is None:
            # Return empty edge image if no club detected
            return np.zeros_like(image), None
        
        x1, y1, x2, y2 = club_bbox
        
        # Ensure valid coordinates
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros_like(image), None
        
        # Extract club region
        club_region = image[y1:y2, x1:x2]
        
        # Convert to grayscale
        gray_region = cv2.cvtColor(club_region, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred_region = cv2.GaussianBlur(gray_region, (5, 5), 0)
        
        # Detect edges using Canny
        edges_region = cv2.Canny(blurred_region, canny_low, canny_high)
        
        # Create output image with edges highlighted
        edge_image = np.zeros_like(image)
        edge_region_3ch = cv2.cvtColor(edges_region, cv2.COLOR_GRAY2BGR)
        edge_image[y1:y2, x1:x2] = edge_region_3ch
        
        return edge_image, club_bbox

    def preprocess_for_training(
        self,
        image: np.ndarray,
        apply_denoise: bool = True,
        denoise_method: str = "fastNlMeans",
        denoise_h: int = 10,
        apply_clahe: bool = True,
        contrast_alpha: float = 1.25,
        contrast_beta: float = 15.0,
        gamma: float = 0.9,
        apply_sharpening: bool = True,
        sharpen_strength: float = 1.0,
        sharpen_method: str = "unsharp",
        apply_edge_enhancement: bool = True,
        edge_alpha: float = 0.3,
        use_adaptive: bool = False,
        adaptive_quality: str = "high",
    ) -> np.ndarray:
        """Preprocess image for pose detection with comprehensive enhancements.
        
        Args:
            image: Input BGR image
            apply_denoise: Whether to apply denoising
            denoise_method: Denoising method ("fastNlMeans", "bilateral", "gaussian")
            denoise_h: Denoising strength
            apply_clahe: Whether to apply CLAHE
            contrast_alpha: Contrast multiplication factor
            contrast_beta: Contrast addition factor
            gamma: Gamma correction value
            apply_sharpening: Whether to apply sharpening
            sharpen_strength: Sharpening strength (0-3)
            sharpen_method: Sharpening method ("unsharp", "laplacian", "kernel")
            apply_edge_enhancement: Whether to enhance edges
            edge_alpha: Edge enhancement strength (0-1)
            use_adaptive: Use adaptive preprocessing based on image analysis
            adaptive_quality: Quality level for adaptive preprocessing ("high", "medium", "fast")
        
        Returns:
            Preprocessed image
        """
        # Use adaptive preprocessing if requested
        if use_adaptive:
            return self.adaptive_preprocess(image, target_quality=adaptive_quality)
        
        # Manual preprocessing pipeline
        # Apply denoising first to reduce noise
        if apply_denoise:
            processed = self.denoise(image, method=denoise_method, h=denoise_h)
        else:
            processed = image
        
        # Color normalization
        processed = self.match_color_style(processed)
        
        # CLAHE for local contrast enhancement
        if apply_clahe:
            processed = self.apply_clahe(processed)
        
        # Global contrast adjustment
        if not np.isclose(contrast_alpha, 1.0) or not np.isclose(contrast_beta, 0.0):
            processed = self.boost_contrast(processed, alpha=contrast_alpha, beta=contrast_beta)
        
        # Gamma correction
        if not np.isclose(gamma, 1.0):
            processed = self.adjust_gamma(processed, gamma=gamma)
        
        # Sharpening for better edge definition
        if apply_sharpening:
            processed = self.sharpen_image(processed, strength=sharpen_strength, method=sharpen_method)
        
        # Edge enhancement for better pose detection
        if apply_edge_enhancement:
            processed = self.enhance_edges(processed, alpha=edge_alpha)
        
        return processed

    def visualize_preprocessing(self, image: np.ndarray, include_club_detection: bool = False, 
                               show_analysis: bool = True) -> Dict[str, np.ndarray]:
        """Visualize all preprocessing steps including new enhancements."""
        denoised = self.denoise(image)
        color_matched = self.match_color_style(denoised)
        clahe_img = self.apply_clahe(color_matched)
        contrast_img = self.boost_contrast(clahe_img, alpha=1.25, beta=15.0)
        gamma_img = self.adjust_gamma(contrast_img, gamma=0.9)
        sharpened = self.sharpen_image(gamma_img, strength=1.0, method="unsharp")
        edge_enhanced = self.enhance_edges(sharpened, alpha=0.3)
        
        result = {
            "Original": image,
            "Denoised": denoised,
            "Color Matched": color_matched,
            "CLAHE Applied": clahe_img,
            "Contrast Boosted": contrast_img,
            "Gamma Adjusted": gamma_img,
            "Sharpened": sharpened,
            "Edge Enhanced": edge_enhanced,
            "Full Pipeline": self.preprocess_for_training(image),
            "Adaptive Pipeline": self.adaptive_preprocess(image, target_quality="high"),
        }
        
        # Show image analysis
        if show_analysis:
            props = self.analyze_image_properties(image)
            # Create text overlay with properties
            analysis_img = image.copy()
            y_offset = 30
            for key, value in props.items():
                text = f"{key}: {value:.1f}"
                cv2.putText(analysis_img, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
            result["Image Analysis"] = analysis_img
        
        if include_club_detection:
            club_bbox = self.detect_club_region(image)
            if club_bbox is not None:
                # Draw club region on original
                img_with_bbox = image.copy()
                x1, y1, x2, y2 = club_bbox
                cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)
                result["Club Detection"] = img_with_bbox
                
                # Add club edges
                edge_image, _ = self.detect_club_edges(image, club_bbox)
                result["Club Edges"] = edge_image
        
        return result


def collect_image_files(root: Path, extensions: Iterable[str] = ("*.jpg", "*.jpeg", "*.png")) -> List[Path]:
    files: List[Path] = []
    for pattern in extensions:
        files.extend(root.rglob(pattern))
    return sorted(files)


def load_reference_image(reference_path: Optional[Path], candidates: List[Path]) -> np.ndarray:
    if reference_path is not None:
        ref_path = reference_path
    elif candidates:
        ref_path = candidates[0]
    else:
        raise ValueError("No images available to derive a reference style.")
    image = cv2.imread(str(ref_path))
    if image is None:
        raise FileNotFoundError(f"Failed to load reference image at {ref_path}.")
    return image


def preprocess_dataset(
    input_root: Path,
    output_root: Path,
    reference_image_path: Optional[Path] = None,
    apply_denoise: bool = True,
    denoise_method: str = "fastNlMeans",
    denoise_h: int = 10,
    apply_clahe: bool = True,
    overwrite: bool = False,
    contrast_alpha: float = 1.25,
    contrast_beta: float = 15.0,
    gamma: float = 0.9,
) -> Dict[str, float]:
    image_files = collect_image_files(input_root)
    if not image_files:
        raise ValueError(f"No images found under {input_root}.")
    reference_image = load_reference_image(reference_image_path, image_files)
    preprocessor = StylePreprocessor(reference_image)
    for image_path in tqdm(image_files, desc="Preprocessing", unit="img"):
        relative = image_path.relative_to(input_root)
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        processed = preprocessor.preprocess_for_training(
            image,
            apply_denoise=apply_denoise,
            denoise_method=denoise_method,
            denoise_h=denoise_h,
            apply_clahe=apply_clahe,
            contrast_alpha=contrast_alpha,
            contrast_beta=contrast_beta,
            gamma=gamma,
        )
        cv2.imwrite(str(destination), processed)
    return preprocessor.reference_stats or {}


def preprocess_video_frames(
    video_path: Path,
    output_root: Path,
    reference_image_path: Optional[Path] = None,
    frame_skip: int = 1,
    apply_denoise: bool = True,
    denoise_method: str = "fastNlMeans",
    denoise_h: int = 10,
    apply_clahe: bool = True,
    overwrite: bool = True,
    preview_frame_count: int = 6,
    contrast_alpha: float = 1.25,
    contrast_beta: float = 15.0,
    gamma: float = 0.9,
    write_video: bool = True,
    video_codec: str = "mp4v",
    video_extension: str = ".mp4",
    detect_club: bool = False,
    save_club_edges: bool = False,
) -> Dict[str, object]:
    if frame_skip < 1:
        raise ValueError("frame_skip must be >= 1.")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video at {video_path}.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_total = total_frames if total_frames > 0 else None
    progress = tqdm(total=progress_total, desc=f"Video {video_path.name}", unit="frame")
    output_dir = output_root / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_image: Optional[np.ndarray] = None
    if reference_image_path is not None:
        reference_image = cv2.imread(str(reference_image_path))
        if reference_image is None:
            raise FileNotFoundError(f"Failed to load reference frame at {reference_image_path}.")
    preprocessor: Optional[StylePreprocessor] = None
    preview_pairs: List[PreviewPair] = []
    saved_frames = 0
    frame_idx = 0
    video_writer: Optional[cv2.VideoWriter] = None
    rendered_video_path: Optional[Path] = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                if preprocessor is None:
                    ref_image = reference_image if reference_image is not None else frame
                    preprocessor = StylePreprocessor(ref_image)
                processed = preprocessor.preprocess_for_training(
                    frame,
                    apply_denoise=apply_denoise,
                    denoise_method=denoise_method,
                    denoise_h=denoise_h,
                    apply_clahe=apply_clahe,
                    contrast_alpha=contrast_alpha,
                    contrast_beta=contrast_beta,
                    gamma=gamma,
                )
                destination = output_dir / f"{video_path.stem}_{saved_frames:05d}.jpg"
                if not destination.exists() or overwrite:
                    cv2.imwrite(str(destination), processed)
                
                # Detect and save club edges if requested
                if detect_club or save_club_edges:
                    club_bbox = preprocessor.detect_club_region(frame)
                    if club_bbox is not None and save_club_edges:
                        edge_image, _ = preprocessor.detect_club_edges(frame, club_bbox)
                        edge_destination = output_dir / f"{video_path.stem}_{saved_frames:05d}_club_edges.jpg"
                        cv2.imwrite(str(edge_destination), edge_image)
                if len(preview_pairs) < preview_frame_count:
                    preview_pairs.append((frame.copy(), processed.copy()))
                if write_video:
                    if video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*video_codec)
                        rendered_video_path = output_dir / f"{video_path.stem}_normalized{video_extension}"
                        frame_size = (processed.shape[1], processed.shape[0])
                        video_writer = cv2.VideoWriter(
                            str(rendered_video_path),
                            fourcc,
                            float(fps),
                            frame_size,
                        )
                    video_writer.write(processed)
                saved_frames += 1
            frame_idx += 1
            progress.update(1)
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        progress.close()
    if preprocessor is None:
        raise RuntimeError("No frames were processed. Check frame_skip or video integrity.")
    return {
        "stats": preprocessor.reference_stats or {},
        "frames_saved": saved_frames,
        "preview_pairs": preview_pairs,
        "preprocessor": preprocessor,
        "output_dir": output_dir,
        "video_path": rendered_video_path,
    }
