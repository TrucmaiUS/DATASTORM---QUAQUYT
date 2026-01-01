from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import albumentations as A
import cv2
import numpy as np
from PIL import Image, ImageEnhance


@dataclass
class AugmentationPreview:
    """Container for visual inspection of deterministic augmentations."""

    name: str
    image: np.ndarray
    output_path: Path | None = None


class DataAugmentor:
    """Reusable augmentation toolbox for golfer frames."""

    def __init__(self) -> None:
        self.transform = A.ReplayCompose(
            [
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomScale(scale_limit=0.2, p=0.5),
            ]
        )

    @staticmethod
    def _pil_enhance(image: np.ndarray, enhancer_cls, factor: float) -> np.ndarray:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = enhancer_cls(pil_image)
        enhanced = enhancer.enhance(factor)
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)

    def augment_brightness(self, image: np.ndarray, factor: float = 1.5) -> np.ndarray:
        return self._pil_enhance(image, ImageEnhance.Brightness, factor)

    def augment_contrast(self, image: np.ndarray, factor: float = 1.5) -> np.ndarray:
        return self._pil_enhance(image, ImageEnhance.Contrast, factor)

    @staticmethod
    def augment_zoom(image: np.ndarray, zoom_factor: float = 1.2) -> np.ndarray:
        h, w = image.shape[:2]
        new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        cropped = image[top : top + new_h, left : left + new_w]
        return cv2.resize(cropped, (w, h))

    @staticmethod
    def augment_rotation(image: np.ndarray, angle: float = 10) -> np.ndarray:
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))

    @staticmethod
    def augment_flip(image: np.ndarray, flip_code: int = 1) -> np.ndarray:
        return cv2.flip(image, flip_code)

    @staticmethod
    def augment_noise(image: np.ndarray, noise_level: float = 25) -> np.ndarray:
        noise = np.random.normal(0, noise_level, image.shape).astype(np.int16)
        noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy

    def run_deterministic_suite(self, image: np.ndarray, output_dir: Path | str | None = None) -> Dict[str, AugmentationPreview]:
        """Produce a fixed set of interpretable augmentations for quick reviews."""
        output_path = Path(output_dir) if output_dir else None
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
        variants = {
            "original": image,
            "brighten": self.augment_brightness(image, 1.5),
            "darken": self.augment_brightness(image, 0.6),
            "high_contrast": self.augment_contrast(image, 1.8),
            "low_contrast": self.augment_contrast(image, 0.6),
            "zoom_in": self.augment_zoom(image, 1.3),
            "zoom_out": self.augment_zoom(image, 0.8),
            "rotate_left": self.augment_rotation(image, -15),
            "rotate_right": self.augment_rotation(image, 15),
            "flip_horizontal": self.augment_flip(image, 1),
            "noise": self.augment_noise(image, 30),
        }
        previews: Dict[str, AugmentationPreview] = {}
        for name, variant in variants.items():
            path = None
            if output_path:
                path = output_path / f"{name}.jpg"
                cv2.imwrite(str(path), variant)
            previews[name] = AugmentationPreview(name=name, image=variant, output_path=path)
        return previews

    def augment_batch(self, image_list: Iterable[np.ndarray], augmentations_per_image: int = 5) -> List[np.ndarray]:
        augmented_batch: List[np.ndarray] = []
        for image in image_list:
            augmented_batch.append(image)
            for _ in range(max(augmentations_per_image, 0)):
                augmented = self.transform(image=image)["image"]
                augmented_batch.append(augmented)
        return augmented_batch

    def augment_sequence_consistently(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if not frames:
            return []
        target_h, target_w = frames[0].shape[:2]
        replay = self.transform(image=frames[0])
        augmented_frames = [self._resize_if_needed(replay["image"], target_w, target_h)]
        for frame in frames[1:]:
            result = A.ReplayCompose.replay(replay["replay"], image=frame)
            augmented_frames.append(self._resize_if_needed(result["image"], target_w, target_h))
        return augmented_frames

    @staticmethod
    def _resize_if_needed(image: np.ndarray, width: int, height: int) -> np.ndarray:
        if image.shape[:2] != (height, width):
            return cv2.resize(image, (width, height))
        return image
