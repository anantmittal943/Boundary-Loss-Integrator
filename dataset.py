"""Dataset and dataloader utilities for semantic segmentation."""

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

from config import BoundaryLossConfig, get_config

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilePair:
    """Pair of image and mask file paths."""

    image_path: Path
    mask_path: Path


def create_augmentation_pipeline(
    image_height: int,
    image_width: int,
    augmentation_config: Dict[str, float],
    is_training: bool,
) -> A.Compose:
    """Create Albumentations transforms for training or evaluation."""

    transforms: List[A.BasicTransform] = []

    if is_training:
        transforms.extend(
            [
                A.HorizontalFlip(p=augmentation_config.get("horizontal_flip_p", 0.5)),
                A.VerticalFlip(p=augmentation_config.get("vertical_flip_p", 0.2)),
                A.ShiftScaleRotate(
                    shift_limit=augmentation_config.get("shift_limit", 0.08),
                    scale_limit=augmentation_config.get("scale_limit", 0.15),
                    rotate_limit=augmentation_config.get("rotate_limit", 15.0),
                    border_mode=0,
                    p=0.75,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=augmentation_config.get("brightness_limit", 0.18),
                    contrast_limit=augmentation_config.get("contrast_limit", 0.18),
                    p=0.45,
                ),
                A.ElasticTransform(
                    alpha=40.0,
                    sigma=5.0,
                    alpha_affine=5.0,
                    border_mode=0,
                    p=augmentation_config.get("elastic_transform_p", 0.2),
                ),
            ]
        )

    transforms.extend(
        [
            A.Resize(height=image_height, width=image_width),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms)


class SemanticSegmentationDataset(Dataset):
    """Robust dataset with safe loading and raw-to-train label remapping."""

    def __init__(
        self,
        file_pairs: Sequence[FilePair],
        config: Optional[BoundaryLossConfig] = None,
        transform: Optional[A.Compose] = None,
    ) -> None:
        self.config = config or get_config()
        self.transform = transform
        self.file_pairs = list(file_pairs)

        if not self.file_pairs:
            raise ValueError("Dataset has no valid image/mask pairs.")

        self._label_lut = np.full(
            self.config.max_raw_label_id + 1,
            self.config.ignore_index,
            dtype=np.uint8,
        )
        for raw_id, train_id in self.config.raw_to_train_id.items():
            self._label_lut[raw_id] = train_id

    def __len__(self) -> int:
        return len(self.file_pairs)

    def _safe_load_rgb(self, path: Path) -> Optional[np.ndarray]:
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                return np.asarray(rgb, dtype=np.uint8)
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            LOGGER.error("Failed to load image %s: %s", path, exc)
            return None

    def _safe_load_mask(self, path: Path) -> Optional[np.ndarray]:
        try:
            with Image.open(path) as mask_image:
                mask = np.asarray(mask_image)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            return mask.astype(np.int32)
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            LOGGER.error("Failed to load mask %s: %s", path, exc)
            return None

    def _remap_mask(self, raw_mask: np.ndarray) -> np.ndarray:
        clipped = np.where(
            (raw_mask >= 0) & (raw_mask <= self.config.max_raw_label_id),
            raw_mask,
            0,
        )
        remapped = self._label_lut[clipped]
        unknown = (raw_mask < 0) | (raw_mask > self.config.max_raw_label_id)
        remapped[unknown] = self.config.ignore_index
        return remapped.astype(np.uint8)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        max_retries = min(3, len(self.file_pairs))
        attempt = 0

        while attempt < max_retries:
            pair = self.file_pairs[(index + attempt) % len(self.file_pairs)]
            image = self._safe_load_rgb(pair.image_path)
            raw_mask = self._safe_load_mask(pair.mask_path)
            if image is None or raw_mask is None:
                attempt += 1
                continue

            mask = self._remap_mask(raw_mask)

            if self.transform is not None:
                transformed = self.transform(image=image, mask=mask)
                image_tensor = transformed["image"]
                mask_tensor = transformed["mask"].long()
            else:
                image_tensor = (
                    torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                )
                mask_tensor = torch.from_numpy(mask).long()

            return {
                "image": image_tensor,
                "mask": mask_tensor,
                "path": str(pair.image_path),
            }

        raise RuntimeError("Unable to load a valid sample after retries.")


def discover_file_pairs(
    dataset_dir: Path,
    image_subdir: str = "images",
    mask_subdir: str = "masks",
    image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
    mask_exts: Tuple[str, ...] = (".png", ".bmp", ".tif", ".tiff"),
) -> List[FilePair]:
    """Discover image-mask file pairs with matching stem names."""

    image_dir = dataset_dir / image_subdir
    mask_dir = dataset_dir / mask_subdir
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"Expected dataset folders missing: {image_dir} and/or {mask_dir}"
        )

    mask_by_stem: Dict[str, Path] = {}
    for mask_path in mask_dir.iterdir():
        if mask_path.is_file() and mask_path.suffix.lower() in mask_exts:
            mask_by_stem[mask_path.stem] = mask_path

    pairs: List[FilePair] = []
    for image_path in image_dir.iterdir():
        if not image_path.is_file() or image_path.suffix.lower() not in image_exts:
            continue
        mask_path = mask_by_stem.get(image_path.stem)
        if mask_path is None:
            LOGGER.warning("No mask found for image %s", image_path.name)
            continue
        pairs.append(FilePair(image_path=image_path, mask_path=mask_path))

    if not pairs:
        raise RuntimeError(f"No valid image/mask pairs found in {dataset_dir}")

    pairs.sort(key=lambda pair: pair.image_path.name)
    LOGGER.info("Discovered %d image/mask pairs.", len(pairs))
    return pairs


def create_dataloaders(
    config: Optional[BoundaryLossConfig] = None,
    dataset_dir: Optional[Path] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create deterministic train/validation dataloaders."""

    cfg = config or get_config()
    root = dataset_dir or cfg.dataset_dir
    pairs = discover_file_pairs(root)

    rng = random.Random(cfg.seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)

    train_count = int(len(shuffled) * cfg.train_split)
    train_pairs = shuffled[:train_count]
    val_pairs = shuffled[train_count:]
    if not val_pairs:
        raise RuntimeError("Validation split is empty. Increase dataset size or lower train_split.")

    train_transform = create_augmentation_pipeline(
        image_height=cfg.image_height,
        image_width=cfg.image_width,
        augmentation_config=cfg.augmentation_config,
        is_training=True,
    )
    val_transform = create_augmentation_pipeline(
        image_height=cfg.image_height,
        image_width=cfg.image_width,
        augmentation_config=cfg.augmentation_config,
        is_training=False,
    )

    train_dataset = SemanticSegmentationDataset(
        file_pairs=train_pairs,
        config=cfg,
        transform=train_transform,
    )
    val_dataset = SemanticSegmentationDataset(
        file_pairs=val_pairs,
        config=cfg,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )

    LOGGER.info(
        "Created dataloaders: train=%d samples, val=%d samples",
        len(train_dataset),
        len(val_dataset),
    )
    return train_loader, val_loader


def create_eval_dataloader(
    config: Optional[BoundaryLossConfig] = None,
    dataset_dir: Optional[Path] = None,
    batch_size: int = 1,
) -> DataLoader:
    """Create a deterministic evaluation dataloader over the full dataset."""

    cfg = config or get_config()
    root = dataset_dir or cfg.dataset_dir
    pairs = discover_file_pairs(root)
    transform = create_augmentation_pipeline(
        image_height=cfg.image_height,
        image_width=cfg.image_width,
        augmentation_config=cfg.augmentation_config,
        is_training=False,
    )
    eval_dataset = SemanticSegmentationDataset(file_pairs=pairs, config=cfg, transform=transform)
    return DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )
