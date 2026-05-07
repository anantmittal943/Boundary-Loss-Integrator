"""Central configuration for the Boundary Loss Integrator project."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ClassSpec:
    """Specification for one semantic class."""

    train_id: int
    raw_id: int
    name: str
    rgb: Tuple[int, int, int]


PROJECT_NAME: str = "Project 28 - Boundary Loss Integrator"
ARCHITECTURE_NAME: str = "U-Net (encoder-decoder) + CE + Active Contour Boundary Loss"
DATASET_STRATEGY: str = "Full dataset; 10 classes"
PRIMARY_CONSTRAINTS: str = "High edge-IoU on Trees (100) and Bushes (500)"

PROJECT_ROOT: Path = Path(__file__).parent.resolve()
DATA_DIR: Path = PROJECT_ROOT / "data"
DATASET_DIR: Path = DATA_DIR / "dataset"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
CHECKPOINTS_DIR: Path = RESULTS_DIR / "checkpoints"
LOGS_DIR: Path = RESULTS_DIR / "logs"
VIS_DIR: Path = RESULTS_DIR / "visuals"

for folder in (DATA_DIR, DATASET_DIR, RESULTS_DIR, CHECKPOINTS_DIR, LOGS_DIR, VIS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

CLASSES: Tuple[ClassSpec, ...] = (
    ClassSpec(train_id=0, raw_id=0, name="Background", rgb=(0, 0, 0)),
    ClassSpec(train_id=1, raw_id=1, name="Road", rgb=(128, 64, 128)),
    ClassSpec(train_id=2, raw_id=2, name="Building", rgb=(70, 70, 70)),
    ClassSpec(train_id=3, raw_id=3, name="Vegetation", rgb=(0, 102, 0)),
    ClassSpec(train_id=4, raw_id=4, name="Sky", rgb=(128, 128, 255)),
    ClassSpec(train_id=5, raw_id=5, name="Water", rgb=(0, 0, 255)),
    ClassSpec(train_id=6, raw_id=6, name="Person", rgb=(255, 0, 0)),
    ClassSpec(train_id=7, raw_id=7, name="Vehicle", rgb=(255, 255, 0)),
    ClassSpec(train_id=8, raw_id=100, name="Trees", rgb=(0, 200, 0)),
    ClassSpec(train_id=9, raw_id=500, name="Bushes", rgb=(34, 139, 34)),
)

NUM_CLASSES: int = len(CLASSES)
IGNORE_INDEX: int = 255
MAX_RAW_LABEL_ID: int = max(class_spec.raw_id for class_spec in CLASSES)
RAW_TO_TRAIN_ID: Dict[int, int] = {class_spec.raw_id: class_spec.train_id for class_spec in CLASSES}
TRAIN_ID_TO_NAME: Dict[int, str] = {class_spec.train_id: class_spec.name for class_spec in CLASSES}
TRAIN_ID_TO_COLOR: Dict[int, Tuple[int, int, int]] = {
    class_spec.train_id: class_spec.rgb for class_spec in CLASSES
}

TREES_TRAIN_ID: int = 8
BUSHES_TRAIN_ID: int = 9
EDGE_FOCUS_CLASSES: Tuple[int, int] = (TREES_TRAIN_ID, BUSHES_TRAIN_ID)

IMAGE_HEIGHT: int = 512
IMAGE_WIDTH: int = 512
TRAIN_SPLIT: float = 0.8

AUGMENTATION_CONFIG: Dict[str, float] = {
    "horizontal_flip_p": 0.5,
    "vertical_flip_p": 0.2,
    "rotate_limit": 15.0,
    "shift_limit": 0.08,
    "scale_limit": 0.15,
    "brightness_limit": 0.18,
    "contrast_limit": 0.18,
    "elastic_transform_p": 0.2,
}

ENCODER_CHANNELS: Tuple[int, ...] = (64, 128, 256, 512)
BOTTLENECK_CHANNELS: int = 1024
DROPOUT_RATE: float = 0.25
FREEZE_ENCODER_EPOCHS: int = 0

CE_LOSS_WEIGHT: float = 1.0
BOUNDARY_LOSS_WEIGHT: float = 1.2
BOUNDARY_KERNEL_SIZE: int = 3

CLASS_WEIGHTS: Dict[int, float] = {
    0: 1.0,
    1: 1.1,
    2: 1.1,
    3: 1.2,
    4: 1.0,
    5: 1.2,
    6: 1.3,
    7: 1.3,
    8: 1.8,
    9: 1.8,
}

BATCH_SIZE: int = 8
NUM_EPOCHS: int = 60
LEARNING_RATE: float = 3e-4
WEIGHT_DECAY: float = 1e-4
GRADIENT_CLIP_NORM: float = 1.0
EARLY_STOPPING_PATIENCE: int = 12
LOG_INTERVAL: int = 20

LR_T_MAX: int = NUM_EPOCHS
LR_ETA_MIN: float = 1e-6

USE_AMP: bool = True
DEVICE: str = "cuda"
NUM_WORKERS: int = 4
PIN_MEMORY: bool = True
PERSISTENT_WORKERS: bool = True

SEED: int = 42
CUDNN_DETERMINISTIC: bool = False
CUDNN_BENCHMARK: bool = True

LOG_LEVEL: str = "INFO"
BEST_CHECKPOINT_NAME: str = "best_model.pt"
LAST_CHECKPOINT_NAME: str = "last_model.pt"

HIGH_CONTRAST_EDGE_WIDTH: int = 2


@dataclass
class BoundaryLossConfig:
    """Mutable runtime configuration container for CLI overrides."""

    project_name: str = PROJECT_NAME
    architecture_name: str = ARCHITECTURE_NAME
    dataset_strategy: str = DATASET_STRATEGY
    primary_constraints: str = PRIMARY_CONSTRAINTS

    project_root: Path = PROJECT_ROOT
    dataset_dir: Path = DATASET_DIR
    checkpoints_dir: Path = CHECKPOINTS_DIR
    logs_dir: Path = LOGS_DIR
    vis_dir: Path = VIS_DIR

    num_classes: int = NUM_CLASSES
    ignore_index: int = IGNORE_INDEX
    max_raw_label_id: int = MAX_RAW_LABEL_ID
    raw_to_train_id: Dict[int, int] = field(default_factory=lambda: dict(RAW_TO_TRAIN_ID))
    train_id_to_name: Dict[int, str] = field(default_factory=lambda: dict(TRAIN_ID_TO_NAME))
    train_id_to_color: Dict[int, Tuple[int, int, int]] = field(default_factory=lambda: dict(TRAIN_ID_TO_COLOR))
    edge_focus_classes: Tuple[int, int] = EDGE_FOCUS_CLASSES

    image_height: int = IMAGE_HEIGHT
    image_width: int = IMAGE_WIDTH
    train_split: float = TRAIN_SPLIT
    augmentation_config: Dict[str, float] = field(default_factory=lambda: dict(AUGMENTATION_CONFIG))

    encoder_channels: Tuple[int, ...] = ENCODER_CHANNELS
    bottleneck_channels: int = BOTTLENECK_CHANNELS
    dropout_rate: float = DROPOUT_RATE
    freeze_encoder_epochs: int = FREEZE_ENCODER_EPOCHS

    ce_loss_weight: float = CE_LOSS_WEIGHT
    boundary_loss_weight: float = BOUNDARY_LOSS_WEIGHT
    boundary_kernel_size: int = BOUNDARY_KERNEL_SIZE
    class_weights: Dict[int, float] = field(default_factory=lambda: dict(CLASS_WEIGHTS))

    batch_size: int = BATCH_SIZE
    num_epochs: int = NUM_EPOCHS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    gradient_clip_norm: float = GRADIENT_CLIP_NORM
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    log_interval: int = LOG_INTERVAL
    lr_t_max: int = LR_T_MAX
    lr_eta_min: float = LR_ETA_MIN

    use_amp: bool = USE_AMP
    device: str = DEVICE
    num_workers: int = NUM_WORKERS
    pin_memory: bool = PIN_MEMORY
    persistent_workers: bool = PERSISTENT_WORKERS

    seed: int = SEED
    cudnn_deterministic: bool = CUDNN_DETERMINISTIC
    cudnn_benchmark: bool = CUDNN_BENCHMARK
    log_level: str = LOG_LEVEL
    best_checkpoint_name: str = BEST_CHECKPOINT_NAME
    last_checkpoint_name: str = LAST_CHECKPOINT_NAME
    high_contrast_edge_width: int = HIGH_CONTRAST_EDGE_WIDTH


def get_config() -> BoundaryLossConfig:
    """Return the default mutable configuration."""

    return BoundaryLossConfig()
