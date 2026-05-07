"""Training entrypoint for Boundary Loss Integrator."""

import argparse
import logging
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import BoundaryLossConfig, get_config
from dataset import create_dataloaders
from model import (
    CombinedSegmentationLoss,
    build_model,
    compute_batch_mean_iou,
    compute_edge_iou_for_class,
)


LOGGER = logging.getLogger("boundary_loss_train")


def set_random_seeds(seed: int, deterministic: bool, benchmark: bool) -> None:
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


def setup_logging(log_dir: Path, log_level: str) -> None:
    """Configure console and file logging."""

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    return images, masks


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    criterion: CombinedSegmentationLoss,
    config: BoundaryLossConfig,
    device: torch.device,
    epoch_idx: int,
) -> Dict[str, float]:
    """Run one training epoch."""

    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_boundary = 0.0
    running_iou = 0.0

    for step, batch in enumerate(dataloader, start=1):
        images, masks = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=config.use_amp):
            logits = model(images)
            total_loss, details = criterion(logits, masks)

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            predictions = logits.argmax(dim=1)
            mean_iou = compute_batch_mean_iou(
                predictions=predictions,
                targets=masks,
                num_classes=config.num_classes,
                ignore_index=config.ignore_index,
            )

        running_loss += float(total_loss.detach().item())
        running_ce += details["ce"]
        running_boundary += details["boundary"]
        running_iou += mean_iou

        if step % config.log_interval == 0:
            LOGGER.info(
                "epoch=%d step=%d/%d loss=%.4f ce=%.4f boundary=%.4f mean_iou=%.4f",
                epoch_idx,
                step,
                len(dataloader),
                running_loss / step,
                running_ce / step,
                running_boundary / step,
                running_iou / step,
            )

    batches = max(len(dataloader), 1)
    return {
        "train_loss": running_loss / batches,
        "train_ce_loss": running_ce / batches,
        "train_boundary_loss": running_boundary / batches,
        "train_mean_iou": running_iou / batches,
    }


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: CombinedSegmentationLoss,
    config: BoundaryLossConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Run model validation and compute edge-focused metrics."""

    model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_boundary = 0.0
    total_iou = 0.0
    trees_edge_iou = 0.0
    bushes_edge_iou = 0.0

    for batch in dataloader:
        images, masks = _to_device(batch, device)

        with autocast(enabled=config.use_amp):
            logits = model(images)
            loss, details = criterion(logits, masks)

        predictions = logits.argmax(dim=1)
        total_loss += float(loss.item())
        total_ce += details["ce"]
        total_boundary += details["boundary"]
        total_iou += compute_batch_mean_iou(
            predictions=predictions,
            targets=masks,
            num_classes=config.num_classes,
            ignore_index=config.ignore_index,
        )
        trees_edge_iou += compute_edge_iou_for_class(
            predictions=predictions,
            targets=masks,
            class_id=config.edge_focus_classes[0],
            ignore_index=config.ignore_index,
        )
        bushes_edge_iou += compute_edge_iou_for_class(
            predictions=predictions,
            targets=masks,
            class_id=config.edge_focus_classes[1],
            ignore_index=config.ignore_index,
        )

    batches = max(len(dataloader), 1)
    return {
        "val_loss": total_loss / batches,
        "val_ce_loss": total_ce / batches,
        "val_boundary_loss": total_boundary / batches,
        "val_mean_iou": total_iou / batches,
        "val_edge_iou_trees": trees_edge_iou / batches,
        "val_edge_iou_bushes": bushes_edge_iou / batches,
    }


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_iou: float,
    config: BoundaryLossConfig,
) -> None:
    """Persist training state for resume/inference."""

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_iou": best_iou,
        "config": asdict(config),
    }
    torch.save(state, checkpoint_path)


def parse_args() -> argparse.Namespace:
    """CLI argument parser."""

    parser = argparse.ArgumentParser(description="Boundary Loss Integrator training script")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Dataset root containing images/ and masks/")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader worker count")
    parser.add_argument("--device", type=str, default=None, help="Device to use: cuda or cpu")
    return parser.parse_args()


def main() -> None:
    """Training program entrypoint."""

    args = parse_args()
    config = get_config()

    if args.dataset_dir is not None:
        config.dataset_dir = args.dataset_dir
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.device is not None:
        config.device = args.device

    setup_logging(config.logs_dir, config.log_level)
    set_random_seeds(config.seed, config.cudnn_deterministic, config.cudnn_benchmark)

    requested_cuda = config.device.lower().startswith("cuda")
    if requested_cuda and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but not available. Falling back to CPU.")
    device = torch.device("cuda" if requested_cuda and torch.cuda.is_available() else "cpu")

    LOGGER.info("Project: %s", config.project_name)
    LOGGER.info("Architecture: %s", config.architecture_name)
    LOGGER.info("Dataset: %s", config.dataset_dir)
    LOGGER.info("Device: %s", device)

    train_loader, val_loader = create_dataloaders(config=config, dataset_dir=config.dataset_dir)

    model = build_model(config).to(device)
    if config.freeze_encoder_epochs > 0:
        model.freeze_encoder()

    class_weights = torch.tensor(
        [config.class_weights[class_id] for class_id in range(config.num_classes)],
        dtype=torch.float32,
        device=device,
    )

    criterion = CombinedSegmentationLoss(
        class_weights=class_weights,
        ce_weight=config.ce_loss_weight,
        boundary_weight=config.boundary_loss_weight,
        focus_classes=config.edge_focus_classes,
        ignore_index=config.ignore_index,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.lr_t_max,
        eta_min=config.lr_eta_min,
    )
    scaler = GradScaler(enabled=config.use_amp)

    best_val_iou = -1.0
    epochs_without_improvement = 0

    best_path = config.checkpoints_dir / config.best_checkpoint_name
    last_path = config.checkpoints_dir / config.last_checkpoint_name

    for epoch in range(1, config.num_epochs + 1):
        if config.freeze_encoder_epochs > 0 and epoch == config.freeze_encoder_epochs + 1:
            model.unfreeze_encoder()
            LOGGER.info("Encoder unfrozen at epoch %d", epoch)

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            config=config,
            device=device,
            epoch_idx=epoch,
        )
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            config=config,
            device=device,
        )
        scheduler.step()

        LOGGER.info(
            "epoch=%d train_loss=%.4f train_iou=%.4f val_loss=%.4f val_iou=%.4f edge_iou_trees=%.4f edge_iou_bushes=%.4f",
            epoch,
            train_metrics["train_loss"],
            train_metrics["train_mean_iou"],
            val_metrics["val_loss"],
            val_metrics["val_mean_iou"],
            val_metrics["val_edge_iou_trees"],
            val_metrics["val_edge_iou_bushes"],
        )

        save_checkpoint(
            checkpoint_path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_iou=best_val_iou,
            config=config,
        )

        if val_metrics["val_mean_iou"] > best_val_iou:
            best_val_iou = val_metrics["val_mean_iou"]
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_iou=best_val_iou,
                config=config,
            )
            LOGGER.info("Saved new best checkpoint: %s (val_iou=%.4f)", best_path, best_val_iou)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.early_stopping_patience:
            LOGGER.info("Early stopping triggered at epoch %d", epoch)
            break

    LOGGER.info("Training complete. Best validation IoU: %.4f", best_val_iou)


if __name__ == "__main__":
    main()
