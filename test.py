"""Inference and evaluation script for Boundary Loss Integrator."""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from config import BoundaryLossConfig, get_config
from dataset import create_eval_dataloader
from model import compute_batch_mean_iou, compute_edge_iou_for_class, build_model


LOGGER = logging.getLogger("boundary_loss_test")


def setup_logging(level: str) -> None:
    """Set up standard logging for inference."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def update_confusion_matrix(
    confusion: np.ndarray,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> None:
    """Update confusion matrix counts for one batch."""

    pred_flat = predictions.detach().cpu().numpy().reshape(-1)
    target_flat = targets.detach().cpu().numpy().reshape(-1)
    valid = target_flat != ignore_index
    pred_valid = pred_flat[valid]
    target_valid = target_flat[valid]

    encoded = target_valid * num_classes + pred_valid
    bins = np.bincount(encoded, minlength=num_classes * num_classes)
    confusion += bins.reshape(num_classes, num_classes)


def compute_iou_from_confusion(confusion: np.ndarray) -> Dict[int, float]:
    """Compute per-class IoU from confusion matrix."""

    iou_by_class: Dict[int, float] = {}
    for class_id in range(confusion.shape[0]):
        tp = float(confusion[class_id, class_id])
        fp = float(confusion[:, class_id].sum() - tp)
        fn = float(confusion[class_id, :].sum() - tp)
        denom = tp + fp + fn
        iou_by_class[class_id] = (tp / denom) if denom > 0.0 else 0.0
    return iou_by_class


def _extract_binary_edges(binary_mask: np.ndarray, edge_width: int) -> np.ndarray:
    """Extract a binary edge map without external cv2 dependency."""

    padded = np.pad(binary_mask.astype(np.uint8), pad_width=1, mode="edge")
    center = padded[1:-1, 1:-1]
    neighbors = [
        padded[:-2, 1:-1],
        padded[2:, 1:-1],
        padded[1:-1, :-2],
        padded[1:-1, 2:],
        padded[:-2, :-2],
        padded[:-2, 2:],
        padded[2:, :-2],
        padded[2:, 2:],
    ]
    edge = np.zeros_like(center, dtype=bool)
    for n in neighbors:
        edge |= n != center

    if edge_width <= 1:
        return edge

    thick = edge.copy()
    for _ in range(edge_width - 1):
        padded_thick = np.pad(thick.astype(np.uint8), pad_width=1, mode="constant")
        neighborhood = [
            padded_thick[:-2, :-2],
            padded_thick[:-2, 1:-1],
            padded_thick[:-2, 2:],
            padded_thick[1:-1, :-2],
            padded_thick[1:-1, 1:-1],
            padded_thick[1:-1, 2:],
            padded_thick[2:, :-2],
            padded_thick[2:, 1:-1],
            padded_thick[2:, 2:],
        ]
        thick = np.any(np.stack(neighborhood, axis=0) > 0, axis=0)
    return thick


def save_high_contrast_edge_visual(
    image_path: Path,
    prediction: np.ndarray,
    target: np.ndarray,
    config: BoundaryLossConfig,
    output_path: Path,
) -> None:
    """Generate high-contrast boundary visualization for report-ready figures."""

    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

    target_trees_edge = _extract_binary_edges(target == config.edge_focus_classes[0], config.high_contrast_edge_width)
    target_bushes_edge = _extract_binary_edges(target == config.edge_focus_classes[1], config.high_contrast_edge_width)
    pred_trees_edge = _extract_binary_edges(prediction == config.edge_focus_classes[0], config.high_contrast_edge_width)
    pred_bushes_edge = _extract_binary_edges(prediction == config.edge_focus_classes[1], config.high_contrast_edge_width)

    overlay = rgb.copy()

    overlay[target_trees_edge] = np.array([255, 255, 0], dtype=np.uint8)
    overlay[target_bushes_edge] = np.array([255, 165, 0], dtype=np.uint8)
    overlay[pred_trees_edge] = np.array([0, 255, 255], dtype=np.uint8)
    overlay[pred_bushes_edge] = np.array([255, 0, 255], dtype=np.uint8)

    blended = (0.55 * rgb + 0.45 * overlay).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(blended).save(output_path)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: BoundaryLossConfig,
    device: torch.device,
    save_visual_count: int,
) -> Dict[str, object]:
    """Run inference, collect timing metrics, IoU metrics, and confusion matrix."""

    model.eval()
    confusion = np.zeros((config.num_classes, config.num_classes), dtype=np.int64)

    per_image_ms: List[float] = []
    total_iou = 0.0
    trees_edge_iou = 0.0
    bushes_edge_iou = 0.0
    sample_count = 0

    visual_saved = 0

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        logits = model(images)
        predictions = logits.argmax(dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        batch_size = images.shape[0]
        per_image_ms.extend([elapsed_ms / batch_size] * batch_size)

        update_confusion_matrix(
            confusion=confusion,
            predictions=predictions,
            targets=targets,
            num_classes=config.num_classes,
            ignore_index=config.ignore_index,
        )

        total_iou += compute_batch_mean_iou(
            predictions=predictions,
            targets=targets,
            num_classes=config.num_classes,
            ignore_index=config.ignore_index,
        )
        trees_edge_iou += compute_edge_iou_for_class(
            predictions=predictions,
            targets=targets,
            class_id=config.edge_focus_classes[0],
            ignore_index=config.ignore_index,
        )
        bushes_edge_iou += compute_edge_iou_for_class(
            predictions=predictions,
            targets=targets,
            class_id=config.edge_focus_classes[1],
            ignore_index=config.ignore_index,
        )
        sample_count += 1

        if visual_saved < save_visual_count:
            pred_np = predictions[0].detach().cpu().numpy()
            target_np = targets[0].detach().cpu().numpy()
            image_path = Path(batch["path"][0])
            output_path = config.vis_dir / f"high_contrast_{visual_saved:04d}.png"
            save_high_contrast_edge_visual(
                image_path=image_path,
                prediction=pred_np,
                target=target_np,
                config=config,
                output_path=output_path,
            )
            visual_saved += 1

    per_class_iou = compute_iou_from_confusion(confusion)
    mean_class_iou = float(np.mean(list(per_class_iou.values()))) if per_class_iou else 0.0

    return {
        "mean_iou": mean_class_iou,
        "batch_mean_iou": total_iou / max(sample_count, 1),
        "edge_iou_trees": trees_edge_iou / max(sample_count, 1),
        "edge_iou_bushes": bushes_edge_iou / max(sample_count, 1),
        "latency_ms_mean": float(np.mean(per_image_ms)) if per_image_ms else 0.0,
        "latency_ms_p95": float(np.percentile(per_image_ms, 95)) if per_image_ms else 0.0,
        "latency_ms_min": float(np.min(per_image_ms)) if per_image_ms else 0.0,
        "latency_ms_max": float(np.max(per_image_ms)) if per_image_ms else 0.0,
        "per_class_iou": per_class_iou,
        "confusion_matrix": confusion.tolist(),
    }


def parse_args() -> argparse.Namespace:
    """CLI argument parser for evaluation."""

    parser = argparse.ArgumentParser(description="Boundary Loss Integrator inference script")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Dataset root containing images/ and masks/")
    parser.add_argument("--device", type=str, default=None, help="Device override: cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference timing")
    parser.add_argument(
        "--save-high-contrast",
        type=int,
        default=8,
        help="Number of high-contrast boundary visualizations to save",
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=Path("results") / "metrics.json",
        help="Path to write final metrics JSON",
    )
    return parser.parse_args()


def main() -> None:
    """Program entrypoint for inference and evaluation."""

    args = parse_args()
    config = get_config()

    if args.dataset_dir is not None:
        config.dataset_dir = args.dataset_dir
    if args.device is not None:
        config.device = args.device

    setup_logging(config.log_level)

    requested_cuda = config.device.lower().startswith("cuda")
    if requested_cuda and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but not available. Falling back to CPU.")
    device = torch.device("cuda" if requested_cuda and torch.cuda.is_available() else "cpu")

    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    eval_loader = create_eval_dataloader(config=config, dataset_dir=config.dataset_dir, batch_size=args.batch_size)

    results = run_inference(
        model=model,
        dataloader=eval_loader,
        config=config,
        device=device,
        save_visual_count=max(0, args.save_high_contrast),
    )

    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.output_metrics.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    LOGGER.info("Inference complete.")
    LOGGER.info("Mean IoU: %.4f", results["mean_iou"])
    LOGGER.info("Edge IoU Trees (raw=100): %.4f", results["edge_iou_trees"])
    LOGGER.info("Edge IoU Bushes (raw=500): %.4f", results["edge_iou_bushes"])
    LOGGER.info(
        "Latency ms/image: mean=%.2f p95=%.2f min=%.2f max=%.2f",
        results["latency_ms_mean"],
        results["latency_ms_p95"],
        results["latency_ms_min"],
        results["latency_ms_max"],
    )
    LOGGER.info("Confusion matrix saved to %s", args.output_metrics)

    # Tradeoff note: to satisfy strict latency targets (e.g. <50ms/image), you may
    # need to reduce input resolution or channel width in config while monitoring
    # edge IoU, because aggressive speed optimization can degrade boundary fidelity.


if __name__ == "__main__":
    main()
