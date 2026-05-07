"""Model, losses, and metrics for boundary-aware semantic segmentation."""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BoundaryLossConfig, get_config


class ConvBlock(nn.Module):
    """Two-layer convolutional block used in U-Net."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class UpBlock(nn.Module):
    """Upsampling + skip-connection fusion block."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(inputs)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat((skip, x), dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net segmentation model."""

    def __init__(
        self,
        num_classes: int,
        encoder_channels: Sequence[int],
        bottleneck_channels: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        if len(encoder_channels) < 2:
            raise ValueError("encoder_channels must have at least two entries.")

        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        in_channels = 3
        for channels in encoder_channels:
            self.encoder_blocks.append(ConvBlock(in_channels, channels, dropout=dropout_rate))
            in_channels = channels

        self.bottleneck = ConvBlock(encoder_channels[-1], bottleneck_channels, dropout=dropout_rate)

        self.decoder_blocks = nn.ModuleList()
        decoder_in = bottleneck_channels
        for skip_channels in reversed(encoder_channels):
            self.decoder_blocks.append(
                UpBlock(
                    in_channels=decoder_in,
                    skip_channels=skip_channels,
                    out_channels=skip_channels,
                    dropout=dropout_rate,
                )
            )
            decoder_in = skip_channels

        self.classifier = nn.Conv2d(encoder_channels[0], num_classes, kernel_size=1)

    def freeze_encoder(self) -> None:
        """Freeze encoder parameters, useful for staged training."""

        for block in self.encoder_blocks:
            for param in block.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters."""

        for block in self.encoder_blocks:
            for param in block.parameters():
                param.requires_grad = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        x = inputs
        for block in self.encoder_blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for decoder_block, skip in zip(self.decoder_blocks, reversed(skips)):
            x = decoder_block(x, skip)

        return self.classifier(x)


class BoundaryLoss(nn.Module):
    """Active contour-inspired boundary loss via gradient alignment."""

    def __init__(self, focus_classes: Iterable[int], ignore_index: int = 255) -> None:
        super().__init__()
        self.focus_classes = tuple(focus_classes)
        self.ignore_index = ignore_index

        sobel_x = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        sobel_y = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)

    def _edge_map(self, tensor: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(tensor, self.sobel_x, padding=1)
        gy = F.conv2d(tensor, self.sobel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        valid_mask = (targets != self.ignore_index).unsqueeze(1).float()

        losses: List[torch.Tensor] = []
        for class_id in self.focus_classes:
            pred_prob = probs[:, class_id : class_id + 1, :, :]
            target_binary = (targets == class_id).float().unsqueeze(1)

            pred_edges = self._edge_map(pred_prob)
            target_edges = self._edge_map(target_binary)

            masked_pred = pred_edges * valid_mask
            masked_target = target_edges * valid_mask
            losses.append(F.l1_loss(masked_pred, masked_target))

        if not losses:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        return torch.stack(losses).mean()


class CombinedSegmentationLoss(nn.Module):
    """Cross-entropy + boundary loss."""

    def __init__(
        self,
        class_weights: torch.Tensor,
        ce_weight: float,
        boundary_weight: float,
        focus_classes: Iterable[int],
        ignore_index: int,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.boundary_weight = boundary_weight
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.boundary = BoundaryLoss(focus_classes=focus_classes, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        ce = self.cross_entropy(logits, targets)
        bl = self.boundary(logits, targets)
        total = self.ce_weight * ce + self.boundary_weight * bl
        return total, {"ce": float(ce.detach().item()), "boundary": float(bl.detach().item())}


def _edge_from_binary(binary: torch.Tensor) -> torch.Tensor:
    max_pool = F.max_pool2d(binary, kernel_size=3, stride=1, padding=1)
    min_pool = -F.max_pool2d(-binary, kernel_size=3, stride=1, padding=1)
    return (max_pool - min_pool) > 0


def compute_batch_mean_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> float:
    """Compute mean IoU over all classes for a batch."""

    ious: List[torch.Tensor] = []
    valid = targets != ignore_index
    for class_id in range(num_classes):
        pred_class = (predictions == class_id) & valid
        target_class = (targets == class_id) & valid
        intersection = (pred_class & target_class).sum().float()
        union = (pred_class | target_class).sum().float()
        if union > 0:
            ious.append(intersection / (union + 1e-8))
    if not ious:
        return 0.0
    return float(torch.stack(ious).mean().item())


def compute_edge_iou_for_class(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_id: int,
    ignore_index: int,
) -> float:
    """Compute IoU on boundary pixels for a specific class."""

    valid = targets != ignore_index
    pred_binary = ((predictions == class_id) & valid).float().unsqueeze(1)
    target_binary = ((targets == class_id) & valid).float().unsqueeze(1)

    pred_edge = _edge_from_binary(pred_binary).squeeze(1)
    target_edge = _edge_from_binary(target_binary).squeeze(1)

    intersection = (pred_edge & target_edge).sum().float()
    union = (pred_edge | target_edge).sum().float()
    if union <= 0:
        return 0.0
    return float((intersection / (union + 1e-8)).item())


def build_model(config: Optional[BoundaryLossConfig] = None) -> UNet:
    """Factory function to create a U-Net instance from config."""

    cfg = config or get_config()
    return UNet(
        num_classes=cfg.num_classes,
        encoder_channels=cfg.encoder_channels,
        bottleneck_channels=cfg.bottleneck_channels,
        dropout_rate=cfg.dropout_rate,
    )
"""Model and loss definitions for boundary-aware semantic segmentation."""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BoundaryLossConfig, get_config


class ConvBlock(nn.Module):
    """Two-layer convolutional block with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """U-Net encoder-decoder segmentation model."""

    def __init__(
        self,
        num_classes: int,
        encoder_channels: Sequence[int],
        bottleneck_channels: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        channels = list(encoder_channels)
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        in_ch = 3
        for ch in channels:
            self.encoder_blocks.append(ConvBlock(in_channels=in_ch, out_channels=ch, dropout_rate=dropout_rate))
            in_ch = ch

        self.bottleneck = ConvBlock(in_channels=channels[-1], out_channels=bottleneck_channels, dropout_rate=dropout_rate)

        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        decoder_in = bottleneck_channels
        for skip_ch in reversed(channels):
            self.upconvs.append(nn.ConvTranspose2d(decoder_in, skip_ch, kernel_size=2, stride=2))
            self.decoder_blocks.append(
                ConvBlock(in_channels=skip_ch * 2, out_channels=skip_ch, dropout_rate=dropout_rate)
            )
            decoder_in = skip_ch

        self.classifier = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def freeze_encoder(self) -> None:
        """Freeze encoder parameters for warmup/fine-tuning phases."""

        for param in self.encoder_blocks.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters."""

        for param in self.encoder_blocks.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []

        for encoder in self.encoder_blocks:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = list(reversed(skips))

        for upconv, decoder, skip in zip(self.upconvs, self.decoder_blocks, skips):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        return self.classifier(x)


class ActiveContourBoundaryLoss(nn.Module):
    """Boundary loss using gradient alignment between prediction and target edges."""

    def __init__(self, kernel_size: int = 3, ignore_index: int = 255) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.ignore_index = ignore_index

        sobel_x = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        sobel_y = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _edge_magnitude(self, tensor_map: torch.Tensor) -> torch.Tensor:
        grad_x = F.conv2d(tensor_map, self.sobel_x, padding=1)
        grad_y = F.conv2d(tensor_map, self.sobel_y, padding=1)
        return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        focus_classes: Optional[Iterable[int]] = None,
    ) -> torch.Tensor:
        num_classes = logits.shape[1]
        probabilities = torch.softmax(logits, dim=1)

        valid_mask = (target != self.ignore_index).float().unsqueeze(1)
        sanitized_target = target.clone()
        sanitized_target[target == self.ignore_index] = 0

        one_hot_target = F.one_hot(sanitized_target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        if focus_classes is not None:
            class_mask = torch.zeros((num_classes,), dtype=logits.dtype, device=logits.device)
            for class_id in focus_classes:
                if 0 <= class_id < num_classes:
                    class_mask[class_id] = 1.0
            class_mask = class_mask.view(1, num_classes, 1, 1)
            probabilities = probabilities * class_mask
            one_hot_target = one_hot_target * class_mask

        pred_edges = []
        target_edges = []
        for class_idx in range(num_classes):
            pred_edges.append(self._edge_magnitude(probabilities[:, class_idx : class_idx + 1]))
            target_edges.append(self._edge_magnitude(one_hot_target[:, class_idx : class_idx + 1]))

        pred_edge_map = torch.cat(pred_edges, dim=1)
        target_edge_map = torch.cat(target_edges, dim=1)

        loss = F.l1_loss(pred_edge_map * valid_mask, target_edge_map * valid_mask)
        return loss


class CombinedSegmentationLoss(nn.Module):
    """Cross-Entropy + Active-Contour Boundary Loss."""

    def __init__(
        self,
        class_weights: torch.Tensor,
        ce_weight: float,
        boundary_weight: float,
        ignore_index: int,
        boundary_kernel_size: int,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.boundary_weight = boundary_weight
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.boundary = ActiveContourBoundaryLoss(kernel_size=boundary_kernel_size, ignore_index=ignore_index)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        focus_classes: Optional[Iterable[int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        ce_loss = self.cross_entropy(logits, target)
        boundary_loss = self.boundary(logits, target, focus_classes=focus_classes)
        total = self.ce_weight * ce_loss + self.boundary_weight * boundary_loss
        return total, {
            "ce_loss": float(ce_loss.detach().item()),
            "boundary_loss": float(boundary_loss.detach().item()),
            "total_loss": float(total.detach().item()),
        }


def _binary_edges(binary_mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Extract a thin edge map from a binary mask using local max/min morphology."""

    float_mask = binary_mask.float().unsqueeze(1)
    max_local = F.max_pool2d(float_mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    min_local = -F.max_pool2d(-float_mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    edges = (max_local - min_local) > 0
    return edges.squeeze(1)


def compute_mean_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> float:
    """Compute mean IoU over classes with valid union."""

    ious: List[float] = []
    valid_mask = targets != ignore_index

    for class_id in range(num_classes):
        pred_class = (predictions == class_id) & valid_mask
        target_class = (targets == class_id) & valid_mask
        intersection = torch.logical_and(pred_class, target_class).sum().item()
        union = torch.logical_or(pred_class, target_class).sum().item()
        if union > 0:
            ious.append(intersection / union)

    if not ious:
        return 0.0
    return float(sum(ious) / len(ious))


def compute_edge_iou_for_class(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_id: int,
    ignore_index: int,
    kernel_size: int = 3,
) -> float:
    """Compute IoU specifically on boundary pixels of one class."""

    valid = targets != ignore_index
    pred_binary = ((predictions == class_id) & valid)
    target_binary = ((targets == class_id) & valid)

    pred_edges = _binary_edges(pred_binary, kernel_size=kernel_size)
    target_edges = _binary_edges(target_binary, kernel_size=kernel_size)

    intersection = torch.logical_and(pred_edges, target_edges).sum().item()
    union = torch.logical_or(pred_edges, target_edges).sum().item()
    if union == 0:
        return 0.0
    return float(intersection / union)


def build_model(config: Optional[BoundaryLossConfig] = None) -> UNet:
    """Factory for model construction from configuration."""

    cfg = config or get_config()
    return UNet(
        num_classes=cfg.num_classes,
        encoder_channels=cfg.encoder_channels,
        bottleneck_channels=cfg.bottleneck_channels,
        dropout_rate=cfg.dropout_rate,
    )
