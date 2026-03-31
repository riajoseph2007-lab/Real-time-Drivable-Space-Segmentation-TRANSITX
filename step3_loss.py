"""
Step 3: Loss Functions
-----------------------

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with optional per-pixel weight map.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    weight_map (B, H, W): multiplied per-pixel before reduction.
    Boundary pixels are given higher weight values (e.g. 3.0 vs 1.0).
    """
    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: torch.Tensor = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma        = gamma
        self.class_weights = class_weights
        self.ignore_index  = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weight_map: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        logits:     (B, C, H, W)
        targets:    (B, H, W) int class labels
        weight_map: (B, H, W) float pixel weights  [optional]
        """
        targets   = targets.long()   # FIX 3a
        log_probs = F.log_softmax(logits, dim=1)
        probs     = log_probs.exp()

        targets_exp = targets.unsqueeze(1).clamp(0)            # (B,1,H,W)
        log_pt      = log_probs.gather(1, targets_exp).squeeze(1)  # (B,H,W)
        pt          = probs.gather(1, targets_exp).squeeze(1)

        focal_weight = (1 - pt) ** self.gamma

        if self.class_weights is not None:
            w     = self.class_weights.to(logits.device)
            alpha = w[targets.clamp(0)]      # FIX 3c
        else:
            alpha = torch.ones_like(pt)

        loss = -alpha * focal_weight * log_pt

        # Apply pixel-level boundary weight map
        if weight_map is not None:
            loss = loss * weight_map.to(loss.device)

        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            loss = loss[mask]

        return loss.mean()


# ─── Dice Loss ────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Soft multi-class Dice loss.
    Dice = (2 * TP + smooth) / (sum_pred + sum_true + smooth)
    Loss = 1 - mean_over_classes(Dice)

    weight_map is not applied here — Dice operates over entire spatial regions
    and applying per-pixel weights would distort the overlap measure.
    Boundary weighting is handled in FocalLoss.
    """
    def __init__(self, smooth: float = 1.0, ignore_index: int = -100):
        super().__init__()
        self.smooth       = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets     = targets.long()   # FIX 3a
        num_classes = logits.shape[1]
        probs       = F.softmax(logits, dim=1)

        valid_mask   = (targets != self.ignore_index).unsqueeze(1).float()
        safe_targets = targets.clamp(0)
        targets_oh   = F.one_hot(safe_targets, num_classes).permute(0, 3, 1, 2).float()
        targets_oh   = targets_oh * valid_mask
        probs        = probs * valid_mask

        dims  = (0, 2, 3)
        inter = (probs * targets_oh).sum(dim=dims)
        p_sum = probs.sum(dim=dims)
        t_sum = targets_oh.sum(dim=dims)
        dice  = (2 * inter + self.smooth) / (p_sum + t_sum + self.smooth)
        return 1.0 - dice.mean()


# ─── Boundary-Aware Combined Loss ─────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    alpha * BoundaryAwareFocalLoss + (1 - alpha) * DiceLoss

    Accepts an optional weight_map (B, H, W) from step1's
    compute_boundary_weight_map(). Pixels near class boundaries get
    weight = BOUNDARY_WEIGHT (default 3.0), all others get 1.0.
    This forces the model to focus on edge-case transitions:
      - road-to-grass
      - road-to-puddle / wet surface
      - road-to-curb / construction barriers

    class_weights: inverse class frequency tensor, e.g. [0.3, 0.7].
    """
    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 2.0,
        class_weights: torch.Tensor = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(
            gamma=gamma,
            class_weights=class_weights,
            ignore_index=ignore_index,
        )
        self.dice  = DiceLoss(ignore_index=ignore_index)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weight_map: torch.Tensor = None,
    ) -> tuple:
        """
        logits:     (B, C, H, W)
        targets:    (B, H, W)
        weight_map: (B, H, W) float  [optional, from compute_boundary_weight_map]

        Returns:
            total_loss (scalar), breakdown dict
        """
        focal_val = self.focal(logits, targets, weight_map=weight_map)
        dice_val  = self.dice(logits, targets)
        total     = self.alpha * focal_val + (1 - self.alpha) * dice_val
        return total, {
            "focal": focal_val.item(),
            "dice":  dice_val.item(),
        }


# ─── Auxiliary Loss ───────────────────────────────────────────────────────────

class AuxiliaryLoss(nn.Module):
    """
    Wraps CombinedLoss for the auxiliary head's logits.
    Applied at reduced weight (aux_weight) relative to the main loss.

    Usage in training loop:
        main_loss, _ = criterion(main_logits, masks, weight_map)
        aux_loss, _  = aux_criterion(aux_logits, masks, weight_map)
        total = main_loss + aux_criterion.aux_weight * aux_loss
    """
    def __init__(
        self,
        aux_weight: float = 0.4,
        alpha: float = 0.5,
        gamma: float = 2.0,
        class_weights: torch.Tensor = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.aux_weight = aux_weight
        self.loss_fn    = CombinedLoss(
            alpha=alpha,
            gamma=gamma,
            class_weights=class_weights,
            ignore_index=ignore_index,
        )

    def forward(
        self,
        aux_logits: torch.Tensor,
        targets: torch.Tensor,
        weight_map: torch.Tensor = None,
    ) -> tuple:
        loss, breakdown = self.loss_fn(aux_logits, targets, weight_map)
        return self.aux_weight * loss, {
            f"aux_{k}": v for k, v in breakdown.items()
        }


# ─── Class weight estimator ───────────────────────────────────────────────────

def estimate_class_weights(
    dataloader, num_classes: int = 2, max_batches: int = 20
) -> torch.Tensor:
    """
    Scan first max_batches batches to estimate inverse-frequency class weights.
    Handles 2-tuple (img, mask) or 3-tuple (img, mask, weight_map) batches.
    """
    counts = torch.zeros(num_classes)
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        masks = batch[1].long()   # FIX 3b: index 1 regardless of tuple length
        for c in range(num_classes):
            counts[c] += (masks == c).sum().item()
    counts  = counts.clamp(min=1)
    weights = 1.0 / counts
    return weights / weights.sum() * num_classes


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, C, H, W = 2, 2, 256, 512
    logits  = torch.randn(B, C, H, W)
    targets = torch.randint(0, C, (B, H, W))

    # Simulate boundary weight map (boundary pixels = 3.0, others = 1.0)
    weight_map = torch.ones(B, H, W)
    weight_map[:, 100:110, :] = 3.0   # fake boundary band

    cw   = torch.tensor([0.3, 0.7])
    loss_fn = CombinedLoss(alpha=0.5, class_weights=cw)

    total, breakdown = loss_fn(logits, targets, weight_map=weight_map)
    print(f"Total loss  : {total.item():.4f}")
    print(f"  Focal     : {breakdown['focal']:.4f}")
    print(f"  Dice      : {breakdown['dice']:.4f}")

    # Test without weight map (backward compat)
    total2, _ = loss_fn(logits, targets)
    print(f"No weight map total: {total2.item():.4f}")

    # Aux loss
    aux_fn    = AuxiliaryLoss(aux_weight=0.4, class_weights=cw)
    aux_val, aux_bd = aux_fn(logits, targets, weight_map)
    print(f"Aux loss    : {aux_val.item():.4f}  {aux_bd}")

    print("✓ Step 3 OK")
