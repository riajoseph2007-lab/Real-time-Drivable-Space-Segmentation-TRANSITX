"""
Step 4: Training Loop (FIXED)
------------------------------
Fixes vs original:
  FIX A — model = FastSCNN(num_classes=..., aux_loss=...) now works because
           step2_model.py accepts aux_loss kwarg.
  FIX B — model.count_params() now returns dict; access via p['trainable'].
  FIX C — run_epoch: when aux_loss=True, model(imgs) in training mode
           correctly returns (main_logits, aux_logits) from updated FastSCNN.
  FIX D — get_dataloaders called with batch_size + seed (now supported by
           updated step1_dataset.py).
"""

import os
import time
import csv
import argparse
from pathlib import Path

import torch
import torch.nn as nn   

try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

from step1_dataset import (
    get_dataloaders, NUM_CLASSES, IMAGE_H, IMAGE_W, seed_everything
)
from step2_model import FastSCNN
from step3_loss  import CombinedLoss, AuxiliaryLoss, estimate_class_weights


# ─── Config ───────────────────────────────────────────────────────────────────

CFG = dict(
    dataroot     = os.environ.get("NUSCENES_DATAROOT", "C:/Users/anwes/Downloads/v1.0-mini"),
    epochs       = 100,
    lr           = 3e-3,
    weight_decay = 1e-4,
    batch_size   = int(os.environ.get("BATCH_SIZE", 8)),
    num_classes  = NUM_CLASSES,
    save_dir     = "./checkpoints",
    log_file     = "./train_log.csv",
    amp          = True,
    grad_clip    = 1.0,
    aux_loss     = True,
    aux_weight   = 0.4,
    seed         = int(os.environ.get("SEED", 42)),
    pct_start    = 0.1,
)


# ─── Metrics ──────────────────────────────────────────────────────────────────

class SegMetrics:
    """Running mIoU and pixel accuracy."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion = torch.zeros(self.num_classes, self.num_classes,
                                     dtype=torch.long)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds   = preds.view(-1).cpu()
        targets = targets.view(-1).cpu()
        valid   = (targets >= 0) & (targets < self.num_classes)
        preds, targets = preds[valid], targets[valid]
        idx = self.num_classes * targets + preds
        self.confusion += (
            torch.bincount(idx, minlength=self.num_classes ** 2)
            .reshape(self.num_classes, self.num_classes)
        )

    def miou(self) -> float:
        cm    = self.confusion.float()
        tp    = cm.diag()
        fp_fn = cm.sum(1) + cm.sum(0) - tp
        iou   = tp / (fp_fn + 1e-6)
        return iou.mean().item()

    def pixel_acc(self) -> float:
        cm = self.confusion.float()
        return (cm.diag().sum() / cm.sum().clamp(min=1)).item()


# ─── Training utilities ───────────────────────────────────────────────────────

def run_epoch(
    model, loader, criterion, aux_criterion,
    optimizer, scaler, device,
    scheduler=None, train: bool = True,
    cfg: dict = None,
) -> dict:
    if cfg is None:
        cfg = CFG
    model.train() if train else model.eval()

    metrics    = SegMetrics(cfg["num_classes"])
    total_loss = 0.0
    n_batches  = 0
    use_amp    = cfg["amp"] and device.type == "cuda"
    use_aux    = cfg.get("aux_loss", False) and train

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            # Dataset now always returns 3-tuple; guard for 2-tuple compat
            if len(batch) == 3:
                imgs, masks, weight_map = batch
                weight_map = weight_map.to(device, non_blocking=True)
            else:
                imgs, masks = batch
                weight_map  = None

            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=use_amp):
                output = model(imgs)
                # FIX C: unpack tuple only when aux head is active
                if use_aux and isinstance(output, (tuple, list)):
                    logits, aux_logits = output
                else:
                    logits = output[0] if isinstance(output, (tuple, list)) else output
                    aux_logits = None

                loss, breakdown = criterion(
                    logits, masks,
                    weight_map=weight_map if train else None,
                )

                if use_aux and aux_logits is not None and aux_criterion is not None:
                    aux_loss, aux_bd = aux_criterion(
                        aux_logits, masks,
                        weight_map=weight_map if train else None,
                    )
                    loss = loss + aux_loss
                    breakdown.update(aux_bd)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                if scheduler:
                    scheduler.step()

            preds = logits.argmax(dim=1)
            metrics.update(preds, masks)
            total_loss += loss.item()
            n_batches  += 1

    return {
        "loss":      total_loss / max(n_batches, 1),
        "miou":      metrics.miou(),
        "pixel_acc": metrics.pixel_acc(),
    }


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(path, epoch, model, optimizer, scheduler, best_miou, cfg):
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "sched_state": scheduler.state_dict(),
        "best_miou":   best_miou,
        "cfg":         cfg,
    }, path)


def load_checkpoint(
    model: FastSCNN,
    path: str,
    device,
    optimizer=None,
    scheduler=None,
) -> dict:
    ckpt = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if unexpected:
        print(f"[Checkpoint] Skipped keys: {unexpected}")
    if missing:
        print(f"[Checkpoint] Missing keys: {missing}")
    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    if scheduler is not None and "sched_state" in ckpt:
        scheduler.load_state_dict(ckpt["sched_state"])
    print(
        f"[Checkpoint] Resumed from epoch {ckpt['epoch']}, "
        f"best mIoU {ckpt.get('best_miou', 0):.4f}"
    )
    return ckpt


# ─── Main training function ───────────────────────────────────────────────────

def train(cfg: dict = CFG, resume: str = None):
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    os.makedirs(cfg["save_dir"], exist_ok=True)

    # FIX D: pass batch_size + seed
    train_loader, val_loader = get_dataloaders(
        cfg["dataroot"],
        batch_size=cfg["batch_size"],
        seed=cfg["seed"],
    )

    print("[Train] Estimating class weights …")
    class_weights = estimate_class_weights(train_loader, cfg["num_classes"])
    print(f"[Train] Class weights: {class_weights.tolist()}")

    # FIX A: aux_loss kwarg now accepted
    model = FastSCNN(
        num_classes=cfg["num_classes"],
        aux_loss=cfg.get("aux_loss", False),
    ).to(device)

    # FIX B: count_params() returns dict
    p = model.count_params()
    print(f"[Train] Parameters: {p['trainable']:,} trainable / {p['total']:,} total")

    criterion = CombinedLoss(
        alpha=0.5, gamma=2.0,
        class_weights=class_weights,
    )
    aux_criterion = AuxiliaryLoss(
        aux_weight=cfg.get("aux_weight", 0.4),
        alpha=0.5, gamma=2.0,
        class_weights=class_weights,
    ) if cfg.get("aux_loss", False) else None

    optimizer   = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    total_steps = cfg["epochs"] * len(train_loader)
    scheduler   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["lr"],
        total_steps=total_steps,
        pct_start=cfg.get("pct_start", 0.1),
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )

    use_amp = cfg["amp"] and device.type == "cuda"
    scaler  = GradScaler(device.type) if use_amp else GradScaler("cpu")

    start_epoch = 1
    best_iou    = 0.0

    if resume and Path(resume).exists():
        ckpt        = load_checkpoint(model, resume, device, optimizer, scheduler)
        start_epoch = ckpt["epoch"] + 1
        best_iou    = ckpt.get("best_miou", 0.0)
        print(f"[Train] Resuming from epoch {start_epoch}")

    log_path   = Path(cfg["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file   = open(log_path, "a" if resume else "w", newline="")
    fieldnames = [
        "epoch", "train_loss", "train_miou", "train_acc",
        "val_loss", "val_miou", "val_acc", "lr", "epoch_time_s"
    ]
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not resume:
        csv_writer.writeheader()

    print(f"\n{'─'*70}")
    print(f"{'Epoch':>6}  {'TrLoss':>8}  {'TrIoU':>7}  {'VlLoss':>8}  {'VlIoU':>7}  {'LR':>8}")
    print(f"{'─'*70}")

    try:
        for epoch in range(start_epoch, cfg["epochs"] + 1):
            t0 = time.time()

            tr = run_epoch(
                model, train_loader, criterion, aux_criterion,
                optimizer, scaler, device,
                scheduler=scheduler, train=True, cfg=cfg,
            )
            vl = run_epoch(
                model, val_loader, criterion, None,
                None, scaler, device,
                train=False, cfg=cfg,
            )

            elapsed = time.time() - t0
            lr_now  = optimizer.param_groups[0]["lr"]

            print(
                f"{epoch:>6}  {tr['loss']:>8.4f}  {tr['miou']:>7.4f}  "
                f"{vl['loss']:>8.4f}  {vl['miou']:>7.4f}  {lr_now:>8.2e}"
            )

            csv_writer.writerow({
                "epoch":        epoch,
                "train_loss":   f"{tr['loss']:.5f}",
                "train_miou":   f"{tr['miou']:.5f}",
                "train_acc":    f"{tr['pixel_acc']:.5f}",
                "val_loss":     f"{vl['loss']:.5f}",
                "val_miou":     f"{vl['miou']:.5f}",
                "val_acc":      f"{vl['pixel_acc']:.5f}",
                "lr":           f"{lr_now:.6e}",
                "epoch_time_s": f"{elapsed:.1f}",
            })
            csv_file.flush()

            if vl["miou"] > best_iou:
                best_iou  = vl["miou"]
                ckpt_path = Path(cfg["save_dir"]) / "best_model.pt"
                save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler,
                                best_iou, cfg)
                print(f"  ↑ New best mIoU {best_iou:.4f} → {ckpt_path}")

            save_checkpoint(
                Path(cfg["save_dir"]) / "latest.pt",
                epoch, model, optimizer, scheduler, best_iou, cfg,
            )

    except KeyboardInterrupt:
        print("\n[Train] Interrupted — saving checkpoint …")
        save_checkpoint(
            Path(cfg["save_dir"]) / "interrupted.pt",
            epoch, model, optimizer, scheduler, best_iou, cfg,
        )

    finally:
        csv_file.close()

    print(f"\n[Train] Done. Best val mIoU: {best_iou:.4f}")
    return model, best_iou


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot",   default=CFG["dataroot"])
    parser.add_argument("--epochs",     type=int,   default=CFG["epochs"])
    parser.add_argument("--lr",         type=float, default=CFG["lr"])
    parser.add_argument("--batch-size", type=int,   default=CFG["batch_size"])
    parser.add_argument("--save-dir",   default=CFG["save_dir"])
    parser.add_argument("--resume",     default=None)
    parser.add_argument("--no-aux",     action="store_true")
    parser.add_argument("--no-amp",     action="store_true")
    parser.add_argument("--seed",       type=int, default=CFG["seed"])
    args = parser.parse_args()

    cfg = dict(CFG)
    cfg.update({
        "dataroot":   args.dataroot,
        "epochs":     args.epochs,
        "lr":         args.lr,
        "batch_size": args.batch_size,
        "save_dir":   args.save_dir,
        "aux_loss":   not args.no_aux,
        "amp":        not args.no_amp,
        "seed":       args.seed,
    })

    train(cfg, resume=args.resume)
