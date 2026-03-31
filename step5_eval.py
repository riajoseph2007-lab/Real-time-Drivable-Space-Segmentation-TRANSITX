"""
Step 5: Evaluation, Visualization & Export (FIXED)
---------------------------------------------------
Fixes vs original:
  FIX A — model.count_params() returns dict; access p['trainable'] / p['total'].
  FIX B — FastSCNN(num_classes=..., aux_loss=False) now accepted by step2_model.
  FIX C — load_checkpoint imported from step4_train (unchanged).
  All other logic unchanged.
"""

import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)
import time
import argparse
import numpy as np
import cv2
from pathlib import Path

import torch
import torch.nn.functional as F

from step1_dataset import (
    get_dataloaders, NuScenesSegDataset, val_transform,
    IMAGENET_MEAN, IMAGENET_STD, IMAGE_W, IMAGE_H, NUM_CLASSES,
    compute_boundary_weight_map, seed_everything, DATAROOT,
)
from step2_model  import FastSCNN
from step4_train  import SegMetrics, load_checkpoint


# ─── Colour palette ───────────────────────────────────────────────────────────

PALETTE = np.array([
    [220,  50,  50],   # non-drivable — red
    [ 50, 205,  50],   # drivable     — green
], dtype=np.uint8)


# ─── 1. Full Evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    metrics          = SegMetrics(NUM_CLASSES)
    boundary_metrics = SegMetrics(NUM_CLASSES)

    for batch in loader:
        if len(batch) == 3:
            imgs, masks, _ = batch
        else:
            imgs, masks = batch

        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(imgs)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        preds = logits.argmax(dim=1)

        metrics.update(preds, masks)

        for b in range(masks.shape[0]):
            mask_np = masks[b].cpu().numpy().astype(np.uint8)
            bw      = compute_boundary_weight_map(mask_np)
            bp = preds[b].cpu() * torch.from_numpy((bw > 1.0).astype(np.int64))
            bt = masks[b].cpu() * torch.from_numpy((bw > 1.0).astype(np.int64))
            boundary_metrics.update(bp.unsqueeze(0), bt.unsqueeze(0))

    cm    = metrics.confusion.float()
    tp    = cm.diag()
    union = cm.sum(1) + cm.sum(0) - tp
    per_class_iou = (tp / (union + 1e-6)).tolist()

    bcm    = boundary_metrics.confusion.float()
    btp    = bcm.diag()
    bunion = bcm.sum(1) + bcm.sum(0) - btp
    boundary_iou = (btp / (bunion + 1e-6)).tolist()

    return {
        "mIoU":                  metrics.miou(),
        "pixel_acc":             metrics.pixel_acc(),
        "iou_background":        per_class_iou[0],
        "iou_drivable":          per_class_iou[1],
        "boundary_mIoU":         boundary_metrics.miou(),
        "boundary_iou_bg":       boundary_iou[0],
        "boundary_iou_drivable": boundary_iou[1],
    }


# ─── 2. FPS Benchmark ─────────────────────────────────────────────────────────

def benchmark_fps(
    model, device,
    input_size: tuple = (1, 3, IMAGE_H, IMAGE_W),
    warmup: int = 20,
    runs:   int = 200,
) -> float:
    model.eval()
    x = torch.randn(*input_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.no_grad():
            for _ in range(runs):
                _ = model(x)
        e.record()
        torch.cuda.synchronize()
        fps = runs / (s.elapsed_time(e) / 1000.0)
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(runs):
                _ = model(x)
        fps = runs / (time.perf_counter() - t0)
    return fps


# ─── 3. Mask Overlay Visualisation ────────────────────────────────────────────

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD ).view(3, 1, 1)
    img  = (tensor.cpu() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


@torch.no_grad()
def visualize_predictions(
    model, loader, device,
    save_dir: str = "./viz",
    n_samples: int = 10,
    alpha: float = 0.45,
):
    """
    Saves 5-panel images:
      [original | ground truth | prediction | overlay | boundary weight map]
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    count = 0

    for batch in loader:
        if count >= n_samples:
            break
        if len(batch) == 3:
            imgs, masks, weight_maps = batch
        else:
            imgs, masks = batch
            weight_maps = None

        imgs_dev = imgs.to(device)
        logits   = model(imgs_dev)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        preds = logits.argmax(dim=1).cpu()

        for i in range(imgs.shape[0]):
            if count >= n_samples:
                break

            img_np   = denormalize(imgs[i])
            gt_np    = masks[i].numpy().astype(np.uint8)
            pred_np  = preds[i].numpy().astype(np.uint8)

            gt_colour   = PALETTE[gt_np]
            pred_colour = PALETTE[pred_np]
            overlay     = (img_np.astype(float) * (1 - alpha) +
                           pred_colour.astype(float) * alpha).astype(np.uint8)

            if weight_maps is not None:
                wm_np = weight_maps[i].numpy()
            else:
                wm_np = compute_boundary_weight_map(gt_np)
            wm_vis  = ((wm_np / wm_np.max()) * 255).astype(np.uint8)
            wm_rgb  = cv2.cvtColor(wm_vis, cv2.COLOR_GRAY2BGR)

            panels = [
                (cv2.cvtColor(img_np,      cv2.COLOR_RGB2BGR), "Original"),
                (cv2.cvtColor(gt_colour,   cv2.COLOR_RGB2BGR), "Ground Truth"),
                (cv2.cvtColor(pred_colour, cv2.COLOR_RGB2BGR), "Prediction"),
                (cv2.cvtColor(overlay,     cv2.COLOR_RGB2BGR), "Overlay"),
                (wm_rgb,                                        "Boundary"),
            ]
            for panel, label in panels:
                cv2.putText(panel, label, (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                            cv2.LINE_AA)

            combined = np.concatenate([p for p, _ in panels], axis=1)
            cv2.imwrite(os.path.join(save_dir, f"sample_{count:04d}.png"), combined)
            count += 1

    print(f"[Viz] Saved {count} images to {save_dir}/")


# ─── 4. ONNX Export ───────────────────────────────────────────────────────────

def export_onnx(model, path: str = "./model.onnx", device=torch.device("cpu")):
    model.eval().to(device)
    dummy = torch.randn(1, 3, IMAGE_H, IMAGE_W, device=device)
    torch.onnx.export(
        model, dummy, path,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )
    print(f"[Export] ONNX saved → {path}")
    try:
        import onnx
        onnx.checker.check_model(onnx.load(path))
        print("[Export] ONNX model verified ✓")
    except ImportError:
        print("[Export] Install 'onnx' to verify the exported model.")


# ─── 5. TorchScript Export ────────────────────────────────────────────────────

def export_torchscript(model, path: str = "./model_ts.pt",
                       device=torch.device("cpu")):
    model.eval().to(device)
    dummy  = torch.randn(1, 3, IMAGE_H, IMAGE_W, device=device)
    traced = torch.jit.trace(model, dummy)
    traced.save(path)
    print(f"[Export] TorchScript saved → {path}")


# ─── Main evaluation runner ───────────────────────────────────────────────────

def run_eval(checkpoint_path: str, dataroot: str = DATAROOT, batch_size: int = 8):
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")

    ckpt_peek = torch.load(checkpoint_path, map_location="cpu")
    saved_cfg = ckpt_peek.get("cfg", {})
    print(f"[Eval] Checkpoint: epoch {ckpt_peek.get('epoch','?')}, "
          f"best mIoU {ckpt_peek.get('best_miou', 0):.4f}, "
          f"aux_loss={saved_cfg.get('aux_loss', False)}")
    del ckpt_peek

    # FIX B: aux_loss=False for inference
    model = FastSCNN(num_classes=NUM_CLASSES, aux_loss=False)
    load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)

    _, val_loader = get_dataloaders(dataroot, batch_size=batch_size)

    print("\n[Eval] Running full validation …")
    results = evaluate(model, val_loader, device)
    print(f"\n{'─'*50}")
    print(f"  mIoU                    : {results['mIoU']:.4f}")
    print(f"  Pixel accuracy          : {results['pixel_acc']:.4f}")
    print(f"  IoU (background)        : {results['iou_background']:.4f}")
    print(f"  IoU (drivable)          : {results['iou_drivable']:.4f}")
    print(f"  Boundary mIoU           : {results['boundary_mIoU']:.4f}")
    print(f"  Boundary IoU (bg)       : {results['boundary_iou_bg']:.4f}")
    print(f"  Boundary IoU (drivable) : {results['boundary_iou_drivable']:.4f}")
    print(f"{'─'*50}\n")

    print("[Eval] Benchmarking FPS …")
    fps_cpu = benchmark_fps(model, torch.device("cpu"), runs=50)
    print(f"  FPS (CPU)   : {fps_cpu:.1f}")
    if torch.cuda.is_available():
        fps_gpu = benchmark_fps(model, torch.device("cuda"), runs=200)
        print(f"  FPS (GPU)   : {fps_gpu:.1f}")
    # FIX A: count_params() returns dict
    p = model.count_params()
    print(f"  Parameters  : {p['trainable']:,} trainable / {p['total']:,} total\n")

    print("[Eval] Saving prediction visualizations …")
    visualize_predictions(model, val_loader, device, n_samples=10)

    print("[Eval] Exporting model …")
    export_onnx(model,        "./model.onnx",  torch.device("cpu"))
    export_torchscript(model, "./model_ts.pt", torch.device("cpu"))

    print("\n[Eval] Complete.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="./checkpoints/best_model.pt")
    parser.add_argument("--dataroot",   default=DATAROOT)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    run_eval(args.checkpoint, dataroot=args.dataroot, batch_size=args.batch_size)
