# Drivable Space Segmentation — Fast-SCNN on nuScenes

A lightweight real-time semantic segmentation pipeline for detecting drivable road surfaces from front-camera images, built on top of the [nuScenes](https://www.nuscenes.org/) dataset using a from-scratch Fast-SCNN architecture.

---

## Overview

This project segments front-camera driving images into two classes:

| Class | Label | Description |
|-------|-------|-------------|
| 0 | Non-drivable | Background, kerbs, buildings, sky, etc. |
| 1 | Drivable | Road surface, including wet/puddle areas |

The pipeline is split into five self-contained steps:

```
step1_dataset.py   →   step2_model.py   →   step3_loss.py
                                                   ↓
                             step5_eval.py  ←  step4_train.py
```

---

## Project Structure

```
├── step1_dataset.py   # nuScenes data loading, mask generation, augmentation
├── step2_model.py     # Fast-SCNN model architecture
├── step3_loss.py      # Focal, Dice, and combined boundary-aware loss functions
├── step4_train.py     # Training loop, metrics, checkpointing
└── step5_eval.py      # Evaluation, FPS benchmark, visualisation, ONNX/TorchScript export
```

---

## Requirements

**Python**: 3.8+

Install dependencies:

```bash
pip install torch torchvision
pip install nuscenes-devkit
pip install albumentations
pip install opencv-python
pip install onnx           # optional, for ONNX export verification
```

**Dataset**: [nuScenes v1.0-mini](https://www.nuscenes.org/download). Download and extract it, then set the path via environment variable or the `DATAROOT` config constant:

```bash
export NUSCENES_DATAROOT=/path/to/v1.0-mini
```

---

## Step-by-Step Guide

### Step 1 — Dataset (`step1_dataset.py`)

Loads nuScenes front-camera (`CAM_FRONT`) images and generates drivable-space segmentation masks using a hybrid image-space heuristic (no 3-D map data required):

- **Cue 1**: Road-colour segmentation in HSV (grey tones = asphalt) in the lower 65% of the image.
- **Cue 2**: Geometric trapezoid prior — the lower image region is assumed drivable.
- **Cue 3**: Sobel edge detection to locate the road/non-road horizon.

The final mask = `cue1 AND (cue2 OR cue3)` with morphological clean-up.

Each `__getitem__` returns a **3-tuple**: `(image [3,H,W], mask [H,W], weight_map [H,W])`. The `weight_map` assigns higher loss weight (`3.0`) to pixels near class boundaries (road edges, kerb transitions, wet-surface boundaries).

**Sanity check:**

```bash
python step1_dataset.py
# Saves sanity_check_overlay.png and prints mask statistics
```

Key constants (editable in the file):

| Constant | Default | Description |
|---|---|---|
| `IMAGE_W / IMAGE_H` | 512 / 256 | Resized image dimensions |
| `CAMERA` | `CAM_FRONT` | nuScenes camera channel |
| `BOUNDARY_WEIGHT` | 3.0 | Weight multiplier for boundary pixels |
| `BOUNDARY_DILATE` | 5 px | Dilation radius for boundary region |

---

### Step 2 — Model (`step2_model.py`)

Implements **Fast-SCNN** from scratch — a lightweight encoder–decoder architecture designed for real-time inference on embedded hardware.

**Architecture:**

```
Input (B,3,H,W)
      │
  LearningToDownsample (LDS)   →  (B, 64,  H/8,  W/8)   3 strided convs
      │
  GlobalFeatureExtractor (GFE) →  (B, 128, H/64, W/64)  9× InvertedResidual + PPM
      │
  FeatureFusionModule (FFM)    →  (B, 128, H/8,  W/8)   fuses LDS + GFE features
      │
  ClassifierHead               →  (B, C,   H,    W)     bilinear upsample to input size
      │
  [AuxHead on GFE output]      →  (B, C,   H,    W)     only active when aux_loss=True
```

**Usage:**

```python
from step2_model import FastSCNN

# Inference (no auxiliary head)
model = FastSCNN(num_classes=2, aux_loss=False)
logits = model(images)             # (B, 2, H, W)

# Training (with auxiliary head)
model = FastSCNN(num_classes=2, aux_loss=True)
model.train()
main_logits, aux_logits = model(images)

# Parameter count
p = model.count_params()
print(p['trainable'], p['total'])
```

**Quick test:**

```bash
python step2_model.py
```

---

### Step 3 — Loss Functions (`step3_loss.py`)

Implements a **boundary-aware combined loss** to improve segmentation accuracy at class transitions (e.g. road-to-kerb, road-to-puddle).

| Class | Description |
|---|---|
| `FocalLoss` | Multi-class focal loss with optional per-class weights and boundary pixel weight map |
| `DiceLoss` | Soft multi-class Dice loss for overlap-based optimisation |
| `CombinedLoss` | `α × FocalLoss + (1−α) × DiceLoss` with optional boundary weight map |
| `AuxiliaryLoss` | Wraps `CombinedLoss` for the auxiliary head at reduced weight |

The `weight_map` (from `compute_boundary_weight_map`) is applied in `FocalLoss` to upweight pixels near boundaries. `DiceLoss` intentionally ignores the weight map to preserve the integrity of the overlap measure.

**Usage:**

```python
from step3_loss import CombinedLoss, AuxiliaryLoss

class_weights = torch.tensor([0.3, 0.7])
criterion     = CombinedLoss(alpha=0.5, gamma=2.0, class_weights=class_weights)

total_loss, breakdown = criterion(logits, masks, weight_map=weight_map)
# breakdown = {"focal": ..., "dice": ...}
```

**Class weight estimation:**

```python
from step3_loss import estimate_class_weights
weights = estimate_class_weights(train_loader, num_classes=2, max_batches=20)
```

**Quick test:**

```bash
python step3_loss.py
```

---

### Step 4 — Training (`step4_train.py`)

Full training loop with mixed-precision, OneCycleLR scheduling, gradient clipping, CSV logging, and checkpoint saving.

**Run training:**

```bash
python step4_train.py \
  --dataroot /path/to/v1.0-mini \
  --epochs 100 \
  --batch-size 8 \
  --lr 3e-3 \
  --save-dir ./checkpoints
```

**Resume from checkpoint:**

```bash
python step4_train.py --resume ./checkpoints/latest.pt
```

**Key flags:**

| Flag | Default | Description |
|---|---|---|
| `--dataroot` | `NUSCENES_DATAROOT` env | Path to nuScenes dataset |
| `--epochs` | 100 | Number of training epochs |
| `--lr` | 3e-3 | Peak learning rate (OneCycleLR) |
| `--batch-size` | 8 | Batch size |
| `--save-dir` | `./checkpoints` | Directory to save checkpoints |
| `--resume` | None | Path to checkpoint to resume from |
| `--no-aux` | False | Disable auxiliary loss head |
| `--no-amp` | False | Disable automatic mixed precision |
| `--seed` | 42 | Random seed |

**Training configuration** (`CFG` dict):

```python
CFG = dict(
    epochs       = 100,
    lr           = 3e-3,
    weight_decay = 1e-4,
    batch_size   = 8,
    amp          = True,       # mixed precision
    grad_clip    = 1.0,
    aux_loss     = True,
    aux_weight   = 0.4,
    pct_start    = 0.1,        # OneCycleLR warm-up fraction
)
```

**Outputs:**

- `./checkpoints/best_model.pt` — best validation mIoU checkpoint
- `./checkpoints/latest.pt` — most recent epoch checkpoint
- `./train_log.csv` — per-epoch metrics log

**Metrics tracked:** training/validation loss, mIoU, pixel accuracy, learning rate, epoch time.

---

### Step 5 — Evaluation & Export (`step5_eval.py`)

Runs full evaluation, FPS benchmarking, prediction visualisation, and model export.

**Run evaluation:**

```bash
python step5_eval.py \
  --checkpoint ./checkpoints/best_model.pt \
  --dataroot /path/to/v1.0-mini \
  --batch-size 8
```

**Metrics reported:**

| Metric | Description |
|---|---|
| mIoU | Mean intersection-over-union across both classes |
| Pixel accuracy | Fraction of correctly classified pixels |
| IoU (background) | Per-class IoU for non-drivable |
| IoU (drivable) | Per-class IoU for road surface |
| Boundary mIoU | mIoU computed only on pixels near class boundaries |
| Boundary IoU (bg / drivable) | Per-class boundary IoU |

**Visualisation:**

Saves 5-panel composite images for 10 validation samples to `./viz/`:

```
[ Original | Ground Truth | Prediction | Overlay | Boundary Weight Map ]
```

**Model export:**

The script automatically exports to both formats after evaluation:

```
./model.onnx      # ONNX (opset 17, dynamic batch axis)
./model_ts.pt     # TorchScript (traced)
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NUSCENES_DATAROOT` | `C:/Users/anwes/Downloads/v1.0-mini` | Path to dataset root |
| `BATCH_SIZE` | 4 (dataset) / 8 (training) | Override batch size |
| `SEED` | 42 | Global random seed |

---

## Architecture Summary

| Module | Input | Output | Purpose |
|---|---|---|---|
| `LearningToDownsample` | (B,3,H,W) | (B,64,H/8,W/8) | Lightweight spatial downsampling |
| `GlobalFeatureExtractor` | (B,64,H/8,W/8) | (B,128,H/64,W/64) | Deep contextual features via InvertedResidual + PPM |
| `FeatureFusionModule` | high + low | (B,128,H/8,W/8) | Fuse multi-scale features |
| `ClassifierHead` | (B,128,H/8,W/8) | (B,C,H,W) | Final segmentation logits |
| `AuxHead` (training only) | (B,128,H/64,W/64) | (B,C,H,W) | Auxiliary deep supervision |

---

## Known Limitations

- Mask generation uses image-space heuristics rather than ground-truth nuScenes map annotations; accuracy may vary in unusual lighting or road conditions.
- `NUM_WORKERS=0` is enforced on Windows to avoid multiprocessing issues with the nuScenes devkit.
- The nuScenes vector map (`NuScenesMap`) is disabled for `v1.0-mini` as vector JSON files are not included in that split.

---

## Model Metrics
   - Best Validation mIoU: 0.7909 (~79.1%)
   - Final Training Pixel Accuracy: ~97.5%
   - Final Training Loss: ~0.043
   - Validation Loss Range: ~0.13 – 0.18
   - Final Training IoU: ~0.97
   - Final Validation IoU: ~0.73 – 0.79
   - Key Observations
  - The model shows consistent improvement in mIoU during training
  - Achieves ~79% validation mIoU, indicating strong segmentation performance
  - High training IoU (~97%) confirms effective learning
  - Boundary-aware loss improves edge detection (road vs non-road)
  - Model is optimized for real-time in

## License

This project is for research and educational use. The nuScenes dataset is subject to its own [Terms of Use](https://www.nuscenes.org/terms-of-use).
