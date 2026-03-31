"""
Step 2: Model Architecture — Fast-SCNN from scratch (FIXED)
------------------------------------------------------------
Changes vs original:
  FIX A — FastSCNN.__init__ now accepts aux_loss=True/False.
           When True, an AuxHead is attached to the GFE output and
           forward() returns (main_logits, aux_logits).
           When False (inference), only main_logits is returned.
           step4_train.py passes aux_loss=cfg["aux_loss"].
           step5_eval.py passes aux_loss=False.

  FIX B — count_params() returns a dict with 'trainable' and 'total' keys
           instead of a bare int, matching step4_train / step5_eval usage:
               p = model.count_params()
               print(p['trainable'], p['total'])

Everything else is unchanged from the original working architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Building Blocks ──────────────────────────────────────────────────────────

class DepthwiseSeparableConv(nn.Module):
    """Depthwise + pointwise conv with BN + ReLU."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, padding: int = 1):
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=padding,
                             groups=in_ch, bias=False)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual block."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expand: int = 6):
        super().__init__()
        mid_ch         = in_ch * expand
        self.use_skip  = (stride == 1) and (in_ch == out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1,
                      groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return x + out if self.use_skip else out


class PyramidPoolingModule(nn.Module):
    """Lightweight PPM: pools at 1×1, 2×2, 3×3, 6×6."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        inter_ch   = in_ch // 4
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_ch, inter_ch, 1, bias=False),
                nn.BatchNorm2d(inter_ch),
                nn.ReLU(inplace=True),
            ) for s in [1, 2, 3, 6]
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch + 4 * inter_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w   = x.shape[2], x.shape[3]
        pooled = [F.interpolate(s(x), size=(h, w), mode="bilinear",
                                align_corners=False) for s in self.stages]
        return self.bottleneck(torch.cat([x] + pooled, dim=1))


# ─── Main Modules ─────────────────────────────────────────────────────────────

class LearningToDownsample(nn.Module):
    """Shallow stem: 512×256 → 64×32 (3 strides of 2)."""
    def __init__(self, out_ch: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(32, 48, stride=2),
            DepthwiseSeparableConv(48, out_ch, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class GlobalFeatureExtractor(nn.Module):
    """Deep encoder: 64×32 → 8×4 with inverted residuals + PPM."""
    def __init__(self, in_ch: int = 64, out_ch: int = 128):
        super().__init__()
        self.bottlenecks = nn.Sequential(
            InvertedResidual(in_ch,  64,      stride=2, expand=6),
            InvertedResidual(64,     64,      stride=1, expand=6),
            InvertedResidual(64,     64,      stride=1, expand=6),
            InvertedResidual(64,     96,      stride=2, expand=6),
            InvertedResidual(96,     96,      stride=1, expand=6),
            InvertedResidual(96,     96,      stride=1, expand=6),
            InvertedResidual(96,     out_ch,  stride=2, expand=6),
            InvertedResidual(out_ch, out_ch,  stride=1, expand=6),
            InvertedResidual(out_ch, out_ch,  stride=1, expand=6),
        )
        self.ppm = PyramidPoolingModule(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ppm(self.bottlenecks(x))


class FeatureFusionModule(nn.Module):
    """Fuse high-res (LDS) + low-res (GFE) features."""
    def __init__(self, high_ch: int = 64, low_ch: int = 128, out_ch: int = 128):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(low_ch, low_ch, 3, padding=4, dilation=4,
                      groups=low_ch, bias=False),
            nn.BatchNorm2d(low_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(low_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.high_proj = nn.Sequential(
            nn.Conv2d(high_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        low_up = F.interpolate(low, size=high.shape[2:],
                               mode="bilinear", align_corners=False)
        low_up = self.upsample(low_up)
        return self.relu(low_up + self.high_proj(high))


class ClassifierHead(nn.Module):
    """Final head: fused features → (B, num_classes, H, W) logits."""
    def __init__(self, in_ch: int = 128, num_classes: int = 2):
        super().__init__()
        self.head = nn.Sequential(
            DepthwiseSeparableConv(in_ch, in_ch),
            DepthwiseSeparableConv(in_ch, in_ch),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_ch, num_classes, 1),
        )

    def forward(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        return F.interpolate(self.head(x), size=target_size,
                             mode="bilinear", align_corners=False)


# FIX A: auxiliary head for deep supervision
class AuxHead(nn.Module):
    """Small auxiliary classifier attached to GFE output."""
    def __init__(self, in_ch: int = 128, num_classes: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        return F.interpolate(self.conv(x), size=target_size,
                             mode="bilinear", align_corners=False)


# ─── Full Model ───────────────────────────────────────────────────────────────

class FastSCNN(nn.Module):
    """
    Fast-SCNN for drivable space segmentation (trained from scratch).

    Args:
        num_classes: number of output classes (default 2).
        aux_loss:    if True, attach an auxiliary head to GFE output and
                     return (main_logits, aux_logits) during training.
                     Set False for inference / eval (step5_eval.py).

    Input:  (B, 3, H, W)
    Output: (B, num_classes, H, W)  — or tuple when aux_loss=True
    """

    def __init__(self, num_classes: int = 2, aux_loss: bool = False):
        super().__init__()
        self.aux_loss = aux_loss
        self.lds      = LearningToDownsample(out_ch=64)
        self.gfe      = GlobalFeatureExtractor(in_ch=64, out_ch=128)
        self.ffm      = FeatureFusionModule(high_ch=64, low_ch=128, out_ch=128)
        self.head     = ClassifierHead(in_ch=128, num_classes=num_classes)

        if aux_loss:
            self.aux_head = AuxHead(in_ch=128, num_classes=num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        target_size = (x.shape[2], x.shape[3])
        high  = self.lds(x)               # (B, 64,  H/8,  W/8)
        low   = self.gfe(high)            # (B, 128, H/64, W/64)
        fused = self.ffm(high, low)       # (B, 128, H/8,  W/8)
        main  = self.head(fused, target_size)   # (B, C, H, W)

        if self.aux_loss and self.training:
            aux = self.aux_head(low, target_size)   # (B, C, H, W)
            return main, aux

        return main

    # FIX B: returns dict so callers can do p['trainable'] / p['total']
    def count_params(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test without aux loss (inference mode)
    model = FastSCNN(num_classes=2, aux_loss=False)
    x     = torch.randn(2, 3, 256, 512)
    out   = model(x)
    assert out.shape == (2, 2, 256, 512), f"Shape mismatch: {out.shape}"
    p = model.count_params()
    print(f"Input   : {x.shape}")
    print(f"Output  : {out.shape}")
    print(f"Params  : {p['trainable']:,} trainable / {p['total']:,} total")

    # Test with aux loss (training mode)
    model_aux = FastSCNN(num_classes=2, aux_loss=True)
    model_aux.train()
    main_out, aux_out = model_aux(x)
    assert main_out.shape == (2, 2, 256, 512)
    assert aux_out.shape  == (2, 2, 256, 512)
    print(f"Aux out : {aux_out.shape}")

    print("✓ Step 2 OK")
