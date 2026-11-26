"""
AdaptNet
===========================================================
"""

import math
import os
import time
import copy
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger("adaptnet_v5")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s %(levelname)s: %(message)s")

# ──────────────────────────────────────────────────────────────────────────────
#  DEVICE / ACCELERATOR DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def _detect_accelerator() -> str:
    if torch.cuda.is_available(): return "cuda"
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
        return "xla"
    except ImportError: pass
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"

ACCELERATOR: str = _detect_accelerator()

def get_device(rank: int = 0) -> torch.device:
    if ACCELERATOR == "cuda": return torch.device(f"cuda:{rank}")
    if ACCELERATOR == "xla":
        import torch_xla.core.xla_model as xm  # type: ignore
        return xm.xla_device()
    if ACCELERATOR == "mps": return torch.device("mps")
    return torch.device("cpu")

# ──────────────────────────────────────────────────────────────────────────────
#  PRIMITIVE BLOCKS (With Reparameterization & Fusion Support)
# ──────────────────────────────────────────────────────────────────────────────

class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + Activation with mathematically sound fusion."""
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = -1,
                 g: int = 1, d: int = 1, bias: bool = False, act: Optional[nn.Module] = None, bn: bool = True) -> None:
        super().__init__()
        if p == -1: p = d * (k - 1) // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, d, g, bias)
        self.bn = nn.BatchNorm2d(out_c) if bn else nn.Identity()
        self.act = act if act is not None else nn.Identity()
        self.is_fused = False

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))

    def fuse(self) -> None:
        if self.is_fused or isinstance(self.bn, nn.Identity): return

        w = self.conv.weight
        b = self.conv.bias if self.conv.bias is not None else torch.zeros(self.conv.out_channels, device=w.device, dtype=w.dtype)

        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        gamma = self.bn.weight
        beta = self.bn.bias
        eps = self.bn.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).view(-1, 1, 1, 1)

        self.conv.weight.data.copy_(w * t)
        if self.conv.bias is None:
            self.conv.bias = nn.Parameter((b - running_mean) * t.view(-1) + beta)
        else:
            self.conv.bias.data.copy_((b - running_mean) * t.view(-1) + beta)

        self.bn = nn.Identity()
        self.is_fused = True

class RepDW(nn.Module):
    """Reparameterizable Depthwise Convolution (3x3 + 1x1 + Identity)."""
    def __init__(self, in_c: int, k: int = 3, s: int = 1, d: int = 1) -> None:
        super().__init__()
        self.in_c = in_c
        self.s = s
        self.d = d
        self.is_reparameterized = False

        self.conv3x3 = ConvBNAct(in_c, in_c, k=3, s=s, p=d, g=in_c, d=d, bn=True)

        self.use_rep = (s == 1)
        self.conv1x1 = ConvBNAct(in_c, in_c, k=1, s=1, p=0, g=in_c, bn=True) if self.use_rep else None
        self.identity = nn.BatchNorm2d(in_c) if self.use_rep else None

    def forward(self, x: Tensor) -> Tensor:
        if self.is_reparameterized:
            return self.conv3x3(x)

        out = self.conv3x3(x)
        if self.use_rep:
            out = out + self.conv1x1(x)
            if self.identity is not None:
                out = out + self.identity(x)
        return out

    def _fuse_bn(self, module: ConvBNAct) -> Tuple[Tensor, Tensor]:
        w = module.conv.weight
        b = module.conv.bias if module.conv.bias is not None else torch.zeros(module.conv.out_channels, device=w.device, dtype=w.dtype)

        if isinstance(module.bn, nn.BatchNorm2d):
            running_mean = module.bn.running_mean
            running_var = module.bn.running_var
            gamma = module.bn.weight
            beta = module.bn.bias
            eps = module.bn.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).view(-1, 1, 1, 1)
            w_fused = w * t
            b_fused = (b - running_mean) * t.view(-1) + beta
        else:
            w_fused = w
            b_fused = b
        return w_fused, b_fused

    def _fuse_bn_identity(self, bn: Optional[nn.BatchNorm2d]) -> Tuple[Tensor, Tensor]:
        w = torch.zeros(self.in_c, 1, 3, 3, device=self.conv3x3.conv.weight.device, dtype=self.conv3x3.conv.weight.dtype)
        w[:, :, 1, 1] = 1.0
        b = torch.zeros(self.in_c, device=w.device, dtype=w.dtype)

        if bn is not None:
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).view(-1, 1, 1, 1)
            w_fused = w * t
            b_fused = (b - running_mean) * t.view(-1) + beta
        else:
            w_fused = w
            b_fused = b
        return w_fused, b_fused

    def reparameterize(self) -> None:
        if self.is_reparameterized: return

        if not self.use_rep:
            self.conv3x3.fuse()
            self.is_reparameterized = True
            return

        w3, b3 = self._fuse_bn(self.conv3x3)
        w1, b1 = self._fuse_bn(self.conv1x1)

        w1_padded = F.pad(w1, [1, 1, 1, 1])
        w_fused = w3 + w1_padded
        b_fused = b3 + b1

        if self.identity is not None:
            wi, bi = self._fuse_bn_identity(self.identity)
            w_fused = w_fused + wi
            b_fused = b_fused + bi

        self.conv3x3.conv.weight.data.copy_(w_fused)
        if self.conv3x3.conv.bias is None:
            self.conv3x3.conv.bias = nn.Parameter(b_fused)
        else:
            self.conv3x3.conv.bias.data.copy_(b_fused)

        self.conv3x3.bn = nn.Identity()
        self.conv3x3.is_fused = True

        self.conv1x1 = None
        self.identity = None
        self.is_reparameterized = True

    def get_fused_weight_bias(self) -> Tuple[Tensor, Tensor]:
        if not self.is_reparameterized:
            self.reparameterize()
        return self.conv3x3.conv.weight, self.conv3x3.conv.bias

class LightSE(nn.Module):
    """Squeeze-and-Excitation with Squeeze Dimension Capping (max 64) and Hardsigmoid."""
    def __init__(self, in_channels: int, expanded_channels: int, ratio: int = 4, min_channels: int = 8) -> None:
        super().__init__()
        squeeze_dim = min(max(min_channels, in_channels // ratio), 64)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(expanded_channels, squeeze_dim, kernel_size=1, bias=True)
        self.act = nn.Hardswish(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_dim, expanded_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        s = self.gap(x)
        s = self.act(self.fc1(s))
        s = self.sigmoid(self.fc2(s))
        return x * s

def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    if drop_prob == 0.0 or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)

# ──────────────────────────────────────────────────────────────────────────────
#  DMSIR-v5 BLOCKS (Fused Early & Super-Kernel Late)
# ──────────────────────────────────────────────────────────────────────────────

class FusedDMSIRv5Block(nn.Module):
    """Dense 3x3 Convolution Block for Early High-Resolution Stages (FastViT Strategy)."""
    def __init__(self, in_c: int, out_c: int, mid_c: int, stride: int, use_se: bool = False, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.fused_conv = ConvBNAct(in_c, mid_c, k=3, s=stride, g=1, act=nn.Hardswish(inplace=True))
        self.se = LightSE(mid_c, mid_c, ratio=4) if use_se else nn.Identity()
        self.project = ConvBNAct(mid_c, out_c, k=1, bn=True, act=None)

        self.use_skip = (stride == 1) and (in_c == out_c)
        self.drop_path = DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self.fused_conv(x)
        out = self.se(out)
        out = self.project(out)
        if self.use_skip:
            out = x + self.drop_path(out)
        return out

    def switch_to_deploy(self) -> None:
        if hasattr(self.fused_conv, 'fuse'): self.fused_conv.fuse()
        if hasattr(self.project, 'fuse'): self.project.fuse()

class DMSIRv5Block(nn.Module):
    """Multi-Scale Context Block with Super-Kernel Reparameterization."""
    def __init__(self, in_c: int, out_c: int, mid_c: int, stride: int, use_se: bool = False, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.expand = ConvBNAct(in_c, mid_c, k=1, act=nn.Hardswish(inplace=True))

        # Asymmetric Split: 75% Local, 25% Context (Warp-32 Aligned)
        self.ctx_channels = _make_divisible(mid_c * 0.25, divisor=32)
        self.local_channels = mid_c - self.ctx_channels

        self.dw_local = RepDW(self.local_channels, k=3, s=stride, d=1)
        self.dw_ctx   = RepDW(self.ctx_channels, k=3, s=stride, d=2)

        self.act = nn.Hardswish(inplace=True)
        self.se = LightSE(mid_c, mid_c, ratio=4) if use_se else nn.Identity()
        self.project = ConvBNAct(mid_c, out_c, k=1, bn=True, act=None)

        self.use_skip = (stride == 1) and (in_c == out_c)
        self.drop_path = DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()
        self.is_deployed = False

    def forward(self, x: Tensor) -> Tensor:
        e = self.expand(x)

        if self.is_deployed:
            out = self.act(self.dw_super(e))
        else:
            e_local, e_ctx = torch.split(e, [self.local_channels, self.ctx_channels], dim=1)
            out1 = self.act(self.dw_local(e_local))
            out2 = self.act(self.dw_ctx(e_ctx))
            out = torch.cat([out1, out2], dim=1)

        out = self.se(out)
        out = self.project(out)

        if self.use_skip:
            out = x + self.drop_path(out)
        return out

    def switch_to_deploy(self) -> None:
        if hasattr(self.expand, 'fuse'): self.expand.fuse()
        if hasattr(self.project, 'fuse'): self.project.fuse()

        # 1. Reparameterize internal parallel branches (1x1, Identity) into 3x3
        w_local, b_local = self.dw_local.get_fused_weight_bias()
        w_ctx, b_ctx = self.dw_ctx.get_fused_weight_bias()

        # 2. Pad local 3x3 to 5x5
        w_local_5x5 = F.pad(w_local, [1, 1, 1, 1])

        # 3. Scatter dilated 3x3 into 5x5 grid (spacing of 2)
        w_ctx_5x5 = torch.zeros(self.ctx_channels, 1, 5, 5, device=w_ctx.device, dtype=w_ctx.dtype)
        w_ctx_5x5[:, :, 0::2, 0::2] = w_ctx

        # 4. Concatenate along the channel dimension
        w_fused = torch.cat([w_local_5x5, w_ctx_5x5], dim=0)
        b_fused = torch.cat([b_local, b_ctx], dim=0)

        # 5. Instantiate the Super-Kernel on the EXACT device/dtype to prevent CUDA/CPU mismatch
        self.dw_super = nn.Conv2d(
            self.local_channels + self.ctx_channels,
            self.local_channels + self.ctx_channels,
            kernel_size=5,
            stride=self.dw_local.s,
            padding=2,
            groups=self.local_channels + self.ctx_channels,
            bias=True,
            device=w_fused.device,
            dtype=w_fused.dtype
        )
        self.dw_super.weight.data.copy_(w_fused)
        self.dw_super.bias.data.copy_(b_fused)

        # 6. Delete original branches to free VRAM and prevent execution
        del self.dw_local
        del self.dw_ctx

        self.is_deployed = True

# ──────────────────────────────────────────────────────────────────────────────
#  SCALING CONFIG & STAGE SPECS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ScalingConfig:
    alpha: float = 1.2
    beta:  float = 1.1
    delta: float = 1.15
    # Asymmetric Compound Scaling: Starve early (1-3), feed late (4-7)
    gamma: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.60, 0.75, 0.90, 1.00)

    def depth_scale(self, phi: float, stage_idx: int) -> float:
        return phi ** (self.beta * self.gamma[stage_idx])

    def width_scale(self, phi: float, stage_idx: int) -> float:
        return phi ** (self.alpha * (1.0 - self.gamma[stage_idx]))

    def resolution_scale(self, phi: float) -> float:
        return phi ** self.delta

@dataclass
class StageSpec:
    in_c:   int
    out_c:  int
    expand: int
    stride: int
    blocks: int
    mid:    int
    use_se: bool = True
    fused:  bool = False

# Progressive Expansion Ratios & Fused Early Stages
BASE_STAGE_SPECS: Tuple[StageSpec, ...] = (
    StageSpec(in_c=32,  out_c=16,  expand=2, stride=1, blocks=1, mid=32,   use_se=False, fused=True),
    StageSpec(in_c=16,  out_c=24,  expand=3, stride=2, blocks=2, mid=48,   use_se=False, fused=True),
    StageSpec(in_c=24,  out_c=32,  expand=4, stride=2, blocks=2, mid=96,   use_se=False, fused=False),
    StageSpec(in_c=32,  out_c=64,  expand=4, stride=2, blocks=3, mid=192,  use_se=True,  fused=False),
    StageSpec(in_c=64,  out_c=96,  expand=6, stride=1, blocks=2, mid=384,  use_se=True,  fused=False),
    StageSpec(in_c=96,  out_c=160, expand=6, stride=2, blocks=3, mid=576,  use_se=True,  fused=False),
    StageSpec(in_c=160, out_c=256, expand=6, stride=1, blocks=1, mid=960,  use_se=True,  fused=False),
)

def _make_divisible(v: float, divisor: int = 32, min_val: Optional[int] = None) -> int:
    """Enforces strict Warp-32 alignment."""
    if min_val is None: min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v: new_v += divisor
    return new_v

def scale_stage_specs(phi: float, scaling: ScalingConfig) -> Tuple[Tuple[StageSpec, ...], int]:
    scaled = []
    for i, s in enumerate(BASE_STAGE_SPECS):
        w  = scaling.width_scale(phi, i)
        d  = scaling.depth_scale(phi, i)
        out_c  = _make_divisible(s.out_c * w)
        mid    = _make_divisible(s.mid   * w)
        blocks = max(1, round(s.blocks * d))
        in_c   = (_make_divisible(BASE_STAGE_SPECS[i - 1].out_c * scaling.width_scale(phi, i - 1)) if i > 0 else 32)
        scaled.append(StageSpec(in_c=in_c, out_c=out_c, expand=s.expand, stride=s.stride, blocks=blocks, mid=mid, use_se=s.use_se, fused=s.fused))

    base_res = round(224 * scaling.resolution_scale(phi) / 32) * 32
    return tuple(scaled), base_res

_VARIANT_PHI: Dict[str, float] = {
    "S0": 1.00, "S1": 1.15, "S2": 1.30,
    "S3": 1.50, "S4": 1.75, "S5": 2.10,
    "S6": 2.50, "S7": 3.00,
}

# ──────────────────────────────────────────────────────────────────────────────
#  ADAPTNET v5 MAIN CLASS
# ──────────────────────────────────────────────────────────────────────────────

class AdaptNetV5(nn.Module):
    def __init__(self, stage_specs: Tuple[StageSpec, ...], num_classes: int = 1000, head_channels: int = 1280, dropout: float = 0.2, in_channels: int = 3, drop_path_rate: float = 0.2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, 32, k=3, s=2, act=nn.ReLU(inplace=True)),
            ConvBNAct(32, 32, k=3, s=1, g=32, act=nn.ReLU(inplace=True)),
        )

        total_blocks = sum(s.blocks for s in stage_specs)
        block_idx = 0
        stages_layers: List[nn.Module] = []

        for spec in stage_specs:
            stage_blocks: List[nn.Module] = []
            for i in range(spec.blocks):
                in_c   = spec.in_c   if i == 0 else spec.out_c
                stride = spec.stride if i == 0 else 1
                mid_c  = spec.mid

                block_use_se = spec.use_se and (i == spec.blocks - 1)
                sd_prob = drop_path_rate * (block_idx / max(1, total_blocks - 1))

                if spec.fused:
                    stage_blocks.append(FusedDMSIRv5Block(in_c=in_c, out_c=spec.out_c, mid_c=mid_c, stride=stride, use_se=block_use_se, drop_prob=sd_prob))
                else:
                    stage_blocks.append(DMSIRv5Block(in_c=in_c, out_c=spec.out_c, mid_c=mid_c, stride=stride, use_se=block_use_se, drop_prob=sd_prob))
                block_idx += 1
            stages_layers.append(nn.Sequential(*stage_blocks))

        self.stages = nn.Sequential(*stages_layers)
        last_c = stage_specs[-1].out_c
        self.head_conv = ConvBNAct(last_c, head_channels, k=1, act=nn.Hardswish(inplace=True))
        self.head_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(p=dropout))
        self.classifier = nn.Linear(head_channels, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.head_conv(x)
        x = self.head_pool(x)
        return self.classifier(x)

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return self.head_conv(x)

    def switch_to_deploy(self) -> None:
        for m in self.modules():
            if m is self: continue
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()
            elif hasattr(m, 'reparameterize'):
                m.reparameterize()
            elif isinstance(m, ConvBNAct):
                m.fuse()

    def param_count(self, detailed: bool = False) -> Union[int, Tuple[int, int]]:
        total = sum(p.numel() for p in self.parameters())
        if not detailed: return total
        backbone = sum(p.numel() for n, p in self.named_parameters() if 'classifier' not in n and 'head' not in n)
        head = total - backbone
        return backbone, head

    @classmethod
    def build(cls, variant: str = "S0", num_classes: int = 1000, **kwargs: Any) -> "AdaptNetV5":
        if variant not in _VARIANT_PHI: raise ValueError(f"Unknown variant '{variant}'.")
        phi = _VARIANT_PHI[variant]
        specs, _ = scale_stage_specs(phi, ScalingConfig())
        return cls(stage_specs=specs, num_classes=num_classes, **kwargs)

# ──────────────────────────────────────────────────────────────────────────────
#  LOSS FUNCTIONS & AUGMENTATIONS
# ──────────────────────────────────────────────────────────────────────────────

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        self.smoothing = smoothing
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        log_prob = F.log_softmax(pred, dim=-1)
        loss     = -log_prob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth   = -log_prob.mean(dim=-1)
        return (1.0 - self.smoothing) * loss.mean() + self.smoothing * smooth.mean()

class SoftTargetCrossEntropy(nn.Module):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        log_prob = F.log_softmax(pred, dim=-1)
        return -(target * log_prob).sum(dim=-1).mean()

def mixup_data(x: Tensor, y: Tensor, alpha: float = 0.2) -> Tuple[Tensor, Tensor, Tensor, float]:
    lam     = float(torch.distributions.Beta(alpha, alpha).sample()) if alpha > 0 else 1.0
    idx     = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[idx]
    return mixed_x, y, y[idx], lam

def mixup_criterion(criterion: nn.Module, pred: Tensor, y_a: Tensor, y_b: Tensor, lam: float) -> Tensor:
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)

def rand_bbox(size: Tuple[int, ...], lam: float) -> Tuple[int, int, int, int]:
    W, H   = size[-1], size[-2]
    cut_w  = int(W * math.sqrt(1.0 - lam))
    cut_h  = int(H * math.sqrt(1.0 - lam))
    cx     = int(torch.randint(W, (1,)).item())
    cy     = int(torch.randint(H, (1,)).item())
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2

def cutmix_data(x: Tensor, y: Tensor, alpha: float = 1.0) -> Tuple[Tensor, Tensor, Tensor, float]:
    lam             = float(torch.distributions.Beta(alpha, alpha).sample()) if alpha > 0 else 1.0
    idx             = torch.randperm(x.size(0), device=x.device)
    x1, y1, x2, y2 = rand_bbox(x.shape, lam)
    x_cut           = x.clone()
    x_cut[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam             = 1.0 - (y2 - y1) * (x2 - x1) / (x.shape[-1] * x.shape[-2])
    return x_cut, y, y[idx], lam

# ──────────────────────────────────────────────────────────────────────────────
#  EMA
# ──────────────────────────────────────────────────────────────────────────────

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay  = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.copy_(self.decay * s + (1.0 - self.decay) * m.detach())
        for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.copy_(m_buf)

    def __call__(self, x: Tensor) -> Tensor:
        return self.shadow(x)

# ──────────────────────────────────────────────────────────────────────────────
#  TRAINING CONFIG & OPTIMIZER UTILS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    variant:             str   = "S0"
    num_classes:         int   = 1000
    epochs:              int   = 300
    batch_size:          int   = 256
    base_lr:             float = 1e-3
    weight_decay:        float = 0.05
    dropout:             float = 0.2
    drop_path_rate:      float = 0.2
    warmup_epochs:       int   = 5
    label_smoothing:     float = 0.1
    mixup_alpha:         float = 0.8
    cutmix_alpha:        float = 1.0
    cutmix_prob:         float = 0.5
    grad_clip:           float = 1.0
    amp:                 bool  = True
    ema_decay:           float = 0.9999
    ema:                 bool  = True
    checkpoint_dir:      str   = "checkpoints"
    log_interval:        int   = 50
    num_workers:         int   = 8
    pin_memory:          bool  = True
    compile_model:       bool  = False
    distributed:         bool  = False
    world_size:          int   = 1
    dist_backend:        str   = "nccl"
    dist_url:            str   = "env://"
    layer_wise_lr_decay: float = 0.8
    lr_schedule:         str   = "cosine"
    seed:                int   = 42
    randaug_num_ops:     int   = 2
    randaug_magnitude:   int   = 9

def build_optimizer(model: AdaptNetV5, cfg: TrainingConfig) -> AdamW:
    params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(params, lr=cfg.base_lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999), eps=1e-8)

def build_scheduler(optimizer: AdamW, cfg: TrainingConfig, steps_per_epoch: int) -> Any:
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    warmup = LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_steps)
    cosine_steps = (cfg.epochs - cfg.warmup_epochs) * steps_per_epoch
    main = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warmup, main], milestones=[warmup_steps])

# ──────────────────────────────────────────────────────────────────────────────
#  METRICS & DEVICE SCHEDULER
# ──────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self) -> None: self.reset()
    def reset(self) -> None: self.val = self.avg = self.sum = 0.0; self.count = 0
    def update(self, val: float, n: int = 1) -> None:
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

def accuracy(output: Tensor, target: Tensor, topk: Tuple[int, ...] = (1, 5)) -> List[Tensor]:
    with torch.no_grad():
        maxk = max(topk); batch = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t(); correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk: res.append(correct[:k].reshape(-1).float().sum(0).mul_(100.0 / batch))
        return res

class _NullContext:
    def __enter__(self): return self
    def __exit__(self, *a): pass

class DeviceScheduler:
    def __init__(self, rank: int = 0, world_size: int = 1, amp: bool = True) -> None:
        self.rank = rank; self.world_size = world_size; self.device = get_device(rank)
        self.amp = amp and (ACCELERATOR == "cuda"); self.scaler = GradScaler("cuda") if self.amp else None
        self._xla = ACCELERATOR == "xla"; self._xm = None
        if self._xla:
            import torch_xla.core.xla_model as xm; self._xm = xm

    def autocast_ctx(self): return autocast("cuda") if self.amp else _NullContext()
    def backward(self, loss: Tensor) -> None:
        if self.amp and self.scaler is not None: self.scaler.scale(loss).backward()
        else: loss.backward()
    def clip_grad(self, model: nn.Module, optimizer: Any, max_norm: float) -> None:
        if self.amp and self.scaler is not None: self.scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    def step(self, optimizer: Any) -> None:
        if self._xla: self._xm.optimizer_step(optimizer)
        elif self.amp and self.scaler is not None: self.scaler.step(optimizer); self.scaler.update()
        else: optimizer.step()
    def mark_step(self) -> None:
        if self._xla: self._xm.mark_step()
    def barrier(self) -> None:
        if self._xla: self._xm.rendezvous("barrier")
        elif dist.is_initialized(): dist.barrier()
    def reduce_mean(self, tensor: Tensor) -> Tensor:
        if self.world_size > 1:
            if self._xla: tensor = self._xm.all_reduce(self._xm.REDUCE_SUM, tensor) / self.world_size
            else: dist.all_reduce(tensor, op=dist.ReduceOp.SUM); tensor.div_(self.world_size)
        return tensor
    def is_main(self) -> bool: return self.rank == 0

# ──────────────────────────────────────────────────────────────────────────────
#  CHECKPOINT MANAGER & TRAINER
# ──────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    def __init__(self, directory: str, keep_top_k: int = 3) -> None:
        self.directory = directory; self.keep_top_k = keep_top_k; self._history = []
        os.makedirs(directory, exist_ok=True)
    def save(self, state: Dict[str, Any], epoch: int, metric: float, is_best: bool = False) -> str:
        path = os.path.join(self.directory, f"epoch_{epoch:04d}_acc{metric:.3f}.pt")
        torch.save(state, path); self._history.append((metric, path)); self._history.sort(key=lambda x: -x[0])
        while len(self._history) > self.keep_top_k:
            _, old_path = self._history.pop()
            if os.path.exists(old_path): os.remove(old_path)
        if is_best: torch.save(state, os.path.join(self.directory, "best.pt"))
        return path
    def load(self, path: str, device: torch.device) -> Dict[str, Any]:
        try: return torch.load(path, map_location=device, weights_only=False)
        except TypeError: return torch.load(path, map_location=device)

class Trainer:
    def __init__(self, model: AdaptNetV5, cfg: TrainingConfig, train_loader: DataLoader, val_loader: DataLoader, rank: int = 0) -> None:
        self.cfg = cfg; self.rank = rank; self.sched = DeviceScheduler(rank, cfg.world_size, cfg.amp)
        self.device = self.sched.device; self.ckpt = CheckpointManager(cfg.checkpoint_dir)
        model = model.to(self.device)
        if cfg.compile_model and hasattr(torch, "compile"): model = torch.compile(model)
        if cfg.distributed and dist.is_initialized(): model = DDP(model, device_ids=[rank], find_unused_parameters=False)
        self.model = model; self.raw_model = model.module if hasattr(model, "module") else model
        self.ema = ModelEMA(self.raw_model, decay=cfg.ema_decay) if cfg.ema else None
        self.optimizer = build_optimizer(self.raw_model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg, steps_per_epoch=len(train_loader))
        self.criterion = LabelSmoothingCrossEntropy(smoothing=cfg.label_smoothing).to(self.device)
        self.train_loader = train_loader; self.val_loader = val_loader
        self.best_acc1 = 0.0; self.start_epoch = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train(); losses = AverageMeter(); top1 = AverageMeter(); top5 = AverageMeter(); t0 = time.time()
        for step, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True); targets = targets.to(self.device, non_blocking=True)
            r = torch.rand(1).item()
            if r < self.cfg.cutmix_prob and self.cfg.cutmix_alpha > 0:
                images, y_a, y_b, lam = cutmix_data(images, targets, self.cfg.cutmix_alpha); use_mix = True
            elif self.cfg.mixup_alpha > 0:
                images, y_a, y_b, lam = mixup_data(images, targets, self.cfg.mixup_alpha); use_mix = True
            else: use_mix = False; lam, y_a, y_b = 1.0, targets, targets
            self.optimizer.zero_grad(set_to_none=True)
            with self.sched.autocast_ctx():
                output = self.model(images)
                loss = mixup_criterion(self.criterion, output, y_a, y_b, lam) if use_mix else self.criterion(output, targets)
            self.sched.backward(loss)
            if self.cfg.grad_clip > 0: self.sched.clip_grad(self.model, self.optimizer, self.cfg.grad_clip)
            self.sched.step(self.optimizer); self.scheduler.step(); self.sched.mark_step()
            if self.ema is not None: self.ema.update(self.raw_model)
            loss_val = loss.detach(); self.sched.reduce_mean(loss_val)
            log_targets = y_a if (use_mix and lam >= 0.5) else (y_b if use_mix else targets)
            acc1_val, acc5_val = accuracy(output.detach(), log_targets, topk=(1, 5))
            bs = images.size(0); losses.update(loss_val.item(), bs); top1.update(acc1_val.item(), bs); top5.update(acc5_val.item(), bs)
            if step % self.cfg.log_interval == 0 and self.sched.is_main():
                logger.info(f"[Epoch {epoch}][{step}/{len(self.train_loader)}] loss={losses.avg:.4f} top1={top1.avg:.2f}% lr={self.optimizer.param_groups[0]['lr']:.2e}")
        return {"loss": losses.avg, "top1": top1.avg, "top5": top5.avg}

    @torch.no_grad()
    def validate(self, use_ema: bool = True) -> Dict[str, float]:
        model = self.ema.shadow if (use_ema and self.ema is not None) else self.model; model.eval()
        losses = AverageMeter(); top1 = AverageMeter(); top5 = AverageMeter()
        for images, targets in self.val_loader:
            images = images.to(self.device, non_blocking=True); targets = targets.to(self.device, non_blocking=True)
            with self.sched.autocast_ctx():
                output = model(images); loss = self.criterion(output, targets)
            acc1_val, acc5_val = accuracy(output, targets, topk=(1, 5))
            self.sched.reduce_mean(loss); self.sched.reduce_mean(acc1_val); self.sched.reduce_mean(acc5_val)
            bs = images.size(0); losses.update(loss.item(), bs); top1.update(acc1_val.item(), bs); top5.update(acc5_val.item(), bs)
        return {"loss": losses.avg, "top1": top1.avg, "top5": top5.avg}

    def fit(self) -> None:
        for epoch in range(self.start_epoch, self.cfg.epochs):
            if isinstance(self.train_loader.sampler, DistributedSampler): self.train_loader.sampler.set_epoch(epoch)
            self.train_epoch(epoch); val_metrics = self.validate(use_ema=True)
            is_best = val_metrics["top1"] > self.best_acc1
            if is_best: self.best_acc1 = val_metrics["top1"]
            if self.sched.is_main():
                state = {"epoch": epoch, "state_dict": self.raw_model.state_dict(), "ema": self.ema.shadow.state_dict() if self.ema else None,
                         "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(), "best_acc1": self.best_acc1, "cfg": self.cfg}
                self.ckpt.save(state, epoch, val_metrics["top1"], is_best=is_best)
            self.sched.barrier()

# ──────────────────────────────────────────────────────────────────────────────
#  INFERENCE & BENCHMARKING UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

class AdaptNetV5Inference:
    def __init__(self, model: AdaptNetV5, device: Optional[torch.device] = None, amp: bool = False, use_compile: bool = True) -> None:
        self.device = device or get_device()
        self.model = model.to(self.device).eval()
        self.model.switch_to_deploy()
        if use_compile and hasattr(torch, "compile") and self.device.type == "cuda":
            self.model = torch.compile(self.model, mode="max-autotune")
        self.amp = amp and (ACCELERATOR == "cuda")

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        x = x.to(self.device)
        with (autocast("cuda") if self.amp else _NullContext()): return self.model(x)

def get_gmacs(model: nn.Module, input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> float:
    try:
        from torch.utils.flop_counter import FlopCounterMode
        model.eval(); flops = FlopCounterMode(display=False)
        with flops: model(torch.randn(*input_shape, device=next(model.parameters()).device))
        return flops.get_total_flops() / 2e9
    except ImportError: return 0.0

def benchmark_v5(variant: str = "S0", batch_size: int = 1, resolution: int = 224, warmup: int = 50, runs: int = 200) -> None:
    device = get_device(); logger.info(f"Benchmarking AdaptNetV5-{variant} on {device}...")
    model = AdaptNetV5.build(variant).to(device).eval()
    model.switch_to_deploy()

    bb_params, head_params = model.param_count(detailed=True)

    # FIX: Calculate GMACs BEFORE torch.compile to avoid Dynamo hook conflicts
    gmacs = get_gmacs(model, (1, 3, resolution, resolution))

    # FIX: Compile AFTER FlopCounterMode for pure latency/throughput benchmarking
    if hasattr(torch, "compile") and device.type == "cuda":
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model, mode="max-autotune")

    dummy = torch.randn(batch_size, 3, resolution, resolution, device=device)

    with torch.no_grad():
        for _ in range(warmup): model(dummy)
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs): model(dummy)
        if device.type == "cuda": torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    lat = elapsed / runs * 1000; thr = batch_size * runs / elapsed
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0

    logger.info("="*60); logger.info(f"[Benchmark Results] AdaptNetV5-{variant}")
    logger.info(f"Backbone Params: {bb_params/1e6:.2f} M | Head Params: {head_params/1e6:.2f} M")
    logger.info(f"GMACs: {gmacs:.2f} | Latency: {lat:.2f} ms (BS={batch_size})")
    logger.info(f"Throughput: {thr:.1f} img/s | Peak VRAM: {peak_vram:.1f} MB"); logger.info("="*60)

# ──────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def adaptnet_s0(num_classes: int = 1000, **kw: Any) -> AdaptNetV5: return AdaptNetV5.build("S0", num_classes=num_classes, **kw)
def adaptnet_s1(num_classes: int = 1000, **kw: Any) -> AdaptNetV5: return AdaptNetV5.build("S1", num_classes=num_classes, **kw)
def adaptnet_s2(num_classes: int = 1000, **kw: Any) -> AdaptNetV5: return AdaptNetV5.build("S2", num_classes=num_classes, **kw)
def adaptnet_s3(num_classes: int = 1000, **kw: Any) -> AdaptNetV5: return AdaptNetV5.build("S3", num_classes=num_classes, **kw)
def adaptnet_s4(num_classes: int = 1000, **kw: Any) -> AdaptNetV5: return AdaptNetV5.build("S4", num_classes=num_classes, **kw)
def adaptnet_s5(num_classes: int = 1000, **kw: Any) -> AdaptNetV5: return AdaptNetV5.build("S5", num_classes=num_classes, **kw)
def adaptnet_s6(num_classes: int = 1000, **kw: Any) -> AdaptNetV5: return AdaptNetV5.build("S6", num_classes=num_classes, **kw)
def adaptnet_s7(num_classes: int = 1000, **kw: Any) -> AdaptNetV5: return AdaptNetV5.build("S7", num_classes=num_classes, **kw)

__all__ = [
    "AdaptNetV5", "DMSIRv5Block", "FusedDMSIRv5Block", "RepDW", "ConvBNAct", "LightSE", "DropPath",
    "ScalingConfig", "StageSpec", "TrainingConfig", "BASE_STAGE_SPECS", "Trainer", "DeviceScheduler",
    "ModelEMA", "CheckpointManager", "LabelSmoothingCrossEntropy", "SoftTargetCrossEntropy",
    "AdaptNetV5Inference", "benchmark_v5", "mixup_data", "cutmix_data", "accuracy", "get_device", "scale_stage_specs",
    "adaptnet_s0", "adaptnet_s1", "adaptnet_s2", "adaptnet_s3", "adaptnet_s4", "adaptnet_s5", "adaptnet_s6", "adaptnet_s7",
]
