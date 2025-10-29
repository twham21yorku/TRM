from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import torch
from torch import nn
import torch.nn.functional as F
from .blocks import RefineCore

class Encoder(nn.Module):
    def __init__(self, in_ch: int = 3, width: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
        )
    def forward(self, x):
        return self.stem(x)  # [B, C, H, W]

class TRMSeg(nn.Module):
    """TRM-style refiner for segmentation.
    - Shared small core run multiple steps.
    - Delta logits are accumulated (additive refinement).
    - q-head estimates halting probability.
    """
    def __init__(self, in_ch: int, num_classes: int, width: int = 64, t_max: int = 6, pixel_act: bool = False):
        super().__init__()
        self.enc = Encoder(in_ch, width)
        self.core = RefineCore(width + num_classes, width)  # concat features + prev prob
        self.delta_head = nn.Conv2d(width, num_classes, 1)
        self.state_proj = nn.Conv2d(width, width, 1)
        self.t_max = t_max
        self.pixel_act = pixel_act

        if pixel_act:
            self.q_head = nn.Sequential(nn.Conv2d(width, 1, 1))  # [B,1,H,W]
        else:
            self.q_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(width, 64),
                nn.SiLU(inplace=True),
                nn.Linear(64, 1),
            )

    def forward(self, x: torch.Tensor, steps: int = None) -> Dict[str, List[torch.Tensor]]:
        # x: [B,C,H,W]
        B = x.shape[0]
        steps = steps or self.t_max
        feats = self.enc(x)                       # [B,W,H,W]
        logits = torch.zeros(B, self.delta_head.out_channels, x.size(2), x.size(3), device=x.device)
        logits_list: List[torch.Tensor] = []
        q_list: List[torch.Tensor] = []

        for t in range(steps):
            probs = logits.softmax(dim=1)
            z = torch.cat([feats, probs], dim=1)
            z = self.core(z)                      # [B,W,H,W]
            delta = self.delta_head(z)            # [B,K,H,W]
            logits = logits + delta
            logits_list.append(logits)
            if self.pixel_act:
                q = self.q_head(z)                # [B,1,H,W]
            else:
                q = self.q_head(z)                # [B,1]
            q_list.append(q)

            feats = self.state_proj(z)            # simple state update

        return { 'logits_list': logits_list, 'q_list': q_list }
