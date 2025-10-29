from __future__ import annotations
from typing import List, Dict
import torch
from torch import nn
import torch.nn.functional as F

def cross_entropy_with_ignore(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 255):
    return F.cross_entropy(logits, target.long(), ignore_index=ignore_index)

def iou_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255):
    # pred: [B,H,W] long ; target: [B,H,W] long
    ious = []
    for c in range(num_classes):
        pred_c = (pred==c)
        tgt_c  = (target==c)
        # remove ignore pixels
        mask = (target!=ignore_index)
        inter = (pred_c & tgt_c & mask).sum().item()
        union = ((pred_c | tgt_c) & mask).sum().item()
        if union==0:
            ious.append(float('nan'))
        else:
            ious.append(inter/union)
    return ious

def mean_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255):
    vals = [v for v in iou_per_class(pred, target, num_classes, ignore_index) if v==v]
    if not vals: return 0.0
    return sum(vals)/len(vals)

def compute_stop_labels(logits_list: List[torch.Tensor], target: torch.Tensor, num_classes: int, eps: float = 0.002, ignore_index: int = 255):
    """Generate stop labels per step based on mIoU improvement."""
    prev_iou = 0.0
    labels = []
    with torch.no_grad():
        for logits in logits_list:
            pred = logits.argmax(dim=1)
            miou = mean_iou(pred, target, num_classes, ignore_index)
            labels.append(1.0 if (miou - prev_iou) < eps else 0.0)
            prev_iou = miou
    return torch.tensor(labels, device=logits_list[0].device, dtype=torch.float32)  # [T]

class TRMLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = 255, alpha: float = 0.3, beta: float = 0.5, gamma: float = 1e-3):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, logits_list: List[torch.Tensor], q_list: List[torch.Tensor], target: torch.Tensor, expected_steps: float = None):
        # CE terms
        ce_final = cross_entropy_with_ignore(logits_list[-1], target, self.ignore_index)
        ce_mid = 0.0
        if len(logits_list)>1:
            for lg in logits_list[:-1]:
                ce_mid = ce_mid + cross_entropy_with_ignore(lg, target, self.ignore_index)
            ce_mid = ce_mid / (len(logits_list)-1)

        # Halting labels
        stop_labels = compute_stop_labels(logits_list, target, self.num_classes, ignore_index=self.ignore_index)  # [T]
        q_losses = 0.0
        for t, q in enumerate(q_list):
            if q.dim()==2:      # [B,1]
                tgt = stop_labels[t] * torch.ones_like(q)
            else:               # [B,1,H,W] pixel-wise (optional)
                tgt = stop_labels[t] * torch.ones_like(q)
            q_losses = q_losses + self.bce(q, tgt)

        # Ponder (expected steps) penalty
        ponder = 0.0
        if expected_steps is not None:
            ponder = expected_steps

        return ce_final + self.alpha*ce_mid + self.beta*q_losses + self.gamma*ponder
