from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple

def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255):
    mask = (target != ignore_index)
    pred = pred[mask]
    target = target[mask]
    k = (target>=0) & (target<num_classes)
    inds = num_classes * target[k].to(torch.int64) + pred[k]
    mat = torch.bincount(inds, minlength=num_classes**2).reshape(num_classes, num_classes).to(torch.float64)
    return mat

def iou_from_confmat(cm: torch.Tensor):
    # cm: [C,C], rows=true, cols=pred
    tp = torch.diag(cm)
    fn = cm.sum(1) - tp
    fp = cm.sum(0) - tp
    denom = tp + fp + fn
    iou = tp / torch.clamp(denom, min=1e-9)
    return iou

def boundary_f1(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = 255) -> float:
    """Simple boundary F1 via morphological gradient on CPU.

    Supports inputs of shape [H, W] or [B, H, W]. Returns the mean BF-score over the batch.
    """

    def grad(x: np.ndarray) -> np.ndarray:
        from numpy.lib.stride_tricks import as_strided
        H, W = x.shape
        pad = np.pad(x, 1, mode='edge')
        windows = as_strided(pad, shape=(H, W, 3, 3), strides=pad.strides * 2)
        return (windows.max(axis=(2, 3)) != windows.min(axis=(2, 3))).astype(np.uint8)

    pred_np = pred.detach().cpu().numpy()
    tgt_np = target.detach().cpu().numpy()

    # Ensure batch dimension
    if pred_np.ndim == 2:
        pred_np = pred_np[None, ...]
        tgt_np = tgt_np[None, ...]

    scores = []
    for p, t in zip(pred_np, tgt_np):
        p_edge = grad(p)
        t_edge = grad(t)
        ign = (t == ignore_index)
        p_edge[ign] = 0
        t_edge[ign] = 0

        tp = np.logical_and(p_edge, t_edge).sum()
        fp = np.logical_and(p_edge, np.logical_not(t_edge)).sum()
        fn = np.logical_and(np.logical_not(p_edge), t_edge).sum()

        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        if prec + rec == 0:
            scores.append(0.0)
        else:
            scores.append(2 * prec * rec / (prec + rec))

    if not scores:
        return 0.0
    return float(np.mean(scores))
