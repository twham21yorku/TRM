from __future__ import annotations
"""
Calibration utilities: pixel-wise ECE and temperature scaling for segmentation.
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ..utils import load_config
from ..data.deepglobe_dataset import DeepGlobePatchDataset
from ..models.trm_refiner import TRMSeg

def ece_segmentation(logits: torch.Tensor, target: torch.Tensor, ignore_index: int, bins: int = 15, T: float = 1.0) -> float:
    """Compute pixel-wise ECE for segmentation logits.
    - logits: [B,C,H,W]
    - target: [B,H,W]
    - ignores 'ignore_index'
    """
    with torch.no_grad():
        if T != 1.0:
            logits = logits / T
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)  # [B,H,W]
        mask = (target != ignore_index)
        conf = conf[mask]
        pred = pred[mask]
        true = target[mask]
        if conf.numel() == 0:
            return 0.0
        # binning
        ece = torch.zeros(1, device=logits.device)
        bin_boundaries = torch.linspace(0, 1, bins+1, device=logits.device)
        for i in range(bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i+1]
            sel = (conf > lo) & (conf <= hi)
            if sel.any():
                conf_bin = conf[sel]
                acc_bin = (pred[sel] == true[sel]).float().mean()
                ece += (sel.float().mean()) * torch.abs(conf_bin.mean() - acc_bin)
        return float(ece.item())

def fit_temperature(model: TRMSeg, loader: DataLoader, ignore_index: int, steps: int) -> float:
    """Fit scalar temperature T by minimizing NLL on validation data.
    Returns T (float).
    """
    device = next(model.parameters()).device
    logT = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=50)

    def _closure():
        opt.zero_grad()
        nll = 0.0
        n = 0
        with torch.no_grad():
            pass
        for img, mask in loader:
            img = img.to(device)
            mask = mask.to(device)
            out = model(img, steps=steps)
            logits = out['logits_list'][-1]  # use final step for calibration
            T = torch.exp(logT)
            nll_batch = F.cross_entropy(logits / T, mask.long(), ignore_index=ignore_index, reduction='mean')
            nll = nll + nll_batch
            n += 1
        loss = nll / max(1, n)
        loss.backward()
        return loss

    opt.step(_closure)
    T = float(torch.exp(logT).item())
    return max(1e-2, min(10.0, T))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--bins', type=int, default=15)
    ap.add_argument('--batch', type=int, default=4)
    args = ap.parse_args()

    cfg = load_config(args.cfg)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ignore_index = cfg['dataset']['classes']['ignore_index']
    steps = cfg['model']['steps']['t_max']

    val_csv = cfg['dataset']['index_dir'].replace('${project_root}', cfg.get('project_root','')) + '/val_patches.csv'
    val_ds = DeepGlobePatchDataset(val_csv, patch_size=cfg['dataset']['patch']['size'], augment=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=cfg['training']['num_workers'])

    model = TRMSeg(cfg['model']['in_channels'], cfg['model']['num_classes'], width=cfg['model']['width'],
                   t_max=steps, pixel_act=cfg['model']['pixel_act']).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state['model'])
    model.eval()

    # Measure ECE before
    with torch.no_grad():
        eces = []
        for img, mask in val_loader:
            img = img.to(device); mask = mask.to(device)
            out = model(img, steps=steps)
            logits = out['logits_list'][-1]
            eces.append(ece_segmentation(logits, mask, ignore_index, bins=args.bins, T=1.0))
        ece_before = float(np.mean(eces)) if len(eces:=eces) else 0.0

    # Fit temperature
    T = fit_temperature(model, val_loader, ignore_index, steps)

    # ECE after
    with torch.no_grad():
        eces = []
        for img, mask in val_loader:
            img = img.to(device); mask = mask.to(device)
            out = model(img, steps=steps)
            logits = out['logits_list'][-1]
            eces.append(ece_segmentation(logits, mask, ignore_index, bins=args.bins, T=T))
        ece_after = float(np.mean(eces)) if len(eces:=eces) else 0.0

    print(f"ECE before: {ece_before:.4f}")
    print(f"Fitted temperature T: {T:.3f}")
    print(f"ECE after: {ece_after:.4f}")

    # Save temperature next to checkpoint
    import os
    out_path = os.path.join(os.path.dirname(args.checkpoint), 'temperature.json')
    with open(out_path, 'w') as f:
        json.dump({'T': T, 'ece_before': ece_before, 'ece_after': ece_after}, f, indent=2)
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    import numpy as np
    from torch.utils.data import DataLoader
    main()
