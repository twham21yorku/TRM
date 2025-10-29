from __future__ import annotations
import os, argparse, csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from .utils import load_config
from .data.deepglobe_dataset import DeepGlobePatchDataset
from .models.trm_refiner import TRMSeg
from .losses import mean_iou

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--delta', type=float, default=0.1, help='risk tolerance for conformal-halting (0..1)')
    ap.add_argument('--checkpoint', default=None, help='optional model checkpoint (.pt) to load')
    ap.add_argument('--save_csv', default=None, help='optional path to save per-step q stats')
    args = ap.parse_args()
    cfg = load_config(args.cfg)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    val_ds = DeepGlobePatchDataset(os.path.join(cfg['dataset']['index_dir'],'val_patches.csv'),
                                   patch_size=cfg['dataset']['patch']['size'], augment=False)
    val_loader = DataLoader(val_ds, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=cfg['training']['num_workers'])

    model = TRMSeg(cfg['model']['in_channels'], cfg['model']['num_classes'], width=cfg['model']['width'],
                   t_max=cfg['model']['steps']['t_max'], pixel_act=cfg['model']['pixel_act']).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        if 'model' in state:
            model.load_state_dict(state['model'])
        else:
            model.load_state_dict(state)
    else:
        print("[warn] No checkpoint provided; calibrating with randomly initialized weights.")

    taus = np.linspace(0.5, 0.9, 9)
    results = []
    q_all = []  # collect sigmoid(q) across steps and samples
    with torch.no_grad():
        for tau in taus:
            steps_used = []
            miou_list = []
            for img, mask in val_loader:
                img = img.to(device); mask = mask.to(device)
                out = model(img, steps=cfg['model']['steps']['t_max'])
                # pick step based on tau
                for b in range(img.size(0)):
                    use_idx = cfg['model']['steps']['t_max']-1
                    for t in range(cfg['model']['steps']['t_min']-1, cfg['model']['steps']['t_max']):
                        q = out['q_list'][t]
                        qq = torch.sigmoid(q[b]).mean() if q.dim()==4 else torch.sigmoid(q[b])
                        q_all.append(float(qq.item()))
                        if qq.item() > tau:
                            use_idx = t
                            break
                    pred = out['logits_list'][use_idx][b:b+1].argmax(dim=1)
                    miou_list.append(mean_iou(pred, mask[b:b+1], cfg['model']['num_classes'], cfg['dataset']['classes']['ignore_index']))
                    steps_used.append(use_idx+1)
            results.append((float(np.mean(miou_list)), float(np.mean(steps_used)), float(tau)))

    # Print Pareto-like summary
    print("tau, mIoU, avg_steps")
    for m,s,tau in results:
        print(f"{tau:.2f}, {m:.4f}, {s:.2f}")

    # Conformal-halting style calibration: choose tau to satisfy coverage @ delta
    if len(q_all):
        q_arr = np.array(q_all)
        # Nonconformity r = 1 - q ; want P(r <= delta) >= 1 - alpha -> q >= 1 - delta with (1-alpha)
        # Here we approximate by tau = quantile(q, 1-delta)
        delta = min(max(args.delta, 0.0), 1.0)
        tau_conf = float(np.quantile(q_arr, 1.0 - delta))
        print(f"Recommended tau for coverage@delta={delta:.2f}: {tau_conf:.3f}")

    # Optional CSV save of per-step q stats (mean/std per step)
    if args.save_csv is not None:
        # recompute simple stats per step
        stats = []
        with torch.no_grad():
            for img, _ in val_loader:
                img = img.to(device)
                out = model(img, steps=cfg['model']['steps']['t_max'])
                for t in range(cfg['model']['steps']['t_min']-1, cfg['model']['steps']['t_max']):
                    q = out['q_list'][t]
                    qsig = torch.sigmoid(q)
                    qmean = float(qsig.mean().item())
                    qstd = float(qsig.std().item())
                    stats.append({'step': t+1, 'q_mean': qmean, 'q_std': qstd})
        # aggregate by step
        by_step = {}
        for r in stats:
            s = r['step']
            by_step.setdefault(s, {'step': s, 'q_mean': [], 'q_std': []})
            by_step[s]['q_mean'].append(r['q_mean'])
            by_step[s]['q_std'].append(r['q_std'])
        rows = []
        for s, rec in sorted(by_step.items()):
            rows.append({'step': s,
                         'q_mean': float(np.mean(rec['q_mean'])),
                         'q_std': float(np.mean(rec['q_std']))})
        os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
        with open(args.save_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['step','q_mean','q_std'])
            w.writeheader(); w.writerows(rows)
        print(f"Saved q stats: {args.save_csv}")

if __name__ == '__main__':
    main()
