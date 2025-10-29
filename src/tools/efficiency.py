from __future__ import annotations
"""
Efficiency utilities: FLOPs/params via THOP and latency via CUDA events.

Notes
- THOP has limitations (may not count some ops; bias terms often ignored; custom modules
  may be treated as zero). Treat numbers as approximate and compare relatively.
"""

import argparse
import time
import numpy as np
import torch
from ..utils import load_config
from ..models.trm_refiner import TRMSeg

def try_thop_profile(model, x, steps: int):
    try:
        from thop import profile
    except Exception:
        return None, None

    # Wrap forward to enforce steps
    class Wrapper(torch.nn.Module):
        def __init__(self, m, s):
            super().__init__()
            self.m = m
            self.s = s
        def forward(self, x):
            out = self.m(x, steps=self.s)
            # return a single tensor to satisfy thop
            return out['logits_list'][-1]

    w = Wrapper(model, 1)
    macs, params = profile(w, inputs=(x,), verbose=False)
    # Per-step MACs -> multiply by steps for total
    return macs * steps, params

def measure_latency(model, x, steps: int, warmup: int = 10, repeat: int = 50):
    device = x.device
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(x, steps=steps)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times = []
        for _ in range(repeat):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x, steps=steps)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1-t0)*1000.0)
    return float(np.mean(times)), float(np.std(times))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--H', type=int, default=None, help='input height; defaults to patch size')
    ap.add_argument('--W', type=int, default=None, help='input width; defaults to patch size')
    ap.add_argument('--steps', type=int, default=None, help='override steps (defaults to t_max)')
    ap.add_argument('--cpu', action='store_true', help='force CPU for latency')
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--repeat', type=int, default=50)
    args = ap.parse_args()

    cfg = load_config(args.cfg)
    device = 'cpu' if args.cpu or (not torch.cuda.is_available()) else 'cuda'
    steps = args.steps if args.steps is not None else cfg['model']['steps']['t_max']

    patch = cfg['dataset']['patch']['size']
    H = args.H or patch
    W = args.W or patch

    model = TRMSeg(
        in_ch=cfg['model']['in_channels'],
        num_classes=cfg['model']['num_classes'],
        width=cfg['model']['width'],
        t_max=cfg['model']['steps']['t_max'],
        pixel_act=cfg['model']['pixel_act'],
    ).to(device)

    x = torch.randn(args.batch, cfg['model']['in_channels'], H, W, device=device)

    # Params
    params_m = sum(p.numel() for p in model.parameters())/1e6

    # FLOPs (MACs) via THOP
    macs, params_thop = try_thop_profile(model, x, steps)
    macs_g = (macs/1e9) if macs is not None else None

    # Latency
    mean_ms, std_ms = measure_latency(model, x, steps, args.warmup, args.repeat)

    print("Efficiency summary")
    print(f"Device: {device}")
    print(f"Input: B{args.batch}x{cfg['model']['in_channels']}x{H}x{W}")
    print(f"Steps: {steps}")
    print(f"Params (M): {params_m:.3f}")
    if macs_g is not None:
        print(f"Approx FLOPs (G MACs): {macs_g:.3f} (THOP, approx)")
    else:
        print("THOP not available; skipping FLOPs.")
    print(f"Latency/patch (ms): {mean_ms:.2f} ± {std_ms:.2f} (warmup={args.warmup}, n={args.repeat})")

if __name__ == '__main__':
    main()

