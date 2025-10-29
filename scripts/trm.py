#!/usr/bin/env python3
from __future__ import annotations
"""
Unified CLI runner for TRM project tasks.

Examples:
  python scripts/trm.py prepare --cfg configs/deepglobe_default.yaml
  python scripts/trm.py train --cfg configs/deepglobe_default.yaml --seed 42
  python scripts/trm.py infer --cfg configs/deepglobe_default.yaml --input Dataset/DeepGlobe/valid --out out_masks --stride 128
  python scripts/trm.py halt-cal --cfg configs/deepglobe_default.yaml --checkpoint experiments/run1/epoch_80.pt --delta 0.1
  python scripts/trm.py ece --cfg configs/deepglobe_default.yaml --checkpoint experiments/run1/epoch_80.pt
  python scripts/trm.py efficiency --cfg configs/deepglobe_default.yaml --steps 6
  python scripts/trm.py sweep --cfg configs/deepglobe_default.yaml
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Ensure local src import precedence
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import load_config  # type: ignore


def run(cmd):
    print("[run]", " ".join(cmd))
    return subprocess.call(cmd)


def cmd_prepare(args):
    cfg = args.cfg
    root = args.root
    if root is None:
        c = load_config(cfg)
        root = c['dataset']['root']
    cmd = [sys.executable, "-m", "src.data.patchify", "--cfg", cfg, "--root", root]
    if args.force:
        cmd.append("--force")
    return run(cmd)


def cmd_train(args):
    cmd = [sys.executable, "-m", "src.train", "--cfg", args.cfg]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.out_dir is not None:
        cmd += ["--out_dir", args.out_dir]
    return run(cmd)


def cmd_infer(args):
    cmd = [sys.executable, "-m", "src.infer", "--cfg", args.cfg, "--input", args.input, "--out", args.out]
    if args.stride is not None:
        cmd += ["--stride", str(args.stride)]
    return run(cmd)


def cmd_halt_cal(args):
    cmd = [sys.executable, "-m", "src.calibrate_halt", "--cfg", args.cfg]
    if args.checkpoint:
        cmd += ["--checkpoint", args.checkpoint]
    if args.delta is not None:
        cmd += ["--delta", str(args.delta)]
    if args.save_csv:
        cmd += ["--save_csv", args.save_csv]
    return run(cmd)


def cmd_ece(args):
    cmd = [sys.executable, "-m", "src.tools.calibration", "--cfg", args.cfg, "--checkpoint", args.checkpoint]
    if args.bins is not None:
        cmd += ["--bins", str(args.bins)]
    if args.batch is not None:
        cmd += ["--batch", str(args.batch)]
    return run(cmd)


def cmd_efficiency(args):
    cmd = [sys.executable, "-m", "src.tools.efficiency", "--cfg", args.cfg]
    if args.steps is not None:
        cmd += ["--steps", str(args.steps)]
    if args.batch is not None:
        cmd += ["--batch", str(args.batch)]
    if args.H is not None:
        cmd += ["--H", str(args.H)]
    if args.W is not None:
        cmd += ["--W", str(args.W)]
    if args.cpu:
        cmd += ["--cpu"]
    return run(cmd)


def cmd_sweep(args):
    script = REPO_ROOT / "scripts" / "run_sweep.sh"
    return run(["bash", str(script), args.cfg])


def main():
    ap = argparse.ArgumentParser(description="TRM unified CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # prepare
    ap_p = sub.add_parser("prepare", help="build patch index CSVs")
    ap_p.add_argument("--cfg", required=True)
    ap_p.add_argument("--root", default=None)
    ap_p.add_argument("--force", action="store_true")
    ap_p.set_defaults(func=cmd_prepare)

    # train
    ap_t = sub.add_parser("train", help="train model")
    ap_t.add_argument("--cfg", required=True)
    ap_t.add_argument("--seed", type=int, default=None)
    ap_t.add_argument("--out_dir", default=None)
    ap_t.set_defaults(func=cmd_train)

    # infer
    ap_i = sub.add_parser("infer", help="run inference on full images")
    ap_i.add_argument("--cfg", required=True)
    ap_i.add_argument("--input", required=True)
    ap_i.add_argument("--out", required=True)
    ap_i.add_argument("--stride", type=int, default=None)
    ap_i.set_defaults(func=cmd_infer)

    # halt calibration
    ap_h = sub.add_parser("halt-cal", help="halting calibration and tau suggestion")
    ap_h.add_argument("--cfg", required=True)
    ap_h.add_argument("--checkpoint", default=None)
    ap_h.add_argument("--delta", type=float, default=0.1)
    ap_h.add_argument("--save_csv", default=None)
    ap_h.set_defaults(func=cmd_halt_cal)

    # ece calibration
    ap_c = sub.add_parser("ece", help="ECE and temperature calibration")
    ap_c.add_argument("--cfg", required=True)
    ap_c.add_argument("--checkpoint", required=True)
    ap_c.add_argument("--bins", type=int, default=15)
    ap_c.add_argument("--batch", type=int, default=4)
    ap_c.set_defaults(func=cmd_ece)

    # efficiency
    ap_e = sub.add_parser("efficiency", help="FLOPs and latency")
    ap_e.add_argument("--cfg", required=True)
    ap_e.add_argument("--steps", type=int, default=None)
    ap_e.add_argument("--batch", type=int, default=None)
    ap_e.add_argument("--H", type=int, default=None)
    ap_e.add_argument("--W", type=int, default=None)
    ap_e.add_argument("--cpu", action="store_true")
    ap_e.set_defaults(func=cmd_efficiency)

    # sweep
    ap_s = sub.add_parser("sweep", help="run 5 seeds")
    ap_s.add_argument("--cfg", required=True)
    ap_s.set_defaults(func=cmd_sweep)

    args = ap.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
