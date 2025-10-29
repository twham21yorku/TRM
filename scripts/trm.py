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
from datetime import datetime
import errno

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Ensure local src import precedence
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from src.utils import load_config  # type: ignore


console = Console()


def _default_cfg_path() -> str:
    cand = REPO_ROOT / "configs" / "deepglobe_default.yaml"
    return str(cand)

def _load_cfg(path: str | None):
    cfg_path = path or _default_cfg_path()
    return cfg_path, load_config(cfg_path)


def _fmt_cmd(cmd):
    return " ".join(cmd)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def run(cmd, desc: str | None = None, log_path: str | None = None):
    label = desc or _fmt_cmd(cmd)
    console.print(Panel.fit(Text(f"$ {_fmt_cmd(cmd)}", style="bold cyan"), title=desc, border_style="cyan"))

    # Open log file (binary to capture control codes)
    fh = None
    if log_path:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = log_file.open('wb')
        fh.write((f"$ {_fmt_cmd(cmd)}\n\n").encode('utf-8'))
        fh.flush()

    rc = 0
    # Prefer PTY on POSIX to preserve TTY semantics for rich/tqdm
    use_pty = (os.name == 'posix')
    if use_pty:
        import pty, select, os as _os, shutil, fcntl, termios, struct
        master, slave = pty.openpty()
        try:
            # Propagate current terminal window size to the PTY slave
            try:
                ts = shutil.get_terminal_size(fallback=(120, 32))
                winsize = struct.pack('HHHH', ts.lines, ts.columns, 0, 0)
                fcntl.ioctl(slave, termios.TIOCSWINSZ, winsize)
            except Exception:
                pass

            proc = subprocess.Popen(cmd, stdin=slave, stdout=slave, stderr=slave, close_fds=True)
            _os.close(slave)
            try:
                while True:
                    r, _, _ = select.select([master], [], [], 0.05)
                    if master in r:
                        try:
                            data = _os.read(master, 4096)
                        except OSError as e:
                            # EIO commonly signals EOF on PTY master when child exits
                            if e.errno in (errno.EIO, 5):
                                break
                            else:
                                raise
                        if not data:
                            if proc.poll() is not None:
                                break
                            continue
                        if fh:
                            fh.write(data)
                        try:
                            _os.write(sys.stdout.fileno(), data)
                        except Exception:
                            sys.stdout.buffer.write(data)
                            sys.stdout.flush()
                    if proc.poll() is not None:
                        # drain remaining
                        while True:
                            try:
                                data = _os.read(master, 4096)
                            except OSError:
                                data = b''
                            if not data:
                                break
                            if fh:
                                fh.write(data)
                            try:
                                _os.write(sys.stdout.fileno(), data)
                            except Exception:
                                sys.stdout.buffer.write(data)
                                sys.stdout.flush()
                        break
                rc = proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                rc = proc.wait()
                if fh:
                    fh.write(b"\nInterrupted by user\n")
                console.print(Text("⚠ Interrupted by user", style="bold yellow"))
        finally:
            try:
                _os.close(master)
            except Exception:
                pass
            if fh:
                fh.flush()
                fh.close()
    else:
        # Fallback: pipe mode (may degrade live progress rendering)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=False, bufsize=0)
        try:
            assert proc.stdout is not None
            while True:
                data = proc.stdout.read(4096)
                if not data:
                    break
                if fh:
                    fh.write(data)
                sys.stdout.buffer.write(data)
                sys.stdout.flush()
            rc = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            rc = proc.wait()
            if fh:
                fh.write(b"\nInterrupted by user\n")
            console.print(Text("⚠ Interrupted by user", style="bold yellow"))
        finally:
            if fh:
                fh.flush()
                fh.close()

    if rc == 0:
        console.print(Text(f"✔ {label}", style="bold green"))
    else:
        console.print(Text(f"✖ {label} (exit {rc})", style="bold red"))
    return rc


def cmd_prepare(args):
    cfg_path, c = _load_cfg(args.cfg)
    root = args.root
    if root is None:
        root = c['dataset']['root']
    cmd = [sys.executable, "-m", "src.data.patchify", "--cfg", cfg_path, "--root", root]
    if args.force:
        cmd.append("--force")
    log_dir = Path(args.log_dir) if getattr(args, 'log_dir', None) else Path("logs") / f"prepare_{_timestamp()}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "prepare.log"
    return run(cmd, desc="Patch index", log_path=str(log_path))


def cmd_train(args):
    cfg_path, c = _load_cfg(args.cfg)
    cmd = [sys.executable, "-m", "src.train", "--cfg", cfg_path]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    run_dir = Path(args.out_dir) if getattr(args, 'out_dir', None) else Path("experiments") / f"run_{_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd += ["--out_dir", str(run_dir)]
    logs_dir = Path(args.log_dir) if getattr(args, 'log_dir', None) else run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "train.log"
    return run(cmd, desc="Training", log_path=str(log_path))


def cmd_infer(args):
    cfg_path, c = _load_cfg(args.cfg)
    inp = args.input
    if inp is None:
        root = c['dataset']['root']
        for sp in [c['dataset'].get('val_split','val'), 'valid', 'val', 'train']:
            cand = Path(root) / sp
            if cand.exists():
                inp = str(cand)
                break
    if inp is None:
        raise SystemExit("Could not determine inference input folder; specify --input explicitly")
    if getattr(args, 'out', None) is None:
        base_dir = Path("experiments") / f"infer_{_timestamp()}"
        out_path = base_dir / "pred_masks"
    else:
        out_path = Path(args.out)
        base_dir = out_path.parent
    out_path.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "src.infer", "--cfg", cfg_path, "--input", inp, "--out", str(out_path)]
    if args.stride is not None:
        cmd += ["--stride", str(args.stride)]
    logs_dir = Path(args.log_dir) if getattr(args, 'log_dir', None) else base_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "infer.log"
    return run(cmd, desc="Inference", log_path=str(log_path))


def _find_latest_ckpt(c) -> str | None:
    out_dir = Path(c['logging']['out_dir'])
    if not out_dir.exists():
        return None
    ckpts = sorted(out_dir.glob('epoch_*.pt'))
    return str(ckpts[-1]) if ckpts else None

def cmd_halt_cal(args):
    cfg_path, c = _load_cfg(args.cfg)
    cmd = [sys.executable, "-m", "src.calibrate_halt", "--cfg", cfg_path]
    ckpt = args.checkpoint or _find_latest_ckpt(c)
    if ckpt:
        cmd += ["--checkpoint", ckpt]
    if args.delta is not None:
        cmd += ["--delta", str(args.delta)]
    if args.save_csv:
        cmd += ["--save_csv", args.save_csv]
    log_dir = Path(args.log_dir) if getattr(args, 'log_dir', None) else Path("logs") / f"halt_{_timestamp()}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "halt_cal.log"
    return run(cmd, desc="Halting calibration", log_path=str(log_path))


def cmd_ece(args):
    cfg_path, c = _load_cfg(args.cfg)
    ckpt = args.checkpoint or _find_latest_ckpt(c)
    if ckpt is None:
        print("[error] checkpoint not found; specify --checkpoint or train first")
        return 1
    cmd = [sys.executable, "-m", "src.tools.calibration", "--cfg", cfg_path, "--checkpoint", ckpt]
    if args.bins is not None:
        cmd += ["--bins", str(args.bins)]
    if args.batch is not None:
        cmd += ["--batch", str(args.batch)]
    log_dir = Path(args.log_dir) if getattr(args, 'log_dir', None) else Path("logs") / f"ece_{_timestamp()}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "ece.log"
    return run(cmd, desc="ECE calibration", log_path=str(log_path))


def cmd_efficiency(args):
    cfg_path, c = _load_cfg(args.cfg)
    cmd = [sys.executable, "-m", "src.tools.efficiency", "--cfg", cfg_path]
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
    log_dir = Path(args.log_dir) if getattr(args, 'log_dir', None) else Path("logs") / f"efficiency_{_timestamp()}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "efficiency.log"
    return run(cmd, desc="Efficiency metrics", log_path=str(log_path))


def cmd_sweep(args):
    script = REPO_ROOT / "scripts" / "run_sweep.sh"
    cfg_path, _ = _load_cfg(args.cfg)
    log_dir = Path(args.log_dir) if getattr(args, 'log_dir', None) else Path("logs") / f"sweep_{_timestamp()}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "sweep.log"
    return run(["bash", str(script), cfg_path], desc="Seed sweep", log_path=str(log_path))


def cmd_quick(args):
    """End-to-end: prepare -> train -> infer with sensible defaults."""
    cfg_path, cfg = _load_cfg(args.cfg)
    run_dir = Path(args.out_dir) if getattr(args, 'out_dir', None) else Path("experiments") / f"run_{_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(args.log_dir) if getattr(args, 'log_dir', None) else run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pred_out = Path(args.pred_out) if getattr(args, 'pred_out', None) else run_dir / "pred_masks"
    pred_out.mkdir(parents=True, exist_ok=True)

    root = args.root or cfg['dataset']['root']

    console.rule("[bold magenta]TRM Quick Pipeline")
    console.print("[bold]Step 1/3: Patch indexing[/]")
    rc = cmd_prepare(argparse.Namespace(cfg=cfg_path, root=root, force=args.force, log_dir=str(logs_dir)))
    if rc != 0:
        console.rule("[bold red]Pipeline failed")
        return rc

    console.print("[bold]Step 2/3: Training[/]")
    rc = cmd_train(argparse.Namespace(cfg=cfg_path, seed=args.seed, out_dir=str(run_dir), log_dir=str(logs_dir)))
    if rc != 0:
        console.rule("[bold red]Pipeline failed")
        return rc

    console.print("[bold]Step 3/3: Inference[/]")
    rc = cmd_infer(argparse.Namespace(cfg=cfg_path, input=args.input, out=str(pred_out), stride=args.stride, log_dir=str(logs_dir)))
    if rc == 0:
        console.print(Text(f"Logs saved to {logs_dir}", style="cyan"))
        console.rule("[bold green]Pipeline complete")
    else:
        console.rule("[bold red]Pipeline failed")
    return rc


def main():
    ap = argparse.ArgumentParser(description="TRM unified CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # prepare
    ap_p = sub.add_parser("prepare", help="build patch index CSVs")
    ap_p.add_argument("--cfg", default=None)
    ap_p.add_argument("--root", default=None)
    ap_p.add_argument("--force", action="store_true")
    ap_p.add_argument("--log_dir", default=None)
    ap_p.set_defaults(func=cmd_prepare)

    # train
    ap_t = sub.add_parser("train", help="train model")
    ap_t.add_argument("--cfg", default=None)
    ap_t.add_argument("--seed", type=int, default=None)
    ap_t.add_argument("--out_dir", default=None)
    ap_t.add_argument("--log_dir", default=None)
    ap_t.set_defaults(func=cmd_train)

    # infer
    ap_i = sub.add_parser("infer", help="run inference on full images")
    ap_i.add_argument("--cfg", default=None)
    ap_i.add_argument("--input", default=None)
    ap_i.add_argument("--out", default=None)
    ap_i.add_argument("--stride", type=int, default=None)
    ap_i.add_argument("--log_dir", default=None)
    ap_i.set_defaults(func=cmd_infer)

    # halt calibration
    ap_h = sub.add_parser("halt-cal", help="halting calibration and tau suggestion")
    ap_h.add_argument("--cfg", default=None)
    ap_h.add_argument("--checkpoint", default=None)
    ap_h.add_argument("--delta", type=float, default=0.1)
    ap_h.add_argument("--save_csv", default=None)
    ap_h.add_argument("--log_dir", default=None)
    ap_h.set_defaults(func=cmd_halt_cal)

    # ece calibration
    ap_c = sub.add_parser("ece", help="ECE and temperature calibration")
    ap_c.add_argument("--cfg", default=None)
    ap_c.add_argument("--checkpoint", default=None)
    ap_c.add_argument("--bins", type=int, default=15)
    ap_c.add_argument("--batch", type=int, default=4)
    ap_c.add_argument("--log_dir", default=None)
    ap_c.set_defaults(func=cmd_ece)

    # efficiency
    ap_e = sub.add_parser("efficiency", help="FLOPs and latency")
    ap_e.add_argument("--cfg", default=None)
    ap_e.add_argument("--steps", type=int, default=None)
    ap_e.add_argument("--batch", type=int, default=None)
    ap_e.add_argument("--H", type=int, default=None)
    ap_e.add_argument("--W", type=int, default=None)
    ap_e.add_argument("--cpu", action="store_true")
    ap_e.add_argument("--log_dir", default=None)
    ap_e.set_defaults(func=cmd_efficiency)

    # sweep
    ap_s = sub.add_parser("sweep", help="run 5 seeds")
    ap_s.add_argument("--cfg", default=None)
    ap_s.add_argument("--log_dir", default=None)
    ap_s.set_defaults(func=cmd_sweep)

    # quick pipeline
    ap_q = sub.add_parser("quick", help="prepare -> train -> infer with defaults")
    ap_q.add_argument("--cfg", default=None)
    ap_q.add_argument("--root", default=None)
    ap_q.add_argument("--force", action="store_true")
    ap_q.add_argument("--seed", type=int, default=None)
    ap_q.add_argument("--out_dir", default=None)
    ap_q.add_argument("--input", default=None)
    ap_q.add_argument("--pred_out", default=None)
    ap_q.add_argument("--stride", type=int, default=None)
    ap_q.add_argument("--log_dir", default=None)
    ap_q.set_defaults(func=cmd_quick)

    args = ap.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
