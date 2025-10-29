from __future__ import annotations
import os, yaml, random, math, time, csv, socket, platform, subprocess, json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from rich.table import Table

console = Console()

def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    # simple ${var} replacement for project_root
    def expand(obj):
        if isinstance(obj, dict):
            return {k: expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [expand(v) for v in obj]
        if isinstance(obj, str):
            return obj.replace("${project_root}", cfg.get("project_root", ""))
        return obj
    return expand(cfg)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

class CSVLogger:
    def __init__(self, path: str, fieldnames: List[str]):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path)
        self.f = open(path, 'a', newline='')
        self.w = csv.DictWriter(self.f, fieldnames=fieldnames)
        if write_header:
            self.w.writeheader()

    def log(self, row: Dict[str, Any]):
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()

def pretty_config(cfg: Dict[str, Any]):
    table = Table(title="Config", show_lines=False)
    table.add_column("Key", style="bold")
    table.add_column("Value")
    def walk(prefix, d):
        for k,v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                walk(key, v)
            else:
                table.add_row(key, str(v))
    walk("", cfg)
    console.print(table)

def save_checkpoint(state: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path: str, map_location=None) -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)

class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0
    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n
    @property
    def avg(self):
        return self.sum / max(1, self.count)

def get_git_commit(root: str) -> str:
    try:
        out = subprocess.check_output(['git', '-C', root, 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"

def write_metadata(out_dir: str, cfg: Dict[str, Any]):
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if hasattr(torch.version, 'cuda') else None,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'cfg': cfg,
        'git_commit': get_git_commit(cfg.get('project_root', '.')),
    }
    with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
