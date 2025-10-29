from __future__ import annotations
import os, argparse, glob, csv, random, json
from typing import List, Tuple, Optional
from PIL import Image
import numpy as np
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

def load_cfg(path):
    with open(path,'r') as f:
        return yaml.safe_load(f)

def find_pairs(root: str, train_split: str = 'train', val_split: str = 'val') -> List[Tuple[str,str]]:
    """Find (image, mask) pairs under root.
    Expects files named <id>_sat.jpg and <id>_mask.png in split subfolders
    (train/val or train/valid) or a single folder.
    """
    pairs: List[Tuple[str, str, str]] = []
    for split in [train_split, val_split]:
        if not split:
            continue
        split_dir = os.path.join(root, split)
        if os.path.isdir(split_dir):
            imgs = glob.glob(os.path.join(split_dir, '*_sat.*'))
            for ip in imgs:
                mid = os.path.basename(ip).split('_sat')[0]
                mp = os.path.join(split_dir, f"{mid}_mask.png")
                if os.path.exists(mp):
                    pairs.append((split, ip, mp))
    if not pairs:
        # single folder fallback
        imgs = glob.glob(os.path.join(root, '*_sat.*'))
        for ip in imgs:
            mid = os.path.basename(ip).split('_sat')[0]
            mp = os.path.join(root, f"{mid}_mask.png")
            if os.path.exists(mp):
                # split later
                pairs.append(('train', ip, mp))
    return pairs

def build_index(csv_path: str, pairs: List[Tuple[str,str,str]], patch: int, stride: int,
                filter_unknown_ratio: float, progress: Optional[Progress] = None,
                task_desc: str = "build"):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    total_tiles = 0
    meta = []
    for split, ip, mp in pairs:
        img = Image.open(ip)
        W, H = img.size
        tiles_x = max(0, ((W - patch) // stride) + 1) if W >= patch else 0
        tiles_y = max(0, ((H - patch) // stride) + 1) if H >= patch else 0
        tiles = tiles_x * tiles_y
        total_tiles += tiles
        meta.append((split, ip, mp, W, H))

    task_id = None
    if progress is not None:
        task_id = progress.add_task(task_desc, total=total_tiles or 1)

    written = 0
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image','mask','x','y','split'])
        w.writeheader()
        for split, ip, mp, W, H in meta:
            img = Image.open(ip).convert('RGB')
            m  = Image.open(mp).convert('RGB')
            for y in range(0, H - patch + 1, stride):
                for x in range(0, W - patch + 1, stride):
                    crop = m.crop((x,y,x+patch,y+patch))
                    crop_np = np.asarray(crop)
                    unknown = np.all(crop_np==0, axis=2).mean()
                    if unknown <= filter_unknown_ratio:
                        w.writerow({'image': ip, 'mask': mp, 'x': x, 'y': y, 'split': split})
                    written += 1
                    if progress is not None and task_id is not None:
                        progress.advance(task_id)
    return written

def _make_meta(cfg, root: str, train_split: str, val_split: str):
    ds = cfg['dataset']
    patch = ds['patch']
    return {
        'dataset_root': os.path.abspath(root),
        'train_split': train_split,
        'val_split': val_split,
        'patch_size': patch['size'],
        'stride': patch['stride'],
        'filter_unknown_ratio': patch['filter_unknown_ratio'],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True, help='path to yaml config')
    ap.add_argument('--root', required=True, help='dataset root containing train/val or images')
    ap.add_argument('--force', action='store_true', help='rebuild indices even if up-to-date')
    args = ap.parse_args()
    cfg = load_cfg(args.cfg)
    patch = cfg['dataset']['patch']['size']
    stride = cfg['dataset']['patch']['stride']
    flt = cfg['dataset']['patch']['filter_unknown_ratio']
    train_split = cfg['dataset'].get('train_split', 'train')
    val_split = cfg['dataset'].get('val_split', 'val')
    index_dir = cfg['dataset']['index_dir'].replace('${project_root}', cfg.get('project_root',''))
    os.makedirs(index_dir, exist_ok=True)

    # Skip if up-to-date
    meta_path = os.path.join(index_dir, 'index_meta.json')
    train_csv = os.path.join(index_dir,'train_patches.csv')
    val_csv   = os.path.join(index_dir,'val_patches.csv')
    curr_meta = _make_meta(cfg, args.root, train_split, val_split)
    if (not args.force) and os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(meta_path):
        try:
            with open(meta_path,'r') as f:
                prev = json.load(f)
            if all(prev.get(k)==curr_meta.get(k) for k in curr_meta.keys()):
                # quick non-empty check
                if os.path.getsize(train_csv) > 64 and os.path.getsize(val_csv) > 64:
                    print(f"Index up-to-date at {index_dir}; use --force to rebuild. Skipping.")
                    return
        except Exception:
            pass

    pairs = find_pairs(args.root, train_split=train_split, val_split=val_split)
    if not pairs:
        raise SystemExit(f"No image/mask pairs found under {args.root}")

    # if only train exists, split into train/val
    splits = set(s for s,_,_ in pairs)
    if splits == {train_split}:
        random.seed(1337)
        random.shuffle(pairs)
        n = len(pairs)
        val_n = max(1, int(0.15*n))
        # keep original split names from config
        pairs = [(val_split if i < val_n else train_split, ip, mp) for i,(_,ip,mp) in enumerate(pairs)]

    train_pairs = [p for p in pairs if p[0]==train_split]
    val_pairs   = [p for p in pairs if p[0]==val_split]

    console = Console(force_terminal=True)
    console_progress = Progress(SpinnerColumn(), "{task.description}", BarColumn(),
                                 TimeElapsedColumn(), TimeRemainingColumn(), console=console)
    with console_progress:
        build_index(train_csv, train_pairs, patch, stride, flt,
                    progress=console_progress, task_desc="train index")
        build_index(val_csv,   val_pairs,   patch, stride, flt,
                    progress=console_progress, task_desc="val index")
    print(f"Wrote CSVs to {index_dir}")

    # write meta
    counts = {}
    try:
        def _count_rows(p):
            with open(p,'r') as f:
                return max(0, sum(1 for _ in f) - 1)
        counts = {
            'train_rows': _count_rows(train_csv),
            'val_rows': _count_rows(val_csv),
        }
    except Exception:
        pass
    meta_out = curr_meta.copy(); meta_out.update(counts)
    with open(meta_path,'w') as f:
        json.dump(meta_out, f, indent=2)

if __name__ == '__main__':
    main()
