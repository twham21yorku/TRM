from __future__ import annotations
import os, argparse, glob
from typing import Tuple
import numpy as np
from PIL import Image
import torch
from .utils import load_config
from .models.trm_refiner import TRMSeg
from .data.label_codec import id_to_color_mask

def tile_image(img: np.ndarray, patch: int, stride: int):
    H,W,_ = img.shape
    tiles = []
    coords = []
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y2 = min(y+patch, H)
            x2 = min(x+patch, W)
            tile = np.zeros((patch,patch,3), dtype=img.dtype)
            tile[:(y2-y), :(x2-x)] = img[y:y2, x:x2]
            tiles.append(tile)
            coords.append((x,y,x2,y2))
    return np.stack(tiles), coords, (H,W)

def stitch_mask(masks: np.ndarray, coords, full_hw):
    H,W = full_hw
    out = np.zeros((H,W), dtype=np.uint8)
    k = 0
    for (x,y,x2,y2) in coords:
        ph, pw = (y2-y), (x2-x)
        out[y:y2, x:x2] = masks[k][:ph, :pw]
        k += 1
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--input', required=True, help='folder with *_sat.jpg')
    ap.add_argument('--checkpoint', default=None, help='path to .pt')
    ap.add_argument('--out', required=True, help='output folder for color masks')
    ap.add_argument('--stride', type=int, default=None, help='optional stride override for tiling')
    args = ap.parse_args()
    cfg = load_config(args.cfg)
    os.makedirs(args.out, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = cfg['model']['num_classes']
    steps = cfg['model']['steps']
    patch = cfg['dataset']['patch']['size']
    stride = args.stride if args.stride is not None else cfg['dataset']['patch'].get('stride', patch)
    # CLI override for stride if provided
    # We reuse argparse but keep backward compatibility: users can still set via config
    if '--stride' in []:
        pass

    model = TRMSeg(cfg['model']['in_channels'], num_classes, width=cfg['model']['width'],
                   t_max=steps['t_max'], pixel_act=cfg['model']['pixel_act']).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state['model'])

    model.eval()
    with torch.no_grad():
        for ip in glob.glob(os.path.join(args.input, '*_sat.*')):
            img = np.asarray(Image.open(ip).convert('RGB'))
            tiles, coords, hw = tile_image(img, patch, stride)
            preds = []
            for t in range(len(tiles)):
                x = torch.from_numpy(tiles[t].transpose(2,0,1)).float().unsqueeze(0)/255.0
                x = x.to(device)
                out = model(x, steps=steps['t_max'])
                # ACT decision
                use_idx = steps['t_max']-1
                for s in range(steps['t_min']-1, steps['t_max']):
                    q = out['q_list'][s]
                    if q.dim()==2:
                        cond = (torch.sigmoid(q) > steps['tau']).all().item()
                    else:
                        cond = (torch.sigmoid(q).mean() > steps['tau']).item()
                    if cond:
                        use_idx = s
                        break
                logits = out['logits_list'][use_idx]
                pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                preds.append(pred)
            full = stitch_mask(np.stack(preds), coords, hw)
            color = id_to_color_mask(full)
            out_path = os.path.join(args.out, os.path.basename(ip).replace('_sat','_pred'))
            Image.fromarray(color).save(out_path)

if __name__ == '__main__':
    main()
