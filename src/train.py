from __future__ import annotations
import os, argparse, time, math
from typing import Dict, Any
import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from .utils import load_config, set_seed, pretty_config, CSVLogger, save_checkpoint
from .data.deepglobe_dataset import DeepGlobePatchDataset
from .models.trm_refiner import TRMSeg
from .losses import TRMLoss
from .metrics import confusion_matrix, iou_from_confmat, boundary_f1

def make_dataloaders(cfg: Dict[str,Any]):
    idx_dir = cfg['dataset']['index_dir']
    bs = cfg['training']['batch_size']
    nw = cfg['training']['num_workers']
    patch = cfg['dataset']['patch']['size']
    train_ds = DeepGlobePatchDataset(os.path.join(idx_dir, 'train_patches.csv'), patch_size=patch, augment=True)
    val_ds   = DeepGlobePatchDataset(os.path.join(idx_dir, 'val_patches.csv'),   patch_size=patch, augment=False)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    return train_loader, val_loader

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--seed', type=int, default=None, help='override random seed')
    ap.add_argument('--out_dir', type=str, default=None, help='override output dir')
    args = ap.parse_args()
    cfg = load_config(args.cfg)
    # apply CLI overrides
    if args.seed is not None:
        cfg['random_seed'] = args.seed
    set_seed(cfg.get('random_seed', 1337))
    pretty_config(cfg)

    train_loader, val_loader = make_dataloaders(cfg)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = cfg['model']['num_classes']
    ignore_index = cfg['dataset']['classes']['ignore_index']
    steps_cfg = cfg['model']['steps']

    model = TRMSeg(
        in_ch=cfg['model']['in_channels'],
        num_classes=num_classes,
        width=cfg['model']['width'],
        t_max=steps_cfg['t_max'],
        pixel_act=cfg['model']['pixel_act'],
    ).to(device)

    loss_fn = TRMLoss(num_classes=num_classes, ignore_index=ignore_index,
                      alpha=cfg['model']['deep_supervision_alpha'],
                      beta=cfg['model']['halting_beta'],
                      gamma=cfg['model']['ponder_gamma'])

    opt = AdamW(model.parameters(), lr=cfg['optimizer']['lr'],
                betas=tuple(cfg['optimizer']['betas']),
                weight_decay=cfg['optimizer']['weight_decay'])
    scaler = GradScaler(enabled=bool(cfg['training']['amp']))

    out_dir = args.out_dir if args.out_dir is not None else cfg['logging']['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    # write metadata for experiment management
    from .utils import write_metadata
    write_metadata(out_dir, cfg)
    logger = CSVLogger(os.path.join(out_dir, 'train_log.csv'),
                       fieldnames=['epoch','lr','train_loss','val_loss','val_miou','val_avg_steps','val_bf1'])

    max_epochs = cfg['training']['max_epochs']
    t_min, t_max, tau = steps_cfg['t_min'], steps_cfg['t_max'], steps_cfg['tau']
    act_on = bool(steps_cfg['act'])

    base_lr = cfg['optimizer']['lr']
    warmup_epochs = int(cfg.get('scheduler',{}).get('warmup_epochs', 0) or 0)

    def set_lr(optim, lr):
        for g in optim.param_groups:
            g['lr'] = lr

    for epoch in range(1, max_epochs+1):
        # Warmup + cosine per-epoch LR
        if warmup_epochs>0 and epoch<=warmup_epochs:
            lr = base_lr * (epoch / max(1, warmup_epochs))
        else:
            # cosine over remaining epochs
            t = epoch - max(1, warmup_epochs)
            T = max(1, max_epochs - warmup_epochs)
            lr = base_lr * 0.5 * (1 + math.cos(math.pi * (t-1) / T))
        set_lr(opt, lr)
        model.train()
        pbar = tqdm(train_loader, desc=f"train {epoch}/{max_epochs}")
        total_loss = 0.0
        for img, mask in pbar:
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=bool(cfg['training']['amp'])):
                out = model(img, steps=t_max)
                loss = loss_fn(out['logits_list'], out['q_list'], mask)
            scaler.scale(loss).backward()
            if cfg['training']['grad_clip_norm']:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip_norm'])
            scaler.step(opt)
            scaler.update()
            total_loss += float(loss.item())
            pbar.set_postfix(lr=f"{lr:.2e}", loss=f"{total_loss/ (pbar.n or 1):.4f}")

        # Validation
        model.eval()
        cm = torch.zeros(num_classes, num_classes, dtype=torch.float64)
        val_loss = 0.0
        total_steps_used = 0
        total_bf1 = 0.0
        n_bf = 0
        with torch.no_grad():
            for img, mask in tqdm(val_loader, desc="val"):
                img = img.to(device); mask = mask.to(device)
                out = model(img, steps=t_max)
                # emulate ACT in validation: pick earliest step satisfying q>tau or t_max
                step_idx = t_max-1
                for t in range(t_min-1, t_max):
                    q = out['q_list'][t]
                    if q.dim()==2:
                        cond = (torch.sigmoid(q) > tau).all().item()
                    else:
                        cond = (torch.sigmoid(q).mean() > tau).item()
                    if cond:
                        step_idx = t
                        break
                logits = out['logits_list'][step_idx]
                loss = loss_fn([logits], [out['q_list'][step_idx]], mask)
                val_loss += float(loss.item())
                pred = logits.argmax(dim=1)
                cm += confusion_matrix(pred, mask, num_classes, ignore_index)
                total_steps_used += (step_idx+1) * img.size(0)
                # boundary F1 on a subset (compute for all here)
                total_bf1 += boundary_f1(pred, mask, ignore_index)
                n_bf += 1

        iou = iou_from_confmat(cm)
        miou = float(torch.nanmean(iou).item())
        avg_steps = total_steps_used / max(1, len(val_loader.dataset))
        bf1 = total_bf1 / max(1, n_bf)
        print(f"Epoch {epoch}: lr={lr:.2e} train_loss={total_loss/len(train_loader):.4f} val_loss={val_loss/len(val_loader):.4f} mIoU={miou:.4f} avg_steps={avg_steps:.2f} bf1={bf1:.3f}")
        logger.log({'epoch': epoch, 'lr': lr, 'train_loss': total_loss/len(train_loader),
                    'val_loss': val_loss/len(val_loader), 'val_miou': miou,
                    'val_avg_steps': avg_steps, 'val_bf1': bf1})

        if epoch % cfg['training']['save_every'] == 0:
            save_checkpoint({'model': model.state_dict(), 'cfg': cfg}, os.path.join(out_dir, f"epoch_{epoch}.pt"))

    logger.close()

if __name__ == '__main__':
    main()
