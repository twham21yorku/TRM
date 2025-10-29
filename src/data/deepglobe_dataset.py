from __future__ import annotations
import os, csv, random
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .label_codec import rgb_to_id_mask
from .transforms import dihedral8, photometric_jitter

class DeepGlobePatchDataset(Dataset):
    """Patch-level dataset using an index CSV (img,mask,x,y).

    The CSV is produced by src/data/patchify.py and supports overlapping windows.
    Each sample returns:
      - image: FloatTensor [C,H,W] in [0,1]
      - mask:  LongTensor  [H,W] with values {0..5, 255(ignore)}
    """
    def __init__(self, index_csv: str, patch_size: int = 256, augment: bool = True):
        super().__init__()
        self.index = []
        with open(index_csv, 'r') as f:
            r = csv.DictReader(f)
            for row in r:
                self.index.append(row)
        self.patch = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.index)

    def _load_pair(self, img_path: str, mask_path: str) -> Tuple[Image.Image, Image.Image]:
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')  # color mask
        return img, mask

    def __getitem__(self, i: int):
        rec = self.index[i]
        img_path, mask_path = rec['image'], rec['mask']
        x, y = int(rec['x']), int(rec['y'])
        p = self.patch

        img, mask_rgb = self._load_pair(img_path, mask_path)
        # crop
        img = img.crop((x, y, x+p, y+p))
        mask_rgb = mask_rgb.crop((x, y, x+p, y+p))

        if self.augment:
            img, mask_rgb = dihedral8(img, mask_rgb)
            img = photometric_jitter(img)

        # to tensors
        img_np = np.asarray(img).astype(np.float32) / 255.0  # HWC
        img_np = img_np.transpose(2,0,1)  # CHW
        mask_np = rgb_to_id_mask(np.asarray(mask_rgb))

        return torch.from_numpy(img_np), torch.from_numpy(mask_np)
