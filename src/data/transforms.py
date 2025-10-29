from __future__ import annotations
import random
import numpy as np
from typing import Tuple
from PIL import Image, ImageOps, ImageEnhance

def dihedral8(img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """Random dihedral (8) transform with the same op for img & mask."""
    k = random.randint(0,7)
    # base: rotations of 0,90,180,270 + optional flip
    if k in (1,5): img, mask = img.transpose(Image.ROTATE_90), mask.transpose(Image.ROTATE_90)
    elif k in (2,6): img, mask = img.transpose(Image.ROTATE_180), mask.transpose(Image.ROTATE_180)
    elif k in (3,7): img, mask = img.transpose(Image.ROTATE_270), mask.transpose(Image.ROTATE_270)
    if k>=4:
        img = ImageOps.mirror(img)
        mask = ImageOps.mirror(mask)
    return img, mask

def photometric_jitter(img: Image.Image, brightness=0.1, contrast=0.1, gamma=0.1) -> Image.Image:
    if brightness>0:
        factor = 1 + random.uniform(-brightness, brightness)
        img = ImageEnhance.Brightness(img).enhance(factor)
    if contrast>0:
        factor = 1 + random.uniform(-contrast, contrast)
        img = ImageEnhance.Contrast(img).enhance(factor)
    if gamma>0:
        g = 1 + random.uniform(-gamma, gamma)
        img = Image.fromarray(np.clip(255. * (np.asarray(img)/255.)**g, 0, 255).astype(np.uint8))
    return img
