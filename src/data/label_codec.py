from __future__ import annotations
import torch
import numpy as np
from typing import Dict, Tuple

# DeepGlobe Land Cover class colors (R,G,B)
# Source: public DeepGlobe repo readme & challenge page.
# Urban:      (0,255,255)
# Agriculture:(255,255,0)
# Rangeland:  (255,0,255)
# Forest:     (0,255,0)
# Water:      (0,0,255)
# Barren:     (255,255,255)
# Unknown:    (0,0,0) -> ignored
COLOR_TO_NAME = {
    (0,255,255): "urban",
    (255,255,0): "agriculture",
    (255,0,255): "rangeland",
    (0,255,0): "forest",
    (0,0,255): "water",
    (255,255,255): "barren",
    (0,0,0): "unknown",
}

NAME_TO_ID = {
    "urban": 0,
    "agriculture": 1,
    "rangeland": 2,
    "forest": 3,
    "water": 4,
    "barren": 5,
    "unknown": 255,  # use 255 as ignore_index
}

ID_TO_NAME = {v:k for k,v in NAME_TO_ID.items()}

def rgb_to_id_mask(mask_rgb: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Convert DeepGlobe color mask (H,W,3 uint8) to class id mask (H,W uint8).
    Robust to compression by binarizing per channel at `threshold`.
    Unknown (0,0,0) -> 255 (ignore).
    """
    assert mask_rgb.ndim == 3 and mask_rgb.shape[2] == 3, "mask must be HxWx3"
    r = (mask_rgb[...,0] >= threshold).astype(np.uint8)
    g = (mask_rgb[...,1] >= threshold).astype(np.uint8)
    b = (mask_rgb[...,2] >= threshold).astype(np.uint8)
    code = (r<<2) + (g<<1) + b  # 3-bit code
    # Mapping from 3-bit code to class id
    mapping = {
        0: 255,  # 000 unknown -> ignore
        3: NAME_TO_ID["urban"],       # 011 -> (0,255,255)
        6: NAME_TO_ID["agriculture"], # 110 -> (255,255,0)
        5: NAME_TO_ID["rangeland"],   # 101 -> (255,0,255)
        2: NAME_TO_ID["forest"],      # 010 -> (0,255,0)
        1: NAME_TO_ID["water"],       # 001 -> (0,0,255)
        7: NAME_TO_ID["barren"],      # 111 -> (255,255,255)
        4: 255,  # 100 shouldn't appear; treat as unknown
    }
    id_mask = np.vectorize(mapping.get)(code).astype(np.uint8)
    return id_mask

def id_to_color_mask(id_mask: np.ndarray) -> np.ndarray:
    """Map class ids back to RGB color mask for visualization."""
    H,W = id_mask.shape
    out = np.zeros((H,W,3), dtype=np.uint8)
    for color, name in COLOR_TO_NAME.items():
        cid = NAME_TO_ID[name]
        if cid == 255:  # unknown/ignore
            continue
        out[id_mask==cid] = np.array(color, dtype=np.uint8)
    return out
