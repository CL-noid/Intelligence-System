"""
segmentation.py
Core ML logic for Module 1: 图像阈值分割
Based on TIF_Final.py — includes denoising, hole-filling, and Gaussian smoothing.
Output: black-and-white binary mask only (no color visualization).
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Union, List, Tuple


# ─────────────────────────────────────────────
#  Post-processing parameters (tunable)
# ─────────────────────────────────────────────
MIN_AREA      = 1000       # remove white blobs smaller than this (pixels)
HOLE_KERNEL   = 19         # morphological closing kernel size (odd number)
SMOOTH_MODE   = 'gaussian' # 'gaussian' or 'median'
SMOOTH_PARAM  = 3.0        # gaussian sigma  OR  median kernel size


# ─────────────────────────────────────────────
#  Model Definition
# ─────────────────────────────────────────────

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet4Layer(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        bc = base_channels
        self.enc1 = DoubleConv(1, bc);          self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(bc, bc*2);       self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(bc*2, bc*4);     self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(bc*4, bc*8);     self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(bc*8, bc*16)
        self.up4  = nn.ConvTranspose2d(bc*16, bc*8,  2, stride=2)
        self.dec4 = DoubleConv(bc*16, bc*8)
        self.up3  = nn.ConvTranspose2d(bc*8,  bc*4,  2, stride=2)
        self.dec3 = DoubleConv(bc*8,  bc*4)
        self.up2  = nn.ConvTranspose2d(bc*4,  bc*2,  2, stride=2)
        self.dec2 = DoubleConv(bc*4,  bc*2)
        self.up1  = nn.ConvTranspose2d(bc*2,  bc,    2, stride=2)
        self.dec1 = DoubleConv(bc*2,  bc)
        self.final = nn.Conv2d(bc, 2, 1)

    def forward(self, x):
        e1 = self.enc1(x);  p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        e4 = self.enc4(p3); p4 = self.pool4(e4)
        bn = self.bottleneck(p4)
        x  = self.dec4(torch.cat([self.up4(bn), e4], 1))
        x  = self.dec3(torch.cat([self.up3(x),  e3], 1))
        x  = self.dec2(torch.cat([self.up2(x),  e2], 1))
        x  = self.dec1(torch.cat([self.up1(x),  e1], 1))
        return self.final(x)


# ─────────────────────────────────────────────
#  Model Loader (singleton – load once)
# ─────────────────────────────────────────────

_model  = None
_device = None
TILE_SIZE = 512


def load_model(model_path: str):
    global _model, _device
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _model  = UNet4Layer(base_channels=32).to(_device)
    _model.load_state_dict(torch.load(model_path, map_location=_device))
    _model.eval()
    print(f"[segmentation] Model loaded on {_device}: {model_path}")


def is_model_loaded() -> bool:
    return _model is not None


# ─────────────────────────────────────────────
#  Post-processing  (from TIF_Final.py)
# ─────────────────────────────────────────────

def _postprocess(binary_img: np.ndarray) -> np.ndarray:
    """
    1. Remove small white blobs (< MIN_AREA pixels)
    2. Fill black holes inside white regions (morphological closing)
    3. Smooth edges (Gaussian or Median)
    Returns 0/255 uint8 image.
    """
    img = binary_img.astype(np.uint8)

    # Step 1 — remove small white noise blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    clean = np.zeros_like(img)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
            clean[labels == i] = 255

    # Step 2 — fill black holes inside white regions
    if HOLE_KERNEL > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (HOLE_KERNEL, HOLE_KERNEL))
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    # Step 3 — edge smoothing
    if SMOOTH_MODE == 'median':
        k = int(round(SMOOTH_PARAM))
        if k % 2 == 0:
            k += 1
        if k >= 3:
            clean = cv2.medianBlur(clean, k)
            _, clean = cv2.threshold(clean, 127, 255, cv2.THRESH_BINARY)
    elif SMOOTH_MODE == 'gaussian':
        sigma = float(SMOOTH_PARAM)
        if sigma > 0:
            blurred = cv2.GaussianBlur(clean, (0, 0), sigma)
            _, clean = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return clean


# ─────────────────────────────────────────────
#  Per-tile inference
# ─────────────────────────────────────────────

def _predict_tile(tile_gray: np.ndarray) -> np.ndarray:
    """Single 512×512 grayscale tile → 0/255 binary mask after post-processing."""
    if tile_gray.shape != (TILE_SIZE, TILE_SIZE):
        tile_gray = cv2.resize(tile_gray, (TILE_SIZE, TILE_SIZE),
                               interpolation=cv2.INTER_NEAREST)

    tensor = (torch.from_numpy(tile_gray.astype(np.float32) / 255.0)
              .unsqueeze(0).unsqueeze(0).to(_device))

    with torch.no_grad():
        pred = torch.argmax(_model(tensor), dim=1).cpu().numpy()[0]

    mask_uint8 = (pred * 255).astype(np.uint8)
    return _postprocess(mask_uint8)


# ─────────────────────────────────────────────
#  Full pipeline
# ─────────────────────────────────────────────

def run_segmentation(image_bytes: bytes) -> bytes:
    """
    1. Decode image bytes → grayscale
    2. Split into 512×512 tiles (edge remainder discarded)
    3. UNet inference + post-processing on each tile
    4. Stitch tiles back into full canvas
    5. Return PNG bytes (black & white binary mask)
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # ── Decode ────────────────────────────────
    nparr    = np.frombuffer(image_bytes, np.uint8)
    img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        pil      = Image.open(BytesIO(image_bytes)).convert('L')
        img_gray = np.array(pil)

    H, W = img_gray.shape

    # ── Split ─────────────────────────────────
    n_rows = max(1, H // TILE_SIZE)
    n_cols = max(1, W // TILE_SIZE)

    # ── Infer each tile ───────────────────────
    processed_tiles = []
    for r in range(n_rows):
        for c in range(n_cols):
            y0, x0 = r * TILE_SIZE, c * TILE_SIZE
            tile   = img_gray[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE]
            # pad if needed (shouldn't happen with integer division, but safe)
            ph, pw = tile.shape
            if ph < TILE_SIZE or pw < TILE_SIZE:
                padded = np.zeros((TILE_SIZE, TILE_SIZE), dtype=tile.dtype)
                padded[:ph, :pw] = tile
                tile = padded
            processed_tiles.append(_predict_tile(tile))

    # ── Stitch ────────────────────────────────
    canvas = np.zeros((n_rows * TILE_SIZE, n_cols * TILE_SIZE), dtype=np.uint8)
    idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            y0, x0 = r * TILE_SIZE, c * TILE_SIZE
            canvas[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE] = processed_tiles[idx]
            idx += 1

    # ── Encode to PNG ─────────────────────────
    ok, buf = cv2.imencode('.png', canvas)
    if not ok or buf is None:
        raise RuntimeError("Failed to encode output image.")
    return buf.tobytes()
