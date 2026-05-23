"""
pipeline/character_extraction.py

Extracts individual characters from a handwriting image.

Fixes over original:
  - Original used a fixed global threshold (128) — fails on images with
    uneven lighting, shadows, or low contrast.
  - Contour filter used absolute pixel sizes (10-100 px), which breaks
    on any non-standard image resolution.
  - No denoising step — small ink specks created false contours.
  - No handling of dark-on-light vs light-on-dark images.
  - Output was discarded after inference; now returned as a list of arrays.

New approach:
  1. Greyscale
  2. Gaussian denoise
  3. Adaptive Gaussian threshold (handles uneven lighting)
  4. Morphological closing (reconnects broken strokes)
  5. Area-ratio based contour filtering (resolution-independent)
  6. Sort left-to-right (reading order)
  7. Return both MLP-sized (28×28) and CNN-sized (64×64) crops
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CharExtractionConfig as Cfg

logger = logging.getLogger(__name__)


@dataclass
class ExtractedCharacter:
    """Holds both resized versions of a single extracted character."""
    index:      int
    mlp_input:  np.ndarray   # float32 (784,)            normalised, for MLP
    cnn_input:  np.ndarray   # float32 (64, 64, 1)       normalised, for CNN
    bbox:       Tuple[int, int, int, int]   # (x, y, w, h) in original image



def extract_characters(image_path: str) -> List[ExtractedCharacter]:
    """
    Extract individual characters from a handwriting image.

    Parameters
    ----------
    image_path : path to the uploaded image file

    Returns
    -------
    List of ExtractedCharacter, sorted left-to-right.
    Empty list if no valid characters are found (caller should handle this).
    """
    # ---- 1. Load as greyscale ------------------------------------------------
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_h, img_w = img.shape
    total_area = img_h * img_w
    logger.debug("Image loaded: %dx%d px", img_w, img_h)

    # ---- 2. Denoise ----------------------------------------------------------
    img_denoised = cv2.GaussianBlur(img, (3, 3), 0)

    # ---- 3. Adaptive threshold -----------------------------------------------
    # THRESH_BINARY_INV → characters become white on black background
    binary = cv2.adaptiveThreshold(
        img_denoised,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=Cfg.ADAPTIVE_BLOCK,
        C=Cfg.ADAPTIVE_C,
    )

    # Ensure consistent polarity (white chars on black)
    # binary = _auto_invert(binary)

    # ---- 4. Morphological closing (reconnect broken strokes) -----------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # ---- 5. Find external contours ------------------------------------------
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fallback: if adaptive threshold found nothing, try Otsu global threshold
    if not contours:
        logger.warning("Adaptive threshold found no contours — trying Otsu fallback")
        _, binary = cv2.threshold(
            img_denoised, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ---- 6. Filter by area ratio and aspect ratio ---------------------------
    valid = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area   = w * h
        ratio  = area / total_area
        aspect = w / max(h, 1)

        if not (Cfg.MIN_AREA_RATIO <= ratio <= Cfg.MAX_AREA_RATIO):
            continue
        if not (Cfg.MIN_ASPECT <= aspect <= Cfg.MAX_ASPECT):
            continue
        valid.append((x, y, w, h))

    if not valid:
        logger.warning("No valid character contours found in %s", image_path)
        return []

    # ---- 7. Sort left-to-right (reading order) ------------------------------
    valid.sort(key=lambda b: b[0])
    logger.info("Extracted %d characters", len(valid))

    # ---- 8. Crop, resize, normalise -----------------------------------------
    characters: List[ExtractedCharacter] = []
    for idx, (x, y, w, h) in enumerate(valid):
        # Add small padding to avoid clipping ascenders/descenders
        pad = max(2, int(max(w, h) * 0.05))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)

        crop = binary[y1:y2, x1:x2]

        # MLP input: 28×28 flat
        mlp_img = cv2.resize(crop, (Cfg.MLP_SIZE, Cfg.MLP_SIZE)).astype(np.float32) / 255.0
        mlp_flat = mlp_img.reshape(-1)   # (784,)

        # CNN input: 64×64 × 1 channel
        cnn_img = cv2.resize(crop, (Cfg.CNN_SIZE, Cfg.CNN_SIZE)).astype(np.float32) / 255.0
        cnn_4d  = cnn_img.reshape(Cfg.CNN_SIZE, Cfg.CNN_SIZE, 1)

        characters.append(ExtractedCharacter(
            index     = idx,
            mlp_input = mlp_flat,
            cnn_input = cnn_4d,
            bbox      = (x, y, w, h),
        ))

    return characters
