"""Direct grain-size measurement: grey-scale granulometry at native image resolution.

For each photo, the flat-fielded, contrast-normalised grey image is opened (bright grains) and closed
(dark grains / shadows) with disks of increasing PHYSICAL radius (mm, via the camera's effective
px/mm). The volume removed between successive radii is the pattern spectrum: the fraction of image
structure whose size lies in that radius band. Its cumulative form is an image-based "passing" curve
in millimetres, computed without any training labels.

Large radii use a pyramid: the image is area-downsampled so that the structuring element never
exceeds KMAX px radius; the removed volume is a mean intensity, which area resampling preserves.
"""
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image, ImageOps

import gsd

CROP = 0.6
RADII_MM = np.round(np.logspace(np.log10(0.05), np.log10(60), 36), 4)  # disk radius; diameter = 2 r
KMAX = 10  # max structuring-element radius in px before downsampling


def _disk(r_px):
    r = max(1, int(round(r_px)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def load_native(path, phone):
    im = ImageOps.exif_transpose(Image.open(path)).convert("L")
    ref_side, ppm = gsd.PHONES[phone]
    eff_ppm = ppm * max(im.size) / ref_side
    w, h = im.size
    cw, ch = int(w * CROP), int(h * CROP)
    im = im.crop(((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch))
    return np.asarray(im).astype(np.float32), eff_ppm


def prepare(gray, ppm):
    """Flat-field (remove illumination on scales > 40 mm) and contrast-normalise."""
    f = max(1, int(round(ppm * 2)))  # background at ~0.5 px/mm
    small = cv2.resize(gray, (max(8, gray.shape[1] // f), max(8, gray.shape[0] // f)), interpolation=cv2.INTER_AREA)
    bg = cv2.GaussianBlur(small, (0, 0), 40 * ppm / f)
    bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    g = gray - bg
    return g / (g.std() + 1e-6)


def spectrum(g, ppm, radii_mm=RADII_MM):
    """Opening and closing volumes (mean intensity) at each physical radius."""
    cache = {}
    vo, vc = [], []
    for r_mm in radii_mm:
        r_px = r_mm * ppm
        f = int(np.ceil(r_px / KMAX)) if r_px > KMAX else 1
        if f not in cache:
            cache[f] = g if f == 1 else cv2.resize(g, (max(4, g.shape[1] // f), max(4, g.shape[0] // f)),
                                                   interpolation=cv2.INTER_AREA)
        img = cache[f]
        k = _disk(r_px / f)
        vo.append(float(cv2.morphologyEx(img, cv2.MORPH_OPEN, k).mean()))
        vc.append(float(cv2.morphologyEx(img, cv2.MORPH_CLOSE, k).mean()))
    v0 = float(g.mean())
    return np.array(vo), np.array(vc), v0


def cumulative(vo, vc, v0):
    """Removed-volume curves normalised to [0,1]: fraction of structure smaller than each radius."""
    ro = np.maximum.accumulate(v0 - vo)  # bright structure removed by opening (monotone up)
    rc = np.maximum.accumulate(vc - v0)  # dark structure filled by closing
    go = ro / max(ro[-1], 1e-9)
    gc = rc / max(rc[-1], 1e-9)
    return go, gc


def image_curve(path, phone):
    gray, ppm = load_native(path, phone)
    g = prepare(gray, ppm)
    vo, vc, v0 = spectrum(g, ppm)
    go, gc = cumulative(vo, vc, v0)
    out = {"eff_ppm": ppm, "min_r_px": RADII_MM[0] * ppm}
    for r, a, b in zip(RADII_MM, go, gc):
        out[f"go_{r}"] = a
        out[f"gc_{r}"] = b
    return out


def extract(root, n_jobs=4, split=None):
    imgs = gsd.list_images(root)
    if split:
        imgs = imgs[imgs.split == split].reset_index(drop=True)
    rows = Parallel(n_jobs=n_jobs)(delayed(image_curve)(p, ph) for p, ph in zip(imgs.path, imgs.phone))
    return pd.concat([imgs.drop(columns="path"), pd.DataFrame(rows)], axis=1)


if __name__ == "__main__":
    t = time.time()
    df = extract(gsd.find_root(), split=sys.argv[2] if len(sys.argv) > 2 else None)
    df.to_csv(sys.argv[1], index=False)
    print(df.shape, f"{time.time() - t:.0f}s")
