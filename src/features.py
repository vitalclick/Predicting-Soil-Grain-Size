"""Physical-scale handcrafted image features.

Every photo is resampled to a common physical resolution (CANON_PPM px/mm) using the camera's
pixels-per-mm, so that texture scales are expressed in millimetres regardless of the phone.
Features are contrast-normalised (robust to exposure / camera tone mapping).
"""
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageOps
from joblib import Parallel, delayed

import gsd

CANON_PPM = 4.0  # px / mm after resampling (training JPGs are ~4.6 px/mm natively)
CROP = 0.6  # central crop fraction per side: avoids tray walls, lamp and vignetting
SIGMAS_MM = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0]
GRAN_MM = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 9.0, 13.0, 18.0, 26.0]  # structuring-element radii
EDGE_MM = [0.5, 1.0, 2.0, 4.0, 8.0]


def load_canonical(path, phone):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    ref_side, ppm = gsd.PHONES[phone]
    eff_ppm = ppm * max(im.size) / ref_side  # effective px/mm of the file we actually have
    s = CANON_PPM / eff_ppm
    w, h = im.size
    cw, ch = int(w * CROP), int(h * CROP)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    im = im.crop((x0, y0, x0 + cw, y0 + ch))
    new = (max(64, int(round(cw * s))), max(64, int(round(ch * s))))
    im = im.resize(new, Image.LANCZOS if s < 1 else Image.BICUBIC)
    return np.asarray(im), eff_ppm


def _disk(r_px):
    r = max(1, int(round(r_px)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def augment(rgb, rng):
    """Camera-style perturbations (blur / sharpening / noise / tone / vignette / JPEG) so that the
    learned mapping does not rely on camera-specific fine-scale rendering (train: Android phones,
    test: iPhones). Physical scale is only jittered by +-4%."""
    x = rgb.astype(np.float32)
    s = rng.uniform(0.96, 1.04)
    if abs(s - 1) > 0.01:
        x = cv2.resize(x, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    sig = rng.uniform(0, 1.0)
    if sig > 0.15:
        x = cv2.GaussianBlur(x, (0, 0), sig)
    amt = rng.uniform(0, 1.2)  # unsharp mask (HDR/"smart" sharpening on phones)
    x = x + amt * (x - cv2.GaussianBlur(x, (0, 0), rng.uniform(0.8, 2.0)))
    x = 255 * np.clip(x / 255, 0, 1) ** rng.uniform(0.8, 1.25)
    x = (x - 128) * rng.uniform(0.8, 1.25) + 128 + rng.uniform(-15, 15)
    h, w = x.shape[:2]  # vignette / illumination gradient
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((yy - h / 2) / h) ** 2 + ((xx - w / 2) / w) ** 2
    x = x * (1 - rng.uniform(0, 0.35) * r2 * 4)[..., None]
    x = x + rng.normal(0, rng.uniform(0, 5), x.shape).astype(np.float32)
    x = np.clip(x, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", x[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(55, 96))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]


def image_features(path, phone, aug_seed=None):
    rgb, eff_ppm = load_canonical(path, phone)
    if aug_seed is not None:
        rgb = augment(rgb, np.random.default_rng(aug_seed))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # flat-field: remove illumination gradients / vignette (large-scale blur ~ 40 mm)
    bg = cv2.GaussianBlur(gray, (0, 0), 40 * CANON_PPM)
    g = gray - bg
    sd = float(g.std()) + 1e-6
    gn = g / sd  # contrast-normalised
    f = {}
    # 1. scale-space energy (LoG, scale-normalised) at physical scales
    resp = []
    for s_mm in SIGMAS_MM:
        sg = s_mm * CANON_PPM
        lap = cv2.Laplacian(cv2.GaussianBlur(gn, (0, 0), sg), cv2.CV_32F) * sg * sg
        e = float(np.mean(np.abs(lap)))
        resp.append(e)
        f[f"log_{s_mm}"] = np.log(e + 1e-6)
        f[f"logkurt_{s_mm}"] = float(np.mean(lap ** 4) / (np.mean(lap ** 2) ** 2 + 1e-9))
    resp = np.array(resp)
    p = resp / resp.sum()
    lx = np.log10(SIGMAS_MM)
    f["scale_centroid"] = float((p * lx).sum())
    f["scale_spread"] = float(np.sqrt((p * (lx - (p * lx).sum()) ** 2).sum()))
    f["scale_argmax"] = float(lx[resp.argmax()])
    # slope of the spectrum in coarse range
    f["slope_coarse"] = float(np.polyfit(lx[-7:], np.log(resp[-7:] + 1e-9), 1)[0])
    f["slope_fine"] = float(np.polyfit(lx[:7], np.log(resp[:7] + 1e-9), 1)[0])
    # 2. grey-scale granulometry: removed brightness (openings) and darkness (closings) vs radius
    g8 = np.clip(gn * 40 + 128, 0, 255).astype(np.uint8)
    for r_mm in GRAN_MM:
        k = _disk(r_mm * CANON_PPM)
        src = g8 if r_mm <= 6 else cv2.resize(g8, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        kk = k if r_mm <= 6 else _disk(r_mm * CANON_PPM / 2)
        op = cv2.morphologyEx(src, cv2.MORPH_OPEN, kk).astype(np.float32)
        cl = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kk).astype(np.float32)
        f[f"open_{r_mm}"] = float((src.astype(np.float32) - op).mean() / 40)
        f[f"close_{r_mm}"] = float((cl - src.astype(np.float32)).mean() / 40)
    # 3. edge density at physical scales
    for s_mm in EDGE_MM:
        b = cv2.GaussianBlur(g8, (0, 0), s_mm * CANON_PPM * 0.5)
        ed = cv2.Canny(b, 10, 30)
        f[f"edge_{s_mm}"] = float((ed > 0).mean() * s_mm)
    # 4. global fine-texture / contrast cues
    f["gray_sd_rel"] = float(np.log(sd))
    f["gray_mean"] = float(gray.mean() / 255)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    f["lab_a"], f["lab_b"] = float(lab[..., 1].mean() - 128), float(lab[..., 2].mean() - 128)
    f["sat"] = float(hsv[..., 1].mean() / 255)
    # 5. dark/bright blob statistics (large stones show as separated bright/dark blobs)
    sm = cv2.GaussianBlur(gn, (0, 0), 1.0 * CANON_PPM)
    for q in (90, 98):
        thr = np.percentile(sm, q)
        m = (sm > thr).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        areas = st[1:, cv2.CC_STAT_AREA] / CANON_PPM ** 2  # mm^2
        f[f"blob_p{q}_n"] = float(np.log1p(n - 1))
        f[f"blob_p{q}_maxd"] = float(np.log(2 * np.sqrt(areas.max() / np.pi) + 1e-3)) if len(areas) else 0.0
    f["eff_ppm"] = eff_ppm
    return f


def extract_all(root, n_aug=0, n_jobs=4, seed=0):
    """Features of every photo; with n_aug>0 also n_aug augmented copies of every TRAIN photo
    (column `aug` = 0 for the original, 1..n_aug for the augmented ones)."""
    imgs = gsd.list_images(root)
    jobs = [(i, 0, None) for i in range(len(imgs))]
    for k in range(1, n_aug + 1):
        jobs += [(i, k, seed * 100003 + k * 1009 + i) for i in range(len(imgs)) if imgs.split[i] == "train"]
    feats = Parallel(n_jobs=n_jobs)(
        delayed(image_features)(imgs.path[i], imgs.phone[i], sd) for i, _, sd in jobs)
    meta = imgs.drop(columns="path").iloc[[j[0] for j in jobs]].reset_index(drop=True)
    meta["aug"] = [j[1] for j in jobs]
    return pd.concat([meta, pd.DataFrame(feats)], axis=1)


if __name__ == "__main__":
    import sys, time
    t = time.time()
    root = gsd.find_root()
    df = extract_all(root, n_aug=int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    df.to_csv(sys.argv[1] if len(sys.argv) > 1 else "features_v1.csv", index=False)
    print(df.shape, f"{time.time()-t:.0f}s")
