"""Shared helpers: grain-size grid, EMD metric, quantile-space transforms, data discovery."""
import glob
import os
import re

import numpy as np
import pandas as pd

SIZES = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2, 6.3, 20, 63, 200])
COLS = ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"]
LOGX = np.log10(SIZES)
DLOG = np.diff(LOGX)

# ppm_updated.csv gives px/mm at the ORIGINAL sensor resolution. The host-provided training JPGs
# were downscaled, so effective ppm = ppm * (max_side / ref_max_side).
PHONES = {  # filename prefix -> (ref max side px, ppm); order matters (most specific first)
    "Motorola_Edge_60_fusion": (4096, 12.465),
    "Motorola_Edge": (4000, 11.492),
    "Samsung_A52": (9248, 26.33),
    "iPhone14": (4032, 13.942),
    "iPhone16": (5712, 19.525),
    "iPhone_16": (5712, 19.525),
}


def emd(true, pred):
    """Log-weighted EMD per row (the competition metric)."""
    true, pred = np.atleast_2d(true), np.atleast_2d(pred)
    return (np.abs(true - pred)[:, :-1] * DLOG).sum(1)


def finalize(curve):
    """Make cumulative curves valid: clip [0,100], non-decreasing, last column == 100."""
    c = np.clip(np.atleast_2d(curve).astype(float), 0, 100)
    c = np.maximum.accumulate(c, axis=1)
    c[:, -1] = 100.0
    return c


# ---- quantile-space representation -------------------------------------------------------
# Log-weighted EMD == integral over p of |logQ(p) - logQhat(p)|, i.e. the L1 distance between
# log-quantile functions. Regressing log-diameters at fixed percentiles is therefore aligned with
# the metric AND extrapolates in size (a linear model on physical-scale features can predict
# coarser-than-seen soils), which a regression on cumulative percentages cannot.
P_GRID = np.r_[0.5, np.arange(1, 100, 1.0), 99.5]  # dense grid keeps the curve round-trip EMD < 1


def curve_to_logq(curve, p=P_GRID):
    """Cumulative curves (n,11) -> log10 diameter at percentiles p (n,len(p))."""
    out = np.empty((len(np.atleast_2d(curve)), len(p)))
    for i, f in enumerate(np.atleast_2d(curve)):
        f = np.maximum.accumulate(np.clip(f, 0, 100))
        # inverse CDF: (F, logx) pairs. Plateaus collapse to the point where F first reaches the
        # value, except the zero plateau where we use the last x with F == 0.
        fx = [0.0]
        lx = [LOGX[np.where(f <= 1e-9)[0].max()] if f[0] <= 1e-9 else LOGX[0] - 1e-6]
        for F, x in zip(f, LOGX):
            if F > fx[-1] + 1e-9:
                fx.append(F)
                lx.append(x)
        out[i] = np.interp(p, fx, lx)
    return out


def logq_to_curve(logq, p=P_GRID):
    """Predicted log-quantiles (n,len(p)) -> cumulative curve at the 11 sizes (n,11)."""
    res = np.empty((len(np.atleast_2d(logq)), 11))
    for i, q in enumerate(np.atleast_2d(logq)):
        q = np.maximum.accumulate(q)  # monotone quantile function
        pp = np.r_[0.0, p, 100.0]
        qq = np.r_[q[0] - 1.0, q, q[-1] + 0.3]
        qq = qq + np.arange(len(qq)) * 1e-9  # strictly increasing for interpolation
        res[i] = np.interp(LOGX, qq, pp, left=0.0, right=100.0)
    return finalize(res)


# ---- data discovery -----------------------------------------------------------------------
def find_root(start=None):
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [start, os.path.join(here, "..", "soil-grain-size-from-photos"),
             "/kaggle/input/soil-grain-size-from-photos", "soil-grain-size-from-photos"]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "Training_labels_updated.csv")):
            return os.path.abspath(c)
    raise FileNotFoundError("competition data not found")


def phone_of(fname):
    for k in PHONES:
        if fname.startswith(k):
            return k
    raise ValueError(fname)


TEST_ID_FIX = {"HPC_Münster_BS6_9,0-10m": "HPC_Muenster_BS6_9_0-10m"}


def sample_of(fname, split):
    base = os.path.splitext(fname)[0]
    if split == "train":
        return re.search(r"_([A-Z]\d{3})_\d+$", base).group(1)
    s = re.sub(r"^iPhone_?\d+_", "", base)
    s = re.sub(r"\s*\(\d+\)$", "", s)
    return TEST_ID_FIX.get(s, s)


def list_images(root):
    rows = []
    for split, d in [("train", "Training-All_Photos_updated/Training-All_Photos_updated"),
                     ("test", "Test_All_Photos/Test_All_Photos")]:
        for f in sorted(glob.glob(os.path.join(root, d, "*"))):
            n = os.path.basename(f)
            rows.append(dict(path=f, file=n, split=split, sample_id=sample_of(n, split), phone=phone_of(n)))
    return pd.DataFrame(rows)
