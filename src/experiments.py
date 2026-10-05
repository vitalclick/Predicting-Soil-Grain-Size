"""Honest CV experiments: leave-one-sample-out (LOSO), cross-phone and coarse-extrapolation."""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

import gsd

FEAT = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "../cache/features_v1.csv")
ROOT = gsd.find_root()
Y = pd.read_csv(f"{ROOT}/Training_labels_updated.csv", index_col=0)
META = ["file", "split", "sample_id", "phone", "eff_ppm"]


def sample_table(df, cols, phones=None):
    d = df if phones is None else df[df.phone.isin(phones)]
    return d.groupby("sample_id")[cols].mean()


def make_model(kind, alpha):
    if kind == "ridge":
        return Ridge(alpha=alpha)
    return PLSRegression(n_components=int(alpha), scale=False)


def fit_predict(Xtr, Ytr_curve, Xte, kind, alpha, target):
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    if target == "logq":
        T = gsd.curve_to_logq(Ytr_curve)
    elif target == "cum":
        T = Ytr_curve
    elif target == "sqrt":
        T = np.sqrt(Ytr_curve)
    mu = T.mean(0)
    m = make_model(kind, alpha).fit(A, T - mu)
    P = np.asarray(m.predict(B)) + mu
    if target == "logq":
        return gsd.logq_to_curve(P)
    if target == "sqrt":
        return gsd.finalize(np.clip(P, 0, None) ** 2)
    return gsd.finalize(P)


def loso(cols, kind, alpha, target, test_phones=None):
    """Train on photo-mean of every other sample; predict held-out sample; EMD over samples."""
    X = sample_table(FEAT[FEAT.split == "train"], cols)
    ids = list(X.index)
    errs = []
    for s in ids:
        tr = [i for i in ids if i != s]
        pred = fit_predict(X.loc[tr].values, Y.loc[tr].values, X.loc[[s]].values, kind, alpha, target)
        errs.append(gsd.emd(Y.loc[[s]].values, pred)[0])
    return np.mean(errs), np.array(errs), ids


def cross_phone(cols, kind, alpha, target):
    """Train on photos of phone family A only, test on phone family B (domain shift check)."""
    tr_df = FEAT[FEAT.split == "train"]
    fam = lambda p: "Samsung" if p.startswith("Samsung") else "Motorola"
    tr_df = tr_df.assign(fam=tr_df.phone.map(fam))
    out = []
    for a, b in [("Motorola", "Samsung"), ("Samsung", "Motorola")]:
        Xa = tr_df[tr_df.fam == a].groupby("sample_id")[cols].mean()
        Xb = tr_df[tr_df.fam == b].groupby("sample_id")[cols].mean()
        common = Xa.index.intersection(Xb.index)
        errs = []
        for s in common:  # leave sample out + switch phone
            tr = [i for i in Xa.index if i != s]
            pred = fit_predict(Xa.loc[tr].values, Y.loc[tr].values, Xb.loc[[s]].values, kind, alpha, target)
            errs.append(gsd.emd(Y.loc[[s]].values, pred)[0])
        out.append(np.mean(errs))
    return np.mean(out)


def extrapolate(cols, kind, alpha, target, k=5):
    """Hold out the k coarsest samples (by D50) and train on the finer rest."""
    lq = gsd.curve_to_logq(Y.values)
    d50 = pd.Series(lq[:, 50], index=Y.index).sort_values()
    te = list(d50.index[-k:])
    X = sample_table(FEAT[FEAT.split == "train"], cols)
    tr = [i for i in X.index if i not in te]
    pred = fit_predict(X.loc[tr].values, Y.loc[tr].values, X.loc[te].values, kind, alpha, target)
    return gsd.emd(Y.loc[te].values, pred).mean()


if __name__ == "__main__":
    allc = [c for c in FEAT.columns if c not in META]
    groups = {
        "all": allc,
        "scale": [c for c in allc if c.startswith(("log_", "scale_", "slope_", "logkurt"))],
        "gran": [c for c in allc if c.startswith(("open_", "close_"))],
        "scale+gran": [c for c in allc if c.startswith(("log_", "scale_", "slope_", "logkurt", "open_", "close_"))],
        "no_color": [c for c in allc if c not in ("lab_a", "lab_b", "sat", "gray_mean", "eff_ppm")],
    }
    print(f"{'feats':11s} {'model':10s} {'target':6s}   LOSO  xPhone  extrap5")
    for g, cols in groups.items():
        for kind, alphas in [("ridge", [3, 10, 30, 100, 300]), ("pls", [1, 2, 3])]:
            for a in alphas:
                for t in ["cum", "logq"]:
                    l, _, _ = loso(cols, kind, a, t)
                    print(f"{g:11s} {kind}{a:<5} {t:6s} {l:6.2f} {cross_phone(cols, kind, a, t):6.2f} {extrapolate(cols, kind, a, t):7.2f}", flush=True)
