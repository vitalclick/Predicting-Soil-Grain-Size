"""End-to-end pipeline: features -> honest leave-one-sample-out CV -> submission.csv

    python run.py                      # uses cache/features.csv if present, else extracts (~6 min on 4 cores)
    python run.py --cv-only | --no-cv
    python run.py --root /kaggle/input/soil-grain-size-from-photos --out submission.csv
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import gsd
from features import extract_all

warnings.filterwarnings("ignore")
META = ["file", "split", "sample_id", "phone", "eff_ppm", "aug"]
RANGE_MARGIN, RANGE_F = None, None  # set from --range-margin
LINEAR_ONLY = False  # set from --linear-only: drop kNN-median parts (they cannot extrapolate past the coarsest train soil)
CLIP_Z = 3.5  # winsorise standardised features: no wild extrapolation on out-of-range test photos


def range_filter(F, cols, margin):
    """Keep features whose every TEST-sample mean lies within the training-sample range (+ margin x
    range). Uses unlabeled test photos only (covariate-shift check), no labels."""
    tr = F[(F.split == "train") & (F.aug == 0)].groupby("sample_id")[cols].mean()
    te = F[F.split == "test"].groupby("sample_id")[cols].mean()
    lo, hi = tr.min(), tr.max()
    rg = hi - lo
    ok = ((te >= lo - margin * rg) & (te <= hi + margin * rg)).all()
    return [c for c in cols if ok[c]]


def feature_sets(cols):
    """Camera-robust subsets. Kurtosis / colour / large-radius closing features were dropped: they
    shift massively between the Android training phones and the iPhone test photos."""
    r1 = [c for c in cols if c.startswith(("log_", "scale_", "slope_", "edge_", "open_"))
          and c not in ("open_13.0", "open_18.0", "open_26.0", "edge_8.0", "log_0.25", "log_0.35")]
    r2 = [c for c in cols if c.startswith(("log_", "edge_"))
          and c not in ("edge_8.0", "log_0.25", "log_0.35", "log_0.5")]
    return {"R1": r1, "R2": r2}


def _wmedian(vals, w):
    out = np.empty(vals.shape[1])
    for j in range(vals.shape[1]):
        o = np.argsort(vals[:, j])
        c = np.cumsum(w[o]) / w.sum()
        out[j] = vals[o[np.searchsorted(c, 0.5)], j]
    return out


class Component:
    """One base learner: sample-level features -> log-quantile function."""

    def __init__(self, cols, kind, alpha=1, bw=0.4):
        self.cols, self.kind, self.alpha, self.bw = cols, kind, alpha, bw

    def _fit_lin(self, X, T):
        self.sc = StandardScaler().fit(X)
        self.mu = T.mean(0)
        A = np.clip(self.sc.transform(X), -CLIP_Z, CLIP_Z)
        self.m = (Ridge(alpha=self.alpha) if self.kind == "ridge"
                  else PLSRegression(1, scale=False)).fit(A, T - self.mu)

    def _pred_lin(self, X):
        return np.asarray(self.m.predict(np.clip(self.sc.transform(X), -CLIP_Z, CLIP_Z))) + self.mu

    def fit(self, X, T):
        self.X, self.T = X, T
        self._fit_lin(X, T)
        if self.kind == "knn":  # latent = predicted log-D50, out-of-sample for the training samples
            self.zt = np.array([self._loo_z(i) for i in range(len(X))])
        return self

    def _loo_z(self, i):
        keep = np.arange(len(self.X)) != i
        c = Component(self.cols, "pls").fit(self.X[keep], self.T[keep])
        return c._pred_lin(self.X[i:i + 1])[0, 50]

    def predict(self, X):
        P = self._pred_lin(X)
        if self.kind != "knn":
            return P
        out = []
        for z in P[:, 50]:  # kernel-weighted MEDIAN of neighbour quantile functions (L1-optimal)
            w = np.exp(-0.5 * ((self.zt - z) / self.bw) ** 2) + 1e-12
            out.append(_wmedian(self.T, w))
        return np.array(out)


def build_components(cols):
    fs = feature_sets(cols)
    if RANGE_MARGIN is not None:
        fs = {k: range_filter(RANGE_F, v, RANGE_MARGIN) for k, v in fs.items()}
    comps = []
    for name, c in fs.items():
        # (ridge variants were tried: weakest in CV and sensitive to alpha, so not in the ensemble)
        comps += [(f"{name}-pls1", Component(c, "pls"))]
        if not LINEAR_ONLY:
            comps += [(f"{name}-knn", Component(c, "knn"))]
    return comps


def fit_predict(train_rows, test_rows, Ylq, cols, weights=None):
    """Fit every component on sample-mean features (original+augmented photos), predict the mean
    log-quantile function over the test photos, return per-component and ensemble curves."""
    ids = sorted(train_rows.sample_id.unique())
    out = {}
    for name, comp in build_components(cols):
        X = train_rows.groupby("sample_id")[comp.cols].mean().loc[ids].values
        comp.fit(X, Ylq.loc[ids].values)
        Xt = test_rows.groupby("sample_id")[comp.cols].mean()
        out[name] = (Xt.index, comp.predict(Xt.values))
    idx = out[next(iter(out))][0]
    ens = np.mean([v[1] for v in out.values()], axis=0)
    return idx, {k: gsd.logq_to_curve(v[1]) for k, v in out.items()}, gsd.logq_to_curve(ens)


def loso(F, Y, Ylq, cols):
    tr = F[F.split == "train"]
    ids = sorted(tr.sample_id.unique())
    rec = {}
    for s in ids:
        _, comp_c, ens_c = fit_predict(tr[tr.sample_id != s], tr[(tr.sample_id == s) & (tr.aug == 0)], Ylq, cols)
        for k, v in {**comp_c, "ENSEMBLE": ens_c}.items():
            rec.setdefault(k, []).append(gsd.emd(Y.loc[[s]].values, v)[0])
    return pd.DataFrame(rec, index=ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--features", default=os.path.join(os.path.dirname(__file__), "..", "cache", "features.csv"))
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--n-aug", type=int, default=8)
    ap.add_argument("--range-margin", type=float, default=None,
                    help="drop features whose test values leave the train range by more than this fraction")
    ap.add_argument("--clip-z", type=float, default=3.5, help="winsorisation of standardised features (0 = off)")
    ap.add_argument("--shift-log10", type=float, default=0.0,
                    help="add a constant to every predicted log10 quantile (>0 = coarser). Leaderboard-informed "
                         "calibration for the Android->iPhone/site shift; NOT validated by CV")
    ap.add_argument("--linear-only", action="store_true", help="PLS-1 components only (extrapolating)")
    ap.add_argument("--cv-only", action="store_true")
    ap.add_argument("--no-cv", action="store_true")
    a = ap.parse_args()
    root = gsd.find_root(a.root)
    if os.path.exists(a.features):
        F = pd.read_csv(a.features)
    else:
        F = extract_all(root, n_aug=a.n_aug)
        os.makedirs(os.path.dirname(os.path.abspath(a.features)), exist_ok=True)
        F.to_csv(a.features, index=False)
    Y = pd.read_csv(f"{root}/Training_labels_updated.csv", index_col=0)[gsd.COLS]
    Ylq = pd.DataFrame(gsd.curve_to_logq(Y.values), index=Y.index)
    cols = [c for c in F.columns if c not in META]
    global RANGE_MARGIN, RANGE_F, LINEAR_ONLY, CLIP_Z
    RANGE_MARGIN, RANGE_F, LINEAR_ONLY = a.range_margin, F, a.linear_only
    CLIP_Z = a.clip_z if a.clip_z > 0 else np.inf

    if not a.no_cv:
        res = loso(F, Y, Ylq, cols)
        print("Leave-one-sample-out EMD (24 samples, lower is better)")
        print(res.mean().round(2).to_string())
        eq = gsd.emd(Y.values, np.linspace(100 / 11, 100, 11)[None]).mean()
        print(f"equal-share baseline {eq:.1f}\nper-sample ENSEMBLE:\n{res['ENSEMBLE'].round(1).to_string()}")
    if a.cv_only:
        return
    tr, te = F[F.split == "train"], F[F.split == "test"]
    idx, comp_c, ens = fit_predict(tr, te, Ylq, cols)
    if a.shift_log10:
        ens = gsd.logq_to_curve(gsd.curve_to_logq(ens) + a.shift_log10)
    sub = pd.DataFrame(ens, index=idx, columns=gsd.COLS).round(4)
    tmpl = pd.read_csv(f"{root}/sample_submission.csv")
    sub = sub.loc[tmpl.sample_id].reset_index().rename(columns={"index": "sample_id", "sample_id": "sample_id"})
    sub.columns = tmpl.columns
    # validity checks mirror the competition's structural rules
    v = sub[gsd.COLS].values
    assert list(sub.columns) == list(tmpl.columns) and len(sub) == len(tmpl)
    assert v.min() >= 0 and v.max() <= 100 and (np.diff(v, axis=1) >= -1e-9).all() and (v[:, -1] == 100).all()
    sub.to_csv(a.out, index=False)
    print(f"wrote {a.out}\n{sub.round(1).to_string(index=False)}")


if __name__ == "__main__":
    main()
