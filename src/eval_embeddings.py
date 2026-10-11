"""Evaluate DINOv2 embeddings (from notebooks/dinov2_embeddings.ipynb) against the handcrafted features.

    python eval_embeddings.py path/to/dinov2_embeddings.csv

The question is not overall LOSO EMD alone but whether the embeddings can RANK the fine/sandy samples by D50: our
features cannot (LOSO Spearman -0.22), which is why the final model pools those samples toward their group median.
"""
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import gsd
import run

warnings.filterwarnings("ignore")
I50 = int(np.argmin(np.abs(gsd.P_GRID - 50)))
PRIMARY = ("tile_mean", 5, "pls")  # fixed before seeing any real embeddings
N_PERM = 200


def loso(X, LQ, ids, model, n_pca=None):
    """Leave-one-sample-out log-quantile predictions; PCA and scaling are fit inside each fold."""
    P = np.zeros_like(LQ)
    for i in range(len(ids)):
        tr = np.arange(len(ids)) != i
        sc = StandardScaler().fit(X[tr])
        A, B = sc.transform(X[tr]), sc.transform(X[i:i + 1])
        if n_pca:
            p = PCA(n_components=min(n_pca, tr.sum() - 1)).fit(A)
            A, B = p.transform(A), p.transform(B)
        mu = LQ[tr].mean(0)
        m = (PLSRegression(1, scale=False) if model == "pls" else Ridge(alpha=model)).fit(A, LQ[tr] - mu)
        P[i] = np.asarray(m.predict(B))[0] + mu
    return P


def report(name, P, LQ, Y, fine):
    emd = gsd.emd(Y, gsd.logq_to_curve(P))
    rho_all = spearmanr(P[:, I50], LQ[:, I50])[0]
    rho_fine = spearmanr(P[fine, I50], LQ[fine, I50])[0]
    print(f"{name:34s} LOSO EMD {emd.mean():5.1f} | fine {emd[fine].mean():5.1f} | "
          f"D50 rank: all {rho_all:+.2f}, fine {rho_fine:+.2f}")
    return rho_fine


def main(path):
    root = gsd.find_root()
    Y = pd.read_csv(f"{root}/Training_labels_updated.csv", index_col=0)[gsd.COLS]
    ids = list(Y.index)
    LQ = gsd.curve_to_logq(Y.values)
    fine = LQ[:, I50] < 0

    E = pd.read_csv(path)
    E = E[E.split == "train"]
    groups = {g: [c for c in E.columns if c.startswith(g)] for g in ("tile_mean", "tile_cls", "whole_mean", "whole_cls")}
    groups["tile_all"] = groups["tile_mean"] + groups["tile_cls"]
    Es = E.groupby("sample_id")[sum(groups.values(), [])].mean().loc[ids]

    F = pd.read_csv(f"{run.os.path.dirname(run.__file__)}/../cache/features.csv")
    F = F[(F.split == "train") & (F.aug == 0)]
    hand = run.feature_sets([c for c in F.columns if c not in run.META])["R2"]
    Fs = F.groupby("sample_id")[hand].mean().loc[ids]

    print("baseline (handcrafted R2 features):")
    best = report("  handcrafted R2, PLS-1", loso(Fs.values, LQ, ids, "pls"), LQ, Y.values, fine)
    print("DINOv2:")
    for g, cols in groups.items():
        for n_pca, model in [(None, "pls"), (5, "pls"), (10, 100.0), (None, 1000.0)]:
            tag = f"  {g}, {'PCA' + str(n_pca) + ', ' if n_pca else ''}{'PLS-1' if model == 'pls' else 'ridge ' + str(model)}"
            report(tag, loso(Es[cols].values, LQ, ids, model, n_pca), LQ, Y.values, fine)

    # Pre-registered primary test. With 12 fine samples one configuration out of many can reach |rho| ~ 0.8 by pure
    # chance, so only this single configuration counts; the rows above are exploratory.
    X = Es[groups[PRIMARY[0]]].values
    P = loso(X, LQ, ids, PRIMARY[2], PRIMARY[1])
    print(f"\nPRIMARY: {PRIMARY[0]}, PCA{PRIMARY[1]}, PLS-1")
    rho = report("  observed", P, LQ, Y.values, fine)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(N_PERM):  # shuffle which sample each embedding belongs to
        Pp = loso(X[rng.permutation(len(ids))], LQ, ids, PRIMARY[2], PRIMARY[1])
        null.append(spearmanr(Pp[fine, I50], LQ[fine, I50])[0])
    p = (1 + np.sum(np.array(null) >= rho)) / (1 + N_PERM)
    print(f"  permutation p-value for fine-sample rho {rho:+.2f}: {p:.3f}  (null 95th pct {np.percentile(null, 95):+.2f})")
    print("\nDecision rule: build a submission only if the PRIMARY fine-sample rho is clearly positive (p < 0.05)\n"
          f"and its LOSO EMD is not worse than the handcrafted baseline (fine rho {best:+.2f}).")


if __name__ == "__main__":
    main(sys.argv[1])
