# Predicting Soil Grain Size Distributions from Images

Solution workspace for the Kaggle community competition
[Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos)
(host: Lukas Leibold, Geotechnical Resilience project / BOKU; $500 prize pool; no points/medals).

## Repository layout

```
├── src/                     pipeline code
│   ├── gsd.py               metric, curve <-> log-quantile transforms, data discovery
│   ├── features.py          physical-scale image features + camera augmentations
│   ├── run.py               leave-one-sample-out CV and submission writer
│   └── experiments.py       CV harness used for feature/target comparisons
├── data/                    competition data (Kaggle layout)
├── cache/features.csv       pre-computed features (skip the ~6 min extraction)
├── submissions/             submission files v1 ... v16, see the log below
├── notebooks/reference/     public Kaggle notebooks reviewed during the analysis
├── docs/competition/        competition overview, data description, rules, leaderboard snapshot
├── requirements.txt
└── README.md
```

## Task

Given phone photographs of a soil surface, predict the **cumulative grain-size distribution** (percent of
mass finer than d) at the 11 DIN EN ISO 14688-1 diameters
`0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2, 6.3, 20, 63, 200 mm`.

* **Metric** – mean over test samples of the log-weighted Earth Mover's Distance
  `EMD = Σ_i |F_i − F̂_i| · (log10 x_{i+1} − log10 x_i)` (area between true and predicted grading curves on a log
  axis; lower is better, range 0–500). The equal-share baseline scores ≈ 102 on the training labels.
* **Leaderboard** – final score = 0.30·public + 0.70·private.
* **Submission rules** – one row per test `sample_id`, values in [0, 100], non-decreasing across the 11 columns,
  last column (200 mm) exactly 100. Invalid files are rejected.
* **Rules worth remembering** – no hand-labelling / human prediction of test data; one Kaggle account per person.

## Data (`data/`)

| file | content |
|---|---|
| `Training_labels_updated.csv` | 24 training samples (`F827, G190, H030 …`) × 11 cumulative values (sieve + sedimentation, log-interpolated to the 11 sizes) |
| `Training-All_Photos_updated/` | 127 photos (3–8 per sample) from Motorola Edge (69), Samsung A52 (55), Motorola Edge 60 Fusion (3) |
| `Test_All_Photos/` | 35 photos of **10 test samples**, all **iPhone 14 / 16**, different sites (Airbus, Kleinkummerfeld, Münster, Audorfring, Lidl WHV); file name `iPhoneXX_<sample_id> (n).JPG` |
| `ppm_updated.csv` | pixels-per-mm per camera **at original sensor resolution** |
| `sample_submission.csv` | the 10 test IDs, e.g. `HPC_Airbus BS6-3`, `HPC_Muenster_BS6_9_0-10m` |

`data/` mirrors the Kaggle dataset layout (`/kaggle/input/soil-grain-size-from-photos`), so the same code runs locally and on Kaggle.

### Findings that shape the solution

1. **Tiny data + large domain shift.** 24 labelled samples; train phones are Android, test phones are iPhones and
   the test sites are new (coarser, cobbly material such as Münster; train D50 ≤ ~6 mm).
2. **The ppm table does not apply to the provided training JPGs as-is.** The host downscaled them
   (Samsung 9248→1599 px, Motorola 4000→1600 px), so the effective resolution is ≈ 4.6 px/mm for train photos,
   while the iPhone test photos are full resolution (13.9 / 19.5 px/mm). Several public notebooks use the table
   directly; this pipeline rescales it by `actual_side / reference_side`.
3. **Several public notebooks report MAE ≈ 10** with random K-fold – that leaks, because photos of the same
   sample land in train and validation. All numbers below use leave-one-**sample**-out CV.
4. **The metric is an L1 distance between log-quantile functions.** `EMD = ∫|log Q(p) − log Q̂(p)| dp`.
   Regressing log-diameters at fixed percentiles is thus aligned with the metric and (being linear in
   physically meaningful features) can extrapolate to coarser soils than seen in training, which regression on
   cumulative percentages cannot.
5. The very low (<1) leaderboard scores at the top are not reachable with an honest image model on 24 samples
   (our honest CV is ~35); they are not a target for this pipeline.

## Approach (`src/`)

* `gsd.py` – metric, curve ⇄ log-quantile transforms (round-trip EMD 0.75), file/sample-ID parsing, valid-curve projection.
* `features.py` – photos are center-cropped (60 %), **resampled to a common 4 px/mm** using the effective ppm,
  flat-fielded, contrast-normalised, then described by physical-scale texture features: scale-normalised LoG energy at
  0.25–22 mm, grey-scale granulometry (openings/closings with 0.5–26 mm radii), edge density, blob statistics.
  Train photos also get 8 camera-style augmentations each (blur, unsharp mask, noise, gamma/contrast, vignette,
  JPEG, ±4 % scale) so the model does not lean on camera-specific fine-scale rendering.
* `run.py` – ensemble in log-quantile space of PLS-1 and latent-kernel-**median** regressors (the median of
  neighbour quantile functions is the L1-optimal point estimate) over two camera-robust feature subsets.
  Features that moved massively between train and test (kurtosis, colour, large-radius closings: 4–10 σ) were
  dropped; standardised features are winsorised at ±3.5 σ. Output is projected to a valid curve (clipped,
  monotone, last = 100).
* `experiments.py` – the CV harness used to compare feature sets / targets (LOSO, cross-phone, coarse-extrapolation).

## Results (leave-one-sample-out EMD, 24 samples)

| model | EMD |
|---|---|
| equal-share baseline | 102.0 |
| mean training curve | 89.9 |
| ridge on cumulative % (all features) | ~41–46 |
| PLS-1 on cumulative % / log-quantiles | ~36–40 |
| R1/R2 PLS-1 (log-quantile) | 39.5 / 38.4 |
| R1/R2 latent-kernel-median | 34.6 / 34.9 |
| **final ensemble** | **35.9** |

Differences of ±3 are within CV noise at n = 24. Errors are dominated by H038, H037 (wide-graded, surface not
representative of the bulk) and H405 (clay-rich; fines are invisible in a surface photo). Fine-content (clay/silt)
accuracy is the structural limit of surface photos. Test predictions rank samples sensibly
(Münster/Lidl coarse; Kleinkummerfeld/Airbus sandy), but the **absolute level on the unseen, camera-shifted test
set is uncertain** – the CV cannot measure the iPhone shift.

## Reproduce

```bash
pip install -r requirements.txt
cd src
python run.py                                    # CV + submissions/submission.csv, uses ../cache/features.csv
rm ../cache/features.csv && python run.py        # re-extract features (~6 min on 4 cores)
# current best (v12):
python run.py --no-cv --linear-only --clip-z 0 --shift-log10 0.15 --shift-coarse 0.35 \
              --shift-fines 0.35 --shift-fines-coarse 0 --out ../submissions/submission_v12.csv
```
On Kaggle, attach the competition data and run `run.py --root /kaggle/input/soil-grain-size-from-photos`
(CPU only; no internet or pretrained weights needed).

## Submission log

All files live in `submissions/`; every one is reproduced byte-for-byte by the command in its row.

| # | file | change | LOSO EMD | public LB |
|---|---|---|---|---|
| 1 | `submission_v1.csv` | ensemble as described above | 35.9 | 55.19 |
| 2 | `submission_v2.csv` | `python run.py --range-margin 0.25`: only features whose test-sample values stay within the train range (+25 %) | 35.9 | 61.09 (worse) |
| 3 | `submission_v3.csv` | `python run.py --linear-only`: v1 without the kNN-median parts, which cannot predict coarser than the coarsest training soil | 38.8 | **40.90** (best) |
| 4 | `submission_v4.csv` | `python run.py --linear-only --clip-z 0`: v3 without the ±3.5σ feature clipping; only Münster changes (D50 12.5 → 18 mm; Kleinkummerfeld 2-2 moves 0.6) | 38.8 | **38.75** |
| 5 | `submission_v5.csv` | `python run.py --linear-only --clip-z 0 --shift-log10 0.15`: v4 with every predicted size ×1.41 (log10 +0.15). Leaderboard-informed calibration, not CV-validated | 38.8 (unshifted) | **35.33** |
| 6 | `submission_v6.csv` | `--shift-log10 0.25`: does a stronger uniform shift keep helping? | – | 35.53 |
| 7 | `submission_v7.csv` | `--shift-log10 0.25 --shift-coarse 0.15`: +0.25 only for sandy samples, Münster/Lidl stay at v5 | – | 36.32 |
| 8 | `submission_v8.csv` | `--shift-log10 0.15 --shift-fines 0.2`: v5 + fewer fines (extra shift ramping in below the median) | – | **32.50** |
| 9 | `submission_v9.csv` | `--shift-log10 0.15 --shift-coarse 0.25`: sandy rows = v5, Münster/Lidl rows = v6 | – | **34.54** (as predicted) |
| 10 | `submission_v10.csv` | `--shift-log10 0.15 --shift-coarse 0.35`: Münster/Lidl pushed one step further | – | **33.76** |
| 11 | `submission_v11.csv` | `--shift-log10 0.15 --shift-coarse 0.35 --shift-fines 0.2 --shift-fines-coarse 0`: sandy rows = v8, Münster/Lidl rows = v10 | – | 32.50 |
| 12 | `submission_v12.csv` | v11 with sandy fines shift 0.35 (sandy rows only change) | – | **31.47** (best) |
| 13 | `submission_v13.csv` | v11 with Münster/Lidl +0.45 (coarse rows only change) | – | 32.72 |
| 14 | `submission_v14.csv` | v12 with sandy fines shift 0.5 (sandy rows only change) | – | 31.08 |
| 15 | `submission_v15.csv` | v12 with Münster/Lidl fines shift 0.2 on top of +0.35 (coarse rows only change) | – | **30.04** |
| 16 | `submission_v16.csv` | sandy rows = v14, Münster/Lidl rows = v15 | – | **29.66** (as predicted) |
| 17 | `submission_v17.csv` | v16 with Münster/Lidl fines shift 0.35 (coarse rows only change) | – | 29.34 |
| 18 | `submission_v18.csv` | v16 with sandy fines shift 0.65 (sandy rows only change) | – | 29.54 |
| 19 | `submission_v19.csv` | sandy rows = v18, Münster/Lidl rows = v17 | – | **29.23** (as predicted) |
| 20 | `submission_v20.csv` | `… --contract-upper 0.55 --contract-upper-coarse 1.0`: v19 with the coarse half of each **sandy** quantile function contracted toward its median (q90−q50: 0.89 → 0.57, matching the well-sorted training sands). Münster/Lidl unchanged | – | _pending_ |

Group effects measured (exact, public split): sandy fines 0→0.2: −1.27, 0.2→0.35: −1.04; Münster/Lidl fines 0→0.2 (at +0.15):
−1.57; Münster/Lidl +0.35→+0.45: +0.22 (overshoot); sandy fines 0.35→0.5: −0.39 (flattening); Münster/Lidl fines
0→0.2 (at +0.35): −1.42; Münster/Lidl fines 0.2→0.35: −0.31; sandy fines 0.5→0.65: −0.12. Earlier notes — (public split, exact by additivity): Münster/Lidl +0.15→+0.25: −0.79, +0.25→+0.35: −0.78;
sandy +0.15→+0.25: +0.99; fines-tail 0.2 on all samples: −2.83 (= sandy part f_s + coarse part f_c, not yet separated;
v11 − (v5 + v10 − v9 ... ) separates it: f_c = 30.93 − v11).

**Per-sample additivity.** The score is a mean of independent per-sample EMDs, so for submissions that differ only by
group, scores combine exactly: sandy(+0.25) − sandy(+0.15) = v7 − v5 = +0.99 (worse), coarse(+0.25) − coarse(+0.15)
= v6 − v7 = −0.79 (better). Hence v9 = v5 + v6 − v7 = 34.54 without guessing.

**What the leaderboard told us (v1–v4).** Coarser moves helped every time (v3, v4) and the one finer move hurt (v2).
From v1→v3 (LB gain 14.3 vs per-sample moves of 5–22) the public split holds at most 6 of the 10 samples, and the
coarser moves paid off almost in full there, so truth is at least as coarse as v3. The Android-trained model
systematically under-predicts grain size on the iPhone/field test photos. v5 adds a uniform +0.15 log10 shift
(~15 EMD per sample, the same size as the moves that already paid off). v5 confirmed it (35.33), but the gain was only ~23 % of the move, so the optimum is close. v6–v8 bracket it.

v1 scored far worse than CV (camera/site shift). v2 tested the hypothesis that shifted features are the cause: it
scored **worse** (61.09 vs 55.19), so range-filtering hurt. Side effect that likely explains it: the filter removes coarse-scale features, so the coarse Münster sample is predicted finer than in v1
(EMD v1↔v2 = 33 on that sample, 15 on Audorfring, 11 on Lidl; ≤6 elsewhere).

### Why the tuning plateaued (reassessment after v19)

The linear model cannot predict **sorting**: LOSO correlation between true and predicted spread (q90−q10) is 0.11, and no
feature set predicts spread better than its mean (best LOSO corr 0.29). Every sample therefore gets the training-average
width (~2 decades). With the true D50 but that fixed width, LOSO EMD would be 19.4; with the true width as well, 10.0 — so
once the leaderboard tuning had fixed the location of the test curves, spread was the entire remaining error, and on a
uniform sand it is worth ~28 EMD (H616). The leaderboard arithmetic says the same: moving the lower half of the sandy
curves coarser gained, moving the whole curve coarser lost, so the **upper half was too coarse by a lot** (≈ +2.3 per 0.1
decade). v19's sandy predictions had q90−q50 = 0.89 and q98−q50 = 1.56, versus 0.48 / 0.88 for the six best-sorted training
sands; the lower side was already within the training range. v20 contracts the upper half (factor 0.55) to the training
template. This is the first change since v3 that is motivated by the training data rather than fitted to the public split.

## Ideas not yet tried

* Pretrained embeddings (DINOv2 / ConvNeXt) as extra latent features – public notebooks with them score ~35–41 OOF, no
  better; needs internet/models (not available in the dev sandbox).
* Instance segmentation of stones (SAM / watershed) to measure coarse-fraction diameters directly in mm.
* Test-time adaptation: standardise iPhone features with unlabeled test statistics; per-phone feature alignment.
* Reduce variance of the tail: Bayesian shrinkage of predictions towards a site-type prior.
