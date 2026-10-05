# Predicting Soil Grain Size Distributions from Images

Solution workspace for the Kaggle community competition
[Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos)
(host: Lukas Leibold, Geotechnical Resilience project / BOKU; $500 prize pool; no points/medals).

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

## Data (`soil-grain-size-from-photos/`)

| file | content |
|---|---|
| `Training_labels_updated.csv` | 24 training samples (`F827, G190, H030 …`) × 11 cumulative values (sieve + sedimentation, log-interpolated to the 11 sizes) |
| `Training-All_Photos_updated/` | 127 photos (3–8 per sample) from Motorola Edge (69), Samsung A52 (55), Motorola Edge 60 Fusion (3) |
| `Test_All_Photos/` | 35 photos of **10 test samples**, all **iPhone 14 / 16**, different sites (Airbus, Kleinkummerfeld, Münster, Audorfring, Lidl WHV); file name `iPhoneXX_<sample_id> (n).JPG` |
| `ppm_updated.csv` | pixels-per-mm per camera **at original sensor resolution** |
| `sample_submission.csv` | the 10 test IDs, e.g. `HPC_Airbus BS6-3`, `HPC_Muenster_BS6_9_0-10m` |

Other files in the repo: `Overview.pdf`, `data.pdf`, `Competition Rules.htm` (screenshots/copies of the competition
pages), `codes/` (20 public notebooks used as reference), the public-leaderboard export.

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
python run.py --out ../submission.csv           # uses ../cache/features.csv (committed)
rm ../cache/features.csv && python run.py        # re-extract features (~6 min on 4 cores)
```
On Kaggle, attach the competition data and run `run.py --root /kaggle/input/soil-grain-size-from-photos`
(CPU only; no internet or pretrained weights needed).

## Submission log

| # | file | change | LOSO EMD | public LB |
|---|---|---|---|---|
| 1 | `submission.csv` | ensemble as described above | 35.9 | 55.19 |
| 2 | `submission_v2.csv` | `python run.py --range-margin 0.25`: only features whose test-sample values stay within the train range (+25 %) | 35.9 | 61.09 (worse) |

v1 scored far worse than CV (camera/site shift). v2 tested the hypothesis that shifted features are the cause: it
scored **worse** (61.09 vs 55.19), so range-filtering hurt. Side effect that likely explains it: the filter removes coarse-scale features, so the coarse Münster sample is predicted finer than in v1
(EMD v1↔v2 = 33 on that sample, 15 on Audorfring, 11 on Lidl; ≤6 elsewhere).

## Ideas not yet tried

* Pretrained embeddings (DINOv2 / ConvNeXt) as extra latent features – public notebooks with them score ~35–41 OOF, no
  better; needs internet/models (not available in the dev sandbox).
* Instance segmentation of stones (SAM / watershed) to measure coarse-fraction diameters directly in mm.
* Test-time adaptation: standardise iPhone features with unlabeled test statistics; per-phone feature alignment.
* Reduce variance of the tail: Bayesian shrinkage of predictions towards a site-type prior.
