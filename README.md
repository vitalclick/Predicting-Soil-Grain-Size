# Predicting Soil Grain Size Distributions from Images

Solution for the Kaggle community competition
[Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos)
(Geotechnical Resilience project / BOKU): predict the cumulative grain-size distribution of a soil sample, as measured
by sieve and sedimentation analysis, from phone photographs of its surface.

**Result:** public leaderboard EMD **22.31**, down from 55.19 for the first submission (equal-share baseline ≈ 102,
lower is better). Uses only the competition data, CPU only, ~6 minutes end to end.

## The problem

* **Target:** percent of mass finer than 11 fixed diameters, `0.002 … 200 mm` (DIN EN ISO 14688-1).
* **Metric:** log-weighted Earth Mover's Distance, `Σ |F_i − F̂_i| · Δlog10(x)`, i.e. the area between the true and
  predicted grading curves on a log axis.
* **Data:** 24 labelled training samples (127 Android photos) and 10 test samples (35 iPhone photos from new sites).
  Tiny data with a camera *and* site shift between train and test.

## Key ideas

1. **Physical scale.** Every photo is resampled to 4 px/mm using the camera's pixels-per-mm. The provided training
   JPGs were downscaled by the host, so the published ppm table has to be rescaled by `actual_side / reference_side`.
2. **Metric-aligned target.** Log-weighted EMD equals the L1 distance between log-quantile functions, so the model
   regresses log-diameters at fixed percentiles. This is aligned with the metric and can extrapolate to coarser soils
   than any training sample.
3. **Honest validation.** Leave-one-*sample*-out CV. Random K-fold leaks, because photos of the same soil sample land
   in both folds.
4. **Camera-robust features.** Scale-space (LoG) energy, grey-scale granulometry and edge density in millimetres, with
   camera-style augmentation (blur, sharpening, noise, tone, vignette, JPEG). Features that shift between Android and
   iPhone photos are dropped.
5. **Shrink what the model cannot predict.** On training data the model's ranking of sandy samples by location and by
   curve shape is no better than noise (LOSO Spearman −0.22), so predictions within the sandy group are pooled toward
   the group median (validated on training data before it was ever submitted; the largest single gain).
6. **Group-level calibration.** The Android-trained model reads iPhone field photos as finer than they are. A small set
   of group-level offsets (sandy vs. gravelly samples, overall size and fine fraction) corrects this.

## Results

| stage | public EMD |
|---|---|
| equal-share baseline | ≈ 102 |
| first submission (PLS + kernel-median ensemble) | 55.19 |
| linear models only, which extrapolate to coarse soils | 40.90 |
| group-level size and fines calibration | 29.23 |
| sandy samples pooled toward the group median | 23.25 |
| + shared sandy curve shape | 22.89 |
| **final (v31)** | **22.31** |

Leave-one-sample-out EMD on the training set is ≈ 36–39 for the base models; the limit is the fine fraction
(clay/silt), which is nearly invisible in a surface photo. Every submission, its score and the reasoning behind it is
in [`docs/development_log.md`](docs/development_log.md).

## Repository layout

```
├── src/
│   ├── gsd.py            metric, curve <-> log-quantile transforms, data discovery
│   ├── features.py       physical-scale image features + camera augmentations
│   ├── run.py            leave-one-sample-out CV, calibration and submission writer
│   ├── experiments.py    CV harness for feature / target comparisons
│   └── granulometry.py   direct native-resolution granulometry (negative result, kept for reference)
├── cache/features.csv    pre-computed features (skips the ~6 min extraction)
├── submissions/          every submission file, v1 … v36
├── docs/development_log.md
├── data/                 competition data goes here (not included, see data/README.md)
├── requirements.txt
└── LICENSE
```

## Reproduce

```bash
pip install -r requirements.txt
# download the competition data into data/ (see data/README.md)
cd src
python run.py --cv-only                                     # leave-one-sample-out CV of the base models
python run.py --no-cv --linear-only --clip-z 0 \
              --shift-log10 0.20 --shift-coarse 0.35 \
              --shift-fines 0.65 --shift-fines-coarse 0.5 \
              --homogenize-sandy 0.65 --homogenize-shape 1.0 \
              --out ../submissions/submission_v31.csv       # final submission, byte-for-byte
```

Delete `cache/features.csv` to re-extract features from the photos. On Kaggle, attach the competition data and pass
`--root /kaggle/input/soil-grain-size-from-photos` (CPU only, no internet or pretrained weights needed).

## What did not work

* **Range-filtering shifted features** (61.09, worse than baseline): it also removed the real coarse-grain signal.
* **Contracting the predicted curve width** toward well-sorted training sands (32.15): the test sands are narrower
  than predicted, but not that narrow.
* **Direct granulometry at native iPhone resolution:** on the training set image-measured sizes do not correlate with
  sieve sizes (ρ ≈ 0.3 for gravels); bright-blob size is set by lighting, not grain outlines.

## Limitations

The public leaderboard covers about 3 test samples, and the group-level calibration was tuned on it. The pooling
steps were validated on training data, but how well the calibration transfers to the private samples is uncertain.
Leaderboard probing (reconstructing hidden labels from scores) was deliberately not used.

## Licence

Code released under the [MIT Licence](LICENSE). The competition data is not included and remains subject to the
competition's own terms.

## Acknowledgements

Competition data by Lukas Leibold and Enrico Soranzo,
[Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos),
Kaggle, 2026.
