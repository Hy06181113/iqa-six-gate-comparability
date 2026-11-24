# Code and data for *The Six-Gate Test*

Companion repository for

> **The Six-Gate Test: Deciding Whether Calibrated Image Quality Scores Can Be Substituted**
> submitted to the *Journal of Electronic Imaging*

It carries the material behind every number printed in the paper: the analysis scripts, the cached per-image scores of the nine open-weight models on the five benchmarks, the fold assignment of every image, the per-judge inference configuration, the derived tables, and the dated pre-registration and revision records.

**No image and no model weight is redistributed here.** The benchmarks and the pretrained weights are third-party materials used under their own licences; see `LICENSE-NOTES.md`.

## Contents

| Path | What it is |
|---|---|
| `code/` | 18 analysis scripts plus the calibration and aggregation code. Each script is named after what it computes |
| `scores/` | Cached per-image quality scores for the nine models on the five benchmarks |
| `folds/` | Fold assignment of every image, per dataset |
| `infer/configs/` | The experiment configurations the cached-score runs were made under |
| `infer/judges/` | Per-judge inference configuration and parsing rules |
| `tables/` | Derived per-distortion-type and per-direction tables |
| `record/` | Dated pre-registration and revision records |

## Reproducing a printed count

The scripts read the cached scores and the fold file; most of them need nothing else:

```
python3 code/diagnose_comparability.py --cfg infer/configs/20261030-p1diagnose-v3.yaml
```

**Two kinds of script behave differently on a fresh machine, and it is worth saying so up front.**

- Scripts that read only `scores/` and `folds/` run as they are.
- Scripts that also read the benchmark label files (mean opinion scores) expect the public benchmarks to be present at the paths written in the configuration. Those benchmarks are available from their official distributors; the files themselves are not included here for licensing reasons.

No GPU is required. The cached-score analyses complete on a CPU in seconds.

## Licence

The scripts in `code/` are released for research use. Third-party components they reference keep their own terms; `LICENSE-NOTES.md` sets out each one, including the non-commercial licence that governs part of the upstream IQA code path.

## Citing

If you use this material, please cite the paper above.
