# 24 distortion types on TID2013 — one judge pair, one family

The cell the journal draft reads in Section V-D: hyperiqa | topiq-nr under logistic5 on TID2013, the widest of the 252 decisions. Domain 01–24 are TID2013's own distortion types.

**Scope.** This file prints **one** of the 36 × 7 = 252 cells. The exhaustive per-type reading — every pair, every family — is `distortion_types_tid2013.csv` (6,048 rows). Nothing here is recomputed: both files are read straight out of `30-experiments/runs/20261030-p1diagnose-v3-tid2013/summary.json`, `datasets.tid2013.pairs[*][*].iso.bins_domain`.

`criterion 3` is the position **and** scale term both held inside ε on that type; `strict` additionally requires each judge's calibrated output within ε of the human score (P_strict). `sd(d)/ε` is the residual dispersion of that type in units of ε.

| Type | n | P | P strict | criterion 3 | mean d | sd(d)/ε |
|---|---:|---:|---:|:---:|---:|---:|
| 01 | 115 | 0.6696 | 0.1304 | fail | -0.0011 | 1.214 |
| 02 | 112 | 0.7500 | 0.0446 | fail | -0.0042 | 0.952 |
| 03 | 114 | 0.5526 | 0.0789 | fail | -0.0148 | 1.145 |
| 04 | 114 | 0.7105 | 0.0263 | fail | +0.0182 | 1.209 |
| 05 | 115 | 0.5478 | 0.0609 | fail | +0.0231 | 1.772 |
| 06 | 109 | 0.8991 | 0.0642 | fail | -0.0016 | 0.739 |
| 07 | 88 | 0.9091 | 0.0682 | pass | -0.0065 | 0.496 |
| 08 | 115 | 0.4957 | 0.1565 | fail | +0.0149 | 1.767 |
| 09 | 115 | 0.6783 | 0.0609 | fail | -0.0020 | 1.700 |
| 10 | 110 | 0.5636 | 0.0364 | fail | -0.0166 | 1.626 |
| 11 | 115 | 0.2261 | 0.0174 | fail | -0.0399 | 2.343 |
| 12 | 115 | 0.9391 | 0.0870 | pass | -0.0020 | 0.525 |
| 13 | 114 | 0.7544 | 0.0439 | fail | -0.0042 | 0.976 |
| 14 | 115 | 1.0000 | 0.0957 | pass | -0.0014 | 0.183 |
| 15 | 115 | 1.0000 | 0.0174 | pass | -0.0006 | 0.174 |
| 16 | 114 | 1.0000 | 0.1053 | pass | -0.0012 | 0.158 |
| 17 | 108 | 0.9907 | 0.0556 | pass | -0.0007 | 0.193 |
| 18 | 104 | 0.9712 | 0.1250 | pass | -0.0016 | 0.284 |
| 19 | 115 | 0.8435 | 0.1739 | fail | -0.0070 | 0.897 |
| 20 | 115 | 0.8522 | 0.0609 | fail | -0.0073 | 0.635 |
| 21 | 114 | 0.6228 | 0.1316 | fail | +0.0116 | 1.420 |
| 22 | 106 | 0.5094 | 0.0283 | fail | +0.0301 | 1.665 |
| 23 | 115 | 0.5391 | 0.0000 | fail | +0.0143 | 1.890 |
| 24 | 115 | 0.4783 | 0.0261 | fail | -0.0003 | 2.389 |

Seven types pass; across the other seventeen the proportion inside ε falls to 0.2261 at type 11 — the two numbers Section V-D prints. Every passing type has sd(d)/ε ≤ 0.525, and the largest |mean d| outside that group (0.0399) belongs to a type carrying sd(d)/ε = 2.343, which is why the section reads the split on the residual multiple rather than on the offset.

