# All twenty ordered transfer directions

The full version of the journal draft's Section VI table, which prints eight of the twenty. Rows are ordered by |ΔR_c|, largest first; the draft prints the seven largest and the smallest, and states that the remaining twelve lie between 0.8 and 12.5 ε.

**What each column is.** `OVL` is per unordered pair, so both directions of a pair share one value. `extrap mean` averages the nine judges after each judge's fold maximum; `extrap max` takes the maximum over folds and judges and names the judge that produced it; criterion 1a fails above 0.05. `ΔR_c` is the position-removed change of equation (6); the last column divides it by ε. `CI excl. 0` counts the 36 pairs × 6 non-`none` families = 216 cells per direction whose conditional interval excludes zero.

**Source.** `30-experiments/runs/20261023-p2-boot-5dom/` — `table_5_3.csv`, `ovl_5dom.csv`, `extrap_5dom.csv`. Cell-level records for all 5,040 cells are in `directions_cells_5040.csv`. Nothing is recomputed here.

| Fit → test | OVL | Extrap mean | Extrap max (judge) | 1a | ΔR_c | \|ΔR_c\|/ε | CI excl. 0 |
|---|---:|---:|---|:---:|---:|---:|---:|
| tid2013 → livechallenge | 0.6776 | 0.002 | 0.013 (maniqa) | pass | -0.602 | 29.9 | 210 |
| tid2013 → csiq | 0.4996 | 0.000 | 0.002 (iqa-t1) | pass | -0.575 | 28.6 | 213 |
| koniq → csiq | 0.5279 | 0.025 | 0.107 (q-align) | fail | -0.430 | 21.4 | 209 |
| koniq → livechallenge | 0.7240 | 0.001 | 0.009 (iqa-t1) | pass | -0.346 | 17.2 | 214 |
| livechallenge → tid2013 | 0.6776 | 0.025 | 0.094 (q-realign-mini) | fail | +0.295 | 14.7 | 207 |
| agiqa3k → csiq | 0.6295 | 0.015 | 0.088 (q-realign-mini) | fail | -0.289 | 14.3 | 208 |
| tid2013 → koniq | 0.8419 | 0.025 | 0.133 (q-align) | fail | -0.276 | 14.1 | 211 |
| csiq → tid2013 | 0.4996 | 0.056 | 0.261 (hyperiqa) | fail | +0.252 | 12.5 | 188 |
| agiqa3k → livechallenge | 0.8416 | 0.003 | 0.029 (clip-iqa) | pass | -0.226 | 11.2 | 210 |
| csiq → agiqa3k | 0.6295 | 0.078 | 0.303 (maniqa) | fail | +0.194 | 9.6 | 200 |
| tid2013 → agiqa3k | 0.7397 | 0.013 | 0.087 (maniqa) | fail | -0.189 | 9.4 | 209 |
| livechallenge → agiqa3k | 0.8416 | 0.043 | 0.109 (maniqa) | fail | +0.178 | 8.9 | 205 |
| agiqa3k → tid2013 | 0.7397 | 0.005 | 0.029 (q-realign-mini) | pass | +0.174 | 8.6 | 202 |
| koniq → tid2013 | 0.8419 | 0.008 | 0.027 (q-align) | pass | +0.137 | 6.8 | 204 |
| csiq → koniq | 0.5279 | 0.132 | 0.320 (topiq-nr) | fail | -0.097 | 5.0 | 191 |
| livechallenge → csiq | 0.6969 | 0.049 | 0.147 (q-realign-mini) | fail | -0.084 | 4.2 | 199 |
| agiqa3k → koniq | 0.7881 | 0.003 | 0.019 (q-align) | pass | -0.041 | 2.1 | 190 |
| csiq → livechallenge | 0.6969 | 0.019 | 0.059 (clip-iqa) | fail | +0.040 | 2.0 | 188 |
| livechallenge → koniq | 0.7240 | 0.040 | 0.125 (q-align) | fail | +0.016 | 0.8 | 185 |
| koniq → agiqa3k | 0.7881 | 0.006 | 0.022 (musiq) | pass | -0.009 | 0.4 | 201 |

