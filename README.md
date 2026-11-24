# Supplementary material

Companion to *The Six-Gate Test: Deciding Whether Calibrated Image Quality
Scores Can Be Substituted* (Journal of Electronic Imaging submission).

This material implements the paper's *Data and code availability* statement: it
carries the cached scores behind every number, the inference configuration,
prompt and parser behind each judge, the experiment configurations with their
dates, the calibration and aggregation code, and the dated revision record of
criterion 3's rule. Every number in the paper can be re-derived from it on a
CPU, in seconds. Nothing here needs a GPU, a model weight or a dataset image.

59 files. `SUPPLEMENTARY-SHA256SUMS.txt`, one level up, lists the digest of each.

## Relationship to the archive that will follow acceptance

If the paper is accepted, the same tree, extended with that deposit's own
publication metadata, will be placed in a public archive. Every file here
appears there at the same relative path and at the same revision: the deposit
was repacked from this one on 2026-11-12, so its code, judge wrappers and
configurations are these files.

What is absent is 6 files: the deposit's `METADATA.md` and
`UPLOAD-CHECKLIST.md`, the figure-drawing program `code/plot_conf.py` (whose
absence the next section explains), the item-by-item licensing record
`LICENSING.md`, and two further configurations under `infer/configs/`
(`20261105-p1diagnose-c4b20000.yaml`, `20261105-p1diagnose-c4block.yaml`). Those
two set no inference and no scoring parameter: they repackage the p1diagnose
configuration at a larger permutation count and with a block permutation, which
is why they sit with the deposit's records rather than with the four
configurations the cached-score runs were made under.

## What is not here, and why

The paper's figure-drawing program is not in this subset. The availability
statement covers the calibration and aggregation code **from which every verdict
and count is computed**; drawing computes neither, and the figures themselves
travel with the manuscript. The log in `record/` still records three revisions to
that program, because the log is a complete record of the project rather than a
subset of this one.

Two programs that build the *thesis's* chapter-level tables are not in this
subset either: `p2_boot_summary.py`, which summarises a three-domain bootstrap,
and `p2_ch5_5dom_c.py`, which produces the per-family and per-reading tables of
that chapter. The test applied to every file here is whether the paper prints a
number it computes, or needs it to re-derive one; those two fail it, and neither
feeds a table in this archive. Two comments elsewhere still name them: in
`code/p2_boot_img.py`, the line saying which program summarises its output, and
in `code/p2_ch5_derive.py`, the line that shares a floating-point-zero tolerance
with `p2_boot_summary.py`. Those comments are left as written, because these
files otherwise match the archive byte for byte.

## Layout

| Path | What it holds |
|---|---|
| `scores/` | 5 files — per-image judge scores and human MOS, one CSV per dataset: `image, y, y_native, fold`, then nine columns of cached model outputs. `y` is the human score mapped onto the common scale by `y = (s - s_min)/(s_max - s_min)`; `y_native` the raw score; `fold` the held-out assignment. This is the input to every criterion. |
| `folds/` | 5 files — the fold definitions (`image, fold`), the same split the paper reads, so the verdicts can be reproduced rather than merely re-read. |
| `infer/` | 9 `judges/<judge>/infer.py` scripts that produced the cached columns, their notes recording backbone and checkpoint, and 5 configurations: the four experiments with their seeds, plus a reader's entry point that reads the merged table in `scores/` instead of the per-judge prediction files. No weights. |
| `code/` | 17 files — the calibration, criteria and aggregation code used for the paper: the gate implementation, the transfer program, the bootstrap and the self-pair control, the split-half calibration of epsilon, the L4 baseline table behind Table I, the programs behind the twenty-direction table, and the two that report the amplitudes' intervals and the within-domain decomposition of the residual. The loader in `pred_table.py` reads either the per-judge prediction files or the merged table in `scores/`, so the table this package carries can drive the code it carries. |
| `tables/` | 5 files — the two tables the paper could not carry: the 24-distortion-type breakdown on TID2013 (6,048 rows) and the complete twenty-direction cross-domain transfer table, each with a readable Markdown extract. |
| `record/CHANGELOG.md` | The dated record of pre-registration and revision, including the revision history of criterion 3's rule that the paper points at. |
| `LICENSE-NOTES.md` | What is and is not redistributable here, and why. |

## Reproducing the main numbers

Everything the paper counts from the cached scores can be recomputed from
`scores/` on a CPU. Two entry points, and one boundary worth stating plainly.

1. **The global homogeneity test.** `code/c4_global.py` run against
   `infer/configs/20261112-reader-c4global.yaml` rebuilds the pair-level test of
   Section V from `scores/p2_tid2013.csv`: 36 pairs × 7 families × 24 domains,
   one permutation per pair acting on all families at once, then 1,000
   image-level resamples per cell. It prints the readings the paper prints — the
   tightest amplitude at 1.0555 epsilon, the interval's lower bound above epsilon
   on 250 of 252 decisions, 4,750 intervals excluding zero and 3,680 amplitudes
   beyond epsilon among 6,048 domain rows — and writes `c4_global.json` and
   `c4_domain_ci.csv`. About 22 minutes; `--limit-pairs 2 --boot 20` is a
   two-minute check that it runs:

       python -X utf8 code/c4_global.py \
           --cfg infer/configs/20261112-reader-c4global.yaml \
           --out runs/reader-c4global

2. **The four criteria.** `code/comparability_criteria.py` applies them: the six
   distributional gate keys of Table II, then the image-level residual gate read
   separately. Criteria 1, 2 and 4 need only the table in `scores/`. Criterion 3
   needs sigma_human, a per-image column the merged table does not carry: it is
   computed from the publishers' label files, which is the one step this package
   cannot do for you.
3. `code/p2_transfer.py` produces the cross-domain readings in `tables/`.
4. `code/sigma_human_median.py` prints the per-image median of the human-side
   uncertainty on each side, the two numbers that calibrate epsilon. Like
   criterion 3 it reads the published label files rather than a cached copy.

The reader configuration is **not** the configuration the paper's numbers were
produced from. The original reads 18 per-judge prediction files and the
publishers' label columns; the reader one reads the merged table. `pred_table.py`
chooses between the two on a single `merged_csv` key and computes nothing
differently. Every reading above reproduces from the merged table: all 36 pairs
clear the Bonferroni line and sit on the p floor, the exceed histogram is 36
zeros, 4,750 intervals exclude zero, 6,048 domain rows are written, and the
interval's lower bound clears epsilon on 250 of 252 decisions — the two
exceptions being the isotonic and logistic5 mappings of `deqa-score | q-align`,
1.1e-3 and 6.6e-4 below it.

The two paths agree to rounding, not bit for bit, and the size of that
disagreement is worth knowing before you compare numbers. The nine judge columns
are exact: the cached prediction files store their scores at six decimals and
`scores/` carries those same values. `y` is not — the merged table writes it to
ten decimal places, while the label path carries it as a double — and the
five-parameter logistic fits amplify that. Pair-level amplitudes agree to 3e-12,
per-domain amplitudes to 6e-6, and bootstrap bounds to 2e-5. Every count above is
identical, and the two decisions that fall short of epsilon miss it by tens of
times that spread.

## What is not reproducible from this material alone

- The per-image inference itself: `infer/` gives the configuration and prompt,
  and the weights are public, but neither images nor weights travel here.
- The calibration of epsilon from the publishers' label columns, which needs
  `mos_std.txt` (TID2013) and the per-image rating counts (KonIQ-10k).

## Language

Comments in the code, the per-judge notes and the experiment configurations are
in Chinese, the working language of the group. The prose files — this README,
the changelog and the licence notes — are in English, and the code uses English
identifiers throughout.

## Licence

See `LICENSE-NOTES.md`. The one restriction a reader must not miss: the pyiqa
code path and the weights it loads are under PolyForm Noncommercial 1.0.0, so
no commercial-use grant extends to anything derived from MUSIQ, MANIQA,
TOPIQ-NR or HyperIQA. No dataset images and no model weights are included.
