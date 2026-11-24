# Dated record of pre-registration and revision

Everything below happened **before** the results it governs were read, unless the
entry says otherwise. This file exists because a deposit that shows only the
final state cannot distinguish a pre-registration from a post-hoc choice.

## How to read this file (added 2026-11-13, not part of the original record)

Ten entries follow, in date order. Two of them change what the paper prints and
were written before the numbers they govern were read; three move no number at
all, being the appearance of a figure or the wording of a description; the rest
are the provenance of this package. The table marks which is which, so that the
entries carrying evidentiary weight can be found without reading the rest.

| Entry | Category |
|---|---|
| 2026-10-16 criterion 3's residual gate moved out | defines when a reading was first produced |
| 2026-10-30 criterion 2's grid repaired | changes printed counts (0/32/220, 0/81/171) |
| 2026-11-03 the second tier returns as a claim | argument order only, numbers unchanged |
| 2026-11-03 ε's two sides reported together | adds one printed quantity |
| 2026-11-03 one factual correction to the transfer section | description only, no number moves |
| 2026-11-10 criterion 4's verdict rule changed | changes every criterion-4 verdict |
| 2026-11-10 Fig. 1, appearances only (three entries) | no number, statistic or verdict moves |
| 2026-11-12 this package and the deposit brought to one revision | provenance of this package |

Two notes on reading the entries as they stand. First, the two entries written
on 2026-11-10 say the compiled manuscript "stays at 13 pages"; that is the state
of the typeset draft on that date, not a reading, and the manuscript has since
been repaginated. Second, the 2026-11-12 entry counts five files that the deposit
carries and this subset does not, and names the licensing ledger separately in
the same sentence; the package README totals them as six. Both statements are
accurate as written, and no file moves between the two packages. Both notes are
left out of the entries themselves, on the same principle as everything else in
this file: what was written stays written.

## 2026-10-16 — criterion 3's residual gate moved out

The residual σ̂ was read as a third term of criterion 3 (calibrated agreement).
A one-sided reading carried no information across the 252 decisions, so σ̂ was
taken out of criterion 3. **It was not deleted**: it was kept, and re-introduced
later as an image-level gate of its own. Any reading of the paper that treats the
image-level gate as invented after seeing the results should note the date: the
statistic was defined and reported on 2026-10-16, only its position in the
verdict rule changed.

## 2026-10-30 — criterion 2's grid repaired, and the three-way counts changed

Criterion 2 (mapping validity) measured monotonicity and stability on a 1,001-point
grid. The grid was built once, over the **first** judge's observed range, and
used for both sides — so the second judge's map was evaluated outside its own
domain whenever the two judges' output ranges differed. Nine judges span three
ranges (`[0,1]`, `[1,5]`, `[0,100]`), so every cross-range pair was affected.

The repair changed *where* the statistic is measured, not how strictly: the
thresholds (0.10 for stability, a −1e-12 step tolerance) and the grid density
(1,001 points) were untouched. Consequences, all recomputed in a fresh run
(`runs/20261030-p1diagnose-v3-*`):

- criterion 2b fails nowhere, 252 of 252 pass, and **stops carrying any
  discrimination** — the paper keeps the column and says so;
- the three-way counts become 0 / 32 / 220 on TID2013 and 0 / 81 / 171 on
  KonIQ-10k, from 0 / 31 / 221 and 0 / 78 / 174;
- all 22 previous 2b failures were cross-range pairs; none were same-range.

The old counts are not reproduced here. This record is the deposit's only
revision log, and the counts the paper prints are the current ones.

## 2026-11-03 — the second tier returns as a claim

The paper's framing was changed from "comparability fails" to a two-tier claim:
distributional comparability (necessary) does not entail interchangeability
(sufficient). The image-level residual gate is what separates them. This is an
ordering of the argument, not a new computation: the gate's readings were first
produced on 2026-10-16 (above), and the numbers are unchanged.

## 2026-11-03 — ε's two sides reported together

TID2013's per-image median σ_human was reported; KonIQ-10k's was not. Both are
now printed, with the argument that they are the same functional — the standard
error of a per-image mean estimate, rescaled by the range that defines `y`. The
script `code/sigma_human_median.py` recomputes both from the published label
files and refuses to print if either ε drifts from the value printed in the
paper.

## 2026-11-03 — one factual correction to the transfer section

The paper described R_dev,c's denominator as an error measured "inside the fit
domain". The code measures it on the **test domain**, refitting both maps there
(`p2_transfer.py`, and the two prediction calls at its lines 535–551). The text
now says so. This is a correction of a description, not of a number: no value in
the tables changes.

## 2026-11-10 — criterion 4's verdict rule changed, and this package was repacked

Criterion 4 (no significant parameter drift across distortion types) was read
against its permutation null: fail if `p < α`. Two things were wrong with that.
The `p` carried no effect-size floor, so a synthetic positive control that is an
affine copy of the other side, `B := 3.7A − 0.42`, was refused at `|δ̂|/ε` of
4.5e-07 and 2.4e-09 — it was measuring permutation count, not drift. And at
`B = 20,000` the resolution floor `1/20001` sat below any Bonferroni line, so all
252 decisions landed on it and were indistinguishable from each other.

The rule is now the amplitude test `M ≤ ε`, with `p` still computed and still
written to `detail.p_value`, but no longer part of the verdict. The cost is
recorded where the rule is: the max-null controlled 24-type multiplicity and the
amplitude test has no counterpart to it, so the family-wise guarantee is gone
and only "M is itself the maximum over 24 types" remains.

The 2026-11-09 revision touched four programs:
`comparability_criteria.py` (`criterion4` gains `eps`), `diagnose_comparability.py`
(ε taken before the call and passed in), `eps_sensitivity.py` (the scan no longer
reads the stored `p` verdict, and `s*` becomes defined at 1.0555 on TID2013) and
`plot_conf.py` (annotation sizes only). This package had been packed on
2026-11-05, from before that revision, so its `eps_sensitivity.py` printed
"`s*` undefined" while the paper prints `s* = 1.0555`. It now carries the same
revisions the paper's numbers were produced with. The two configurations behind
the 20,000-permutation run and the block permutation stay with the deposit
rather than here, for the reason recorded when they were added to it: they set
no inference or scoring parameter.

No number in the paper changes as a result: the substantive readings were
already produced under the new rule, and the `s = 1` grid point is the one at
which the amplitude rule and the old `p` rule agree decision for decision on all
504 cells.

## 2026-11-10 — Fig. 1's palette, and nothing else

`code/plot_conf.py` was revised once more after the entry above. `fig_structure()`
now draws its five stage boxes on a five-step light ramp that deepens left to
right, so the shade carries a box's position on the chain and not its
importance; the body lines under each stage name begin with a capital, which
separates them from the bold stage names by case as well as by weight and size;
and the horizontal padding went from 6.0 to 5.5 pt because the capitals widened
each box by a few points and the script refuses to print a figure whose
inter-box gap falls below 15 pt (it was 14.4 pt at the old padding).

This is a change of appearance only. No number, statistic, threshold or verdict
in the paper moves, no gate in this file's code path is touched, and the figure
caption is unchanged: Fig. 1 carries no quantity by design, and this entry
records the restyle so that a reader comparing this package against the paper's
Fig. 1 does not read the difference as drift in the science.

## 2026-11-10 — Fig. 1 rebuilt as two bands

`code/plot_conf.py` was revised again. `fig_structure()` no longer draws a single
row of five stage boxes. It now draws two labelled bands: the upper band is the
input any comparability check needs (nine open-weight judges with their four
output mechanisms and three score ranges, 36 unordered pairs, seven calibration
families, 252 decisions per judge set), and the lower band is what the paper adds
(four criteria as gates, each with its own threshold, and a three-way verdict,
each verdict carrying the condition that produces it). The earlier version named
the stages without saying what any of them held, and a structure figure that
carries no content in a paper whose contribution is the structure is not worth
its space.

Three errors in an earlier sketch of this figure were corrected while it was
redrawn: the not-comparable count is 220 on TID2013, not 32 (32 is the number of
decisions that pass every gate and are stopped by criterion 4, hence
conditionally comparable); a decision whose drift is measured and found is
conditionally comparable, not not comparable; and the range 1.28 to 11.19 epsilon
belongs to TID2013 alone, KonIQ-10k running 1.46 to 9.67.

As before, this is a change of appearance only. No number in the paper moves and
no gate in this file's code path is touched. The figure still carries no result
count: the 252 to 32 to 220 funnel belongs to Fig. 2, and the caption says so in
one clause. The figure is now 174.96 pt tall against 82.08 pt before, and the
caption was cut from six lines to three so that the compiled manuscript stays at
13 pages.

## 2026-11-10 — Fig. 1 set at one body size

A fourth pass over `code/plot_conf.py`, again appearance only. The two-band layout
had left one inconsistency: gate 3 is the only gate whose threshold carries
mathtext subscripts (`y_95`, `y_05` in its second line), and mathtext sets
subscripts at 0.7 of the base size, so a 7.0 pt base gave 4.9 pt subscripts, below
the 5 pt floor this repository's text audit enforces. The earlier fix raised that
one line to 7.2 pt, which put its subscripts at 5.04 pt but also put two sizes in
one card, 7.2 pt against 7.0 pt.

The ratio itself is now raised instead, to 0.72, scoped to `fig_structure()`. A
7.0 pt base gives 5.04 pt subscripts and the gate 3 line returns to 7.0 pt, so the
figure body is one size from band label to verdict. The embedded font inventory is
now 5.04 / 6.4 / 7.0 / 7.8 pt and nothing else. This leans on a private matplotlib
constant; if a future release renames it the line fails silently and the
subscripts fall back to 4.9 pt, which is exactly what the 5 pt audit gate reports.
That gate is the safety net for this change and must not be relaxed to make an
audit pass.

The verdict cards also gained 1.0 pt of separation between each verdict name and
the condition that produces it, from 1.5 to 2.5 pt, by moving both 0.5 pt toward
their own card edges: names sit 1.08 pt below the card top and conditions 1.38 pt
above the card bottom. Card height and figure height are unchanged, the canvas is
still 515.52 by 174.96 pt, and the compiled manuscript is still 13 pages.

One detail was examined and deliberately left alone. The hat on delta-hat and
beta-hat sits about a point above the glyph rather than touching it, because
matplotlib places an accent using the font's own accent offset and exposes no knob
for it. A Unicode combining circumflex was tested as a substitute and strikes
through the delta instead of sitting on it, which reads worse than the gap. The
mathtext form stays.

## 2026-11-12 — this package and the deposit brought to one revision

The two had diverged: this package carried the 2026-11-09 change and the deposit
did not, and comment-level edits made since then existed on the repository side
only. Both were resynced to the project's current revision on 2026-11-12, so the
seventeen programs here, the nine judge wrappers and the four configurations are
the files the deposit now carries as well.

Two programs were dropped from both at the same time, `p2_boot_summary.py` and
`p2_ch5_5dom_c.py`. They aggregate the thesis's chapter-level tables — a
three-domain bootstrap summary and the per-family, per-reading tables of that
chapter — and the paper prints nothing they compute. Everything else stays: the
twenty-direction table's OVL and extrapolation columns are built from
`p2_ch5_boot5.py`, and the OVL definition of the paper's equation 2 is
`derive_ovl()` in `p2_ch5_derive.py`.

Five programs were added to both packages the same day, `c4_global.py`,
`selfpair_control.py`, `within_domain_residual.py`, `l4_baseline_table.py` and
`l4_validate.py`. The availability statement says this material carries the code
behind every verdict and count so that a printed count can be recomputed, and
these compute readings the paper does print: the amplitude intervals with the
per-domain counts, the self-pair control, the within-domain decomposition of the
residual, and the L4 matrix behind Table I. `l4_validate.py` is there because
`l4_baseline_table.py` imports `plcc_logistic` from it. They enter with their
internal pointers removed and nothing else changed.

The code in `code/` can now be run against the table this package carries.
Running it here used to fail twice over: the relative-path root resolved one
level above the package (`pred_table.REPO_ROOT` was "two directories up from the
script", which is the repository root there and this package's parent here), and
the experiment configurations ask for the publishers' label columns and eighteen
per-judge prediction files, neither of which travels with this package. The
loader in `pred_table.py` now finds its root by marker and can build its table
from `scores/p2_<dataset>.csv`, and `infer/configs/20261112-reader-c4global.yaml`
is a reader's entry point that uses it. Nothing is recomputed by the new path:
every judge column in the merged table is that judge's cached score and `y` is
the same mapped human scale, so the branch swaps I/O only. It was checked by
rerunning `c4_global.py` through it on all 36 pairs from inside this package:
every verdict and every count matches the run the paper's counts come from (36 of
36 on the p floor, the same 4,750 and 6,048, the same 250 of 252 interval lower
bounds above epsilon), and a pair run on the original label path returns the same
amplitude to the last digit. The products are not byte-identical, and the gap is
bounded here rather than left to be found: `y` is stored in `scores/` to ten
decimal places where the label path carries a double, so amplitudes agree to 6e-6
at worst and bootstrap bounds to 2e-5, the largest differences falling on the
five-parameter logistic mappings. Criterion 3 is the exception and stays one: it
needs sigma_human, a per-image column the merged table does not carry.

Nothing was recomputed and no number moves. What remains different between the
two packages is what each README says: the deposit carries five files this
subset does not, this subset carries no figure-drawing program, and the deposit's
`record/` holds the licensing ledger.
