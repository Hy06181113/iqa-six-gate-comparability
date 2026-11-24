# Licensing notes for this deposit

This file summarises what may be redistributed here and why. The item-by-item
record is `LICENSING.md`, which ships with the companion archive: for each
dataset and each judge it names the document that was read and the date it was
read. Two
standing rules apply throughout: **anything the official source does not declare
is treated as not redistributable**, and **licensing is judged on the copy we
actually loaded or copied**, not on what a paper or a README says about the
original release.

## The four object classes

| Class | In this deposit | Status |
|---|---|---|
| **Images** | **Absent.** No dataset image is redistributed here. | Per-dataset terms differ; several are research-use only. Obtain each dataset from its publisher. |
| **Labels / human scores** | **Derived scalars only** — per-image MOS and its uncertainty, in `scores/` and in the tables. | Redistribution of a derived per-image statistic is asserted as fair use of the underlying labels, **not** as a blanket licence grant. If a publisher disagrees, the affected column will be removed on request. |
| **Code** | **Our own scripts only**, in `code/` and `infer/`. No third-party tree is vendored. | Project-authored; released under the licence declared in `METADATA.md`. |
| **Models / weights** | **Absent.** No checkpoint is redistributed. `infer/` carries configuration and prompt text only. | Weights stay with their publishers. |

## The one restriction a reader must not miss

`pyiqa` — the reference implementation this project loads four judges through —
is **not** MIT. Its `LICENSE` is **PolyForm Noncommercial License 1.0.0**, pinned
at the vendored commit `6e48c5ca9e3e58bd632af52f06a2ca976ccf2722`. Four of the
nine judges in this study (MUSIQ, MANIQA, TOPIQ-NR, HyperIQA) run through
`pyiqa` code paths, and the weight repository
`chaofengc/IQA-PyTorch-Weights` is **CC BY-NC-SA 4.0** and self-describes as
mixed-origin. **This deposit therefore carries no commercial-use grant for
anything derived from those four judges**, and none of their weights are
included.

## What is deliberately not here

- **Dataset images.** TID2013, CSIQ, KonIQ-10k, LIVE Challenge and AGIQA-3K are
  obtained from their publishers. Only per-image model outputs and human scores
  travel with this deposit.
- **Model checkpoints.** Every judge is identified by the backbone and the
  checkpoint we actually loaded (see `infer/judges/*/README.md`); the weights
  themselves are re-obtainable from their sources.
- **Third-party source trees.** `pyiqa` and its dependencies are cited by commit
  and licence, never copied in.

## Third-party assets referenced, not redistributed

| Asset | Where it comes from | Licence as recorded |
|---|---|---|
| pyiqa | github.com/chaofengc/IQA-PyTorch @ `6e48c5ca…` | PolyForm Noncommercial 1.0.0 |
| IQA-PyTorch-Weights | github.com/chaofengc/IQA-PyTorch-Weights | CC BY-NC-SA 4.0 (mixed origin) |
| TID2013 | tampere.fi / published archive | research use; no standard SPDX identifier |
| CSIQ | vision.okstate.edu | research use; no standard SPDX identifier |
| KonIQ-10k | DARUS / DataCite record | per-image CC licences; remainder CC-BY 4.0 |
| LIVE Challenge | live.ece.utexas.edu | research use; no standard SPDX identifier |
| AGIQA-3K | published archive | research use; no standard SPDX identifier |

Seven of the datasets in the wider study carry **no standard SPDX identifier**.
"Free for research" is not the same as "redistributable", which is why the
images stay out of this deposit.
