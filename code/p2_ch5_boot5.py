#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第五章（P2）5.1.4／5.2 的派生件：五域 `ΔR_c` 条件区间 ＋ `OVL` ＋ 外推。

本脚本**不产生新实验**，只把 `20261023-p2-boot-5dom/` 的 5,040 格重采样与
五个标签 CSV 汇总成正文可引用的四份表。纯 CPU，数秒跑完，不读图像、不重拟合。

════════════════════════ 预注册（先写定义，再跑） ════════════════════════

【派生件 1】table_5_3.csv —— 20 个有序方向 × 4 列
  真源 `runs/20261023-p2-boot-5dom/boot_cells.csv`（5,040 行）。
  口径与 `p2_ch5_boot4.py` 逐项相同：方向级 `ΔR_c` = 六族中位再取中位；
  「不含 0」按 216 个非 `none` 格计。区间是**条件** bootstrap（固定已拟合映射、
  只重采样评测图），是**偏窄下界**，不得读作 95% 置信区间。

【派生件 2】ovl_5dom.csv —— 10 个无序域对
  公式、bin 数、阈值与 `p2_ch5_derive.py` 的 `derive_ovl()` 逐项相同：
  `OVL = Σ min(p_A, p_B)`，在已归一化的 `y` 上做 20 等宽 bin，阈值 0.50。
  另给 10/20/40/100 四个 bin 数下的取值与判定，用于「是否贴着阈值」的敏感性。

【派生件 3】extrap_5dom.csv —— 20 个方向 × 9 个评委 = 180 行
  判据 1a：对拟合域第 f 折之外的图取该评委的分数范围 [lo, hi]，统计测试域
  落在其外的比例；逐折算后取均值与最大值。阈值 `c1_extrapolation_max = 0.05`。
  **不得**把各折校准集并起来算（并起来支撑退化，比例恒近 0）。

【summary.json（ch5_boot5_summary.json）】同一批数的汇总与四道闸：
  ① `none` 族 `ΔR_c` 的最大绝对值（应为浮点零）
  ② `lo ≤ med ≤ hi` 的违反数
  ③ `ovl_5dom.csv` 的 6 个域对与 `20261022-p2-boot-4dom/ovl_4dom.csv` 逐位比对
     （3 个三域域对另与 `20261021-p2-ch5-r2/ovl.csv` 逐位比对）
  ④ 点值：`boot_cells.csv` 的 `abs_diff_R_dev` / `abs_diff_R_pair` 最大值（对 5 域归档）

@created: 2026-10-23
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, "runs")

OUT_DIR = os.path.join(RUNS, "20261023-p2-boot-5dom")
BOOT_CELLS = os.path.join(OUT_DIR, "boot_cells.csv")
OVL4_CSV = os.path.join(RUNS, "20261022-p2-boot-4dom", "ovl_4dom.csv")
OVL3_CSV = os.path.join(RUNS, "20261021-p2-ch5-r2", "ovl.csv")

DOMAINS = {
    "tid2013": r"D:/datasets/tid2013/labels/p2_tid2013.csv",
    "koniq": r"D:/datasets/koniq10k/labels/p2_koniq.csv",
    "agiqa3k": r"D:/datasets/agiqa3k/labels/p2_agiqa3k.csv",
    "csiq": r"D:/datasets/csiq/labels/p2_csiq.csv",
    "livechallenge": r"D:/datasets/live_challenge/labels/p2_livechallenge.csv",
}
FAMILIES = ["none", "minmax", "zscore", "quantile", "platt", "logistic5", "isotonic"]
FAMS_RATED = [f for f in FAMILIES if f != "none"]

N_BINS = 20
BIN_SENSITIVITY = (10, 20, 40, 100)
OVL_MIN = 0.50
EXTRAP_MAX = 0.05
EPS = 1e-12
EXPECT_CELLS = 5040


def _f(x):
    return float(x)


def _sha8(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:8]


def load_cells(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def directions(rows):
    seen = []
    for r in rows:
        k = (r["fit_ds"], r["test_ds"])
        if k not in seen:
            seen.append(k)
    return seen


# ──────────────────────── 1. table_5_3 ────────────────────────
def derive_table_5_3(rows):
    out = []
    for fa, fb in directions(rows):
        fam_med, width_all = {}, []
        for fam in FAMS_RATED:
            sel = [r for r in rows if r["fit_ds"] == fa and r["test_ds"] == fb
                   and r["mapping"] == fam]
            if len(sel) != 36:
                raise SystemExit("%s→%s %s 行数 %d，期望 36" % (fa, fb, fam, len(sel)))
            fam_med[fam] = float(np.median([_f(r["deltaR_c"]) for r in sel]))
            width_all.extend((_f(r["deltaR_c_hi"]) - _f(r["deltaR_c_lo"])) for r in sel)
        sel_all = [r for r in rows if r["fit_ds"] == fa and r["test_ds"] == fb
                   and r["mapping"] != "none"]
        out.append({
            "fit_ds": fa, "test_ds": fb,
            "deltaR_c_med": "%.17g" % float(np.median(list(fam_med.values()))),
            "n_ci_excl_0": sum(1 for r in sel_all if r["deltaR_c_ci_contains_0"] == "False"),
            "ci_width_med": "%.17g" % float(np.median(width_all)),
        })
    return out


# ──────────────────────── 2. OVL ────────────────────────
def _load_y(name):
    with open(DOMAINS[name], encoding="utf-8") as fh:
        return np.array([float(r["y"]) for r in csv.DictReader(fh)])


def _ovl_at(ya, yb, n_bins):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    pa, _ = np.histogram(np.clip(ya, 0.0, 1.0), bins=edges)
    pb, _ = np.histogram(np.clip(yb, 0.0, 1.0), bins=edges)
    pa = pa / max(pa.sum(), 1)
    pb = pb / max(pb.sum(), 1)
    return float(np.minimum(pa, pb).sum())


def derive_ovl():
    ys = {k: _load_y(k) for k in DOMAINS}
    out = []
    keys = sorted(DOMAINS)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ovl = _ovl_at(ys[a], ys[b], N_BINS)
            sens = {str(n): "%.17g" % _ovl_at(ys[a], ys[b], n) for n in BIN_SENSITIVITY}
            verdicts = {n: ("fail" if _ovl_at(ys[a], ys[b], n) < OVL_MIN else "pass")
                        for n in BIN_SENSITIVITY}
            out.append({
                "ds_a": a, "ds_b": b, "n_bins": N_BINS,
                "n_a": len(ys[a]), "n_b": len(ys[b]),
                "ovl": "%.17g" % ovl,
                "threshold": OVL_MIN,
                "verdict": "fail" if ovl < OVL_MIN else "pass",
                "ovl_bins10": sens["10"], "ovl_bins40": sens["40"],
                "ovl_bins100": sens["100"],
                "verdict_stable_10_100": "yes" if len(set(verdicts.values())) == 1 else "no",
            })
    return out


# ──────────────────────── 3. 外推 1a ────────────────────────
def _load_domain(name):
    with open(DOMAINS[name], encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    judges = [c for c in rows[0] if c not in ("image", "y", "y_native", "fold")]
    return rows, judges


def derive_extrap():
    data = {k: _load_domain(k) for k in DOMAINS}
    judges = data["tid2013"][1]
    if len(judges) != 9:
        raise SystemExit("评委列数 %d，期望 9" % len(judges))
    for k, (_, js) in data.items():
        if js != judges:
            raise SystemExit("%s 的评委列与 tid2013 不一致" % k)
    folds = sorted({int(r["fold"]) for r in data["tid2013"][0]})
    if len(folds) != 5:
        raise SystemExit("折数 %d，期望 5" % len(folds))

    out = []
    for fa in DOMAINS:
        for fb in DOMAINS:
            if fa == fb:
                continue
            ra = data[fa][0]
            rb = data[fb][0]
            fa_fold = np.array([int(r["fold"]) for r in ra])
            for j in judges:
                xa = np.array([float(r[j]) for r in ra])
                xb = np.array([float(r[j]) for r in rb])
                per_fold = []
                for f in folds:
                    cal = xa[fa_fold != f]
                    if len(cal) == 0:
                        raise SystemExit("%s 第 %d 折无校准图" % (fa, f))
                    lo, hi = float(cal.min()), float(cal.max())
                    per_fold.append(float(np.mean((xb < lo) | (xb > hi))))
                out.append({
                    "fit_ds": fa, "test_ds": fb, "judge": j,
                    "n_fit": len(xa), "n_test": len(xb),
                    "mean_over_folds": "%.17g" % float(np.mean(per_fold)),
                    "max_over_folds": "%.17g" % float(np.max(per_fold)),
                    "per_fold": ";".join("%.6g" % v for v in per_fold),
                })
    return out


def extrap_by_direction(ext):
    out = []
    for fa in DOMAINS:
        for fb in DOMAINS:
            if fa == fb:
                continue
            sel = [r for r in ext if r["fit_ds"] == fa and r["test_ds"] == fb]
            mx = max(sel, key=lambda r: _f(r["max_over_folds"]))
            v = _f(mx["max_over_folds"])
            out.append({
                "fit_ds": fa, "test_ds": fb, "n_judges": len(sel),
                "mean_of_means": "%.17g" % float(np.mean([_f(r["mean_over_folds"]) for r in sel])),
                "max_over_judges": "%.17g" % v,
                "judge_at_max": mx["judge"],
                "verdict": "fail" if v > EXTRAP_MAX else "pass",
            })
    return out


# ──────────────────────── 4. 自证闸 ────────────────────────
def check_none(rows):
    v = [_f(r["deltaR_c"]) for r in rows if r["mapping"] == "none"]
    return {"n_cells": len(v), "max_abs": max(abs(x) for x in v)}


def check_order(rows):
    bad = 0
    for r in rows:
        lo, med, hi = _f(r["deltaR_c_lo"]), _f(r["deltaR_c_med"]), _f(r["deltaR_c_hi"])
        if not (lo <= med <= hi):
            bad += 1
    return {"n_rows": len(rows), "n_violation": bad}


def check_point_values(rows):
    d = max(max(_f(r["abs_diff_R_dev"]), _f(r["abs_diff_R_pair"])) for r in rows)
    return {"n_rows": len(rows), "max_abs_diff_vs_archive": d}


def check_ovl(ovl):
    def _cmp(path):
        old = {frozenset((r["ds_a"], r["ds_b"])): r["ovl"] for r in
               csv.DictReader(open(path, encoding="utf-8", newline=""))}
        n, bad = 0, 0
        for r in ovl:
            o = old.get(frozenset((r["ds_a"], r["ds_b"])))
            if o is None:
                continue
            n += 1
            if r["ovl"] != o:
                bad += 1
        return {"ref": path, "n_compared": n, "n_mismatch": bad}
    return {"vs_ovl_4dom": _cmp(OVL4_CSV), "vs_ovl_3dom": _cmp(OVL3_CSV)}


def coverage(rows):
    def _grp(pred):
        sub = [r for r in rows if pred(r) and r["mapping"] != "none"]
        excl = sum(1 for r in sub if r["deltaR_c_ci_contains_0"] == "False")
        return {"n_cells": len(sub), "n_ci_excl_0": excl,
                "share_ci_excl_0": (excl / len(sub)) if sub else None,
                "abs_deltaR_c_med": float(np.median([abs(_f(r["deltaR_c"])) for r in sub])),
                "ci_width_med": float(np.median([_f(r["deltaR_c_hi"]) - _f(r["deltaR_c_lo"])
                                                 for r in sub]))}
    def _has(r, d):
        return d in (r["fit_ds"], r["test_ds"])
    return {
        "all": _grp(lambda r: True),
        "three_dom": _grp(lambda r: not _has(r, "csiq") and not _has(r, "livechallenge")),
        "four_dom_no_live": _grp(lambda r: not _has(r, "livechallenge")),
        "live_involved": _grp(lambda r: _has(r, "livechallenge")),
        "live_as_fit": _grp(lambda r: r["fit_ds"] == "livechallenge"),
        "live_as_test": _grp(lambda r: r["test_ds"] == "livechallenge"),
        "csiq_involved": _grp(lambda r: _has(r, "csiq")),
        "csiq_as_fit": _grp(lambda r: r["fit_ds"] == "csiq"),
        "csiq_as_test": _grp(lambda r: r["test_ds"] == "csiq"),
    }


def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="第五章：五域 ΔR_c 条件区间、OVL 与外推")
    ap.add_argument("--boot-cells", default=BOOT_CELLS)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()

    rows = load_cells(a.boot_cells)
    if len(rows) != EXPECT_CELLS:
        raise SystemExit("boot_cells 行数 %d，期望 %d" % (len(rows), EXPECT_CELLS))
    if any(r["status"] != "ok" for r in rows):
        raise SystemExit("存在 status != ok 的格")

    t53 = derive_table_5_3(rows)
    ovl = derive_ovl()
    ext = derive_extrap()
    ext_dir = extrap_by_direction(ext)
    checks = {
        "none_deltaR_c": check_none(rows),
        "interval_order": check_order(rows),
        "point_values": check_point_values(rows),
        "ovl_identical": check_ovl(ovl),
    }
    s = {
        "generated": "2026-10-23",
        "batch": "2026-10-23（五域条件 bootstrap）",
        "inputs": {a.boot_cells: _sha8(a.boot_cells),
                   OVL4_CSV: _sha8(OVL4_CSV), OVL3_CSV: _sha8(OVL3_CSV),
                   **{DOMAINS[k]: _sha8(DOMAINS[k]) for k in DOMAINS}},
        "definitions": {
            "deltaR_c": "R_pair_c − R_dev_c（同一评委对配对作差），方向级 = 六族中位再取中位",
            "interval": "图像级**条件** bootstrap（B=2000，按折分层，固定已拟合映射），"
                        "偏窄下界，不得读作 95% 置信区间",
            "ovl": "Σ min(p_A, p_B)，归一化 y 上 %d 等宽 bin，阈值 %.2f（与 P1 判据 1b 同式）"
                   % (N_BINS, OVL_MIN),
            "extrap": "判据 1a：逐折取拟合域分数范围，统计测试域落在其外的比例；"
                      "方向级取跨折 × 跨评委最大值，阈值 %.2f" % EXTRAP_MAX,
        },
        "checks": checks,
        "coverage": coverage(rows),
        "ovl": ovl,
        "extrap_by_direction": ext_dir,
    }
    os.makedirs(a.out, exist_ok=True)
    _write_csv(os.path.join(a.out, "table_5_3.csv"), t53)
    _write_csv(os.path.join(a.out, "ovl_5dom.csv"), ovl)
    _write_csv(os.path.join(a.out, "extrap_5dom.csv"), ext)
    with open(os.path.join(a.out, "ch5_boot5_summary.json"), "w",
              encoding="utf-8", newline="") as fh:
        json.dump(s, fh, ensure_ascii=False, indent=1)
        fh.write("\n")

    print("table_5_3 (%d 方向):" % len(t53))
    for r in t53:
        print("  %-14s→%-14s  ΔR_c_med=%+.4f  ci_excl_0=%d/216  width=%.4f"
              % (r["fit_ds"], r["test_ds"], _f(r["deltaR_c_med"]),
                 r["n_ci_excl_0"], _f(r["ci_width_med"])))
    print("ovl (%d 域对):" % len(ovl))
    for r in ovl:
        print("  %-14s %-14s OVL=%s %s  bins10/40/100=%s/%s/%s stable=%s"
              % (r["ds_a"], r["ds_b"], r["ovl"], r["verdict"],
                 r["ovl_bins10"], r["ovl_bins40"], r["ovl_bins100"],
                 r["verdict_stable_10_100"]))
    print("extrap (%d 方向):" % len(ext_dir))
    for r in ext_dir:
        print("  %-14s→%-14s mean=%.4f max=%.4f (%s) %s"
              % (r["fit_ds"], r["test_ds"], _f(r["mean_of_means"]),
                 _f(r["max_over_judges"]), r["judge_at_max"], r["verdict"]))
    print("checks:", json.dumps(checks, ensure_ascii=False))


if __name__ == "__main__":
    main()
