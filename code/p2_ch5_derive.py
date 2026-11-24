#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第五章（P2）正文的派生件：42 格聚合表、跨域 OVL、跨域外推比例。

本脚本**不产生新实验**，只把已有产物重新汇总成论文表格所需的三份 CSV。
纯 CPU，几分钟内跑完，不读图像、不推理、不重拟合。

════════════════════════ 预注册（先写定义，再跑） ════════════════════════

【派生件 1】table_5_1.csv —— 6 方向 × 7 族 = 42 行

  唯一真源：`runs/20261020-p2-uncertainty/boot_cells.csv`（1,512 行，跨域格）。
  ⚠️ 2026-10-21：三条输入路径与输出目录改为 **CLI 参数**
  （`--boot-cells` / `--boot-summary` / `--matrix` / `--out`），默认值仍是上面这几条
  旧路径。口径重跑（`*-r2`）时**必须显式传参**，不得手改输出文件。
  ⚠️ **不**用 `p2_matrix.csv` 取点值：该文件把 `R_dev_med` / `R_pair_med` 以
  `%.10g` 写盘，先舍入再相减会在 `ΔR` 上引入 ~6e-10 的二次舍入；而
  `boot_cells.csv` 的 `R_dev` / `R_pair` / `deltaR` 是同一份浮点数全精度写出，
  `R_pair − R_dev` 与 `deltaR` 逐行严格相等。故点值一律取自 boot_cells。

  列定义：
    R_dev_med      36 个评委对的 `R_dev` 取中位
    R_pair_med     36 个评委对的 `R_pair` 取中位
    deltaR_med     36 个评委对的 `deltaR` 取中位（**先作差再取中位**，配对统计）
    n_deltaR_pos   `deltaR > EPS`（EPS = 1e-12）的评委对数；`none` 族的 deltaR 是
                   ±2e-16 的浮点噪声，按 EPS 一律计 0，故该列在 `none` 行为 0
    n_ci_excl_0    条件 bootstrap 区间不含 0 的评委对数
    ci_width_med   `deltaR_hi − deltaR_lo` 的中位
  `none` 族的 6 行 `deltaR` 恒为 0，作内部校验。

【派生件 2】ovl.csv —— 3 个无序域对

  定义：`OVL = Σ min(p_A, p_B)`，在**已归一化的 `y` 列**上做 20 等宽 bin 直方图。
  与 `metrics/comparability_criteria.py:54-62` 的判据 1b **同一公式、同一 bin 数**。
  y 直接取自 PDE 域输入件（三域均已归一到 [0,1]，见 `make_p2_inputs.py`），
  **不做二次归一化** —— 二次归一化会把「位置/尺度差」抹掉，测出来就只剩形状，
  与「分数分布是否不重叠」的原意不符。
  阈值沿用 `configs/20261016-p1diagnose-v2.yaml`：`c1_overlap_min = 0.50`。

【派生件 3】extrap.csv —— 6 个有序方向 × 9 评委（逐折）

  定义：判据 1a 的跨域版。对拟合域 A 的每个评委 k、每一折 f：
    支撑 = A 中**不在第 f 折**（即校准集）的图的 k 分数范围 [min, max]；
    外推比例 = B 中 k 分数落在该范围之外的图占比。
  取「跨折 × 跨评委」的最大值作为该方向的读数 —— 与
  `comparability_criteria.py:38-48` 的 1a 同口径（**必须逐折算**：把所有折的
  校准集并起来，支撑会退化成全体取值范围，这一条就白算了）。
  阈值沿用 `c1_extrapolation_max = 0.05`。

@created: 2026-10-21
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, "runs")

BOOT_CELLS = os.path.join(RUNS, "20261020-p2-uncertainty", "boot_cells.csv")
BOOT_SUMMARY = os.path.join(RUNS, "20261020-p2-uncertainty", "boot_summary.json")
MATRIX = os.path.join(RUNS, "20261019-p2transfer-3dom", "p2_matrix.csv")
OUT_DIR = os.path.join(RUNS, "20261021-p2-ch5")

DOMAINS = {
    "tid2013": r"D:/datasets/tid2013/labels/p2_tid2013.csv",
    "koniq": r"D:/datasets/koniq10k/labels/p2_koniq.csv",
    "agiqa3k": r"D:/datasets/agiqa3k/labels/p2_agiqa3k.csv",
}
DIRECTIONS = [("tid2013", "koniq"), ("tid2013", "agiqa3k"), ("koniq", "tid2013"),
              ("koniq", "agiqa3k"), ("agiqa3k", "tid2013"), ("agiqa3k", "koniq")]
FAMILIES = ["none", "minmax", "zscore", "quantile", "platt", "logistic5", "isotonic"]

N_BINS = 20
OVL_MIN = 0.50          # c1_overlap_min
EXTRAP_MAX = 0.05       # c1_extrapolation_max
EPS = 1e-12             # 与 p2_boot_summary.py 同一量级的「浮点零」判据


def _f(x):
    return float(x)


def _sha8(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:8]


# ──────────────────────────── 1. table_5_1 ────────────────────────────
def derive_table_5_1():
    rows = list(csv.DictReader(open(BOOT_CELLS, encoding="utf-8")))
    if len(rows) != 1512:
        raise SystemExit("boot_cells 行数 %d，期望 1512" % len(rows))

    out = []
    for fa, fb in DIRECTIONS:
        for fam in FAMILIES:
            sel = [r for r in rows if r["fit_ds"] == fa and r["test_ds"] == fb
                   and r["mapping"] == fam]
            if len(sel) != 36:
                raise SystemExit("%s→%s %s 行数 %d，期望 36" % (fa, fb, fam, len(sel)))
            if any(r["status"] != "ok" for r in sel):
                raise SystemExit("%s→%s %s 有非 ok 行" % (fa, fb, fam))
            rd = np.array([_f(r["R_dev"]) for r in sel])
            rp = np.array([_f(r["R_pair"]) for r in sel])
            dr = np.array([_f(r["deltaR"]) for r in sel])
            if not np.allclose(rp - rd, dr, rtol=0.0, atol=0.0):
                raise SystemExit("%s→%s %s: R_pair − R_dev 与 deltaR 不严格相等" % (fa, fb, fam))
            w = np.array([_f(r["deltaR_hi"]) - _f(r["deltaR_lo"]) for r in sel])
            out.append({
                "fit_ds": fa, "test_ds": fb, "mapping": fam,
                "R_dev_med": "%.17g" % float(np.median(rd)),
                "R_pair_med": "%.17g" % float(np.median(rp)),
                "deltaR_med": "%.17g" % float(np.median(dr)),
                "n_deltaR_pos": int(np.sum(dr > EPS)),
                "n_ci_excl_0": sum(1 for r in sel if r["deltaR_ci_contains_0"] == "False"),
                "ci_width_med": "%.17g" % float(np.median(w)),
            })
    return out


# ──────────────────────────── 2. OVL ────────────────────────────
def _load_y(name):
    with open(DOMAINS[name], encoding="utf-8") as fh:
        return np.array([float(r["y"]) for r in csv.DictReader(fh)])


def derive_ovl():
    ys = {k: _load_y(k) for k in DOMAINS}
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    out = []
    for i, a in enumerate(sorted(DOMAINS)):
        for b in sorted(DOMAINS)[i + 1:]:
            pa, _ = np.histogram(np.clip(ys[a], 0.0, 1.0), bins=edges)
            pb, _ = np.histogram(np.clip(ys[b], 0.0, 1.0), bins=edges)
            pa = pa / max(pa.sum(), 1)
            pb = pb / max(pb.sum(), 1)
            ovl = float(np.minimum(pa, pb).sum())
            out.append({
                "ds_a": a, "ds_b": b, "n_bins": N_BINS,
                "n_a": len(ys[a]), "n_b": len(ys[b]),
                "ovl": "%.17g" % ovl,
                "y_min_a": "%.17g" % float(ys[a].min()), "y_max_a": "%.17g" % float(ys[a].max()),
                "y_min_b": "%.17g" % float(ys[b].min()), "y_max_b": "%.17g" % float(ys[b].max()),
                "threshold": OVL_MIN,
                "verdict": "fail" if ovl < OVL_MIN else "pass",
            })
    return out


# ──────────────────────────── 3. 外推比例 ────────────────────────────
def _load_domain(name):
    with open(DOMAINS[name], encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    judges = [c for c in rows[0] if c not in ("image", "y", "y_native", "fold")]
    return rows, judges


def derive_extrap():
    data = {}
    for k in DOMAINS:
        data[k] = _load_domain(k)
    judges = data["tid2013"][1]
    if len(judges) != 9:
        raise SystemExit("评委列数 %d，期望 9" % len(judges))
    folds = sorted({int(r["fold"]) for r in data["tid2013"][0]})
    if len(folds) != 5:
        raise SystemExit("折数 %d，期望 5" % len(folds))

    out = []
    for fa, fb in DIRECTIONS:
        ra, _ = data[fa]
        rb, _ = data[fb]
        for j in judges:
            xa = np.array([float(r[j]) for r in ra])
            xb = np.array([float(r[j]) for r in rb])
            fa_fold = np.array([int(r["fold"]) for r in ra])
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


# ──────────────────────────── 汇总 ────────────────────────────
def build_summary(t51, ovl, extrap):
    # 与 boot_summary.json 的 by_cell 对照（同一条管线的自证）
    bc = {(r["fit_ds"], r["test_ds"], r["mapping"]): r
          for r in json.load(open(BOOT_SUMMARY, encoding="utf-8"))["by_cell"]}
    mt = {"deltaR_med": 0.0, "n_ci_excl_0": 0}
    for r in t51:
        k = (r["fit_ds"], r["test_ds"], r["mapping"])
        mt["deltaR_med"] = max(mt["deltaR_med"], abs(_f(r["deltaR_med"]) - bc[k]["deltaR_median"]))
        mt["n_ci_excl_0"] = max(mt["n_ci_excl_0"],
                                abs(r["n_ci_excl_0"] - bc[k]["n_deltaR_ci_excludes_0"]))
    # 注：by_cell 的 deltaR_sign_share 是「与自身中位同号」的比例，与本表的
    # 「deltaR > 0」不是同一个量，故不比对；`none` 族二者必然不同（恒零时按 |x|<=EPS 计）。

    # 与归档矩阵的点值对照（%.10g 口径，同 Gate 2）
    mx = {}
    for r in csv.DictReader(open(MATRIX, encoding="utf-8")):
        if r["diagonal"] == "False":
            mx[(r["fit_ds"], r["test_ds"], r["mapping"], r["judge_i"], r["judge_j"])] = r
    arch_bad = 0
    for r in csv.DictReader(open(BOOT_CELLS, encoding="utf-8")):
        a = mx[(r["fit_ds"], r["test_ds"], r["mapping"], r["judge_i"], r["judge_j"])]
        for col_b, col_a in (("R_dev", "R_dev_med"), ("R_pair", "R_pair_med")):
            if "%.10g" % _f(r[col_b]) != "%.10g" % _f(a[col_a]):
                arch_bad += 1

    # 外推的方向级读数
    dir_stats = []
    for fa, fb in DIRECTIONS:
        sel = [r for r in extrap if r["fit_ds"] == fa and r["test_ds"] == fb]
        vals = [_f(r["mean_over_folds"]) for r in sel]
        mxv = max(_f(r["max_over_folds"]) for r in sel)
        worst = max(sel, key=lambda r: _f(r["max_over_folds"]))
        dir_stats.append({
            "fit_ds": fa, "test_ds": fb,
            "extrap_mean_of_judges": float(np.mean(vals)),
            "extrap_max_over_judges": mxv,
            "worst_judge": worst["judge"],
            "verdict_1a": "fail" if mxv > EXTRAP_MAX else "pass",
        })

    return {
        "generated": "2026-10-21",
        "batch": "2026-10-21（R 口径 r2 重跑）",
        "inputs": {
            BOOT_CELLS: _sha8(BOOT_CELLS),
            BOOT_SUMMARY: _sha8(BOOT_SUMMARY),
            MATRIX: _sha8(MATRIX),
            **{DOMAINS[k]: _sha8(DOMAINS[k]) for k in DOMAINS},
        },
        "definitions": {
            "deltaR": "R_pair − R_dev（同一评委对配对作差），42 格各取 36 对中位",
            "ovl": "Σ min(p_A, p_B)，归一化 y 上 %d 等宽 bin，阈值 %.2f" % (N_BINS, OVL_MIN),
            "extrap": "判据 1a 跨域版：逐折取拟合域校准集支撑，测测试域落在其外的比例，"
                      "跨折×跨评委取最大；阈值 %.2f" % EXTRAP_MAX,
        },
        "checks": {
            "n_table_5_1_rows": len(t51),
            "max_abs_diff_deltaR_vs_by_cell": mt["deltaR_med"],
            "max_abs_diff_n_ci_excl_0_vs_by_cell": mt["n_ci_excl_0"],
            "fmt10g_mismatch_vs_archive": arch_bad,
            "none_deltaR_all_zero": all(_f(r["deltaR_med"]) == 0.0
                                        for r in t51 if r["mapping"] == "none"),
            "n_extrap_rows": len(extrap),
            "n_ovl_rows": len(ovl),
        },
        "ovl": ovl,
        "extrap_by_direction": dir_stats,
    }


def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _cli():
    ap = argparse.ArgumentParser(description="第五章派生件：table_5_1 / ovl / extrap")
    ap.add_argument("--boot-cells", default=BOOT_CELLS)
    ap.add_argument("--boot-summary", default=BOOT_SUMMARY)
    ap.add_argument("--matrix", default=MATRIX)
    ap.add_argument("--out", default=OUT_DIR)
    return ap.parse_args()


def main():
    global BOOT_CELLS, BOOT_SUMMARY, MATRIX, OUT_DIR
    a = _cli()
    BOOT_CELLS, BOOT_SUMMARY, MATRIX, OUT_DIR = (
        a.boot_cells, a.boot_summary, a.matrix, a.out)
    os.makedirs(OUT_DIR, exist_ok=True)
    t51 = derive_table_5_1()
    ovl = derive_ovl()
    extrap = derive_extrap()
    _write_csv(os.path.join(OUT_DIR, "table_5_1.csv"), t51)
    _write_csv(os.path.join(OUT_DIR, "ovl.csv"), ovl)
    _write_csv(os.path.join(OUT_DIR, "extrap.csv"), extrap)
    s = build_summary(t51, ovl, extrap)
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8", newline="") as fh:
        json.dump(s, fh, ensure_ascii=False, indent=1)
        fh.write("\n")

    print("table_5_1 rows =", len(t51))
    print("ovl:")
    for r in ovl:
        print("  %-8s %-8s OVL=%s  %s" % (r["ds_a"], r["ds_b"], r["ovl"], r["verdict"]))
    print("extrap by direction:")
    for r in s["extrap_by_direction"]:
        print("  %-8s->%-8s mean=%.4f max=%.4f (%s) %s"
              % (r["fit_ds"], r["test_ds"], r["extrap_mean_of_judges"],
                 r["extrap_max_over_judges"], r["worst_judge"], r["verdict_1a"]))
    print("checks:", json.dumps(s["checks"], ensure_ascii=False))


if __name__ == "__main__":
    main()
