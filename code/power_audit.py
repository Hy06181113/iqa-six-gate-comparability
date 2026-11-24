# -*- coding: utf-8 -*-
"""基准的统计功效审计：分辨两个模型的 SRCC 差，需要多少张图？

════════════════════════════════════════════════════════════════════
预注册（2026-11-23 建，跑之前写死；改任何一条都要在 progress.md 记账）
════════════════════════════════════════════════════════════════════

问题
----
文献里大量出现「模型 A 的 SRCC 比 B 高 0.003」。在现有基准规模下，
这个差**分辨得出来吗**？本脚本对每一对模型给出：给定 n 张图，
80% 功效下能分辨的最小 SRCC 差（MDD），以及要分辨某个目标差需要多少图（N*）。

为什么这个量值得算
------------------
它把「两模型谁更好」从一句口号变成一个有刻度的判断：读者拿到自己基准的
图像数，就能查表知道「这个规模的基准根本分辨不出 0.005」。这是**正面的、
可直接使用的交付物**，与本文其余部分的「不可比」结论互补。

统计量（只这一条，不换）
------------------------
    Δ = SRCC(A, MOS) − SRCC(B, MOS)      在同一批图上算，配对设计

不采用「SRCC(A,B) 模型互相关」：那不是「谁更接近人类」的问题。

估计方式
--------
- **配对重抽样**：从 N 张图里无放回抽 n 张，重算 Δ；重复 M 次，
  取 SD 作为 SE(n)。图是抽样单位，两模型在同一子样本上配对。
- **MDD(n) = (z_{0.975} + z_{0.80}) · SE(n) = 2.801585 · SE(n)**
  （双侧 α=0.05、功效 0.80；两个 z 值是常数，不是拟合出来的）
- **N\* 由 SE(n) ≈ c / √n 反解**：c = median(SE(n)·√n)，
  N\* = (2.801585 · c / Δ)²。⚠️ 这条 1/√n 外推**必须验**：
  脚本同时报拟合值与实测 SE 的相对偏差，偏差 >10% 时在 report 里标红。

预注册的取值（不得看数改）
--------------------------
    SEED      = 20261123
    M         = 400                     # 每个 (对, n) 的重抽样次数
    N_GRID    = [250, 500, 1000, 2000]  # 超过该数据集 N 的档位跳过
    ALPHA     = 0.05  (双侧)   POWER = 0.80
    Z_SUM     = 1.959964 + 0.841621 = 2.801585
    TARGETS   = [0.005, 0.010, 0.020]   # 要反解 N* 的目标差
    数据集    = tid2013、koniq10k（P1 的两个，其余三域不参与）
    模型对    = 9 个评委的全部 36 对无序对

做什么 / 不做什么
-----------------
做：只读缓存的逐图分数（pred_*.jsonl）与标签 CSV，算 SRCC，配对重抽样。
不做：不重跑任何模型、不改任何既有读数、不触碰 ε 与判据。
      ⇒ 本脚本的产出**不是**判据的一部分，是一条独立的基准审计读数。

两条边界（先说，免得被读成更强的结论）
--------------------------------------
① **图不是独立观测**。TID2013 的 2,687 张图来自 23 个参考图，每个参考图下
   有多个失真版本；本脚本按**图**重抽样，等于假定图与图独立，这会**低估** SE。
   ⇒ TID2013 的 MDD 是**偏乐观**的下界；真实 MDD 只会更大。KonIQ 的图彼此独立，
   不受此影响。若要做严格版，须按参考图分组重抽样——本批未做，登记为已知边界。
② **N\* 是外推，不是实测**。它由全样本 SE 按 1/√n 反解；子抽样得到的
   SE·√n 与全样本反推的 c 存在 20–30% 漂移（脚本会报），故 N\* 只作
   **量级**使用，不作精确值。**可信的是各 n 档的 MDD 本身。**

产物
----
    runs/<run>/power_per_pair.csv   每 (数据集, 对, n) 一行
    runs/<run>/summary.json         汇总 + 拟合诊断
    runs/<run>/report.md            人读版
"""
import argparse
import csv
import json
import os
import sys
from itertools import combinations

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

SEED = 20261123
M = 400
N_GRID = [250, 500, 1000, 2000]
Z_SUM = 1.959964 + 0.841621      # 双侧 0.05 + 功效 0.80
TARGETS = [0.005, 0.010, 0.020]

DATASETS = {
    "tid2013": {
        "labels": r"D:\datasets\tid2013\labels\tid2013_eval_labels.csv",
        "label_col": "MOS",
        "key_col": "image_name",
        "preds": {
            "q-align": "runs/20261015-qalign-tid2013-l4/pred_qalign_tid2013.jsonl",
            "q-realign-mini": "runs/20261014-qrealign-tid2013-l4/pred_qrealign_test.jsonl",
            "deqa-score": "runs/20261015-deqascore-tid2013-l4/pred_deqascore_tid2013.jsonl",
            "iqa-t1": "runs/20261015-iqat1-tid2013-l4/pred_iqat1_tid2013.jsonl",
            "clip-iqa": "runs/20261014-clipiqa-tid2013-l4/pred_clipiqa_test.jsonl",
            "maniqa": "runs/20261014-maniqa-tid2013-l4/pred_maniqa_test.jsonl",
            "musiq": "runs/20261014-musiq-tid2013-l4/pred_musiq_test.jsonl",
            "hyperiqa": "runs/20261014-hyperiqa-tid2013-l4/pred_hyperiqa_test.jsonl",
            "topiq-nr": "runs/20261014-topiqnr-tid2013-l4/pred_topiqnr_test.jsonl",
        },
    },
    "koniq10k": {
        "labels": r"D:\datasets\koniq10k\labels\koniq10k_scores_and_distributions.csv",
        "label_col": "MOS",
        "key_col": "image_name",
        "preds": {
            "q-align": "runs/20261013-qalign-koniq-l4/pred_qalign_test.jsonl",
            "q-realign-mini": "runs/20261014-qrealign-koniq-l4/pred_qrealign_test.jsonl",
            "deqa-score": "runs/20261013-deqascore-koniq-l4/pred_deqascore_test.jsonl",
            "iqa-t1": "runs/20261013-iqat1-koniq-l4/pred_iqat1_test.jsonl",
            "clip-iqa": "runs/20261014-clipiqa-koniq-l4/pred_clipiqa_test.jsonl",
            "maniqa": "runs/20261009-maniqa-koniq-l4/pred_maniqa_test.jsonl",
            "musiq": "runs/20261010-musiq-koniq-l4/pred_musiq_test.jsonl",
            "hyperiqa": "runs/20261010-hyperiqa-koniq-l4/pred_hyperiqa_test.jsonl",
            "topiq-nr": "runs/20261010-topiqnr-koniq-l4/pred_topiqnr_test.jsonl",
        },
    },
}


def load_labels(path, key_col, val_col):
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            k = row[key_col].strip().lower()
            try:
                out[k] = float(row[val_col])
            except (TypeError, ValueError):
                continue
    return out


def load_pred(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            k = os.path.basename(d["image"]).lower()
            out[k] = float(d["score"])
    return out


def fast_spearman(a, b):
    """秩上的 Pearson；对本题的规模比 scipy.stats.spearmanr 快一个量级。"""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--m", type=int, default=M)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    rows, summary = [], {"seed": SEED, "M": args.m, "z_sum": Z_SUM,
                         "n_grid": N_GRID, "targets": TARGETS, "datasets": {}}

    for ds, cfg in DATASETS.items():
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        labels = load_labels(cfg["labels"], cfg["key_col"], cfg["label_col"])
        scores = {}
        for judge, rel in cfg["preds"].items():
            p = os.path.join(base, rel)
            if os.path.exists(p):
                scores[judge] = load_pred(p)
            else:
                print("  [skip] %s %s (no file)" % (ds, judge))
        keys = set(labels)
        for s in scores.values():
            keys &= set(s)
        keys = sorted(keys)
        N = len(keys)
        print("[%s] judges=%d  aligned images=%d" % (ds, len(scores), N))
        if len(scores) < 2 or N < 400:
            summary["datasets"][ds] = {"error": "insufficient data", "N": N}
            continue

        y = np.array([labels[k] for k in keys])
        S = {j: np.array([s[k] for k in keys]) for j, s in scores.items()}
        # ⚠️ 只在小样本档子抽样：n 逼近 N 时**无放回抽样几乎没有方差**，
        #    koniq（N=2015）在 n=2000 处 SE 会塌到 0.002，那是抽样几何
        #    的产物、不是功效变好。故子抽样网格取 n ≤ N/2；全样本另用
        #    **有放回 bootstrap** 单独估 SE_full（首版踩到，2026-11-23 订正）。
        grid = [n for n in N_GRID if n <= N // 2]

        ds_rows, c_est = [], []
        full = {}
        for ja, jb in combinations(sorted(S), 2):
            for n in grid:
                deltas = np.empty(args.m)
                for m in range(args.m):
                    idx = rng.choice(N, size=n, replace=False)
                    deltas[m] = (fast_spearman(S[ja][idx], y[idx])
                                 - fast_spearman(S[jb][idx], y[idx]))
                se = float(np.std(deltas, ddof=1))
                ds_rows.append({"dataset": ds, "judge_a": ja, "judge_b": jb,
                                "n": n, "se": se, "mdd": Z_SUM * se, "M": args.m,
                                "kind": "subsample"})
                c_est.append(se * np.sqrt(n))
            # 全样本：有放回 bootstrap
            idxb = rng.integers(0, N, size=(args.m, N))
            dl = (np.array([fast_spearman(S[ja][i], y[i]) for i in idxb])
                  - np.array([fast_spearman(S[jb][i], y[i]) for i in idxb]))
            se_full = float(np.std(dl, ddof=1))
            full[(ja, jb)] = se_full
            ds_rows.append({"dataset": ds, "judge_a": ja, "judge_b": jb,
                            "n": N, "se": se_full, "mdd": Z_SUM * se_full,
                            "M": args.m, "kind": "full_bootstrap"})
        rows.extend(ds_rows)

        se_full_med = float(np.median(list(full.values())))
        c = se_full_med * np.sqrt(N)          # 由全样本 SE 反推 c（更稳）
        agg = {}
        for n in grid + [N]:
            v = [r["mdd"] for r in ds_rows if r["n"] == n]
            agg[str(n)] = {"median": float(np.median(v)),
                           "p10": float(np.percentile(v, 10)),
                           "p90": float(np.percentile(v, 90))}
        # 拟合诊断：子抽样得到的 SE·√n 是否与全样本的 c 一致（1/√n 的前提）
        diag = {}
        for n in grid:
            v = [r["se"] * np.sqrt(n) for r in ds_rows
                 if r["n"] == n and r["kind"] == "subsample"]
            diag[str(n)] = float(np.median(v))
        drift = max(abs(v - c) / c for v in diag.values()) if diag else float("nan")
        # N*：由全样本 SE 外推（c 已达 N，不再依赖子抽样几何）
        nstar = {("%.3f" % t): float(N * (se_full_med * Z_SUM / t) ** 2)
                 for t in TARGETS}
        summary["datasets"][ds] = {"N_aligned": N, "n_grid": grid + [N],
                                   "c_median": c, "se_full": se_full_med,
                                   "mdd_by_n": agg, "sqrt_n_drift": drift,
                                   "N_star": nstar}

    with open(os.path.join(args.out, "power_per_pair.csv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = ["# 基准统计功效审计（预注册见 `metrics/power_audit.py` 头注）", ""]
    for ds, d in summary["datasets"].items():
        if "error" in d:
            lines.append("- %s：%s" % (ds, d["error"]))
            continue
        lines.append("## %s（N = %d）" % (ds, d["N_aligned"]))
        lines.append("")
        lines.append("| n | 中位 MDD | p10 | p90 |")
        lines.append("|---|---|---|---|")
        for n, v in d["mdd_by_n"].items():
            lines.append("| %s | %.4f | %.4f | %.4f |"
                         % (n, v["median"], v["p10"], v["p90"]))
        lines.append("")
        lines.append("最后一行 n = N 是**全样本有放回 bootstrap**（不是子抽样）；"
                     "子抽样档只取 n ≤ N/2，否则无放回抽样本身会把 SE 压塌。")
        lines.append("")
        lines.append("1/√n 外推的漂移：**%.1f%%**%s"
                     % (d["sqrt_n_drift"] * 100,
                        "  ⚠️ 超过 10%，故 N\\* 只作量级使用" if d["sqrt_n_drift"] > 0.10 else ""))
        lines.append("")
        lines.append("| 目标差 Δ | 所需图像数 N* | 该基准实际有 |")
        lines.append("|---|---|---|")
        for t, v in d["N_star"].items():
            lines.append("| %s | %s | %d |"
                         % (t, ("%.0f" % v) if v < 1e7 else ">1e7", d["N_aligned"]))
        lines.append("")
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write("\n".join(lines))
    print("written:", args.out)
    print("\n".join(lines[:18]))


if __name__ == "__main__":
    main()
