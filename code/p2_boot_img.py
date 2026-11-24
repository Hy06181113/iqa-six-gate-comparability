# -*- coding: utf-8 -*-
"""P2 · 图像级**条件** bootstrap：`R` / `ΔR` 的不确定性（2026-10-20）。

状态：**重算 + 重采样**。不跑推理、不加载权重、不重拟合 P1；只读取缓存的评委
分数（经 `p2_transfer.load_domain`），重跑一次跨域矩阵并**保留逐图误差**，
再在评测图上重采样。

═══ 这个脚本回答的问题 ═══════════════════════════════════════════════════════

Q1（「换掉 6 对」的抖动）由 `p2_pair_subsample.py` 用**精确组合枚举**回答。
本脚本回答的是另一个问题：

  · 单个格的 `ΔR`（例如 `agiqa→koniq` logistic5 的 −0.0457）**能不能与 0 区分**？
  · 单个格的 `R_dev` 跨 1 的两套读数，差别是不是被评测图的抽样误差淹没？

═══ 它是「条件」bootstrap，不是无条件 CI ═══════════════════════════════════

重采样**只针对评测图**，映射固定为本次拟合出来的那 20 份（拟合域 5 折 × 2 评委
× 2 侧）。因此：

  · 它含 **评测图抽样** 的方差；
  · 它**不含**「重新拟合映射」的方差（那需要每次重采样都重拟合，成本 ≈ 30 倍）；
  · 它**不含**「换数据集 / 换评委」的方差（数据集与评委都是固定集合）。

⇒ 报出的区间是**偏窄的下界**。文档与正文**必须**写成「条件 bootstrap 区间」，
   不得写成「95% 置信区间」而不加条件。

═══ 重采样设计（**先写死，再看结果**）══════════════════════════════════════

  · **按折分层**：每折内独立重采样同数量的图（保持折大小），不把折结构打散；
  · `B = 2000`；每格种子 = `20261021 + 1000 × 评委对序号 + 族序号`
    （沿用 `p2_transfer.py:608` 的构造，保证可复现）；
  · 每次重采样重算 4 个 RMSE；`sd(y_test)` 仍算（诊断与状态位用），但**不进 R**；
  · 池化口径与 `compute_R` 完全一致：
        R = RMSE_num / RMSE_den，
        RMSE = sqrt(Σ 平方误差 / N)，N = 2 × 评测图数。
    ⚠️ **2026-10-21 删掉了旧的 sd 因子**。旧式是
    `(RMSE_num / sd_fit) / (RMSE_den / sd_test)`，分子分母不在同一单位上；
    见 `p2_transfer.py` 模块头「本文件不做的事」一节。

═══ 产物 ═══════════════════════════════════════════════════════════════════

  `boot_cells.csv`   n(n−1) 方向 × 7 族 × 36 对（对角不跑，R 恒 1）
                     · 三域 spec ⇒ 6 方向、**1,512 行**（初版）
                     · 四域 spec ⇒ 12 方向、**3,024 行**

═══ 去位置口径 `R_c` 的区间（2026-10-22 加）════════════════════════════════

  同一次重采样里**顺带**算去位置的比值，写进 `R_dev_c*` / `R_pair_c*` /
  `deltaR_c*` 三组列。定义与 `compute_R._rmse_c` **逐字同式**：

      rmse_c = sqrt(max(mean(r²) − mean(r)², 0.0))，R_c = rmse_c(分子)/rmse_c(分母)

  为此 `compute_R(keep_errs=True)` 的回传里新增了四个**有符号**残差数组
  （`s_num_dev` / `s_den_dev` / `s_num_pair` / `s_den_pair`）：只给 `Σr²` 是
  还原不出 `mean(r)` 的。计数口径与 `compute_R` 一致——`dev` 类用 2·n_tot
  （两位评委拼接），`pair` 类用 n_tot；比值里 2 倍会约掉，但**不得混用**。

  ⚠️ 未去位置的列**一个字节都没变**：同一格、同一种子，`R_dev` / `R_pair` /
  `deltaR` 与旧版产出的值必须逐行全等，这是本脚本的复现闸门。

`boot_summary.json`（计数、Gate 2 的比对结果、逐格摘要）**不由本脚本写** ——
它由 `p2_boot_summary.py` 从上面的 CSV 汇总生成，这样「跑」与「验」解耦。

⚠️ **每 50 格 flush**（教训：只在结束时写盘，杀掉即全丢）。
⚠️ **断点续跑**：已存在的行（按五元键判）直接跳过，无开关。

用法：
  cd 30-experiments
  python -u metrics/p2_boot_img.py --spec D:/datasets/p2_inputs3/p2_spec_full.json \
      --archive runs/20261019-p2transfer-3dom/p2_matrix_records.csv \
      --out runs/20261020-p2-uncertainty
"""
import argparse
import csv
import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p2_transfer as P                                          # noqa: E402

FAMILIES = ("none", "minmax", "zscore", "quantile", "platt", "logistic5", "isotonic")
B_DEFAULT = 2000
SEED_BASE = 20261021
# 三域归档的跨域格数（1,512）不再是硬编码：`_load_archive` 按 spec 的域集合推导

COLS = [
    "fit_ds", "test_ds", "mapping", "judge_i", "judge_j",
    "status", "n_eval", "n_folds", "sd_fit", "sd_test", "n_boot",
    "R_dev", "R_pair", "deltaR",
    "R_dev_lo", "R_dev_med", "R_dev_hi",
    "R_pair_lo", "R_pair_med", "R_pair_hi",
    "deltaR_lo", "deltaR_med", "deltaR_hi",
    "deltaR_ci_contains_0", "R_dev_ci_contains_1", "R_pair_ci_contains_1",
    "arch_R_dev_pooled", "arch_R_pair_pooled",
    "abs_diff_R_dev", "abs_diff_R_pair",
    # ── 2026-10-22：去位置口径（`R_c`）的条件区间 ──
    "R_dev_c", "R_pair_c", "deltaR_c",
    "R_dev_c_lo", "R_dev_c_med", "R_dev_c_hi",
    "R_pair_c_lo", "R_pair_c_med", "R_pair_c_hi",
    "deltaR_c_lo", "deltaR_c_med", "deltaR_c_hi",
    "deltaR_c_ci_contains_0", "R_dev_c_ci_contains_1", "R_pair_c_ci_contains_1",
]


def _boot_one(errs, sd_fit, rng, B):
    """按折分层重采样，返回 `R_dev`、`R_pair`、`R_dev_c`、`R_pair_c` 四个数组。

    `errs` 是 `compute_R(keep_errs=True)` 的回传：每折的 y、四种逐图**平方**误差，
    以及同口径的**有符号**残差 `s_*`（2026-10-22 加）。前者算未去位置的
    `R`，后者算去位置的 `R_c` —— 去位置要 `mean(r)`，只从 `mean(r²)` 还原不出来。
    """
    num_dev = np.zeros(B)
    den_dev = np.zeros(B)
    num_pair = np.zeros(B)
    den_pair = np.zeros(B)
    s_num_dev = np.zeros(B)
    s_den_dev = np.zeros(B)
    s_num_pair = np.zeros(B)
    s_den_pair = np.zeros(B)
    sum_y = np.zeros(B)
    sum_y2 = np.zeros(B)
    n_tot = 0
    for e in errs:
        n = int(e["n"])
        n_tot += n
        w = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float64)
        num_dev += w @ e["num_dev"]
        den_dev += w @ e["den_dev"]
        num_pair += w @ e["num_pair"]
        den_pair += w @ e["den_pair"]
        s_num_dev += w @ e["s_num_dev"]
        s_den_dev += w @ e["s_den_dev"]
        s_num_pair += w @ e["s_num_pair"]
        s_den_pair += w @ e["s_den_pair"]
        y = e["y"]
        sum_y += w @ y
        sum_y2 += w @ (y * y)
    # 每条重采样里评测图总数 = n_tot（分层保持折大小），误差项各占 2 份
    rmse = lambda s: np.sqrt(s / (2.0 * n_tot))          # noqa: E731

    def rmse_c(s2, s1, cnt):
        """去位置 RMSE = 残差的标准差，与 `compute_R._rmse_c` 同式。

        `cnt` 是残差个数：`dev` 类 2·n_tot（两位评委拼接），`pair` 类 n_tot。
        """
        m = s1 / cnt
        return np.sqrt(np.maximum(s2 / cnt - m * m, 0.0))

    c_num_dev = rmse_c(num_dev, s_num_dev, 2.0 * n_tot)
    c_den_dev = rmse_c(den_dev, s_den_dev, 2.0 * n_tot)
    c_num_pair = rmse_c(num_pair, s_num_pair, float(n_tot))
    c_den_pair = rmse_c(den_pair, s_den_pair, float(n_tot))

    mean_y = sum_y / n_tot
    var_y = np.maximum(sum_y2 / n_tot - mean_y * mean_y, 0.0)
    sd_test = np.sqrt(var_y)
    ok = (den_dev > 0) & (den_pair > 0) & (sd_test > 0)
    rd = np.full(B, np.nan)
    rp = np.full(B, np.nan)
    # 2026-10-21：分子分母同为测试域上的 RMSE，不再各除各的 sd
    # （`sd_fit` 参数保留在签名里，只用于状态位/诊断，不参与公式）
    rd[ok] = rmse(num_dev[ok]) / rmse(den_dev[ok])
    rp[ok] = rmse(num_pair[ok]) / rmse(den_pair[ok])

    ok_c = ok & (c_den_dev > 0) & (c_den_pair > 0)
    rdc = np.full(B, np.nan)
    rpc = np.full(B, np.nan)
    rdc[ok_c] = c_num_dev[ok_c] / c_den_dev[ok_c]
    rpc[ok_c] = c_num_pair[ok_c] / c_den_pair[ok_c]
    return rd, rp, rdc, rpc


def _pct(a, q):
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None, None, None
    return (float(np.percentile(a, q)), float(np.percentile(a, 50)),
            float(np.percentile(a, 100 - q)))


def _load_archive(path, domains):
    """读归档的跨域记录，**按 spec 的域集合过滤**。

    ⚠️ 2026-10-21 改：原来断言「跨域格 == 1,512」是写死三域的。此后
    归档是 4 域（2,688 格），而本章的 bootstrap 仍只跑三域 ⇒ 改成按 spec 的
    域名过滤、期望格数由域数推出：n(n−1) × 7 族 × C(评委,2)。
    """
    names = {d.name for d in domains}
    n_judges = len(domains[0].judges) if domains else 0
    n_pairs = n_judges * (n_judges - 1) // 2
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["diagonal"] == "True":
                continue
            if r["fit_ds"] not in names or r["test_ds"] not in names:
                continue
            out[(r["fit_ds"], r["test_ds"], r["mapping"], r["judge_i"], r["judge_j"])] = r
    exp = len(names) * (len(names) - 1) * len(FAMILIES) * n_pairs
    assert len(out) == exp, "归档三域子集跨域格 %d ≠ 期望 %d" % (len(out), exp)
    return out


def _existing(path):
    keys = set()
    if not os.path.exists(path):
        return keys
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            keys.add((r["fit_ds"], r["test_ds"], r["mapping"], r["judge_i"], r["judge_j"]))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="D:/datasets/p2_inputs3/p2_spec_full.json")
    ap.add_argument("--archive",
                    default=os.path.join(os.path.dirname(HERE), "runs",
                                         "20261019-p2transfer-3dom",
                                         "p2_matrix_records.csv"))
    ap.add_argument("--out",
                    default=os.path.join(os.path.dirname(HERE), "runs",
                                         "20261020-p2-uncertainty"))
    ap.add_argument("--boot", type=int, default=B_DEFAULT)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 格（冒烟用）")
    ap.add_argument("--families", default=",".join(FAMILIES))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "boot_cells.csv")
    done = _existing(path)
    new_file = not os.path.exists(path)
    f = open(path, "a", encoding="utf-8", newline="")
    w = csv.DictWriter(f, fieldnames=COLS, lineterminator="\n")
    if new_file:
        w.writeheader()

    spec = json.load(open(args.spec, encoding="utf-8"))
    doms = [P.load_domain(s).validate() for s in spec["datasets"]]
    by_name = {d.name: d for d in doms}
    judges = list(doms[0].judges)
    fams = [s for s in args.families.split(",") if s]
    arch = _load_archive(args.archive, doms)
    cfg = {"seed": 20261019, "n_splits": 5, "boot": 0, "families": list(P.ALL_FAMILIES)}

    tasks = []
    for a, b in itertools.product(doms, repeat=2):
        if a is b:
            continue
        for k, (ji, jj) in enumerate(itertools.combinations(judges, 2)):
            for fi, fam in enumerate(fams):
                key = (a.name, b.name, fam, ji, jj)
                if key in done:
                    continue
                tasks.append((a, b, ji, jj, k, fam, fi, key))
    if args.limit:
        tasks = tasks[:args.limit]
    print("domains %s | judges %d | tasks %d | B %d | resume_skipped %d"
          % ({d.name: d.n for d in doms}, len(judges), len(tasks), args.boot,
             len(done)), flush=True)

    t0 = time.time()
    n_written = 0
    for idx, (a, b, ji, jj, k, fam, fi, key) in enumerate(tasks):
        rng = np.random.default_rng(SEED_BASE + 1000 * k + fi)
        t1 = time.time()
        r = P.compute_R(a, b, None, None, fam, ji, jj, cfg, keep_errs=True)
        row = {"fit_ds": a.name, "test_ds": b.name, "mapping": fam,
               "judge_i": ji, "judge_j": jj, "status": r.get("status"),
               "n_eval": r.get("n_eval", ""), "n_folds": r.get("n_folds", ""),
               "sd_fit": r.get("sd_fit", ""), "sd_test": r.get("sd_test", ""),
               "n_boot": args.boot}
        if r.get("status") == "ok":
            rd, rp, rdc, rpc = _boot_one(r["errs"], float(r["sd_fit"]), rng, args.boot)
            dl = rp - rd
            lo, med, hi = _pct(rd, 2.5)
            row.update({"R_dev": r["R_dev_pooled"], "R_dev_lo": lo,
                        "R_dev_med": med, "R_dev_hi": hi})
            lo, med, hi = _pct(rp, 2.5)
            row.update({"R_pair": r["R_pair_pooled"], "R_pair_lo": lo,
                        "R_pair_med": med, "R_pair_hi": hi})
            lo, med, hi = _pct(dl, 2.5)
            row.update({"deltaR": r["R_pair_pooled"] - r["R_dev_pooled"],
                        "deltaR_lo": lo, "deltaR_med": med, "deltaR_hi": hi,
                        "deltaR_ci_contains_0":
                            ("" if lo is None else bool(lo <= 0.0 <= hi)),
                        "R_dev_ci_contains_1":
                            ("" if row["R_dev_lo"] is None
                             else bool(row["R_dev_lo"] <= 1.0 <= row["R_dev_hi"])),
                        "R_pair_ci_contains_1":
                            ("" if row["R_pair_lo"] is None
                             else bool(row["R_pair_lo"] <= 1.0 <= row["R_pair_hi"]))})
            # 去位置口径：点值仍取 compute_R 的 pooled 值（不进重采样），
            # 区间来自同一批重采样的 `R_c` 分布。分母为 0 的格（`compute_R`
            # 记 None）不报，留空 —— 与 `R_dev_c` 列的 None 语义一致。
            if r.get("R_dev_c_pooled") is not None and r.get("R_pair_c_pooled") is not None:
                row.update({"R_dev_c": r["R_dev_c_pooled"],
                            "R_pair_c": r["R_pair_c_pooled"],
                            "deltaR_c": r["R_pair_c_pooled"] - r["R_dev_c_pooled"]})
                lo, med, hi = _pct(rdc, 2.5)
                row.update({"R_dev_c_lo": lo, "R_dev_c_med": med, "R_dev_c_hi": hi,
                            "R_dev_c_ci_contains_1":
                                ("" if lo is None else bool(lo <= 1.0 <= hi))})
                lo, med, hi = _pct(rpc, 2.5)
                row.update({"R_pair_c_lo": lo, "R_pair_c_med": med, "R_pair_c_hi": hi,
                            "R_pair_c_ci_contains_1":
                                ("" if lo is None else bool(lo <= 1.0 <= hi))})
                lo, med, hi = _pct(rpc - rdc, 2.5)
                row.update({"deltaR_c_lo": lo, "deltaR_c_med": med, "deltaR_c_hi": hi,
                            "deltaR_c_ci_contains_0":
                                ("" if lo is None else bool(lo <= 0.0 <= hi))})
            ar = arch.get(key)
            if ar is not None:
                row["arch_R_dev_pooled"] = ar["R_dev_pooled"]
                row["arch_R_pair_pooled"] = ar["R_pair_pooled"]
                row["abs_diff_R_dev"] = abs(float(ar["R_dev_pooled"])
                                            - r["R_dev_pooled"])
                row["abs_diff_R_pair"] = abs(float(ar["R_pair_pooled"])
                                             - r["R_pair_pooled"])
        w.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in COLS})
        n_written += 1
        if n_written % 50 == 0:
            f.flush()
            el = time.time() - t0
            print("  %d/%d  %.0fs  eta %.1f h" % (idx + 1, len(tasks), el,
                                                  el / (idx + 1) * (len(tasks) - idx - 1) / 3600.0),
                  flush=True)
        del r
        if idx == 0:
            print("  first cell %.1fs" % (time.time() - t1), flush=True)
    f.flush()
    f.close()
    print("done %d rows, %.1f s" % (n_written, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
