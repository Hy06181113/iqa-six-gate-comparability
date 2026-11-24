# -*- coding: utf-8 -*-
"""判据 4 的**对级全局同质性检验**与**幅度量的自助置信区间**。

起因
----
2026-11-09 的复审意见把三件事分开要求：① 一场全局同质性检验；② 各域效应量与
置信区间；③ 明确说明所用置换次数下的 p 值分辨率极限。本脚本给前两样，第三样由本脚本
打印的「检验数 → 阈值 → 对地板倍数」对照表直接读出。

**先把检验数这件事说清（这是「BH 最多支持 7 次拒绝」那条意见的症结）**
--------------------------------------------------------------
Table VI 原设计的 `6,048` 是 `252 判定 × 24 域` 的**格子数**，不是判据 4 跑过的检验数。
判据 4 对每个判定的统计量是 `M = max_c |δ̂(c) − δ̂_global|` —— 跨 24 个域**只取一次最大值**，
空分布也是这个 max 的分布 ⇒ 一个判定**一次**检验。TID2013 上就是 252 次，不是 6,048 次，
更不是 6,048 次互相独立的小检验。

于是 Bonferroni 线的可达性只取决于 `m`：

    p 的 Monte Carlo 地板 = 1/(B+1)；B = 20,000 ⇒ 5.00e-5
    可达 ⇔ 0.05/m > 5.00e-5  ⇔  m < 1,000

| m | 路径 | 阈值 0.05/m | 对地板 5.00e-5 |
|---:|---|---:|---|
| 6,048 | Table VI 原设计（判定×域） | 8.27e-6 | **线下 6.0 倍 ⇒ 结构上不可达** |
| 252 | 判据 4 实际（每判定一次） | 1.98e-4 | 线上 4.0 倍 ⇒ 可达 |
| **36** | **本脚本（评委对层面）** | **1.39e-3** | **线上 27.8 倍 ⇒ 可达且富余** |

⇒ 那条意见说「BH 在最小 p 处同为 α/m、最多只能支持 7 次拒绝」**前半句对、后半句错**：
`i·α/m ≥ 1/(B+1)` 解出 `i ≥ 6.05`，得到的是「**最先可被拒绝的秩位是第 7 位**」，
不是「拒绝个数上限是 7」。且在 `m = 36` 上这个死结根本不存在。

本脚本做什么
------------
1. **对级全局检验**：对 36 个评委对各做**一次**置换检验，统计量取该对在 **7 个校准族 ×
   24 个域**上的最大 `|δ̂ − δ̂_global|`。置换时**同一置换同时作用于全部 7 族**（它们共享
   同一批图与同一套域标签，独立置换会抹掉族间相关）。
2. **幅度量的自助 CI**：对每个 (对, 族) 单元，按图有放回重采样，重估 `β̂` 与 24 个域均值，
   给出 `M` 的 95% 百分位区间；**同一批重复同时留下 24 个 `δ̂(c)` 各自的区间**
   （复审要的「各域效应量与置信区间」——`M` 的区间只是其中最大的那一个，
   给出它不等于给出了各域），落 `c4_domain_ci.csv`，供补充材料整表。
3. **检验数对照表**（上面那张），直接回答「分辨率极限在哪」。

**预期与实际不符要报**：252 个单判定 p 在 B = 20,000 下全部等于地板
（`c4_pvalue_audit.json` 的 `exceed_histogram = {'0': 252}`），所以对级统计量更大 ⇒
对级 p 也必然落在地板。本脚本的价值不在「得到新 p」，而在：给出**正式的 36 次检验读数**、
证明**降检验数就能解决分辨率问题**、以及给出**幅度量的区间**（幅度才是判决依据）。

跑法
----
    python -X utf8 30-experiments/metrics/c4_global.py \
        --cfg 30-experiments/configs/20261105-p1diagnose-c4b20000.yaml \
        --out 30-experiments/runs/20261109-c4global-tid2013

（`-X utf8` 必需；默认控制台代码页会把中文判定标签打成乱码。）
"""
import argparse
import io
import itertools
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comparability_criteria as CC                                # noqa: E402
import mapping_families as MF                                      # noqa: E402
import pred_table as PT                                            # noqa: E402
import stratified_split as SS                                      # noqa: E402


def _domain_means(v, codes, k):
    sums = np.bincount(codes, weights=v, minlength=k)
    cnts = np.bincount(codes, minlength=k).astype(np.float64)
    cnts[cnts == 0] = np.nan
    return sums / cnts


def _pair_stat(stats, codes, k):
    """跨 (族, 域) 的最大 |δ̂(c) − δ̂_global| —— 对级全局统计量。

    `stats` = [(v, mean(v)), ...]，`v = d − β̂·y` 每个族一份；置换只重排 `codes`，
    所以 `v` 与其均值在循环外预计算一次即可（等价，纯提速）。
    """
    worst = 0.0
    for v, g in stats:
        means = _domain_means(v, codes, k)
        worst = max(worst, float(np.nanmax(np.abs(means - g))))
    return worst


def _percentile_ci(x, lo=2.5, hi=97.5):
    """百分位区间。用 nanpercentile 而非 percentile：某个域的图若在某次自助重采样里
    一张都没抽到，该域的均值是 NaN；`M` 那一列取的是跨域最大值，不受影响，
    但**逐域**的列会，故统一按非缺失值取分位（对 `M` 与 `np.percentile` 逐位相同）。"""
    return float(np.nanpercentile(x, lo)), float(np.nanpercentile(x, hi))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="tid2013")
    ap.add_argument("--boot", type=int, default=1000, help="自助重采样次数")
    ap.add_argument("--limit-pairs", type=int, default=0, help=">0 时只跑前 N 对（冒烟）")
    args = ap.parse_args()

    cfg_path = PT.resolve(args.cfg)
    with io.open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg["seed"])
    thr = dict(cfg["criteria"])
    B = int(thr["c4_perm_B"])
    floor = 1.0 / (B + 1)
    out_dir = PT.resolve(args.out)
    os.makedirs(out_dir, exist_ok=True)

    table = PT.build_all(cfg, verbose=False)[args.dataset]
    if table.domains is None:
        sys.exit("数据集 %s 没有域标签，判据 4 未被调用（与 KonIQ 同形）" % args.dataset)
    names = table.judge_names
    folds = SS.make_folds(table, cfg["cv"]["n_splits"], seed)
    ranges = {j: cfg["judges"][j]["score_range"] for j in names}
    y = table.y
    n = table.n
    dom = np.asarray(table.domains)
    cats, codes = np.unique(dom, return_inverse=True)
    k = cats.size
    pairs = list(itertools.combinations(range(len(names)), 2))
    if args.limit_pairs:
        pairs = pairs[:args.limit_pairs]

    logf = open(os.path.join(out_dir, "c4_global.log"), "w", encoding="utf-8",
                newline="\n")

    def log(m):
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), m)
        logf.write(line + "\n")
        logf.flush()
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"), flush=True)

    # 检验数对照表（回答「分辨率极限」）—— 只依赖 B，先打印。
    lines = []
    for m, path in ((6048, "Table VI 原设计（判定×域）"),
                    (252, "判据 4 实际（每判定一次）"),
                    (len(pairs), "本脚本（评委对层面）")):
        t = 0.05 / m
        ratio = floor / t
        lines.append({"m": m, "path": path, "threshold": t, "p_floor": floor,
                      "floor_over_threshold": ratio, "reachable": bool(t > floor)})
    log("检验数 → Bonferroni 阈值 → 对地板（B=%d，地板=%.6e）" % (B, floor))
    for r in lines:
        log("  m=%-5d %-26s 0.05/m=%.4e  地板/阈值=%.3f  %s"
            % (r["m"], r["path"], r["threshold"], r["floor_over_threshold"],
               "可达" if r["reachable"] else "**不可达**"))

    rows, boot_rows, domain_rows = [], [], []
    t0 = time.time()
    for pi, (ia, ib) in enumerate(pairs):
        pk = "%s | %s" % (names[ia], names[ib])
        xa, xb = table.X[ia], table.X[ib]
        ds, betas, fam_names = [], [], []
        for mp in cfg["mappings"]:
            za = np.full(n, np.nan)
            zb = np.full(n, np.nan)
            for cal, test in folds:
                ma = MF.fit_all(mp, xa[cal], y[cal], judge_range=ranges[names[ia]])
                mb = MF.fit_all(mp, xb[cal], y[cal], judge_range=ranges[names[ib]])
                za[test] = ma.transform(xa[test])
                zb[test] = mb.transform(xb[test])
            d = za - zb
            _, beta, _ = CC._decompose(d, y)
            ds.append(d)
            betas.append(beta)
            fam_names.append(mp)

            # ── 自助 CI：按图有放回重采样，重估 β̂ 与域均值 ──
            rng_b = np.random.default_rng(seed + 700000 + 1000 * pi
                                          + MF.MAPPING_NAMES.index(mp))
            Ms = np.empty(args.boot, dtype=np.float64)
            Dev = np.empty((args.boot, k), dtype=np.float64)   # 每次重复的 k 个 δ̂(c)
            for b in range(args.boot):
                idx = rng_b.integers(0, n, n)
                db, yb, cb = d[idx], y[idx], codes[idx]
                _, bb, _ = CC._decompose(db, yb)
                vb = db - float(bb) * yb
                dev_b = _domain_means(vb, cb, k) - float(np.mean(vb))
                Dev[b] = dev_b
                Ms[b] = float(np.nanmax(np.abs(dev_b)))
            lo, hi = _percentile_ci(Ms)
            v_obs = d - float(beta) * y
            dev_obs = _domain_means(v_obs, codes, k) - float(np.mean(v_obs))
            obs_single = float(np.nanmax(np.abs(dev_obs)))
            boot_rows.append({"pair": pk, "mapping": mp,
                              "M": obs_single, "M_ci_lo": lo, "M_ci_hi": hi,
                              "boot_sd": float(np.std(Ms)), "n_boot": args.boot})
            # ── 各域效应量 δ̂(c) 及其区间：与上面是**同一批重复**，零额外成本 ──
            for ci_, cname in enumerate(cats):
                col = Dev[:, ci_]
                if np.all(np.isnan(col)):
                    continue
                dlo, dhi = _percentile_ci(col)
                domain_rows.append({
                    "pair": pk, "mapping": mp, "domain": str(cname),
                    "delta": float(dev_obs[ci_]), "ci_lo": dlo, "ci_hi": dhi,
                    "boot_sd": float(np.std(col)), "n_boot": args.boot,
                    "excludes_zero": bool(dlo > 0.0 or dhi < 0.0),
                    "is_worst": bool(abs(abs(dev_obs[ci_]) - obs_single) < 1e-12)})

        # ── 对级全局置换检验：同一置换作用于全部 7 族 ──
        stats = [(d - float(beta) * y, float(np.mean(d - float(beta) * y)))
                 for d, beta in zip(ds, betas)]
        obs = _pair_stat(stats, codes, k)
        rng = np.random.default_rng(seed + 900000 + 1000 * pi)
        null = np.empty(B, dtype=np.float64)
        for b in range(B):
            null[b] = _pair_stat(stats, codes[rng.permutation(codes.size)], k)
        exceed = int(np.sum(null >= obs))
        p = float((1 + exceed) / (B + 1))
        rows.append({"pair": pk, "obs": obs, "p": p, "exceed": exceed, "B": B,
                     "n_families": len(fam_names), "n_domains": int(k),
                     "bonferroni_36": 0.05 / len(pairs),
                     "passes_bonferroni": bool(p < 0.05 / len(pairs)),
                     "at_floor": bool(abs(p - floor) < 1e-15)})
        log("  %2d/%2d %-42s M=%.6f p=%.6e 超=%d %.0fs"
            % (pi + 1, len(pairs), pk, obs, p, exceed, time.time() - t0))

    n_pass = sum(1 for r in rows if r["passes_bonferroni"])
    n_floor = sum(1 for r in rows if r["at_floor"])
    summary = {
        "dataset": args.dataset, "B": B, "p_floor": floor,
        "n_pairs": len(rows), "n_domains": int(k),
        "n_passes_bonferroni_36": n_pass, "n_at_floor": n_floor,
        "bonferroni_line": 0.05 / len(rows),
        "test_count_table": lines,
        "exceed_histogram": {str(v): sum(1 for r in rows if r["exceed"] == v)
                             for v in sorted({r["exceed"] for r in rows})},
        "rows": rows,
        "boot_cells": boot_rows,
        "n_domain_rows": len(domain_rows),
        "n_domain_ci_excludes_zero": sum(1 for r in domain_rows
                                         if r["excludes_zero"]),
        "domain_ci_note": ("δ̂(c) 的区间由与 M 相同的那一批自助重复给出（百分位法，"
                           "2.5/97.5），不是另跑一次重采样"),
    }
    with io.open(os.path.join(out_dir, "c4_global.json"), "w", encoding="utf-8",
                 newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    # 各域效应量整表 —— 补充材料可直接排版，正文只印摘要（2026-11-09 复审的 (b) 项）
    with io.open(os.path.join(out_dir, "c4_domain_ci.csv"), "w", encoding="utf-8",
                 newline="\n") as f:
        f.write("pair,mapping,domain,delta,ci_lo,ci_hi,boot_sd,n_boot,"
                "excludes_zero,is_worst\n")
        for r in domain_rows:
            f.write("%s,%s,%s,%.8f,%.8f,%.8f,%.8f,%d,%d,%d\n"
                    % (r["pair"], r["mapping"], r["domain"], r["delta"],
                       r["ci_lo"], r["ci_hi"], r["boot_sd"], r["n_boot"],
                       int(r["excludes_zero"]), int(r["is_worst"])))

    log("对级全局检验：%d/%d 过 36 路 Bonferroni（线 %.4e）；落在地板的 %d 条"
        % (n_pass, len(rows), 0.05 / len(rows), n_floor))
    log("超越次数直方图：%s" % summary["exceed_histogram"])
    log("各域效应量：%d 行（%d 族 × %d 对 × %d 域），其区间不含 0 的 %d 行"
        % (len(domain_rows), len(cfg["mappings"]), len(rows), int(k),
           summary["n_domain_ci_excludes_zero"]))
    logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
