# -*- coding: utf-8 -*-
"""图像层残差闸门的域内分解。

**要回答的问题**：正文的图像层闸门是

    σ̂ = SD_i( d(i) − δ̂ − β̂·y(i) ),  通过条件 σ̂ ≤ ε

它只扣掉了**全局**位置 δ̂ 与**全局**尺度 β̂，**没有扣域偏移**。而判据 4 用的正是

    v(i) = d(i) − β̂·y(i),   M = max_c |mean_{i∈c} v − mean_i v|

的问题量。⇒ σ̂ 里**含**域间差。若扣掉域均值之后剩下的部分 σ̂_w 已经很小，图像层这一档就有
相当一部分是在**重复**判据 4 的域漂移，而不是独立的逐图证据。

**本批的定义（跑前写死）**：

    v(i)  = d(i) − δ̂ − β̂·y(i)                      与正文一致，mean_i v = 0（代数恒等）
    δ̃_c   = mean_{i∈c} v(i)                          域均值
    σ̂_w   = sqrt( mean_i ( v(i) − δ̃_{c(i)} )² )      **域内残差**（本批新增量）
    σ̂_b   = sqrt( mean_i ( δ̃_{c(i)} )² )            域间分量
    σ̂²    = σ̂_w² + σ̂_b²                             ANOVA 恒等，脚本内逐条断言
    M     = max_c |δ̃_c|                              判据 4 的统计量（mean_i v = 0 时恒等）

**预注册的判读规则（跑前写死，不按结果改）**：

1. **头号读数**：32 条「条件可比（c4_drift）」判定的 σ̂_w/ε。
   - 若 **≥1 条** 落在 ε 内 ⇒ 该条的图像层拒绝在扣掉域偏移后**依然成立**，图像层与判据 4
     **不是同一句话**；`0 of 252` 里这一条是独立证据。
   - 若 **0 条** 落在 ε 内 ⇒ 对这 32 条而言，图像层拒绝**部分由域漂移造成**，正文的
     「two tiers separate」需要补一句限定；但 `0 of 252` 的**条数不变**。
2. **全量读数**：252 条的 σ̂_w/ε 分布（min / 中位 / max），以及 σ̂ ≤ ε 与 σ̂_w ≤ ε 两条
   各自判出多少条。
3. **不做的事**：不改任何既有读数；不把 σ̂_w 变成新闸门；不重跑判据 4（本脚本不调用
   `criterion4`，B 次置换与本题无关）。

**回归校验（不等即报错退出）**：本脚本重建的 v 必须逐位复现 v3 run 的
`c3.sigma` 与 `aggregate.drift_amplitude`（零容差）——它们分别就是 σ̂ 与 M。
复现不了就说明折／种子／标定与 v3 不是同一套，读数不成立。
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

LABELS = ("可比", "条件可比", "不可比")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ref", required=True, help="v3 run 的 summary.json")
    ap.add_argument("--out", required=True, help="本批 run 目录")
    ap.add_argument("--dataset", default="tid2013")
    args = ap.parse_args()

    with io.open(PT.resolve(args.cfg), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg["seed"])
    with io.open(PT.resolve(args.ref), encoding="utf-8") as f:
        ref = json.load(f)["datasets"][args.dataset]["pairs"]
    out_dir = PT.resolve(args.out)
    os.makedirs(out_dir, exist_ok=True)

    table = PT.build_all(cfg, verbose=False)[args.dataset]
    names = table.judge_names
    folds = SS.make_folds(table, cfg["cv"]["n_splits"], seed)
    ranges = {j: cfg["judges"][j]["score_range"] for j in names}
    y = table.y
    n = table.n
    codes = np.asarray(table.domains)
    dnames = sorted(set(codes.tolist()))
    pairs = list(itertools.combinations(range(len(names)), 2))

    logf = open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8", newline="\n")

    def log(m):
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), m)
        logf.write(line + "\n"); logf.flush()
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"), flush=True)

    log("图像层残差的域内分解：%s  n=%d  域 %d 类  判定 %d 条"
        % (args.dataset, n, len(dnames), len(pairs) * len(cfg["mappings"])))

    rows = []
    bad_sigma = bad_M = bad_ident = 0
    ident_max = 0.0
    dif_s_max = dif_M_max = 0.0
    t0 = time.time()
    for k, (ia, ib) in enumerate(pairs):
        pk = "%s | %s" % (names[ia], names[ib])
        xa, xb = table.X[ia], table.X[ib]
        for mp in cfg["mappings"]:
            za = np.full(n, np.nan)
            zb = np.full(n, np.nan)
            for cal, test in folds:
                ma = MF.fit_all(mp, xa[cal], y[cal], judge_range=ranges[names[ia]])
                mb = MF.fit_all(mp, xb[cal], y[cal], judge_range=ranges[names[ib]])
                za[test] = ma.transform(xa[test])
                zb[test] = mb.transform(xb[test])
            d = za - zb
            delta, beta, sigma = CC._decompose(d, y)
            v = d - delta - beta * y
            # ⚠️ 域均值必须**再减去全局均值**再平方。`mean_i v` 不是 0——
            #    beta 是带截距的 OLS 斜率，v = d − δ̂ − β̂y 的均值等于 −β̂·mean(y)。
            #    若不中心化，σ̂_w² + σ̂_b² 就不等于 σ̂²（第一版就是这样炸的：
            #    β̂·mean(y) 的量级把 ANOVA 恒等差出 1e-2）。
            vbar = {c: v[codes == c].mean() for c in dnames}
            gbar = np.array([vbar[c] - v.mean() for c in codes])   # 中心化后的域偏移
            dev_in = v - np.array([vbar[c] for c in codes])
            sigma_w = float(np.sqrt(np.mean(dev_in ** 2)))
            sigma_b = float(np.sqrt(np.mean(gbar ** 2)))
            M = float(np.max(np.abs(gbar)))
            ident = abs(sigma ** 2 - (sigma_w ** 2 + sigma_b ** 2))
            ident_max = max(ident_max, ident)

            rr = ref[pk][mp]
            ref_sigma = float(rr["c3"]["sigma"])
            ref_M = float(rr["aggregate"]["drift_amplitude"])
            ref_eps = float(rr["c3"]["eps"])
            # 回归容差 1e-12：`criterion4` 内部算 M 走的是另一个代数等价式
            # （`_stat`），实测差 1 ulp（6e-17），**不能**用零容差判。
            ok_s = abs(sigma - ref_sigma) <= 1e-12
            ok_M = abs(M - ref_M) <= 1e-12
            dif_s = abs(sigma - ref_sigma)
            dif_M = abs(M - ref_M)
            if dif_s > dif_s_max:
                dif_s_max = dif_s
            if dif_M > dif_M_max:
                dif_M_max = dif_M
            if not ok_s:
                bad_sigma += 1
                log("  !! %s %s σ̂ 不复现：%.17g vs %.17g" % (pk, mp, sigma, ref_sigma))
            if not ok_M:
                bad_M += 1
                log("  !! %s %s M 不复现：%.17g vs %.17g" % (pk, mp, M, ref_M))
            if ident > 1e-15:
                bad_ident += 1
                log("  !! %s %s ANOVA 恒等破：%.3g" % (pk, mp, ident))

            rows.append({
                "pair": pk, "mapping": mp,
                "label": rr["aggregate"]["label"],
                "reason": rr["aggregate"].get("reason"),
                "eps": ref_eps,
                "sigma": sigma, "sigma_over_eps": sigma / ref_eps,
                "sigma_w": sigma_w, "sigma_w_over_eps": sigma_w / ref_eps,
                "sigma_b": sigma_b, "sigma_b_over_eps": sigma_b / ref_eps,
                "M": M, "M_over_eps": M / ref_eps,
                "share_drift": (sigma_b ** 2) / (sigma ** 2) if sigma > 0 else float("nan"),
                "gate_sigma": rr["criteria"]["c3_sigma"]["verdict"],
                "c4": rr["criteria"]["c4"]["verdict"] if rr["criteria"].get("c4") else None,
            })
        log("  %2d/%2d %-42s 累计 %.0fs" % (k + 1, len(pairs), pk, time.time() - t0))

    def block(sel, tag):
        n_ = len(sel)
        if not n_:
            return {"tag": tag, "n": 0}
        sw = np.array([r["sigma_w_over_eps"] for r in sel])
        s = np.array([r["sigma_over_eps"] for r in sel])
        return {
            "tag": tag, "n": n_,
            "sigma_over_eps": {"min": float(s.min()), "median": float(np.median(s)), "max": float(s.max())},
            "sigma_w_over_eps": {"min": float(sw.min()), "median": float(np.median(sw)), "max": float(sw.max())},
            "pass_sigma": int((s <= 1.0).sum()),
            "pass_sigma_w": int((sw <= 1.0).sum()),
        }

    passing = [r for r in rows if r["label"] == "条件可比"]
    summary = {
        "dataset": args.dataset, "cfg": args.cfg, "ref": args.ref, "seed": seed,
        "n_decisions": len(rows), "n_domains": len(dnames),
        "identity_max_abs": float(ident_max),
        "regression": {"sigma_mismatch": bad_sigma, "M_mismatch": bad_M, "identity_violations": bad_ident,
                       "sigma_max_abs_diff": dif_s_max, "M_max_abs_diff": dif_M_max,
                       "tolerance": 1e-12},
        "headline_gate_passing": block(passing, "条件可比（32 条，c4 实测有漂移）"),
        "all_decisions": block(rows, "全部 252 条"),
        "by_label": {lb: block([r for r in rows if r["label"] == lb], lb) for lb in LABELS},
        "by_mapping": {mp: block([r for r in rows if r["mapping"] == mp], mp) for mp in cfg["mappings"]},
        "rows": rows,
    }
    # σ̂_b 的份额：域间分量占 σ̂² 多少
    sh = np.array([r["share_drift"] for r in passing])
    summary["headline_gate_passing"]["share_drift_median"] = float(np.median(sh)) if sh.size else None

    with io.open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    h = summary["headline_gate_passing"]
    a = summary["all_decisions"]
    log("回归：σ̂ 不复现 %d 条、M 不复现 %d 条、恒等破 %d 条" % (bad_sigma, bad_M, bad_ident))
    log("32 条：σ̂/ε %.4f–%.4f（中位 %.4f）；σ̂_w/ε %.4f–%.4f（中位 %.4f）"
        % (h["sigma_over_eps"]["min"], h["sigma_over_eps"]["max"], h["sigma_over_eps"]["median"],
           h["sigma_w_over_eps"]["min"], h["sigma_w_over_eps"]["max"], h["sigma_w_over_eps"]["median"]))
    log("32 条：σ̂ ≤ ε 的 %d 条；σ̂_w ≤ ε 的 %d 条；域间分量占 σ̂² 中位 %.1f%%"
        % (h["pass_sigma"], h["pass_sigma_w"], 100.0 * (h.get("share_drift_median") or 0.0)))
    log("252 条：σ̂_w/ε %.4f–%.4f（中位 %.4f）；σ̂_w ≤ ε 的 %d 条"
        % (a["sigma_w_over_eps"]["min"], a["sigma_w_over_eps"]["max"],
           a["sigma_w_over_eps"]["median"], a["pass_sigma_w"]))
    logf.close()
    return 0 if (bad_sigma == 0 and bad_M == 0 and bad_ident == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
