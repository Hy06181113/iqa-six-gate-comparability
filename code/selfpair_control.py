# -*- coding: utf-8 -*-
"""A-vs-A 正向对照（同一位评委与它自己配对，只换标定族）。

要回答的是：主跑的「0/252 不可比」里，有多少是**评委之间的差异**、
多少是**标定族的选择**。做法是把 A 与它自己配对，令两侧用**不同**的标定族。

两个变体（缺一不可）：

    A0  同族自配对   f_A = f_B = f（7 个族各一次）
        harness 自检：x_A ≡ x_B ⇒ d ≡ 0 ⇒ 判据 3 三项应精确为 0、六门全过。
        同时，A0 的 `c2b` 逐侧 p95 与 `c1a` 给出**每个 (评委, 族) 单元自己稳不稳**——
        这是 A1 条件样本的分母，A1 本身给不出来。

    A1  跨族自配对   f_A ≠ f_B（C(7,2) = 21 个**无序**族对）
        主量。无序即可：d = z_A − z_B 换序只翻 δ̂/β̂ 的符号，
        |δ̂|、β̂·span、σ̂、判据 1b/2b、判据 4 的 max|·| 全不变 ⇒ 21 个而非 42 个。

⚠️ **读法边界（不许越过）**：主跑的每一条判定两侧**永远是同一个族**。故 A1 的失败
   只能说明「判决对族的选择敏感」，**不能**反推「主跑的 0/252 是标定 artifact」。
⚠️ **σ̂ 不是门槛**（GATE_KEYS 只有 c1a/c1b/c2a/c2b/c3_delta/c3_beta）。族的形状差异
   主要落在 σ̂ 上，而它不进判决 ⇒ A1「通过」**不等于**「两族输出一致」。
⚠️ **A1 的失败会混进「族本身不稳」**（c2b 取两侧最大值），故必须同时给**条件样本**：
   只统计「该评委的两个族各自单侧全门通过」的单元。只报全样本会夸大。

规格在读结果之前写定，不按结果改。
"""
import argparse
import csv
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
from diagnose_comparability import run_pair_mapping, stage_script   # noqa: E402

#: 预注册的判定界线（跑之前写死，不看结果再定）
MAJORITY = 2.0 / 3.0

#: 自配对的种子偏移。主跑用 `seed + 1000*k + idx(mp)`（k ≤ 35 ⇒ ≤ 35006），
#: 这里整体抬到 900000 以上，两者不可能相撞。
SEED_BASE = 900000


def _exceed_from_p(p, B):
    """由 p 反推超越次数 c：p = (1 + c) / (B + 1)。只用于记账，不改判据。"""
    return int(round(float(p) * (B + 1))) - 1


def collect(table, ja, folds, thr, cfg, seed, rng_b):
    """对一位评委跑完 A0（7 条）与 A1（21 条）。"""
    agg_cfg, iso_cfg = cfg.get("aggregate"), cfg.get("iso")
    r = cfg["judges"][table.judge_names[ja]]["score_range"]
    out = []
    for fam_a, fam_b in [(m, m) for m in cfg["mappings"]] + \
                        list(itertools.combinations(cfg["mappings"], 2)):
        i_a = MF.MAPPING_NAMES.index(fam_a)
        i_b = MF.MAPPING_NAMES.index(fam_b)
        s = seed + SEED_BASE + 1000 * rng_b + 100 * i_a + i_b
        res = run_pair_mapping(table, ja, ja, fam_a, folds, thr, s, r, r,
                               agg_cfg=agg_cfg, iso_cfg=iso_cfg, mapping_b=fam_b)
        ag = res["aggregate"]
        c4d = res["c4_detail"] or {}
        p = c4d.get("p_value")
        B = int(thr["c4_perm_B"])
        out.append({
            "judge": table.judge_names[ja],
            "variant": "A0" if fam_a == fam_b else "A1",
            "family_a": fam_a, "family_b": fam_b,
            "label": ag["label"], "reason": ag["reason"],
            "c4": ag["c4"], "drift_domains": ag["drift_domains"],
            "undecidable": ag["undecidable"],
            "gate": dict(ag["gate"]),
            "binding": sorted(k for k, v in ag["gate"].items() if v != "pass"),
            "c2b_p95_a": res["criteria"]["c2b"]["detail"]["p95_pointwise_sd_a"],
            "c2b_p95_b": res["criteria"]["c2b"]["detail"]["p95_pointwise_sd_b"],
            "c1a": res["criteria"]["c1a"]["value"],
            "c1b": res["criteria"]["c1b"]["value"],
            "eps": res["c3"]["eps"],
            "delta": res["c3"]["delta"], "beta_amp": res["c3"]["beta_amp"],
            "sigma": res["c3"]["sigma"],
            "abs_delta_over_eps": res["c3"]["abs_delta_over_eps"],
            "sigma_over_eps": res["c3"]["sigma_over_eps"],
            "c4_p": p,
            "c4_exceed": (_exceed_from_p(p, B) if p is not None else None),
            "seed": s,
        })
    return out


def standalone_ok(rows_a0, thr):
    """每个 (评委, 族) 单元「自己稳不稳」：c2b 逐侧 p95 ≤ 阈值 且 c1a ≤ 阈值。

    只从 A0 取：A0 的 `c2b_p95_a` 就是该族在该评委的原始分上 5 折重拟合的逐点 SD，
    `c1a` 是该评委自己的外推比例。A1 给不出这两个量（它两侧已经不同族）。
    """
    okm = {}
    for row in rows_a0:
        f = row["family_b"]                       # A0 里 fam_a == fam_b
        okm[(row["judge"], f)] = bool(
            row["c2b_p95_a"] <= thr["c2_stability_max"] and row["c1a"] <= thr["c1_extrapolation_max"])
    return okm


def summarize(rows, okm):
    def _count(sub):
        n = len(sub)
        lab = {}
        for r in sub:
            lab[r["label"]] = lab.get(r["label"], 0) + 1
        return n, lab

    a0 = [r for r in rows if r["variant"] == "A0"]
    a1 = [r for r in rows if r["variant"] == "A1"]
    cond = [r for r in a1 if okm[(r["judge"], r["family_a"])] and okm[(r["judge"], r["family_b"])]]

    n0, l0 = _count(a0)
    n1, l1 = _count(a1)
    nc, lc = _count(cond)
    binding = {}
    for r in a1:
        for g in r["binding"]:
            binding[g] = binding.get(g, 0) + 1
        if r["c4"] == "fail":
            binding["c4(p only)"] = binding.get("c4(p only)", 0) + 1
        if r["drift_domains"]:
            binding["c4(eps drift)"] = binding.get("c4(eps drift)", 0) + 1
    return {
        "A0": {"n": n0, "labels": l0,
               "all_comparable": l0.get("可比", 0) == n0,
               "max_abs_c3": max(max(abs(r["delta"]), abs(r["sigma"]),
                                     abs(r["beta_amp"] or 0.0)) for r in a0)},
        "A1_full": {"n": n1, "labels": l1,
                    "pass_rate": (l1.get("可比", 0) / n1) if n1 else None},
        "A1_conditional": {"n": nc, "labels": lc,
                           "pass_rate": (lc.get("可比", 0) / nc) if nc else None,
                           "definition": "两侧的族在 standalone_ok 里都为 True"},
        "A1_binding": binding,
        "standalone_cells": {"%s | %s" % k: v for k, v in sorted(okm.items())},
        "preregistered_majority": MAJORITY,
    }


def write_csv(path, rows):
    cols = ["dataset", "judge", "variant", "family_a", "family_b", "label", "reason",
            "c4", "drift_domains", "binding", "c1a", "c1b", "c2b_p95_a", "c2b_p95_b",
            "eps", "delta", "beta_amp", "sigma", "abs_delta_over_eps", "sigma_over_eps",
            "c4_p", "c4_exceed", "seed"]

    def _n(x):
        return "" if x is None else ("%.12g" % x if isinstance(x, float) else str(x))

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r["dataset"], r["judge"], r["variant"], r["family_a"], r["family_b"],
                        r["label"], r["reason"], r["c4"], ",".join(r["drift_domains"]) or "—",
                        ",".join(r["binding"]) or "—",
                        _n(r["c1a"]), _n(r["c1b"]), _n(r["c2b_p95_a"]), _n(r["c2b_p95_b"]),
                        _n(r["eps"]), _n(r["delta"]), _n(r["beta_amp"]), _n(r["sigma"]),
                        _n(r["abs_delta_over_eps"]), _n(r["sigma_over_eps"]),
                        _n(r["c4_p"]), _n(r["c4_exceed"]), r["seed"]])


def write_report(path, summary, ds):
    s = summary["datasets"][ds]
    L = ["# A-vs-A 自配对对照 —— %s" % ds, "",
         "配置 `%s`；%d 条判定（每位评委 7 条 A0 ＋ 21 条 A1）。" % (summary["config"], s["A0"]["n"] + s["A1_full"]["n"]),
         "", "## A0 同族自配对（harness 自检）", "",
         "n = %d，标签分布 %s，全 comparable = **%s**；max|c3 分量| = %.3e。"
         % (s["A0"]["n"], s["A0"]["labels"], s["A0"]["all_comparable"], s["A0"]["max_abs_c3"]),
         "", "## A1 跨族自配对", "",
         "| 口径 | n | 可比 | 条件可比 | 不可比 | 可比率 |", "|---|---|---|---|---|---|",
         "| 全样本 | %d | %d | %d | %d | %s |"
         % (s["A1_full"]["n"], s["A1_full"]["labels"].get("可比", 0),
            s["A1_full"]["labels"].get("条件可比", 0),
            s["A1_full"]["labels"].get("不可比", 0),
            "—" if s["A1_full"]["pass_rate"] is None else "%.3f" % s["A1_full"]["pass_rate"]),
         "| 条件样本 | %d | %d | %d | %d | %s |"
         % (s["A1_conditional"]["n"], s["A1_conditional"]["labels"].get("可比", 0),
            s["A1_conditional"]["labels"].get("条件可比", 0),
            s["A1_conditional"]["labels"].get("不可比", 0),
            "—" if s["A1_conditional"]["pass_rate"] is None else "%.3f" % s["A1_conditional"]["pass_rate"]),
         "", "条件样本 = 两侧的族在该评委上各自单侧全门通过（c2b ≤ %.2f 且 c1a ≤ %.2f）。"
         % (summary["thresholds"]["c2_stability_max"], summary["thresholds"]["c1_extrapolation_max"]),
         "", "### A1 的绑定门计数", "",
         "| 门 | 次数 |", "|---|---|"]
    for k, v in sorted(s["A1_binding"].items(), key=lambda kv: -kv[1]):
        L.append("| `%s` | %d |" % (k, v))
    L += ["", "预注册的「多数」界线 = %.3f。**结果不在本报告里解读**——解读与预注册结局"
          "的对照由论文给出。" % MAJORITY,
          "", "⚠️ 主跑每条判定的两侧永远是**同一个族**；A1 的失败只能说「判决对族的选择敏感」，"
          "不能反推主跑结论是标定 artifact。⚠️ `σ̂` 不是门槛，A1「通过」不等于两族输出一致。", ""]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", default=None, help="逗号分隔，默认配置里全部")
    args = ap.parse_args()

    cfg_path = PT.resolve(args.cfg)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg["seed"])
    thr = cfg["criteria"]
    out_dir = PT.resolve(args.out)
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8", newline="\n")

    def log(msg):
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
        logf.write(line + "\n")
        logf.flush()
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"), flush=True)

    log("A-vs-A 自配对对照 —— 配置 %s" % cfg_path)
    log("种子 %d；折数 %d；族 %s" % (seed, cfg["cv"]["n_splits"], cfg["mappings"]))
    log("A0 = %d 条/评委；A1 = C(%d,2) = %d 条/评委"
        % (len(cfg["mappings"]), len(cfg["mappings"]),
           len(cfg["mappings"]) * (len(cfg["mappings"]) - 1) // 2))

    tables = PT.build_all(cfg, verbose=False)
    want = args.datasets.split(",") if args.datasets else list(tables)

    summary = {"config": os.path.relpath(cfg_path, PT.REPO_ROOT).replace(os.sep, "/"),
               "seed": seed, "schema_version": 1,
               "mappings": cfg["mappings"],
               "thresholds": {k: thr[k] for k in ("c1_extrapolation_max", "c1_overlap_min",
                                                  "c2_stability_max", "c4_perm_B", "c4_alpha")},
               "preregistered_majority": MAJORITY,
               "note": "主跑两侧同族；本对照两侧可不同族。A1 失败不得反推主跑是标定 artifact。",
               "datasets": {}}
    all_rows = []
    t0 = time.time()
    for ds in want:
        table = tables[ds]
        folds = SS.make_folds(table, cfg["cv"]["n_splits"], seed)
        log("=== %s（n=%d，%d 位评委）===" % (ds, table.n, len(table.judge_names)))
        rows = []
        for ja in range(len(table.judge_names)):
            rows.extend(collect(table, ja, folds, thr, cfg, seed, ja))
            log("  %2d/%2d  %-16s  A0 %d + A1 %d 条  累计 %.1fs"
                % (ja + 1, len(table.judge_names), table.judge_names[ja],
                   len(cfg["mappings"]),
                   len(cfg["mappings"]) * (len(cfg["mappings"]) - 1) // 2, time.time() - t0))
        for r in rows:
            r["dataset"] = ds
        okm = standalone_ok([r for r in rows if r["variant"] == "A0"], thr)
        summary["datasets"][ds] = summarize(rows, okm)
        summary["datasets"][ds]["rows"] = rows
        all_rows.extend(rows)
        s = summary["datasets"][ds]
        log("  A0 全 comparable = %s；max|c3 分量| = %.3e"
            % (s["A0"]["all_comparable"], s["A0"]["max_abs_c3"]))
        log("  A1 全样本 %d 条：%s" % (s["A1_full"]["n"], s["A1_full"]["labels"]))
        log("  A1 条件样本 %d 条：%s" % (s["A1_conditional"]["n"], s["A1_conditional"]["labels"]))
        log("  A1 绑定门：%s" % s["A1_binding"])

    write_csv(os.path.join(out_dir, "pairs.csv"), all_rows)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    for ds in want:
        write_report(os.path.join(out_dir, "report-%s.md" % ds), summary, ds)

    stage_script(out_dir, cfg_path, log, note=(
        "本 run 由 `selfpair_control.py` 驱动。`run_pair_mapping` 相对主跑版本的唯一改动是"
        "新增可选参数 `mapping_b=None`（缺省两侧同族）；主路径行为不变的回归闸门见"
        "该批 run 记录里对本脚本的说明。\n\n"
        "⚠️ 本对照与主跑的读法边界：主跑每条判定的两侧**永远是同一个族**；"
        "A1（跨族）的失败只能说明「判决对族的选择敏感」，不能反推主跑结论是标定 artifact。"))
    # 补登本脚本自身（stage_script 的 MODULE_FILES 不含它）
    import hashlib
    import shutil
    me = os.path.abspath(__file__)
    shutil.copyfile(me, os.path.join(out_dir, "_script", os.path.basename(me)))
    h = hashlib.sha256(open(me, "rb").read()).hexdigest()
    with open(os.path.join(out_dir, "_script", "README.md"), "a",
              encoding="utf-8", newline="\n") as f:
        f.write("\n## 补登（stage_script 的 MODULE_FILES 不含本脚本）\n\n")
        f.write("| 文件 | 当次 sha256（前 16 位） | 说明 |\n|---|---|---|\n")
        f.write("| `%s` | `%s…` | 本 run 的驱动器；与本目录内副本逐字节相同（运行时刻） |\n"
                % (os.path.basename(me), h[:16]))

    log("产物：pairs.csv / summary.json / report-*.md / run.log / _script/ → %s" % out_dir)
    logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
