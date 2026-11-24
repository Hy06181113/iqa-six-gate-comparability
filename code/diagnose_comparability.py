# -*- coding: utf-8 -*-
"""P1 可比性诊断 —— 端到端 CLI。

     python 30-experiments/metrics/diagnose_comparability.py \
            --cfg 30-experiments/configs/20261015-p1diagnose.yaml \
            --out 30-experiments/runs/20261015-p1diagnose-tid2013

流程 = 六步里可离线做的五步：
    ① 分数采集（pred_table） ② 分层切分（stratified_split） ③ 边际校准（mapping_families）
    ④ 可比性诊断（comparability_criteria） ⑤ 条件校准与聚合（P3，本模块不做）
    ⑥ 失效条件刻画 = 判据 3 的 σ̂ 分解 + 判据 4 的跨失真漂移（本模块的产出）

**阶段 C 的三组对照都在本模块内跑，结果无论正负都写进产物**：
    · 阳性对照：B := a·A + b（已知仿射）⇒ 判据 3 应全过
    · 阴性对照：把 B 的分数**打乱**（保边际、破对应）⇒ 判据 3 应挂尺度项与残差项、位置项通过。
      ⚠️ 打乱方式**随数据集而变**，产物以 `synthetic_negative_mode` 标注：
      TID2013 有域标签 ⇒ `by_domain`（块内打乱）；KonIQ 无任何标签 ⇒ `global`（全局打乱，
      保的是全局边际）。两者**不是同一个对照**，跨数据集比时不得混读。
      〔2026-10-16 修：本段原先被 `table.domains is not None` 这道过宽的闸门整个跳过，
      连不需要任何标签的**阳性**对照也一起挡了 ⇒ KonIQ 此前一条合成对照都没跑。〕
    · 方法学对照：成对选择准确率在族 0–5 上应与映射无关，
      而判据 3 的三分量有区分 —— 本模块**实测**这一点并断言。
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

PAIR_NAMES = ("A", "B")


# ───────────────────────────── 单对 × 单映射 ─────────────────────────────
def run_pair_mapping(table, ia, ib, mapping, folds, thr, seed, judge_range_a, judge_range_b,
                     x_override=None, agg_cfg=None, iso_cfg=None, mapping_b=None):
    """跑完一对评委在一个映射族上的 5 折，返回判据结果与诊断量。

    `x_override` = (x_a, x_b)，用于合成对照（替换原始分数）。
    `agg_cfg` / `iso_cfg` = config 的 `aggregate` / `iso` 两段（三分类与可互换区间的
    跑前预注册口径）。缺省时用模块内默认值。
    `mapping_b` = B 侧单独指定的族，**缺省 `None` 即两侧同族**（= 全部历史 run 的行为）。
    只有 A-vs-A 自配对对照（`selfpair_control.py`）会传它；主路径 `run_dataset` 一律不传。
    """
    xa = table.X[ia] if x_override is None else x_override[0]
    xb = table.X[ib] if x_override is None else x_override[1]
    y = table.y
    n = table.n

    zA_test = np.full(n, np.nan)
    zB_test = np.full(n, np.nan)
    cal_chunks, zA_cal, zB_cal, cal_idx_all = [], [], [], []
    maps_a, maps_b, folds_a, folds_b = [], [], [], []
    pa_fold = []
    # 秩不变性**必须折内测**：拼接口径下相邻两张图可能来自不同折、用不同映射，
    # 跨折比较真会翻序（那正是判据 2b 在量的「CV 不稳」），与浮点噪声无关。
    flip_acc = {}

    for cal, test in folds:
        ma = MF.fit_all(mapping, xa[cal], y[cal], judge_range=judge_range_a)
        mb = MF.fit_all(mapping if mapping_b is None else mapping_b,
                        xb[cal], y[cal], judge_range=judge_range_b)
        maps_a.append(ma)
        maps_b.append(mb)
        zA_test[test] = ma.transform(xa[test])
        zB_test[test] = mb.transform(xb[test])
        zA_cal.append(ma.transform(xa[cal]))
        rf = CC.rank_flip_stats(ma.transform(xa[test]), xa[test])
        flip_acc = {k: flip_acc.get(k, 0) + v for k, v in rf.items()}
        zB_cal.append(mb.transform(xb[cal]))
        cal_idx_all.append(cal)
        folds_a.append((xa[cal], xa[test]))
        folds_b.append((xb[cal], xb[test]))
        pa_fold.append(CC.pairwise_accuracy(ma.transform(xa[test]), y[test]))

    if np.any(~np.isfinite(zA_test)) or np.any(~np.isfinite(zB_test)):
        raise SystemExit("存在未填的样本外格子 ⇒ 拼接有漏")

    # 样本内对照取**第 1 折的**校准集：不拼接 —— 五折校准集拼起来会让同一张图出现 5 次，
    # 均值不受影响但离散度会被系统性低估，作为「样本内 vs 样本外」的对照会失真。
    cal0 = folds[0][0]
    d_test = zA_test - zB_test
    d_cal = zA_cal[0] - zB_cal[0]
    y_cal = y[cal0]

    blocks_test = np.asarray(table.groups) if table.groups is not None else None

    sd_y = float(np.std(y)) or 1.0
    spread_a = float(np.std(zA_test) / sd_y)
    spread_b = float(np.std(zB_test) / sd_y)

    c1 = CC.criterion1(folds_a, folds_b, zA_test, zB_test, thr)
    # ⚠️ 两侧各用自己的观测分范围（2026-10-30 修）：只取 A 的范围会让 B 的映射在自己的
    #    定义域外被评估，跨量纲对的 2b 因此整批假失败。见 `criterion2` 的 docstring。
    c2 = CC.criterion2(maps_a, maps_b,
                       (float(np.min(xa)), float(np.max(xa))),
                       (float(np.min(xb)), float(np.max(xb))), thr)
    c3 = CC.criterion3(d_cal, y_cal, d_test, y, blocks_test,
                       table.sigma_human, thr, seed,
                       spread_ratio_a=spread_a, spread_ratio_b=spread_b)
    _, beta, _ = CC._decompose(d_test, y)
    # ★ eps 必须先取出来再调 criterion4 —— 判据 4 已改判**幅度规则**
    #   `M > ε`，置换 p 只作诊断。旧顺序是 c4 之后才取 eps，现前移（值不变）。
    c3d, c3b, c3s = c3["c3_delta"], c3["c3_beta"], c3["c3_sigma"]
    eps = c3d["detail"].get("eps")
    if table.domains is not None:
        c4 = CC.criterion4(d_test, y, beta, table.domains, thr, seed, eps=eps)
    else:
        c4 = {"c4": {"verdict": "na", "value": None, "ci_lo": None, "ci_hi": None,
                     "threshold_ref": "KonIQ 无失真类型、无内容标注 ⇒ 结构上不可测",
                     "detail": {"reason": "no_domain_labels"}}}

    c3_sum = {
        "delta": c3d["value"], "beta": c3b["value"], "sigma": c3s["value"], "eps": eps,
        "abs_delta_over_eps": c3d["detail"].get("ratio_abs_delta_over_eps"),
        "beta_amp": c3b["detail"].get("beta_amp"), "y_span": c3b["detail"].get("y_span"),
        "sigma_over_eps": (c3s["value"] / eps) if eps else None,
        # σ̂ 已于 2026-10-16 退出门槛（只作分级读数）；它的 verdict 是 `info`。
        "sigma_verdict": c3s["verdict"],
    }

    crit = dict(list(c1.items()) + list(c2.items()) + list(c3.items()) + list(c4.items()))

    # ── 「可互换区间」与三分类（2026-10-16 新增）──
    # ⚠️ 退化闸门必须与判据 3 的退化护栏**同源** —— 用 `c3_delta.verdict == "na"` 判，
    #    不能自己另设一套。否则映射塌成常函数时 d ≡ 0 ⇒ P 假性 = 1、报「100% 可互换」。
    c3_degen = (c3d["verdict"] == "na")
    iso = CC.interchangeability(
        zA_test, zB_test, y, eps, domains=table.domains, cfg=iso_cfg,
        degenerate=c3_degen,
        degenerate_reason=(c3d["detail"].get("reason") if c3_degen else None))
    agg = CC.aggregate_verdict(crit, c4["c4"]["detail"], eps, cfg=agg_cfg)

    return {
        "criteria": crit,
        "aggregate": agg,
        "iso": iso,
        "c3": c3_sum,
        "c4_detail": c4["c4"]["detail"],
        "pairwise_accuracy": CC.pairwise_accuracy(zA_test, y),
        "pairwise_accuracy_raw": CC.pairwise_accuracy(xa, y),
        "pairwise_accuracy_fold_sd": float(np.std(pa_fold)),
        "rank_flip": flip_acc,             # 折内累加（见函数开头注释）
        "z_spread_ratio_a": spread_a,
        "z_spread_ratio_b": spread_b,
    }


def aggregate_summary(pairs, gate_keys=None):
    """三分类的**计数汇总**（数据集级）：逐 pair × mapping 统计标签与理由分布。

    ⚠️ 单给总数不够：必须**同时给理由分布** —— 否则「不可比 252/252」看不出究竟是
       门槛挂了，还是映射退化（`gate_na_degenerate_mapping`）。两者处置完全不同。
    """
    labels, reasons, per_pair = {}, {}, {}
    n = 0
    for pair, per in pairs.items():
        c = {}
        for _mp, r in per.items():
            ag = r.get("aggregate")
            if not ag:
                continue
            n += 1
            labels[ag["label"]] = labels.get(ag["label"], 0) + 1
            key = "gate_failed" if str(ag["reason"]).startswith("gate_failed:") else ag["reason"]
            reasons[key] = reasons.get(key, 0) + 1
            c[ag["label"]] = c.get(ag["label"], 0) + 1
        per_pair[pair] = c
    return {"n": n, "labels": labels, "reasons": reasons, "per_pair": per_pair,
            "gate_keys": list(gate_keys or CC.GATE_KEYS)}


# ───────────────────────────── 数据集级 ─────────────────────────────
def run_dataset(ds_name, table, cfg, seed, log):
    thr = cfg["criteria"]
    names = table.judge_names
    folds = SS.make_folds(table, cfg["cv"]["n_splits"], seed)
    ranges = {j: cfg["judges"][j]["score_range"] for j in names}

    log("  [%s] 折已切：每折测试 %d 张；分组键 %s"
        % (ds_name, folds[0][1].size,
           "已强制隔离" if table.groups is not None else "真空条件（无参考图变体）"))

    results = {}
    pairs = list(itertools.combinations(range(len(names)), 2))
    t0 = time.time()
    for k, (ia, ib) in enumerate(pairs):
        key = "%s | %s" % (names[ia], names[ib])
        results[key] = {}
        for mp in cfg["mappings"]:
            results[key][mp] = run_pair_mapping(
                table, ia, ib, mp, folds, thr,
                seed + 1000 * k + MF.MAPPING_NAMES.index(mp), ranges[names[ia]], ranges[names[ib]],
                agg_cfg=cfg.get("aggregate"), iso_cfg=cfg.get("iso"))
        log("    %2d/%2d  %-42s  %.1fs" % (k + 1, len(pairs), key, time.time() - t0))

    # ── 方法学对照的断言 ──
    # 直接比 `pairwise_accuracy` 的数值**不可靠**：float64 下仿射映射在近并列对上会翻符号
    # （实测 none/minmax/zscore 给出 0.6711/0.6708/0.6705 这类不相等的数）。**可证的断言**是
    # 「精确算术下的秩不变性」= 排除近并列对后翻转数应为 0，由 `rank_flip_stats` 给出。
    viol = []
    for key, per in results.items():
        for m in cfg["mappings"]:
            if m in MF.STRICT_MONOTONE and per[m]["rank_flip"]["flips_outside_nearties"] != 0:
                viol.append([key, m, per[m]["rank_flip"]["flips_outside_nearties"]])
    pa_range = max(
        (max(per[m]["pairwise_accuracy"] for m in cfg["mappings"] if m in MF.STRICT_MONOTONE)
         - min(per[m]["pairwise_accuracy"] for m in cfg["mappings"] if m in MF.STRICT_MONOTONE))
        for per in results.values())
    log("    ★ 秩不变性（族 0–5，排除近并列对后翻转数）：%s"
        % ("全部为 0 ✓" if not viol else "%d 处违例 %s" % (len(viol), viol[:3])))
    log("    ★ 成对准确率的原始跨映射极差 %.2e（浮点噪声 + 平台段所致，非数学性质）"
        % pa_range)

    return results, folds, viol


# ───────────────────────────── 合成对照 ─────────────────────────────
def run_synthetic(ds_name, table, cfg, seed, folds, log):
    sc = cfg["synthetic"]
    j = sc["positive"]["judge"]
    ia = table.judge_names.index(j)
    xa = table.X[ia].copy()
    a, b = sc["positive"]["affine"]["a"], sc["positive"]["affine"]["b"]
    xb_pos = a * xa + b
    lo, hi = cfg["judges"][j]["score_range"]
    rng_b = (a * lo + b, a * hi + b)

    rng = np.random.default_rng(seed)
    xb_neg = xa.copy()
    if table.domains is not None:
        dom = np.asarray(table.domains)
        for c in np.unique(dom):
            sel = np.where(dom == c)[0]
            xb_neg[sel] = xa[rng.permutation(sel)]
        neg_mode = "by_domain"      # 保块内边际、只破块内的图–分对应
    else:
        xb_neg = xa[rng.permutation(table.n)]
        neg_mode = "global"         # 无域标签 ⇒ 只能全局打乱（保全局边际）

    out = {}
    for tag, xb, rb in (("positive", xb_pos, rng_b), ("negative", xb_neg, cfg["judges"][j]["score_range"])):
        # ⚠️ 阳/阴对照的 A 侧用**原始分**，B 侧用合成分；两者共用同一套折与阈值
        per = {}
        for mp in cfg["mappings"]:
            per[mp] = run_pair_mapping(table, ia, ia, mp, folds, cfg["criteria"],
                                       seed + 7919, (lo, hi), rb, x_override=(xa, xb),
                                       agg_cfg=cfg.get("aggregate"), iso_cfg=cfg.get("iso"))
        out[tag] = per
        for mp, r in per.items():
            c3 = [r["criteria"][k]["verdict"] for k in ("c3_delta", "c3_beta", "c3_sigma")]
            log("    合成对照 %-8s %-10s 判据3 = %-22s 聚合 = %s"
                % (tag, mp, "/".join(c3), r["aggregate"]["label"]))
    # ⚠️ `neg_mode` **必须走 dataset 级、不能塞进 `out`**：summary 构造里对
    #    `entry["synthetic"].items()` 是按 `{tag: {mp: {...}}}` 迭代的，塞一个字符串进去
    #    会让 `per.items()` 在字符串上抛 AttributeError，整个 run 倒在最后一步。
    return out, neg_mode


# ───────────────────────────── 数据自检 ─────────────────────────────
def self_checks(tables, log):
    ok = True
    for ds, t in tables.items():
        n_exp = {"tid2013": 2687, "koniq": 2015}.get(ds)
        if n_exp is not None and t.n != n_exp:
            log("  !! [%s] 样本数 %d ≠ 期望 %d" % (ds, t.n, n_exp)); ok = False
        for j in t.judge_names:
            r = t.row(j)
            if not np.all(np.isfinite(r)):
                log("  !! [%s] %s 存在非有限分数" % (ds, j)); ok = False
        if not np.all(np.isfinite(t.y)) or not np.all(np.isfinite(t.sigma_human)):
            log("  !! [%s] y / sigma_human 存在非有限值" % (ds,)); ok = False
        log("  ✓ [%s] n=%d，9 份 key 集合同一、join 0 缺失 0 重复、y 与 σ_human 全有限" % (ds, t.n))
    return ok


#: 本模块新增/依赖的模块清单（须进 run 的 `_script/` 并记 sha256）
MODULE_FILES = ["pred_table.py", "mapping_families.py", "stratified_split.py",
                "comparability_criteria.py", "diagnose_comparability.py", "render_report.py"]


# ───────────────────────────── 产物 ─────────────────────────────
def write_csv(path, results, pairwise_rows):
    cols = ["dataset", "pair", "mapping", "criterion", "verdict", "value", "ci_lo", "ci_hi",
            "threshold_ref"]

    def _n(x, spec="%.10g"):
        return "" if x is None else spec % x

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for ds, per_ds in results.items():
            for pair, per_map in per_ds["pairs"].items():
                for mp, r in per_map.items():
                    for ck, cv in r["criteria"].items():
                        w.writerow([ds, pair, mp, ck, cv["verdict"],
                                    _n(cv.get("value")), _n(cv.get("ci_lo")), _n(cv.get("ci_hi")),
                                    cv["threshold_ref"]])
                    # ── 三分类（`criterion="aggregate"`，verdict 列 = 标签）──
                    ag = r.get("aggregate")
                    if ag:
                        w.writerow([ds, pair, mp, "aggregate", ag["label"], "", "", "",
                                    "reason=%s；gate=%s；c4=%s；drift_domains=%s；undecidable=%s"
                                    % (ag["reason"],
                                       " ".join("%s=%s" % (k, v)
                                                for k, v in sorted(ag["gate"].items())),
                                       ag["c4"], ",".join(ag["drift_domains"]) or "—",
                                       ag["undecidable"])])
                    # ── 「可互换区间」摘要（`criterion="iso"`，value 列 = P 主口径）──
                    iso = r.get("iso")
                    if iso:
                        w.writerow([ds, pair, mp, "iso", iso.get("verdict", "info"),
                                    _n(iso.get("value")), "", "",
                                    "可互换区间（两版并列）：主口径 P=mean(|f_A−f_B|<ε)；"
                                    "严格版 P_strict=%s；MOS 分箱达标 %s/%s；失真域达标 %s/%s；"
                                    "ε=%s；n_bins=%s；reason=%s"
                                    % (_n(iso.get("p_strict"), "%.6g"),
                                       _n(iso.get("n_ok_bins_mos"), "%s"),
                                       _n(iso.get("n_eval_bins_mos"), "%s"),
                                       _n(iso.get("n_ok_bins_domain"), "%s"),
                                       _n(iso.get("n_eval_bins_domain"), "%s"),
                                       _n(iso.get("eps"), "%.6g"),
                                       _n(iso.get("n_bins_effective"), "%s"),
                                       iso.get("reason") or "—")])
        for row in pairwise_rows:
            w.writerow(row)
    return path


def dataset_notes(ds, cfg):
    """报告里每个数据集必须自带的口径注（三条：切分性质、污染、判据 4 是否可测）。"""
    d = cfg["datasets"][ds]
    n_splits = cfg["cv"]["n_splits"]
    if d["split"]["kind"] == "group_stratified":
        split_note = ("切分 = `StratifiedGroupKFold`（groups = 参考图编号，y = MOS 五分位，K = %d）。"
                      "「**同一参考图的不同失真版本不得同时出现在校准集与测试集**」由分组机制"
                      "**在结构上**保证，并由 `stratified_split.py` 的断言复核。" % n_splits)
    else:
        split_note = ("切分 = `StratifiedKFold`（y = MOS 五分位，K = %d）。⚠️ 「同一参考图不得跨集」"
                      "在本数据集上是**真空条件**（无参考图变体）——**不是「通过了这条约束」**，"
                      "而是**没有可违反的对象**，引用时不得读成前者。" % n_splits)

    dirty = [(j, cfg["judges"][j].get("koniq_seen_frac"))
             for j in cfg["judges"] if (cfg["judges"][j].get("koniq_seen_frac") or 0) > 0]
    if ds == "koniq" and dirty:
        pol = ("**污染对照列**：本数据集的 %s 在 KonIQ-10k 上训练过、且见过我们的官方 test 子集 "
               "—— %s。⇒ 这一列**只作对照**，主结论不在它上面出。"
               % ("、".join("`%s`" % j for j, _ in dirty),
                  "、".join("`%s` 见过 %.1f%%" % (j, 100.0 * f) for j, f in dirty)))
    else:
        clean = [j for j in cfg["judges"] if cfg["judges"][j].get("tid2013_clean") is True]
        unclear = [j for j in cfg["judges"] if cfg["judges"][j].get("tid2013_clean") is None]
        # ⚠️ 这里**不能**写「%d 个评委的 tid2013_clean 全为真」了事：9 个评委里
        #    `clip-iqa` 是 `null`（其 IQA 层干净，但 CLIP 骨干的 WIT-400M 语料不可核
        #    ⇒ 骨干层不可判）。写「8 个全为真」会让人以为本数据集只有 8 个评委，
        #    把「一项不可判」静默成「一项不存在」。
        tail = ""
        if unclear:
            tail = ("；`%s` 为 `null` —— 其 IQA 层干净、但骨干层不可核"
                    " ⇒ **不可判，不是干净**" % "`、`".join(unclear))
        pol = ("训练集含 TID2013 的评委：**无**（逐评委审计：9 个评委中 %d 个 "
               "`tid2013_clean` 为真%s）⇒ **主结论在本数据集上出**，"
               "符合「主结论只在干净格子上出」。" % (len(clean), tail))

    dom_na = None
    if not d.get("domain"):
        dom_na = ("标签 CSV 只有 `image_name / c1..c5 / c_total / MOS / SD / set`，**既无失真类型"
                  "也无内容标注**；TID2013 的文件名 `I02_01_1.bmp` 才编码参考图 / 失真类型 / 强度级。")
    return split_note, pol, dom_na


def stage_script(out_dir, cfg_path, log, note=None):
    """run 目录里留**当次真正执行的那份脚本**，并按原名复制。

    ⚠️ 被 import 的模块（`mapping_families` 等、`l4_validate`）也是「当次执行件」，一并入册。
    ⚠️ 只把 sha256 打进日志不够（2026-10-13 踩过：事后能证明「不一致」，证明不了「差在哪」）。
    `note`：run 级附注，原样写进 `_script/README.md` —— 当「入册的脚本」与「产出数字的那一版」
    不完全对应时（例如某步事后单独重跑过），**这一句是唯一能说清差别的地方**，不要省略。
    """
    import hashlib
    import shutil

    here = os.path.dirname(os.path.abspath(__file__))
    names = list(MODULE_FILES) + ["l4_validate.py"]
    dst = os.path.join(out_dir, "_script")
    os.makedirs(dst, exist_ok=True)
    rows = []
    for nm in names:
        src = os.path.join(here, nm)
        if not os.path.exists(src):
            rows.append((nm, "缺失", "—", "**仓库侧不存在该文件**"))
            continue
        shutil.copyfile(src, os.path.join(dst, nm))
        h = hashlib.sha256(open(src, "rb").read()).hexdigest()
        rows.append((nm, h[:16] + "…", h[:16] + "…", "与仓库当前副本**逐字节相同**（运行时刻）"))
    cfg_copy = os.path.basename(cfg_path)
    shutil.copyfile(cfg_path, os.path.join(dst, cfg_copy))
    h = hashlib.sha256(open(cfg_path, "rb").read()).hexdigest()
    rows.append((cfg_copy, h[:16] + "…", h[:16] + "…", "配置（当次）"))

    with open(os.path.join(dst, "README.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# 当次执行的脚本\n\n")
        f.write("本目录下按**原名**存放产出这批数字时**真正执行**的那一版文件。\n")
        f.write("仓库里的脚本还会被继续改，**这里这一份才是产出摘要数字的那一版**。\n\n")
        f.write("| 文件 | 当次 sha256（前 16 位） | 仓库副本 | 与之关系 |\n|---|---|---|---|\n")
        for nm, h1, h2, rel in rows:
            f.write("| `%s` | `%s` | `%s` | %s |\n" % (nm, h1, h2, rel))
        f.write("\n**标准跑法**：本模块是**一次性诊断脚本**（不是某个评委的标准跑法）⇒ "
                "`judges/<评委>/run_local.sh` 那一档**不适用**，故上表第三列只作「运行时刻"
                "与仓库副本是否相同」的记账。⚠️ 该列是**运行时刻**的快照：此后若改过仓库副本，"
                "两者会不再相同，比对时须以本目录的字节为准。\n")
        if note:
            f.write("\n## 本 run 的附注（覆盖上表的适用范围）\n\n%s\n" % note)
    log("  _script/ 已入册 %d 个文件（含配置与 `l4_validate.py`）" % len(rows))


def main():
    # 本模块的日志含 `✓` 与中文；Windows 控制台默认是 GBK（cp936），`print` 会直接抛
    # `UnicodeEncodeError` 把整个 run 打断（实测踩到过：挂在第一处 `✓`，产物只剩 run.log 头两行）。
    # 正确修法是**声明流编码**，不是把记号换成 ASCII —— 换掉只是把缺陷藏起来。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", default=None, help="产物目录；默认写到 runs/<配置名去后缀>/")
    ap.add_argument("--datasets", default=None, help="逗号分隔，默认全部")
    ap.add_argument("--no-synthetic", action="store_true")
    args = ap.parse_args()

    cfg_path = PT.resolve(args.cfg)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg["seed"])

    if args.out:
        out_dir = PT.resolve(args.out)
    else:
        stem = os.path.basename(cfg_path).replace(".yaml", "").replace(".yml", "")
        out_dir = PT.resolve(os.path.join("30-experiments", "runs", stem))
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8", newline="\n")

    def log(msg):
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
        logf.write(line + "\n")           # 文件一律 utf-8：产物优先于控制台
        logf.flush()
        try:
            print(line, flush=True)
        except UnicodeEncodeError:        # 终端编码再意外，也不得中断 run
            print(line.encode("ascii", "replace").decode("ascii"), flush=True)

    log("P1 可比性诊断 —— 配置 %s" % cfg_path)
    log("种子 %d；折数 %d；映射 %s" % (seed, cfg["cv"]["n_splits"], cfg["mappings"]))

    log("① 分数采集")
    tables = PT.build_all(cfg, verbose=False)
    if not self_checks(tables, log):
        raise SystemExit("数据自检失败，中止")

    want = args.datasets.split(",") if args.datasets else list(tables)
    all_results = {}
    for ds in want:
        table = tables[ds]
        log("② – ④ %s（%s）" % (ds, cfg["datasets"][ds]["role"]))
        res, folds, viol = run_dataset(ds, table, cfg, seed, log)
        entry = {"pairs": res, "strict_monotone_violations": viol, "n": table.n,
                 "aggregate_summary": aggregate_summary(
                     res, (cfg.get("aggregate") or {}).get("gate_keys"))}
        if not args.no_synthetic:
            log("  ～ 合成对照（阶段 C）")
            entry["synthetic"], entry["synthetic_negative_mode"] = run_synthetic(
                ds, table, cfg, seed, folds, log)
            log("    阴性对照模式 = %s（%s）" % (
                entry["synthetic_negative_mode"],
                "按失真类型分块打乱" if entry["synthetic_negative_mode"] == "by_domain"
                else "**全局打乱**：本数据集无域标签，只能保全局边际"))
        all_results[ds] = entry

    # ── 方法学对照长表 ──
    pairwise_rows = []
    for ds, entry in all_results.items():
        for pair, per_map in entry["pairs"].items():
            for mp, r in per_map.items():
                pairwise_rows.append([ds, pair, mp, "pairwise_accuracy", "",
                                      "%.10g" % r["pairwise_accuracy"], "", "",
                                      "对任何严格单调变换不变（纯秩指标）"])

    write_csv(os.path.join(out_dir, "pairs.csv"), all_results, pairwise_rows)

    summary = {
        "config": os.path.relpath(cfg_path, PT.REPO_ROOT).replace(os.sep, "/"),
        "seed": seed,
        # ⚠️ schema_version = 2（2026-10-16）：新增 `aggregate` / `iso` 两个**兄弟键**与
        #    `aggregate_summary` / `iso_eps` / `synthetic_negative_mode` 三处 dataset 级字段。
        #    **更早的 run 的 summary 无这些键，两者不可混比**，渲染层按缺失降级。
        "schema_version": 2,
        "mappings": cfg["mappings"],
        "criteria_thresholds": cfg["criteria"],
        "aggregate_cfg": cfg.get("aggregate"),
        "iso_cfg": cfg.get("iso"),
        "n_pairs": 36,
        "self_check": "pass",
        "datasets": {},
    }
    for ds, entry in all_results.items():
        split_note, pol, dom_na = dataset_notes(ds, cfg)
        # ε 是**每个数据集一个常数**（只取决于 median(sigma_human)），故取任一条即可。
        iso_eps = next((r["c3"].get("eps") for per_map in entry["pairs"].values()
                        for r in per_map.values() if r["c3"].get("eps")), None)
        summary["datasets"][ds] = {
            "role": cfg["datasets"][ds]["role"],
            "n": entry["n"],
            "split_note": split_note,
            "pollution_note": pol,
            "domain_na_reason": dom_na,
            "strict_monotone_violations": entry["strict_monotone_violations"],
            "iso_eps": iso_eps,
            "aggregate_summary": entry.get("aggregate_summary"),
            "synthetic_negative_mode": entry.get("synthetic_negative_mode"),
            "pairs": {p: {mp: {"criteria": {k: {"verdict": v["verdict"], "value": v.get("value"),
                                                "ci_lo": v.get("ci_lo"), "ci_hi": v.get("ci_hi"),
                                                "threshold_ref": v["threshold_ref"]}
                                         for k, v in r["criteria"].items()},
                           # ⚠️ `aggregate` / `iso` 是 `criteria` 的**兄弟键**，绝不放进
                           #    `criteria` 里：渲染层 `build_report` 的 `crit_keys` 取自
                           #    `criteria.keys()`，混进去会引入意料外的键、连环破坏表头与列数。
                           "aggregate": r.get("aggregate"),
                           "iso": r.get("iso"),
                           "pairwise_accuracy": r["pairwise_accuracy"],
                           "pairwise_accuracy_fold_sd": r["pairwise_accuracy_fold_sd"],
                           "rank_flip": r["rank_flip"],
                           "c3": r["c3"],
                           "c4_detail": r["c4_detail"],
                           "z_spread_ratio_a": r["z_spread_ratio_a"],
                           "z_spread_ratio_b": r["z_spread_ratio_b"]}
                      for mp, r in per_map.items()}
                      for p, per_map in entry["pairs"].items()},
            "synthetic": {tag: {mp: {"criteria": {k: v["verdict"] for k, v in r["criteria"].items()},
                                     "c3": r["c3"],
                                     "aggregate": (r.get("aggregate") or {}).get("label"),
                                     "iso_P": (r.get("iso") or {}).get("value")}
                                for mp, r in per.items()}
                          for tag, per in entry.get("synthetic", {}).items()},
        }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=False)

    # ── 报告渲染（从落盘的 summary.json 重新读入，保证报告是派生物）──
    import render_report as RR
    sum_path = os.path.join(out_dir, "summary.json")
    rep = RR.render(sum_path, os.path.join(out_dir, "report.md"), strict=MF.STRICT_MONOTONE)
    log("报告草稿：%s" % rep)

    # ── 当次执行件入册 ──
    stage_script(out_dir, cfg_path, log)

    log("产物：pairs.csv / summary.json / report.md / run.log / _script/ → %s" % out_dir)
    logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
