# -*- coding: utf-8 -*-
"""第一层基线（L4）的 9 评委 × 5 数据集矩阵：SRCC / PLCC(raw) / PLCC(log) / KROCC。

2026-11-05 建。动机：期刊稿正文从头到尾没印过一个 PLCC 或 SRCC
（`.md` 与 `main.tex` 全文 grep `PLCC|SRCC|Spearman|Pearson` 均 0 命中），读者无法判断
九个评委本身靠不靠谱；而主结论「252 条判定无一条可比」缺基线时，最省事的反驳就是
「你的评委不行」。本脚本产出那张表所需的 45 格。

★ **为什么从 `pred_*.jsonl` 重算，而不是抄 `run.log`**（三条理由，缺一条都不成立）：
  1. **46 个 `-l4` 目录里没有一个 `summary.json`**（2026-11-05 逐目录实测）。`l4_validate.py`
     只打印、不落盘 ⇒ 指标只以文本块散在日志里。
  2. **日志块只覆盖 32/45 格**：三个卡上栈在 KonIQ / TID2013 / AGIQA-3K 上的跑没有内联块，
     那几个值只存在于各评委 README 的散文里。
  3. **CSIQ 的日志值符号是反的**：L4 校验用的是数据集原列（CSIQ 的 `MOS` 实为 DMOS，
     越高越差），所以日志里 CSIQ 的 `SRCC / KROCC / PLCC(raw)` 九个全负，而 `PLCC(log)`
     因 4 参数 logistic 自己把方向翻回来而是正的。转录日志就会把 CSIQ 印成负相关。
  ⇒ 本脚本**一个口径算全部 45 格**：y 取配置的 `(raw − offset)/scale`，CSIQ 因此被翻正，
    全表「越高越好」。日志块只用来**对账**（见下），不用来取值。
  实现上复用 `pred_table.build_table()`（不重写 I/O：它的 key 集合自检、逐图对齐、
  y 归一化都已验过），本脚本只负责取对齐后的 `X`/`y` 并算四个指标。

★ **口径四件事的落点**（纪律第 9 条；本表是三段栈的**披露后并列**，不是同一口径）：
  ① split       —— 取自各 run 目录的 `test_split.txt` / 官方 test 集，本脚本不重排；
  ② 输入口径     —— 如实记录在 pred 的逐行字段里（img_size / crops / patch_size …），本脚本不碰；
  ③ 指标算法     —— PLCC(raw) = 直接 Pearson；PLCC(log) = 4 参数 logistic 拟合后 Pearson
                    （`l4_validate.plcc_logistic`，IQA 惯例；**拟合在原始标签列上做**，
                    与历史日志逐位一致，理由见 compute_cells() 里的实测注）；SRCC / KROCC 为秩相关；
  ④ 运行栈       —— 三段：`local`（6 个，Py 3.11.15 / torch 2.6.0 / TF 5.19.0 / GTX 1650）、
                    `card-qalign`（2 个，3.10.22 / 2.0.1 / 4.36.1 / RTX 3090）、
                    `card-iqat1`（1 个，3.11.17 / 2.6.0 / 5.19.0 / 3090）。
                    唯一实测的栈差：同一评委仅换栈，SRCC 位移 ≈5e-3（MUSIQ，本机 ↔ 3090）。

★ **对账不是可有可无的装饰**：本脚本会去每个 run 目录里找 `l4_validate.py` 打的那块文本，
  与重算值逐格比。CSIQ 那五列按**绝对值**比（原因见上），并在报告里标 `sign=`。
  差 > 5e-4 的格会被点名，必须逐条解释，不得含糊（纪律第 6 条）。

用法：
    python -I 30-experiments/metrics/l4_baseline_table.py            # 默认五数据集、写 runs/20261105-l4baseline-9x5
    python -I 30-experiments/metrics/l4_baseline_table.py --out DIR  # 指定产物目录
    python -I 30-experiments/metrics/l4_baseline_table.py --datasets koniq,tid2013

产物（`--out` 目录下）：
    summary.json  45 格 ＋ 每格 n / 耗时 / 日志块 / 对账差 ＋ 数据集与评委元数据
    report.md     人读版：两张矩阵 ＋ 栈与污染标注 ＋ 对账表
    run.log       逐格进度与耗时（stdout 同步）

⚠️ 实测：与本脚本等价的循环若把结果只往 stdout 打，会在个别格之后**无输出阻塞**
  （单格单独跑都是 0.1–0.2 s，已验 7 格与既有记录逐位相同）。故本脚本**先写文件后打印**，
  且每格都落盘——中途被杀也不会丢掉已算完的格。
"""

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pred_table as PT                                   # noqa: E402
from l4_validate import plcc_logistic                      # noqa: E402
from scipy.stats import spearmanr, kendalltau, pearsonr    # noqa: E402

SCHEMA_VERSION = "l4-baseline-1"

# 数据集出场顺序＝期刊稿 §III「Five datasets in three domain classes」的顺序：
# 合成失真（TID2013、CSIQ）→ 真实失真（KonIQ-10k、LIVE Challenge）→ AIGC（AGIQA-3K）。
# 配置文件里 `datasets:` 的书写顺序**不是**这个顺序 ⇒ 必须显式排。
DS_ORDER = ("tid2013", "csiq", "koniq", "livechallenge", "agiqa3k")

DS_LABEL = {
    "tid2013":       "TID2013",
    "csiq":          "CSIQ",
    "koniq":         "KonIQ-10k",
    "livechallenge": "LIVE Challenge",
    "agiqa3k":       "AGIQA-3K",
}

DS_CLASS = {
    "tid2013":       "synthetic",
    "csiq":          "synthetic",
    "koniq":         "authentic",
    "livechallenge": "authentic",
    "agiqa3k":       "AIGC",
}

# 栈标记：只需区分「哪一段栈」，版本号不进表（进报告）。
STACK_MARK = {
    "local":       "†",
    "card-qalign": "‡",
    "card-iqat1":  "§",
}

STACK_DESC = {
    "local":       "本机栈：Python 3.11.15 / torch 2.6.0+cu124 / transformers 5.19.0 / GTX 1650",
    "card-qalign": "卡上 qalign 系：Python 3.10.22 / torch 2.0.1+cu117 / transformers 4.36.1 / RTX 3090",
    "card-iqat1":  "卡上 iqat1：Python 3.11.17 / torch 2.6.0+cu124 / transformers 5.19.0 / RTX 3090",
}

JUDGE_LABEL = {
    "q-align":         "Q-Align",
    "q-realign-mini":  "Q-ReAlign Mini 0.8B",
    "deqa-score":      "DeQA-Score",
    "clip-iqa":        "CLIP-IQA",
    "musiq":           "MUSIQ",
    "maniqa":          "MANIQA",
    "hyperiqa":        "HyperIQA",
    "topiq-nr":        "TOPIQ-NR",
    "iqa-t1":          "IQA-T1",
}

# `l4_validate.py` 打的文本块（ASCII 标签，跨编码稳定）。标签后可能跟行内注释
# （如 `PLCC(raw) : 0.6320   <- 直接 Pearson`），故数值后一律吃到行尾 `[^\n]*\n`。
BLOCK_RE = re.compile(
    r"SRCC\s*:[^\S\n]*(-?\d+\.\d+)[^\n]*\n"
    r"\s*KROCC\s*:[^\S\n]*(-?\d+\.\d+)[^\n]*\n"
    r"\s*PLCC\(raw\)\s*:[^\S\n]*(-?\d+\.\d+)[^\n]*\n"
    r"\s*PLCC\(log\)\s*:[^\S\n]*(-?\d+\.\d+)")

# 找日志块的优先次序：先主日志，再校验日志，最后其余 .log / .txt（按名字）。
_LOG_PRIORITY = ("run.log", "run_l4.log", "l4_validate.log", "l4_livechallenge.txt")

TOL = 5e-4          # 对账容差：超过就算不一致、必须解释


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_log_block(run_dir):
    """在一个 run 目录里找 `l4_validate.py` 的指标块。找不到返回 None。

    同一目录可能同时有 smoke 日志（6 张）与全量日志 ⇒ 只认**优先级最高且有块**的那个文件，
    并取该文件里的**最后**一个块（多次调用时最后那次才是最终读数）。
    """
    if not os.path.isdir(run_dir):
        return None
    files = [f for f in os.listdir(run_dir) if f.lower().endswith((".log", ".txt"))]

    def rank(fn):
        low = fn.lower()
        for i, pref in enumerate(_LOG_PRIORITY):
            if low == pref:
                return i
        return len(_LOG_PRIORITY)

    for fn in sorted(files, key=lambda f: (rank(f), f)):
        path = os.path.join(run_dir, fn)
        try:
            text = io.open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        hits = BLOCK_RE.findall(text)
        if hits:
            srcc, krocc, plcc_raw, plcc_log = hits[-1]
            return {
                "file": fn,
                "srcc": float(srcc),
                "krocc": float(krocc),
                "plcc_raw": float(plcc_raw),
                "plcc_log": float(plcc_log),
                "note": "同一文件内 %d 个块，取最后一个" % len(hits),
            }
    return None


def load_config(cfg_path):
    import yaml
    with io.open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_cells(cfg, datasets, out_dir, log):
    """返回 (cells, ds_meta, judge_meta)。cells[ds][judge] = {...}。"""
    cells, ds_meta = {}, {}

    for ds in datasets:
        dc = cfg["datasets"][ds]
        t_ds = time.time()

        # ★ 复用 `pred_table.build_table`：它已做「9 份 pred 的 key 集合逐对相同」「标签侧
        #   文件名与缓存逐一对齐」「y = (raw − offset)/scale」三件事，且 CSIQ 的
        #   `scale: -1.0` 就在它第 181 行翻正。本脚本**不重写 I/O**。
        #   注：`sigma_human.kind = not_available`（AGIQA-3K / CSIQ）只填 NaN、不抛，
        #   故可安全地在五个数据集上通用——本脚本不碰 σ_human。
        t = PT.build_table(ds, dc, cfg["judges"], verbose=False)
        names, y = t.names, t.y

        ycfg = dc["y"]
        src_col = ycfg["source_column"]
        ds_meta[ds] = {
            "n": t.n,
            "label": DS_LABEL.get(ds, ds),
            "domain_class": DS_CLASS.get(ds, "?"),
            "y_source_column": src_col,
            "y_label_file": dc.get("aux_labels_csv") or dc.get("labels_csv"),
            "y_offset": float(ycfg.get("offset", 0.0)),
            "y_scale": float(ycfg["scale"]),
            "y_min": float(y.min()),
            "y_max": float(y.max()),
            "y_note": dc.get("y", {}).get("note"),
            "split": dc.get("split"),
            "sign_flipped": float(ycfg["scale"]) < 0.0,
        }
        log("## %s  n=%d  y=[%.5f, %.5f]  scale=%g%s"
            % (ds, t.n, y.min(), y.max(), float(ycfg["scale"]),
               "（DMOS 已翻正）" if float(ycfg["scale"]) < 0.0 else ""))

        cells[ds] = {}
        for judge in sorted(dc["preds"]):
            pred_rel = dc["preds"][judge]
            pred_abs = PT.resolve(pred_rel)
            t0 = time.time()
            pv = np.asarray(t.X[t.judge_names.index(judge)], dtype=np.float64)
            if not np.all(np.isfinite(pv)):
                raise SystemExit("评委 %s 在 %s 上有非有限分（%d 个）⇒ 中止，不补数"
                                 % (judge, ds, int((~np.isfinite(pv)).sum())))
            if not np.all(np.isfinite(y)):
                raise SystemExit("数据集 %s 的 y 含非有限值 ⇒ 中止" % ds)

            srcc = float(spearmanr(pv, y).statistic)
            krocc = float(kendalltau(pv, y).statistic)
            plcc_raw = float(pearsonr(pv, y).statistic)
            # ★ PLCC(log) 的拟合**在原始标签列上做**（`y·scale + offset`，即 CSV 里那一列），
            #   与 `l4_validate.py` 逐位一致。这不是等价变换的随意选择，而是一处实测的
            #   舍入平局：Q-Align／TID2013 的真值 ≈ 0.84585000，**正好压在 4 位小数的
            #   平局点 0.84585 上**——归一化 y 拟合得 0.84585006（印 0.8459），原始 y 拟合得
            #   0.84585000（印 0.8458）。差 6e-8，但 4 位小数会翻。取原始 y 是因为
            #   `judges/q-align/README.md` 与既有记录记的都是 **0.8458**，
            #   表不能与自己的留痕差最后一位。45 格里只有这一格会翻（2026-11-05 实测）。
            y_raw = y * float(ycfg["scale"]) + float(ycfg.get("offset", 0.0))
            plcc_log, _popt = plcc_logistic(pv, y_raw)

            run_dir = os.path.dirname(pred_abs)
            blk = find_log_block(run_dir)
            recon = None
            if blk is not None:
                # CSIQ 的日志块用的是未翻正的 DMOS ⇒ 按绝对值比，并记符号关系。
                flip = ds_meta[ds]["sign_flipped"]
                recon = {}
                for k in ("srcc", "plcc_raw", "plcc_log", "krocc"):
                    mine = locals()[k]
                    d = abs(abs(blk[k]) - abs(mine)) if flip else abs(blk[k] - mine)
                    recon[k] = {"recomputed": mine, "log": blk[k], "absdiff": d,
                                "sign_relation": ("flipped" if flip and blk[k] * mine < 0
                                                  else "same")}

            cells[ds][judge] = {
                "srcc": srcc,
                "krocc": krocc,
                "plcc_raw": plcc_raw,
                "plcc_log": None if plcc_log is None else float(plcc_log),
                "n": len(names),
                "pred": pred_rel,
                "run_dir": os.path.relpath(run_dir, PT.REPO_ROOT).replace("\\", "/"),
                "seconds": round(time.time() - t0, 3),
                "log_block": blk,
                "reconcile": recon,
            }
            log("   %-16s SRCC=%.4f  PLCCraw=%.4f  PLCClog=%s  KROCC=%.4f  [%s%.2fs]"
                % (judge, srcc, plcc_raw,
                   "None" if plcc_log is None else "%.4f" % plcc_log,
                   krocc,
                   ("" if recon is None else "对账:%s " % (
                       "ok" if max(v["absdiff"] for v in recon.values()) <= TOL else "差")),
                   time.time() - t0))

        log("   -- %s 用时 %.1fs" % (ds, time.time() - t_ds))

        # ── 标签列对照（只在有 aux_labels_csv 的数据集上跑；五个里只有 KonIQ） ──
        # 背景：KonIQ 的官方包里有两个带 MOS 的文件——`koniq10k_distributions_sets.csv`
        # （1–100）与 `koniq10k_scores_and_distributions.csv`（1–5）。本课题的 P1 全链
        # （含 σ_human = SD/√n、252 条判定）用的是**后者**；而历史 L4 日志的 KonIQ 那一列
        # 用的是**前者**。两个 MOS 列不完全同源（2015 张 test 上二者 SRCC 0.9917，不是 1），
        # 所以同一评委的 SRCC 会差到 2e-3。此块把差值算出来，供报告点名，不靠转述。
        if dc.get("aux_labels_csv"):
            try:
                _, alt_rows = PT._load_labels(dc)
                alt = {os.path.basename(r["image_name"]): float(r[src_col]) for r in alt_rows}
                if all(n in alt for n in names):
                    alt_y = np.asarray([alt[n] for n in names], dtype=np.float64)
                    ctl = {"file": dc["labels_csv"], "column": src_col,
                           "n_covered": len(names),
                           "srcc_between_label_columns": float(
                               spearmanr(t.y, alt_y).statistic),
                           "per_judge": {}}
                    for judge in sorted(dc["preds"]):
                        pv = np.asarray(t.X[t.judge_names.index(judge)], dtype=np.float64)
                        s_alt = float(spearmanr(pv, alt_y).statistic)
                        ctl["per_judge"][judge] = {
                            "srcc_alt_column": s_alt,
                            "delta_vs_primary": s_alt - cells[ds][judge]["srcc"],
                        }
                    ctl["max_abs_delta_srcc"] = max(
                        abs(v["delta_vs_primary"]) for v in ctl["per_judge"].values())
                    ds_meta[ds]["label_column_control"] = ctl
                    log("   标签列对照（%s）：两列之间 SRCC=%.4f，逐评委 |ΔSRCC| 最大 %.4f"
                        % (os.path.basename(dc["labels_csv"]),
                           ctl["srcc_between_label_columns"], ctl["max_abs_delta_srcc"]))
                else:
                    ds_meta[ds]["label_column_control"] = {
                        "status": "not_run",
                        "reason": "labels_csv 未覆盖全部 %d 张" % len(names)}
            except Exception as e:                                # noqa: BLE001
                ds_meta[ds]["label_column_control"] = {
                    "status": "failed", "error": "%s: %s" % (type(e).__name__, e)}
                log("   标签列对照失败：%s: %s（如实记，不补数）" % (type(e).__name__, e))

    judge_meta = {}
    for judge, jc in cfg.get("judges", {}).items():
        stack = jc.get("stack")
        judge_meta[judge] = {
            "label": JUDGE_LABEL.get(judge, judge),
            "stack": stack,
            "stack_mark": STACK_MARK.get(stack, "?"),
            "group": jc.get("group"),
            "backbone": jc.get("backbone"),
            "koniq_seen_frac": jc.get("koniq_seen_frac"),
            "koniq_clean": jc.get("koniq_clean"),
            "koniq_confidence": jc.get("koniq_confidence"),
            "tid2013_clean": jc.get("tid2013_clean"),
        }
    return cells, ds_meta, judge_meta


def build_report(cells, ds_meta, judge_meta, datasets, cfg_path, cfg_sha):
    judges = sorted(cells[datasets[0]].keys())
    L = []
    L.append("# L4 基线：9 评委 × 5 数据集（SRCC / PLCC）")
    L.append("")
    L.append("> 2026-11-05 产物。**从 `pred_*.jsonl` 重算**，不从 `run.log` 转录。")
    L.append("> 配置：`%s`（sha256 `%s…`）。" % (cfg_path, cfg_sha[:16]))
    L.append("")
    L.append("**口径**：PLCC(raw) = 直接 Pearson；PLCC(log) = 4 参数 logistic 拟合后 Pearson "
             "（IQA 惯例，拟合在**原始标签列**上做，与历史日志逐位一致）；SRCC / KROCC 为秩相关。"
             "y 一律取配置的 `(raw − offset)/scale`，故 **CSIQ 已按 `scale = −1.0` 从 DMOS 翻正**，"
             "全表「越高越好」。")
    L.append("")
    L.append("**栈**：" + "；".join("%s %s" % (m, STACK_DESC[s])
                                   for s, m in STACK_MARK.items()))
    L.append("")
    L.append("**污染**：KonIQ-10k 一列里 MANIQA 见过我们 test 的 1,607/2,015（79.8%）、"
             "MUSIQ ≈80%（种子未公布）⇒ 该列只作对照，不作可比性证据。")
    L.append("")

    for metric, title in (("srcc", "SRCC"), ("plcc_log", "PLCC（4 参数 logistic 后）")):
        L.append("## %s" % title)
        L.append("")
        L.append("| 评委 | 栈 | " + " | ".join("%s（%d）" % (ds_meta[ds]["label"], ds_meta[ds]["n"])
                                               for ds in datasets) + " |")
        L.append("|---|---|" + "---:|" * len(datasets))
        for j in judges:
            jm = judge_meta.get(j, {})
            mark = jm.get("stack_mark", "?")
            row = []
            for ds in datasets:
                v = cells[ds][j][metric]
                row.append("—" if v is None else "%.4f" % v)
            L.append("| %s %s | %s | %s |"
                     % (jm.get("label", j), mark, mark, " | ".join(row)))
        L.append("")
        L.append("† 本机栈 ／ ‡ 卡上 qalign 系 ／ § 卡上 iqat1。KonIQ-10k 列中 MANIQA、MUSIQ 两行非留出，"
                 "阅读时按 Table 1 的污染列处理。")
        L.append("")

    # ── 对账表 ──
    L.append("## 与 run 目录内联日志块的对账")
    L.append("")
    L.append("容差 %g（CSIQ 按绝对值比，因为日志用的是未翻正的 DMOS）。" % TOL)
    L.append("")
    L.append("| 数据集 | 评委 | SRCC 重算 / 日志 | PLCC(log) 重算 / 日志 | 最大 |Δ| | 判定 |")
    L.append("|---|---|---|---|---:|---|")
    n_cells = n_block = n_bad = 0
    worst = (0.0, None)
    for ds in datasets:
        for j in judges:
            c = cells[ds][j]
            n_cells += 1
            rec = c.get("reconcile")
            if not rec:
                L.append("| %s | %s | — | — | — | 无内联块 |" % (ds_meta[ds]["label"], j))
                continue
            n_block += 1
            d = max(v["absdiff"] for v in rec.values())
            verdict = "ok" if d <= TOL else "**差**"
            if d > TOL:
                n_bad += 1
                # 若换用 labels_csv 那一列能对上日志 ⇒ 差异是**标签列口径**，不是算错。
                ctl = ds_meta[ds].get("label_column_control") or {}
                alt = (ctl.get("per_judge") or {}).get(j)
                if alt and abs(abs(alt["srcc_alt_column"]) - abs(rec["srcc"]["log"])) <= TOL:
                    verdict = "差（标签列口径，见下节）"
            if d > worst[0]:
                worst = (d, "%s/%s" % (ds, j))
            L.append("| %s | %s | %.4f / %.4f | %.4f / %.4f | %.1e | %s |"
                     % (ds_meta[ds]["label"], j,
                        rec["srcc"]["recomputed"], rec["srcc"]["log"],
                        rec["plcc_log"]["recomputed"] if rec["plcc_log"]["recomputed"] is not None else float("nan"),
                        rec["plcc_log"]["log"], d, verdict))
    L.append("")
    L.append("**汇总**：%d 格中 %d 格有内联日志块可比；超容差 **%d** 格；最大 |Δ| **%.1e**（%s）。"
             % (n_cells, n_block, n_bad, worst[0], worst[1] or "—"))
    L.append("")
    L.append("没有内联块的格（三个卡上栈在 KonIQ / TID2013 / AGIQA-3K 上的跑）**只能用重算值**——"
             "它们的旧读数只存在于各评委 README 的散文里，"
             "没有可机器核对的当次产物。")
    L.append("")

    # ── 标签列对照 ──
    for ds in datasets:
        ctl = ds_meta[ds].get("label_column_control")
        if not ctl:
            continue
        if ctl.get("status"):
            L.append("## %s 的标签列对照（未跑成）" % ds_meta[ds]["label"])
            L.append("")
            L.append("`%s`：%s" % (ctl["status"], ctl.get("reason") or ctl.get("error")))
            L.append("")
            continue
        L.append("## %s 的标签列对照：两个官方文件里的 `%s` 列不是同一列"
                 % (ds_meta[ds]["label"], ctl["column"]))
        L.append("")
        L.append("**发现（本批首次量化）**：`%s` 与本脚本取用的 `%s` 都带 `%s` 列，"
                 "但两列在同样 %d 张图上**不肯重合**——两列之间 SRCC 仅 **%.4f**，"
                 "并非正比关系。本课题 P1 全链（σ_human、252 条判定……）用的是后者，"
                 "而历史 L4 日志的 KonIQ 那一列用的是前者 ⇒ 各评委 README 里 KonIQ 的 SRCC/PLCC "
                 "与本节重算差到 **%.1e**，是**标签列口径差**，不是算错。"
                 % (ctl["file"], ds_meta[ds]["y_label_file"], ctl["column"], ctl["n_covered"],
                    ctl["srcc_between_label_columns"], ctl["max_abs_delta_srcc"]))
        L.append("")
        L.append("| 评委 | SRCC（本表基准列） | SRCC（另一列） | Δ |")
        L.append("|---|---:|---:|---:|")
        for j in judges:
            v = ctl["per_judge"][j]
            L.append("| %s | %.4f | %.4f | %+.4f |"
                     % (j, cells[ds][j]["srcc"], v["srcc_alt_column"], v["delta_vs_primary"]))
        L.append("")
        L.append("**处置**：正文表用**本表基准列**（与论文其余部分同一人类锚点）；"
                 "各评委 README 里 KonIQ 那一列**不改、不回填**，差异登记在此与 `_script/README.md`，"
                 "引用那些数值时须声明它出自另一个标签文件。")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="L4 基线：9 评委 × 5 数据集的 SRCC/PLCC 矩阵")
    ap.add_argument("--cfg", default="30-experiments/configs/20261016-p1diagnose-v2.yaml",
                    help="五个数据集的配置（含全部 45 个 pred 路径与 y 的 offset/scale）")
    ap.add_argument("--out", default="30-experiments/runs/20261105-l4baseline-9x5",
                    help="产物目录")
    ap.add_argument("--datasets", default="",
                    help="逗号分隔；缺省＝按 DS_ORDER 全跑")
    args = ap.parse_args()

    out_dir = PT.resolve(args.out)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")
    log_fh = io.open(log_path, "w", encoding="utf-8", newline="\n")

    def log(msg):
        print(msg, flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    cfg_path = PT.resolve(args.cfg)
    cfg_sha = sha256(cfg_path)
    cfg = load_config(cfg_path)

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        unknown = [d for d in datasets if d not in cfg["datasets"]]
        if unknown:
            raise SystemExit("配置里没有这些数据集：%s" % unknown)
    else:
        missing = [d for d in DS_ORDER if d not in cfg["datasets"]]
        if missing:
            raise SystemExit("配置缺数据集：%s（本脚本按五数据集设计）" % missing)
        datasets = list(DS_ORDER)

    log("== L4 基线 %s ==" % SCHEMA_VERSION)
    log("配置：%s" % args.cfg)
    log("配置 sha256：%s" % cfg_sha)
    log("数据集：%s" % ", ".join(datasets))
    log("")

    t_all = time.time()
    cells, ds_meta, judge_meta = compute_cells(cfg, datasets, out_dir, log)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "built": "2026-11-05",
        "batch": "2026-11-05",
        "config": {"path": args.cfg, "sha256": cfg_sha},
        "metric_definitions": {
            "srcc": "Spearman rank correlation between the judge's cached scalar and y",
            "krocc": "Kendall tau",
            "plcc_raw": "Pearson on raw (pred, y)",
            "plcc_log": "Pearson after a 4-parameter logistic fit (l4_validate.plcc_logistic)",
            "y": "(raw_label - offset) / scale, per dataset config",
            "plcc_log_fit_input": "the raw label column (y*scale + offset), not the normalized y",
        },
        "known_rounding_ties": [
            {
                "cell": "tid2013/q-align/plcc_log",
                "value": 0.8458500043,
                "why": "真值正压 4 位小数平局点 0.84585；归一化 y 拟合得 0.84585006（印 0.8459），"
                       "原始 y 拟合得 0.84585000（印 0.8458）。评委 README 记 0.8458，故取原始 y。",
            }
        ],
        "stack_marks": STACK_MARK,
        "stack_desc": STACK_DESC,
        "sign_flipped_datasets": [d for d in datasets if ds_meta[d]["sign_flipped"]],
        "reconcile_tolerance": TOL,
        "datasets": ds_meta,
        "judges": judge_meta,
        "cells": cells,
        "elapsed_seconds": round(time.time() - t_all, 1),
    }

    summary_path = os.path.join(out_dir, "summary.json")
    with io.open(summary_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    report = build_report(cells, ds_meta, judge_meta, datasets, args.cfg, cfg_sha)
    report_path = os.path.join(out_dir, "report.md")
    with io.open(report_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(report)

    log("")
    log("总用时 %.1fs" % (time.time() - t_all))
    log("summary.json  %s  sha256=%s  %d B"
        % (os.path.basename(summary_path), sha256(summary_path), os.path.getsize(summary_path)))
    log("report.md     %s  sha256=%s  %d B"
        % (os.path.basename(report_path), sha256(report_path), os.path.getsize(report_path)))
    log_fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
