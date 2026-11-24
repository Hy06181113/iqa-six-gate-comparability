# -*- coding: utf-8 -*-
"""人类评分者分半对照（KonIQ-10k）：同一把 ε 下「人与自己」的可互换率。

用法：python -I metrics/human_split_half.py configs/20261102-human-splithalf-koniq.yaml

**这个脚本回答什么**：期刊稿的主结论是「252 个判定里没有一个 comparable」。
审稿人的第一反应必然是「那是模型不行，还是你那把尺子太细？」。把人类评分者
拆成两半、让**它们**去跑同一个 `P`，就能把这个反问变成读数——不需要另立一把尺子。

**口径（写死；改动前须先更新本节与对应配置）**

1. **只有 KonIQ-10k 能做**。它的 `c1..c5` 是 1–5 级的个体评分**计数**，
   可以按计数拆。TID2013 只发聚合 `MOS` 与 `mos_std.txt`，**没有逐人票** ⇒
   本对照不覆盖它。正文必须如实写成 limitation，不得拿模型顶替。
2. **拆分是按计数的二项近似**：每个 k 级上 `c_k → Binomial(c_k, p)`，两个半
   互为补。它**不是**「同一批评分者精确对半分」——原始数据里没有评分者 ID。
   后果要跟着报：两半各约 `c_total/2` 票 ⇒ 各自的 MOS 估计更噪 ⇒ 算出的 `P`
   是**偏严的下界**，真人两个整半会更宽。
3. **`ε` 不复算**，直接读 `20261030-p1diagnose-v3-koniq` 的 `iso_eps`
   （0.01956538834685546），保证与 252 格模型判定用的是同一把尺子。
   本脚本**不写**任何 `eps` 到既有产物里。
4. **只取官方 `set == test` 的 2,015 张**，与 P1 同批图。

**复用而不重写**：`comparability_criteria.interchangeability()` 给出 `P`／
`P_strict`／ε 阶梯（与模型的 252 格逐字同一定义）；`_decompose()` 给出
`δ̂`／`β̂`／`σ̂`。两者都按原样调用，不改一行。

**要看的是 `β̂` 那一列**：两半各带独立的测量误差 ⇒ `d = e₁ − e₂` 与 `y` 无关
⇒ `β̂ ≈ 0`；而模型的尺度项在 TID2013 上有 217/252 挂。这条判别**不需要域标签**，
所以在 KonIQ 上也能做——它把「模型分歧是结构性的、人类分歧是噪声」从断言变成读数。
⚠️ 若实测下来这一列分不开，**如实报分不开**，不改数、不删列。
"""
import io
import json
import os
import shutil
import sys
import time
import hashlib

import numpy as np
import yaml

# 本机控制台是 GBK，而日志里有 δ̂ / ε 这类组合字符 ⇒ 显式转 UTF-8，否则 print 就崩
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import comparability_criteria as cc  # noqa: E402
import pred_table as pt  # noqa: E402

_LOG = []


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    _LOG.append(line)
    print(line)
    sys.stdout.flush()


def sha256_of(path):
    h = hashlib.sha256()
    with io.open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_script(outdir, cfg_path, notes):
    """把当次真正执行的文件按原名复制进 run 的 `_script/`。"""
    sd = os.path.join(outdir, "_script")
    os.makedirs(sd, exist_ok=True)
    files = [os.path.abspath(sys.argv[0]), os.path.abspath(cfg_path),
             os.path.join(HERE, "comparability_criteria.py"),
             os.path.join(HERE, "pred_table.py")]
    rows = []
    for src in files:
        base = os.path.basename(src)
        dst = os.path.join(sd, base)
        shutil.copyfile(src, dst)
        same = sha256_of(src) == sha256_of(dst)
        rows.append((base, sha256_of(dst)[:16], "相同" if same else "不同"))
    lines = [
        "# 当次执行的脚本",
        "",
        "本目录按原名存放**当次真正执行**的文件。仓库副本可能已被继续修改，"
        "**以本目录的字节为准**——产出这批数字的是这一版。",
        "",
        "| 文件 | 当次 sha256（前 16 位） | 与仓库副本 |",
        "|---|---|---|",
    ]
    for base, sha, same in rows:
        lines.append("| `%s` | `%s` | %s |" % (base, sha, same))
    lines += [
        "",
        "**正典位**：本脚本是**一次性诊断**（不是某个评委的标准跑法），"
        "故不适用 `judges/<评委>/run_local.sh`；第三列只是运行时刻的快照。",
        "",
        notes,
    ]
    io.open(os.path.join(sd, "README.md"), "w", encoding="utf-8",
            newline="\n").write("\n".join(lines) + "\n")


def half_counts(counts, rng, p):
    """按级二项拆分：返回两个半 panel 的各级计数，二者严格互补。"""
    a = rng.binomial(counts, p)
    return a, counts - a


def mos_from_counts(half, levels):
    tot = half.sum(axis=1)
    # 空半 panel 理论上不可能（c_total ≥ 94），但除法前仍显式挡一下
    tot = np.where(tot > 0, tot, 1)
    return (half * levels).sum(axis=1) / tot


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        REPO_ROOT, "30-experiments", "configs", "20261102-human-splithalf-koniq.yaml")
    cfg = yaml.safe_load(io.open(cfg_path, encoding="utf-8"))
    ds = cfg["dataset"]
    R = int(cfg["R"])
    p = float(cfg["split"]["p"])
    outdir = pt.resolve(cfg["out"])
    os.makedirs(outdir, exist_ok=True)

    log("配置：%s" % cfg_path)
    log("R = %d，p = %g" % (R, p))

    # ── ε：只读不算 ────────────────────────────────────────────────
    eps_run = pt.resolve(cfg["eps_source"]["run"])
    summary_path = os.path.join(eps_run, "summary.json")
    with io.open(summary_path, encoding="utf-8") as f:
        eps = float(json.load(f)["datasets"]["koniq"]["iso_eps"])
    log("ε 取自 %s/datasets.koniq.iso_eps = %.17g" % (cfg["eps_source"]["run"], eps))
    if not (eps > 0):
        raise SystemExit("ε 非正：%r" % eps)

    # ── 图名单：官方 set == test ───────────────────────────────────
    ds_cfg = {"labels_csv": ds["labels_csv"], "label_filter": ds.get("label_filter")}
    _, label_rows = pt._load_labels(ds_cfg)
    names = [r["image_name"] for r in label_rows]
    log("test 图 %d 张" % len(names))

    _, aux_rows = pt._read_csv_rows(pt.resolve(ds["aux_labels_csv"]))
    aux = {r["image_name"]: r for r in aux_rows}
    missing = [n for n in names if n not in aux]
    if missing:
        raise SystemExit("%d 张 test 图在 aux 表里找不到，例：%s"
                         % (len(missing), missing[:3]))

    lv_cols = list(ds["counts_columns"])
    counts = np.asarray([[int(aux[n][c]) for c in lv_cols] for n in names],
                        dtype=np.int64)
    c_tot = np.asarray([int(aux[n][ds["total_column"]]) for n in names],
                       dtype=np.int64)
    mos_full = np.asarray([float(aux[n][ds["mos_column"]]) for n in names],
                          dtype=np.float64)

    # ── 自检 1：计数之和 = c_total；计数还原的 MOS = MOS 列 ─────────
    if not np.array_equal(counts.sum(axis=1), c_tot):
        raise SystemExit("c1..c5 之和与 c_total 不等，取错表了（'%s' 里的 c1..c5 "
                         "是归一化比例，不是计数）" % os.path.basename(ds["aux_labels_csv"]))
    levels = np.arange(1, counts.shape[1] + 1, dtype=np.float64)
    mos_rebuilt = mos_from_counts(counts, levels)
    dev = float(np.max(np.abs(mos_rebuilt - mos_full)))
    if dev > 1e-9:
        raise SystemExit("计数还原的 MOS 与 MOS 列差 %.3g，口径不一致" % dev)
    log("自检 1 通过：c1..c5 之和 = c_total（%d–%d 票）；计数还原 MOS 与 MOS 列最大差 %.2e"
        % (c_tot.min(), c_tot.max(), dev))

    scale = float(ds["y"]["scale"])
    offset = float(ds["y"]["offset"])
    y = (mos_full - offset) / scale
    span = float(np.percentile(y, 95) - np.percentile(y, 5))

    # ── 重复拆分 ───────────────────────────────────────────────────
    seed = int(cfg["seed"])
    rows = []
    for rep in range(R):
        rng = np.random.default_rng(seed + rep)
        ca, cb = half_counts(counts, rng, p)
        za = (mos_from_counts(ca, levels) - offset) / scale
        zb = (mos_from_counts(cb, levels) - offset) / scale
        inter = cc.interchangeability(za, zb, y, eps, domains=None,
                                      cfg=cfg["interchangeability"])
        delta, beta, sigma = cc._decompose(za - zb, y)
        rows.append({
            "rep": rep,
            "P": inter["value"],
            "P_strict": inter["p_strict"],
            "P_1x": inter["ladder"]["1x"],
            "P_2x": inter["ladder"]["2x"],
            "P_5x": inter["ladder"]["5x"],
            "delta": delta,
            "beta": beta,
            "sigma": sigma,
            "beta_span_over_eps": abs(beta) * span / eps,
            "abs_delta_over_eps": abs(delta) / eps,
        })
        if (rep + 1) % 50 == 0:
            log("  %d/%d" % (rep + 1, R))

    def med(key):
        return float(np.median([r[key] for r in rows]))

    def band(key):
        v = [r[key] for r in rows]
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    metrics = {}
    for key in ("P", "P_strict", "P_1x", "P_2x", "P_5x", "delta", "beta",
                "sigma", "beta_span_over_eps", "abs_delta_over_eps"):
        lo, hi = band(key)
        metrics[key] = {"median": med(key), "lo": lo, "hi": hi}

    # ── 自检 2：两条结构性预期（不是断言结果，是把「该为 0 的」核出来）──
    checks = {
        "n": int(len(names)),
        "reps": R,
        "seed": seed,
        "eps": eps,
        "c_total_min": int(c_tot.min()),
        "c_total_med": float(np.median(c_tot)),
        "c_total_max": int(c_tot.max()),
        "y_span_p05_p95": span,
        "counts_sum_equals_c_total": True,
        "mos_rebuild_max_abs_dev": dev,
        # 期望：两半互为补 ⇒ 平均下来应与全量 MOS 无系统偏移
        "delta_over_eps_median": metrics["delta"]["median"] / eps,
    }
    log("自检 2：δ̂/ε 中位 = %.4f（|·| < 1 说明两半没有系统偏移）"
        % checks["delta_over_eps_median"])

    summary = {
        "schema_version": "human-splithalf-1",
        "config": cfg_path,
        "dataset": ds["name"],
        "caliber": {
            "split": "按级二项近似（c_k → Binomial(c_k, p)）；无评分者 ID，"
                     "不是同一批评分者精确对半分",
            "conservative": "两半各约 c_total/2 票 ⇒ P 是偏严下界，真两整半会更宽",
            "not_covering": "TID2013 无逐人票，本对照不覆盖该数据集",
            "eps": "只读 20261030-p1diagnose-v3-koniq 的 iso_eps，未复算",
        },
        "check": checks,
        "metrics": metrics,
        "per_rep": rows,
    }
    s_path = os.path.join(outdir, "summary.json")
    io.open(s_path, "w", encoding="utf-8", newline="\n").write(
        json.dumps(summary, ensure_ascii=False, indent=1) + "\n")

    # splits.csv
    cols = ["rep", "P", "P_strict", "P_1x", "P_2x", "P_5x", "delta", "beta",
            "sigma", "beta_span_over_eps", "abs_delta_over_eps"]
    buf = [",".join(cols)]
    for r in rows:
        buf.append(",".join([str(r["rep"])] + ["%.10g" % r[c] for c in cols[1:]]))
    io.open(os.path.join(outdir, "splits.csv"), "w", encoding="utf-8",
            newline="\n").write("\n".join(buf) + "\n")

    # report.md
    md = [
        "# 人类评分者分半对照（KonIQ-10k）",
        "",
        "> 由 `metrics/human_split_half.py` 生成，真源是同目录 `summary.json` 与 "
        "`splits.csv`；当次执行件的 sha256 见 `_script/README.md`。",
        "",
        "## 0. 本次运行",
        "",
        "| 项 | 值 |",
        "|---|---|",
        "| 数据集 | KonIQ-10k（官方 `set == test`） |",
        "| 图数 | **%d** |" % checks["n"],
        "| ε（只读，未复算） | **%.17g** |" % eps,
        "| 重复拆分 | **%d** 次，种子 %d 起逐次 +1 |" % (R, seed),
        "| 每图票数 `c_total` | %d ／ 中位 %.0f ／ %d |"
        % (checks["c_total_min"], checks["c_total_med"], checks["c_total_max"]),
        "",
        "## 1. 口径（三条边界，引用时必须跟着报）",
        "",
        "1. **只有 KonIQ-10k 能做**：`c1..c5` 是 1–5 级的个体评分计数。"
        "TID2013 只发聚合 `MOS` 与 `mos_std.txt`，没有逐人票 ⇒ 本对照不覆盖它。",
        "2. **拆分是按计数的二项近似**（每级 `c_k → Binomial(c_k, %.2f)`），"
        "两个半互为补；原始数据没有评分者 ID，所以它不是「同一批评分者精确对半」。"
        "两半各约 `c_total/2` 票 ⇒ 估计更噪 ⇒ **`P` 是偏严下界**。" % p,
        "3. **`ε` 只读不算**，取自 `20261030-p1diagnose-v3-koniq` 的 `iso_eps`，"
        "与 252 格模型判定同一把尺子。本脚本不往任何既有产物写 `eps`。",
        "",
        "## 2. 读数（%d 次拆分的中位，括号内为 2.5–97.5 分位）" % R,
        "",
        "| 量 | 中位（区间） |",
        "|---|---|",
    ]
    label = {
        "P": "`P`（主口径，`|z_a − z_b| < ε` 的比例）",
        "P_strict": "`P_strict`（各自与人类真值都在 ε 内）",
        "P_1x": "ε 阶梯 1×",
        "P_2x": "ε 阶梯 2×",
        "P_5x": "ε 阶梯 5×",
        "delta": "`δ̂`（两半 MSE 估计之差的位置项）",
        "beta": "`β̂`（`d` 对 `y` 的斜率）",
        "sigma": "`σ̂`（残差项）",
        "beta_span_over_eps": "`|β̂|·span/ε`（与判据 3 尺度项同尺）",
        "abs_delta_over_eps": "`|δ̂|/ε`（与判据 3 位置项同尺）",
    }
    for key in cols[1:]:
        m = metrics[key]
        md.append("| %s | **%.4f**（%.4f – %.4f） |"
                  % (label[key], m["median"], m["lo"], m["hi"]))
    md += [
        "",
        "## 3. 自检",
        "",
        "| 项 | 结果 |",
        "|---|---|",
        "| `c1..c5` 之和 = `c_total` | 是（%d–%d 票） |"
        % (checks["c_total_min"], checks["c_total_max"]),
        "| 计数还原的 MOS 与 `MOS` 列 | 最大绝对差 **%.2e** |"
        % checks["mos_rebuild_max_abs_dev"],
        "| `δ̂/ε` 中位 | **%.4f**（两半互为补，不应有系统偏移） |"
        % checks["delta_over_eps_median"],
        "",
        "## 4. 读法",
        "",
        "⚠️ **这不是第二把尺子。** 它不产生新阈值、不改任何判决，只是把「0/252 是"
        "模型不行还是尺子太细」这个反问变成同一把尺子上的另一组读数。",
        "",
        "⚠️ **不得读成「模型与人类同样可互换」。** 判别落在分歧的**结构**上："
        "两半带独立测量误差 ⇒ `d = e₁ − e₂` 与 `y` 无关 ⇒ `β̂ ≈ 0`；"
        "而模型的尺度项在 TID2013 上 217/252 挂。`β̂·span/ε` 这一行才是可比的那一列。",
        "",
        "⚠️ **`P` 是偏严下界**（两半只有半个 panel 的票）。真人两个整半会更宽；"
        "所以这里读出多少，都只能当下界用。",
    ]
    io.open(os.path.join(outdir, "report.md"), "w", encoding="utf-8",
            newline="\n").write("\n".join(md) + "\n")

    log("写入：summary.json / splits.csv / report.md")
    stage_script(outdir, cfg_path,
                 "本次为一次性诊断 run（人类分半对照），无正典位。")
    log("初始 _script/ 完成")

    io.open(os.path.join(outdir, "run.log"), "w", encoding="utf-8",
            newline="\n").write("\n".join(_LOG) + "\n")


if __name__ == "__main__":
    main()
