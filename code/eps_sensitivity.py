# -*- coding: utf-8 -*-
"""D2：ε 阈值敏感性与临界倍数 `s*`（只读，纯 CPU，不加载任何模型）。

首版 2026-10-28。**本文件产出阈值敏感性与临界倍数的全部读数**，本文件就是可复现的手段。

口径
----
ε 乘性缩放 `s·ε`，**逐数据集各按自身 ε 缩放**；判据 3 的位置项与尺度项、判据 4 的
「漂移域」上界同步缩放——只缩一处就是调参。

★ **2026-11-09：判据 4 改判幅度规则 `M > ε` 后重扫。**
改判之前 `label_at` 里还有一项 `has_drift = c4v == "fail"`（读落盘的旧 `p` 判决），
它是「无论 `s` 多大都不可能可比」的唯一来源；现在**删掉**，判据 4 只看幅度，与
判据 3 同尺。冻结的 `summary.json` 一字未改，仍按 `c4_detail` 重算（不重跑管线）。

**做法是读已冻结产物，不重跑判据管线。** `s` 能影响的只有三个比较：
    c3_delta : |δ̂|      <= s·ε   <=>  c3["abs_delta_over_eps"] <= s
    c3_beta  : beta_amp  <= s·ε   <=>  c3["beta_amp"] / c3["eps"] <= s
    c4 drift : |δ̂(c)−δ̂_global| <= s·ε（`c4_detail["per_domain_delta"]` / `delta_global`）
其余量与 ε 无关，改 `s` 不会动它们：
    · 判据 1 `c1a`／`c1b`、判据 2 `c2a`／`c2b`（分箱重叠、单调性、跨折稳定性）
    · 判据 4 的置换 `p`（统计量是跨域 max，零分布由置换生成，**不含 ε**）
    · 判据 3 的退化护栏 `na`（判 `min(SD(f_k)/SD(y)) >= c3_min_spread_ratio`，不含 ε）

**自校验（硬闸门）**：`s = 1` 必须逐项复现已冻结的门槛字典与聚合标签，否则本脚本
以非零码退出、拒绝报到任何结果。已核：两数据集各 252 个单元，门槛字典不一致 0、
标签不一致 0（504/504 全等）。

跑法
----
    python -X utf8 30-experiments/metrics/eps_sensitivity.py

（`-X utf8` 是必需的：默认控制台代码页会把中文判定标签打成乱码，数字仍对但读不了。）

结论（2026-11-09 重扫）
----------------------
**`s*` 有定义：TID2013 上 `s* = 1.0555`**（门槛全过的 32 条里最紧的一条，
`M/ε = 1.055490`）。可比计数 0 → 1 → 3 → 7 → 78 → 183 → 238（`s` = 1 → 1.0555 →
1.5 → 2 → 5 → 10 → 50）。**KonIQ-10k 上任何 `s` 都给不出可比**，因为那里的判据 4
是 `na`（无域标签）——那一条与 ε 无关，换的是「测不了」。
★ 改判之前（p 规则）的结论是「`s*` 无定义、可比恒为 0」；那一版**不删不改**。
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = {
    # ★ 2026-10-30：数据源由 v2 换 v3。v3 与 v2 的唯一差别是判据 2 的
    #   两侧网格（每侧各用自己评委的观测分范围）。c2b 在 v3 上是 252/252 全过，
    #   故「硬底」里原先属于 2b 的那部分（TID2013 9 条、KonIQ 13 条）随之消失，
    #   扫参的每一档计数都变了。旧 run 与其 `_script/` 副本一字未动，仍可复现。
    "tid2013": os.path.join(REPO, "30-experiments", "runs", "20261030-p1diagnose-v3-tid2013"),
    "koniq": os.path.join(REPO, "30-experiments", "runs", "20261030-p1diagnose-v3-koniq"),
}
GATE_KEYS_C1C2 = ("c1a", "c1b", "c2a", "c2b")
LABELS = ("可比", "条件可比", "不可比")
GRID = (0.25, 0.5, 1.0, 1.0555, 1.5, 2.0, 5.0, 10.0, 50.0, 1e9)


def label_at(cell, s):
    """按 ε 尺度 s 重建三分类判定（规则同 `comparability_criteria.aggregate_verdict`）。"""
    crit, c3 = cell["criteria"], cell["c3"]
    eps = float(c3["eps"])
    gate = {k: (crit.get(k) or {}).get("verdict") for k in GATE_KEYS_C1C2}
    gate["c3_delta"] = "pass" if c3["abs_delta_over_eps"] <= s else "fail"
    gate["c3_beta"] = "pass" if c3["beta_amp"] / eps <= s else "fail"
    if any(v == "na" for v in gate.values()):
        return "不可比"
    if any(v == "fail" for v in gate.values()):
        return "不可比"
    c4v = (crit.get("c4") or {}).get("verdict")
    if c4v is None or c4v == "na":
        return "条件可比"
    det = cell.get("c4_detail") or {}
    g, per = det.get("delta_global"), det.get("per_domain_delta") or {}
    # ★ 2026-11-09：判据 4 是**幅度规则**，判决只看 `M > s·ε`。
    #   这里刻意**不读**落盘的 `crit["c4"]["verdict"]` —— 那是改判前按 p 写下的旧语义。
    has_drift = bool(g is not None and per and
                     any(abs(float(v) - float(g)) > s * eps for v in per.values()))
    return "条件可比" if has_drift else "可比"


def load_cells(ds_name):
    d = json.load(open(os.path.join(RUNS[ds_name], "summary.json"), encoding="utf-8"))
    return d["datasets"][ds_name], [
        c for fams in d["datasets"][ds_name]["pairs"].values() for c in fams.values()
    ]


def guard(ds_name, cells):
    """s = 1 必须逐项复现已冻结的门槛字典与标签。"""
    bad_gate = bad_label = 0
    for c in cells:
        c3 = c["c3"]
        eps = float(c3["eps"])
        gate = {k: (c["criteria"].get(k) or {}).get("verdict") for k in GATE_KEYS_C1C2}
        gate["c3_delta"] = "pass" if c3["abs_delta_over_eps"] <= 1.0 else "fail"
        gate["c3_beta"] = "pass" if c3["beta_amp"] / eps <= 1.0 else "fail"
        bad_gate += (gate != c["aggregate"]["gate"])
        bad_label += (label_at(c, 1.0) != c["aggregate"]["label"])
    print("  guard(%s): n=%d  gate_mismatch=%d  label_mismatch=%d"
          % (ds_name, len(cells), bad_gate, bad_label))
    if bad_gate or bad_label:
        sys.exit("GUARD FAILED for %s -- refusing to report any s*" % ds_name)


def main():
    for name in ("tid2013", "koniq"):
        ds, cells = load_cells(name)
        print("== %s ==  n=%d  eps=%r" % (name, ds["n"], float(cells[0]["c3"]["eps"])))
        guard(name, cells)
        print("  recorded:", json.dumps(ds["aggregate_summary"]["reasons"], ensure_ascii=True))
        for s in GRID:
            cnt = {k: 0 for k in LABELS}
            for c in cells:
                cnt[label_at(c, s)] += 1
            print("  s=%-10g  可比=%d  条件可比=%3d  不可比=%3d"
                  % (s, cnt["可比"], cnt["条件可比"], cnt["不可比"]))
        pv = [((c["criteria"].get("c4") or {}).get("verdict")) for c in cells]
        ps = sorted((c.get("c4_detail") or {}).get("p_value") or -1 for c in cells)
        print("  c4 verdicts:", {v: pv.count(v) for v in set(pv)},
              " p: min=%g max=%g" % (ps[0], ps[-1]))
        # ★ 判据 4 的临界倍数 s* = min(M/ε)，只在前三道门槛能过的
        #   单元上取（过不了的单元无论 s 多大都不可比，取进来会把 s* 污染成 0）。
        amps = []
        for c in cells:
            c3 = c["c3"]
            eps = float(c3["eps"])
            gate = {k: (c["criteria"].get(k) or {}).get("verdict") for k in GATE_KEYS_C1C2}
            gate["c3_delta"] = "pass" if c3["abs_delta_over_eps"] <= 1.0 else "fail"
            gate["c3_beta"] = "pass" if c3["beta_amp"] / eps <= 1.0 else "fail"
            if any(v in ("na", "fail") for v in gate.values()):
                continue
            det = c.get("c4_detail") or {}
            g, per = det.get("delta_global"), det.get("per_domain_delta") or {}
            if g is None or not per:
                continue
            amps.append(max(abs(float(v) - float(g)) for v in per.values()) / eps)
        if amps:
            amps.sort()
            # ⚠️ 中位必须取真中位（偶数个时取中间两个的均值）。取 `amps[len//2]` 会在
            #    n = 32 时得到第 17 个值 4.115675，而正文 §V-A 报的是 4.106009 ——
            #    两者差 0.0097，是「上中位 vs 中位数」的口径差，不是数据差。
            _n = len(amps)
            med = (amps[_n // 2] if _n % 2 else
                   0.5 * (amps[_n // 2 - 1] + amps[_n // 2]))
            print("  s* = %.6f   (gate-passing n=%d, M/eps min/med/max = "
                  "%.6f / %.6f / %.6f)"
                  % (amps[0], _n, amps[0], med, amps[-1]))
        else:
            print("  s* undefined (no gate-passing decision carries a c4 amplitude)")


if __name__ == "__main__":
    main()
