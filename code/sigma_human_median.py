# -*- coding: utf-8 -*-
"""两个数据集的逐图中位 σ_human（只读数据集标签，不经 run 目录、不加载模型）。

起因：2026-11-03 的一次定向复核——稿子只给了 TID2013 侧的 `mos_std` 口径与
推导，**KonIQ-10k 侧的中位 σ_human 从未报出**，也没写两侧是同一泛函。

口径（与 `pred_table.py` 的 `kind == "column_se"` 分支逐字一致）
----------------------------------------------------------------
    σ_human(i) = [每图 MOS 均值估计量的标准误] / [定义 y 的量程]
    y          = (s − s_min) / (s_max − s_min)

  · **TID2013**：数据集直接给标准误（`mos_std.txt`，一格一个数、按官方
    25 参考 × 24 失真 × 5 强度的顺序排列），量程 0–9 ⇒ 除以 **9**。
  · **KonIQ-10k**：数据集给逐图评分计数，标准误要自己造：
    `SD / √c_total`（SD 是评分者间的标准差，c_total 是人数），量程 1–5 ⇒ 除以 **4**。

两侧因此是**同一个泛函**——「逐图 MOS 均值估计量的标准误，按该数据集的名义量程
换算到 y 上」；差别只在数据集把哪个量直接印出来。`SD(ŷ) = SD(ŷ_native)/range`
是 y 的线性归一化的直接推论，不是近似。

**为什么这条不进 `runs/`**：它不跑判据管线、不产生任何数据产物，只读两份公开标签
文件并打印一个中位数。它的两个输出值就是论文里 `σ_human` 中位数的来源。

跑法
----
    python -X utf8 30-experiments/metrics/sigma_human_median.py

硬闸门（任一条不过即非零退出，拒绝报数）
--------------------------------------
  · TID2013 参与统计的图数 = 2,687（官方 3,000 减去不可用项）
  · KonIQ-10k 参与统计的图数 = 2,015（官方 test 划分）
  · `mos_std.txt` 行数 = 3,000，且能按官方顺序与评测名单逐一对上
  · KonIQ 的 c1..c5 计数之和 = c_total（逐行）
  · √2 × 中位 σ_human 与稿中印的 ε（0.02014 / 0.01957）小数后 5 位相同
"""
import csv
import math
import os
import statistics
import sys

TID_ROOT = r"D:\datasets\tid2013"
KONIQ_ROOT = r"D:\datasets\koniq10k"

TID_EVAL_LIST = os.path.join(TID_ROOT, "labels", "tid2013_eval_list.txt")
TID_MOS_STD = os.path.join(TID_ROOT, "mos_std.txt")
KONIQ_SCORES = os.path.join(KONIQ_ROOT, "labels", "koniq10k_scores_and_distributions.csv")
KONIQ_SETS = os.path.join(KONIQ_ROOT, "labels", "koniq10k_distributions_sets.csv")

TID_SCALE = 9.0     # 名义量程 0–9
KONIQ_SCALE = 4.0   # 名义量程 1–5

EXPECTED = {"tid2013": 0.02014, "koniq": 0.01957}   # 稿中印的 ε


def fail(msg):
    sys.exit("闸门未过：%s" % msg)


def tid2013_sigma():
    """返回 (中位数, n)。按官方顺序把 mos_std 对齐到图名，再筛评测名单。"""
    evals = set()
    with open(TID_EVAL_LIST, encoding="utf-8") as f:
        for line in f:
            s = line.strip().lower()
            if s:
                evals.add(s)

    vals = []
    with open(TID_MOS_STD, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                vals.append(float(s))
    if len(vals) != 3000:
        fail("mos_std.txt 行数 = %d，应为 3000" % len(vals))

    # 官方顺序：参考图 01–25 × 失真类型 01–24 × 强度 1–5，扩展名 .bmp
    canon = ["i%02d_%02d_%d.bmp" % (r, t, level)
             for r in range(1, 26) for t in range(1, 25) for level in range(1, 6)]
    if len(set(canon)) != 3000:
        fail("官方顺序生成的图名有重复")

    picked = [(canon[i], vals[i]) for i in range(3000) if canon[i] in evals]
    if len(picked) != len(evals):
        fail("mos_std 对齐后 %d 条，评测名单 %d 条——有图名对不上"
             % (len(picked), len(evals)))
    if len(picked) != 2687:
        fail("参与统计的图数 = %d，应为 2687" % len(picked))

    sig = sorted(v / TID_SCALE for _, v in picked)
    return statistics.median(sig), len(sig)


def koniq_sigma():
    """返回 (中位数, n)。SD 与 c_total 取自 scores_and_distributions，划分取自 sets。"""
    scores = {}
    with open(KONIQ_SCORES, encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        need = {"image_name", "c1", "c2", "c3", "c4", "c5", "c_total", "SD"}
        missing = need - set(rd.fieldnames or [])
        if missing:
            fail("KonIQ 分数表缺列：%s" % sorted(missing))
        for row in rd:
            n = sum(int(row["c%d" % k]) for k in range(1, 6))
            if n != int(row["c_total"]):
                fail("%s 的 c1..c5 之和 %d ≠ c_total %s"
                     % (row["image_name"], n, row["c_total"]))
            scores[row["image_name"]] = (float(row["SD"]), float(row["c_total"]))

    test = []
    with open(KONIQ_SETS, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["set"].strip().lower() == "test":
                test.append(row["image_name"])
    if len(test) != 2015:
        fail("KonIQ test 划分 = %d 条，应为 2015" % len(test))

    sig = []
    for name in test:
        if name not in scores:
            fail("test 图 %s 在分数表里找不到" % name)
        sd, c = scores[name]
        sig.append((sd / math.sqrt(c)) / KONIQ_SCALE)
    sig.sort()
    return statistics.median(sig), len(sig)


def main():
    out = {}
    for key, fn in (("tid2013", tid2013_sigma), ("koniq", koniq_sigma)):
        med, n = fn()
        eps = math.sqrt(2.0) * med
        out[key] = (med, eps, n)
        print("%-8s n = %4d   中位 σ_human = %.6f   ε = √2·σ = %.6f"
              % (key, n, med, eps))
        if round(eps, 5) != EXPECTED[key]:
            fail("%s 的 ε = %.6f 与稿中 %.5f 不符，口径漂了"
                 % (key, eps, EXPECTED[key]))

    t, k = out["tid2013"][0], out["koniq"][0]
    print("两侧中位之比 = %.4f；ε 之比 = %.4f（稿中写 2.8%% 的差 = %.4f）"
          % (t / k, out["tid2013"][1] / out["koniq"][1],
             out["tid2013"][1] / out["koniq"][1] - 1.0))
    print("闸门通过：两侧中位与稿中 ε 逐位相符。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
