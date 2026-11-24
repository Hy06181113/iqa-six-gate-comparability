# -*- coding: utf-8 -*-
"""L4 复现验证：把一个评委的分数缓存对官方标签，算 SRCC / PLCC / KROCC。

用途与位置
----------
这是 `30-experiments/README.md` 里三层指标的**第一层（单模型基线）**。
「L4 验证」指：在 KonIQ-10k **官方 test split** 上把该评委的分数对官方标签算相关。

★ **关于 `--target-srcc / --target-plcc`（2026-10-09 修订，勿再当门槛用）**
    这两个参数只是「顺手比一下」，**不是判据**。理由：
      · MANIQA 常被引作目标的 SRCC **0.930** / PLCC **0.946**，出处是作者仓库 **checkpoint
        表的表头**，表里写 KonIQ-10K **8058 训练 / 2015 测试**。而 8058 = 官方 7058 train
        + 1000 validation，2015 = 官方 test ⇒ **测试集与我们的相同**，split 不是差异来源。
      · 那两个数是在作者**自有协议与分辨率**上算的；本机跑的是 pyiqa 默认 **512×384**。
        2026-10-09 实测本机得 SRCC 0.9042 / PLCC(log) 0.9349，与自报差 −0.0258 / −0.0111。
      · ⇒ 拿自报值当门槛会**误判封装有误**。第一层基线的真门槛是「对官方示例图的复现精度」
        （MANIQA 为 0.4%），见 `judges/maniqa/README.md`「L4 验证」节。
    **给 target 是为了留痕，不是为了判过不过。**

★ 三个必须钉死的口径（错一个数字就对不上，且不会报错）
    1. **split**：用 `koniq10k_distributions_sets.csv` 的 `set` 列。
       ⚠️ 官方是 training 7058 / validation 1000 / **test 2015**。
       常见误传的「8,049 / 2,024」来自 pyiqa 里 `split_ratio='802'+seed123` 那条分支，
       但 KonIQ 的配置写的是 `split_index: 'official_split'`（字符串），**不走那条分支**。
    2. **MOS**：用同一份官方 CSV 的 `MOS` 列（**0–100 制**）。
       本项目另存的 `koniq10k_scores_and_distributions.csv` 是 **1–5 制**，
       两者 Spearman 仅 0.9919，**不是同一组标签**，不可混用。
    3. **PLCC 是否做非线性拟合**：IQA 文献里 PLCC 惯例是**先做 4 参数 logistic 回归再算
       Pearson**（因分数与主观感受非线性）。本脚本 raw 与 logistic **都算**，
       用哪个以「能否对上论文自报值」为准，不凭印象。

用法
----
    python l4_validate.py --pred <缓存.jsonl> --labels <官方sets.csv> \
                          [--split test] [--target-srcc 0.930] [--target-plcc 0.946]
    python l4_validate.py --pred ... --labels ... --emit-list test --out-list test_split.txt
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr, kendalltau


def _logistic4(x, b1, b2, b3, b4):
    """标准 4 参数 logistic（IQA 文献惯例，见 VQEG / ITU-T P.1401 的做法）。"""
    return b1 + (b2 - b1) / (1.0 + np.exp(-(x - b3) / np.abs(b4)))


def plcc_logistic(pred, mos):
    """先拟合 logistic 再取 Pearson。拟合失败时返回 (nan, None) 而不是抛。"""
    p0 = [min(mos), max(mos), np.mean(pred), np.std(pred) or 1.0]
    try:
        popt, _ = curve_fit(_logistic4, pred, mos, p0=p0, maxfev=20000)
        fitted = _logistic4(pred, *popt)
        return float(pearsonr(mos, fitted).statistic), popt.tolist()
    except Exception as e:                                        # noqa: BLE001
        print("[warn] logistic 拟合失败：%s: %s" % (type(e).__name__, e), file=sys.stderr)
        return float("nan"), None


def load_pred(path):
    """读 infer.py 产出的 jsonl：{image: 路径, score: 分数}。"""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[os.path.basename(r["image"])] = float(r["score"])
    return out


def load_labels(path):
    """读官方 CSV：返回 {image_name: (MOS, set)}。列名按官方文件硬校验。"""
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        need = {"image_name", "MOS", "set"}
        missing = need - set(rd.fieldnames or [])
        if missing:
            sys.exit("标签文件缺列 %s，实际列：%s" % (sorted(missing), rd.fieldnames))
        return {r["image_name"]: (float(r["MOS"]), r["set"]) for r in rd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="评委分数缓存 .jsonl")
    ap.add_argument("--labels", required=True, help="官方 koniq10k_distributions_sets.csv")
    ap.add_argument("--split", default="test",
                    help="set 列取值：training / validation / test（默认 test）")
    ap.add_argument("--target-srcc", type=float, default=None)
    ap.add_argument("--target-plcc", type=float, default=None)
    ap.add_argument("--emit-list", default=None,
                    help="只导出该 split 的图片名到 --out-list，不计算指标")
    ap.add_argument("--out-list", default=None)
    args = ap.parse_args()

    labels = load_labels(args.labels)
    if args.emit_list:
        names = sorted(k for k, v in labels.items() if v[1] == args.emit_list)
        if not args.out_list:
            sys.exit("--emit-list 需要同时给 --out-list")
        with open(args.out_list, "w", encoding="utf-8") as f:
            f.write("\n".join(names) + "\n")
        print("已导出 %s 的 %d 个文件名 -> %s" % (args.emit_list, len(names), args.out_list))
        return 0

    pred = load_pred(args.pred)

    # 只取该 split，且**只取两边都有的图**——不静默补全，缺了要说出来
    in_split = sorted(k for k, v in labels.items() if v[1] == args.split)
    used = [k for k in in_split if k in pred]
    n_split, n_used = len(in_split), len(used)
    if n_used == 0:
        sys.exit("该 split 下一个可用样本都没有：pred %d 条，split=%s 共 %d 张"
                 % (len(pred), args.split, n_split))
    if n_used != n_split:
        print("[warn] split=%s 共 %d 张，但只有 %d 张在预测缓存里（缺 %d 张）"
              % (args.split, n_split, n_used, n_split - n_used), file=sys.stderr)

    # 按文件名排序，保证结果与顺序无关（也便于换人复算）
    p = np.array([pred[k] for k in used], dtype=np.float64)
    y = np.array([labels[k][0] for k in used], dtype=np.float64)

    srcc = float(spearmanr(p, y).statistic)
    krocc = float(kendalltau(p, y).statistic)
    plcc_raw = float(pearsonr(p, y).statistic)
    plcc_log, popt = plcc_logistic(p, y)

    print("=" * 66)
    print("L4 复现验证   split = %s" % args.split)
    print("=" * 66)
    print("  缓存      : %s" % args.pred)
    print("  标签      : %s  (MOS 制式见 --labels 文件本身)" % args.labels)
    print("  样本      : %d / %d 张" % (n_used, n_split))
    print("  预测范围  : %.4f .. %.4f" % (p.min(), p.max()))
    print("  标签范围  : %.4f .. %.4f" % (y.min(), y.max()))
    print("-" * 66)
    print("  SRCC      : %.4f" % srcc)
    print("  KROCC     : %.4f" % krocc)
    print("  PLCC(raw) : %.4f   <- 直接 Pearson" % plcc_raw)
    print("  PLCC(log) : %.4f   <- 4 参数 logistic 拟合后 Pearson（IQA 惯例）"
          % plcc_log)
    if args.target_srcc is not None:
        print("-" * 66)
        print("  目标 SRCC : %.4f   →  差 %+.4f" % (args.target_srcc, srcc - args.target_srcc))
    if args.target_plcc is not None:
        print("  目标 PLCC : %.4f   →  raw 差 %+.4f ／ log 差 %+.4f"
              % (args.target_plcc, plcc_raw - args.target_plcc, plcc_log - args.target_plcc))
        # 哪个口径对得上，就写哪个——不预设
        near = "log" if abs(plcc_log - args.target_plcc) < abs(plcc_raw - args.target_plcc) else "raw"
        print("  ⇒ 更接近目标的是 **%s** 口径" % near)
        print("  ⚠️ 上面这个「差」**不判封装对错**——目标是作者自报、在作者自有分辨率算的，")
        print("     与 512×384 口径不同源。判据见本脚本 docstring 与评委 README。")
    print("=" * 66)
    print("★ 提醒一（纪律第 3 条）：本验证只为确认**封装实现无误**，")
    print("  不代表该评委在 KonIQ 上无污染——MANIQA 正是在 KonIQ-10k 上微调的。")
    print("★ 提醒二：硬门槛是「对官方 demo 默认图的复现精度」（MANIQA 实测差 0.4%），")
    print("  不是这里的 --target-*。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
