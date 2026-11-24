# -*- coding: utf-8 -*-
"""P1 第①步「分数采集」的 I/O 层：把 9 份评委分数缓存与数据集标签读成**已对齐**的矩阵。

口径（先于任何计算写死，对应下列两条声明）：

  1. **一律用 `os.path.basename` 对齐**：评委 jsonl 的 `image` 字段是完整路径。
  2. **TID2013 的文件名是结构化标签**：`I02_01_1.bmp` = 参考图 02 / 失真类型 01 / 强度级 1。
     判据 4 的分块键与分层切分的分组键都从这里解析。
  3. **人类尺度 y 与 σ_human 必须用同一个归一因子**（这是判据 3 能比的前提）：
       TID2013  y = MOS / 9                      σ_human = mos_std / 9
       KonIQ    y = (m − 1) / 4                  σ_human = (SD_c / √c_total) / 4
                （m、SD_c 均为 1–5 制；c_total 是该图的**个体评分数**，实测 94–156）
     ★ **σ_human 的语义 = 「该图人类 MOS 估计量的不确定度」（标准误）**，两列都是这个量。
       2026-10-15 晚改定：KonIQ 原先直取 `SD_c`，但 `SD_c` 是**单个评分者绕图上均值的离散度**
       （已用 `c1..c5` 计数自算 ddof=1 的 SD 逐图比对证实：10,073 行最大差 2.4e-15），
       与 TID2013 的 `mos_std`（官方 readme「standard deviation of MOS」、SPIC 2015 论文
       称其为 RMSE of MOS）**不是同一类量** —— 占人类尺度幅宽的比例 14.25% ↔ 1.42%
       （差 10.01 倍，与两侧 ε 之比逐位一致）。
       两者混用会让判据 3 残差项在两个数据集上得出相反判定（0/252 ↔ 252/252）。
       理由与证据链见上；本文件的 `column_se` 分支就是这次改定的落点。
     ⚠️ 判据 3 的全部判定（三分量与 ε 的比较）在 `y → αy + β`（α > 0）下**不变**，
        所以上表里除以 4 还是除以 5 **不影响任何结论**；
        这个选择只影响判据 1b（固定 [0,1] 的 20 等宽 bin）与判据 2b（归一化幅宽）。
        选「把评分量表的名义量程投到 [0,1]」是为了让这两条也可读。
  4. ⚠️ **KonIQ 的锚点与 SD 不得取自 `distributions_sets.csv` 那个叫 `MOS` 的 0–100 列**：
     实测它是 `scores_and_distributions.csv` 里的 `MOS_zscore`（逐被试 z-score 后的重标度量），
     对 1–5 加权均值只有 R²=0.9908 的仿射关系、最大偏差 4.5 个 MOS 点，重标度配方未公开。
     改用 1–5 制那张表则 `MOS` 与由 `c1..c5` 计数算出的加权均值相差 6.2e-15 —— 零假设。
  5. ⚠️ **TID2013 的 `mos_std.txt` 每行只有一个数、不带文件名**，须与 `mos_with_names.txt`
     （`<MOS> <文件名>`，同为 3,000 行、同一官方序）**按行号 zip**。该对齐已独立证实：
     我们的标签 CSV 的 `MOS` 与 `mos_with_names.txt` 逐图比对 **maxabs = 0.000000**（n = 2,687）。
  6. ★ **`sigma_human.kind = not_available`（2026-10-19 新增，为 AGIQA-3K 而设）**：
     该数据集**不存在**可用的「人类 MOS 估计量不确定度」口径（AGIQA-3K 的 `SD` 列定义未核实、
     且无逐图 `n`）⇒ `Table.sigma_human` 返回**全 NaN**，`reason` 键必填。
     ⚠️ **这类数据集只能在 P2（无量纲 `R` / `D`）里用，不得进 P1 诊断**——判据 3 的 ε 由
     `median(σ_human)` 定义，NaN 会穿透成 NaN 判定，而不是给出一个看起来合理的结果。
     本分支**只**解决「P2 的输入适配器要穿过 `build_table` 取 y 与评委列」这一件事。
"""
import csv
import io
import os
import re
import sys

import numpy as np

# 复用 `l4_validate.load_pred`（jsonl → {basename: float}）。它被 import 即属当次执行件，
# 须一并进 run 目录的 `_script/`。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l4_validate import load_pred                                  # noqa: E402

def _find_root():
    """相对路径的基准目录。**在仓库里与在发布包里必须是同一个语义**（「这棵树的根」）。

    仓库里本文件在 `30-experiments/metrics/` ⇒ 根在两上级；发布包里它在 `code/` ⇒ 根在
    上一级。原实现写死「两上级」，脚本搬进包内后基准就跑到包**外面**去了（2026-11-12
    实测：包里 `resolve("infer/configs/…")` 解析到 `80-release/infer/…`，配置根本找不到）。
    判据用两个各自唯一的标记：根下有 `30-experiments/`（仓库）或 `scores/`（发布包）。
    两上级优先——仓库的形态不变，取值与改前逐字节相同。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "..", ".."), os.path.join(here, "..")):
        cand = os.path.normpath(cand)
        if (os.path.isdir(os.path.join(cand, "30-experiments"))
                or os.path.isdir(os.path.join(cand, "scores"))):
            return cand
    return os.path.normpath(os.path.join(here, "..", ".."))


REPO_ROOT = _find_root()

# TID2013 官方命名约定（固定，不由配置驱动 —— 放进 YAML 只会多一层转义风险）
TID_NAME_RE = re.compile(r"^i(\d+)_(\d+)_(\d+)\.(bmp)$", re.IGNORECASE)


class Table(object):
    """一个数据集上、**已逐图对齐**的分数矩阵与标签。"""

    def __init__(self, dataset, names, judge_names, X, y, sigma_human, groups, domains):
        self.dataset = dataset
        self.names = names                    # list[str]，顺序即所有矩阵的列序
        self.judge_names = judge_names        # list[str]
        self.X = X                            # (n_judges, n_images) 原始分
        self.y = y                            # (n_images,) 归一到幅宽 1 的人类尺度
        self.sigma_human = sigma_human        # (n_images,) 与 y 同尺度的**人类真值不确定度**（MOS 估计量的标准误）
        self.groups = groups                  # list[str] | None，分层切分的分组键
        self.domains = domains                # list[str] | None，判据 4 的分块键
        self.n = len(names)

    def row(self, judge):
        return self.X[self.judge_names.index(judge)]


def resolve(path):
    """相对路径按**仓库根**解析（不是按 cwd、也不是按配置文件目录）——避免 cwd 依赖。"""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(REPO_ROOT, path))


def parse_tid2013_name(name):
    """`I02_01_1.bmp` → ('02', '01', '1')。失败抛 ValueError（不静默）。"""
    m = TID_NAME_RE.match(os.path.basename(name))
    if not m:
        raise ValueError("不是 TID2013 官方命名：%r" % name)
    return m.group(1), m.group(2), m.group(3)


def _read_csv_rows(path):
    with io.open(path, encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        return rd.fieldnames, list(rd)


def _load_labels(ds_cfg):
    """返回 {basename: y}（未归一）与选中的文件名列表，供两种数据集共用。"""
    cols, rows = _read_csv_rows(resolve(ds_cfg["labels_csv"]))
    filt = ds_cfg.get("label_filter") or {}
    fcol, fvals = filt.get("column"), filt.get("values")
    if fcol is not None and fvals is not None:
        if fcol not in cols:
            raise SystemExit("标签文件缺过滤列 %r，实际列：%s" % (fcol, cols))
        rows = [r for r in rows if r[fcol] in fvals]
    return cols, rows


def _sigma_human_tid(cfg, names):
    """与 `mos_with_names.txt` 按行号 zip 取 std。两文件行数不等即抛。"""
    wpath, spath = resolve(cfg["mos_with_names"]), resolve(cfg["mos_std"])
    wnames, wmos = [], []
    for line in io.open(wpath, encoding="utf-8"):
        p = line.split()
        if len(p) >= 2:
            wmos.append(float(p[0]))
            wnames.append(os.path.basename(p[1]))
    std = [float(l) for l in io.open(spath, encoding="utf-8") if l.strip()]
    if not (len(wnames) == len(wmos) == len(std)):
        raise SystemExit("mos_with_names(%d/%d) 与 mos_std(%d) 行数不等，按行号 zip 不成立"
                         % (len(wnames), len(wmos), len(std)))
    smap = {n.lower(): s for n, s in zip(wnames, std)}
    out = []
    for n in names:
        k = os.path.basename(n).lower()
        if k not in smap:
            raise SystemExit("mos_std 未覆盖：%s" % n)
        out.append(smap[k] / float(cfg["scale"]))
    return np.asarray(out, dtype=np.float64)


def _table_from_merged(ds_name, ds_cfg, judges_cfg, verbose=True):
    """从**已并好的** `scores/p2_<数据集>.csv` 建表（发布包内的读者路径）。

    ⚠️ 这条路径**不是**论文读数产生时走的那条。论文的每个数都由 `labels_csv` ＋ 逐评委
    `preds` 经 `build_table()` 的主分支算出；本分支 2026-11-12 才加，
    只为一件事实：发布包里发的是**并好的表**，没有逐评委缓存与标签原件的副本。
    两边的数值同源——并表里每个评委列就是该评委 jsonl 的 `score`，`y` 就是标签算出的
    归一人类尺度（已抽样逐值核对，2,687 行无缺）——所以本分支**不重算任何量，只换 I/O**。

    σ_human **不在**并表里 ⇒ 给全 NaN 并打印一行提示。消费它的判据（判据 3）仍须走
    `labels_csv` 路径；判据 1／2／4 与对级全局检验只用 `y` 与评委矩阵，本路径够用。
    """
    path = resolve(ds_cfg["merged_csv"])
    if not os.path.isfile(path):
        raise SystemExit("并表不存在：%s" % path)
    cols, rows = _read_csv_rows(path)
    for need in ("image", "y"):
        if need not in cols:
            raise SystemExit("并表缺列 %r：%s" % (need, path))

    judge_names = list(ds_cfg.get("merged_judges") or sorted(judges_cfg.keys()))
    miss = [j for j in judge_names if j not in cols]
    if miss:
        raise SystemExit("并表缺评委列 %s（实际列：%s）" % (miss, cols))

    names = [os.path.basename(r["image"]) for r in rows]
    if len(set(names)) != len(names):
        raise SystemExit("并表出现重复文件名 ⇒ 对齐不可靠")
    y = np.asarray([float(r["y"]) for r in rows], dtype=np.float64)
    X = np.asarray([[float(r[j]) for r in rows] for j in judge_names], dtype=np.float64)

    groups = domains = None
    kind = ds_cfg["split"]["kind"]
    if kind == "group_stratified":
        parsed = [parse_tid2013_name(n) for n in names]
        groups = [p[0] for p in parsed]
        if ds_cfg.get("domain") == "distortion_type":
            domains = [p[1] for p in parsed]
        else:
            raise SystemExit("group_stratified 目前只配 domain=distortion_type")

    sigma_col = ds_cfg.get("merged_sigma_column")
    if sigma_col:
        if sigma_col not in cols:
            raise SystemExit("并表缺 σ_human 列 %r：%s" % (sigma_col, path))
        sigma_human = np.asarray([float(r[sigma_col]) for r in rows], dtype=np.float64)
    else:
        sigma_human = np.full(len(names), np.nan, dtype=np.float64)

    if verbose:
        print("  [%s] 并表路径 n=%d  评委 %d 个  分组键 %s  分块键 %s"
              % (ds_name, len(names), len(judge_names),
                 "%d 类" % len(set(groups)) if groups else "无（真空条件）",
                 "%d 类" % len(set(domains)) if domains else "无（判据 4 记 na）"))
        print("        y ∈ [%.5f, %.5f]   σ_human: %s"
              % (y.min(), y.max(),
                 "取自 %r 列" % sigma_col if sigma_col
                 else "**不可用**（并表不含该列；判据 3 请走 labels_csv 路径）"))
    return Table(ds_name, names, judge_names, X, y, sigma_human, groups, domains)


def build_table(ds_name, ds_cfg, judges_cfg, verbose=True):
    """读一个数据集的全部评委缓存 + 标签，返回 `Table`。任一自检失败即抛。"""
    if ds_cfg.get("merged_csv"):
        return _table_from_merged(ds_name, ds_cfg, judges_cfg, verbose=verbose)
    cols, rows = _load_labels(ds_cfg)
    aux_rows = None
    if ds_cfg.get("aux_labels_csv"):
        _, aux_rows = _read_csv_rows(resolve(ds_cfg["aux_labels_csv"]))
        aux_rows = {r["image_name"]: r for r in aux_rows}

    # ── 评委分数：先读成 {basename: score}，再按标签侧的文件名集合对齐 ──
    preds = ds_cfg["preds"]
    per_judge = {}
    for j, p in preds.items():
        p = resolve(p)
        if not os.path.isfile(p):
            raise SystemExit("评委 %s 的缓存不存在：%s" % (j, p))
        per_judge[j] = load_pred(p)

    judge_names = sorted(preds.keys())
    # 9 份 key 集合必须逐对相同（阶段 C 的数据自检断言之一）
    base = set(per_judge[judge_names[0]])
    for j in judge_names[1:]:
        if set(per_judge[j]) != base:
            d1 = sorted(base - set(per_judge[j]))[:3]
            d2 = sorted(set(per_judge[j]) - base)[:3]
            raise SystemExit("评委 %s 的 key 集合与 %s 不同：多 %d 缺 %d（例 %s / %s）"
                             % (j, judge_names[0], len(d2), len(d1), d2, d1))

    # 标签侧的文件名，按有序列表固定下来（顺序即列序，保证与折无关、可复算）
    names = [os.path.basename(r["image_name"]) for r in rows]
    if len(set(names)) != len(names):
        raise SystemExit("标签侧出现重复文件名 ⇒ casefold 后仍非唯一，对齐不可靠")
    missing = [n for n in names if n not in base]
    if missing:
        raise SystemExit("%d 张标签图在评委缓存里没有（例 %s）" % (len(missing), missing[:3]))
    extra = sorted(base - set(names))
    if extra:
        raise SystemExit("评委缓存里有 %d 张不在本数据集标签内（例 %s）" % (len(extra), extra[:3]))

    X = np.asarray([[per_judge[j][n] for n in names] for j in judge_names], dtype=np.float64)

    # ── 人类锚点 y 与 σ_human ──
    ycfg = ds_cfg["y"]
    if ycfg.get("kind") != "linear":
        raise SystemExit("未知的 y.kind：%r" % ycfg.get("kind"))
    if ds_cfg.get("aux_labels_csv"):
        raw = np.asarray([float(aux_rows[n][ycfg["source_column"]]) for n in names])
    else:
        raw = np.asarray([float(r[ycfg["source_column"]]) for r in rows])
    y = (raw - float(ycfg.get("offset", 0.0))) / float(ycfg["scale"])

    scfg = ds_cfg["sigma_human"]
    if scfg["kind"] == "mos_std_positional":
        sigma_human = _sigma_human_tid(scfg, names)
    elif scfg["kind"] == "column":
        # ⚠️ 保留但不建议再用：直取某一列当 σ_human，**不保证它是「真值不确定度」**。
        # KonIQ 就栽在这里（`SD` 列是个体评分 SD，不是标准误）⇒ 新配置一律用 column_se。
        src = aux_rows if scfg.get("source") == "aux_labels_csv" else None
        if src is None:
            raise SystemExit("sigma_human.kind=column 目前只支持 source=aux_labels_csv")
        sigma_human = np.asarray([float(src[n][scfg["column"]]) for n in names]) / float(scfg["scale"])
    elif scfg["kind"] == "column_se":
        # ★ 2026-10-15 晚改定的 KonIQ 口径：σ_human = 个体评分 SD / √(个体评分数)。
        # 它才是「该图 MOS 估计量的不确定度」（标准误），与 TID2013 的 mos_std 同类。
        # 取 kind=column 直用 SD 会把「单个评分者的离散度」当参照量（错，见模块 docstring 第 3 条）。
        src = aux_rows if scfg.get("source") == "aux_labels_csv" else None
        if src is None:
            raise SystemExit("sigma_human.kind=column_se 目前只支持 source=aux_labels_csv")
        sd = np.asarray([float(src[n][scfg["sd_column"]]) for n in names])
        n_obs = np.asarray([float(src[n][scfg["n_column"]]) for n in names])
        if np.any(n_obs <= 0.0):
            raise SystemExit("sigma_human.kind=column_se 的 %s 列出现非正值" % scfg["n_column"])
        if np.any(sd < 0.0):
            raise SystemExit("sigma_human.kind=column_se 的 %s 列出现负值" % scfg["sd_column"])
        sigma_human = (sd / np.sqrt(n_obs)) / float(scfg["scale"])
    elif scfg["kind"] == "not_available":
        # ★ 2026-10-19：本数据集**没有**可用的人类真值不确定度口径（AGIQA-3K 的 SD 不是
        #   MOS 估计量的标准误、且无逐图 n）⇒ 全 NaN。**P1 诊断不得在此类数据集上跑**
        #   （NaN 会穿透判据 3，而不是伪装成一个合理的 ε）。本分支只服务 P2 输入适配。
        if not scfg.get("reason"):
            raise SystemExit("sigma_human.kind=not_available 必须给出 reason（口径不成立的理由）")
        sigma_human = np.full(len(names), np.nan, dtype=np.float64)
    else:
        raise SystemExit("未知的 sigma_human.kind：%r" % scfg["kind"])

    # ── 分组键与分块键 ──
    groups = domains = None
    kind = ds_cfg["split"]["kind"]
    if kind == "group_stratified":
        parsed = [parse_tid2013_name(n) for n in names]
        groups = [p[0] for p in parsed]
        if ds_cfg.get("domain") == "distortion_type":
            domains = [p[1] for p in parsed]
        else:
            raise SystemExit("group_stratified 目前只配 domain=distortion_type")

    if verbose:
        print("  [%s] n=%d  评委 %d 个  分组键 %s  分块键 %s"
              % (ds_name, len(names), len(judge_names),
                 "%d 类" % len(set(groups)) if groups else "无（真空条件）",
                 "%d 类" % len(set(domains)) if domains else "无（判据 4 记 na）"))
        if scfg["kind"] == "not_available":
            print("        y ∈ [%.5f, %.5f]   σ_human: 不适用（%s）"
                  % (y.min(), y.max(), scfg["reason"]))
        else:
            print("        y ∈ [%.5f, %.5f]   σ_human ∈ [%.5f, %.5f]  中位 %.6f"
                  % (y.min(), y.max(), sigma_human.min(), sigma_human.max(),
                     float(np.median(sigma_human))))
        for j in judge_names:
            r = X[judge_names.index(j)]
            print("        %-16s 原始分 ∈ [%.4f, %.4f]" % (j, r.min(), r.max()))

    return Table(ds_name, names, judge_names, X, y, sigma_human, groups, domains)


def build_all(cfg, verbose=True):
    out = {}
    for ds_name, ds_cfg in cfg["datasets"].items():
        out[ds_name] = build_table(ds_name, ds_cfg, cfg["judges"], verbose=verbose)
    return out
