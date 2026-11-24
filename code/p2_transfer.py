# -*- coding: utf-8 -*-
"""P2 跨域迁移矩阵：R（迁移劣化比）与 D（参数漂移）——可执行骨架（2026-10-19）。

状态：**骨架 ＋ 合成分数自测**。不产出任何真实结论，不需要 GPU，不读任何真实数据集。
本文件是 R 与 D 的**算法真源**，文档只给语义，具体实现以本文件为准
（与 P1 的 `comparability_criteria.py` 同一分工）。

═══ 定义 ═══════════════════════════════════════════════════════════════════

对拟合域 A、测试域 B、映射族 f、评委对 (i, j)：

  · **R（迁移劣化比）** = 在 A 上拟合的 θ 用到 B 上的误差 ÷ 在 B 上自拟合的误差。
    **分子与分母都是同一个测试域 B 上的 RMSE，相除即无量纲**；`y` 在进拟合前已按
    理论量程归一到 [0,1]，故 R 内**不再除以 sd**。

      R_dev = RMSE_B(θ_A) / RMSE_B(θ_B)
              RMSE 对**人类尺度 y** 算（两位评委各算一遍再平均）
      R_pair = RMSE_B^pair(θ_A) / RMSE_B^pair(θ_B)
              RMSE 对**两位评委校准后之差** d = f_i(x_i) − f_j(x_j) 算

    ⚠️ **2026-10-21 删掉了旧式里的 `sd` 因子**。旧式是
        [RMSE_B(θ_A)/sd(y_A)] / [RMSE_B(θ_B)/sd(y_B)] = 上式 × sd(y_B)/sd(y_A)，
    分子的误差算在 B 上却除以 A 的散布，两侧不是同一单位。删掉之后恒等映射
    精确满足 `R_dev(none) ≡ R_pair(none) ≡ 1`（旧式下它等于 sd(y_B)/sd(y_A)）。
    `sd_fit` / `sd_test` 仍落盘，但降级为**诊断量**（CSIQ 的位置差分析靠它）。

  · **去位置读数（同批新增，供方向性结论排伪影）**：
      bias      = mean(r)，r 为残差（dev 用 θ_A 在 B 上的预测减 y；pair 用两侧差）
      RMSE_c    = sqrt(max(mean(r²) − bias², 0))      # 残差的标准差，位置无关
      R_dev_c   = RMSE_c(num_dev) / RMSE_c(den_dev)，R_pair_c 同理
      ΔR_c      = R_pair_c − R_dev_c
    `R_pair` 本来就约掉了共同的位置偏移，`R_dev` 不约掉 ⇒ 比较 `ΔR` 与 `ΔR_c`
    的符号结构，就能判「方向不对称是不是 y 位置差的伪影」。

    `R_dev` 回答「映射能不能迁移」；`R_pair` 回答「校准后的分歧能不能迁移」。两者
    分开报，因为一个可以随尺度项一起漂，另一个只由分歧构成。

  · **D（参数漂移）** 两版并列：
      D_fun   = 在归一化输入网格上，两个映射**标准化输出**的 RMS 差
                f_A 与 f_B 各用自己的 y 的 μ/σ 标准化 ⇒ 纯 y 单位不同不会制造漂移
      D_param = 逐系数的 |θ_A − θ_B| / sqrt(SE_A² + SE_B²)（bootstrap 的标准误归一）
    **D_fun 是跨数据集的主读数**，对七个族都定义（含 quantile / isotonic）；
    D_param 只在参数族上有定义，且**只在两个数据集的 y 单位相同时可直读**——y 尺度
    差 100 倍时 μ_y 的差也会是 100 倍，那是单位差不是漂移。逐系数表照报，另给
    `sd_ratio = sd(y_A)/sd(y_B)` 供读者自己判断。
    ⚠️ **D_fun 有 O(1/√n) 的噪声底**：标准化用的是各域**自己的** y 的 μ/σ，两域是
    两批独立样本 ⇒ 即使两个映射完全相同（族 0），D_fun 也不恒为 0。实测 n=120 时
    ≈0.11，n=2000 时 ≈0.057。判读真实结果时必须与这个底比：几千条记录时底在 0.05
    量级，`0.05` 这种小值**不能**当漂移；要报 `D_fun` 的量级差异，不要报「是否 > 0」。

═══ 三条口径决策（写在这里，改之前先改文档）═══════════════════════════════════

1. **对角线 R ≡ 1**。A 与 B 是同一数据集时，分母的每折映射就是分子的映射（任何别的
   折的校准集都包含该测试折的图 ⇒ 用它算分子就是泄漏）。⇒ 对角线只报**域内基线
   RMSE**，不报 R。这是数学事实，不是实现取巧。
2. **y 先各自归一，再拟合**（`y_norm`）：
     `native`       直接用原值（只用于 0–1 口径都一致的对照）
     `theoretical`  y / 理论量程（LIVE ／ SPAQ / 100、AGIQA / 5、KADID `(5−DMOS)/4`…）
     `zscore`       **按块内** z 化（PIQ23 的 50 个场景各自一个尺度，必须逐场景）
   ⇒ R / D 用的是归一后的 y；**ε 仍用原单位**（PIQ23 用 JOD），两条路径不混。
3. **块（block）就是拟合单位**。PIQ23 传 `scene` 作块键 ⇒ 每个场景各自拟合、各自
   归一、各自算 R / D，最后跨场景汇总**中位 ＋ IQR**；跨场景汇总的不确定度用**场景
   重采样**（resample records，不重跑拟合）。不得把 50 个尺度混在一起拟一个 θ。

═══ 与 P1 的接口 ═══════════════════════════════════════════════════════════

映射族、拟合、五折协议全部沿用 P1：
  · `mapping_families.fit_all`（七族，族 0 需要 `judge_range`）
  · `stratified_split.make_folds`（MOS 五分位分层；块内则逐块调用）
  · 因子侧每折拟合后取**五折平均**作该侧的 bagged 映射；分母侧用**该折自己的**样本外映射。
    ⚠️ bagging 用的是 A 的全部折 ⇒ 比分母侧（80% 数据）训练得更多，R 因此**偏保守**
    （分子更容易小）——R > 1 的结论只会更稳，不会被这个偏向制造出来。

═══ 输入约定（真实分数到位后直接吃）══════════════════════════════════════════

一个数据集一个 CSV，至少含：`y` 列、每个评委一列；可选 `block`（场景）、`fold`。
`--domains spec.json` 的形状见 `load_domain` 的 docstring。合成自测用 `--synthetic`。

═══ 用法 ════════════════════════════════════════════════════════════════════

  python 30-experiments/metrics/p2_transfer.py --synthetic \
         --out C:/tmp/p2recon/p2run

  python 30-experiments/metrics/p2_transfer.py --domains spec.json \
         --out 30-experiments/runs/<run> --boot 200

═══ 本文件不做的事 ══════════════════════════════════════════════════════════

  · 不读真实数据集、不跑推理、不写 `runs/` 之外的目录；
  · 不报矩阵总量：**「每格每数据集一条」与「拟合数据集 × 测试数据集逐对展开」两种
    读法差一个量级**，口径未定 ⇒ 只给逐记录产物，不给总数；
  · 不碰 ε 三分类（那是 `comparability_criteria.py` 的路径，输入是原单位 y）；
  · 不判断「哪些格能对外报」——那要等真实数据的 `σ_human` 审计结果。
"""
import argparse
import csv
import itertools
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mapping_families as MF                                     # noqa: E402
import stratified_split as SS                                     # noqa: E402

CLASSES = ("合成", "真实", "AIGC")
ALL_FAMILIES = tuple(MF.MAPPING_NAMES)
NO_FREE_PARAMS = frozenset(("none",))
NONPARAMETRIC = frozenset(("quantile", "isotonic"))
PARAMETRIC = frozenset(("minmax", "zscore", "platt", "logistic5"))
N_GRID = 101


# ═════════════════════════════ 域与输入 ═════════════════════════════
class Domain(object):
    """一个数据集的「拟合视图」。

    `y`          已按 `y_norm` 归一的目标（R / D 用它；ε 另走原单位）
    `y_native`   原值，只作记录与报告，不参与任何计算
    `X`          (n, m) 评委分数；列序 = `judges`
    `ranges`     {评委: (理论下限, 理论上限)}；族 0 必需
    `blocks`     None 或长度 n 的块标签（PIQ23 传 scene）
    `folds`      None 或长度 n 的折号；块内数据集**每块各自 0..K−1**
    """

    __slots__ = ("name", "cls", "y", "y_native", "y_norm", "X", "judges", "ranges",
                 "blocks", "folds", "_judge_index", "_fold_cache")

    def __init__(self, name, cls, y, X, judges, ranges, y_native=None, y_norm="native",
                 blocks=None, folds=None):
        self.name = name
        self.cls = cls
        self.y = np.asarray(y, dtype=np.float64)
        self.X = np.asarray(X, dtype=np.float64)
        self.judges = tuple(judges)
        self.ranges = dict(ranges)
        self.y_native = None if y_native is None else np.asarray(y_native, dtype=np.float64)
        self.y_norm = y_norm
        self.blocks = None if blocks is None else np.asarray(blocks)
        self.folds = None if folds is None else np.asarray(folds, dtype=np.int64)
        self._judge_index = {j: k for k, j in enumerate(self.judges)}
        self._fold_cache = None

    @property
    def n(self):
        return int(self.y.size)

    def col(self, judge):
        return self.X[:, self._judge_index[judge]]

    def validate(self):
        n = self.n
        if self.X.ndim != 2 or self.X.shape[0] != n:
            raise SystemExit("[%s] X 形状 %r 与 y 长度 %d 不符" % (self.name, self.X.shape, n))
        if self.X.shape[1] != len(self.judges):
            raise SystemExit("[%s] X 列数 %d ≠ 评委数 %d"
                             % (self.name, self.X.shape[1], len(self.judges)))
        if not np.all(np.isfinite(self.X)):
            raise SystemExit("[%s] 评委分数含非有限值" % self.name)
        if not np.all(np.isfinite(self.y)):
            raise SystemExit("[%s] y 含非有限值" % self.name)
        if float(np.std(self.y)) <= 0:
            raise SystemExit("[%s] y 无变异，R / D 不可定义" % self.name)
        for j in self.judges:
            if j not in self.ranges:
                raise SystemExit("[%s] 评委 %s 缺理论量程" % (self.name, j))
            lo, hi = float(self.ranges[j][0]), float(self.ranges[j][1])
            if not hi > lo:
                raise SystemExit("[%s] 评委 %s 的量程非正：%r" % (self.name, j, self.ranges[j]))
        if self.blocks is not None and self.blocks.size != n:
            raise SystemExit("[%s] blocks 长度 %d ≠ n=%d" % (self.name, self.blocks.size, n))
        if self.folds is not None and self.folds.size != n:
            raise SystemExit("[%s] folds 长度 %d ≠ n=%d" % (self.name, self.folds.size, n))
        if self.cls not in CLASSES:
            raise SystemExit("[%s] 类别 %r 不在 %s" % (self.name, self.cls, CLASSES))
        return self


def normalize_y(y, blocks, how, scale=None):
    """按口径归一 y。返回 (y_fit, 说明字符串)。`how` 见模块头第 2 条。"""
    y = np.asarray(y, dtype=np.float64)
    if how == "native":
        return y.copy(), "native（原值）"
    if how == "theoretical":
        if scale is None or not (float(scale) > 0):
            raise SystemExit("y_norm=theoretical 需要正的 y_scale")
        return y / float(scale), "theoretical（÷ %g）" % float(scale)
    if how == "zscore":
        out = np.empty_like(y)
        if blocks is None:
            sd = float(np.std(y))
            if sd <= 0:
                raise SystemExit("y_norm=zscore：y 无变异")
            out[:] = (y - float(np.mean(y))) / sd
            return out, "zscore（全局）"
        for b in np.unique(blocks):
            idx = np.where(blocks == b)[0]
            sd = float(np.std(y[idx]))
            if sd <= 0:
                raise SystemExit("y_norm=zscore：块 %r 内 y 无变异" % (b,))
            out[idx] = (y[idx] - float(np.mean(y[idx]))) / sd
        return out, "zscore（逐块：%d 块）" % int(np.unique(blocks).size)
    raise SystemExit("未知 y_norm=%r（可选 native / theoretical / zscore）" % (how,))


def load_domain(spec):
    """从一个 JSON 规格读一个域。规格字段：

    name, class, csv, y_col, judge_cols[], ranges{评委:[lo,hi]},
    y_norm ("native"|"theoretical"|"zscore"), y_scale（theoretical 时必需）,
    block_col（可选）, fold_col（可选）

    ⚠️ `fold` 列若存在，**必须**是块内折号（PIQ23 的 `make_piq23_lists.py` 就是这么写的）；
    本函数不重排、不猜。
    """
    path = spec["csv"]
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("[%s] CSV 为空：%s" % (spec["name"], path))
    need = [spec["y_col"]] + list(spec["judge_cols"])
    if spec.get("block_col"):
        need.append(spec["block_col"])
    if spec.get("fold_col"):
        need.append(spec["fold_col"])
    miss = [c for c in need if c not in rows[0]]
    if miss:
        raise SystemExit("[%s] CSV 缺列 %s（实际列：%s）"
                         % (spec["name"], miss, list(rows[0])))

    def _f(v, what):
        try:
            return float(v)
        except (TypeError, ValueError):
            raise SystemExit("[%s] %s 列出现非数值：%r" % (spec["name"], what, v))

    y_native = np.array([_f(r[spec["y_col"]], spec["y_col"]) for r in rows], dtype=np.float64)
    X = np.array([[_f(r[j], j) for j in spec["judge_cols"]] for r in rows], dtype=np.float64)
    blocks = (np.array([str(r[spec["block_col"]]) for r in rows], dtype=object)
              if spec.get("block_col") else None)
    folds = (np.array([int(r[spec["fold_col"]]) for r in rows], dtype=np.int64)
             if spec.get("fold_col") else None)
    y_fit, _ = normalize_y(y_native, blocks, spec.get("y_norm", "native"),
                           spec.get("y_scale"))
    return Domain(spec["name"], spec["class"], y_fit, X, spec["judge_cols"], spec["ranges"],
                  y_native=y_native, y_norm=spec.get("y_norm", "native"),
                  blocks=blocks, folds=folds)


# ═════════════════════════════ 折 ═════════════════════════════
class _FoldHost(object):
    """`stratified_split.make_folds` 需要的最小接口（.y / .n / .groups）。"""

    def __init__(self, y, groups=None):
        self.y = np.asarray(y, dtype=np.float64)
        self.n = int(self.y.size)
        self.groups = groups


def ensure_folds(domain, n_splits, seed):
    """返回长度 n 的折号数组。块内数据集**逐块**生成，折号每块各自 0..K−1。

    已给 `domain.folds`（PIQ23 的名单文件）时只校验不重排：每块必须出现 0..K−1
    的完整集合，且每折两侧非空。
    """
    if domain._fold_cache is not None:
        return domain._fold_cache
    if domain.folds is not None:
        folds = domain.folds.copy()
        for b in block_list(domain):
            sel = np.arange(domain.n) if b is None else np.where(domain.blocks == b)[0]
            ids = np.unique(folds[sel])
            if ids.size < 2:
                raise SystemExit("[%s] 块 %r 只有 %d 个折，无法留出" % (domain.name, b, ids.size))
            if ids.min() != 0 or ids.max() != ids.size - 1:
                raise SystemExit("[%s] 块 %r 的折号不是 0..%d：%s"
                                 % (domain.name, b, ids.size - 1, ids.tolist()))
        domain._fold_cache = folds
        return folds

    folds = np.empty(domain.n, dtype=np.int64)
    for b in block_list(domain):
        sel = np.arange(domain.n) if b is None else np.where(domain.blocks == b)[0]
        k = min(int(n_splits), int(sel.size))
        if k < 2:
            raise SystemExit("[%s] 块 %r 只有 %d 张，无法切折" % (domain.name, b, sel.size))
        if k < int(n_splits):
            print("  ! [%s] 块 %r 只有 %d 张 ⇒ 折数降为 %d"
                  % (domain.name, b, sel.size, k))
        sub = SS.make_folds(_FoldHost(domain.y[sel]), k, seed)
        for kk, (_cal, test) in enumerate(sub):
            folds[sel[test]] = kk
    domain._fold_cache = folds
    return folds


def block_list(domain):
    if domain.blocks is None:
        return [None]
    return sorted(np.unique(domain.blocks), key=str)


def _split_by_folds(domain, block, folds):
    """返回 [(cal_idx, test_idx), ...]。块内数据集先按块过滤。"""
    base = (np.arange(domain.n) if block is None
            else np.where(domain.blocks == block)[0])
    ids = np.unique(folds[base])
    out = []
    for k in ids:
        cal = base[folds[base] != k]
        test = base[folds[base] == k]
        if cal.size == 0 or test.size == 0:
            raise SystemExit("[%s] 块 %r 第 %d 折一侧为空" % (domain.name, block, k))
        out.append((cal, test))
    return out


# ═════════════════════════════ 拟合与预测 ═════════════════════════════
def _fit(domain, idx, family, judge):
    try:
        lo, hi = domain.ranges[judge]
        m = MF.fit_all(family, domain.col(judge)[idx], domain.y[idx],
                       judge_range=(float(lo), float(hi)))
        return m, None
    except Exception as e:                                   # noqa: BLE001
        return None, "%s: %s" % (type(e).__name__, e)


def _bag_transform(maps, x):
    return np.mean([np.asarray(m.transform(x), dtype=np.float64) for m in maps], axis=0)


def _sd(a):
    s = float(np.std(np.asarray(a, dtype=np.float64)))
    return s if s > 0 else None


def _normalized_curve(m, grid, y_ref):
    """把映射在归一化网格上的输出按 y_ref 的 μ/σ 标准化（D_fun 的原子）。"""
    mu, sd = float(np.mean(y_ref)), _sd(y_ref)
    if sd is None:
        return None
    return (np.asarray(m.transform(grid), dtype=np.float64) - mu) / sd


# ═════════════════════════════ D ═════════════════════════════
def _boot_fit(domain, idx, family, judge, n_boot, rng):
    """在块内重采样拟合，返回 (参数列表, 标准化曲线列表, 失败数)。"""
    params, curves, failed = [], [], 0
    lo, hi = domain.ranges[judge]
    grid = lo + np.linspace(0.0, 1.0, N_GRID) * (hi - lo)
    for _ in range(int(n_boot)):
        pos = rng.integers(0, idx.size, idx.size)
        sub = idx[pos]
        m, err = _fit(domain, sub, family, judge)
        if err is not None:
            failed += 1
            continue
        params.append(dict(m.params))
        c = _normalized_curve(m, grid, domain.y[sub])
        curves.append(c)
    return params, curves, failed


def _param_block(pa, pb, sea, seb):
    """逐系数 |θ_A−θ_B| / sqrt(SE²+SE²)。返回 (D_param, 明细, 状态)。"""
    if not pa or not pb:
        return None, [], "no_free_params"
    keys = sorted(set(pa) & set(pb))
    if not keys:
        return None, [], "no_shared_params"
    rows, zs, n_inf = [], [], 0
    for k in keys:
        d = float(pa[k]) - float(pb[k])
        s = math.sqrt(float(sea.get(k, 0.0)) ** 2 + float(seb.get(k, 0.0)) ** 2)
        if s > 0:
            z = d / s
            zs.append(z)
        elif d == 0.0:
            z = 0.0
            zs.append(z)
        else:
            z = float("inf")
            n_inf += 1
        rows.append({"param": k, "theta_a": float(pa[k]), "theta_b": float(pb[k]),
                     "diff": d, "se_pooled": s, "z": z})
    dpar = float(np.sqrt(np.mean(np.square(zs)))) if zs else None
    return dpar, rows, ("ok" if n_inf == 0 else "ok_with_inf")


def compute_D(fit_dom, eval_dom, fit_block, eval_block, family, judges, cfg, rng):
    """一个拟合块 × 一个测试块上的 D（逐评委算，再合成评委对）。"""
    fit_idx = (np.arange(fit_dom.n) if fit_block is None
               else np.where(fit_dom.blocks == fit_block)[0])
    eval_idx = (np.arange(eval_dom.n) if eval_block is None
                else np.where(eval_dom.blocks == eval_block)[0])
    diagonal = (fit_dom is eval_dom and fit_block == eval_block)
    out = {"status": "ok", "per_judge": {}, "diagonal": diagonal}

    for j in judges:
        lo, hi = fit_dom.ranges[j]
        grid = lo + np.linspace(0.0, 1.0, N_GRID) * (hi - lo)
        ma, ea = _fit(fit_dom, fit_idx, family, j)
        mb, eb = _fit(eval_dom, eval_idx, family, j)
        if ea or eb:
            out["per_judge"][j] = {"status": "fit_failed",
                                   "error": ea or eb, "D_fun": None,
                                   "D_param_z": None, "params": None}
            out["status"] = "fit_failed"
            continue

        if diagonal:
            d_fun = 0.0
            d_fun_se = 0.0
        else:
            ca = _normalized_curve(ma, grid, fit_dom.y[fit_idx])
            cb = _normalized_curve(mb, grid, eval_dom.y[eval_idx])
            d_fun = float(np.sqrt(np.mean(np.square(ca - cb))))
            d_fun_se = None

        # ── D_param 的分派：族 0 / 非参族在有定义之前先钉状态，绝不落进 z 计算 ──
        if family in NO_FREE_PARAMS:
            dpar, rows, pstat = 0.0, [], "no_free_params"
        elif family in NONPARAMETRIC:
            dpar, rows, pstat = None, [], "nonparametric"
        else:
            # 没有 bootstrap 就不能把 |θ_A−θ_B| 标准化 ⇒ D_param 记 no_bootstrap，
            # **不报一个拿 0 当 SE 算出来的 z**（那会把「没算」伪装成「漂移极大」）。
            dpar, rows, pstat = None, [], "no_bootstrap"

        boot = {"n": 0, "failed": 0}
        if cfg.get("boot", 0) and not diagonal and family in PARAMETRIC:
            nb = int(cfg["boot"])
            pa, ca, f_a = _boot_fit(fit_dom, fit_idx, family, j, nb, rng)
            pb, cb, f_b = _boot_fit(eval_dom, eval_idx, family, j, nb, rng)
            boot = {"n": min(len(pa), len(pb)), "failed": f_a + f_b}
            if len(pa) >= 3 and len(pb) >= 3:
                # 中心用**全量拟合**的 θ（不是 bootstrap 均值），SE 用 bootstrap
                keys = sorted(set(ma.params) & set(mb.params))
                sea = {k: float(np.std([p[k] for p in pa])) for k in keys}
                seb = {k: float(np.std([p[k] for p in pb])) for k in keys}
                dpar, rows, pstat = _param_block(dict(ma.params), dict(mb.params), sea, seb)
                # D_fun 的 bootstrap：两侧重采样独立
                ds = []
                for k in range(min(len(ca), len(cb))):
                    if ca[k] is None or cb[k] is None:
                        continue
                    ds.append(float(np.sqrt(np.mean(np.square(ca[k] - cb[k])))))
                if len(ds) >= 3:
                    d_fun_se = float(np.std(ds))

        sd_a, sd_b = _sd(fit_dom.y[fit_idx]), _sd(eval_dom.y[eval_idx])
        out["per_judge"][j] = {
            "status": "ok", "D_fun": d_fun, "D_fun_se": d_fun_se,
            "D_param_z": dpar, "param_status": pstat,
            "sd_ratio_a_over_b": (sd_a / sd_b if sd_a and sd_b else None),
            "params": {"A": dict(ma.params), "B": dict(mb.params)} if family in PARAMETRIC else None,
            "param_rows": rows, "boot": boot,
        }

    ok = [v for v in out["per_judge"].values() if v["status"] == "ok"]
    if len(ok) == len(judges):
        dvals = [v["D_fun"] for v in ok if v["D_fun"] is not None]
        out["D_pair_fun"] = (float(np.sqrt(np.mean(np.square(dvals)))) if dvals else None)
        out["D_pair_fun_se"] = (float(np.sqrt(np.mean([v["D_fun_se"] ** 2 for v in ok
                                                       if v["D_fun_se"] is not None])))
                                if any(v["D_fun_se"] is not None for v in ok) else None)
        pv = [v["D_param_z"] for v in ok if v["D_param_z"] is not None]
        out["D_pair_param_z"] = (float(np.sqrt(np.mean(np.square(pv)))) if pv else None)
        out["D_param_status"] = sorted({v["param_status"] for v in ok})
    else:
        out["D_pair_fun"] = None
        out["D_pair_fun_se"] = None
        out["D_pair_param_z"] = None
        out["D_param_status"] = ["fit_failed"]
    return out


# ═════════════════════════════ R ═════════════════════════════
def compute_R(fit_dom, eval_dom, fit_block, eval_block, family, ji, jj, cfg,
              keep_errs=False):
    """一个拟合块 × 一个测试块上的 R（逐折算，再汇总池化值）。

    `keep_errs=True` 时额外返回 `errs`（逐折、逐图的平方误差分解），供
    `p2_boot_img.py` 做**图像级条件 bootstrap**。默认 `False`，返回体的其余
    字段与写盘内容**一个字节不变**（2026-10-20 加，为不改动旧产物）。
    """
    fit_folds = ensure_folds(fit_dom, cfg["n_splits"], cfg["seed"])
    eval_folds = ensure_folds(eval_dom, cfg["n_splits"], cfg["seed"])
    fit_folds = _split_by_folds(fit_dom, fit_block, fit_folds)
    eval_folds = _split_by_folds(eval_dom, eval_block, eval_folds)
    diagonal = (fit_dom is eval_dom and fit_block == eval_block)

    # ── 拟合侧：每折一份映射（bagging 用）──
    maps_a = {j: [] for j in (ji, jj)}
    for j in (ji, jj):
        for cal, _test in fit_folds:
            m, err = _fit(fit_dom, cal, family, j)
            if err is not None:
                return {"status": "fit_failed", "error": "[%s/%s] %s"
                        % (fit_dom.name, j, err)}
            maps_a[j].append(m)

    # ── 测试侧：每折一份映射（分母）──
    maps_b = {j: [] for j in (ji, jj)}
    for j in (ji, jj):
        for cal, _test in eval_folds:
            m, err = _fit(eval_dom, cal, family, j)
            if err is not None:
                return {"status": "fit_failed", "error": "[%s/%s] %s"
                        % (eval_dom.name, j, err)}
            maps_b[j].append(m)

    fit_base = (np.arange(fit_dom.n) if fit_block is None
                else np.where(fit_dom.blocks == fit_block)[0])
    eval_base = (np.arange(eval_dom.n) if eval_block is None
                 else np.where(eval_dom.blocks == eval_block)[0])
    sd_a, sd_b = _sd(fit_dom.y[fit_base]), _sd(eval_dom.y[eval_base])
    if sd_a is None or sd_b is None:
        return {"status": "zero_variance", "error": "拟合侧或测试侧 y 无变异"}

    per_fold, n_eval = [], 0
    errs = []
    # 累加平方和（num_*/den_*）与**和**（s*_，用于去位置的 bias）。
    # ⚠️ 计数分两个：`n` = 两位评委拼接后的样本数（dev 类用它，也是 `rmse_*` 字段
    #    口径，**不得改**——改了会让 rmse 列翻倍，破坏「只改了 R」的复现闸门）；
    #    `n_pair` = 评委间差的样本数（只有 bias/去位置要用它；比值里 2 倍会约掉）。
    acc = {"num_dev": 0.0, "den_dev": 0.0, "num_pair": 0.0, "den_pair": 0.0,
           "s_num_dev": 0.0, "s_den_dev": 0.0, "s_num_pair": 0.0, "s_den_pair": 0.0,
           "n": 0, "n_pair": 0}
    for k, (cal, test) in enumerate(eval_folds):
        xa = eval_dom.X[test]
        if diagonal:
            pred_a = {j: np.asarray(maps_b[j][k].transform(xa[:, eval_dom._judge_index[j]]),
                                    dtype=np.float64) for j in (ji, jj)}
        else:
            pred_a = {j: _bag_transform(maps_a[j], xa[:, eval_dom._judge_index[j]])
                      for j in (ji, jj)}
        pred_b = {j: np.asarray(maps_b[j][k].transform(xa[:, eval_dom._judge_index[j]]),
                                dtype=np.float64) for j in (ji, jj)}
        yt = eval_dom.y[test]
        n = int(yt.size)
        n_eval += n

        def _rms(a):
            return float(np.sqrt(np.mean(np.square(a))))

        num_dev = _rms(np.concatenate([pred_a[ji] - yt, pred_a[jj] - yt]))
        den_dev = _rms(np.concatenate([pred_b[ji] - yt, pred_b[jj] - yt]))
        num_pair = _rms(pred_a[ji] - pred_a[jj])
        den_pair = _rms(pred_b[ji] - pred_b[jj])
        acc["num_dev"] += float(np.sum(np.square(pred_a[ji] - yt))) + \
            float(np.sum(np.square(pred_a[jj] - yt)))
        acc["den_dev"] += float(np.sum(np.square(pred_b[ji] - yt))) + \
            float(np.sum(np.square(pred_b[jj] - yt)))
        acc["num_pair"] += float(np.sum(np.square(pred_a[ji] - pred_a[jj])))
        acc["den_pair"] += float(np.sum(np.square(pred_b[ji] - pred_b[jj])))
        acc["s_num_dev"] += float(np.sum(pred_a[ji] - yt)) + float(np.sum(pred_a[jj] - yt))
        acc["s_den_dev"] += float(np.sum(pred_b[ji] - yt)) + float(np.sum(pred_b[jj] - yt))
        acc["s_num_pair"] += float(np.sum(pred_a[ji] - pred_a[jj]))
        acc["s_den_pair"] += float(np.sum(pred_b[ji] - pred_b[jj]))
        acc["n"] += 2 * n
        acc["n_pair"] += n
        if keep_errs:
            # 逐图平方误差（两位评委相加，与 acc 的 num_* 同口径）
            # 以及**同口径的有符号**残差（`s_*`）：去位置 RMSE 要 Σr，
            # 只有 Σr² 无法在重采样里还原 mean(r)²。2026-10-22 加。
            errs.append({
                "fold": int(k), "n": n, "y": np.array(yt, dtype=np.float64),
                "num_dev": np.square(pred_a[ji] - yt) + np.square(pred_a[jj] - yt),
                "den_dev": np.square(pred_b[ji] - yt) + np.square(pred_b[jj] - yt),
                "num_pair": np.square(pred_a[ji] - pred_a[jj]),
                "den_pair": np.square(pred_b[ji] - pred_b[jj]),
                "s_num_dev": (pred_a[ji] - yt) + (pred_a[jj] - yt),
                "s_den_dev": (pred_b[ji] - yt) + (pred_b[jj] - yt),
                "s_num_pair": pred_a[ji] - pred_a[jj],
                "s_den_pair": pred_b[ji] - pred_b[jj],
            })
        per_fold.append({
            "fold": int(k), "n": n,
            # 2026-10-21：不再除以 sd（分子分母同域、同单位，比值已无量纲）
            "R_dev": (num_dev / den_dev) if den_dev > 0 else None,
            "R_pair": (num_pair / den_pair) if den_pair > 0 else None,
            "num_dev_rmse": num_dev, "den_dev_rmse": den_dev,
            "num_pair_rmse": num_pair, "den_pair_rmse": den_pair,
        })

    if acc["den_dev"] <= 0 or acc["den_pair"] <= 0:
        return {"status": "zero_denominator", "error": "分母侧 RMSE 为 0，R 不可定义"}

    rmse_num_dev = math.sqrt(acc["num_dev"] / acc["n"])
    rmse_den_dev = math.sqrt(acc["den_dev"] / acc["n"])
    rmse_num_pair = math.sqrt(acc["num_pair"] / acc["n"])
    rmse_den_pair = math.sqrt(acc["den_pair"] / acc["n"])

    def _rmse_c(sum2, s1, cnt):
        """去位置 RMSE = 残差的标准差：sqrt(max(mean(r²) − mean(r)², 0))。

        `sum2` 是 Σr²、`s1` 是 Σr、`cnt` 是 r 的个数（dev 类 2n、pair 类 n）。
        """
        if cnt <= 0:
            return None
        m = s1 / cnt
        var = sum2 / cnt - m * m
        return math.sqrt(var) if var > 1e-18 else 0.0

    bias_dev = acc["s_num_dev"] / acc["n"]
    bias_pair = acc["s_num_pair"] / acc["n_pair"]
    c_num_dev = _rmse_c(acc["num_dev"], acc["s_num_dev"], acc["n"])
    c_den_dev = _rmse_c(acc["den_dev"], acc["s_den_dev"], acc["n"])
    c_num_pair = _rmse_c(acc["num_pair"], acc["s_num_pair"], acc["n_pair"])
    c_den_pair = _rmse_c(acc["den_pair"], acc["s_den_pair"], acc["n_pair"])

    def _ratio(num, den):
        return (num / den) if (num is not None and den) else None

    pooled = {
        "R_dev": _ratio(rmse_num_dev, rmse_den_dev),
        "R_pair": _ratio(rmse_num_pair, rmse_den_pair),
        # 去位置读数：分母接近 0 的格（域内基线被拟合到极低）记 None，不报大数
        "R_dev_c": _ratio(c_num_dev, c_den_dev),
        "R_pair_c": _ratio(c_num_pair, c_den_pair),
        "bias_dev": bias_dev, "bias_pair": bias_pair,
        "num_dev_rmse": rmse_num_dev,
        "den_dev_rmse": rmse_den_dev,
        "num_pair_rmse": rmse_num_pair,
        "den_pair_rmse": rmse_den_pair,
    }
    devs = [f["R_dev"] for f in per_fold if f["R_dev"] is not None]
    pairs = [f["R_pair"] for f in per_fold if f["R_pair"] is not None]
    return {
        "status": "ok", "diagonal": diagonal, "n_eval": n_eval,
        "n_folds": len(eval_folds), "per_fold": per_fold,
        "R_dev_pooled": pooled["R_dev"], "R_pair_pooled": pooled["R_pair"],
        "R_dev_c_pooled": pooled["R_dev_c"],
        "R_pair_c_pooled": pooled["R_pair_c"],
        "bias_dev": pooled["bias_dev"], "bias_pair": pooled["bias_pair"],
        "R_dev_med": float(np.median(devs)) if devs else None,
        "R_dev_q25": float(np.percentile(devs, 25)) if devs else None,
        "R_dev_q75": float(np.percentile(devs, 75)) if devs else None,
        "R_pair_med": float(np.median(pairs)) if pairs else None,
        "R_pair_q25": float(np.percentile(pairs, 25)) if pairs else None,
        "R_pair_q75": float(np.percentile(pairs, 75)) if pairs else None,
        "rmse_num_dev": pooled["num_dev_rmse"], "rmse_den_dev": pooled["den_dev_rmse"],
        "rmse_num_pair": pooled["num_pair_rmse"], "rmse_den_pair": pooled["den_pair_rmse"],
        "sd_fit": sd_a, "sd_test": sd_b,
        "errs": errs if keep_errs else None,
    }


# ═════════════════════════════ 矩阵 ═════════════════════════════
def _block_pairs(a, b):
    """返回 [(fit_block, test_block), ...]。规则见模块头第 3 条。"""
    ab, bb = block_list(a), block_list(b)
    if a is b:
        return [(x, x) for x in ab]
    if a.blocks is not None and b.blocks is not None:
        common = sorted(set(ab) & set(bb), key=str)
        return [(x, x) for x in common]
    if a.blocks is not None:
        return [(x, None) for x in ab]
    if b.blocks is not None:
        return [(None, x) for x in bb]
    return [(None, None)]


def run_matrix(domains, cfg):
    """返回逐记录列表。只算 R / D，不做跨格汇总、不报总数。"""
    records = []
    seed = int(cfg["seed"])
    for a, b in itertools.product(domains, repeat=2):
        pairs = _block_pairs(a, b)
        if not pairs:
            for jp in itertools.combinations(a.judges, 2):
                for mp in cfg["families"]:
                    records.append({"fit_ds": a.name, "test_ds": b.name,
                                    "fit_class": a.cls, "test_class": b.cls,
                                    "fit_block": None, "test_block": None,
                                    "mapping": mp, "judge_i": jp[0], "judge_j": jp[1],
                                    "status": "no_matching_blocks"})
            continue
        for (fb, tb) in pairs:
            for k, (ji, jj) in enumerate(itertools.combinations(a.judges, 2)):
                for mp in cfg["families"]:
                    rng = np.random.default_rng(seed + 1000 * k + MF.MAPPING_NAMES.index(mp))
                    t0 = time.time()
                    r = compute_R(a, b, fb, tb, mp, ji, jj, cfg)
                    d = compute_D(a, b, fb, tb, mp, (ji, jj), cfg, rng)
                    rec = {
                        "fit_ds": a.name, "test_ds": b.name,
                        "fit_class": a.cls, "test_class": b.cls,
                        "fit_block": fb, "test_block": tb,
                        "mapping": mp, "judge_i": ji, "judge_j": jj,
                        "status": ("ok" if (r["status"] == "ok" and d["status"] == "ok")
                                   else "failed"),
                        "seconds": time.time() - t0,
                    }
                    for src in (r, d):
                        for kk, vv in src.items():
                            if kk not in rec:
                                rec[kk] = vv
                    if r["status"] != "ok":
                        rec["error"] = r.get("error")
                    if d["status"] != "ok":
                        rec["D_error"] = d.get("status")
                    records.append(rec)
    return records


def _med_iqr(vals):
    v = [float(x) for x in vals if x is not None]
    if not v:
        return None, None, None
    return (float(np.median(v)), float(np.percentile(v, 75) - np.percentile(v, 25)),
            len(v))


def _scene_boot_median(vals, rng, B=1000):
    """跨块（场景）重采样中位数的置信区间——只在 n_records ≥ 3 时有意义。"""
    v = np.asarray([x for x in vals if x is not None], dtype=np.float64)
    if v.size < 3:
        return None, None
    meds = np.empty(B, dtype=np.float64)
    for b in range(B):
        meds[b] = np.median(v[rng.integers(0, v.size, v.size)])
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def aggregate(records, seed):
    """按 (fit_ds, test_ds, mapping, judge 对) 汇总；块间给中位 ＋ IQR ＋ 场景 bootstrap CI。"""
    groups = {}
    for r in records:
        key = (r["fit_ds"], r["test_ds"], r["mapping"], r["judge_i"], r["judge_j"])
        groups.setdefault(key, []).append(r)

    # ── D_fun 的「同域对恒等基线」：同一 (fit_ds,test_ds,块对) 上 `none` 的 D_fun ──
    # 为什么要这一列（2026-10-19 冒烟实测）：`D_fun` 用**各自的** y μ/σ 标准化，
    # 两域 y 的均值/尺度差会变成一个**与映射无关的常数项**——真实 2×2 上
    # `none` 的 D_fun = 0.3148，而两域同分布时它才接近 0（合成 n=2000 ≈0.057）。
    # ⇒ 读「该族的映射漂移」必须减去这一项；`D_fun_med` 本身**不得**当漂移幅度。
    # `none` 的 D_fun 与评委对无关（映射恒等），按块对匹配即可。
    base = {}
    for r in records:
        if (r.get("mapping") == "none" and r.get("status") == "ok"
                and not r.get("diagonal") and r.get("D_pair_fun") is not None):
            base[(r["fit_ds"], r["test_ds"],
                  r.get("fit_block"), r.get("test_block"))] = float(r["D_pair_fun"])

    rows = []
    rng = np.random.default_rng(seed + 31337)
    for key in sorted(groups, key=lambda t: tuple(str(x) for x in t)):
        recs = groups[key]
        ok = [r for r in recs if r.get("status") == "ok"]
        dev = [r.get("R_dev_pooled") for r in ok]
        pair = [r.get("R_pair_pooled") for r in ok]
        dfun = [r.get("D_pair_fun") for r in ok]
        den = [r.get("rmse_den_dev") for r in ok]
        num = [r.get("rmse_num_dev") for r in ok]
        dev_c = [r.get("R_dev_c_pooled") for r in ok]
        pair_c = [r.get("R_pair_c_pooled") for r in ok]
        bias = [r.get("bias_dev") for r in ok]
        dm, di, dn = _med_iqr(dev)
        pm, pi, _ = _med_iqr(pair)
        fm, fi, _ = _med_iqr(dfun)
        dcm, dci, _ = _med_iqr(dev_c)
        pcm, pci, _ = _med_iqr(pair_c)
        bm_dev, _, _ = _med_iqr(bias)
        # ΔR_c 必须**先作差再取中位**（与正文的配对汇总口径一致）；
        # 取「中位之差」会得到另一个量。
        drc = [p - d for p, d in zip(pair_c, dev_c)
               if p is not None and d is not None]
        drm, dri, _ = _med_iqr(drc)
        bm, _, _ = _med_iqr(den)
        nm, _, _ = _med_iqr(num)
        lo, hi = _scene_boot_median(dev, rng)
        # 逐块扣除恒等基线后再汇总（减中位数 ≠ 中位数的差，故按记录逐条扣）
        exc = []
        for r in ok:
            if key[2] == "none" or r.get("diagonal"):
                continue
            b = base.get((r["fit_ds"], r["test_ds"],
                          r.get("fit_block"), r.get("test_block")))
            if b is not None and r.get("D_pair_fun") is not None:
                exc.append(float(r["D_pair_fun"]) - b)
        em, ei, en = _med_iqr(exc)
        rows.append({
            "fit_ds": key[0], "test_ds": key[1], "mapping": key[2],
            "judge_i": key[3], "judge_j": key[4],
            "fit_class": recs[0]["fit_class"], "test_class": recs[0]["test_class"],
            "n_records": len(recs), "n_ok": len(ok), "n_failed": len(recs) - len(ok),
            "n_blocks": len({(r.get("fit_block"), r.get("test_block")) for r in ok}),
            "diagonal": bool(recs[0].get("diagonal")),
            "R_dev_med": dm, "R_dev_iqr": di, "R_dev_n": dn,
            "R_dev_ci_lo": lo, "R_dev_ci_hi": hi,
            "R_pair_med": pm, "R_pair_iqr": pi,
            "R_dev_c_med": dcm, "R_dev_c_iqr": dci,
            "R_pair_c_med": pcm, "R_pair_c_iqr": pci,
            "dR_c_med": drm, "dR_c_iqr": dri,
            "bias_dev_med": bm_dev,
            "D_fun_med": fm, "D_fun_iqr": fi,
            "D_fun_excess_med": em, "D_fun_excess_iqr": ei, "D_fun_excess_n": en,
            "rmse_den_dev_med": bm, "rmse_num_dev_med": nm,
        })
    return rows


# ═════════════════════════════ 合成自测 ═════════════════════════════
def _monotone_curve(v, kind):
    """由潜在质量 v∈[0,1] 生成评委的「真实」单调响应。"""
    v = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0)
    if kind == "same":
        return v + 0.06 * np.sin(2.0 * np.pi * v)
    if kind == "power":
        return np.power(v, 1.6)
    raise SystemExit("未知合成曲线 %r" % (kind,))


def synthetic_domains(seed, real_curve="same", n=120, n_judges=2, n_scenes=0,
                      scene_n=0):
    """造三个普通域（合成 / 真实 / AIGC）＋（可选）一个场景域。

    评委的真实响应 = a_k·g_d(v) + b_k，**g_d 只随域变**；y 的**原值尺度**刻意不同
    （1 / 100 / 5），但 `y_norm` 统一成 theoretical ⇒ 归一后都是 v。场景域的 y 原值
    再乘一个**逐场景仿射**（模拟 PIQ23 的 50 个尺度），用 zscore 逐块归一。
    """
    rng = np.random.default_rng(seed)
    judges = ["J%d" % (k + 1) for k in range(n_judges)]
    ranges = {j: (0.0, 100.0) for j in judges}
    a = {j: 0.80 + 0.10 * k for k, j in enumerate(judges)}
    b = {j: -0.04 + 0.03 * k for k, j in enumerate(judges)}

    def _scores(v, g):
        X = np.empty((v.size, len(judges)), dtype=np.float64)
        for k, j in enumerate(judges):
            x = 100.0 * (a[j] * _monotone_curve(v, g) + b[j])
            x = x + rng.normal(0.0, 2.0, v.size)
            X[:, k] = np.clip(x, 0.0, 100.0)
        return X

    def _domain(name, cls, scale, g, n_rows):
        v = rng.beta(2.0, 2.0, n_rows)
        X = _scores(v, g)
        y_native = scale * v
        y_fit, _ = normalize_y(y_native, None, "theoretical", scale)
        return Domain(name, cls, y_fit, X, judges, ranges, y_native=y_native,
                      y_norm="theoretical")

    out = [
        _domain("syn", "合成", 1.0, "same", n),
        _domain("real", "真实", 100.0, real_curve, n),
        _domain("aigc", "AIGC", 5.0, "same", n),
    ]
    if n_scenes:
        v = rng.beta(2.0, 2.0, n_scenes * scene_n)
        scene = np.repeat(np.arange(n_scenes), scene_n)
        X = _scores(v, "same")
        alpha = rng.uniform(0.5, 1.5, n_scenes)[scene]
        beta = rng.normal(0.0, 1.0, n_scenes)[scene]
        y_native = 2.0 * (v - 0.5) * alpha + beta
        y_fit, _ = normalize_y(y_native, scene, "zscore")
        out.append(Domain("piq", "真实", y_fit, X, judges, ranges, y_native=y_native,
                          y_norm="zscore", blocks=scene))
    return out


def _fmt(x, spec="%.4f"):
    return "—" if x is None else (spec % x)


def _check(label, ok, detail, results):
    results.append({"test": label, "pass": bool(ok), "detail": detail})
    print("  [%s] %-46s %s" % ("PASS" if ok else "FAIL", label, detail))
    return ok


def run_self_test(out_dir, seed, no_assert=False):
    """合成自测：定义能算、能对方向响应、该挡的挡得住。返回 (results, records)。"""
    print("── 合成自测（种子 %d）──" % seed)
    results = []

    base_cfg = {"seed": seed, "n_splits": 3, "boot": 0, "families": list(ALL_FAMILIES)}

    # T1 正对照：两个域同一条曲线 ⇒ R ≈ 1。
    # ⚠️ 2026-10-21 订正：原注释写「跨原值尺度 1 / 100 / 5 仍应成立」，**这句话是空的**
    #    ——`synthetic_domains` 用 `y_norm: theoretical`（本文件 `synthetic_domains` 的
    #    `_domain`），三个域在进 R 之前已被除以各自的理论量程、归一成同一个 v ⇒
    #    `sd(y_fit) ≡ sd(y_test)`。所以 T1 从来没有覆盖过「尺度」这件事。真正的
    #    跨尺度对照是 T6。
    print("\nT1 正对照：两个域同一条真实曲线 ⇒ R_dev ≈ 1")
    doms = synthetic_domains(seed, real_curve="same")
    recs = run_matrix([doms[0], doms[1]], base_cfg)
    for mp in ALL_FAMILIES:
        v = [r.get("R_dev_pooled") for r in recs if r["mapping"] == mp
             and r.get("status") == "ok" and not r.get("diagonal")]
        _check("T1 %s：跨域 R_dev" % mp,
               bool(v) and all(0.85 <= x <= 1.15 for x in v),
               "R_dev = %s" % [_fmt(x) for x in v], results)

    # T2 负对照：真实域的曲线换成 v^1.6 ⇒ 有参数的族在**至少一个方向**上 R > 1。
    # ⚠️ 曾经断言「柔性族更稳」——**被实测否证**：isotonic 在 syn→real 上 R≈6.02，
    # 反而是 minmax 的 1.44 最小。柔性族能拟合源域的形状，不等于它**迁移**得更好；
    # 哪个族迁移得最好是要在真实矩阵上回答的经验问题，
    # 不能写成自测里的恒等式。
    print("\nT2 负对照：真实域曲线换成 v^1.6 ⇒ 有参数的族应出现迁移劣化")
    doms_neg = synthetic_domains(seed, real_curve="power")
    recs_neg = run_matrix([doms_neg[0], doms_neg[1]], base_cfg)

    def _r(mp, a, b):
        v = [r["R_dev_pooled"] for r in recs_neg
             if r["mapping"] == mp and r["fit_ds"] == a and r["test_ds"] == b
             and r.get("status") == "ok"]
        return float(np.median(v)) if v else None

    # ⚠️ **不假设「柔性族迁移得更好」**。这是**待检验的预期**，
    #    不是本骨架的正确性条件；合成 fixture 上实测反而是 isotonic 劣化得更多。
    #    骨架只断言「有参数的族全都响应劣化」——那是 R 定义正确的必要条件。
    fitted = [mp for mp in ALL_FAMILIES if mp != "none"]
    fwd = {mp: _r(mp, "syn", "real") for mp in fitted}
    bwd = {mp: _r(mp, "real", "syn") for mp in fitted}
    bad = [mp for mp in fitted if fwd[mp] is None or fwd[mp] <= 1.10]
    _check("T2 六个有参数的族在 syn→real 上全部劣化", not bad,
           ", ".join("%s=%.2f" % (mp, fwd[mp] or float("nan")) for mp in fitted), results)
    # 反向**不要求全部劣化**：R 是**有方向的**量，迁移劣化本来就不对称。实测 minmax
    # 在 real→syn 上回到 0.91——两个域的 x 观测区间几乎重合时，仿射映射搬过去损失很小。
    # 该断言只要求「六个族中每个至少在一个方向上响应」，这是 R 定义正确的必要条件。
    best = {mp: max(fwd[mp] if fwd[mp] is not None else -1,
                    bwd[mp] if bwd[mp] is not None else -1) for mp in fitted}
    bad2 = [mp for mp in fitted if best[mp] <= 1.10]
    _check("T2 每个有参数的族至少在一个方向上劣化", not bad2,
           ", ".join("%s=max(%.2f,%.2f)" % (mp, fwd[mp] or 0, bwd[mp] or 0)
                     for mp in fitted), results)
    r_none = _r("none", "syn", "real")
    _check("T2 族 0 无参数 ⇒ R 不响应（边界，不是缺陷）",
           r_none is not None and abs(r_none - 1.0) < 0.05,
           "R_none = %s" % _fmt(r_none), results)

    # T3 D：跨域漂移 > 0；同域 D ≈ 0；非参族 D_param 记 na、D_fun 仍可算
    print("\nT3 D：跨域漂移、对角归零、非参族的 D_param 状态")
    cfg_d = dict(base_cfg, boot=6, families=["minmax", "zscore", "quantile", "isotonic"])
    recs_d = run_matrix([doms[0], doms[1]], cfg_d)
    cross = [r["D_pair_fun"] for r in recs_d if r["mapping"] == "zscore"
             and r["fit_ds"] == "syn" and r["test_ds"] == "real" and r.get("status") == "ok"]
    diag = [r["D_pair_fun"] for r in recs_d if r["mapping"] == "zscore"
            and r.get("diagonal")]
    _check("T3 跨域 D_fun(zscore) > 0", bool(cross) and max(cross) > 0.10,
           "D_fun = %s" % [_fmt(x) for x in cross], results)
    _check("T3 对角 D_fun ≡ 0", bool(diag) and all(x == 0.0 for x in diag),
           "D_fun = %s" % [_fmt(x) for x in diag], results)
    st = {}
    for r in recs_d:
        st.setdefault(r["mapping"], set()).update(r.get("D_param_status") or [])
    _check("T3 quantile 的 D_param = nonparametric",
           "nonparametric" in st.get("quantile", set()), str(sorted(st.get("quantile", []))),
           results)
    _check("T3 none 的 D_param = no_free_params",
           True, "（none 未入本组；由 T5 单测）", results)

    # T4 场景内：逐场景记录数 = 场景数；逐场景拟合优于全局混合拟合
    print("\nT4 场景内：逐场景拟合 vs 全局混合拟合")
    doms_s = synthetic_domains(seed, real_curve="same", n=120, n_judges=2,
                               n_scenes=6, scene_n=40)
    piq = doms_s[3]
    cfg_s = dict(base_cfg, families=["zscore", "isotonic"])
    recs_s = run_matrix([doms_s[0], piq], cfg_s)
    n_blocks = len({r["test_block"] for r in recs_s
                    if r["test_ds"] == "piq" and r.get("status") == "ok"})
    _check("T4 piq 的逐场景记录数 = 6", n_blocks == 6, "n_blocks = %d" % n_blocks, results)

    def _scene_bias_range(fam, blocked):
        """跨场景的「场景级平均残差」展开。**用样本内拟合**：这里问的是
        「一个 θ 能不能同时服务 50 个尺度」，不是泛化误差 —— 后者会被「每场景样本少
        ⇒ 方差大」混淆（实测 isotonic 在 scene_n=40 时逐场景 CV 反而更差，那是方差，
        不是口径错）。"""
        j = piq.judges[0]
        out = []
        if blocked:
            for b in block_list(piq):
                idx = np.where(piq.blocks == b)[0]
                m, err = _fit(piq, idx, fam, j)
                if err:
                    return None
                out.append(float(np.mean(m.transform(piq.col(j)[idx]) - piq.y[idx])))
        else:
            m, err = _fit(piq, np.arange(piq.n), fam, j)
            if err:
                return None
            z = m.transform(piq.col(j)) - piq.y
            for b in block_list(piq):
                idx = np.where(piq.blocks == b)[0]
                out.append(float(np.mean(z[idx])))
        return max(out) - min(out)

    for fam in ("zscore", "isotonic"):
        rb, rg = _scene_bias_range(fam, True), _scene_bias_range(fam, False)
        _check("T4 %s：逐场景拟合消除场景级偏置" % fam,
               rb is not None and rg is not None and rb < rg,
               "跨场景偏置展开：逐场景 %.4f < 全局 %.4f" % (rb or 0.0, rg or 0.0), results)

    # T5 护栏：坏折号 / NaN / 缺量程 / `none` 的 D
    print("\nT5 护栏：坏输入必须拒绝，族 0 的 D 必须显式记 0")
    bad_fold = synthetic_domains(seed, n=60, n_judges=2)[0]
    bad_fold.folds = np.zeros(bad_fold.n, dtype=np.int64)
    try:
        ensure_folds(bad_fold, 3, seed)
        ok_fold = False
    except SystemExit:
        ok_fold = True
    _check("T5 单折输入被拒绝", ok_fold, "SystemExit", results)

    bad_nan = synthetic_domains(seed, n=60, n_judges=2)[0]
    bad_nan.X[0, 0] = np.nan
    try:
        bad_nan.validate()
        ok_nan = False
    except SystemExit:
        ok_nan = True
    _check("T5 NaN 分数被拒绝", ok_nan, "SystemExit", results)

    bad_rng = synthetic_domains(seed, n=60, n_judges=2)[0]
    bad_rng.ranges = {}
    try:
        bad_rng.validate()
        ok_rng = False
    except SystemExit:
        ok_rng = True
    _check("T5 缺理论量程被拒绝", ok_rng, "SystemExit", results)

    # D_fun 的标准化用的是**各域自己的 y 的 μ/σ**，而两个域的 y 是两批独立样本 ⇒
    # 即使映射完全相同（族 0 无参数），也会有 O(1/√n) 的抽样差：n=120 时实测 ≈0.11，
    # n=2000 时 ≈0.057。这个「底」不是缺陷，但**必须与真实漂移分开判**。
    # ⇒ 在同一对域上同时算族 0（纯噪声底）与 zscore（形状改变带来的真实漂移），
    #    断言「底 < 0.10」且「底 < 漂移的一半」。只断言「≈0」会把有限样本噪声
    #    误读成实现缺陷；只断言「> 0」又会把噪声当漂移。
    doms_none = synthetic_domains(seed + 11, real_curve="power", n=2000, n_judges=2)
    cfg_none = dict(base_cfg, families=["none", "zscore"], boot=0)
    recs_none = run_matrix([doms_none[0], doms_none[1]], cfg_none)
    dn = [r for r in recs_none if r["mapping"] == "none" and not r.get("diagonal")]
    dz = [r for r in recs_none if r["mapping"] == "zscore" and not r.get("diagonal")]
    dfuns = [r.get("D_pair_fun") for r in dn]
    dzuns = [r.get("D_pair_fun") for r in dz]
    floor = float(np.median(dfuns)) if dfuns and all(x is not None for x in dfuns) else None
    drift = float(np.median(dzuns)) if dzuns and all(x is not None for x in dzuns) else None
    _check("T5 none：D_param = no_free_params 且 D_fun 落在噪声底（n=2000）",
           bool(dn) and all("no_free_params" in (r.get("D_param_status") or []) for r in dn)
           and floor is not None and drift is not None
           and floor < 0.10 and floor < 0.5 * drift,
           "状态 = %s；噪声底 = %s；同域 zscore 漂移 = %s"
           % (sorted({tuple(r.get("D_param_status") or []) for r in dn}),
              _fmt(floor), _fmt(drift)), results)

    # T6 跨尺度正对照 —— 补 T1 的空覆盖（2026-10-21）。
    # T1 用的域全是 `y_norm: theoretical`：`synthetic_domains` 在 R 之前就把量程
    # 1 / 100 / 5 各除以自己的理论量程、归一成同一个 v ⇒ `sd(y_fit) ≡ sd(y_test)`
    # ⇒ 旧式里那个 sd 因子恒等于 1，**T1 对它既不能证成也不能证伪**。
    # 这里造两个 `y_norm: native` 的域：同一批 x、同一潜在质量 v，但
    #   y_A = v、y_B = 2·v  ⇒  sd(y_test)/sd(y_fit) = 2（位置也不同）。
    # 恒等映射不改分数 ⇒ 两侧给出同一组预测 ⇒ R 必须**精确**为 1。旧式会算成 2.0。
    print("\nT6 跨尺度正对照：y 原值尺度差 2 倍、恒等映射 ⇒ R ≡ 1（旧式会给 2）")
    rng6 = np.random.default_rng(seed + 60606)
    v6 = rng6.beta(2.0, 2.0, 240)
    judges6 = ["J1", "J2"]
    ranges6 = {j: (0.0, 100.0) for j in judges6}
    X6 = np.empty((v6.size, len(judges6)), dtype=np.float64)
    for k, j in enumerate(judges6):
        x = 100.0 * (0.9 * v6 + 0.05) + rng6.normal(0.0, 2.0, v6.size)
        X6[:, k] = np.clip(x, 0.0, 100.0)
    dom_a6 = Domain("scaleA", "合成", v6, X6, judges6, ranges6,
                    y_native=v6, y_norm="native")
    dom_b6 = Domain("scaleB", "合成", 2.0 * v6, X6, judges6, ranges6,
                    y_native=2.0 * v6, y_norm="native")
    recs6 = run_matrix([dom_a6, dom_b6], dict(base_cfg, families=["none"]))
    off6 = [r for r in recs6 if not r.get("diagonal") and r.get("status") == "ok"]
    sd_ratio6 = [float(r["sd_test"]) / float(r["sd_fit"]) for r in off6]
    # 矩阵跑两个方向 ⇒ 比值是 2（A→B）与 1/2（B→A），两种都必须出现
    ratio_ok6 = (bool(sd_ratio6)
                 and all(abs(x - 2.0) < 1e-6 or abs(x - 0.5) < 1e-6 for x in sd_ratio6)
                 and any(abs(x - 2.0) < 1e-6 for x in sd_ratio6)
                 and any(abs(x - 0.5) < 1e-6 for x in sd_ratio6))
    _check("T6 两个方向的 sd 之比 = 2 与 1/2（测试本身不空）", ratio_ok6,
           "sd_test/sd_fit = %s" % [_fmt(x) for x in sd_ratio6], results)
    _check("T6 跨尺度恒等映射 R_dev ≡ R_pair ≡ 1（旧式会给 2）",
           bool(off6)
           and all(abs(float(r["R_dev_pooled"]) - 1.0) < 1e-12 for r in off6)
           and all(abs(float(r["R_pair_pooled"]) - 1.0) < 1e-12 for r in off6),
           "R_dev = %s；R_pair = %s"
           % ([_fmt(r["R_dev_pooled"]) for r in off6],
              [_fmt(r["R_pair_pooled"]) for r in off6]), results)

    return results, recs


# ═════════════════════════════ 产物 ═════════════════════════════
REC_COLS = [
    "fit_ds", "fit_class", "test_ds", "test_class", "fit_block", "test_block",
    "mapping", "judge_i", "judge_j", "status", "diagonal", "n_eval", "n_folds",
    "R_dev_pooled", "R_dev_med", "R_dev_q25", "R_dev_q75",
    "R_pair_pooled", "R_pair_med", "R_pair_q25", "R_pair_q75",
    # 2026-10-21 新增：去位置读数与两个 bias（y 单位）。sd_fit/sd_test 降级为诊断量。
    "R_dev_c_pooled", "R_pair_c_pooled", "bias_dev", "bias_pair",
    "rmse_num_dev", "rmse_den_dev", "rmse_num_pair", "rmse_den_pair",
    "sd_fit", "sd_test",
    # ⚠️ 键名必须与 `compute_D` 返回的 dict 一致（`D_pair_fun` 等），
    # 不是「D_fun_pair」。2026-10-19 冒烟实测：写错过一次 ⇒ 逐记录 CSV 的
    # D 三列全空，而汇总表正常（汇总直接读 dict）⇒ 只看汇总不会发现。
    "D_pair_fun", "D_pair_fun_se", "D_pair_param_z", "D_param_status",
]

AGG_COLS = [
    "fit_ds", "fit_class", "test_ds", "test_class", "mapping", "judge_i", "judge_j",
    "n_records", "n_ok", "n_failed", "n_blocks", "diagonal",
    "R_dev_med", "R_dev_iqr", "R_dev_n", "R_dev_ci_lo", "R_dev_ci_hi",
    "R_pair_med", "R_pair_iqr",
    "R_dev_c_med", "R_dev_c_iqr", "R_pair_c_med", "R_pair_c_iqr",
    "dR_c_med", "dR_c_iqr", "bias_dev_med",
    "D_fun_med", "D_fun_iqr",
    "D_fun_excess_med", "D_fun_excess_iqr", "D_fun_excess_n",
    "rmse_den_dev_med", "rmse_num_dev_med",
]


def _cell(v):
    if isinstance(v, (list, tuple)):
        return "|".join(str(x) for x in v)
    if isinstance(v, float):
        return "%.10g" % v
    return "" if v is None else str(v)


def _guard_cols(name, cols, rows):
    """列名 ↔ 记录键 的守卫：某列在所有行里都取不到 ⇒ 名字写错了。

    ⚠️ 2026-10-19 冒烟实测踩过：`REC_COLS` 写 `D_fun_pair`、记录键是 `D_pair_fun`，
    逐记录 CSV 的 D 三列**静默全空**，而汇总表正常（汇总直接读 dict）——
    只核汇总发现不了，必须在写盘前拦住。
    """
    if not rows:
        return
    seen = set()
    for r in rows:
        seen |= set(r)
    ghost = [c for c in cols if c not in seen]
    if ghost:
        raise SystemExit("%s 有 %d 列在行里不存在（键名写错？）：%s"
                         % (name, len(ghost), ghost))


def write_products(out_dir, records, agg, meta, selftest=None):
    _guard_cols("REC_COLS", REC_COLS, records)
    _guard_cols("AGG_COLS", AGG_COLS, agg)
    os.makedirs(out_dir, exist_ok=True)
    rec_path = os.path.join(out_dir, "p2_matrix_records.csv")
    with open(rec_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(REC_COLS)
        for r in records:
            w.writerow([_cell(r.get(c)) for c in REC_COLS])

    agg_path = os.path.join(out_dir, "p2_matrix.csv")
    with open(agg_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(AGG_COLS)
        for r in agg:
            w.writerow([_cell(r.get(c)) for c in AGG_COLS])

    payload = {"meta": meta,
               "n_records": len(records), "n_aggregate_rows": len(agg),
               "status_counts": {},
               "aggregate": agg}
    for r in records:
        s = r.get("status")
        payload["status_counts"][s] = payload["status_counts"].get(s, 0) + 1
    if selftest is not None:
        payload["selftest"] = selftest
    with open(os.path.join(out_dir, "p2_matrix_summary.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = ["# P2 跨域迁移矩阵 —— 骨架产物（%s）" % time.strftime("%Y-%m-%d %H:%M:%S"),
             "",
             "> **本文件是派生物**：真源是 `p2_matrix_records.csv` 与 `p2_matrix_summary.json`。",
             "> 本次运行**不报矩阵总量**（两种记录枚举口径差一个量级，口径未定）。",
             "",
             "## 口径", ""]
    for k, v in meta.items():
        lines.append("- **%s**：%s" % (k, v))
    lines.append("")
    lines.append(
        "> ⚠️ **`D_fun` 的读法（2026-10-19 冒烟实测）**：`D_fun` 用各域**自己的** "
        "y μ/σ 标准化，所以两域 y 的均值/尺度差会变成一个与映射无关的常数项。"
        "真实 2×2 上 `none`（恒等映射）的 `D_fun` = **0.3148**，而合成同分布时只有 "
        "≈0.057 ⇒ **`D_fun` 本身不是漂移幅度**。漂移读 `D_fun_excess_*`"
        "（逐块减去同一域对 `none` 的 `D_fun`），它才是拟合映射相对恒等映射多出来的形状差。")
    lines.append("")
    lines.append("## 状态计数")
    lines.append("")
    for k in sorted(payload["status_counts"], key=str):
        lines.append("- `%s`：%d" % (k, payload["status_counts"][k]))
    if selftest is not None:
        lines += ["", "## 合成自测", ""]
        for t in selftest:
            lines.append("- [%s] %s —— %s" % ("PASS" if t["pass"] else "FAIL",
                                             t["test"], t["detail"]))
    lines += ["", "## 产物", "",
              "- `p2_matrix_records.csv`：逐记录（拟合块 × 测试块）。",
              "- `p2_matrix.csv`：按数据集对 × 映射 × 评委对汇总（中位 ＋ IQR ＋ 场景 bootstrap CI）。",
              "- `p2_matrix_summary.json`：同数据 ＋ 口径 ＋ 状态计数。",
              "- `p2_matrix_report.md`：本文件。", ""]
    with open(os.path.join(out_dir, "p2_matrix_report.md"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    return rec_path, agg_path


def _sha256(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="用合成分数跑自测")
    ap.add_argument("--domains", default=None, help="真实数据的 JSON 规格")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=0, help="D 的 bootstrap 次数（真实运行建议 200）")
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--families", default=",".join(ALL_FAMILIES))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--no-assert", action="store_true", help="自测失败仍写产物（仅供排查）")
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    cfg = {"seed": args.seed, "n_splits": args.n_splits, "boot": args.boot,
           "families": [s for s in args.families.split(",") if s]}
    unknown = [m for m in cfg["families"] if m not in MF.MAPPING_NAMES]
    if unknown:
        raise SystemExit("未知映射族：%s（可选 %s）" % (unknown, list(MF.MAPPING_NAMES)))

    meta = {
        "定义": "R = RMSE_B(θ_A) / RMSE_B(θ_B)（同域同单位，比值无量纲；2026-10-21 起不再除以 sd）；"
                "R_dev_c / R_pair_c 为去位置读数；D 两版（D_fun 主、D_param 辅）",
        "对角线": "A 与 B 是同一数据集时 R ≡ 1，只报域内基线 RMSE",
        "y 归一": "theoretical / zscore（逐块）；ε 另走原单位，不混",
        "块": "PIQ23 传 scene ⇒ 逐场景拟合、归一、算 R/D，跨场景汇总中位＋IQR",
        "记录枚举": "逐记录输出，**不报总量**（口径未定）",
        "种子": args.seed,
        "折数": args.n_splits,
        "bootstrap": args.boot,
        "映射族": ",".join(cfg["families"]),
    }

    selftest = None
    if args.synthetic:
        tests, _recs = run_self_test(out_dir, args.seed, no_assert=args.no_assert)
        selftest = tests
        n_fail = sum(1 for t in tests if not t["pass"])
        print("\n自测：%d/%d 通过" % (len(tests) - n_fail, len(tests)))
        # 产物用的矩阵比自测再小一档：自测已经覆盖各分支，这里只求「产物能生成」。
        doms = synthetic_domains(args.seed, real_curve="power", n=100, n_judges=2,
                                 n_scenes=4, scene_n=30)
        records = run_matrix(doms, cfg)
        meta["模式"] = "synthetic（合成分数，非任何数据集）"
    elif args.domains:
        spec = json.load(open(args.domains, encoding="utf-8"))
        doms = [load_domain(s).validate() for s in spec["datasets"]]
        ref = set(doms[0].judges)
        for d in doms[1:]:
            if set(d.judges) != ref:
                raise SystemExit("[%s] 评委集合与 %s 不一致：%r vs %r"
                                 % (d.name, doms[0].name, d.judges, doms[0].judges))
        records = run_matrix(doms, cfg)
        meta["模式"] = "domains（%s）" % os.path.basename(args.domains)
    else:
        raise SystemExit("必须给 --synthetic 或 --domains")

    agg = aggregate(records, args.seed)
    meta["耗时秒"] = round(time.time() - t0, 2)
    rec_path, agg_path = write_products(out_dir, records, agg, meta, selftest)
    print("\n产物：%s" % out_dir)
    print("  %s（sha256 %s…）" % (os.path.basename(rec_path), _sha256(rec_path)[:16]))
    print("  %s（sha256 %s…）" % (os.path.basename(agg_path), _sha256(agg_path)[:16]))
    print("  状态计数：%s" % json.dumps(
        {k: sum(1 for r in records if r.get("status") == k)
         for k in sorted({r.get("status") for r in records})}, ensure_ascii=False))

    if selftest is not None and not args.no_assert:
        bad = [t for t in selftest if not t["pass"]]
        if bad:
            print("\n[ABORT] 自测未全过（%d 项）——产物已写出，供排查" % len(bad))
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
