# -*- coding: utf-8 -*-
"""P1 第④步「可比性诊断」：四条判据。

**本模块是四条判据的算法真源**：预注册的语义与阈值来源见同仓 `configs/` 下的配置，
具体算法以本文件为准。

每条判据返回统一形状：
    {"verdict": "pass" | "fail" | "na", "value": float | None,
     "threshold_ref": str, "detail": {...}}

★ 判据 3 是整套里最关键的一条，用**水平条件残差分解**：
    令 d(i) = f_A(q_A(i)) − f_B(q_B(i))   （校准后差）
      δ̂ = mean(d)                      位置项，对应 D1 的 δ_k
      β̂ = slope(d ~ y)                 尺度项，对应 δ_k(c) 的连续形态
      σ̂ = SD(d − δ̂ − β̂·y)             残差项，对应 D1 的 ξ_ik
  **判定只取样本外测试集**（校准集上 δ̂ 被拟合强行压到 ≈0，会产生「校准集全过、测试集
  全挂」的形态 —— 这个对照本身是要报的方法学发现，故两个集都算、都报）。
  σ̂ 的参照量 `σ_human` = **人类 MOS 估计量的不确定度（该图 MOS 的标准误）**
  〔2026-10-15 晚改定：原写「人类评分者自身的离散度」，那是**另一种量**——它在 KonIQ 上被
  直取为个体评分 `SD`、在 TID2013 上是 MOS 估计的 RMSE，两侧混用会让判据得出相反结论；
  见 `pred_table.py` 模块 docstring 第 3 条〕。
  尺度换算：δ̂ 是**两个评委之差**的统计量，其真值不确定度是单个人类真值的 √2 倍
  ⇒ 预注册阈值 `σ̂ > √2 · median(σ_human)`（`c3_equiv_factor` 为其倍率，默认 1.0）。
"""
import numpy as np

SQRT2 = float(np.sqrt(2.0))


# ──────────────────────────── 判据 1 ────────────────────────────
def criterion1(folds_a, folds_b, z_a, z_b, thr):
    """边际分布有充分重叠。

    1a 外推比例：测试集的原始分落在**该折的**校准集支撑 `[min_cal, max_cal]` 之外的比例。
        ⚠️ 必须**逐折**算 —— 若把所有折的校准集并起来，支撑会退化成全体取值范围，
        外推比例恒 ≈ 0，这一条就白算了。取「跨折 × 两评委」的最大值。
    1b 重叠系数：`OVL = Σ min(p_A, p_B)`，在**校准后的人类尺度**上做 20 等宽 bin 直方图。

    参数：`folds_a` / `folds_b` 是 `[(x_cal, x_test), ...]`；`z_a` / `z_b` 是拼接后的
    样本外**校准输出**（已在人类尺度上）。
    """
    def _extrap(folds):
        vals, spans = [], []
        for xc, xt in folds:
            lo, hi = float(np.min(xc)), float(np.max(xc))
            vals.append(float(np.mean((xt < lo) | (xt > hi))))
            spans.append([lo, hi])
        return max(vals), vals, spans

    ea, ea_all, span_a = _extrap(folds_a)
    eb, eb_all, span_b = _extrap(folds_b)
    e = max(ea, eb)

    n_bins = int(thr["c1_n_bins"])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    pa, _ = np.histogram(np.clip(z_a, 0.0, 1.0), bins=edges)
    pb, _ = np.histogram(np.clip(z_b, 0.0, 1.0), bins=edges)
    pa = pa / max(pa.sum(), 1)
    pb = pb / max(pb.sum(), 1)
    ovl = float(np.minimum(pa, pb).sum())
    out_of_unit = float(np.mean((z_a < 0) | (z_a > 1)) + np.mean((z_b < 0) | (z_b > 1))) / 2.0

    return {
        "c1a": {
            "verdict": "fail" if e > thr["c1_extrapolation_max"] else "pass",
            "value": e,
            "threshold_ref": "c1_extrapolation_max = %.2f" % thr["c1_extrapolation_max"],
            "detail": {"extrap_a_per_fold": ea_all, "extrap_b_per_fold": eb_all,
                       "cal_span_a_per_fold": span_a, "cal_span_b_per_fold": span_b},
        },
        "c1b": {
            "verdict": "fail" if ovl < thr["c1_overlap_min"] else "pass",
            "value": ovl,
            "threshold_ref": "c1_overlap_min = %.2f" % thr["c1_overlap_min"],
            "detail": {"n_bins": n_bins, "frac_outside_unit_interval": out_of_unit},
        },
    }


# ──────────────────────────── 判据 2 ────────────────────────────
def criterion2(maps_per_fold_a, maps_per_fold_b, range_a, range_b, thr):
    """映射单调且交叉验证稳定。

    2a 单调性：**每个评委各自的观测分范围**上建 1001 点等距网格，验 `diff ≥ −tol`，各折都要过。
    2b CV 稳定性：5 折重拟合后在**同一原则（每侧自己的观测分范围）**上逐点求 SD，取
       `p95(SD)`；输出已在人类尺度上（该尺度跨度已归一到 1），故阈值可直接比较。

    ⚠️ **两侧必须各用自己的网格**（2026-10-30 修）。九评委量程分三档（`[0,1]`／`[1,5]`／
    `[0,100]`），早先版本只取 A 的范围造一条网格、两边共用 ⇒ **B 的映射被拿到自己的定义域外
    评估**，2b 的跨折 SD 被外推发散撑大，跨量纲对因此整批假失败（实测 TID2013 9 个、
    KonIQ 13 个，全部是跨量纲对；同量纲对 0 个）。本函数的 docstring 一直写「该评委」，
    实现却只收一组边界——是**代码与注释不一致**，不是口径调整。
    另报参数方差 —— 即「不同校准集下的参数方差」这一层读数。
    """
    tol = float(thr["c2_monotone_tol"])

    def _side(maps, rng_):
        grid = np.linspace(float(rng_[0]), float(rng_[1]), int(thr["c2_grid_n"]))
        worst = 1.0
        all_ok = True
        for m in maps:
            ok, mind = m.monotone_ok(grid, tol=tol)
            all_ok = all_ok and ok
            worst = min(worst, mind)
        Z = np.vstack([np.asarray(m.transform(grid), dtype=np.float64).ravel() for m in maps])
        sd_pt = np.std(Z, axis=0)
        return all_ok, worst, float(np.percentile(sd_pt, 95)), Z, grid

    ok_a, min_a, stab_a, Z_a, grid_a = _side(maps_per_fold_a, range_a)
    ok_b, min_b, stab_b, Z_b, grid_b = _side(maps_per_fold_b, range_b)
    stab = max(stab_a, stab_b)

    keys = sorted({k for m in maps_per_fold_a + maps_per_fold_b for k in m.params})
    pvar = {}
    for k in keys:
        for tag, maps in (("A", maps_per_fold_a), ("B", maps_per_fold_b)):
            v = np.array([m.params.get(k, np.nan) for m in maps], dtype=np.float64)
            if np.all(np.isfinite(v)) and v.size and np.mean(v) != 0:
                pvar["%s.%s" % (tag, k)] = float(np.std(v) / abs(np.mean(v)))

    return {
        "c2a": {
            "verdict": "pass" if (ok_a and ok_b) else "fail",
            "value": min(min_a, min_b),
            "threshold_ref": "diff >= -c2_monotone_tol (%.0e)，各折都要过" % tol,
            "detail": {"min_diff_a": min_a, "min_diff_b": min_b,
                       "grid_n": grid_a.size,
                       "grid_span_a": [float(grid_a[0]), float(grid_a[-1])],
                       "grid_span_b": [float(grid_b[0]), float(grid_b[-1])]},
        },
        "c2b": {
            "verdict": "fail" if stab > thr["c2_stability_max"] else "pass",
            "value": stab,
            "threshold_ref": "c2_stability_max = %.2f" % thr["c2_stability_max"],
            "detail": {"p95_pointwise_sd_a": stab_a, "p95_pointwise_sd_b": stab_b,
                       "param_cv": pvar},
        },
    }


# ──────────────────────────── 判据 3 ────────────────────────────
def _decompose(d, y):
    """水平条件残差分解：返回 (δ̂, β̂, σ̂)。OLS 斜率用总体矩（ddof=0）。"""
    d = np.asarray(d, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    delta = float(np.mean(d))
    vy = float(np.var(y))
    beta = float(np.mean((d - delta) * (y - np.mean(y))) / vy) if vy > 0 else 0.0
    sigma = float(np.std(d - delta - beta * y))
    return delta, beta, sigma


def _block_indices(blocks):
    """按块分块重采样所需的「块内下标」列表。blocks=None ⇒ 逐图独立。"""
    if blocks is None:
        return None
    uniq = {}
    for i, b in enumerate(blocks):
        uniq.setdefault(b, []).append(i)
    return [np.asarray(v, dtype=np.int64) for v in uniq.values()]


def _boot_decompose(d, y, blocks, B, rng):
    idx_blocks = _block_indices(blocks)
    n = d.size
    out = np.empty((B, 3), dtype=np.float64)
    for b in range(B):
        if idx_blocks is None:
            idx = rng.integers(0, n, n)
        else:
            pick = rng.integers(0, len(idx_blocks), len(idx_blocks))
            idx = np.concatenate([idx_blocks[p] for p in pick])
        out[b] = _decompose(d[idx], y[idx])
    return out


def criterion3(d_cal, y_cal, d_test, y_test, blk_test, sigma_ref, thr, seed,
               spread_ratio_a=None, spread_ratio_b=None):
    """校准后两评委分数差与人类 MOS 差的一致性（★ 主判据，只看样本外）。

    参数：
      `d_cal` / `y_cal`  = **某一折校准集**上的校准后差与人类尺度（样本内对照用）
      `d_test` / `y_test`= **全部样本外**拼接后的校准后差与人类尺度（判定用；长度 = n）
      `blk_test`         = 与 `d_test` **同长**的分块标签（TID2013 用参考图编号；KonIQ 传 None）
      `sigma_ref`        = **人类 MOS 估计量的不确定度**（该图 MOS 的标准误；口径见
                           `pred_table.py` docstring 第 3 条），**取全数据集**的中位数 ——
                           它是人类真值的属性，与折无关，故不随折变化
                           （避免「拼接校准集」带来的重复计数）
      `spread_ratio_*`   = `SD(f_k(测试集)) / SD(y)`。**退化护栏**，见下。

    ★ **退化护栏（阈值预注册为 `c3_min_spread_ratio`）** —— 判据规格本身的漏洞：
      若某个映射族退化成**近乎常函数**（输出 spread → 0），则两个评委的输出都趋同，
      `δ̂ → 0`、`β̂ → 0`、`σ̂ → 0`，**判据 3 三分量会同时「通过」** —— 而它实际上什么
      都没测。这类映射**既非过也非挂，而是不可判**，故三分量一律记 **`na`**（与判据 4 在
      KonIQ 上的 `na` 同一语义：结构上测不了，不是测出来好）。实测触发例见 run 目录的产物。
    """
    B = int(thr["c3_boot_B"])
    alpha = float(thr["c3_alpha"])
    lo_q, hi_q = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)

    rng = np.random.default_rng(seed)
    boot = _boot_decompose(d_test, y_test, blk_test, B, rng)
    ci = np.percentile(boot, [lo_q, hi_q], axis=0)

    delta, beta, sigma = _decompose(d_test, y_test)
    d_in, b_in, s_in = _decompose(d_cal, y_cal)        # 样本内对照（校准集）

    # σ_human 的参照量：人类真值的**不确定度**，再换算成「两个评委之差」的不确定度（×√2）
    # ⚠️ sigma_ref 是**人类 MOS 估计量的标准误**，不是「个体评分者之间的离散度」——
    #    两者的差别在 KonIQ 上实测为 10.30 倍，混用会让残差项判定整批翻转。
    ref = SQRT2 * float(np.median(sigma_ref))
    ratio = sigma / ref if ref > 0 else float("inf")

    d_excl = ci[0, 0] > 0.0 or ci[1, 0] < 0.0
    b_excl = ci[0, 1] > 0.0 or ci[1, 1] < 0.0
    s_excl = ci[0, 2] > 0.0 or ci[1, 2] < 0.0

    # ── 判定式：等价性检验（不是零假设显著性检验）────────────────────
    # ⚠️ 旧判定式「95% CI 不含 0 ⇒ fail」已由**阳性对照证伪**（见模块头的阳性对照说明）：
    #   阳性对照 B := 3.7·A − 0.42 是**精确仿射**、三分量真值恒为 0，实测三分量数值全在
    #   1e-17 ~ 1e-7 的浮点级，**却仍被 7 个映射中的 4 个报 fail** —— 因为真值为 0 时
    #   bootstrap CI 约有 **50%** 的概率落在 0 的一侧。⇒ 旧判法下「两个**完全相同**的评委
    #   会以约一半概率被判不可互换」，且判定**随样本量漂移**（n 越大越容易 fail）。
    # 现改为与同一把尺 ε 比：ε = c3_equiv_factor × √2 × median(σ_human)，
    # 语义 = 「两个评委之间的分歧幅度是否**小于人类真值自身的不确定度**」。
    # ⚠️ 语义在 2026-10-15 晚随 σ_human 口径一并改定：原写「人类评分者之间的天然分歧」，
    #    那是**个体评分者**的离散度（另一种量）；改定后参照量是**真值的不确定度**。
    # CI 仍照算照报，但**降为辅助证据**（写进 detail）—— 它回答的是「δ̂ 被数据定位到多准」，
    # 与「δ̂ 是否大到不可忽略」是两个不同的问题。
    eps = float(thr.get("c3_equiv_factor", 1.0)) * ref

    # 尺度项量纲不同，**必须先把斜率换算回幅度**再比：β̂ 是「Δd/Δy」，量纲为一；
    # ε 是「人类尺度上的一个幅度」。乘上 y 的 5–95 分位跨度即得「y 的主要变动区间内，
    # 两评委之差漂移了多少」——这才与 δ̂、σ̂ 同量纲。
    p_lo, p_hi = thr.get("c3_beta_prob_span", [5.0, 95.0])
    y_span = float(np.percentile(y_test, p_hi) - np.percentile(y_test, p_lo))
    beta_amp = abs(beta) * y_span

    d_fail = abs(delta) > eps
    b_fail = beta_amp > eps
    # ⚠️ 残差项**不作门槛**（原写法 `s_fail = sigma > eps` 已删）。
    #    理由：口径统一之后，该二值判定在
    #    **这两批评委集上是退化的** —— 两数据集均 252/252 一边倒（TID2013 全挂、KonIQ 全挂），
    #    `pass/fail` 这个比特**不携带区分信息**。若仍把它计入门槛，叠加「四条判据全过」的
    #    旧总口径，**「可比」将恒为空集**，与三分类口径冲突。
    #    ⇒ σ̂ 只以**倍数 σ̂/ε** 作**连续读数**，落在「可互换区间」里（`interchangeability`）；
    #    σ̂ / ε / 倍数三者在此处与 `c3_sigma.detail` 里照算照报，可复现性不断。

    ci_aux = {"ci_excludes_zero": [bool(d_excl), bool(b_excl), bool(s_excl)],
              "note": "CI 仅作辅助证据：它给的是定位精度，不是判定依据"}

    min_ratio = thr.get("c3_min_spread_ratio")
    degenerated = (min_ratio is not None
                   and spread_ratio_a is not None and spread_ratio_b is not None
                   and min(spread_ratio_a, spread_ratio_b) < float(min_ratio))
    dg_detail = {"degenerate": True, "min_spread_ratio": min(spread_ratio_a, spread_ratio_b),
                 "threshold": min_ratio}
    if degenerated:
        return {k: {"verdict": "na", "value": v, "ci_lo": None, "ci_hi": None,
                    "threshold_ref": "退化护栏：min(SD(f_k)/SD(y)) >= %.3f" % min_ratio,
                    "detail": dict(dg_detail, reason="degenerate_mapping", ratio=ratio,
                                   eps=eps, median_sigma_human=float(np.median(sigma_ref)),
                                   sqrt2_median_sigma_human=ref, y_span=y_span,
                                   in_sample={"delta": d_in, "beta": b_in, "sigma": s_in},
                                   B=B)}
                for k, v in (("c3_delta", delta), ("c3_beta", beta), ("c3_sigma", sigma))}

    return {
        "c3_delta": {
            "verdict": "fail" if d_fail else "pass",
            "value": delta, "ci_lo": float(ci[0, 0]), "ci_hi": float(ci[1, 0]),
            "threshold_ref": "等价性检验：|δ̂| <= ε（ε = %.1f × √2 × median(σ_human)）" %
                             float(thr.get("c3_equiv_factor", 1.0)),
            "detail": {"eps": eps, "abs_delta": abs(delta), "in_sample_delta": d_in,
                       "median_sigma_human": float(np.median(sigma_ref)),
                       "sqrt2_median_sigma_human": ref, "ratio_abs_delta_over_eps":
                       (abs(delta) / eps if eps > 0 else float("inf")),
                       "B": B, **ci_aux},
        },
        "c3_beta": {
            "verdict": "fail" if b_fail else "pass",
            "value": beta, "ci_lo": float(ci[0, 1]), "ci_hi": float(ci[1, 1]),
            "threshold_ref": "等价性检验：|β̂| · (y_p%g − y_p%g) <= ε（β̂ 须先换算回幅度）"
                             % (p_hi, p_lo),
            "detail": {"eps": eps, "beta_amp": beta_amp, "y_span": y_span,
                       "ratio_amp_over_eps": (beta_amp / eps if eps > 0 else float("inf")),
                       "in_sample_beta": b_in, "B": B, **ci_aux},
        },
        "c3_sigma": {
            # ⚠️ verdict 是 **`info`**，**不是** `pass`/`fail`/`na` 三者之一（2026-10-16 改定）：
            #    `na` 在本库语义已固定为「**结构上测不了**」（见本函数退化护栏与判据 4 在
            #    KonIQ 上的 `na`），而 σ̂ 在这里是**测得出、只是不作门槛** ⇒ 必须用第三个记号，
            #    不得借 `na` 把「不作门槛」混成「测不了」。
            "verdict": "info",
            "value": sigma, "ci_lo": float(ci[0, 2]), "ci_hi": float(ci[1, 2]),
            "threshold_ref": "分级读数：σ̂/ε（**不作门槛**；2026-10-16 改定）",
            "detail": {"eps": eps, "ratio": ratio, "sqrt2_median_sigma_human": ref,
                       "median_sigma_human": float(np.median(sigma_ref)),
                       "in_sample_sigma": s_in, "B": B, **ci_aux},
        },
    }


# ──────────────────────────── 判据 4 ────────────────────────────
def criterion4(d, y, beta_global, domains, thr, seed, name="c4", groups=None, eps=None):
    """跨失真类型无显著参数漂移（水平已先扣掉，否则块间差会被分数区间混淆）。

    δ̂(c) 定义为**块内 `d − β̂_global·y` 的均值**（而不是 `d` 的均值）：同一失真类型下
    5 个强度级跨越不同 MOS，若不扣掉水平项，`δ̂(c)` 的块间差主要反映的是**分数区间差**，
    而不是失真类型差 —— 那样这一条就退化成判据 3 的重述。

    置换检验：统计量 = 跨 c 的 `|δ̂(c) − δ̂_global|` 的**最大值**，用置换的 max 分布控多重比较。

    ★ **2026-11-09：判决改为幅度规则 `M > ε`。**
      `eps` 非空时，`verdict` 由 `obs > eps` 决定，置换 `p` **仍算、仍落盘，但不进判决**
      （`detail.p_value` 保留，另给 `gate_rule` 与 `amp_over_eps`）。三条理由：
      ① B = 20,000 的 p 分辨率下限是 `1/20001 = 5.00e-5`，TID2013 上 252/252 全打在这个
      下限上（`min = max`），任何 α 都分不开一条判定与另一条；
      ② 它在**数值零**上开火 —— 合成正向对照 `B := 3.7A − 0.42` 在 platt／logistic5 上
      以 `|δ̂|/ε = 4.51e-7` 与 `2.39e-9` 被判 fail，拒绝一个自己的仿射副本，量到的是
      置换次数不是漂移；
      ③ 改成幅度后判据 4 与判据 1–3 落在**同一把尺**上（`drift_domains` 的口径本来
      就写着「与判据 3 同尺」），三分类不再把等价问题交给显著性检验。
      **代价（务必连同读）**：`p` 的 max-null 原本控 24 个域的多重比较，幅度规则没有
      对应物，只剩「M 本身就是这 24 个域的最大值」这一层，**family-wise 保证不再成立**。

    `eps=None` 走**旧语义**（按 `p < c4_alpha` 判），专供只审 p 的三个脚本
    （`c4_pvalue_audit` / `c4_block_perm` / `c4_identify`）原样调用，不因此改变口径。

    `groups`（**默认 None ＝ 原路径逐位不变**）：
      · `None`：自由置换 —— 24 类标签在整个样本上重排，**不看内容分块**。
      · 非空：**分块置换** —— 在每个块（TID2013 的参考图）内重排标签。块内多重集守恒
        ⇒ 各域计数与自由置换一样不变；不同的是**同内容的图不再被打散到各域**，
        内容方差从域均值里扣掉 ⇒ 空分布**更窄**（方向已用只读试跑实测，不是推测）。
        ⚠️ 因此这不是「更保守」的空分布，而是**与分块设计匹配**的那一个。
    """
    d = np.asarray(d, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    v = d - float(beta_global) * y
    dom = np.asarray(domains)
    cats, codes = np.unique(dom, return_inverse=True)
    k = cats.size
    B = int(thr["c4_perm_B"])

    def _stat(cod):
        sums = np.bincount(cod, weights=v, minlength=k)
        cnts = np.bincount(cod, minlength=k).astype(np.float64)
        cnts[cnts == 0] = np.nan
        means = sums / cnts
        g = float(np.mean(v))
        return float(np.nanmax(np.abs(means - g))), means

    obs, means = _stat(codes)
    rng = np.random.default_rng(seed)
    null = np.empty(B, dtype=np.float64)
    if groups is None:
        for b in range(B):
            null[b] = _stat(codes[rng.permutation(codes.size)])[0]
    else:
        # ★ 分块置换：每块内重排标签，块内多重集守恒。
        #   填充位用 +inf 键排到行尾 ⇒ 每行前 b_i 列恰是该块标签的一个均匀随机排列。
        g_arr = np.asarray(groups)
        if g_arr.size != d.size:
            raise ValueError("groups 长度 %d ≠ d 长度 %d" % (g_arr.size, d.size))
        blk = {}
        for i, g in enumerate(g_arr):
            blk.setdefault(g, []).append(i)
        blocks = [np.asarray(v_, dtype=np.int64) for v_ in blk.values()]
        K = len(blocks)
        S = max(b.size for b in blocks)
        pad = np.zeros((K, S), dtype=np.int64)
        msk = np.zeros((K, S), dtype=bool)
        for i, b in enumerate(blocks):
            pad[i, :b.size] = b
            msk[i, :b.size] = True
        good = pad[msk]
        code_pad = np.zeros((K, S), dtype=np.int64)
        code_pad[msk] = codes[good]
        scratch = np.empty(codes.size, dtype=np.int64)
        for b in range(B):
            keys = rng.random((K, S))
            keys[~msk] = np.inf
            scratch[good] = np.take_along_axis(code_pad, np.argsort(keys, axis=1),
                                               axis=1)[msk]
            null[b] = _stat(scratch)[0]
    p = float((1 + np.sum(null >= obs)) / (B + 1))

    # ★ 幅度规则 vs 旧 p 规则（见 docstring）。`eps=None` 走旧语义。
    if eps is None:
        verdict, rule = ("fail" if p < thr["c4_alpha"] else "pass"), "permutation"
        thresh_ref = ("置换检验 p < %.2f（B = %d，统计量取跨域 max，控多重比较）"
                      % (thr["c4_alpha"], B))
    else:
        eps_f = float(eps)
        verdict, rule = ("fail" if obs > eps_f else "pass"), "amplitude"
        thresh_ref = ("幅度规则 M > ε（ε = %.6g）；置换 p = %.6g 仅作诊断、不进判决（B = %d）"
                      % (eps_f, p, B))

    return {
        name: {
            "verdict": verdict,
            "value": obs, "ci_lo": None, "ci_hi": None,
            "threshold_ref": thresh_ref,
            "detail": {"p_value": p, "n_domains": int(k),
                       "delta_global": float(np.mean(v)),
                       "per_domain_delta": {str(c): float(m) for c, m in zip(cats, means)},
                       "drift_span": float(np.nanmax(means) - np.nanmin(means)),
                       # 判决口径与「幅度／ε」倍数，供正文直接引用。
                       "gate_rule": rule,
                       "amp_over_eps": (float(obs) / float(eps)) if eps else None,
                       # 空分布上界与比值 —— 「narrow null」这一问
                       # 只有把空分布自己的尺度印出来才答得了。只增字段，不改既有语义。
                       "perm_scheme": "free" if groups is None else "block",
                       "null_max": float(np.max(null)),
                       "null_max_over_obs": float(np.max(null) / obs) if obs > 0 else None},
        }
    }


# ══════════════ 三分类聚合与「可互换区间」（2026-10-16 新增）══════════════════════
# 这两块兑现两处早已写出、实验侧一直没有的承诺：
#   · 「按四条判据逐一检验，输出 可比 / 不可比 / 条件可比 三种判定」
#   · 「输出『可互换区间』」
# 规则**全部跑前预注册**在同仓 `configs/20261016-p1diagnose-v2.yaml` 的
# `aggregate` / `iso` 两段；改口径只改配置，不改本文件。
# ⚠️ 本节不改变任何已登记的实测结论（判据 1–4 的计算一字未动），只把它们接上定义。

#: 参与三分类的**门槛键**。⚠️ **不含 `c3_sigma`** —— 残差项的二值判定在两批评委集上
#: 退化（见 `criterion3` 内注释），故它只作分级读数、不作门槛。
#: config 的 `aggregate.gate_keys` 可覆盖本默认值。
GATE_KEYS = ("c1a", "c1b", "c2a", "c2b", "c3_delta", "c3_beta")

LABEL_COMPARABLE = "可比"
LABEL_CONDITIONAL = "条件可比"
LABEL_INCOMPARABLE = "不可比"


def _dom_key(s):
    """域标签排序：数字域按数值序（TID2013 的失真类型是 1…24，字符串序会把 10 排在 2 前）。"""
    try:
        return (0, float(s), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(s))


def drift_domains(c4_detail, eps):
    """按**跑前预注册的判据**找出发生漂移的域：`|δ̂(c) − δ̂_global| > ε`。

    ⚠️ 这是一个**自设口径**，`criterion4` 只给置换检验的 `p` 与逐域 `δ̂(c)`，没有
    「哪些域漂移」的现成定义。**选「与判据 3 同尺」的理由**：漂移域与「位置失配」在
    同一容忍度 ε 下读，口径唯一；否则会被质疑「漂移域是事后从 24 个域里挑出来的」。
    ⇒ 返回 `(漂移域列表, δ̂_global, 最大漂移幅度)`。
    """
    if not c4_detail or eps is None or not (eps > 0):
        return [], None, None
    per = c4_detail.get("per_domain_delta") or {}
    g = c4_detail.get("delta_global")
    if g is None or not per:
        return [], None, None
    g = float(g)
    dev = {str(c): abs(float(v) - g) for c, v in per.items()}
    doms = sorted((c for c, dv in dev.items() if dv > float(eps)), key=_dom_key)
    return doms, g, (max(dev.values()) if dev else None)


def aggregate_verdict(criteria, c4_detail=None, eps=None, cfg=None, c4_name="c4"):
    """三分类聚合：`可比` / `条件可比` / `不可比`。规则 = 配置 `aggregate` 段预注册。

    ★ **两条「不可判」必须分开**（本层最容易混的一处）：
      · 门槛键为 `na` ⇒ **不可比** ＋ `undecidable=True`。`na` 只出现在**映射塌成常函数**的
        退化护栏（判据 3）里，那是「结构上测不了」—— **不得因「没报 fail」就判可比**。
      · 只有 `c4 == na` ⇒ **条件可比**（`c4_na_no_domain_labels`）。那是有判据、缺标签。
    两者语义不同、处置不同，实现上不得共用分支。
    """
    cfg = cfg or {}
    gate_keys = tuple(cfg.get("gate_keys") or GATE_KEYS)
    c4_na_policy = cfg.get("c4_na_policy", "conditional")

    gate = {k: ((criteria.get(k) or {}).get("verdict")) for k in gate_keys}
    failed = sorted(k for k, v in gate.items() if v == "fail")
    na_keys = sorted(k for k, v in gate.items() if v == "na")
    missing = sorted(k for k, v in gate.items() if v is None)
    c4v = (criteria.get(c4_name) or {}).get("verdict")
    doms, g, amp = drift_domains(c4_detail, eps)
    has_drift = (c4v == "fail") or bool(doms)

    def _mk(label, reason, undecidable=False):
        return {"label": label, "reason": reason, "gate": gate, "c4": c4v,
                "c4_na_policy": c4_na_policy, "failed": failed, "na": na_keys + missing,
                "drift_domains": doms, "drift_amplitude": amp, "delta_global": g,
                "undecidable": bool(undecidable)}

    if na_keys or missing:
        return _mk(LABEL_INCOMPARABLE,
                   "gate_na_degenerate_mapping" if na_keys else "gate_missing", True)
    if failed:
        return _mk(LABEL_INCOMPARABLE, "gate_failed:" + ",".join(failed))
    if c4v is None or c4v == "na":
        if c4_na_policy == "conditional":
            return _mk(LABEL_CONDITIONAL, "c4_na_no_domain_labels")
        return _mk(LABEL_INCOMPARABLE, "c4_na_no_domain_labels", True)
    if has_drift:
        if c4v == "fail" and doms:
            rs = "c4_drift"
        elif c4v == "fail":
            rs = "c4_drift_p_only"      # 置换检验拒了，但没有域超过 ε ⇒ 只报条件化
        else:
            rs = "c4_drift_eps_only"    # 置换检验过了，但有域超过 ε ⇒ 保守地条件化
        return _mk(LABEL_CONDITIONAL, rs)
    return _mk(LABEL_COMPARABLE, "all_gate_pass_c4_no_drift")


def interchangeability(z_a, z_b, y, eps, domains=None, cfg=None, degenerate=False,
                       degenerate_reason=None):
    """「可互换区间」：在容忍 ε 下报出**哪些分数区间 / 失真类型**上两个评委可互换。

    两版**并列报**（2026-10-16 定）：
      · **主口径**（口径原文的字面写法）：`P_main = mean[ |z_a − z_b| < ε ]`
      · **严格版**：`P_strict = mean[ |z_a − y| < ε 且 |z_b − y| < ε ]`
        —— 「**且与人类判断一致**」那半句由严格版单独兑现（主口径只约束 A、B 互差）。
    `z_a`/`z_b` 是**各自的**校准后输出、`y` 是人类尺度，三者等长、均为**样本外**测试集。

    ★ **退化映射必须挡住**（本函数最高危的一处）：映射塌成常函数时 `z_a ≈ z_b` ⇒ `d ≡ 0`
      ⇒ `|d| < ε` 恒真 ⇒ **`P` 假性 = 1**，会报「100% 可互换」。`degenerate` 必须与
      `criterion3` 的退化护栏**同源**（用 `c3_delta.verdict == "na"`），否则两处口径不一致。
      退化时**直接记 `na`**，不报任何比例。

    ★ 区间以**箱的集合**为主陈列，**不假称连续区间**；箱数按实际分位边界取，**不写死 10**。
    ★ **`ladder` 必报**：ε 已钉在「人类不确定度」水平，单点 `P` 大概率很低、达标箱可能为空，
      那不是实现失败；不报阶梯会被读成「实现没产出结果」。
    """
    cfg = cfg or {}
    n_bins = int(cfg.get("n_bins", 10))
    min_bin_n = int(cfg.get("min_bin_n", 30))
    target_P = float(cfg.get("target_P", 0.90))
    ladder = tuple(float(v) for v in (cfg.get("eps_ladder") or (1.0, 2.0, 5.0)))
    by_domain = bool(cfg.get("by_domain", True))

    za = np.asarray(z_a, dtype=np.float64)
    zb = np.asarray(z_b, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    out = {"n": int(yy.size), "eps": (None if eps is None else float(eps)),
           "target_P": target_P, "min_bin_n": min_bin_n, "n_bins_requested": n_bins,
           "degenerate": bool(degenerate), "reason": degenerate_reason,
           "value": None, "p_strict": None, "ladder": None, "bins_mos": None,
           "bins_domain": None, "n_bins_effective": None,
           "n_ok_bins_mos": None, "n_ok_bins_domain": None, "n_ok_bins_strict_mos": None}

    if degenerate:
        out["verdict"] = "na"
        out["reason"] = degenerate_reason or "degenerate_mapping"
        return out
    if eps is None or not (float(eps) > 0) or yy.size == 0:
        out["verdict"] = "na"
        out["reason"] = out["reason"] or "no_eps"
        return out
    if not (za.size == zb.size == yy.size):
        raise ValueError("interchangeability: z_a / z_b / y 长度必须相同")

    eps = float(eps)
    d = za - zb
    ad = np.abs(d)
    pe_a, pe_b = np.abs(za - yy), np.abs(zb - yy)

    out["value"] = float(np.mean(ad < eps))
    out["p_strict"] = float(np.mean((pe_a < eps) & (pe_b < eps)))
    out["ladder"] = {"%gx" % f: float(np.mean(ad < eps * f)) for f in ladder}

    def _pack(mask):
        nb = int(mask.sum())
        if nb < 1:
            return None
        pk = float(np.mean(ad[mask] < eps))
        ps = float(np.mean((pe_a[mask] < eps) & (pe_b[mask] < eps)))
        insuff = nb < min_bin_n
        return {"n": nb, "P": pk, "P_strict": ps,
                "ok": (None if insuff else bool(pk >= target_P)),
                "ok_strict": (None if insuff else bool(ps >= target_P)),
                "insufficient": insuff,
                "mean_d": float(np.mean(d[mask])), "abs_mean_d": float(abs(np.mean(d[mask]))),
                "sd_d_over_eps": float(np.std(d[mask])) / eps}

    # ── MOS 十分位箱（实际边界去重后可能少于 10 箱：TID2013 的 MOS 是离散的）──
    edges = np.unique(np.percentile(yy, np.linspace(0.0, 100.0, n_bins + 1)))
    bins_mos = []
    if edges.size >= 3:
        idx = np.digitize(yy, edges[1:-1])
        out["n_bins_effective"] = int(edges.size - 1)
        for b in range(edges.size - 1):
            rec = _pack(idx == b)
            if rec is None:
                continue
            rec["lo"], rec["hi"] = float(edges[b]), float(edges[b + 1])
            bins_mos.append(rec)
    out["bins_mos"] = bins_mos or None
    okd = [r["ok"] for r in bins_mos if r["ok"] is not None]
    out["n_ok_bins_mos"] = sum(1 for v in okd if v)
    out["n_eval_bins_mos"] = len(okd)
    out["n_ok_bins_strict_mos"] = sum(1 for r in bins_mos if r["ok_strict"] is True)

    # ── 失真类型 / 内容域箱（TID2013 用 24 类失真类型；KonIQ 无域标签 ⇒ 跳过）──
    if domains is not None and by_domain:
        dom = np.asarray(domains)
        if dom.size == yy.size:
            bd = []
            for c in sorted(np.unique(dom), key=_dom_key):
                rec = _pack(dom == c)
                if rec is None:
                    continue
                rec["domain"] = str(c)
                bd.append(rec)
            out["bins_domain"] = bd or None
            okd2 = [r["ok"] for r in bd if r["ok"] is not None]
            out["n_ok_bins_domain"] = sum(1 for v in okd2 if v)
            out["n_eval_bins_domain"] = len(okd2)
            out["n_ok_bins_strict_domain"] = sum(1 for r in bd if r["ok_strict"] is True)

    out["verdict"] = "info"
    return out


# ─────────────── 方法学对照：成对选择准确率 ───────────────
def pairwise_accuracy(score, y):
    """成对选择准确率 = 同序对比例。

    ⚠️ **对任何严格单调变换不变**（只依赖 `score` 的大小关系）⇒ 它**原理上测不出校准**。
    本函数存在的唯一目的，是**实测**这件事：在族 0–5 六类严格单调映射上，它的值应当
    逐位相同，而判据 3 的三分量**有区分**。
    Tie in y 计入分母、不计入分子（与 Kendall τ_a 的口径一致）。
    """
    s = np.asarray(score, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ds = np.sign(s[:, None] - s[None, :])
    dy = np.sign(y[:, None] - y[None, :])
    iu = np.triu_indices(s.size, k=1)
    a, b = ds[iu], dy[iu]
    nz = b != 0
    if not np.any(nz):
        return float("nan")
    return float(np.mean(a[nz] == b[nz]))


def rank_flip_stats(z, x, tol_rel=1.0e-9):
    """**精确算术下的秩不变性** —— 这才是「成对指标对单调变换不变」的可证形式。

    直接比 `pairwise_accuracy` 的数值会看到一个假象：`none` / `minmax` / `zscore` 三者都是
    严格单调的，但仍给出 0.6711 / 0.6708 / 0.6705 这类**不相等**的数。根因是 float64 ——
    对仿射映射 `(αx_i + β) − (αx_j + β)` **不逐位等于** `α(x_i − x_j)`，在被评分几乎并列的
    那一小撮对上会翻符号。那是浮点误差，不是数学性质。

    本函数把这件事量出来：把**近并列对**（`|x_i − x_j| <= tol_rel · 量程`）单独计数，
    只对剩下的对核 `sign(z_i − z_j) == sign(x_i − x_j)`。
    ⇒ 严格单调映射应当给出 `flips_outside_nearties == 0`；`flips_inside_nearties` 就是
    浮点噪声的规模。**isotonic 与分位数映射还会另引入平台段**（`z_i == z_j` 而 `x_i != x_j`），
    那是它们与另外四类的**唯一实质差别**，由 `plateau_pairs` 单列。
    """
    z = np.asarray(z, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    iu = np.triu_indices(x.size, k=1)
    dx = x[:, None] - x[None, :]
    dz = z[:, None] - z[None, :]
    span = float(np.max(x) - np.min(x)) or 1.0
    near = np.abs(dx[iu]) <= tol_rel * span
    plateau = (dz[iu] == 0.0) & (dx[iu] != 0.0)
    # ⚠️ 翻转 = **真正的序反转**（z 的大小关系与 x 相反），**不含平台**。
    #   `plateau`（z 并列而 x 不并列）在 `sign` 意义下也满足 `sign(dz) != sign(dx)`，
    #   若不显式排除就会被计进翻转 —— 实测过：quantile 的 1573 个「翻转」与它的
    #   `plateau_pairs` 逐个相等，说明那全是平台、不是反转。两类必须分开计。
    flipper = (np.sign(dz[iu]) != np.sign(dx[iu])) & (~near) & (~plateau)
    return {
        "n_pairs": int(iu[0].size),
        "near_tie_pairs": int(near.sum()),
        "flips_outside_nearties": int(flipper.sum()),
        "flips_inside_nearties": int(((np.sign(dz[iu]) != np.sign(dx[iu])) & near).sum()),
        "plateau_pairs": int(plateau.sum()),
    }
