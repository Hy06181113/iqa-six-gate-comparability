# -*- coding: utf-8 -*-
"""P1 第②步「分层切分」：分层键 + 分组键，固定种子。

两个数据集**不是同一件事**，必须分开说：

  · **TID2013**（主结论）：文件名 `I02_01_1.bmp` 的第 1 段是**参考图编号**，用它作分组键，
    用 MOS 五分位作分层键，跑 `StratifiedGroupKFold`。⇒「**同一参考图的不同失真版本不得
    同时出现在校准集和测试集**」这条硬约束由分组机制**在结构上**保证，本模块再断言一遍。
  · **KonIQ**（污染对照）：无参考图变体 ⇒ 跑 `StratifiedKFold`（只按 MOS 五分位分层）。
    ⚠️ **该硬约束在本数据集上是真空条件**（不存在参考图的多个失真版本），必须显式声明——
    否则会读成「KonIQ 上通过了这条约束」，而实际是「没有可违反的对象」。

主结果 = **5 折的样本外预测拼接**；逐折值另存，用于报稳定性。
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def mos_bins(y, n_bins):
    """按人类尺度 y 的分位数切箱，返回 [0, n_bins) 的整数标签。

    用 `np.quantile` 的边界 + `np.digitize`（而非 `qcut` 那类依赖 pandas 的做法）。
    边界若因大量并列而退化为非严格递增，仍保证落在 [0, n_bins) 内。
    """
    y = np.asarray(y, dtype=np.float64)
    qs = np.quantile(y, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    qs = np.unique(qs)
    return np.clip(np.digitize(y, qs), 0, n_bins - 1)


def make_folds(table, n_splits, seed):
    """返回 [(cal_idx, test_idx), ...]，长度为 n_splits。"""
    y = table.y
    strata = mos_bins(y, 5)
    idx = np.arange(table.n)

    if table.groups is not None:
        groups = np.asarray(table.groups)
        if np.unique(groups).size < n_splits:
            raise SystemExit("分组数 %d < 折数 %d，StratifiedGroupKFold 不可用"
                             % (np.unique(groups).size, n_splits))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        it = splitter.split(idx, strata, groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        it = splitter.split(idx, strata)

    folds = [(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)) for a, b in it]

    # ── 自检：分组键不得跨集（阶段 C 的数据自检断言之一）──
    if table.groups is not None:
        g = np.asarray(table.groups)
        for k, (cal, test) in enumerate(folds):
            inter = set(g[cal].tolist()) & set(g[test].tolist())
            if inter:
                raise SystemExit("第 %d 折违反分组隔离：参考图 %s 同时出现在校准集与测试集"
                                 % (k + 1, sorted(inter)[:5]))
    # ── 自检：每折两侧非空，且并集恰好覆盖全体、交集为空 ──
    seen = np.concatenate([t for _, t in folds])
    if np.unique(seen).size != table.n or seen.size != table.n:
        raise SystemExit("样本外测试集拼接后不是「每张恰好一次」")
    for k, (cal, test) in enumerate(folds):
        if cal.size == 0 or test.size == 0:
            raise SystemExit("第 %d 折一侧为空" % (k + 1))

    return folds
