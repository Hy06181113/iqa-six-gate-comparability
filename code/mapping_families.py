# -*- coding: utf-8 -*-
"""P1 第③步「边际校准」：七类映射到共同人类尺度。

⚠️ **「六类映射」与「七类映射」两种说法指的不是同一份清单，本模块实现的是七个**：
    「六类」指**不含族 0（不校准）**的那六个（前五类严格单调 ＋ isotonic 有平台段）；
    多处旧文写的「六类」实列 7 个。两种说法都成立、只是集合不同，本模块一律按七个。

统一接口：
    m = make_mapping("isotonic")
    m.fit(x_cal, y_cal, judge_range=(L, U))
    z = m.transform(x)          # 任意形状
    m.monotone_ok(grid)         # 非递减（容差 1e-12）
    m.params                    # dict，供「不同校准集下的参数方差」记账

  族 0 `none`      理论量程线性压缩（实践中「把分数当质量」的朴素做法）
  族 1 `minmax`    观测 min-max 压到 [0, 1]
  族 2 `zscore`    标准化后投到人类尺度的 μ/σ
  族 3 `quantile`  经验 CDF 匹配（非参）
  族 4 `platt`     双参数逻辑，**约束 a ≥ 1e-6**（否则会出现单调递减）
  族 5 `logistic5` 五参数单调逻辑，**约束 β1 < β2 且 β5 > 0**
  族 6 `isotonic`  保序回归（非参，有平台段）

⚠️ 族 1 与族 2 都是 2 参数仿射，**但输出尺度不同** —— 族 1 只压 [0,1]、**不投到人类
尺度**，其零点与人类尺度零点不重合 ⇒ 判据 3 的位置项会对族 1 整族报「位置失配」。
**这是真实结论、不是实现错误**，报告里必须显式写明，否则会被读成 bug。
"""
import numpy as np
from scipy.optimize import least_squares
from sklearn.isotonic import IsotonicRegression

MAPPING_NAMES = ["none", "minmax", "zscore", "quantile", "platt", "logistic5", "isotonic"]

#: 严格单调（在样本上）的族 ⇒ 成对选择准确率应当逐位相同。isotonic 因平台段除外。
STRICT_MONOTONE = {"none", "minmax", "zscore", "quantile", "platt", "logistic5"}


class Mapping(object):
    """基类：所有族的 `transform` 都必须是**非递减**的。"""

    name = None
    n_params = 0

    def fit(self, x_cal, y_cal, judge_range=None):
        raise NotImplementedError

    def transform(self, x):
        raise NotImplementedError

    @property
    def params(self):
        return {}

    def monotone_ok(self, grid, tol=1.0e-12):
        d = np.diff(self.transform(np.asarray(grid, dtype=np.float64)))
        return bool(np.min(d) >= -tol), float(np.min(d))


class _None(Mapping):
    """族 0：把理论量程线性压到 [0, 1]。0 参数。"""

    name = "none"

    def fit(self, x_cal, y_cal, judge_range=None):
        if judge_range is None:
            raise ValueError("族 0 需要 judge_range（理论量程），不得用观测极值")
        self.L, self.U = float(judge_range[0]), float(judge_range[1])
        if not self.U > self.L:
            raise ValueError("理论量程非正：%r" % (judge_range,))
        return self

    def transform(self, x):
        return (np.asarray(x, dtype=np.float64) - self.L) / (self.U - self.L)


class _MinMax(Mapping):
    """族 1：观测 min-max 压到 [0, 1]。2 参数。"""

    name = "minmax"
    n_params = 2

    def fit(self, x_cal, y_cal, judge_range=None):
        self.lo, self.hi = float(np.min(x_cal)), float(np.max(x_cal))
        if not self.hi > self.lo:
            raise ValueError("校准集分数无变异，min-max 不可定义")
        return self

    def transform(self, x):
        return (np.asarray(x, dtype=np.float64) - self.lo) / (self.hi - self.lo)

    @property
    def params(self):
        return {"min_cal": self.lo, "max_cal": self.hi}


class _ZScore(Mapping):
    """族 2：标准化后投到人类尺度的 μ/σ。2 参数。"""

    name = "zscore"
    n_params = 2

    def fit(self, x_cal, y_cal, judge_range=None):
        self.mu_x, self.sd_x = float(np.mean(x_cal)), float(np.std(x_cal))
        self.mu_y, self.sd_y = float(np.mean(y_cal)), float(np.std(y_cal))
        if self.sd_x <= 0:
            raise ValueError("校准集分数标准差为 0，z-score 不可定义")
        return self

    def transform(self, x):
        return self.mu_y + self.sd_y * (np.asarray(x, dtype=np.float64) - self.mu_x) / self.sd_x

    @property
    def params(self):
        return {"mu_x": self.mu_x, "sd_x": self.sd_x, "mu_y": self.mu_y, "sd_y": self.sd_y}


class _Quantile(Mapping):
    """族 3：经验 CDF 匹配 f(x) = F_y⁻¹(F_x^cal(x))。非参。"""

    name = "quantile"

    def fit(self, x_cal, y_cal, judge_range=None):
        self.x_sorted = np.sort(np.asarray(x_cal, dtype=np.float64))
        m = self.x_sorted.size
        self.y_sorted = np.sort(np.asarray(y_cal, dtype=np.float64))
        # F_y⁻¹ 的横轴：q = k/m ⇒ 第 k 个次序统计量（k = 1..m）
        self.q_grid = np.arange(1, m + 1, dtype=np.float64) / float(m)
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        q = np.searchsorted(self.x_sorted, x, side="right").astype(np.float64) / self.x_sorted.size
        q = np.clip(q, self.q_grid[0], 1.0)
        return np.interp(q, self.q_grid, self.y_sorted)

    @property
    def params(self):
        return {"m": int(self.x_sorted.size)}


def _sigmoid(t):
    return 1.0 / (1.0 + np.exp(-np.clip(t, -500.0, 500.0)))


class _Platt(Mapping):
    """族 4：f(x) = 1 / (1 + exp(−(a·x + b)))，约束 a ≥ 1e-6。2 参数。"""

    name = "platt"
    n_params = 2

    def fit(self, x_cal, y_cal, judge_range=None):
        x = np.asarray(x_cal, dtype=np.float64)
        y = np.asarray(y_cal, dtype=np.float64)
        sx = float(np.std(x)) or 1.0
        x0 = np.array([1.0 / sx, -float(np.mean(x)) / sx])
        res = least_squares(lambda p: _sigmoid(p[0] * x + p[1]) - y, x0,
                            bounds=([1.0e-6, -np.inf], [np.inf, np.inf]), max_nfev=20000)
        self.a, self.b = float(res.x[0]), float(res.x[1])
        return self

    def transform(self, x):
        return _sigmoid(self.a * np.asarray(x, dtype=np.float64) + self.b)

    @property
    def params(self):
        return {"a": self.a, "b": self.b}


class _Logistic5(Mapping):
    """族 5：f(x) = β1 + (β2 − β1) / (1 + exp(−(x − β3) / |β4|))^β5。

    约束 β1 < β2、β5 > 0 —— 由重参数化**结构性保证**（β2 = β1 + exp(p1)、β4 = exp(p4)、
    β5 = exp(p5)），因此无需带界优化。多起点 `least_squares`。
    """

    name = "logistic5"
    n_params = 5

    def _f(self, p, x):
        b1, b2 = p[0], p[0] + np.exp(p[1])
        b3, b4, b5 = p[2], np.exp(p[3]), np.exp(p[4])
        return b1 + (b2 - b1) / np.power(1.0 + np.exp(-np.clip((x - b3) / b4, -500.0, 500.0)), b5)

    def _resid(self, p, x, y):
        # ⚠️⚠️ **必须传残差，不能传模型本身**。`least_squares` 最小化的是「传入函数的
        #   **返回值**的平方和」——`y` 不在它的视野里，靠闭包/args 自己带进来。
        #   2026-10-15 实测踩到过：写成 `least_squares(self._f, p0, args=(x,), ...)` 时，
        #   被最小化的量成了 `f` 本身，优化器于是老老实实把 `f` 推到 0（β1→0、
        #   幅度 β2−β1→下限、β4→上限、β5→下限），产出 R² = −12.85 的**常函数**，
        #   `SD(f)` 低到 5e-11，还**连带触发判据 3 的退化护栏**。
        #   ⚠️ 该现象一度被误记为「五参数逻辑的两条退化方向、无界最小二乘会掉进去」并把
        #   多起点改成带界——**那个归因是错的**：退化方向的排查、带界、25 个起点都对
        #   常函数无能为力，因为真正的病根是这一行调用形态。带界本身保留（无害的护栏），
        #   但它**不是**修好这个 bug 的原因。
        return self._f(p, x) - y

    def fit(self, x_cal, y_cal, judge_range=None):
        x = np.asarray(x_cal, dtype=np.float64)
        y = np.asarray(y_cal, dtype=np.float64)
        ylo, yhi = float(np.min(y)), float(np.max(y))
        sy = max(yhi - ylo, 1e-6)
        xlo, xhi = float(np.min(x)), float(np.max(x))
        sx = float(np.std(x)) or 1.0

        # 带界：把 β2−β1、β4、β5 都放在**有界**的 log 参数上（由重参数化给出 β1<β2、β5>0）。
        # 界是给最优化的一块护栏，不是本族能用的必要条件；真正必须的是 `_resid`（见上）。
        lb = np.array([ylo - 2 * sy, np.log(1e-4 * sy), xlo - 0.5 * sx,
                       np.log(sx / 1000.0), np.log(0.05)])
        ub = np.array([yhi + 2 * sy, np.log(10.0 * sy), xhi + 0.5 * sx,
                       np.log(sx * 1000.0), np.log(20.0)])
        lo = np.minimum(lb, ub - 1e-9)

        best, best_cost = None, np.inf
        for b3 in np.quantile(x, [0.1, 0.35, 0.5, 0.65, 0.9]):
            for b5 in (0.25, 0.5, 1.0, 2.0, 4.0):
                p0 = np.clip(np.array([ylo, np.log(0.5 * sy), b3, np.log(sx), np.log(b5)]),
                             lo + 1e-12, ub - 1e-12)
                try:
                    res = least_squares(self._resid, p0, args=(x, y), bounds=(lo, ub),
                                        max_nfev=4000, x_scale="jac")
                except Exception:                                  # noqa: BLE001
                    continue
                if res.cost < best_cost:
                    best, best_cost = res.x, res.cost
        if best is None:
            raise RuntimeError("五参数逻辑全部起点失败")
        self._p = best
        resid = self._f(best, x) - y
        ss = float(np.sum(resid ** 2))
        st = float(np.sum((y - y.mean()) ** 2))
        self.r2 = 1.0 - ss / st if st > 0 else float("nan")
        return self

    def transform(self, x):
        return self._f(self._p, np.asarray(x, dtype=np.float64))

    @property
    def params(self):
        p = self._p
        return {"beta1": float(p[0]), "beta2": float(p[0] + np.exp(p[1])),
                "beta3": float(p[2]), "beta4": float(np.exp(p[3])), "beta5": float(np.exp(p[4]))}


class _Isotonic(Mapping):
    """族 6：保序回归，out_of_bounds='clip'。非参，有平台段。"""

    name = "isotonic"

    def fit(self, x_cal, y_cal, judge_range=None):
        self.ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
        self.ir.fit(np.asarray(x_cal, dtype=np.float64), np.asarray(y_cal, dtype=np.float64))
        return self

    def transform(self, x):
        return self.ir.predict(np.asarray(x, dtype=np.float64).ravel()).reshape(np.shape(x))

    @property
    def params(self):
        return {"n_knots": int(len(np.unique(self.ir.f_.x)) if hasattr(self.ir, "f_") else 0)}


_REGISTRY = {c.name: c for c in
             (_None, _MinMax, _ZScore, _Quantile, _Platt, _Logistic5, _Isotonic)}


def make_mapping(name):
    if name not in _REGISTRY:
        raise KeyError("未知映射族 %r；可用：%s" % (name, sorted(_REGISTRY)))
    return _REGISTRY[name]()


def fit_all(name, x_cal, y_cal, judge_range=None):
    return make_mapping(name).fit(x_cal, y_cal, judge_range=judge_range)
