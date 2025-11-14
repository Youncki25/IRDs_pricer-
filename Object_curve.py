from __future__ import annotations
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Sequence
import numpy as np
import QuantLib as ql

# Auto-diff
from Auto_dif import Dual, _real, _grad_from_dual

# math utils pour interpolation
from math import log as _logf, exp as _expf

# ===========================
# Interpolation & utilities
# ===========================

def interpolation(x, X, Y, method="log-linear"):
    """
    x : scalaire (jours)
    X : [x0, x1]
    Y : [y0, y1] (floats ou Dual)
    """
    def _is_dual_val(v):
        return hasattr(v, "real") and hasattr(v, "dual")

    def ln(v):
        return v.__log__() if _is_dual_val(v) else _logf(v)

    def ex(v):
        return v.__exp__() if _is_dual_val(v) else _expf(v)

    x0, x1 = X[0], X[1]
    y0, y1 = Y[0], Y[1]
    t = (x - x0) / (x1 - x0)

    if method == "linear":
        return y0 + (y1 - y0) * t
    elif method == "log-linear":
        return ex(ln(y0) + (ln(y1) - ln(y0)) * t)
    elif method == "cubic":
        import numpy as _np
        try:
            from scipy.interpolate import CubicSpline
        except Exception as e:
            raise RuntimeError("Interpolation 'cubic' requiert SciPy.") from e
        X_np = _np.array(X, dtype=float)
        Y_np = _np.array([yv.real if hasattr(yv, "real") else float(yv) for yv in Y], dtype=float)
        cs = CubicSpline(X_np, Y_np)
        return cs(x)
    else:
        raise ValueError("method doit être 'linear' | 'log-linear' | 'cubic'.")

def add_months(start: datetime, months: int) -> datetime:
    month = start.month - 1 + months
    year  = start.year + month // 12
    month = month % 12 + 1
    day   = start.day
    while True:
        try:
            return datetime(year, month, day)
        except ValueError:
            day -= 1

def ql_to_dt(d: ql.Date) -> datetime:
    return datetime(d.year(), d.month(), d.dayOfMonth())

# ===========================
# Curve / Schedule / Swap
# ===========================

class Curve:
    def __init__(self, nodes: dict, interpolation_method: str = "linear"):
        self.nodes = dict(nodes)
        self.nodes_dates = sorted(self.nodes.keys())
        self.method = interpolation_method

    def __repr__(self):
        lines = [f"node_date  | value (metho :{self.method})",
                 "------------|------------"]
        for d in self.nodes_dates:
            v = self.nodes[d]
            val = v.real if hasattr(v, "real") else float(v)
            lines.append(f"{d:%Y-%b-%d} | {val:0.6f}")
        return "\n".join(lines)
    __str__ = __repr__

    def __getitem__(self, date: datetime):
        if date <= self.nodes_dates[0]:
            return self.nodes[self.nodes_dates[0]]
        for i in range(len(self.nodes_dates)-1):
            d0, d1 = self.nodes_dates[i], self.nodes_dates[i+1]
            if d0 <= date <= d1:
                x = (date - d0).days
                X = [0.0, (d1 - d0).days]
                Y = [self.nodes[d0], self.nodes[d1]]
                return interpolation(x, X, Y, method=self.method)
        return self.nodes[self.nodes_dates[-1]]

class Schedule:
    def __init__(self, start: datetime, tenor_months: int, period: int, calendar=None):
        self.start = start
        self.tenor = tenor_months
        self.period = period
        self.calendar = calendar or ql.TARGET()
        self._build()

    def __repr__(self):
        out = ["period_start      | period_end        | period_DCF",
               "------------------+-------------------+-----------"]
        for d0, d1, dcf in self.data:
            out.append(f"{d0:%Y-%b-%d}       | {d1:%Y-%b-%d}       | {dcf:.6f}")
        return "\n".join(out)

    def _build(self):
        qldc = ql.Actual365Fixed()
        ql_start = ql.Date(self.start.day, self.start.month, self.start.year)
        end_dt = add_months(self.start, self.tenor)
        ql_end = ql.Date(end_dt.day, end_dt.month, end_dt.year)

        ql_sched = ql.Schedule(
            ql_start, ql_end,
            ql.Period(self.period, ql.Months),
            self.calendar,
            ql.ModifiedFollowing, ql.ModifiedFollowing,
            ql.DateGeneration.Forward, False
        )
        self.data = []
        for i in range(len(ql_sched)-1):
            d0, d1 = ql_sched[i], ql_sched[i+1]
            dcf = qldc.yearFraction(d0, d1)
            self.data.append([ql_to_dt(d0), ql_to_dt(d1), float(dcf)])

class Swap:
    def __init__(self, start: datetime, tenor: int, period_fix: int, period_float: int):
        self.start = start
        self.end = add_months(start, tenor)
        self.schedule_fix = Schedule(start, tenor, period_fix)
        self.schedule_float = Schedule(start, tenor, period_float)

    def __repr__(self):
        return f"\n<Swap : {self.start:%Y-%b-%d} -> {self.end:%Y-%b-%d}>\n"

    def annuity(self, curve: Curve) -> object:
        return sum(curve[p1] * dcf for (_, p1, dcf) in self.schedule_fix.data)

    def rate(self, curve: Curve) -> object:
        A = self.annuity(curve)      # DF-weighted accruals (fixed leg)
        DF0 = curve[self.start]
        DFT = curve[self.end]
        return (DF0 - DFT) / A * 100.0

    def npv(self, curve: Curve, fixed_rate: float, notional: float = 1e6) -> object:
        A = self.annuity(curve)
        DF0 = curve[self.start]
        DFT = curve[self.end]
        pv_fixed = (fixed_rate / 100.0) * A * notional
        pv_float = (DF0 - DFT) * notional
        return pv_fixed - pv_float

    def risk(self, curve: "SolvedCurve", fixed_rate: float, notional: float = 1e6):
        """
        ∇_S P = (∂v/∂S)^T ∇_v P  — renvoie (m x 1) en P&L / 1% de quote.
        Pour 1bp: multiplier par 0.01.
        """
        # courbe Dual sur nodes (tag sur variables libres)
        nodes = {}
        for j, d in enumerate(curve.node_dates):
            tag = {f"v{j}": 1.0} if j in curve.free_idx else {}
            nodes[d] = Dual(curve.v[j], tag)
        c_dual = Curve(nodes=nodes, interpolation_method=curve.base.method)
        # ∇_v P
        pv = self.npv(c_dual, fixed_rate=fixed_rate, notional=notional)  # Dual
        grad_v_P = np.array([pv.dual.get(f"v{j}", 0.0) for j in range(len(curve.v))], dtype=float).reshape(-1, 1)
        # K
        K = curve.K  # (n x m)
        grad_s_P = K.T @ grad_v_P       # (m x 1)
        return grad_s_P / 100.0         # par 1%

# ===========================
# Solver (LM/GN/GD)
# ===========================

@dataclass
class SolvedCurveInput:
    base_curve: Curve
    swaps: List[Swap]
    target_rates: np.ndarray    # en %
    weights: np.ndarray | None = None
    vary_first_node: bool = False

class SolvedCurve:
    def __init__(self, data: SolvedCurveInput, algo: str = "levenberg_marquardt", lam0: float = 1000.0):
        self.base = data.base_curve
        self.swaps = data.swaps
        self.S = np.array(data.target_rates, dtype=float)
        self.m = len(self.S)

        self.node_dates: List[datetime] = list(self.base.nodes_dates)
        self.v = np.array([_real(self.base.nodes[d]) for d in self.node_dates], dtype=float)

        # EXACTEMENT 3 paramètres libres (≈ 2y, 5y, 10y) : indices 2,4,6 sur la grille donnée
        self.free_idx = [2, 4, 6]

        self.param_names = [f"v{k}" for k in self.free_idx]
        self.algo = algo
        self.lam = float(lam0)
        self.w = np.ones(self.m) if data.weights is None else np.array(data.weights, dtype=float)

        self.J = None
        self.grad = None
        self.f = None
        self.r = None
        self.grad_s_v = None  # cache pour K

    def _curve_from_vector(self, v_vec: np.ndarray) -> Curve:
        nodes: Dict[datetime, object] = {}
        for j, d in enumerate(self.node_dates):
            if j in self.free_idx:
                key = f"v{j}"
                nodes[d] = Dual(v_vec[j], {key: 1.0})
            else:
                nodes[d] = float(v_vec[j])
        return Curve(nodes=nodes, interpolation_method=self.base.method)

    def calculate_metrics(self, v_vec: np.ndarray):
        curve_dual = self._curve_from_vector(v_vec)
        r_list, d_list = [], []
        names = [f"v{j}" for j in self.free_idx]
        for sw in self.swaps:
            ri = sw.rate(curve_dual)
            r_list.append(_real(ri))
            d_list.append(_grad_from_dual(ri, names))
        r = np.array(r_list, dtype=float)
        J = np.zeros((self.m, len(self.free_idx)))
        if len(self.free_idx) > 0:
            J[:, :] = np.vstack(d_list)
        resid = r - self.S
        W = np.diag(self.w)
        fval = float(resid.T @ W @ resid)
        grad_free = 2.0 * J.T @ (W @ resid)
        grad_full = np.zeros_like(v_vec)
        for k, j in enumerate(self.free_idx):
            grad_full[j] = grad_free[k]
        self.r, self.J, self.f, self.grad = r, J, fval, grad_full
        return fval

    def _update_step_gradient(self, v_vec: np.ndarray):
        if len(self.free_idx) == 0:
            return v_vec.copy(), np.zeros_like(v_vec)
        Jfull = np.zeros((self.m, len(v_vec)))
        for k, j in enumerate(self.free_idx):
            Jfull[:, j] = self.J[:, k]
        y = Jfull.T @ self.grad
        denom = float(y.T @ y) if float(y.T @ y) > 1e-18 else 1e-18
        alpha = float(y.T @ (self.r - self.S)) / denom
        delta = -alpha * self.grad
        v_new = v_vec + delta
        if 0 not in self.free_idx:
            v_new[0] = v_vec[0]
        return v_new, delta

    def _update_step_gauss_newton(self, v_vec: np.ndarray):
        if len(self.free_idx) == 0:
            return v_vec.copy(), np.zeros_like(v_vec)
        A = self.J.T @ self.J + 1e-12 * np.eye(self.J.shape[1])
        b = 0.5 * self.grad[self.free_idx]
        delta_free = np.linalg.solve(A, b)
        delta = np.zeros_like(v_vec)
        for k, j in enumerate(self.free_idx):
            delta[j] = delta_free[k]
        v_new = v_vec + delta
        if 0 not in self.free_idx:
            v_new[0] = v_vec[0]
        return v_new, delta

    def _update_step_levenberg_marquardt(self, v_vec: np.ndarray):
        if len(self.free_idx) == 0:
            return v_vec.copy(), np.zeros_like(v_vec)
        A = self.J.T @ self.J + self.lam * np.eye(self.J.shape[1])
        b = 0.5 * self.grad[self.free_idx]
        delta_free = np.linalg.solve(A, b)
        delta = np.zeros_like(v_vec)
        for k, j in enumerate(self.free_idx):
            delta[j] = delta_free[k]
        v_new = v_vec + delta
        if 0 not in self.free_idx:
            v_new[0] = v_vec[0]
        return v_new, delta

    def iterate(self, max_iter: int = 2000, tol: float = 1e-10, verbose: bool = False):
        v = self.v.copy()
        self.calculate_metrics(v)
        f_prev = self.f
        f_hist = [f_prev]
        for it in range(max_iter):
            if self.algo == "gradient_descent":
                v_trial, delta = self._update_step_gradient(v)
            elif self.algo == "gauss_newton":
                v_trial, delta = self._update_step_gauss_newton(v)
            else:
                v_trial, delta = self._update_step_levenberg_marquardt(v)

            v_save = v.copy()
            self.calculate_metrics(v_trial)
            improved = self.f < f_prev

            if self.algo == "levenberg_marquardt":
                self.lam = self.lam / 2.0 if improved else self.lam * 2.0

            if verbose and (it % 10 == 0 or improved):
                print(f"it={it:4d}  f={self.f:.6e}  lam={self.lam:.3e}  |Δ|={np.linalg.norm(delta):.3e}")

            if improved:
                if abs(f_prev - self.f) < tol:
                    v = v_trial
                    break
                v = v_trial
                f_prev = self.f
            else:
                if self.algo in ("gradient_descent", "gauss_newton"):
                    v = v_save
                    break
                else:
                    v = v_save
            f_hist.append(f_prev)
        self.v = v
        self.calculate_metrics(self.v)
        return {
            "iters": len(f_hist)-1,
            "f": self.f,
            "f_hist": np.array(f_hist+[self.f]),
            "v": self.v.copy(),
            "rates_model": self.r.copy(),
            "jacobian": self.J.copy(),
            "grad": self.grad.copy(),
        }

    # ---- K = ∂v/∂S par différences finies (forward) ----
    def compute_grad_sv(self, ds: float = 0.01, max_iter: int = 800, tol: float = 1e-10):
        """
        Calcule K = ∂v/∂S (n x m) en bumpant chaque quote S_i de +ds (en %).
        ds=0.01 ≈ 1 bp si S est en pourcentage (ex: 2.50 -> 2.51).
        """
        m, n = self.m, len(self.v)
        K = np.zeros((n, m), dtype=float)
        base_v = self.v.copy()

        for i in range(m):
            S_bump = self.S.copy()
            S_bump[i] += ds
            data_bump = SolvedCurveInput(
                base_curve=self.base,
                swaps=self.swaps,
                target_rates=S_bump,
                weights=self.w,
                vary_first_node=(0 in self.free_idx)
            )
            # lambda initial plus souple pour éviter la paralysie
            s_cv = SolvedCurve(data_bump, algo=self.algo, lam0=100.0)
            s_cv.v = base_v.copy()  # warm start
            s_cv.iterate(max_iter=max_iter, tol=max(tol, 1e-10), verbose=False)

            dv = s_cv.v - base_v
            K[:, i] = dv / ds

        self.grad_s_v = K
        return K

    @property
    def K(self):
        if getattr(self, "grad_s_v", None) is None:
            return self.compute_grad_sv()
        return self.grad_s_v

# ===========================
# Demo: calibration + risk table (3 maturities)
# ===========================

if __name__ == "__main__":
    # Courbe initiale (DFs aux noeuds)
    curve0 = Curve(
        nodes={
            datetime(2024,1,2): 1.00,
            datetime(2025,1,2): 0.963,
            datetime(2026,1,2): 0.931,
            datetime(2027,1,2): 0.902,
            datetime(2029,1,2): 0.860,
            datetime(2031,1,2): 0.825,
            datetime(2034,1,2): 0.790,
            datetime(2039,1,2): 0.730,
            datetime(2044,1,2): 0.690,
            datetime(2054,1,2): 0.635,
        },
        interpolation_method="log-linear"
    )

    spot = datetime(2024,1,2)

    # ===== Calibration sur 3 maturités (2y, 5y, 10y) =====
    PAR_MATS = (2, 5, 10)
    par_swaps = [Swap(spot, 12*y, period_fix=12, period_float=6) for y in PAR_MATS]

    # COTATIONS marché alignées (en %)
    S = np.array([3.40, 2.80, 2.55], dtype=float)

    data = SolvedCurveInput(
        base_curve=curve0,
        swaps=par_swaps,
        target_rates=S,
        weights=None,
        vary_first_node=False
    )
    solver = SolvedCurve(data, algo="levenberg_marquardt", lam0=1000.0)
    out = solver.iterate(max_iter=500, tol=1e-12, verbose=True)

    print("\n--- Résultat calibration ---")
    print("f final :", out["f"])
    print("r(v*)   :", out["rates_model"])
    print("DF calibrés :")
    for d, val in zip(solver.node_dates, out["v"]):
        print(f"  {d:%Y-%m-%d} : {val:0.8f}")

    # ------------- Risk table (∂PV/∂bp) -------------
    import pandas as pd

    def curve_from_solver_floats(solver: SolvedCurve) -> Curve:
        return Curve(
            {d: float(v) for d, v in zip(solver.node_dates, solver.v)},
            interpolation_method=solver.base.method
        )

    def par_swaps_dict(spot_dt, par_list=PAR_MATS):
        return {f"{y}y": Swap(spot_dt, tenor=12*y, period_fix=12, period_float=6) for y in par_list}

    def fwd_swaps_dict(spot_dt):
        specs = ((0,1),(1,1),(2,3),(5,5))
        d = {}
        for off, ten in specs:
            start = add_months(spot_dt, 12*off)
            label = f"{off}y{ten}y" if off>0 else f"{ten}y"
            d[label] = Swap(start, tenor=12*ten, period_fix=12, period_float=6)
        return d

    def mid_rate(sw, curve_float):
        return _real(sw.rate(curve_float))

    def risk_vector_bp(sw, solver, fixed_rate_pct, notional=100e6):
        # risk() renvoie par 1% -> *0.01 pour 1 bp
        return sw.risk(solver, fixed_rate=fixed_rate_pct, notional=notional) * 0.01

    par_swaps_map = par_swaps_dict(spot)
    fwd_swaps_map = fwd_swaps_dict(spot)
    curve_star = curve_from_solver_floats(solver)

    row_labels = list(fwd_swaps_map.keys())
    col_labels = list(par_swaps_map.keys())  # ["2y","5y","10y"]
    assert len(col_labels) == solver.m, "Colonnes ≠ nb de quotes calibrées."

    fwd_mids = {lab: mid_rate(sw, curve_star) for lab, sw in fwd_swaps_map.items()}

    risk_matrix = np.zeros((len(row_labels), len(col_labels)))
    for i, rlab in enumerate(row_labels):
        sw = fwd_swaps_map[rlab]
        risk_matrix[i,:] = risk_vector_bp(sw, solver, fwd_mids[rlab], 100e6).ravel()

    df = pd.DataFrame(risk_matrix, index=row_labels, columns=col_labels)
    print("\nRaw ∂PV/∂bp (per par quote column):")
    print(df.round(3))

    df_norm = df.divide(df.sum(axis=0), axis=1)
    print("\nNormalised (each column sums to 1):")
    print(df_norm.round(3))

