# Auto_dif.py — Dual + helpers
from __future__ import annotations
from typing import Sequence
import math
import numpy as np

class Dual:
    def __init__(self, real, dual=None):
        self.real = float(real) if not isinstance(real, Dual) else real.real
        self.dual = {} if dual is None else dict(dual)

    def __repr__(self):
        lines = [f"f = {self.real:.10f}"]
        for k, v in self.dual.items():
            lines.append(f"df/d{k} = {v:.10f}")
        return "\n".join(lines)

    # ----- unary
    def __neg__(self):
        return Dual(-self.real, {k: -v for k, v in self.dual.items()})

    # ----- comparisons
    def __eq__(self, other): return isinstance(other, Dual) and self.real == other.real and self.dual == other.dual
    def __ne__(self, other): return not self == other

    # ----- addition
    def __add__(self, other):
        if isinstance(other, Dual):
            real = self.real + other.real
            dual = self.dual.copy()
            for k, v in other.dual.items():
                dual[k] = dual.get(k, 0.0) + v
            return Dual(real, dual)
        return Dual(self.real + other, self.dual)
    __radd__ = __add__

    # ----- subtraction
    def __sub__(self, other):  return self + (-other)
    def __rsub__(self, other): return Dual(other) + (-self)

    # ----- multiplication
    def __mul__(self, other):
        if isinstance(other, Dual):
            real = self.real * other.real
            keys = set(self.dual) | set(other.dual)
            dual = {k: self.real * other.dual.get(k, 0.0) + other.real * self.dual.get(k, 0.0) for k in keys}
            return Dual(real, dual)
        return Dual(self.real * other, {k: v * other for k, v in self.dual.items()})
    __rmul__ = __mul__

    # ----- division
    def __truediv__(self, other):
        if isinstance(other, Dual):
            a, b = self.real, other.real
            real = a / b
            keys = set(self.dual) | set(other.dual)
            dual = {k: (b * self.dual.get(k, 0.0) - a * other.dual.get(k, 0.0)) / (b * b) for k in keys}
            return Dual(real, dual)
        return Dual(self.real / other, {k: v / other for k, v in self.dual.items()})
    def __rtruediv__(self, other):
        a, b = other, self.real
        real = a / b
        dual = {k: (-a * v) / (b * b) for k, v in self.dual.items()}
        return Dual(real, dual)

    # ----- power
    def __pow__(self, p):
        real = self.real ** p
        dual = {k: p * (self.real ** (p - 1)) * v for k, v in self.dual.items()}
        return Dual(real, dual)
    def __rpow__(self, base):
        real = base ** self.real
        dual = {k: real * math.log(base) * v for k, v in self.dual.items()}
        return Dual(real, dual)

    # ----- exp/log (+compat avec l'interpolation)
    def exp(self):
        real = math.exp(self.real)
        dual = {k: real * v for k, v in self.dual.items()}
        return Dual(real, dual)
    def log(self):
        real = math.log(self.real)
        dual = {k: v / self.real for k, v in self.dual.items()}
        return Dual(real, dual)
    def __exp__(self): return self.exp()
    def __log__(self): return self.log()


# ----- helpers module-level -----
def _is_dual(x):
    return isinstance(x, Dual) or (hasattr(x, "real") and hasattr(x, "dual"))

def _real(x):
    if _is_dual(x):
        return x.real
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return float(x)

def _grad_from_dual(d, names: Sequence[str]):
    if not _is_dual(d):
        return np.zeros(len(names), dtype=float)
    return np.array([d.dual.get(n, 0.0) for n in names], dtype=float)
