"""Rectilinear (optionally non-uniform) Yee grid.

Each axis is described by its node coordinates x_0 < x_1 < ... < x_N.
  primal spacing  dp[i] = x_{i+1} - x_i            (N values, cell edges)
  dual spacing    dd[i] = (dp[i-1] + dp[i]) / 2     (N+1 values, at nodes;
                                                     half cell at the ends)
Non-uniform spacing keeps the ADI tridiagonal structure and its unconditional
stability (the energy norm only gets volume weights).
"""
from __future__ import annotations

import numpy as np


def build_axis(spec):
    """Node coordinates from a segment spec.

    spec = (x0, [(x1, n1, r1), (x2, n2, r2), ...])
    Each segment from the previous point to x_k is split into n_k cells.
    r_k = (last cell size)/(first cell size) of the segment (geometric
    grading); r_k = 1 gives uniform cells.
    """
    x0, segs = spec
    nodes = [float(x0)]
    for x1, n, r in segs:
        a = nodes[-1]
        L = float(x1) - a
        if L <= 0 or n < 1:
            raise ValueError("axis segments must be increasing with n >= 1")
        if n == 1 or abs(r - 1.0) < 1e-12:
            w = np.full(n, 1.0)
        else:
            q = r ** (1.0 / (n - 1))
            w = q ** np.arange(n)
        cum = np.cumsum(w) / w.sum() * L
        nodes.extend((a + cum).tolist())
    return np.array(nodes)


class Grid:
    def __init__(self, xn, yn, zn):
        self.nodes = [np.asarray(v, dtype=float) for v in (xn, yn, zn)]
        for v in self.nodes:
            if v.size < 3 or np.any(np.diff(v) <= 0):
                raise ValueError("grid node coordinates must be strictly increasing (>= 2 cells)")
        self.N = tuple(v.size - 1 for v in self.nodes)
        self.dp = [np.diff(v) for v in self.nodes]
        self.dd = []
        for p in self.dp:
            d = np.empty(p.size + 1)
            d[1:-1] = 0.5 * (p[:-1] + p[1:])
            d[0] = 0.5 * p[0]
            d[-1] = 0.5 * p[-1]
            self.dd.append(d)
        self.centers = [0.5 * (v[:-1] + v[1:]) for v in self.nodes]
        self.uniform = all(np.allclose(p, p[0], rtol=1e-9) for p in self.dp)

    @classmethod
    def uniform_grid(cls, shape, cell, origin=(0.0, 0.0, 0.0)):
        return cls(*[origin[a] + cell[a] * np.arange(shape[a] + 1) for a in range(3)])

    @property
    def dmin(self):
        return tuple(float(p.min()) for p in self.dp)

    def node_index(self, a, x):
        """Nearest node index on axis a for coordinate x."""
        return int(np.abs(self.nodes[a] - x).argmin())

    def cell_index(self, a, x):
        """Index of the cell containing coordinate x (clipped)."""
        i = int(np.searchsorted(self.nodes[a], x, side="right") - 1)
        return min(max(i, 0), self.N[a] - 1)

    def describe(self):
        out = []
        for a, nm in enumerate("xyz"):
            p = self.dp[a]
            out.append(f"{nm}: {self.N[a]} cells, [{self.nodes[a][0]:.4g}, {self.nodes[a][-1]:.4g}] m, "
                       f"d = {p.min():.3e} .. {p.max():.3e} m")
        return out
