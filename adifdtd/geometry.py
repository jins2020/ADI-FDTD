"""Mapping of structures (boxes and unstructured meshes) onto the Yee grid.

Cell-based material arrays (eps_r, mu_r, sigma, sigma_m) and edge-based PEC
masks are built in the order the structures appear in the input file (later
entries overwrite earlier ones, PEC always wins over materials).

Unstructured meshes
-------------------
* volume region (tetrahedra, or a closed STL surface): its closed boundary
  surface is voxelised by ray parity along z. Material cells get the volume
  fraction computed from ``subcell`` x ``subcell`` x ``subcell`` sample
  points per cell (eps, mu, sigma blended by fraction). PEC volumes zero every
  E edge whose midpoint lies inside or on the surface.
* surface region (triangles)  -> PEC sheet: triangles are sampled densely,
  samples snap to the nearest grid node, and every edge between two sheet
  nodes becomes PEC.
* line region (2-node lines)  -> PEC wire, snapped the same way.
"""
from __future__ import annotations

import numpy as np

from .meshfile import Mesh

# marker values of the cell geometry map (used for drawing)
GEO_VACUUM, GEO_PEC, GEO_MATERIAL, GEO_PORT = 0, 1, 2, 3


# --------------------------------------------------------------- voxelising
def boundary_triangles(tets):
    """Faces that belong to exactly one tetrahedron."""
    if len(tets) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    f = tets[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]].reshape(-1, 3)
    key = np.sort(f, axis=1)
    _, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    return f[idx[cnt == 1]]


def _parity(V, xs, ys, zs):
    """Inside test of lattice points by ray parity along +z.

    V : (m, 3, 3) triangle vertices of a closed surface.
    """
    nx, ny, nz = len(xs), len(ys), len(zs)
    toggles = np.zeros((nx, ny, nz + 1), dtype=np.int32)
    lo = V.min(axis=1)
    hi = V.max(axis=1)
    i0s = np.searchsorted(xs, lo[:, 0], "left")
    i1s = np.searchsorted(xs, hi[:, 0], "right")
    j0s = np.searchsorted(ys, lo[:, 1], "left")
    j1s = np.searchsorted(ys, hi[:, 1], "right")
    for t in np.nonzero((i1s > i0s) & (j1s > j0s))[0]:
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = V[t]
        det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(det) < 1e-30:
            continue                       # triangle parallel to the ray
        X = xs[i0s[t]:i1s[t], None]
        Y = ys[None, j0s[t]:j1s[t]]
        l0 = ((y1 - y2) * (X - x2) + (x2 - x1) * (Y - y2)) / det
        l1 = ((y2 - y0) * (X - x2) + (x0 - x2) * (Y - y2)) / det
        l2 = 1.0 - l0 - l1
        hit = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        if not hit.any():
            continue
        ii, jj = np.nonzero(hit)
        zi = (l0 * z0 + l1 * z1 + l2 * z2)[ii, jj]
        kk = np.searchsorted(zs, zi, "right")
        np.add.at(toggles, (ii + i0s[t], jj + j0s[t], kk), 1)
    return (np.cumsum(toggles, axis=2)[:, :, :nz] % 2) == 1


def inside_lattice(V, xs, ys, zs, closed=False, eps=None):
    """Points of the lattice xs x ys x zs inside the closed surface V.

    The lattice is shifted by a tiny, non-symmetric offset so that rays never
    run exactly along mesh edges. ``closed=True`` also counts points lying on
    the surface (union of the 8 diagonal shifts).
    """
    if eps is None:
        span = max(np.ptp(xs), np.ptp(ys), np.ptp(zs), 1e-30)
        eps = 1e-7 * span
    shift = np.array([1.0, 0.6180339887, 0.3819660113]) * eps
    m = _parity(V, xs + shift[0], ys + shift[1], zs + shift[2])
    if closed:      # points on faces/edges/corners: try all 8 shift directions
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    if sx == sy == sz == 1:
                        continue
                    m |= _parity(V, xs + sx * shift[0], ys + sy * shift[1], zs + sz * shift[2])
    return m


def _snap(coords, x):
    """Nearest index in sorted coords for an array of x."""
    i = np.clip(np.searchsorted(coords, x), 1, len(coords) - 1)
    left = coords[i - 1]
    right = coords[i]
    return np.where(np.abs(x - left) <= np.abs(right - x), i - 1, i)


def _in_domain(grid, P, tol):
    ok = np.ones(len(P), dtype=bool)
    for a in range(3):
        ok &= (P[:, a] >= grid.nodes[a][0] - tol[a]) & (P[:, a] <= grid.nodes[a][-1] + tol[a])
    return ok


def sheet_node_mask(grid, nodes, tris):
    """Grid nodes touched by a triangulated sheet."""
    mask = np.zeros(tuple(n + 1 for n in grid.N), dtype=bool)
    h = min(grid.dmin) / 3.0
    tol = [0.5 * p.max() for p in grid.dp]
    for tri in tris:
        A, B, C = nodes[tri]
        n = int(np.ceil(max(np.linalg.norm(B - A), np.linalg.norm(C - A),
                            np.linalg.norm(C - B)) / h)) + 1
        u, v = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1), indexing="ij")
        keep = (u + v) <= 1 + 1e-12
        P = A + u[keep, None] * (B - A) + v[keep, None] * (C - A)
        P = P[_in_domain(grid, P, tol)]
        if len(P):
            idx = tuple(_snap(grid.nodes[a], P[:, a]) for a in range(3))
            mask[idx] = True
    return mask


def wire_paths(grid, nodes, lines):
    """Yield (axis, i, j, k) edges of the snapped staircase path of each line."""
    h = min(grid.dmin) / 4.0
    tol = [0.5 * p.max() for p in grid.dp]
    for a_, b_ in lines:
        A, B = nodes[a_], nodes[b_]
        n = int(np.ceil(np.linalg.norm(B - A) / h)) + 1
        P = A + np.linspace(0, 1, n + 1)[:, None] * (B - A)
        P = P[_in_domain(grid, P, tol)]
        if len(P) < 2:
            continue
        I = np.stack([_snap(grid.nodes[a], P[:, a]) for a in range(3)], axis=1)
        for p, q in zip(I[:-1], I[1:]):
            cur = p.copy()
            for ax in range(3):            # staircase: one axis at a time
                while cur[ax] != q[ax]:
                    step = 1 if q[ax] > cur[ax] else -1
                    lo = min(cur[ax], cur[ax] + step)
                    e = cur.copy()
                    e[ax] = lo
                    yield (ax, *e)
                    cur[ax] += step


# ---------------------------------------------------------------- materials
class Materials:
    """Cell material arrays, PEC edge masks and a drawing map."""

    def __init__(self, grid):
        self.grid = grid
        N = grid.N
        Nx, Ny, Nz = N
        self.eps = np.ones(N)
        self.mu = np.ones(N)
        self.sig = np.zeros(N)
        self.sigm = np.zeros(N)
        self.geo = np.zeros(N, dtype=np.int8)
        self.eshape = [(Nx, Ny + 1, Nz + 1), (Nx + 1, Ny, Nz + 1), (Nx + 1, Ny + 1, Nz)]
        self.pec = [np.zeros(s, dtype=bool) for s in self.eshape]

    # -- helpers
    def _cell_slices(self, box):
        sl = []
        for a, (lo, hi) in enumerate(box.ranges()):
            if hi > lo:
                sl.append(slice(lo, hi))
            else:
                lo = min(lo, self.grid.N[a] - 1)
                sl.append(slice(lo, lo + 1))
        return tuple(sl)

    def _blend(self, frac, props):
        f = frac
        self.eps = (1 - f) * self.eps + f * props.get("eps", 1.0)
        self.mu = (1 - f) * self.mu + f * props.get("mu", 1.0)
        self.sig = (1 - f) * self.sig + f * props.get("sigma", 0.0)
        self.sigm = (1 - f) * self.sigm + f * props.get("sigma_m", 0.0)
        self.geo[(f >= 0.5) & (self.geo != GEO_PEC)] = GEO_MATERIAL

    # -- boxes
    def add_box_material(self, box, props):
        (i0, i1), (j0, j1), (k0, k1) = box.ranges()
        f = np.zeros(self.grid.N)
        f[i0:i1, j0:j1, k0:k1] = 1.0
        self._blend(f, props)

    def add_box_pec(self, box):
        for c in range(3):
            sl = tuple(slice(lo, hi) if a == c else slice(lo, hi + 1)
                       for a, (lo, hi) in enumerate(box.ranges()))
            self.pec[c][sl] = True
        self.geo[self._cell_slices(box)] = GEO_PEC

    # -- meshes
    def _sub_lattice(self, s):
        g = self.grid
        out = []
        for a in range(3):
            frac = (np.arange(s) + 0.5) / s
            out.append((g.nodes[a][:-1, None] + frac[None, :] * g.dp[a][:, None]).ravel())
        return out

    def add_mesh_region(self, mesh: Mesh, region, kind, props, subcell=2):
        g = self.grid
        tets, tris, lines = region.arrays()
        if len(tets) or (region.dim == 3 and len(tris)):
            surf = boundary_triangles(tets) if len(tets) else tris
            V = mesh.nodes[surf]
            if kind == "pec":
                for c in range(3):
                    xs = [g.centers[a] if a == c else g.nodes[a] for a in range(3)]
                    self.pec[c] |= inside_lattice(V, *xs, closed=True)
                self.geo[inside_lattice(V, *g.centers)] = GEO_PEC
            else:
                s = max(1, int(subcell))
                m = inside_lattice(V, *self._sub_lattice(s))
                Nx, Ny, Nz = g.N
                frac = m.reshape(Nx, s, Ny, s, Nz, s).mean(axis=(1, 3, 5))
                self._blend(frac, props)
            return
        if kind != "pec":
            raise ValueError(f"region {region.tag}: surfaces/lines can only be type=pec")
        if len(tris):
            nm = sheet_node_mask(g, mesh.nodes, tris)
            self.pec[0] |= nm[:-1] & nm[1:]
            self.pec[1] |= nm[:, :-1] & nm[:, 1:]
            self.pec[2] |= nm[:, :, :-1] & nm[:, :, 1:]
            self._mark_nodes(nm)
        if len(lines):
            nm = np.zeros(tuple(n + 1 for n in g.N), dtype=bool)
            for ax, i, j, k in wire_paths(g, mesh.nodes, lines):
                self.pec[ax][i, j, k] = True
                nm[i, j, k] = True
            self._mark_nodes(nm)

    def _mark_nodes(self, nm):
        """Mark cells around flagged nodes for drawing."""
        idx = np.nonzero(nm)
        for o in ((0, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1),
                  (-1, -1, 0), (-1, 0, -1), (0, -1, -1), (-1, -1, -1)):
            ii = [np.clip(idx[a] + o[a], 0, self.grid.N[a] - 1) for a in range(3)]
            self.geo[tuple(ii)] = GEO_PEC

    # -- derived edge / face quantities
    def edge_average(self, cell, c):
        """Area-weighted average of the 4 cells around each E_c edge."""
        g = self.grid
        t = [a for a in range(3) if a != c]
        pad = [(0, 0) if a == c else (1, 1) for a in range(3)]
        P = np.pad(cell, pad, mode="edge")
        w = [np.pad(g.dp[a], 1, mode="edge") for a in t]
        acc = 0.0
        wsum = 0.0
        shape = self.eshape[c]
        for o1 in (0, 1):
            for o2 in (0, 1):
                sl = [slice(None)] * 3
                sl[t[0]] = slice(o1, o1 + shape[t[0]])
                sl[t[1]] = slice(o2, o2 + shape[t[1]])
                w0 = w[0][o1:o1 + shape[t[0]]]
                w1 = w[1][o2:o2 + shape[t[1]]]
                wt = _b(w0, t[0]) * _b(w1, t[1])
                acc = acc + wt * P[tuple(sl)]
                wsum = wsum + wt
        return np.broadcast_to(acc / wsum, shape).copy()

    def face_harmonic(self, cell, c):
        """Length-weighted harmonic mean of the 2 cells sharing each H_c face."""
        g = self.grid
        pad = [(1, 1) if a == c else (0, 0) for a in range(3)]
        P = np.pad(cell, pad, mode="edge")
        w = np.pad(g.dp[c], 1, mode="edge")
        n = g.N[c] + 1
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[c] = slice(0, n)
        sl1[c] = slice(1, n + 1)
        w0 = _b(w[0:n], c)
        w1 = _b(w[1:n + 1], c)
        return (w0 + w1) / (w0 / P[tuple(sl0)] + w1 / P[tuple(sl1)])

    def face_mean(self, cell, c):
        g = self.grid
        pad = [(1, 1) if a == c else (0, 0) for a in range(3)]
        P = np.pad(cell, pad, mode="edge")
        w = np.pad(g.dp[c], 1, mode="edge")
        n = g.N[c] + 1
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[c] = slice(0, n)
        sl1[c] = slice(1, n + 1)
        w0 = _b(w[0:n], c)
        w1 = _b(w[1:n + 1], c)
        return (w0 * P[tuple(sl0)] + w1 * P[tuple(sl1)]) / (w0 + w1)


def _b(v, axis):
    s = [1, 1, 1]
    s[axis] = v.size
    return v.reshape(s)
