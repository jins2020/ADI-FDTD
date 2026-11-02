"""3D ADI-FDTD solver: non-uniform rectilinear grid, eps/mu materials,
graded matched-loss PML and matched resistive ports.

Scheme
------
Zheng/Namiki ADI (Peaceman-Rachford) splitting of the curl operator.

Sub-step 1 (n -> n+1/2), implicit derivative pairs (sign +1):
    Ex <- dHz/dy,  Ey <- dHx/dz,  Ez <- dHy/dx     (and Hz <- dEx/dy, ...)
Sub-step 2 (n+1/2 -> n+1), implicit derivative pairs (sign -1):
    Ex <- dHy/dz,  Ey <- dHz/dx,  Ez <- dHx/dy
The other derivative of each equation is explicit (previous half step).
Eliminating H from each implicit pair gives tridiagonal systems along grid
lines, solved with a pre-factorised Thomas algorithm.

Media: eps_r, sigma (electric loss) on E edges, mu_r, sigma_m (magnetic
loss) on H faces. All losses are diagonal and time-averaged in both
sub-steps. With A' = A - S/2, B' = B - S/2 (A, B skew in the volume- and
eps/mu-weighted energy norm, S >= 0) every sub-step is dissipative in the
same norm: the scheme is unconditionally stable on non-uniform grids too.

Derivatives on the non-uniform grid:
    dH/dw at an E node   : (H[j] - H[j-1]) / dd[j]   (dual spacing)
    dE/dw at an H point  : (E[j+1] - E[j]) / dp[j]   (primal spacing)

PML
---
Graded impedance-matched conductive layer (sigma_e/eps = sigma_m/mu, also
inside dielectric/magnetic media):
    uniaxial (default): in a layer normal to axis d only the tangential
                        components (c != d) are lossy.
    isotropic         : every component is lossy.
(CPML/split-field PMLs lose the common energy norm under ADI and grow
exponentially at large time steps, so they are not used.)

Port
----
Lumped resistive source on the port edges:
    eps dE/dt + sigma_p E = curl H + sigma_p * E_s(t)
sigma_p gives a total port resistance R. A matched R absorbs returning waves.
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import GEO_PORT, Materials
from .inputfile import SimConfig, finalize
from .meshfile import read_mesh
from .waveforms import make_waveform

EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6
C0 = 1.0 / math.sqrt(EPS0 * MU0)
ETA0 = math.sqrt(MU0 / EPS0)

# (E component, derivative axis, H component, sign)
PAIRS_STEP1 = [(0, 1, 2, +1.0), (1, 2, 0, +1.0), (2, 0, 1, +1.0)]
PAIRS_STEP2 = [(0, 2, 1, -1.0), (1, 0, 2, -1.0), (2, 1, 0, -1.0)]


def _bshape(arr1d, axis):
    s = [1, 1, 1]
    s[axis] = arr1d.size
    return arr1d.reshape(s)


def _mv(a, d):
    """View with axis d moved to the front."""
    return np.moveaxis(a, d, 0)


def _col(v):
    return v.reshape(-1, 1, 1)


class ADIFDTD:
    def __init__(self, cfg: SimConfig, verbose: bool = False):
        if cfg.grid is None:
            finalize(cfg)
        self.cfg = cfg
        self.verbose = verbose
        self.g = g = cfg.grid
        self.dtype = np.dtype(cfg.dtype)
        self.N = g.N
        self.d = g.dmin                       # smallest spacing per axis
        Nx, Ny, Nz = self.N

        self.dt_cfl = 1.0 / (C0 * math.sqrt(sum(1 / v**2 for v in self.d)))
        self.dt = cfg.dt if cfg.dt is not None else cfg.cfl * self.dt_cfl
        self.h = 0.5 * self.dt

        self.eshape = [(Nx, Ny + 1, Nz + 1), (Nx + 1, Ny, Nz + 1), (Nx + 1, Ny + 1, Nz)]
        self.hshape = [(Nx + 1, Ny, Nz), (Nx, Ny + 1, Nz), (Nx, Ny, Nz + 1)]
        self.E = [np.zeros(self.eshape[c], self.dtype) for c in range(3)]
        self.H = [np.zeros(self.hshape[c], self.dtype) for c in range(3)]

        self._build_materials()
        self._build_ports()
        self._build_coefficients()
        self._factorize()
        self.t = 0.0
        self.n = 0

    # ------------------------------------------------------------------ setup
    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _build_materials(self):
        cfg, g = self.cfg, self.g
        M = Materials(g)
        for d in cfg.dielectrics:
            M.add_box_material(d.box, d.props())
        for b in cfg.conductors:
            M.add_box_pec(b)
        meshes = {}
        pec_ops = []
        for kind, obj in cfg.shapes:
            if kind == "mat":
                M.add_box_material(obj.box, obj.props())
            elif kind == "pec":
                pec_ops.append(("box", obj))
            else:
                key = (obj.file, obj.scale, obj.offset)
                if key not in meshes:
                    meshes[key] = read_mesh(obj.file).transformed(obj.scale, obj.offset)
                    self._log(f"mesh {obj.file}\n{meshes[key].summary()}")
                region = meshes[key].find(obj.region)
                if obj.kind == "pec":
                    pec_ops.append(("mesh", (meshes[key], region)))
                else:
                    M.add_mesh_region(meshes[key], region, "material", obj.props, obj.subcell)
        for kind, obj in pec_ops:      # PEC after materials: always wins
            if kind == "box":
                M.add_box_pec(obj)
            else:
                M.add_mesh_region(obj[0], obj[1], "pec", {})
        self.mat = M

        # outer PEC walls: tangential E on the domain faces
        for c in range(3):
            for a in range(3):
                if a != c:
                    sl = [slice(None)] * 3
                    sl[a] = 0
                    M.pec[c][tuple(sl)] = True
                    sl[a] = -1
                    M.pec[c][tuple(sl)] = True
        self.pec = M.pec
        self.eps_e = [M.edge_average(M.eps, c) for c in range(3)]
        self.sig_e = [M.edge_average(M.sig, c) for c in range(3)]
        self.mu_h = [M.face_harmonic(M.mu, c) for c in range(3)]
        self.sigm_h = [M.face_mean(M.sigm, c) for c in range(3)]

        # cell volumes associated with each component (energy norm)
        dp, dd = g.dp, g.dd
        def vol(c, along, across):
            t = [a for a in range(3) if a != c]
            return _bshape(along[c], c) * _bshape(across[t[0]], t[0]) * _bshape(across[t[1]], t[1])
        self.vol_e = [vol(c, dp, dd) for c in range(3)]
        self.vol_h = [vol(c, dd, dp) for c in range(3)]

    def pml_sigma(self, a, half):
        """Graded PML conductivity along axis a at nodes (half=False) or cell centres."""
        g = self.g
        x = g.centers[a] if half else g.nodes[a]
        Lmin, Lmax = self.cfg.pml[2 * a], self.cfg.pml[2 * a + 1]
        m, R0 = self.cfg.pml_order, self.cfg.pml_R0
        sig = np.zeros_like(x)
        xn = g.nodes[a]
        N = g.N[a]
        for L, inner, sgn in ((Lmin, xn[Lmin], -1.0), (Lmax, xn[N - Lmax], +1.0)):
            if L > 0:
                outer = xn[0] if sgn < 0 else xn[-1]
                thick = abs(outer - inner)
                smax = -(m + 1) * math.log(R0) / (2 * ETA0 * thick)
                depth = np.clip(sgn * (x - inner) / thick, 0, 1)
                sig = np.maximum(sig, smax * depth**m)
        return sig

    def _pml_field_sigma(self, c, electric):
        shape = self.eshape[c] if electric else self.hshape[c]
        s = np.zeros(shape)
        axes = range(3) if self.cfg.pml_type == "isotropic" else [a for a in range(3) if a != c]
        for a in axes:
            half = (a == c) if electric else (a != c)
            s = s + _bshape(self.pml_sigma(a, half), a)
        return s

    def _build_ports(self):
        g = self.g
        self.ports = []
        self.sig_port = [np.zeros(self.eshape[c]) for c in range(3)]
        for p in self.cfg.ports:
            c = p.axis
            sl = tuple(slice(lo, hi) if a == c else slice(lo, hi + 1)
                       for a, (lo, hi) in enumerate(p.box.ranges()))
            nL = sl[c].stop - sl[c].start
            ncell = int(np.prod([s.stop - s.start for s in sl]))
            nA = ncell // nL
            t = [a for a in range(3) if a != c]
            length = _bshape(g.dp[c][sl[c]], c)                        # edge lengths
            area = _bshape(g.dd[t[0]][sl[t[0]]], t[0]) * _bshape(g.dd[t[1]][sl[t[1]]], t[1])
            R_edge = p.R * nA / nL
            sig = np.broadcast_to(length / (R_edge * area), tuple(s.stop - s.start for s in sl))
            self.sig_port[c][sl] += sig
            self.pec[c][sl] = False
            self.mat.geo[self.mat._cell_slices(p.box)] = GEO_PORT
            self.ports.append(dict(cfg=p, comp=c, sl=sl, sig=np.array(sig), nL=nL, nA=nA,
                                   wave=make_waveform(p.wave, p.params, self.dt),
                                   lengths=np.broadcast_to(length, sig.shape).copy(),
                                   length=float(g.dp[c][sl[c]].sum())))

    def _build_coefficients(self):
        h = self.h
        self.ca, self.cb, self.da, self.db = [], [], [], []
        for c in range(3):
            eps = EPS0 * self.eps_e[c]
            # PML sigma scaled by eps_r / mu_r keeps sigma/eps = sigma_m/mu in any medium
            sig = (self.sig_e[c] + self.sig_port[c]
                   + self._pml_field_sigma(c, True) * self.eps_e[c])
            k = sig * h / (2 * eps)
            self.ca.append(((1 - k) / (1 + k)).astype(self.dtype))
            self.cb.append(((h / eps) / (1 + k)).astype(self.dtype))
            mu = MU0 * self.mu_h[c]
            sigm = self.sigm_h[c] + self._pml_field_sigma(c, False) * MU0 / EPS0 * self.mu_h[c]
            k = sigm * h / (2 * mu)
            self.da.append(((1 - k) / (1 + k)).astype(self.dtype))
            self.db.append(((h / mu) / (1 + k)).astype(self.dtype))
        self.notpec = [(~m).astype(self.dtype) for m in self.pec]

    def _factorize(self):
        """Thomas factorisation of the 6 tridiagonal systems (implicit axis first)."""
        g = self.g
        self.lu = {}
        for c, d, hc, s in PAIRS_STEP1 + PAIRS_STEP2:
            F = _mv(self.cb[c], d) / _col(g.dd[d])      # (N+1, ...)
            G = _mv(self.db[hc], d) / _col(g.dp[d])     # (N, ...)
            n = F.shape[0]
            a = np.zeros_like(F)
            b = np.ones_like(F)
            cc = np.zeros_like(F)
            a[1:-1] = -F[1:-1] * G[:-1]
            cc[1:-1] = -F[1:-1] * G[1:]
            b[1:-1] = 1 + F[1:-1] * (G[1:] + G[:-1])
            pec = _mv(self.pec[c], d)
            a[pec] = 0.0
            cc[pec] = 0.0
            b[pec] = 1.0
            cp = np.zeros_like(F)
            inv_m = np.ones_like(F)
            for j in range(1, n):
                m = b[j] - a[j] * cp[j - 1]
                inv_m[j] = 1.0 / m
                cp[j] = cc[j] * inv_m[j]
            asc = lambda x: np.ascontiguousarray(x, self.dtype)
            self.lu[(c, d)] = (asc(a), asc(inv_m), asc(cp), asc(F), asc(G))
        self.inv_dd = [_col(1.0 / g.dd[a]).astype(self.dtype) for a in range(3)]
        self.inv_dp = [_col(1.0 / g.dp[a]).astype(self.dtype) for a in range(3)]

    # ------------------------------------------------------------ field access
    def Etot(self, c):
        return self.E[c]

    def Htot(self, c):
        return self.H[c]

    # ------------------------------------------------------------- operators
    def _source(self, c, t):
        src = None
        for p in self.ports:
            if p["comp"] != c:
                continue
            if src is None:
                src = np.zeros(self.eshape[c], dtype=self.dtype)
            es = p["cfg"].sign * p["cfg"].amp * p["wave"](t)
            sl = p["sl"]
            src[sl] += self.cb[c][sl] * p["sig"] * es
        return src

    def _explicit_terms(self, pairs_expl):
        """Explicit derivative terms from the fields of the previous half step."""
        tE, tH = {}, {}
        for c, d, hc, s in pairs_expl:
            Hm = _mv(self.H[hc], d)
            DH = np.zeros(_mv(self.E[c], d).shape, self.dtype)
            DH[1:-1] = (Hm[1:] - Hm[:-1]) * (s * self.inv_dd[d][1:-1])
            tE[c] = np.moveaxis(DH, 0, d)
            Em = _mv(self.E[c], d)
            tH[hc] = np.moveaxis((Em[1:] - Em[:-1]) * (s * self.inv_dp[d]), 0, d)
        return tE, tH

    def _implicit(self, c, d, hc, s, tE, tH, t):
        a, inv_m, cp, F, G = self.lu[(c, d)]
        Hs = self.da[hc] * self.H[hc] + self.db[hc] * tH[hc]       # H without implicit term
        r0 = self.ca[c] * self.E[c] + self.cb[c] * tE[c]
        src = self._source(c, t)
        if src is not None:
            r0 += src

        R = _mv(r0, d)
        Hm = _mv(Hs, d)
        r = np.empty(R.shape, self.dtype)
        r[1:-1] = R[1:-1] + s * F[1:-1] * (Hm[1:] - Hm[:-1])
        r[0] = 0.0
        r[-1] = 0.0
        r *= _mv(self.notpec[c], d)

        n = r.shape[0]
        x = r
        x[0] *= inv_m[0]
        for j in range(1, n):
            x[j] = (x[j] - a[j] * x[j - 1]) * inv_m[j]
        for j in range(n - 2, -1, -1):
            x[j] -= cp[j] * x[j + 1]

        _mv(self.H[hc], d)[...] = Hm + s * G * (x[1:] - x[:-1])
        _mv(self.E[c], d)[...] = x

    def _substep(self, pairs_impl, pairs_expl, t_mid):
        tE, tH = self._explicit_terms(pairs_expl)
        for c, d, hc, s in pairs_impl:
            self._implicit(c, d, hc, s, tE, tH, t_mid)

    def step(self):
        t = self.t
        self._substep(PAIRS_STEP1, PAIRS_STEP2, t + 0.25 * self.dt)
        self._substep(PAIRS_STEP2, PAIRS_STEP1, t + 0.75 * self.dt)
        self.n += 1
        self.t = self.n * self.dt

    # ------------------------------------------------------------ diagnostics
    def port_state(self, p):
        """(E_source, mean E on port, port voltage, port current)."""
        c, sl, q = p["comp"], p["sl"], p["cfg"]
        es = q.sign * q.amp * p["wave"](self.t)
        E = self.E[c][sl]
        e_mean = float(E.mean())
        V = float((E * p["lengths"]).sum()) / p["nA"] * q.sign      # mean line integral
        I = (es * p["length"] * q.sign - V) / q.R
        return es, e_mean, V, I

    def energy(self):
        we = sum(0.5 * EPS0 * float(np.sum(self.eps_e[c] * self.vol_e[c] * self.E[c] ** 2))
                 for c in range(3))
        wh = sum(0.5 * MU0 * float(np.sum(self.mu_h[c] * self.vol_h[c] * self.H[c] ** 2))
                 for c in range(3))
        return we + wh

    def summary(self):
        c = self.cfg
        M = self.mat
        lines = ["grid        : " + ("uniform" if self.g.uniform else "non-uniform")]
        lines += ["  " + s for s in self.g.describe()]
        lines += [
            f"dt          : {self.dt:.4e} s  (= {self.dt / self.dt_cfl:.2f} x Yee CFL limit of smallest cell)",
            f"steps       : {c.steps}  (T = {c.steps * self.dt:.4e} s)",
            f"PML (cells) : {c.pml}  type={c.pml_type} order={c.pml_order} R0={c.pml_R0}",
            f"materials   : eps_r {M.eps.min():.3g}..{M.eps.max():.3g}, mu_r {M.mu.min():.3g}..{M.mu.max():.3g}, "
            f"PEC edges {sum(int(m.sum()) for m in self.pec)}",
        ]
        for p in self.ports:
            q = p["cfg"]
            lines.append(f"port {q.name:7s}: E{'xyz'[q.axis]} sign={q.sign:+.0f} amp={q.amp} "
                         f"R={q.R} ohm wave={q.wave} edges={p['nL']}x{p['nA']} L={p['length']:.3e} m")
        return "\n".join(lines)
