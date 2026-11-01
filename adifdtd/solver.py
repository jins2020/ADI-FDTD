"""3D ADI-FDTD solver with a graded matched-loss PML and matched resistive ports.

Scheme
------
Zheng/Namiki ADI (Peaceman-Rachford) splitting of the curl operator.

Sub-step 1 (n -> n+1/2), implicit derivative pairs (sign +1):
    Ex <- dHz/dy,  Ey <- dHx/dz,  Ez <- dHy/dx     (and Hz <- dEx/dy, ...)
Sub-step 2 (n+1/2 -> n+1), implicit derivative pairs (sign -1):
    Ex <- dHy/dz,  Ey <- dHz/dx,  Ez <- dHx/dy
The other derivative of each equation is explicit (previous half step).
Eliminating H from each implicit pair gives tridiagonal systems, solved with
a pre-factorised (Thomas) algorithm along each grid line.

All losses (PML, dielectric conductivity, port resistance) are diagonal and
time-averaged (Crank-Nicolson) in both sub-steps. With
    A' = A - S/2,  B' = B - S/2      (A, B skew in the eps/mu energy norm, S >= 0)
each sub-step operator is dissipative in the *same* norm, so the scheme is
unconditionally stable for any dt.

PML
---
Graded, impedance-matched conductive layer (sigma_e/eps = sigma_m/mu):
    uniaxial  (default): in a layer normal to axis d, only the field
                         components tangential to the layer (c != d) are lossy.
                         This is exact for normal incidence and equals the
                         UPML tensor apart from its 1/s_d normal component.
    isotropic          : every component is lossy.
Unsplit stretched-coordinate (CPML) or split-field (Berenger) PMLs are not
used: combined with ADI their split operators lose the common energy norm
and grow exponentially at large time steps.

Grid (Yee), Nx*Ny*Nz cells, nodes 0..N:
    Ex (Nx, Ny+1, Nz+1)  Ey (Nx+1, Ny, Nz+1)  Ez (Nx+1, Ny+1, Nz)
    Hx (Nx+1, Ny, Nz)    Hy (Nx, Ny+1, Nz)    Hz (Nx, Ny, Nz+1)
Outer boundary is PEC (behind the PML).

Port
----
Lumped resistive source. On the port edges
    eps dE/dt + sigma_p E = curl H + sigma_p * E_s(t)
with sigma_p chosen such that the total port resistance is R. A port whose
R matches the line impedance absorbs the returning wave (no re-reflection).
"""
from __future__ import annotations

import math

import numpy as np

from .inputfile import SimConfig
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


class ADIFDTD:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.dtype = np.dtype(cfg.dtype)
        self.N = tuple(cfg.shape)
        self.d = tuple(float(v) for v in cfg.cell)
        Nx, Ny, Nz = self.N
        dx, dy, dz = self.d

        self.dt_cfl = 1.0 / (C0 * math.sqrt(1 / dx**2 + 1 / dy**2 + 1 / dz**2))
        self.dt = cfg.dt if cfg.dt is not None else cfg.cfl * self.dt_cfl
        self.h = 0.5 * self.dt   # sub-step length

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
    def _build_materials(self):
        cfg, N = self.cfg, self.N
        eps_cell = np.ones(N)
        sig_cell = np.zeros(N)
        for diel in cfg.dielectrics:
            (i0, i1), (j0, j1), (k0, k1) = diel.box.ranges()
            eps_cell[i0:i1, j0:j1, k0:k1] = diel.eps_r
            sig_cell[i0:i1, j0:j1, k0:k1] = diel.sigma
        self.eps_cell, self.sig_cell = eps_cell, sig_cell

        # edge values = average of the 4 cells sharing the edge
        self.eps_e, self.sig_e, self.pec = [], [], []
        for c in range(3):
            pad = [(0, 0) if a == c else (1, 1) for a in range(3)]
            pe = np.pad(eps_cell, pad, mode="edge")
            ps = np.pad(sig_cell, pad, mode="edge")
            t = [a for a in range(3) if a != c]
            acc_e = np.zeros(self.eshape[c])
            acc_s = np.zeros(self.eshape[c])
            for o1 in (0, 1):
                for o2 in (0, 1):
                    sl = [slice(None)] * 3
                    sl[t[0]] = slice(o1, o1 + self.eshape[c][t[0]])
                    sl[t[1]] = slice(o2, o2 + self.eshape[c][t[1]])
                    acc_e += pe[tuple(sl)]
                    acc_s += ps[tuple(sl)]
            self.eps_e.append(acc_e / 4.0)
            self.sig_e.append(acc_s / 4.0)

            mask = np.zeros(self.eshape[c], dtype=bool)
            for a in t:                        # outer PEC walls (tangential E)
                sl = [slice(None)] * 3
                sl[a] = 0
                mask[tuple(sl)] = True
                sl[a] = -1
                mask[tuple(sl)] = True
            for b in cfg.conductors:
                sl = []
                for a, (lo, hi) in enumerate(b.ranges()):
                    # edge centre (i+1/2 along c, node otherwise) inside [lo, hi]
                    sl.append(slice(lo, hi) if a == c else slice(lo, hi + 1))
                mask[tuple(sl)] = True
            self.pec.append(mask)

    def pml_sigma(self, a, half):
        """Graded PML conductivity along axis a (at nodes or half nodes)."""
        Na, da = self.N[a], self.d[a]
        Lmin, Lmax = self.cfg.pml[2 * a], self.cfg.pml[2 * a + 1]
        m, R0 = self.cfg.pml_order, self.cfg.pml_R0
        u = np.arange(Na) + 0.5 if half else np.arange(Na + 1).astype(float)
        sig = np.zeros_like(u)
        for L, depth in ((Lmin, Lmin - u), (Lmax, u - (Na - Lmax))):
            if L > 0:
                smax = -(m + 1) * math.log(R0) / (2 * ETA0 * L * da)
                sig = np.maximum(sig, smax * np.clip(depth / L, 0, 1) ** m)
        return sig

    def _pml_field_sigma(self, c, electric):
        """PML conductivity (vacuum value) on the grid of component c."""
        shape = self.eshape[c] if electric else self.hshape[c]
        s = np.zeros(shape)
        axes = range(3) if self.cfg.pml_type == "isotropic" else [a for a in range(3) if a != c]
        for a in axes:
            # E_c: half-integer along c, integer otherwise; H_c: the opposite
            half = (a == c) if electric else (a != c)
            s = s + _bshape(self.pml_sigma(a, half), a)
        return s

    def _build_ports(self):
        self.ports = []
        self.sig_port = [np.zeros(self.eshape[c]) for c in range(3)]
        dx = self.d
        for p in self.cfg.ports:
            c = p.axis
            sl = tuple(slice(lo, hi) if a == c else slice(lo, hi + 1)
                       for a, (lo, hi) in enumerate(p.box.ranges()))
            nL = p.box.ranges()[c][1] - p.box.ranges()[c][0]
            ncell = int(np.prod([s.stop - s.start for s in sl]))
            nA = ncell // nL
            area = np.prod([dx[a] for a in range(3) if a != c])
            R_edge = p.R * nA / nL           # nA parallel strings of nL edges
            sig = dx[c] / (R_edge * area)
            self.sig_port[c][sl] += sig
            self.pec[c][sl] = False
            self.ports.append(dict(cfg=p, comp=c, sl=sl, sig=sig, nL=nL, nA=nA,
                                   wave=make_waveform(p.wave, p.params, self.dt),
                                   length=nL * dx[c]))

    def _build_coefficients(self):
        h = self.h
        self.ca, self.cb, self.da, self.db = [], [], [], []
        for c in range(3):
            eps = EPS0 * self.eps_e[c]
            # PML sigma scaled by eps_r keeps sigma/eps = sigma*/mu0 in dielectrics
            sig = self.sig_e[c] + self.sig_port[c] + self._pml_field_sigma(c, True) * self.eps_e[c]
            k = sig * h / (2 * eps)
            self.ca.append(((1 - k) / (1 + k)).astype(self.dtype))
            self.cb.append(((h / eps) / (1 + k)).astype(self.dtype))
            sigm = self._pml_field_sigma(c, False) * MU0 / EPS0
            k = sigm * h / (2 * MU0)
            self.da.append(((1 - k) / (1 + k)).astype(self.dtype))
            self.db.append(((h / MU0) / (1 + k)).astype(self.dtype))
        self.notpec = [(~m).astype(self.dtype) for m in self.pec]

    def _factorize(self):
        """Thomas factorisation of the 6 tridiagonal systems (implicit axis first)."""
        self.lu = {}
        for c, d, hc, s in PAIRS_STEP1 + PAIRS_STEP2:
            F = _mv(self.cb[c], d) / self.d[d]        # (N+1, ...)
            G = _mv(self.db[hc], d) / self.d[d]       # (N, ...)
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
            dd = self.d[d]
            Hm = _mv(self.H[hc], d)
            DH = np.zeros(_mv(self.E[c], d).shape, self.dtype)
            DH[1:-1] = (Hm[1:] - Hm[:-1]) * (s / dd)
            tE[c] = np.moveaxis(DH, 0, d)
            Em = _mv(self.E[c], d)
            tH[hc] = np.moveaxis((Em[1:] - Em[:-1]) * (s / dd), 0, d)
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
        e_mean = float(self.E[c][sl].mean())
        V = e_mean * p["length"] * q.sign
        I = (es - e_mean) * q.sign * p["length"] / q.R
        return es, e_mean, V, I

    def energy(self):
        vol = self.d[0] * self.d[1] * self.d[2]
        we = sum(0.5 * EPS0 * float(np.sum(self.eps_e[c] * self.E[c] ** 2)) for c in range(3))
        wh = sum(0.5 * MU0 * float(np.sum(self.H[c] ** 2)) for c in range(3))
        return (we + wh) * vol

    def summary(self):
        Nx, Ny, Nz = self.N
        c = self.cfg
        lines = [
            f"grid        : {Nx} x {Ny} x {Nz} cells",
            f"cell        : {self.d[0]:.3e} x {self.d[1]:.3e} x {self.d[2]:.3e} m",
            f"dt          : {self.dt:.4e} s  (= {self.dt / self.dt_cfl:.2f} x Yee CFL limit)",
            f"steps       : {c.steps}  (T = {c.steps * self.dt:.4e} s)",
            f"PML (cells) : {c.pml}  type={c.pml_type} order={c.pml_order} R0={c.pml_R0}",
            f"conductors  : {len(c.conductors)}   dielectrics: {len(c.dielectrics)}",
        ]
        for p in self.ports:
            q = p["cfg"]
            lines.append(f"port {q.name:7s}: E{'xyz'[q.axis]} sign={q.sign:+.0f} amp={q.amp} "
                         f"R={q.R} ohm wave={q.wave} edges={p['nL']}x{p['nA']}")
        return "\n".join(lines)
