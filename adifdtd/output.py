"""Field snapshots on a plane and GIF writing."""
from __future__ import annotations

import os

import numpy as np

AX = {"x": 0, "y": 1, "z": 2}


def e_center(sim, c):
    """E component c interpolated to cell centres, shape (Nx, Ny, Nz)."""
    F = sim.Etot(c)
    t = [a for a in range(3) if a != c]
    out = 0.0
    for o1 in (0, 1):
        for o2 in (0, 1):
            sl = [slice(None)] * 3
            sl[t[0]] = slice(o1, o1 + sim.N[t[0]])
            sl[t[1]] = slice(o2, o2 + sim.N[t[1]])
            out = out + F[tuple(sl)]
    return out / 4.0


def h_center(sim, c):
    F = sim.Htot(c)
    sl0 = [slice(None)] * 3
    sl1 = [slice(None)] * 3
    sl0[c] = slice(0, -1)
    sl1[c] = slice(1, None)
    return 0.5 * (F[tuple(sl0)] + F[tuple(sl1)])


def _plane_slice(arr, plane, index):
    normal = AX[({"x", "y", "z"} - set(plane)).pop()]
    sl = [slice(None)] * 3
    sl[normal] = index
    return arr[tuple(sl)]       # remaining axes in increasing order


class Recorder:
    def __init__(self, sim, outputs, outdir):
        self.sim = sim
        self.outputs = outputs
        self.outdir = outdir
        self.frames = [[] for _ in outputs]
        self.times = [[] for _ in outputs]

    def sample(self):
        sim = self.sim
        for k, o in enumerate(self.outputs):
            if sim.n % o.interval:
                continue
            fn = e_center if o.field == "E" else h_center
            if o.comp == "mag":
                v = np.sqrt(sum(fn(sim, c) ** 2 for c in range(3)))
            else:
                v = fn(sim, AX[o.comp])
            self.frames[k].append(np.array(_plane_slice(v, o.plane, o.index), dtype=np.float32))
            self.times[k].append(sim.t)

    def write(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter

        os.makedirs(self.outdir, exist_ok=True)
        sim = self.sim
        files = []
        geo = geometry_map(sim)
        for k, o in enumerate(self.outputs):
            frames = self.frames[k]
            if not frames:
                continue
            stack = np.stack(frames)
            # default colour limit: 99.5th percentile (robust against source hot spots)
            vmax = o.clim if o.clim else float(np.percentile(np.abs(stack), 99.5)) or 1.0
            if o.comp == "mag":
                cmap, vmin = "inferno", 0.0
            else:
                cmap, vmin = "RdBu_r", -vmax
            a1, a2 = o.plane[0], o.plane[1]
            g = _plane_slice(geo, o.plane, o.index)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(stack[0].T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                           interpolation="nearest", aspect=sim.d[AX[a2]] / sim.d[AX[a1]])
            unit = "V/m" if o.field == "E" else "A/m"
            label = f"|{o.field}|" if o.comp == "mag" else f"{o.field}{o.comp}"
            fig.colorbar(im, ax=ax, label=f"{label} [{unit}]")
            # geometry: conductor (1), dielectric (2), port (3)
            for lvl, col in ((1, "k"), (2, "lime"), (3, "cyan")):
                m = (g == lvl).T.astype(float)
                if m.any():
                    ax.contour(m, levels=[0.5], colors=col, linewidths=0.8)
            _draw_pml(ax, sim, a1, a2)
            ax.set_xlabel(f"{a1} [cell]")
            ax.set_ylabel(f"{a2} [cell]")
            normal = ({"x", "y", "z"} - set(o.plane)).pop()
            title = ax.set_title("")

            def update(i):
                im.set_data(stack[i].T)
                title.set_text(f"{label}  {o.plane}-plane {normal}={o.index}  "
                               f"t={self.times[k][i]:.3e} s")
                return im, title

            anim = FuncAnimation(fig, update, frames=len(stack), blit=False)
            path = os.path.join(self.outdir, o.file)
            anim.save(path, writer=PillowWriter(fps=o.fps))
            plt.close(fig)
            files.append(path)
        return files


def _draw_pml(ax, sim, a1, a2):
    p = sim.cfg.pml
    i1, i2 = AX[a1], AX[a2]
    lo1, hi1 = p[2 * i1], sim.N[i1] - p[2 * i1 + 1]
    lo2, hi2 = p[2 * i2], sim.N[i2] - p[2 * i2 + 1]
    xs = [lo1 - 0.5, hi1 - 0.5, hi1 - 0.5, lo1 - 0.5, lo1 - 0.5]
    ys = [lo2 - 0.5, lo2 - 0.5, hi2 - 0.5, hi2 - 0.5, lo2 - 0.5]
    ax.plot(xs, ys, "w--", lw=0.7)


def geometry_map(sim):
    """Cell-centred integer map: 0 vacuum, 1 conductor, 2 dielectric, 3 port."""
    g = np.zeros(sim.N, dtype=np.int8)

    def cells(box):
        sl = []
        for a, (lo, hi) in enumerate(box.ranges()):
            if hi > lo:
                sl.append(slice(lo, hi))
            else:                       # zero-thickness sheet: mark one cell layer
                lo = min(lo, sim.N[a] - 1)
                sl.append(slice(lo, lo + 1))
        return tuple(sl)

    for d in sim.cfg.dielectrics:
        g[cells(d.box)] = 2
    for b in sim.cfg.conductors:
        g[cells(b)] = 1
    for p in sim.cfg.ports:
        g[cells(p.box)] = 3
    return g
