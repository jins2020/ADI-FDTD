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
            i1, i2 = AX[a1], AX[a2]
            g = _plane_slice(sim.mat.geo, o.plane, o.index)
            grid = sim.g
            X = grid.nodes[i1] * 1e3
            Y = grid.nodes[i2] * 1e3
            xc = grid.centers[i1] * 1e3
            yc = grid.centers[i2] * 1e3
            w = X[-1] - X[0]
            hgt = Y[-1] - Y[0]
            fw = 6.5
            fig, ax = plt.subplots(figsize=(fw, max(2.5, min(9.0, fw * hgt / w * 0.85 + 0.6))))
            im = ax.pcolormesh(X, Y, stack[0].T, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
            ax.set_aspect("equal")
            unit = "V/m" if o.field == "E" else "A/m"
            label = f"|{o.field}|" if o.comp == "mag" else f"{o.field}{o.comp}"
            fig.colorbar(im, ax=ax, label=f"{label} [{unit}]", shrink=0.85)
            # geometry: conductor (black), material (lime), port (cyan)
            for lvl, col in ((1, "k"), (2, "lime"), (3, "cyan")):
                m = (g == lvl).T.astype(float)
                if m.any() and min(m.shape) > 1:
                    ax.contour(xc, yc, m, levels=[0.5], colors=col, linewidths=0.8)
            _draw_pml(ax, sim, i1, i2)
            if not grid.uniform and o.grid_lines:
                for v in X:
                    ax.axvline(v, color="0.5", lw=0.15)
                for v in Y:
                    ax.axhline(v, color="0.5", lw=0.15)
            ax.set_xlabel(f"{a1} [mm]")
            ax.set_ylabel(f"{a2} [mm]")
            normal = ({"x", "y", "z"} - set(o.plane)).pop()
            npos = grid.centers[AX[normal]][o.index] * 1e3
            title = ax.set_title("", fontsize=10)
            fig.tight_layout()

            def update(i):
                im.set_array(stack[i].T.ravel())
                title.set_text(f"{label}  {o.plane}-plane {normal}={npos:.2f} mm  "
                               f"t={self.times[k][i]:.3e} s")
                return im, title

            anim = FuncAnimation(fig, update, frames=len(stack), blit=False)
            path = os.path.join(self.outdir, o.file)
            anim.save(path, writer=PillowWriter(fps=o.fps))
            plt.close(fig)
            files.append(path)
        return files


def _draw_pml(ax, sim, i1, i2):
    p = sim.cfg.pml
    g = sim.g
    x = g.nodes[i1] * 1e3
    y = g.nodes[i2] * 1e3
    lo1, hi1 = x[p[2 * i1]], x[g.N[i1] - p[2 * i1 + 1]]
    lo2, hi2 = y[p[2 * i2]], y[g.N[i2] - p[2 * i2 + 1]]
    ax.plot([lo1, hi1, hi1, lo1, lo1], [lo2, lo2, hi2, hi2, lo2], "--", color="0.55", lw=0.7)
