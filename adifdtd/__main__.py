"""Command line entry:  python -m adifdtd input.txt"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from .inputfile import parse_input
from .output import Recorder
from .solver import ADIFDTD


def run(path, outdir=None, quiet=False):
    cfg = parse_input(path)
    if outdir:
        cfg.outdir = outdir
    sim = ADIFDTD(cfg, verbose=not quiet)
    if not quiet:
        print(sim.summary())
    os.makedirs(cfg.outdir, exist_ok=True)
    rec = Recorder(sim, cfg.outputs, cfg.outdir)
    rec.sample()

    port_rows = []
    t0 = time.time()
    report = max(1, cfg.steps // 10)
    for _ in range(cfg.steps):
        sim.step()
        rec.sample()
        port_rows.append([sim.t] + [v for p in sim.ports for v in sim.port_state(p)])
        if not quiet and (sim.n % report == 0 or sim.n == cfg.steps):
            el = time.time() - t0
            print(f"step {sim.n:6d}/{cfg.steps}  t={sim.t:.3e} s  W={sim.energy():.3e} J  "
                  f"({el:.1f} s, {el / sim.n * 1e3:.1f} ms/step)")
        if not np.isfinite(sim.Etot(0)).all():
            raise FloatingPointError("field diverged")

    if sim.ports:
        hdr = ["t"]
        for p in sim.ports:
            n = p["cfg"].name
            hdr += [f"{n}_Esrc", f"{n}_Eport", f"{n}_V", f"{n}_I"]
        np.savetxt(os.path.join(cfg.outdir, "ports.csv"), np.array(port_rows),
                   delimiter=",", header=",".join(hdr), comments="")
    files = rec.write()
    if not quiet:
        for f in files:
            print("wrote", f)
        if sim.ports:
            print("wrote", os.path.join(cfg.outdir, "ports.csv"))
    return sim


def main():
    ap = argparse.ArgumentParser(description="3D ADI-FDTD solver with PML and resistive ports")
    ap.add_argument("input", help="input text file")
    ap.add_argument("-o", "--outdir", help="override output directory")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args()
    run(a.input, a.outdir, a.quiet)


if __name__ == "__main__":
    main()
