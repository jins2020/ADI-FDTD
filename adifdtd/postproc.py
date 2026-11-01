"""Port post-processing: reflection coefficient S11 from ports.csv.

    python -m adifdtd.postproc output/ports.csv [--port feed] [--z0 50] [--fmax 6e9]

Incident / reflected power waves at the port terminals (V, I from ports.csv):
    a = (V + Z0 I) / (2 sqrt(Z0)),  b = (V - Z0 I) / (2 sqrt(Z0)),  S11 = B(f)/A(f)
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def s11(csv_path, port=None, z0=50.0, fmin=0.0, fmax=None, pad=8):
    with open(csv_path) as fh:
        hdr = fh.readline().strip().split(",")
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    t = data[:, 0]
    names = sorted({h.rsplit("_", 1)[0] for h in hdr[1:]})
    port = port or names[0]
    V = data[:, hdr.index(f"{port}_V")]
    I = data[:, hdr.index(f"{port}_I")]
    dt = t[1] - t[0]
    a = (V + z0 * I) / (2 * np.sqrt(z0))
    b = (V - z0 * I) / (2 * np.sqrt(z0))
    n = pad * len(t)
    f = np.fft.rfftfreq(n, dt)
    A = np.fft.rfft(a, n)
    B = np.fft.rfft(b, n)
    fmax = fmax or f[-1]
    keep = (f >= fmin) & (f <= fmax) & (np.abs(A) > 1e-3 * np.abs(A).max())
    return f[keep], B[keep] / A[keep], port


def main():
    ap = argparse.ArgumentParser(description="S11 from ports.csv")
    ap.add_argument("csv")
    ap.add_argument("--port")
    ap.add_argument("--z0", type=float, default=50.0)
    ap.add_argument("--fmin", type=float, default=0.0)
    ap.add_argument("--fmax", type=float)
    a = ap.parse_args()
    f, S, port = s11(a.csv, a.port, a.z0, a.fmin, a.fmax)
    out = os.path.join(os.path.dirname(a.csv), f"S11_{port}")
    np.savetxt(out + ".csv", np.c_[f, S.real, S.imag, 20 * np.log10(np.abs(S))],
               delimiter=",", header="f,Re,Im,dB", comments="")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(f / 1e9, 20 * np.log10(np.abs(S)))
    ax.set_xlabel("f [GHz]")
    ax.set_ylabel("|S11| [dB]")
    ax.set_title(f"port {port}, Z0 = {a.z0:g} ohm")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=110)
    k = np.argmin(np.abs(S))
    print(f"min |S11| = {20 * np.log10(np.abs(S[k])):.1f} dB at {f[k] / 1e9:.3f} GHz")
    print("wrote", out + ".csv", out + ".png")


if __name__ == "__main__":
    main()
