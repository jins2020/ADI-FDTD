"""Time waveforms for port excitation (dimensionless, peak ~ 1)."""
from __future__ import annotations

import math


def make_waveform(name: str, p: dict, dt: float):
    name = name.lower()
    if name == "gauss":
        tau = p.get("tau", 30 * dt)
        t0 = p.get("t0", 4 * tau)
        return lambda t: math.exp(-((t - t0) / tau) ** 2)
    if name in ("modgauss", "gausssin", "pulse"):
        f0 = p["f0"]
        tau = p.get("tau", 1.0 / f0)
        t0 = p.get("t0", 4 * tau)
        return lambda t: math.exp(-((t - t0) / tau) ** 2) * math.sin(2 * math.pi * f0 * (t - t0))
    if name in ("sin", "sine", "cw"):
        f0 = p["f0"]
        ramp = p.get("ramp", 3.0) / f0   # smooth turn-on time
        def w(t):
            env = 1.0 if t >= ramp else 0.5 * (1 - math.cos(math.pi * max(t, 0.0) / ramp))
            return env * math.sin(2 * math.pi * f0 * t)
        return w
    if name == "ricker":
        f0 = p["f0"]
        t0 = p.get("t0", 1.5 / f0)
        def w(t):
            a = (math.pi * f0 * (t - t0)) ** 2
            return (1 - 2 * a) * math.exp(-a)
        return w
    raise ValueError(f"unknown waveform '{name}' (gauss, modgauss, sin, ricker)")
