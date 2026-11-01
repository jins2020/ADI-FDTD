"""Basic checks: run with `python -m pytest tests` or `python tests/test_solver.py`."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adifdtd.inputfile import SimConfig, Box, Dielectric, Port, parse_input  # noqa: E402
from adifdtd.solver import ADIFDTD  # noqa: E402


def _spectral_radius(cfg):
    """Spectral radius of the one-step amplification matrix (source off)."""
    s = ADIFDTD(cfg)
    L = s.E + s.H
    n = sum(a.size for a in L)
    M = np.zeros((n, n))
    for k in range(n):
        v = np.zeros(n)
        v[k] = 1.0
        o = 0
        for a in L:
            a.ravel()[:] = v[o:o + a.size]
            o += a.size
        s.step()
        M[:, k] = np.concatenate([a.ravel() for a in L])
    return np.abs(np.linalg.eigvals(M)).max()


def test_unconditional_stability():
    """|lambda| <= 1 for large time steps with PML, dielectric and a port."""
    for cfl in (1.0, 10.0, 100.0):
        cfg = SimConfig(shape=(8, 6, 4), cell=(1e-3, 1.3e-3, 0.7e-3), cfl=cfl,
                        pml=(2, 2, 1, 0, 0, 1),
                        dielectrics=[Dielectric(Box(1, 5, 0, 4, 1, 3), 4.0, 0.05)],
                        conductors=[Box(6, 6, 1, 4, 0, 2)],
                        ports=[Port(Box(4, 4, 3, 3, 1, 3), 2, 1.0, 0.0, 50.0, "gauss", {}, "p")])
        rho = _spectral_radius(cfg)
        assert rho < 1 + 1e-9, (cfl, rho)


def test_pec_and_port_run(tmp_path=None):
    here = os.path.dirname(__file__)
    cfg = parse_input(os.path.join(here, "..", "examples", "dipole.txt"))
    cfg.shape = (24, 24, 24)
    cfg.pml = (6,) * 6
    cfg.conductors = [Box(12, 12, 12, 12, 7, 11), Box(12, 12, 12, 12, 13, 17)]
    cfg.ports[0].box = Box(12, 12, 12, 12, 11, 13)
    s = ADIFDTD(cfg)
    for _ in range(60):
        s.step()
    assert np.isfinite(s.energy())
    # tangential E on the wire is zero
    assert np.all(s.E[2][12, 12, 7:11] == 0)
    assert s.energy() > 0


if __name__ == "__main__":
    test_unconditional_stability()
    test_pec_and_port_run()
    print("all tests passed")
