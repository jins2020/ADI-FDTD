"""Mesh reader / voxeliser checks: `python tests/test_mesh.py`."""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from adifdtd.geometry import Materials  # noqa: E402
from adifdtd.grid import Grid, build_axis  # noqa: E402
from adifdtd.meshfile import read_mesh  # noqa: E402

# unit cube [0,1]^3 split into 5 tetrahedra + a triangle sheet + a line, gmsh v2.2
MSH22 = """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
3
1 7 "wire"
2 8 "sheet"
3 5 "cube"
$EndPhysicalNames
$Nodes
11
1 0 0 0
2 1 0 0
3 0 1 0
4 1 1 0
5 0 0 1
6 1 0 1
7 0 1 1
8 1 1 1
9 0 0 2
10 1 0 2
11 0 1 2
$EndNodes
$Elements
8
1 4 2 5 1 1 2 3 5
2 4 2 5 1 2 4 3 8
3 4 2 5 1 2 5 6 8
4 4 2 5 1 3 5 7 8
5 4 2 5 1 2 3 5 8
6 2 2 8 2 9 10 11
7 1 2 7 3 1 9
8 15 2 9 4 1
$EndElements
"""


def _cube_mesh():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "cube.msh")
    with open(p, "w") as fh:
        fh.write(MSH22)
    return read_mesh(p)


def test_gmsh22_regions():
    m = _cube_mesh()
    t, s, l = m.find("cube").arrays()
    assert len(t) == 5 and len(m.find("sheet").arrays()[1]) == 1
    assert len(m.find(7).arrays()[2]) == 1


def test_tet_volume_fraction():
    m = _cube_mesh().transformed(1e-3, (2e-3, 2e-3, 2e-3))        # cube 2..3 mm
    g = Grid.uniform_grid((10, 10, 10), (0.5e-3,) * 3)
    M = Materials(g)
    M.add_mesh_region(m, m.find("cube"), "material", dict(eps=3.0, mu=2.0), subcell=2)
    vol = (M.eps - 1).sum() / 2 * 0.5e-3 ** 3
    assert abs(vol - 1e-9) < 1e-12, vol
    assert np.isclose(M.mu.max(), 2.0)


def test_pec_volume_and_sheet():
    m = _cube_mesh().transformed(1e-3, (1e-3, 1e-3, 1e-3))
    g = Grid.uniform_grid((6, 6, 6), (1e-3,) * 3)
    M = Materials(g)
    M.add_mesh_region(m, m.find("cube"), "pec", {})
    # closed PEC cube [1,2]mm: Ex edges with x-centre 1.5 and y,z nodes in {1,2}
    assert M.pec[0][1, 1:3, 1:3].all() and M.pec[0].sum() == 4
    M2 = Materials(g)
    M2.add_mesh_region(m, m.find("sheet"), "pec", {})
    # triangle (1,1)-(2,1)-(1,2) mm at z=3 mm: legs become PEC edges, no normal edges
    assert M2.pec[0][1, 1, 3] and M2.pec[1][1, 1, 3] and M2.pec[2].sum() == 0


def test_example_meshes():
    m = read_mesh(os.path.join(ROOT, "examples", "mesh_patch", "patch.msh"))
    assert {r.name for r in m.regions.values()} == {"substrate", "ground", "patch", "probe"}
    s = read_mesh(os.path.join(ROOT, "examples", "mesh_sphere", "sphere.stl")).transformed(1e-3)
    g = Grid.uniform_grid((24, 24, 24), (1e-3,) * 3, (-12e-3,) * 3)
    M = Materials(g)
    M.add_mesh_region(s, s.find(1), "material", dict(eps=2.0), subcell=2)
    vol = (M.eps - 1).sum() * 1e-9
    assert abs(vol / (4 / 3 * np.pi * 8e-3 ** 3) - 1) < 0.02


def test_graded_axis():
    x = build_axis((0.0, [(1.0, 10, 1.0), (3.0, 5, 4.0)]))
    d = np.diff(x)
    assert len(x) == 16 and np.isclose(x[-1], 3.0) and np.isclose(d[-1] / d[10], 4.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all mesh tests passed")
