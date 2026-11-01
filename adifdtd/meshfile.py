"""Readers for external unstructured meshes.

Supported formats
-----------------
* Gmsh ``.msh`` ASCII, version 2.2 and 4.1
    - 4-node tetrahedra  (type 4)  -> volume regions
    - 3-node triangles   (type 2)  -> surface regions (thin sheets)
    - 2-node lines       (type 1)  -> wire regions
  Regions are the physical groups (number or name from $PhysicalNames).
  In v4.1 files without physical groups the elementary entity tag is used.
* STL, ASCII or binary: one closed surface = one volume region (tag 1).

Second-order elements are accepted; only their corner nodes are used.
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

import numpy as np

# gmsh element type -> (kind, number of nodes, number of corner nodes)
_GMSH_TYPES = {
    1: ("line", 2, 2), 8: ("line", 3, 2),
    2: ("tri", 3, 3), 9: ("tri", 6, 3),
    4: ("tet", 4, 4), 11: ("tet", 10, 4),
    15: ("point", 1, 1),
}
_GMSH_NNODES = {1: 2, 2: 3, 3: 4, 4: 4, 5: 8, 6: 6, 7: 5, 8: 3, 9: 6, 10: 9, 11: 10,
                12: 27, 13: 18, 14: 14, 15: 1, 16: 8, 17: 20, 18: 15, 19: 13}


@dataclass
class Region:
    tag: int
    name: str = ""
    dim: int = 3
    tets: list = field(default_factory=list)
    tris: list = field(default_factory=list)
    lines: list = field(default_factory=list)

    def arrays(self):
        f = lambda L, n: np.array(L, dtype=np.int64).reshape(-1, n)
        return f(self.tets, 4), f(self.tris, 3), f(self.lines, 2)


@dataclass
class Mesh:
    nodes: np.ndarray                      # (n, 3) coordinates
    regions: dict                          # tag -> Region
    source: str = ""

    def find(self, key):
        """Region by physical tag (int or numeric string) or by name."""
        if isinstance(key, str) and not key.lstrip("-").isdigit():
            for r in self.regions.values():
                if r.name == key:
                    return r
            raise KeyError(f"region '{key}' not in {self.source} "
                           f"(available: {[r.name or r.tag for r in self.regions.values()]})")
        tag = int(key)
        if tag not in self.regions:
            raise KeyError(f"region {tag} not in {self.source} (available: {sorted(self.regions)})")
        return self.regions[tag]

    def transformed(self, scale=1.0, offset=(0.0, 0.0, 0.0)):
        return Mesh(self.nodes * scale + np.asarray(offset, float), self.regions, self.source)

    def summary(self):
        out = []
        for r in self.regions.values():
            t, s, l = r.arrays()
            out.append(f"  region {r.tag:3d} {r.name!r:16s} tets={len(t)} tris={len(s)} lines={len(l)}")
        return "\n".join(out)


def read_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".msh":
        return read_gmsh(path)
    if ext == ".stl":
        return read_stl(path)
    raise ValueError(f"unsupported mesh format '{ext}' (use .msh or .stl)")


# ----------------------------------------------------------------------- gmsh
def _sections(lines):
    sec, i = {}, 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("$") and not ln.startswith("$End"):
            name = ln[1:]
            j = i + 1
            while not lines[j].strip().startswith("$End" + name):
                j += 1
            sec[name] = lines[i + 1:j]
            i = j
        i += 1
    return sec


def read_gmsh(path):
    with open(path, "rb") as fh:
        head = fh.read(200)
    if b"$MeshFormat" not in head:
        raise ValueError(f"{path}: not a gmsh file")
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    sec = _sections(lines)
    ver, ftype = sec["MeshFormat"][0].split()[:2]
    if int(ftype) != 0:
        raise ValueError(f"{path}: binary .msh is not supported, save as ASCII (Mesh.Binary = 0)")
    names = {}
    if "PhysicalNames" in sec:
        for ln in sec["PhysicalNames"][1:]:
            p = ln.split(maxsplit=2)
            names[(int(p[0]), int(p[1]))] = p[2].strip().strip('"')
    if ver.startswith("2"):
        nodes, regions = _gmsh2(sec, names)
    elif ver.startswith("4"):
        nodes, regions = _gmsh4(sec, names)
    else:
        raise ValueError(f"{path}: unsupported gmsh version {ver}")
    return Mesh(nodes, regions, path)


def _add(regions, tag, dim, names, kind, conn):
    r = regions.get(tag)
    if r is None:
        r = regions[tag] = Region(tag, names.get((dim, tag), ""), dim)
    {"tet": r.tets, "tri": r.tris, "line": r.lines}[kind].append(conn)


def _gmsh2(sec, names):
    nl = sec["Nodes"]
    n = int(nl[0])
    ids = np.empty(n, dtype=np.int64)
    xyz = np.empty((n, 3))
    for k, ln in enumerate(nl[1:n + 1]):
        p = ln.split()
        ids[k] = int(p[0])
        xyz[k] = [float(v) for v in p[1:4]]
    index = {int(i): k for k, i in enumerate(ids)}
    regions = {}
    el = sec["Elements"]
    for ln in el[1:int(el[0]) + 1]:
        p = [int(v) for v in ln.split()]
        etype, ntag = p[1], p[2]
        if etype not in _GMSH_TYPES:
            continue
        kind, nn, nc = _GMSH_TYPES[etype]
        if kind == "point":
            continue
        phys = p[3] if ntag > 0 else 0
        conn = [index[v] for v in p[3 + ntag:3 + ntag + nc]]
        dim = {"line": 1, "tri": 2, "tet": 3}[kind]
        _add(regions, phys, dim, names, kind, conn)
    return xyz, regions


def _gmsh4(sec, names):
    # entity tag -> physical tags
    ent_phys = {}
    if "Entities" in sec:
        it = iter(sec["Entities"])
        counts = [int(v) for v in next(it).split()]
        for dim, cnt in enumerate(counts):
            for _ in range(cnt):
                p = next(it).split()
                if dim == 0:
                    nphys = int(p[4])
                    phys = [int(v) for v in p[5:5 + nphys]]
                else:
                    nphys = int(p[7])
                    phys = [int(v) for v in p[8:8 + nphys]]
                ent_phys[(dim, int(p[0]))] = phys
    # nodes
    nl = sec["Nodes"]
    nblocks, nnodes = [int(v) for v in nl[0].split()[:2]]
    ids, xyz = [], []
    i = 1
    for _ in range(nblocks):
        _, _, param, nb = [int(v) for v in nl[i].split()]
        i += 1
        ids.extend(int(nl[i + k]) for k in range(nb))
        for k in range(nb):
            xyz.append([float(v) for v in nl[i + nb + k].split()[:3]])
        i += 2 * nb
    index = {v: k for k, v in enumerate(ids)}
    xyz = np.array(xyz).reshape(-1, 3)
    # elements
    regions = {}
    el = sec["Elements"]
    nblocks = int(el[0].split()[0])
    i = 1
    for _ in range(nblocks):
        edim, etag, etype, nb = [int(v) for v in el[i].split()]
        i += 1
        rows = el[i:i + nb]
        i += nb
        if etype not in _GMSH_TYPES or _GMSH_TYPES[etype][0] == "point":
            continue
        kind, nn, nc = _GMSH_TYPES[etype]
        phys = ent_phys.get((edim, etag)) or ([etag] if not names and not any(ent_phys.values()) else [])
        if not phys:
            continue
        for ln in rows:
            p = ln.split()
            conn = [index[int(v)] for v in p[1:1 + nc]]
            for tag in phys:
                _add(regions, tag, edim, names, kind, conn)
    return xyz, regions


# ------------------------------------------------------------------------ stl
def read_stl(path):
    with open(path, "rb") as fh:
        data = fh.read()
    tris = None
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                                      ("a", "<u2")]), count=n, offset=84)
            tris = rec["v"].astype(float)
    if tris is None:
        txt = data.decode("utf-8", errors="replace").split()
        v = [[float(txt[k + 1]), float(txt[k + 2]), float(txt[k + 3])]
             for k in range(len(txt)) if txt[k] == "vertex"]
        tris = np.array(v).reshape(-1, 3, 3)
    pts = tris.reshape(-1, 3)
    uniq, inv = np.unique(np.round(pts, 12), axis=0, return_inverse=True)
    conn = inv.reshape(-1, 3)
    r = Region(1, os.path.splitext(os.path.basename(path))[0], 3)
    r.tris = conn.tolist()
    return Mesh(uniq, {1: r}, path)
