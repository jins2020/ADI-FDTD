"""Text input-file parser for the 3D ADI-FDTD solver.

Each line is ``keyword  positional...  key=value...``. ``#`` starts a comment.
See README.md for the full list of keywords.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .grid import Grid, build_axis

AXES = {"x": 0, "y": 1, "z": 2}
MAT_KEYS = ("eps", "mu", "sigma", "sigma_m")


@dataclass
class Box:
    i0: float
    i1: float
    j0: float
    j1: float
    k0: float
    k1: float
    phys: bool = False      # True: coordinates in metres, resolved to nodes later

    def ranges(self):
        return ((self.i0, self.i1), (self.j0, self.j1), (self.k0, self.k1))

    def resolve(self, grid: Grid) -> "Box":
        if not self.phys:
            return self
        v = [grid.node_index(a, x) for a, r in enumerate(self.ranges()) for x in r]
        return Box(*v)


@dataclass
class Dielectric:
    box: Box
    eps_r: float = 1.0
    sigma: float = 0.0
    mu_r: float = 1.0
    sigma_m: float = 0.0

    def props(self):
        return dict(eps=self.eps_r, mu=self.mu_r, sigma=self.sigma, sigma_m=self.sigma_m)


@dataclass
class MeshShape:
    file: str
    region: str             # physical tag / name ("1" for STL)
    kind: str               # "material" or "pec"
    props: dict
    scale: float = 1.0
    offset: tuple = (0.0, 0.0, 0.0)
    subcell: int = 2


@dataclass
class Port:
    box: Box
    axis: int            # 0,1,2 -> x,y,z
    sign: float          # +1 / -1
    amp: float           # impressed E-field amplitude [V/m]
    R: float             # internal resistance [ohm]
    wave: str
    params: dict
    name: str


@dataclass
class Output:
    field: str           # "E" or "H"
    comp: str            # "x","y","z","mag"
    plane: str           # "xy","xz","yz"
    index: int           # cell index along plane normal
    interval: int
    file: str
    fps: int = 10
    clim: float | None = None
    pos: float | None = None   # physical position of the plane [m]
    grid_lines: bool = False   # draw grid lines (non-uniform grids)


@dataclass
class SimConfig:
    shape: tuple = (40, 40, 40)
    cell: tuple = (1e-3, 1e-3, 1e-3)
    origin: tuple = (0.0, 0.0, 0.0)
    lines: list = field(default_factory=lambda: [None, None, None])   # non-uniform axes
    dt: float | None = None
    cfl: float = 1.0
    steps: int = 200
    pml: tuple = (8, 8, 8, 8, 8, 8)
    pml_order: float = 3.0
    pml_R0: float = 1e-4
    pml_type: str = "uniaxial"  # "uniaxial" or "isotropic" matched-loss layer
    conductors: list = field(default_factory=list)    # Box (applied first)
    dielectrics: list = field(default_factory=list)   # Dielectric (applied first)
    shapes: list = field(default_factory=list)        # ordered: ("pec"|"mat", Box|Dielectric|MeshShape)
    ports: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    outdir: str = "output"
    dtype: str = "float64"
    grid: Grid | None = None

    def make_grid(self) -> Grid:
        axes = []
        for a in range(3):
            if self.lines[a] is not None:
                axes.append(build_axis(self.lines[a]))
            else:
                axes.append(Grid.uniform_grid(self.shape, self.cell, self.origin).nodes[a])
        self.grid = Grid(*axes)
        self.shape = self.grid.N
        return self.grid


def _split(tokens):
    pos, kw = [], {}
    for t in tokens:
        if "=" in t:
            k, v = t.split("=", 1)
            kw[k.strip().lower()] = v.strip()
        else:
            pos.append(t)
    return pos, kw


def _box(pos, kw):
    if len(pos) < 6:
        raise ValueError("box needs 6 values i0 i1 j0 j1 k0 k1")
    phys = kw.get("coords", "index").lower() in ("phys", "m", "physical", "meter", "metre")
    conv = float if phys else int
    v = [conv(p) for p in pos[:6]]
    for a in (0, 2, 4):
        if v[a] > v[a + 1]:
            v[a], v[a + 1] = v[a + 1], v[a]
    return Box(*v, phys=phys), pos[6:]


def _mat_props(kw, extra=()):
    p = dict(eps=1.0, mu=1.0, sigma=0.0, sigma_m=0.0)
    if extra:
        p["eps"] = float(extra[0])
    if len(extra) > 1:
        p["sigma"] = float(extra[1])
    for k in MAT_KEYS:
        if k in kw:
            p[k] = float(kw[k])
    for alias, k in (("eps_r", "eps"), ("mu_r", "mu"), ("sigma_e", "sigma")):
        if alias in kw:
            p[k] = float(kw[alias])
    if p["eps"] <= 0 or p["mu"] <= 0 or p["sigma"] < 0 or p["sigma_m"] < 0:
        raise ValueError("need eps > 0, mu > 0, sigma >= 0, sigma_m >= 0")
    return p


def _lines(pos):
    """xlines x0 x1:n1[:r1] x2:n2[:r2] ..."""
    x0 = float(pos[0])
    segs = []
    for tok in pos[1:]:
        p = tok.split(":")
        if len(p) not in (2, 3):
            raise ValueError(f"segment '{tok}' must be x:n or x:n:ratio")
        segs.append((float(p[0]), int(p[1]), float(p[2]) if len(p) == 3 else 1.0))
    if not segs:
        raise ValueError("need at least one segment x:n")
    return (x0, segs)


def _vec(s):
    v = [float(t) for t in s.replace(";", ",").split(",")]
    if len(v) != 3:
        raise ValueError(f"'{s}' must be x,y,z")
    return tuple(v)


def _kind(kw):
    t = kw.get("type", "material").lower()
    if t in ("pec", "conductor", "metal"):
        return "pec"
    if t in ("material", "dielectric", "magnetic"):
        return "material"
    raise ValueError(f"unknown type '{t}' (material | pec)")


def parse_input(path: str) -> SimConfig:
    cfg = SimConfig()
    base = os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()

    last_mesh = None
    for lineno, raw in enumerate(lines, 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, *rest = line.split()
        key = key.lower()
        pos, kw = _split(rest)
        try:
            if key == "grid":
                cfg.shape = tuple(int(p) for p in pos[:3])
            elif key == "cell":
                vals = [float(p) for p in pos]
                cfg.cell = tuple(vals * 3 if len(vals) == 1 else vals[:3])
            elif key == "origin":
                cfg.origin = tuple(float(p) for p in pos[:3])
            elif key in ("xlines", "ylines", "zlines"):
                cfg.lines[AXES[key[0]]] = _lines(pos)
            elif key == "dt":
                cfg.dt = float(pos[0])
            elif key == "cfl":
                cfg.cfl = float(pos[0])
            elif key == "steps":
                cfg.steps = int(pos[0])
            elif key == "pml":
                vals = [int(p) for p in pos]
                if len(vals) == 1:
                    vals *= 6
                elif len(vals) == 3:
                    vals = [vals[0], vals[0], vals[1], vals[1], vals[2], vals[2]]
                if len(vals) != 6:
                    raise ValueError("pml needs 1, 3 or 6 values")
                cfg.pml = tuple(vals)
                if "order" in kw:
                    cfg.pml_order = float(kw["order"])
                if "r0" in kw:
                    cfg.pml_R0 = float(kw["r0"])
                if "type" in kw:
                    cfg.pml_type = kw["type"].lower()
            elif key == "pml_order":
                cfg.pml_order = float(pos[0])
            elif key in ("pml_r0", "pml_reflection"):
                cfg.pml_R0 = float(pos[0])
            elif key == "pml_type":
                cfg.pml_type = pos[0].lower()
            elif key in ("conductor", "pec"):
                box, _ = _box(pos, kw)
                cfg.shapes.append(("pec", box))
            elif key in ("dielectric", "material"):
                box, extra = _box(pos, kw)
                p = _mat_props(kw, extra)
                cfg.shapes.append(("mat", Dielectric(box, p["eps"], p["sigma"], p["mu"], p["sigma_m"])))
            elif key in ("mesh", "stl"):
                f = kw.get("file", pos[0] if pos else None)
                if f is None:
                    raise ValueError("mesh needs file=")
                f = f if os.path.isabs(f) else os.path.join(base, f)
                if not os.path.exists(f):
                    raise ValueError(f"mesh file not found: {f}")
                last_mesh = dict(file=f, scale=float(kw.get("scale", 1.0)),
                                 offset=_vec(kw["offset"]) if "offset" in kw else (0.0, 0.0, 0.0),
                                 subcell=int(kw.get("subcell", 2)))
                if key == "stl" or f.lower().endswith(".stl"):
                    if "type" in kw or any(k in kw for k in MAT_KEYS):
                        cfg.shapes.append(("mesh", MeshShape(region="1", kind=_kind(kw),
                                                             props=_mat_props(kw), **last_mesh)))
            elif key == "region":
                if last_mesh is None:
                    raise ValueError("'region' must follow a 'mesh' line")
                if not pos:
                    raise ValueError("region needs a physical tag or name")
                cfg.shapes.append(("mesh", MeshShape(region=pos[0], kind=_kind(kw),
                                                     props=_mat_props(kw), **last_mesh)))
            elif key == "port":
                box, extra = _box(pos, kw)
                d = kw.get("dir", extra[0] if extra else "+z").lower()
                sign = -1.0 if d.startswith("-") else 1.0
                ax = AXES[d.lstrip("+-")]
                params = {k: float(v) for k, v in kw.items()
                          if k not in ("dir", "amp", "r", "wave", "name", "coords")}
                cfg.ports.append(Port(
                    box=box, axis=ax, sign=sign,
                    amp=float(kw.get("amp", 1.0)),
                    R=float(kw.get("r", 50.0)),
                    wave=kw.get("wave", "gauss").lower(),
                    params=params,
                    name=kw.get("name", f"port{len(cfg.ports) + 1}"),
                ))
            elif key == "output":
                fld = pos[0].upper() if pos else kw.get("field", "E").upper()
                if fld not in ("E", "H"):
                    raise ValueError("output field must be E or H")
                comp = kw.get("comp", "mag").lower()
                plane = kw.get("plane", "xy").lower()
                if plane not in ("xy", "xz", "yz", "yx", "zx", "zy"):
                    raise ValueError(f"bad plane {plane}")
                plane = "".join(sorted(plane))
                interval = int(kw.get("interval", 10))
                n = len([o for o in cfg.outputs if o.field == fld]) + 1
                fname = kw.get("file", f"{fld}{comp}_{plane}_{n}.gif")
                cfg.outputs.append(Output(
                    field=fld, comp=comp, plane=plane,
                    index=int(kw.get("index", -1)), interval=interval,
                    file=fname, fps=int(kw.get("fps", 10)),
                    clim=float(kw["clim"]) if "clim" in kw else None,
                    pos=float(kw["pos"]) if "pos" in kw else None,
                    grid_lines=kw.get("gridlines", "no").lower() in ("1", "yes", "true", "on"),
                ))
            elif key == "outdir":
                cfg.outdir = pos[0]
            elif key == "dtype":
                cfg.dtype = pos[0]
            else:
                raise ValueError(f"unknown keyword '{key}'")
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(f"{path}:{lineno}: {raw.strip()} -> {exc}") from exc

    finalize(cfg)
    return cfg


def finalize(cfg: SimConfig):
    """Build the grid, resolve physical coordinates and validate."""
    grid = cfg.make_grid()
    cfg.conductors = [b.resolve(grid) for b in cfg.conductors]
    for d in cfg.dielectrics:
        d.box = d.box.resolve(grid)
    shapes = []
    for kind, obj in cfg.shapes:
        if kind == "pec":
            obj = obj.resolve(grid)
        elif kind == "mat":
            obj.box = obj.box.resolve(grid)
        shapes.append((kind, obj))
    cfg.shapes = shapes
    for p in cfg.ports:
        p.box = p.box.resolve(grid)
    for o in cfg.outputs:
        normal = AXES[({"x", "y", "z"} - set(o.plane)).pop()]
        if o.pos is not None:
            o.index = grid.cell_index(normal, o.pos)
    _validate(cfg)
    return cfg


def _validate(cfg: SimConfig):
    N = cfg.grid.N
    if cfg.pml_type not in ("uniaxial", "isotropic"):
        raise ValueError("pml_type must be uniaxial or isotropic")
    for p in cfg.pml:
        if p < 0:
            raise ValueError("pml thickness must be >= 0")
    for a in range(3):
        if cfg.pml[2 * a] + cfg.pml[2 * a + 1] >= N[a]:
            raise ValueError("PML thicker than the domain")
    boxes = (cfg.conductors + [d.box for d in cfg.dielectrics] + [p.box for p in cfg.ports]
             + [o if k == "pec" else o.box for k, o in cfg.shapes if k in ("pec", "mat")])
    for b in boxes:
        for a, (lo, hi) in enumerate(b.ranges()):
            if lo < 0 or hi > N[a]:
                raise ValueError(f"box {b} exceeds grid {N}")
    for p in cfg.ports:
        lo, hi = p.box.ranges()[p.axis]
        if hi <= lo:
            raise ValueError(f"port {p.name}: extent along dir must be >= 1 cell")
        if p.R <= 0:
            raise ValueError(f"port {p.name}: R must be > 0")
    for o in cfg.outputs:
        normal = ({"x", "y", "z"} - set(o.plane)).pop()
        a = AXES[normal]
        if o.index < 0:
            o.index = N[a] // 2
        if o.index >= N[a]:
            raise ValueError(f"output index {o.index} outside 0..{N[a]-1}")
