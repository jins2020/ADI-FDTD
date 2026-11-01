"""Text input-file parser for the 3D ADI-FDTD solver.

Each line is ``keyword  positional...  key=value...``. ``#`` starts a comment.
See README.md for the full list of keywords.
"""
from __future__ import annotations

from dataclasses import dataclass, field

AXES = {"x": 0, "y": 1, "z": 2}


@dataclass
class Box:
    i0: int
    i1: int
    j0: int
    j1: int
    k0: int
    k1: int

    def ranges(self):
        return ((self.i0, self.i1), (self.j0, self.j1), (self.k0, self.k1))


@dataclass
class Dielectric:
    box: Box
    eps_r: float = 1.0
    sigma: float = 0.0


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


@dataclass
class SimConfig:
    shape: tuple = (40, 40, 40)
    cell: tuple = (1e-3, 1e-3, 1e-3)
    dt: float | None = None
    cfl: float = 1.0
    steps: int = 200
    pml: tuple = (8, 8, 8, 8, 8, 8)
    pml_order: float = 3.0
    pml_R0: float = 1e-4
    pml_type: str = "uniaxial"  # "uniaxial" or "isotropic" matched-loss layer
    conductors: list = field(default_factory=list)
    dielectrics: list = field(default_factory=list)
    ports: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    outdir: str = "output"
    dtype: str = "float64"


def _split(tokens):
    pos, kw = [], {}
    for t in tokens:
        if "=" in t:
            k, v = t.split("=", 1)
            kw[k.strip().lower()] = v.strip()
        else:
            pos.append(t)
    return pos, kw


def _box(pos, kw, lineno):
    if len(pos) < 6:
        raise ValueError(f"line {lineno}: box needs 6 indices i0 i1 j0 j1 k0 k1")
    v = [int(p) for p in pos[:6]]
    for a in (0, 2, 4):
        if v[a] > v[a + 1]:
            v[a], v[a + 1] = v[a + 1], v[a]
    return Box(*v), pos[6:]


def parse_input(path: str) -> SimConfig:
    cfg = SimConfig()
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()

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
                box, _ = _box(pos, kw, lineno)
                cfg.conductors.append(box)
            elif key == "dielectric":
                box, extra = _box(pos, kw, lineno)
                eps = float(kw.get("eps", extra[0] if extra else 1.0))
                sig = float(kw.get("sigma", extra[1] if len(extra) > 1 else 0.0))
                cfg.dielectrics.append(Dielectric(box, eps, sig))
            elif key == "port":
                box, extra = _box(pos, kw, lineno)
                d = kw.get("dir", extra[0] if extra else "+z").lower()
                sign = -1.0 if d.startswith("-") else 1.0
                ax = AXES[d.lstrip("+-")]
                params = {k: float(v) for k, v in kw.items()
                          if k not in ("dir", "amp", "r", "wave", "name")}
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
                ))
            elif key == "outdir":
                cfg.outdir = pos[0]
            elif key == "dtype":
                cfg.dtype = pos[0]
            else:
                raise ValueError(f"unknown keyword '{key}'")
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(f"{path}:{lineno}: {raw.strip()} -> {exc}") from exc

    _validate(cfg)
    return cfg


def _validate(cfg: SimConfig):
    N = cfg.shape
    if cfg.pml_type not in ("uniaxial", "isotropic"):
        raise ValueError("pml_type must be uniaxial or isotropic")
    for p in cfg.pml:
        if p < 0:
            raise ValueError("pml thickness must be >= 0")
    for a in range(3):
        if cfg.pml[2 * a] + cfg.pml[2 * a + 1] >= N[a]:
            raise ValueError("PML thicker than the domain")
    boxes = cfg.conductors + [d.box for d in cfg.dielectrics] + [p.box for p in cfg.ports]
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
