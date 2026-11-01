"""Generate patch.msh (Gmsh ASCII v4.1, units: mm) for the patch-antenna example.

Physical groups (used by the 'region' lines of patch_antenna.txt):
  3D  1 "substrate" : FR-4 slab 40 x 40 x 1.6 mm  (tetrahedra)
  2D  2 "ground"    : ground plane at z = 0     (triangles -> PEC sheet)
  2D  3 "patch"     : 30 x 22 mm patch at z = 1.6 (triangles -> PEC sheet)
  1D  4 "probe"     : feed pin from z = 0.4 to 1.6 mm (line -> PEC wire)

Run:  python make_mesh.py      (needs `pip install gmsh`)
The same geometry can be built in the Gmsh GUI from patch.geo.
"""
import os

import gmsh

here = os.path.dirname(os.path.abspath(__file__))
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("patch")
occ = gmsh.model.occ

sub = occ.addBox(-20, -20, 0, 40, 40, 1.6)
gnd = occ.addRectangle(-20, -20, 0, 40, 40)
pat = occ.addRectangle(-15, -11, 1.6, 30, 22)
p0 = occ.addPoint(0, -5, 0.4)
p1 = occ.addPoint(0, -5, 1.6)
pin = occ.addLine(p0, p1)
occ.synchronize()

gmsh.model.addPhysicalGroup(3, [sub], 1, "substrate")
gmsh.model.addPhysicalGroup(2, [gnd], 2, "ground")
gmsh.model.addPhysicalGroup(2, [pat], 3, "patch")
gmsh.model.addPhysicalGroup(1, [pin], 4, "probe")

gmsh.option.setNumber("Mesh.MeshSizeMax", 4.0)
gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
gmsh.option.setNumber("Mesh.Binary", 0)
gmsh.model.mesh.generate(3)
gmsh.write(os.path.join(here, "patch.msh"))
gmsh.finalize()
print("wrote patch.msh")
