"""Generate sphere.stl (binary STL, units: mm): sphere of radius 8 mm at the origin.

Run:  python make_mesh.py      (needs `pip install gmsh`)
Any CAD tool that exports a closed STL surface can be used instead.
"""
import os

import gmsh

here = os.path.dirname(os.path.abspath(__file__))
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("sphere")
gmsh.model.occ.addSphere(0, 0, 0, 8.0)
gmsh.model.occ.synchronize()
gmsh.option.setNumber("Mesh.MeshSizeMax", 1.2)
gmsh.option.setNumber("Mesh.Binary", 1)
gmsh.model.mesh.generate(2)
gmsh.write(os.path.join(here, "sphere.stl"))
gmsh.finalize()
print("wrote sphere.stl")
