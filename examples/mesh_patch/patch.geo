// Same geometry as make_mesh.py, for the Gmsh GUI (units: mm).
// File > Export > patch.msh (Version 4 ASCII)
SetFactory("OpenCASCADE");
Box(1) = {-20, -20, 0, 40, 40, 1.6};          // substrate
Rectangle(101) = {-20, -20, 0, 40, 40};        // ground plane
Rectangle(102) = {-15, -11, 1.6, 30, 22};      // patch
Point(201) = {0, -5, 0.4};
Point(202) = {0, -5, 1.6};
Line(301) = {201, 202};                        // feed pin
Physical Volume("substrate", 1) = {1};
Physical Surface("ground", 2) = {101};
Physical Surface("patch", 3) = {102};
Physical Curve("probe", 4) = {301};
Mesh.MeshSizeMax = 4;
