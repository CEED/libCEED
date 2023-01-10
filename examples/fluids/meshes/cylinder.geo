DefineConstant[
  zspan = {0.2, Min .1, Max 10, Step .1,
    Name "Parameters/Zspan"}
];

DefineConstant[
  H = {4.5, Min 1, Max 10, Step .1,
    Name "Parameters/Height"}
];

DefineConstant[
  xL = {4.5, Min 1, Max 10, Step .1,
    Name "Parameters/XL"}
];

DefineConstant[
  xR = {11, Min .2, Max 30, Step .1,
    Name "Parameters/XR"}
];

DefineConstant[
  cyldiameter = {1.0, Min .1, Max 1.0, Step .1,
    Name "Parameters/CylDiameter"}
];

DefineConstant[
  N = {20, Min 2, Max 80, Step 2,
    Name "Parameters/ElementsPerQuadrant"}
];

DefineConstant[
  RadialRatio = {1, Min .2, Max 5, Step .1,
  Name "Parameters/RadialRatio"}
];

Nx1 = N + 1; Rx1 = 1.00;
Nx2 = Ceil(N * 9 / 11 * xR / (2*xL)) + 1; Rx2 = 1.00;
Ny  = N + 1; Ry = 2.00;
Nb  = Ceil(RadialRatio * N) + 1; Rb = 0.9;
Nc  = N + 1; Rc = 1.00;

// Characteristic element length
h = 2 * H / N;

// Exterior corners
Point(1) = {-xL, -H, 0};
Point(2) = {xL, -H, 0};
Point(3) = {xL + xR, -H, 0};
Point(4) = {-xL, H, 0};
Point(5) = {xL, H, 0};
Point(6) = {xL + xR, H, 0};

// Coordinates for Cylinder points
Point(7) = {-cyldiameter/Sqrt(8), -cyldiameter/Sqrt(8), 0};
Point(8) = {cyldiameter/Sqrt(8), -cyldiameter/Sqrt(8), 0};
Point(9) = {-cyldiameter/Sqrt(8), cyldiameter/Sqrt(8), 0};
Point(10) = {cyldiameter/Sqrt(8), cyldiameter/Sqrt(8), 0};
Point(11) = {0, 0, 0};

// Top and bottom edges
Line(1) = {1, 2}; Transfinite Curve {1} = Nx1 Using Progression Rx1;
Line(2) = {2, 3}; Transfinite Curve {2} = Nx2 Using Progression Rx2;
Line(3) = {4, 5}; Transfinite Curve {3} = Nx1 Using Progression Rx1;
Line(4) = {5, 6}; Transfinite Curve {4} = Nx2 Using Progression Rx2;
// Vertical edges and internal interface
Line(5) = {4, 1}; Transfinite Curve {5} = Ny Using Bump Ry;
Line(6) = {5, 2}; Transfinite Curve {6} = Ny Using Bump Ry;
Line(7) = {6, 3}; Transfinite Curve {7} = Ny Using Bump Ry;

// Cylinder arcs
Circle(8) = {7, 11, 8}; Transfinite Curve {8} = Nc Using Progression Rc;
Circle(9) = {8, 11, 10}; Transfinite Curve {9} = Nc Using Progression Rc;
Circle(10) = {10, 11, 9}; Transfinite Curve {10} = Nc Using Progression Rc;
Circle(11) = {9, 11, 7}; Transfinite Curve {11} = Nc Using Progression Rc;

// Radial lines
Line(12) = {1, 7}; Transfinite Curve {12} = Nb Using Progression Rb;
Line(13) = {2, 8}; Transfinite Curve {13} = Nb Using Progression Rb;
Line(14) = {5, 10}; Transfinite Curve {14} = Nb Using Progression Rb;
Line(15) = {4, 9}; Transfinite Curve {15} = Nb Using Progression Rb;

// surfaces
Curve Loop(1) = {12, 8, -13, -1};
Plane Surface(1) = {1};
Curve Loop(2) = {13, 9, -14, 6};
Plane Surface(2) = {2};
Curve Loop(3) = {14, 10, -15, 3};
Plane Surface(3) = {3};
Curve Loop(4) = {15, 11, -12, -5};
Plane Surface(4) = {4};
Curve Loop(5) = {2, -7, -4, 6};
Plane Surface(5) = {5};
Transfinite Surface {1};
Transfinite Surface {2};
Transfinite Surface {3};
Transfinite Surface {4};
Transfinite Surface {5};
Recombine Surface {1};
Recombine Surface {2};
Recombine Surface {3};
Recombine Surface {4};
Recombine Surface {5};
Extrude {0, 0, zspan} {
  Surface{1, 2, 3, 4, 5};
  Layers {Ceil(zspan / h)};
  Recombine;
}

Physical Surface("inlet") = {102}; // inlet
Physical Surface("outlet") = {116}; // outlet
Physical Surface("top") = {80, 120}; // top
Physical Surface("bottom") = {36, 112}; // bottom
Physical Surface("cylinderwalls") = {94, 28, 50, 72}; // cylinderwalls
Physical Surface("frontandback") = {37, 1, 4, 103, 3, 81, 2, 59, 5, 125}; // frontandback
Physical Volume("mesh") = {1:5};
