// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "../elasticity.h"

// -----------------------------------------------------------------------------
// Boundary Functions
// -----------------------------------------------------------------------------
// Note: If additional boundary conditions are added, an update is needed in
//         elasticity.h for the boundaryOptions variable.

// BCMMS boundary function
// ss : (sideset)
// MMS: Boundary corresponding to the Method of Manufactured Solutions
// Cylinder with a whole in the middle (see figure ..\meshes\surface999-9.png)
// Also check ..\meshes\cyl-hol.8.jou
//
// left:  sideset 999
// right: sideset 998
// outer: sideset 997
// inner: sideset 996
//
//   / \-------------------\              y
//  /   \                   \             |
// (  O  )                   )      x ____|
//  \   /                   /              \    Coordinate axis
//   \ /-------------------/                \ z
//
// Values on all points of the mesh is set based on given solution below
// for u[0], u[1], u[2]
PetscErrorCode BCMMS(PetscInt dim, PetscReal loadIncrement,
                     const PetscReal coords[], PetscInt ncompu,
                     PetscScalar *u, void *ctx) {
  PetscScalar x = coords[0];
  PetscScalar y = coords[1];
  PetscScalar z = coords[2];

  PetscFunctionBeginUser;

  u[0] = exp(2*x)*sin(3*y)*cos(4*z) / 1e8 * loadIncrement;
  u[1] = exp(3*y)*sin(4*z)*cos(2*x) / 1e8 * loadIncrement;
  u[2] = exp(4*z)*sin(2*x)*cos(3*y) / 1e8 * loadIncrement;

  PetscFunctionReturn(0);
};

// BCBend2_ss boundary function
// ss : (sideset)
// 2_ss : two sides of the geometry
// Cylinder with a whole in the middle (see figure ..\meshes\surface999-9.png)
// Also check ..\meshes\cyl-hol.8.jou
//
// left:  sideset 999
// right: sideset 998
//
//   / \-------------------\              y
//  /   \                   \             |
// (  O  )                   )      x ____|
//  \   /                   /              \    Coordinate axis
//   \ /-------------------/                \ z
//
//  0 values on the left side of the cyl-hole (sideset 999)
// -1 values on y direction of the right side of the cyl-hole (sideset 999)
PetscErrorCode BCBend2_ss(PetscInt dim, PetscReal loadIncrement,
                          const PetscReal coords[], PetscInt ncompu,
                          PetscScalar *u, void *ctx) {
  PetscInt *faceID = (PetscInt *)ctx;

  PetscFunctionBeginUser;

  switch (*faceID) {
  case 999:                      // left side of the cyl-hol
    u[0] = 0;
    u[1] = 0;
    u[2] = 0;
    break;
  case 998:                      // right side of the cyl-hol
    u[0] = 0;
    u[1] = -1.0 * loadIncrement; // bend in the -y direction
    u[2] = 0;
    break;
  }

  PetscFunctionReturn(0);
};

// BCBend1_ss boundary function
// ss : (sideset)
// 1_ss : 1 side (left side) of the geometry
// Cylinder with a whole in the middle (see figure ..\meshes\surface999-9.png)
// Also check ..\meshes\cyl-hol.8.jou
//
// left: sideset 999
//
//   / \-------------------\              y
//  /   \                   \             |
// (  O  )                   )      x ____|
//  \   /                   /              \    Coordinate axis
//   \ /-------------------/                \ z
//
//  0 values on the left side of the cyl-hole (sideset 999)
PetscErrorCode BCBend1_ss(PetscInt dim, PetscReal loadIncrement,
                          const PetscReal coords[], PetscInt ncompu,
                          PetscScalar *u, void *ctx) {
  PetscFunctionBeginUser;

  u[0] = 0;
  u[1] = 0;
  u[2] = 0;

  PetscFunctionReturn(0);
};
