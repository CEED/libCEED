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

/// @file
/// Boundary condition functions for solid mechanics example using PETSc

#include "../elasticity.h"

// -----------------------------------------------------------------------------
// Boundary Functions
// -----------------------------------------------------------------------------
// Note: If additional boundary conditions are added, an update is needed in
//         elasticity.h for the boundaryOptions variable.

// BCMMS - boundary function
// Values on all points of the mesh is set based on given solution below
// for u[0], u[1], u[2]
PetscErrorCode BCMMS(PetscInt dim, PetscReal load_increment,
                     const PetscReal coords[], PetscInt num_comp_u,
                     PetscScalar *u, void *ctx) {
  PetscScalar x = coords[0];
  PetscScalar y = coords[1];
  PetscScalar z = coords[2];

  PetscFunctionBeginUser;

  u[0] = exp(2*x)*sin(3*y)*cos(4*z) / 1e8 * load_increment;
  u[1] = exp(3*y)*sin(4*z)*cos(2*x) / 1e8 * load_increment;
  u[2] = exp(4*z)*sin(2*x)*cos(3*y) / 1e8 * load_increment;

  PetscFunctionReturn(0);
};

#ifndef M_PI
#  define M_PI    3.14159265358979323846
#endif

// BCClamp - fix boundary values with affine transformation at fraction of load
//   increment
PetscErrorCode BCClamp(PetscInt dim, PetscReal load_increment,
                       const PetscReal coords[], PetscInt num_comp_u,
                       PetscScalar *u, void *ctx) {
  PetscScalar x = coords[0];
  PetscScalar y = coords[1];
  PetscScalar z = coords[2];
  PetscScalar (*clampMax) = (PetscScalar(*))ctx;

  PetscFunctionBeginUser;
  PetscScalar
  // Translation vector
  lx = clampMax[0]*load_increment,
  ly = clampMax[1]*load_increment,
  lz = clampMax[2]*load_increment,
  // Normalized rotation axis
  kx = clampMax[3],
  ky = clampMax[4],
  kz = clampMax[5],
  // Rotation polynomial
  c_0 = clampMax[6] * M_PI,
  c_1 = clampMax[7] * M_PI,
  cx = kx * x + ky * y + kz * z,
  // Rotation magnitude
  theta = (c_0 + c_1 * cx) * load_increment;
  PetscScalar c = cos(theta), s = sin(theta);

  u[0] = lx + s*(-kz*y + ky*z) + (1-c)*(-(ky*ky+kz*kz)*x + kx*ky*y + kx*kz*z);
  u[1] = ly + s*(kz*x + -kx*z) + (1-c)*(kx*ky*x + -(kx*kx+kz*kz)*y + ky*kz*z);
  u[2] = lz + s*(-ky*x + kx*y) + (1-c)*(kx*kz*x + ky*kz*y + -(kx*kx+ky*ky)*z);
  PetscFunctionReturn(0);
};
