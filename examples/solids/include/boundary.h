// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef libceed_solids_examples_boundary_h
#define libceed_solids_examples_boundary_h

#include <petsc.h>

typedef PetscErrorCode BCFunc(PetscInt, PetscReal, const PetscReal *, PetscInt, PetscScalar *, void *);
BCFunc                 BCMMS, BCZero, BCClamp;

// -----------------------------------------------------------------------------
// Boundary Functions
// -----------------------------------------------------------------------------
// Note: If additional boundary conditions are added, an update is needed in
//         elasticity.h for the boundaryOptions variable.

// BCMMS - boundary function
// Values on all points of the mesh is set based on given solution below
// for u[0], u[1], u[2]
PetscErrorCode BCMMS(PetscInt dim, PetscReal load_increment, const PetscReal coords[], PetscInt num_comp_u, PetscScalar *u, void *ctx);

// BCClamp - fix boundary values with affine transformation at fraction of load
//   increment
PetscErrorCode BCClamp(PetscInt dim, PetscReal load_increment, const PetscReal coords[], PetscInt num_comp_u, PetscScalar *u, void *ctx);

#endif  // libceed_solids_examples_boundary_h
