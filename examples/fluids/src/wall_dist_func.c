// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// General wall distance functions for Navier-Stokes example using PETSc
/// We do this by solving the Poisson equation ∇^{2} φ  = -1

#include "../qfunctions/wall_dist_func.h"

#include "../navierstokes.h"
#include "../qfunctions/newtonian_state.h"

// General distance functions
static PetscErrorCode Distance_Function_NS(DM dm, User user) {
  DM       dmDist;
  PetscFE  fe;
  PetscInt dim = 3;
  SNES     snesDist;

  PetscFunctionBeginUser;
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, NULL, PETSC_DETERMINE, &fe));
  PetscBool distance_snes_monitor = PETSC_FALSE;
  PetscCall(PetscOptionsHasName(NULL, NULL, "-distance_snes_monitor", &distance_snes_monitor));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snesDist));
  PetscObjectSetOptionsPrefix((PetscObject)snesDist, "distance_");
  PetscCall(DMClone(dm, &dmDist));
  PetscCall(DMAddField(dmDist, NULL, (PetscObject)fe));

  PetscFunctionReturn(0);
}
