// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for setting up vortex shedding problem behind a cylinder

#include "../navierstokes.h"
#include "../qfunctions/vortexshedding.h"
#include "stg_shur14.h"

PetscErrorCode NS_VORTEXSHEDDING(ProblemData *problem, DM dm, void *ctx) {

  PetscInt ierr;
  User      user    = *(User *)ctx;
  MPI_Comm  comm    = PETSC_COMM_WORLD;
  PetscBool use_stg = PETSC_FALSE;
  VortexsheddingContext vortexshedding_ctx;
  NewtonianIdealGasContext newtonian_ig_ctx;
  CeedQFunctionContext vortexshedding_context;

  PetscFunctionBeginUser;
  ierr = NS_NEWTONIAN_IG(problem, dm, ctx); CHKERRQ(ierr);
  ierr = PetscCalloc1(1, &vortexshedding_ctx); CHKERRQ(ierr);

  // ------------------------------------------------------
  //               SET UP Vortex Shedding
  // ------------------------------------------------------
  problem->ics.qfunction        = ICsVortexshedding;
  problem->ics.qfunction_loc    = ICsVortexshedding_loc;

  CeedScalar P0                 = 1.e5;      // Pa
  PetscBool  weakT              = PETSC_FALSE; // weak density or temperature


}