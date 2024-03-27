// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for setting up Taylor-Green Vortex

#include "../qfunctions/taylorgreen.h"

#include "../navierstokes.h"

PetscErrorCode NS_TAYLOR_GREEN(ProblemData *problem, DM dm, void *ctx, SimpleBC bc) {
  PetscFunctionBeginUser;
  PetscCall(NS_NEWTONIAN_IG(problem, dm, ctx, bc));

  problem->ics.qfunction     = ICsTaylorGreen;
  problem->ics.qfunction_loc = ICsTaylorGreen_loc;
  PetscFunctionReturn(PETSC_SUCCESS);
}
