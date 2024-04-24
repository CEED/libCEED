// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for setting up slip boundary condition

#include "../qfunctions/bc_slip.h"

#include <ceed.h>
#include <petscdm.h>

#include "../navierstokes.h"
#include "../qfunctions/newtonian_types.h"

PetscErrorCode SlipBCSetup(ProblemData problem, DM dm, void *ctx, CeedQFunctionContext newtonian_ig_qfctx) {
  User user = *(User *)ctx;
  Ceed ceed = user->ceed;

  PetscFunctionBeginUser;
  switch (user->phys->state_var) {
    case STATEVAR_CONSERVATIVE:
      problem->apply_slip.qfunction              = Slip_Conserv;
      problem->apply_slip.qfunction_loc          = Slip_Conserv_loc;
      problem->apply_slip_jacobian.qfunction     = Slip_Jacobian_Conserv;
      problem->apply_slip_jacobian.qfunction_loc = Slip_Jacobian_Conserv_loc;
      break;
    case STATEVAR_PRIMITIVE:
      problem->apply_slip.qfunction              = Slip_Prim;
      problem->apply_slip.qfunction_loc          = Slip_Prim_loc;
      problem->apply_slip_jacobian.qfunction     = Slip_Jacobian_Prim;
      problem->apply_slip_jacobian.qfunction_loc = Slip_Jacobian_Prim_loc;
      break;
  }

  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(newtonian_ig_qfctx, &problem->apply_slip.qfunction_context));
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(newtonian_ig_qfctx, &problem->apply_slip_jacobian.qfunction_context));
  PetscFunctionReturn(PETSC_SUCCESS);
}
