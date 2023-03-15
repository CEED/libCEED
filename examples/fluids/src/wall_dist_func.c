// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// General wall distance functions for Navier-Stokes example using PETSc
/// We do this by solving the Poisson equation ∇^{2} φ  = -1 with weak form ∫ ∇v ⋅ ∇ φ - v

#include "../qfunctions/wall_dist_func.h"

#include "../navierstokes.h"
#include "../qfunctions/newtonian_state.h"

// General distance functions
static PetscErrorCode Distance_Function_NS(DM dm, User user) {
  PetscScalar  *r;
  DM            dmDist;
  PetscFE       fe;
  PetscInt     *xl_size, dim = 3;
  PetscInt      distance_function, distance_function_loc;
  SNES          snesDist;
  Vec          *X_loc, rhs, rhs_loc;
  PetscMemType  mem_type;
  Ceed          ceed;
  CeedVector    rhs_ceed;
  CeedOperator  op_distance_function;
  CeedQFunction qf_distance_function;

  PetscFunctionBeginUser;
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, NULL, PETSC_DETERMINE, &fe));
  PetscBool distance_snes_monitor = PETSC_FALSE;
  PetscCall(PetscOptionsHasName(NULL, NULL, "-distance_snes_monitor", &distance_snes_monitor));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snesDist));
  PetscObjectSetOptionsPrefix((PetscObject)snesDist, "distance_");
  PetscCall(DMClone(dm, &dmDist));
  PetscCall(DMAddField(dmDist, NULL, (PetscObject)fe));

  // Create RHS vector
  PetscCall(VecDuplicate(X_loc, &rhs_loc));
  PetscCall(VecZeroEntries(rhs_loc));
  PetscCall(VecGetArrayAndMemType(rhs_loc, &r, &mem_type));
  CeedVectorCreate(ceed, xl_size, &rhs_ceed);
  CeedVectorSetArray(rhs_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, r);

  // Gather RHS
  CeedVectorTakeArray(rhs_ceed, MemTypeP2C(mem_type), NULL);
  PetscCall(VecRestoreArrayAndMemType(rhs_loc, &r));
  PetscCall(VecZeroEntries(rhs));
  PetscCall(DMLocalToGlobal(dmDist, rhs_loc, ADD_VALUES, rhs));
  CeedVectorDestroy(&rhs_ceed);

  // Create and Add QFunction fields
  CeedQFunctionCreateInterior(ceed, dim, distance_function, distance_function_loc, &qf_distance_function);
  CeedQFunctionAddInput(qf_distance_function, "v", dim, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_distance_function, "dphi", dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_distance_function, "q_data", dim * (dim + 1) / 2, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_distance_function, "dv", dim, CEED_EVAL_GRAD);

  CeedOperatorCreate(ceed, qf_distance_function, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_distance_function);

  PetscFunctionReturn(0);
}
