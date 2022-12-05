// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Miscellaneous utility functions

#include "../navierstokes.h"

PetscErrorCode ICs_FixMultiplicity(DM dm, CeedData ceed_data, User user, Vec Q_loc, Vec Q, CeedScalar time) {
  PetscFunctionBeginUser;

  // ---------------------------------------------------------------------------
  // Update time for evaluation
  // ---------------------------------------------------------------------------
  if (user->phys->ics_time_label) CeedOperatorContextSetDouble(ceed_data->op_ics, user->phys->ics_time_label, &time);

  // ---------------------------------------------------------------------------
  // ICs
  // ---------------------------------------------------------------------------
  // -- CEED Restriction
  CeedVector q0_ceed;
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &q0_ceed, NULL);

  // -- Place PETSc vector in CEED vector
  CeedScalar  *q0;
  PetscMemType q0_mem_type;
  PetscCall(VecGetArrayAndMemType(Q_loc, (PetscScalar **)&q0, &q0_mem_type));
  CeedVectorSetArray(q0_ceed, MemTypeP2C(q0_mem_type), CEED_USE_POINTER, q0);

  // -- Apply CEED Operator
  CeedOperatorApply(ceed_data->op_ics, ceed_data->x_coord, q0_ceed, CEED_REQUEST_IMMEDIATE);

  // -- Restore vectors
  CeedVectorTakeArray(q0_ceed, MemTypeP2C(q0_mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(Q_loc, (const PetscScalar **)&q0));

  // -- Local-to-Global
  PetscCall(VecZeroEntries(Q));
  PetscCall(DMLocalToGlobal(dm, Q_loc, ADD_VALUES, Q));

  // ---------------------------------------------------------------------------
  // Fix multiplicity for output of ICs
  // ---------------------------------------------------------------------------
  // -- CEED Restriction
  CeedVector mult_vec;
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &mult_vec, NULL);

  // -- Place PETSc vector in CEED vector
  CeedScalar  *mult;
  PetscMemType m_mem_type;
  Vec          multiplicity_loc;
  PetscCall(DMGetLocalVector(dm, &multiplicity_loc));
  PetscCall(VecGetArrayAndMemType(multiplicity_loc, (PetscScalar **)&mult, &m_mem_type));
  CeedVectorSetArray(mult_vec, MemTypeP2C(m_mem_type), CEED_USE_POINTER, mult);

  // -- Get multiplicity
  CeedElemRestrictionGetMultiplicity(ceed_data->elem_restr_q, mult_vec);

  // -- Restore vectors
  CeedVectorTakeArray(mult_vec, MemTypeP2C(m_mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(multiplicity_loc, (const PetscScalar **)&mult));

  // -- Local-to-Global
  Vec multiplicity;
  PetscCall(DMGetGlobalVector(dm, &multiplicity));
  PetscCall(VecZeroEntries(multiplicity));
  PetscCall(DMLocalToGlobal(dm, multiplicity_loc, ADD_VALUES, multiplicity));

  // -- Fix multiplicity
  PetscCall(VecPointwiseDivide(Q, Q, multiplicity));
  PetscCall(VecPointwiseDivide(Q_loc, Q_loc, multiplicity_loc));

  // -- Restore vectors
  PetscCall(DMRestoreLocalVector(dm, &multiplicity_loc));
  PetscCall(DMRestoreGlobalVector(dm, &multiplicity));

  // Cleanup
  CeedVectorDestroy(&mult_vec);
  CeedVectorDestroy(&q0_ceed);

  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexInsertBoundaryValues_NS(DM dm, PetscBool insert_essential, Vec Q_loc, PetscReal time, Vec face_geom_FVM, Vec cell_geom_FVM,
                                             Vec grad_FVM) {
  Vec Qbc, boundary_mask;
  PetscFunctionBegin;

  // Mask (zero) Dirichlet entries
  PetscCall(DMGetNamedLocalVector(dm, "boundary mask", &boundary_mask));
  PetscCall(VecPointwiseMult(Q_loc, Q_loc, boundary_mask));
  PetscCall(DMRestoreNamedLocalVector(dm, "boundary mask", &boundary_mask));

  PetscCall(DMGetNamedLocalVector(dm, "Qbc", &Qbc));
  PetscCall(VecAXPY(Q_loc, 1., Qbc));
  PetscCall(DMRestoreNamedLocalVector(dm, "Qbc", &Qbc));

  PetscFunctionReturn(0);
}

// Compare reference solution values with current test run for CI
PetscErrorCode RegressionTests_NS(AppCtx app_ctx, Vec Q) {
  Vec         Qref;
  PetscViewer viewer;
  PetscReal   error, Qrefnorm;
  PetscFunctionBegin;

  // Read reference file
  PetscCall(VecDuplicate(Q, &Qref));
  PetscCall(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)Q), app_ctx->file_path, FILE_MODE_READ, &viewer));
  PetscCall(VecLoad(Qref, viewer));

  // Compute error with respect to reference solution
  PetscCall(VecAXPY(Q, -1.0, Qref));
  PetscCall(VecNorm(Qref, NORM_MAX, &Qrefnorm));
  PetscCall(VecScale(Q, 1. / Qrefnorm));
  PetscCall(VecNorm(Q, NORM_MAX, &error));

  // Check error
  if (error > app_ctx->test_tol) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Test failed with error norm %g\n", (double)error));
  }

  // Cleanup
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecDestroy(&Qref));

  PetscFunctionReturn(0);
}

PetscErrorCode GetConservativeFromPrimitive_NS(Vec Q_prim, Vec Q_cons) {
  PetscInt           Q_size;
  PetscInt           num_comp = 5;
  PetscScalar        cv       = 2.5;
  const PetscScalar *Y;
  PetscScalar       *U;

  PetscFunctionBegin;
  PetscCall(VecGetSize(Q_prim, &Q_size));
  PetscCall(VecGetArrayRead(Q_prim, &Y));
  PetscCall(VecGetArrayWrite(Q_cons, &U));
  for (PetscInt i = 0; i < Q_size; i += num_comp) {
    // Primitive variables
    PetscScalar P = Y[i + 0], u[3] = {Y[i + 1], Y[i + 2], Y[i + 3]}, T = Y[i + 4];
    PetscScalar rho = P / T;
    U[i + 0]        = rho;
    for (int j = 0; j < 3; j++) U[i + 1 + j] = rho * u[j];
    U[i + 4] = rho * (cv * T + .5 * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]));
  }
  PetscCall(VecRestoreArrayRead(Q_prim, &Y));
  PetscCall(VecRestoreArrayWrite(Q_cons, &U));
  PetscFunctionReturn(0);
}

PetscErrorCode GetPrimitiveFromConservative_NS(Vec Q_cons, Vec Q_prim) {
  PetscInt           Q_size;
  PetscInt           num_comp = 5;
  PetscScalar        cv       = 2.5;
  const PetscScalar *U;
  PetscScalar       *Y;

  PetscFunctionBegin;
  PetscCall(VecGetSize(Q_cons, &Q_size));
  PetscCall(VecGetArrayRead(Q_cons, &U));
  PetscCall(VecGetArrayWrite(Q_prim, &Y));
  for (PetscInt i = 0; i < Q_size; i += num_comp) {
    // Conservative variables
    PetscScalar rho = U[i + 0], M[3] = {U[i + 1], U[i + 2], U[i + 3]}, E = U[i + 4];
    // Primitive variables
    for (int j = 0; j < 3; j++) Y[i + 1 + j] = M[j] / rho;
    Y[i + 4] = (E - 0.5 * (M[0] * M[0] + M[1] * M[1] + M[2] * M[2]) / rho) / (cv * rho);
    Y[i + 0] = (E - 0.5 * (M[0] * M[0] + M[1] * M[1] + M[2] * M[2]) / rho) / cv;
  }
  PetscCall(VecRestoreArrayRead(Q_cons, &U));
  PetscCall(VecRestoreArrayWrite(Q_prim, &Y));
  PetscFunctionReturn(0);
}

// Get error for problems with exact solutions
PetscErrorCode GetError_NS(CeedData ceed_data, DM dm, User user, Vec Q, PetscScalar final_time) {
  PetscInt       loc_nodes;
  Vec            Q_exact, Q_exact_loc, Q_loc;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (user->phys->ics_time_label) CeedOperatorContextSetDouble(ceed_data->op_ics, user->phys->ics_time_label, &final_time);

  // Get exact solution at final time
  ierr = DMCreateGlobalVector(dm, &Q_exact);
  CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &Q_exact_loc);
  CHKERRQ(ierr);
  ierr = VecGetSize(Q_exact_loc, &loc_nodes);
  CHKERRQ(ierr);
  ierr = ICs_FixMultiplicity(dm, ceed_data, user, Q_exact_loc, Q_exact, final_time);
  CHKERRQ(ierr);

  PetscCall(DMGlobalToLocal(dm, Q_exact, INSERT_VALUES, Q_exact_loc));
  PetscCall(DMPlexInsertBoundaryValues(dm, PETSC_TRUE, Q_exact_loc, final_time, NULL, NULL, NULL));

  ierr = DMGetLocalVector(dm, &Q_loc);
  CHKERRQ(ierr);
  PetscCall(DMGlobalToLocal(dm, Q, INSERT_VALUES, Q_loc));
  PetscCall(DMPlexInsertBoundaryValues(dm, PETSC_TRUE, Q_loc, final_time, NULL, NULL, NULL));

  NormType norm_type   = NORM_MAX;
  MPI_Op   norm_reduce = MPI_MAX;
  // Get |exact solution - obtained solution|
  Vec Q_target_exact, Q_target;

  ierr = VecDuplicate(Q_loc, &Q_target);
  CHKERRQ(ierr);
  ierr = VecDuplicate(Q_exact_loc, &Q_target_exact);
  CHKERRQ(ierr);

  if (user->phys->state_var == STATEVAR_PRIMITIVE) {
    ierr = GetConservativeFromPrimitive_NS(Q_loc, Q_target);
    CHKERRQ(ierr);
    ierr = GetConservativeFromPrimitive_NS(Q_exact_loc, Q_target_exact);
    CHKERRQ(ierr);
  } else {
    PetscCall(GetPrimitiveFromConservative_NS(Q_loc, Q_target));
    PetscCall(GetPrimitiveFromConservative_NS(Q_exact_loc, Q_target_exact));
  }
  // Get norm of each component
  PetscReal norm_exact[5], norm_error[5], rel_error[5];
  ierr = VecStrideNormAll(Q_target_exact, norm_type, norm_exact);
  CHKERRQ(ierr);
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, norm_exact, 5, MPIU_REAL, norm_reduce, PETSC_COMM_WORLD));
  ierr = VecAXPY(Q_target, -1.0, Q_target_exact);
  CHKERRQ(ierr);
  ierr = VecStrideNormAll(Q_target, norm_type, norm_error);
  CHKERRQ(ierr);
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, norm_error, 5, MPIU_REAL, norm_reduce, PETSC_COMM_WORLD));
  if (user->phys->state_var == STATEVAR_PRIMITIVE) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Relative Error converted from primitive to conservative:\n");
    CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Relative Error converted from conservative to primitive:\n");
    CHKERRQ(ierr);
  }
  for (int i = 0; i < 5; i++) {
    rel_error[i] = norm_error[i] / norm_exact[i];
    ierr         = PetscPrintf(PETSC_COMM_WORLD, "Component %d: %g (%g, %g)\n", i, (double)rel_error[i], norm_error[i], norm_exact[i]);
    CHKERRQ(ierr);
  }
  PetscCall(VecDestroy(&Q_target_exact));
  PetscCall(VecDestroy(&Q_target));
  // Report errors in source variables
  ierr = VecStrideNormAll(Q_exact_loc, norm_type, norm_exact);
  CHKERRQ(ierr);
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, norm_exact, 5, MPIU_REAL, norm_reduce, PETSC_COMM_WORLD));
  ierr = VecAXPY(Q_loc, -1.0, Q_exact_loc);
  CHKERRQ(ierr);
  ierr = VecStrideNormAll(Q_loc, norm_type, norm_error);
  CHKERRQ(ierr);
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, norm_error, 5, MPIU_REAL, norm_reduce, PETSC_COMM_WORLD));
  if (user->phys->state_var == STATEVAR_PRIMITIVE) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Relative Error in primitive:\n");
    CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Relative Error in conservative:\n");
    CHKERRQ(ierr);
  }
  for (int i = 0; i < 5; i++) {
    rel_error[i] = norm_error[i] / norm_exact[i];
    ierr         = PetscPrintf(PETSC_COMM_WORLD, "Component %d: %g (%g, %g)\n", i, (double)rel_error[i], norm_error[i], norm_exact[i]);
    CHKERRQ(ierr);
  }
  // Cleanup
  ierr = DMRestoreLocalVector(dm, &Q_exact_loc);
  CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &Q_loc);
  CHKERRQ(ierr);
  ierr = VecDestroy(&Q_exact);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode PostProcess_NS(TS ts, CeedData ceed_data, DM dm, ProblemData *problem, User user, Vec Q, PetscScalar final_time) {
  PetscInt steps;
  PetscFunctionBegin;

  // Print relative error
  if (problem->non_zero_time && !user->app_ctx->test_mode) {
    PetscCall(GetError_NS(ceed_data, dm, user, Q, final_time));
  }

  // Print final time and number of steps
  PetscCall(TSGetStepNumber(ts, &steps));
  if (!user->app_ctx->test_mode) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Time integrator took %" PetscInt_FMT " time steps to reach final time %g\n", steps, (double)final_time));
  }

  // Output numerical values from command line
  PetscCall(VecViewFromOptions(Q, NULL, "-vec_view"));

  // Compare reference solution values with current test run for CI
  if (user->app_ctx->test_mode) {
    PetscCall(RegressionTests_NS(user->app_ctx, Q));
  }

  PetscFunctionReturn(0);
}

// Gather initial Q values in case of continuation of simulation
PetscErrorCode SetupICsFromBinary(MPI_Comm comm, AppCtx app_ctx, Vec Q) {
  PetscViewer viewer;

  PetscFunctionBegin;

  // Read input
  PetscCall(PetscViewerBinaryOpen(comm, app_ctx->cont_file, FILE_MODE_READ, &viewer));

  // Load Q from existent solution
  PetscCall(VecLoad(Q, viewer));

  // Cleanup
  PetscCall(PetscViewerDestroy(&viewer));

  PetscFunctionReturn(0);
}

// Record boundary values from initial condition
PetscErrorCode SetBCsFromICs_NS(DM dm, Vec Q, Vec Q_loc) {
  Vec Qbc, boundary_mask;
  PetscFunctionBegin;

  PetscCall(DMGetNamedLocalVector(dm, "Qbc", &Qbc));
  PetscCall(VecCopy(Q_loc, Qbc));
  PetscCall(VecZeroEntries(Q_loc));
  PetscCall(DMGlobalToLocal(dm, Q, INSERT_VALUES, Q_loc));
  PetscCall(VecAXPY(Qbc, -1., Q_loc));
  PetscCall(DMRestoreNamedLocalVector(dm, "Qbc", &Qbc));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm, "DMPlexInsertBoundaryValues_C", DMPlexInsertBoundaryValues_NS));

  PetscCall(DMGetNamedLocalVector(dm, "boundary mask", &boundary_mask));
  PetscCall(DMGetGlobalVector(dm, &Q));
  PetscCall(VecZeroEntries(boundary_mask));
  PetscCall(VecSet(Q, 1.0));
  PetscCall(DMGlobalToLocal(dm, Q, INSERT_VALUES, boundary_mask));
  PetscCall(DMRestoreNamedLocalVector(dm, "boundary mask", &boundary_mask));

  PetscFunctionReturn(0);
}

// Free a plain data context that was allocated using PETSc; returning libCEED error codes
int FreeContextPetsc(void *data) {
  if (PetscFree(data)) return CeedError(NULL, CEED_ERROR_ACCESS, "PetscFree failed");
  return CEED_ERROR_SUCCESS;
}
