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
  PetscInt Q_size;
  PetscInt num_comp = 5;
  PetscScalar cv = 2.5;
  PetscErrorCode ierr;
  const PetscScalar *Y;
  PetscScalar *U;

  PetscFunctionBegin;
  ierr = VecGetSize(Q_prim, &Q_size); CHKERRQ(ierr);
  PetscCall(VecGetArrayRead(Q_prim, &Y));
  PetscCall(VecGetArrayWrite(Q_cons, &U));
  for (PetscInt i=0; i<Q_size; i+=num_comp) {
    // Primitive variables
    PetscScalar P = Y[i+0], u[3] = {Y[i+1], Y[i+2], Y[i+3]}, T = Y[i+4];
    PetscScalar rho = P / T;
    U[i+0] = rho;
    for (int j=0; j<3; j++) U[i+1+j] = rho * u[j];
    U[i+4] = rho * (cv * T + .5*(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]));
  }
  PetscCall(VecRestoreArrayRead(Q_prim, &Y));
  PetscCall(VecRestoreArrayWrite(Q_cons, &U));
  PetscFunctionReturn(0);
}

PetscErrorCode GetPrimitiveFromConservative_NS(Vec Q_cons, Vec Q_prim) {
  PetscInt Q_size;
  PetscInt num_comp = 5;
  PetscScalar cv = 2.5;
  PetscErrorCode ierr;
  const PetscScalar *U;
  PetscScalar *Y;

  PetscFunctionBegin;
  ierr = VecGetSize(Q_cons, &Q_size); CHKERRQ(ierr);
  PetscCall(VecGetArrayRead(Q_cons, &U));
  PetscCall(VecGetArrayWrite(Q_prim, &Y));
  for (PetscInt i=0; i<Q_size; i+=num_comp) {
    // Conservative variables
    PetscScalar rho = U[i+0], M[3] = {U[i+1], U[i+2], U[i+3]}, E = U[i+4];
    // Primitive variables
    for (int j=0; j<3; j++) Y[i+1+j] = M[j] / rho;
    Y[i+4] = (E - 0.5*(M[0]*M[0] + M[1]*M[1] + M[2]*M[2])/rho) / (cv*rho);
    Y[i+0] = (E - 0.5*(M[0]*M[0] + M[1]*M[1] + M[2]*M[2])/rho) / cv;
  }
  PetscCall(VecRestoreArrayRead(Q_cons, &U));
  PetscCall(VecRestoreArrayWrite(Q_prim, &Y));
  PetscFunctionReturn(0);
}

PetscErrorCode GetError_NS(CeedData ceed_data, DM dm, User user, Vec Q, PetscScalar final_time) {
  PetscInt  loc_nodes;
  Vec       Q_exact, Q_exact_loc;
  Vec       Q_loc = Q;
  PetscReal rel_error, norm_error, norm_exact;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  // Get exact solution at final time
  PetscCall(DMCreateGlobalVector(dm, &Q_exact));
  PetscCall(DMGetLocalVector(dm, &Q_exact_loc));
  PetscCall(VecGetSize(Q_exact_loc, &loc_nodes));
  PetscCall(ICs_FixMultiplicity(dm, ceed_data, user, Q_exact_loc, Q_exact, final_time));

  if(user->phys->state_var == STATEVAR_PRIMITIVE) { //If primitive, get conservative
    Vec Q_cons, Q_cons_exact;
    PetscReal rel_error_cons, norm_error_cons, norm_exact_cons;
    
    ierr = VecDuplicate(Q_loc, &Q_cons); CHKERRQ(ierr);
    ierr = GetConservativeFromPrimitive_NS(Q_loc, Q_cons); CHKERRQ(ierr);

    ierr = VecDuplicate(Q_exact, &Q_cons_exact); CHKERRQ(ierr);
    ierr = GetConservativeFromPrimitive_NS(Q_exact, Q_cons_exact); CHKERRQ(ierr);

    // Get |exact solution - obtained solution|
    PetscCall(VecNorm(Q_exact, NORM_1, &norm_exact));
    PetscCall(VecAXPY(Q_loc, -1.0, Q_exact));
    PetscCall(VecNorm(Q_loc, NORM_1, &norm_error));

    PetscCall(VecNorm(Q_cons_exact, NORM_1, &norm_exact_cons));
    PetscCall(VecAXPY(Q_cons, -1.0, Q_cons_exact));
    PetscCall(VecNorm(Q_cons, NORM_1, &norm_error_cons));

    // Compute relative error
    rel_error = norm_error / norm_exact;
    rel_error_cons = norm_error_cons / norm_exact_cons;

    // Output relative error
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Relative Error in primitive variables: %g\n", (double)rel_error));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Relative Error in conservative variables:: %g\n", (double)rel_error_cons));
    
  } else {
    Vec Q_prim, Q_prim_exact;
    PetscReal rel_error_prim, norm_error_prim, norm_exact_prim;
    
    ierr = VecDuplicate(Q_loc, &Q_prim); CHKERRQ(ierr);
    ierr = GetPrimitiveFromConservative_NS(Q_loc, Q_prim); CHKERRQ(ierr);

    ierr = VecDuplicate(Q_exact, &Q_prim_exact); CHKERRQ(ierr);
    ierr = GetPrimitiveFromConservative_NS(Q_exact, Q_prim_exact); CHKERRQ(ierr);

    // Get |exact solution - obtained solution|
    PetscCall(VecNorm(Q_exact, NORM_1, &norm_exact));
    PetscCall(VecAXPY(Q_loc, -1.0, Q_exact));
    PetscCall(VecNorm(Q_loc, NORM_1, &norm_error));

    PetscCall(VecNorm(Q_prim_exact, NORM_1, &norm_exact_prim));
    PetscCall(VecAXPY(Q_prim, -1.0, Q_prim_exact));
    PetscCall(VecNorm(Q_prim, NORM_1, &norm_error_prim));

    // Compute relative error
    rel_error = norm_error / norm_exact;
    rel_error_prim = norm_error_prim / norm_exact_prim;

    // Output relative error
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Relative Error in primitive variables: %g\n", (double)rel_error_prim));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Relative Error in conservative variables: %g\n", (double)rel_error));
  }

  // Cleanup
  PetscCall(DMRestoreLocalVector(dm, &Q_exact_loc));
  PetscCall(VecDestroy(&Q_exact));
  
  PetscFunctionReturn(0);
}

// Post-processing
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
