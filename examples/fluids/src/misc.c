// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Miscellaneous utility functions

#include "../navierstokes.h"

PetscErrorCode ICs_FixMultiplicity(DM dm, CeedData ceed_data, User user,
                                   Vec Q_loc, Vec Q,
                                   CeedScalar time) {
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // ---------------------------------------------------------------------------
  // Update time for evaluation
  // ---------------------------------------------------------------------------
  if (user->phys->ics_time_label)
    CeedOperatorContextSetDouble(ceed_data->op_ics, user->phys->ics_time_label,
                                 &time);

  // ---------------------------------------------------------------------------
  // ICs
  // ---------------------------------------------------------------------------
  // -- CEED Restriction
  CeedVector q0_ceed;
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &q0_ceed, NULL);

  // -- Place PETSc vector in CEED vector
  CeedScalar *q0;
  PetscMemType q0_mem_type;
  ierr = VecGetArrayAndMemType(Q_loc, (PetscScalar **)&q0, &q0_mem_type);
  CHKERRQ(ierr);
  CeedVectorSetArray(q0_ceed, MemTypeP2C(q0_mem_type), CEED_USE_POINTER, q0);

  // -- Apply CEED Operator
  CeedOperatorApply(ceed_data->op_ics, ceed_data->x_coord, q0_ceed,
                    CEED_REQUEST_IMMEDIATE);

  // -- Restore vectors
  CeedVectorTakeArray(q0_ceed, MemTypeP2C(q0_mem_type), NULL);
  ierr = VecRestoreArrayReadAndMemType(Q_loc, (const PetscScalar **)&q0);
  CHKERRQ(ierr);

  // -- Local-to-Global
  ierr = VecZeroEntries(Q); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dm, Q_loc, ADD_VALUES, Q); CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // Fix multiplicity for output of ICs
  // ---------------------------------------------------------------------------
  // -- CEED Restriction
  CeedVector mult_vec;
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &mult_vec, NULL);

  // -- Place PETSc vector in CEED vector
  CeedScalar *mult;
  PetscMemType m_mem_type;
  Vec multiplicity_loc;
  ierr = DMGetLocalVector(dm, &multiplicity_loc); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(multiplicity_loc, (PetscScalar **)&mult,
                               &m_mem_type);
  CHKERRQ(ierr);
  CeedVectorSetArray(mult_vec, MemTypeP2C(m_mem_type), CEED_USE_POINTER, mult);
  CHKERRQ(ierr);

  // -- Get multiplicity
  CeedElemRestrictionGetMultiplicity(ceed_data->elem_restr_q, mult_vec);

  // -- Restore vectors
  CeedVectorTakeArray(mult_vec, MemTypeP2C(m_mem_type), NULL);
  ierr = VecRestoreArrayReadAndMemType(multiplicity_loc,
                                       (const PetscScalar **)&mult); CHKERRQ(ierr);

  // -- Local-to-Global
  Vec multiplicity;
  ierr = DMGetGlobalVector(dm, &multiplicity); CHKERRQ(ierr);
  ierr = VecZeroEntries(multiplicity); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dm, multiplicity_loc, ADD_VALUES, multiplicity);
  CHKERRQ(ierr);

  // -- Fix multiplicity
  ierr = VecPointwiseDivide(Q, Q, multiplicity); CHKERRQ(ierr);
  ierr = VecPointwiseDivide(Q_loc, Q_loc, multiplicity_loc); CHKERRQ(ierr);

  // -- Restore vectors
  ierr = DMRestoreLocalVector(dm, &multiplicity_loc); CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &multiplicity); CHKERRQ(ierr);

  // Cleanup
  CeedVectorDestroy(&mult_vec);
  CeedVectorDestroy(&q0_ceed);

  PetscFunctionReturn(0);
}


// Note: The BCs must be inserted *before* the other values are inserted into Q_loc
PetscErrorCode DMPlexInsertBoundaryValues_NS(DM dm,
    PetscBool insert_essential, Vec Q_loc, PetscReal time, Vec face_geom_FVM,
    Vec cell_geom_FVM, Vec grad_FVM) {

  Vec            Qbc;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = DMGetNamedLocalVector(dm, "Qbc", &Qbc); CHKERRQ(ierr);
  ierr = VecZeroEntries(Q_loc); CHKERRQ(ierr);
  ierr = VecAXPY(Q_loc, 1., Qbc); CHKERRQ(ierr);
  ierr = DMRestoreNamedLocalVector(dm, "Qbc", &Qbc); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Compare reference solution values with current test run for CI
PetscErrorCode RegressionTests_NS(AppCtx app_ctx, Vec Q) {

  Vec            Qref;
  PetscViewer    viewer;
  PetscReal      error, Qrefnorm;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Read reference file
  ierr = VecDuplicate(Q, &Qref); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PetscObjectComm((PetscObject)Q),
                               app_ctx->file_path, FILE_MODE_READ,
                               &viewer); CHKERRQ(ierr);
  ierr = VecLoad(Qref, viewer); CHKERRQ(ierr);

  // Compute error with respect to reference solution
  ierr = VecAXPY(Q, -1.0, Qref); CHKERRQ(ierr);
  ierr = VecNorm(Qref, NORM_MAX, &Qrefnorm); CHKERRQ(ierr);
  ierr = VecScale(Q, 1./Qrefnorm); CHKERRQ(ierr);
  ierr = VecNorm(Q, NORM_MAX, &error); CHKERRQ(ierr);

  // Check error
  if (error > app_ctx->test_tol) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "Test failed with error norm %g\n",
                       (double)error); CHKERRQ(ierr);
  }

  // Cleanup
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  ierr = VecDestroy(&Qref); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Get error for problems with exact solutions
PetscErrorCode GetError_NS(CeedData ceed_data, DM dm, User user, Vec Q,
                           PetscScalar final_time) {
  PetscInt       loc_nodes;
  Vec            Q_exact, Q_exact_loc;
  PetscReal      rel_error, norm_error, norm_exact;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Get exact solution at final time
  ierr = DMCreateGlobalVector(dm, &Q_exact); CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &Q_exact_loc); CHKERRQ(ierr);
  ierr = VecGetSize(Q_exact_loc, &loc_nodes); CHKERRQ(ierr);
  ierr = ICs_FixMultiplicity(dm, ceed_data, user, Q_exact_loc, Q_exact,
                             final_time);
  CHKERRQ(ierr);

  // Get |exact solution - obtained solution|
  ierr = VecNorm(Q_exact, NORM_1, &norm_exact); CHKERRQ(ierr);
  ierr = VecAXPY(Q, -1.0, Q_exact);  CHKERRQ(ierr);
  ierr = VecNorm(Q, NORM_1, &norm_error); CHKERRQ(ierr);

  // Compute relative error
  rel_error = norm_error / norm_exact;

  // Output relative error
  ierr = PetscPrintf(PETSC_COMM_WORLD,
                     "Relative Error: %g\n",
                     (double)rel_error); CHKERRQ(ierr);
  // Cleanup
  ierr = DMRestoreLocalVector(dm, &Q_exact_loc); CHKERRQ(ierr);
  ierr = VecDestroy(&Q_exact); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Post-processing
PetscErrorCode PostProcess_NS(TS ts, CeedData ceed_data, DM dm,
                              ProblemData *problem, User user,
                              Vec Q, PetscScalar final_time) {
  PetscInt       steps;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Print relative error
  if (problem->non_zero_time && !user->app_ctx->test_mode) {
    ierr = GetError_NS(ceed_data, dm, user, Q, final_time); CHKERRQ(ierr);
  }

  // Print final time and number of steps
  ierr = TSGetStepNumber(ts, &steps); CHKERRQ(ierr);
  if (!user->app_ctx->test_mode) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "Time integrator took %" PetscInt_FMT " time steps to reach final time %g\n",
                       steps, (double)final_time); CHKERRQ(ierr);
  }

  // Output numerical values from command line
  ierr = VecViewFromOptions(Q, NULL, "-vec_view"); CHKERRQ(ierr);

  // Compare reference solution values with current test run for CI
  if (user->app_ctx->test_mode) {
    ierr = RegressionTests_NS(user->app_ctx, Q); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

// Gather initial Q values in case of continuation of simulation
PetscErrorCode SetupICsFromBinary(MPI_Comm comm, AppCtx app_ctx, Vec Q) {

  PetscViewer    viewer;
  char           file_path[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Read input
  ierr = PetscSNPrintf(file_path, sizeof file_path, "%s/ns-solution.bin",
                       app_ctx->output_dir); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm, file_path, FILE_MODE_READ, &viewer);
  CHKERRQ(ierr);

  // Load Q from existent solution
  ierr = VecLoad(Q, viewer); CHKERRQ(ierr);

  // Cleanup
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Record boundary values from initial condition
PetscErrorCode SetBCsFromICs_NS(DM dm, Vec Q, Vec Q_loc) {

  Vec            Qbc;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = DMGetNamedLocalVector(dm, "Qbc", &Qbc); CHKERRQ(ierr);
  ierr = VecCopy(Q_loc, Qbc); CHKERRQ(ierr);
  ierr = VecZeroEntries(Q_loc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm, Q, INSERT_VALUES, Q_loc); CHKERRQ(ierr);
  ierr = VecAXPY(Qbc, -1., Q_loc); CHKERRQ(ierr);
  ierr = DMRestoreNamedLocalVector(dm, "Qbc", &Qbc); CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)dm,
                                    "DMPlexInsertBoundaryValues_C", DMPlexInsertBoundaryValues_NS);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Free a plain data context that was allocated using PETSc; returning libCEED error codes
int FreeContextPetsc(void *data) {
  if (PetscFree(data)) return CeedError(NULL, CEED_ERROR_ACCESS,
                                          "PetscFree failed");
  return CEED_ERROR_SUCCESS;
}
