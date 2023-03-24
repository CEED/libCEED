// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Miscellaneous utility functions

#include <ceed.h>
#include <petscdm.h>
#include <petscts.h>

#include "../navierstokes.h"
#include "../qfunctions/mass.h"

PetscErrorCode ICs_FixMultiplicity(DM dm, CeedData ceed_data, User user, Vec Q_loc, Vec Q, CeedScalar time) {
  PetscFunctionBeginUser;

  // ---------------------------------------------------------------------------
  // Update time for evaluation
  // ---------------------------------------------------------------------------
  if (user->phys->ics_time_label) CeedOperatorSetContextDouble(ceed_data->op_ics, user->phys->ics_time_label, &time);

  // ---------------------------------------------------------------------------
  // ICs
  // ---------------------------------------------------------------------------
  // -- CEED Restriction
  CeedVector q0_ceed;
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &q0_ceed, NULL);

  // -- Place PETSc vector in CEED vector
  PetscMemType q0_mem_type;
  PetscCall(VecP2C(Q_loc, &q0_mem_type, q0_ceed));

  // -- Apply CEED Operator
  CeedOperatorApply(ceed_data->op_ics, ceed_data->x_coord, q0_ceed, CEED_REQUEST_IMMEDIATE);

  // -- Restore vectors
  PetscCall(VecC2P(q0_ceed, q0_mem_type, Q_loc));

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
  PetscMemType m_mem_type;
  Vec          multiplicity_loc;
  PetscCall(DMGetLocalVector(dm, &multiplicity_loc));
  PetscCall(VecP2C(multiplicity_loc, &m_mem_type, mult_vec));

  // -- Get multiplicity
  CeedElemRestrictionGetMultiplicity(ceed_data->elem_restr_q, mult_vec);

  // -- Restore vectors
  PetscCall(VecC2P(mult_vec, m_mem_type, multiplicity_loc));

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

// @brief Load vector from binary file, possibly with embedded solution time and step number
PetscErrorCode LoadFluidsBinaryVec(MPI_Comm comm, PetscViewer viewer, Vec Q, PetscReal *time, PetscInt *step_number) {
  PetscInt  token, file_step_number;
  PetscReal file_time;
  PetscFunctionBeginUser;

  // Attempt
  PetscCall(PetscViewerBinaryRead(viewer, &token, 1, NULL, PETSC_INT));
  if (token == FLUIDS_FILE_TOKEN) {  // New style format; we're reading a file with step number and time in the header
    PetscCall(PetscViewerBinaryRead(viewer, &file_step_number, 1, NULL, PETSC_INT));
    PetscCall(PetscViewerBinaryRead(viewer, &file_time, 1, NULL, PETSC_REAL));
    if (time) *time = file_time;
    if (step_number) *step_number = file_step_number;
  } else if (token == VEC_FILE_CLASSID) {  // Legacy format of just the vector, encoded as [VEC_FILE_CLASSID, length, ]
    PetscInt length, N;
    PetscCall(PetscViewerBinaryRead(viewer, &length, 1, NULL, PETSC_INT));
    PetscCall(VecGetSize(Q, &N));
    PetscCheck(length == N, comm, PETSC_ERR_ARG_INCOMP, "File Vec has length %" PetscInt_FMT " but DM has global Vec size %" PetscInt_FMT, length, N);
    PetscCall(PetscViewerBinarySetSkipHeader(viewer, PETSC_TRUE));
  } else SETERRQ(comm, PETSC_ERR_FILE_UNEXPECTED, "Not a fluids header token or a PETSc Vec in file");

  // Load Q from existent solution
  PetscCall(VecLoad(Q, viewer));

  PetscFunctionReturn(0);
}

// Compare reference solution values with current test run for CI
PetscErrorCode RegressionTests_NS(AppCtx app_ctx, Vec Q) {
  Vec         Qref;
  PetscViewer viewer;
  PetscReal   error, Qrefnorm;
  MPI_Comm    comm = PetscObjectComm((PetscObject)Q);
  PetscFunctionBegin;

  // Read reference file
  PetscCall(VecDuplicate(Q, &Qref));
  PetscCall(PetscViewerBinaryOpen(comm, app_ctx->test_file_path, FILE_MODE_READ, &viewer));
  PetscCall(LoadFluidsBinaryVec(comm, viewer, Qref, NULL, NULL));

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

// Get error for problems with exact solutions
PetscErrorCode GetError_NS(CeedData ceed_data, DM dm, User user, Vec Q, PetscScalar final_time) {
  PetscInt  loc_nodes;
  Vec       Q_exact, Q_exact_loc;
  PetscReal rel_error, norm_error, norm_exact;
  PetscFunctionBegin;

  // Get exact solution at final time
  PetscCall(DMCreateGlobalVector(dm, &Q_exact));
  PetscCall(DMGetLocalVector(dm, &Q_exact_loc));
  PetscCall(VecGetSize(Q_exact_loc, &loc_nodes));
  PetscCall(ICs_FixMultiplicity(dm, ceed_data, user, Q_exact_loc, Q_exact, final_time));

  // Get |exact solution - obtained solution|
  PetscCall(VecNorm(Q_exact, NORM_1, &norm_exact));
  PetscCall(VecAXPY(Q, -1.0, Q_exact));
  PetscCall(VecNorm(Q, NORM_1, &norm_error));

  // Compute relative error
  rel_error = norm_error / norm_exact;

  // Output relative error
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Relative Error: %g\n", (double)rel_error));
  // Cleanup
  PetscCall(DMRestoreLocalVector(dm, &Q_exact_loc));
  PetscCall(VecDestroy(&Q_exact));

  PetscFunctionReturn(0);
}

// Post-processing
PetscErrorCode PostProcess_NS(TS ts, CeedData ceed_data, DM dm, ProblemData *problem, User user, Vec Q, PetscScalar final_time) {
  PetscInt          steps;
  TSConvergedReason reason;
  PetscFunctionBegin;

  // Print relative error
  if (problem->non_zero_time && user->app_ctx->test_type == TESTTYPE_NONE) {
    PetscCall(GetError_NS(ceed_data, dm, user, Q, final_time));
  }

  // Print final time and number of steps
  PetscCall(TSGetStepNumber(ts, &steps));
  PetscCall(TSGetConvergedReason(ts, &reason));
  if (user->app_ctx->test_type == TESTTYPE_NONE) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Time integrator %s on time step %" PetscInt_FMT " with final time %g\n", TSConvergedReasons[reason],
                          steps, (double)final_time));
  }

  // Output numerical values from command line
  PetscCall(VecViewFromOptions(Q, NULL, "-vec_view"));

  // Compare reference solution values with current test run for CI
  if (user->app_ctx->test_type == TESTTYPE_SOLVER) {
    PetscCall(RegressionTests_NS(user->app_ctx, Q));
  }

  PetscFunctionReturn(0);
}

const PetscInt FLUIDS_FILE_TOKEN = 0xceedf00;

// Gather initial Q values in case of continuation of simulation
PetscErrorCode SetupICsFromBinary(MPI_Comm comm, AppCtx app_ctx, Vec Q) {
  PetscViewer viewer;

  PetscFunctionBegin;

  PetscCall(PetscViewerBinaryOpen(comm, app_ctx->cont_file, FILE_MODE_READ, &viewer));
  PetscCall(LoadFluidsBinaryVec(comm, viewer, Q, &app_ctx->cont_time, &app_ctx->cont_steps));
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

// Return mass qfunction specification for number of components N
PetscErrorCode CreateMassQFunction(Ceed ceed, CeedInt N, CeedInt q_data_size, CeedQFunction *qf) {
  CeedQFunctionUser qfunction_ptr;
  const char       *qfunction_loc;
  PetscFunctionBeginUser;

  switch (N) {
    case 1:
      qfunction_ptr = Mass_1;
      qfunction_loc = Mass_1_loc;
      break;
    case 5:
      qfunction_ptr = Mass_5;
      qfunction_loc = Mass_5_loc;
      break;
    case 9:
      qfunction_ptr = Mass_9;
      qfunction_loc = Mass_9_loc;
      break;
    case 22:
      qfunction_ptr = Mass_22;
      qfunction_loc = Mass_22_loc;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, -1, "Could not find mass qfunction of size %d", N);
  }
  CeedQFunctionCreateInterior(ceed, 1, qfunction_ptr, qfunction_loc, qf);

  CeedQFunctionAddInput(*qf, "u", N, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(*qf, "qdata", q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(*qf, "v", N, CEED_EVAL_INTERP);
  PetscFunctionReturn(0);
}

/* @brief L^2 Projection of a source FEM function to a target FEM space
 *
 * To solve system using a lumped mass matrix, pass a KSP object with ksp_type=preonly, pc_type=jacobi, pc_jacobi_type=rowsum.
 *
 * @param[in]  source_vec    Global Vec of the source FEM function. NULL indicates using rhs_matop_ctx->X_loc
 * @param[out] target_vec    Global Vec of the target (result) FEM function. NULL indicates using rhs_matop_ctx->Y_loc
 * @param[in]  rhs_matop_ctx MatopApplyContext for performing the RHS evaluation
 * @param[in]  ksp           KSP for solving the consistent projection problem
 */
PetscErrorCode ComputeL2Projection(Vec source_vec, Vec target_vec, MatopApplyContext rhs_matop_ctx, KSP ksp) {
  PetscFunctionBeginUser;

  PetscCall(ApplyLocal_Ceed(source_vec, target_vec, rhs_matop_ctx));
  PetscCall(KSPSolve(ksp, target_vec, target_vec));

  PetscFunctionReturn(0);
}
