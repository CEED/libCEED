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
  if (user->phys->ics_time_label) CeedOperatorSetContextDouble(ceed_data->op_ics_ctx->op, user->phys->ics_time_label, &time);

  // ---------------------------------------------------------------------------
  // ICs
  // ---------------------------------------------------------------------------
  // -- CEED Restriction
  CeedVector q0_ceed;
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &q0_ceed, NULL);

  // -- Place PETSc vector in CEED vector
  PetscCall(ApplyCeedOperatorLocalToGlobal(NULL, Q, ceed_data->op_ics_ctx));

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

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexInsertBoundaryValues_NS(DM dm, PetscBool insert_essential, Vec Q_loc, PetscReal time, Vec face_geom_FVM, Vec cell_geom_FVM,
                                             Vec grad_FVM) {
  Vec Qbc, boundary_mask;
  PetscFunctionBegin;

  // Mask (zero) Strong BC entries
  PetscCall(DMGetNamedLocalVector(dm, "boundary mask", &boundary_mask));
  PetscCall(VecPointwiseMult(Q_loc, Q_loc, boundary_mask));
  PetscCall(DMRestoreNamedLocalVector(dm, "boundary mask", &boundary_mask));

  PetscCall(DMGetNamedLocalVector(dm, "Qbc", &Qbc));
  PetscCall(VecAXPY(Q_loc, 1., Qbc));
  PetscCall(DMRestoreNamedLocalVector(dm, "Qbc", &Qbc));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Load vector from binary file, possibly with embedded solution time and step number
PetscErrorCode LoadFluidsBinaryVec(MPI_Comm comm, PetscViewer viewer, Vec Q, PetscReal *time, PetscInt *step_number) {
  PetscInt   file_step_number;
  PetscInt32 token;
  PetscReal  file_time;
  PetscFunctionBeginUser;

  // Attempt
  PetscCall(PetscViewerBinaryRead(viewer, &token, 1, NULL, PETSC_INT32));
  if (token == FLUIDS_FILE_TOKEN_32 || token == FLUIDS_FILE_TOKEN_64 ||
      token == FLUIDS_FILE_TOKEN) {  // New style format; we're reading a file with step number and time in the header
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

  PetscFunctionReturn(PETSC_SUCCESS);
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

  PetscFunctionReturn(PETSC_SUCCESS);
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

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

const PetscInt32 FLUIDS_FILE_TOKEN    = 0xceedf00;  // for backwards compatibility
const PetscInt32 FLUIDS_FILE_TOKEN_32 = 0xceedf32;
const PetscInt32 FLUIDS_FILE_TOKEN_64 = 0xceedf64;

// Gather initial Q values in case of continuation of simulation
PetscErrorCode SetupICsFromBinary(MPI_Comm comm, AppCtx app_ctx, Vec Q) {
  PetscViewer viewer;

  PetscFunctionBegin;

  PetscCall(PetscViewerBinaryOpen(comm, app_ctx->cont_file, FILE_MODE_READ, &viewer));
  PetscCall(LoadFluidsBinaryVec(comm, viewer, Q, &app_ctx->cont_time, &app_ctx->cont_steps));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscFunctionReturn(PETSC_SUCCESS);
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

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Free a plain data context that was allocated using PETSc; returning libCEED error codes
int FreeContextPetsc(void *data) {
  if (PetscFree(data)) return CeedError(NULL, CEED_ERROR_ACCESS, "PetscFree failed");
  return CEED_ERROR_SUCCESS;
}

// Return mass qfunction specification for number of components N
PetscErrorCode CreateMassQFunction(Ceed ceed, CeedInt N, CeedInt q_data_size, CeedQFunction *qf) {
  PetscFunctionBeginUser;

  switch (N) {
    case 1:
      CeedQFunctionCreateInterior(ceed, 1, Mass_1, Mass_1_loc, qf);
      break;
    case 5:
      CeedQFunctionCreateInterior(ceed, 1, Mass_5, Mass_5_loc, qf);
      break;
    case 7:
      CeedQFunctionCreateInterior(ceed, 1, Mass_7, Mass_7_loc, qf);
      break;
    case 9:
      CeedQFunctionCreateInterior(ceed, 1, Mass_9, Mass_9_loc, qf);
      break;
    case 22:
      CeedQFunctionCreateInterior(ceed, 1, Mass_22, Mass_22_loc, qf);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Could not find mass qfunction of size %d", N);
  }

  CeedQFunctionAddInput(*qf, "u", N, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(*qf, "qdata", q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(*qf, "v", N, CEED_EVAL_INTERP);
  PetscFunctionReturn(PETSC_SUCCESS);
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
PetscErrorCode ComputeL2Projection(Vec source_vec, Vec target_vec, OperatorApplyContext rhs_matop_ctx, KSP ksp) {
  PetscFunctionBeginUser;

  PetscCall(ApplyCeedOperatorGlobalToGlobal(source_vec, target_vec, rhs_matop_ctx));
  PetscCall(KSPSolve(ksp, target_vec, target_vec));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NodalProjectionDataDestroy(NodalProjectionData context) {
  PetscFunctionBeginUser;
  if (context == NULL) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(DMDestroy(&context->dm));
  PetscCall(KSPDestroy(&context->ksp));

  PetscCall(OperatorApplyContextDestroy(context->l2_rhs_ctx));

  PetscCall(PetscFree(context));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * @brief Open a PHASTA *.dat file, grabbing dimensions and file pointer
 *
 * This function opens the file specified by `path` using `PetscFOpen` and passes the file pointer in `fp`.
 * It is not closed in this function, thus `fp` must be closed sometime after this function has been called (using `PetscFClose` for example).
 *
 * Assumes that the first line of the file has the number of rows and columns as the only two entries, separated by a single space.
 *
 * @param[in]  comm           MPI_Comm for the program
 * @param[in]  path           Path to the file
 * @param[in]  char_array_len Length of the character array that should contain each line
 * @param[out] dims           Dimensions of the file, taken from the first line of the file
 * @param[out] fp File        pointer to the opened file
 */
PetscErrorCode PHASTADatFileOpen(const MPI_Comm comm, const char path[PETSC_MAX_PATH_LEN], const PetscInt char_array_len, PetscInt dims[2],
                                 FILE **fp) {
  int    ndims;
  char   line[char_array_len];
  char **array;

  PetscFunctionBeginUser;
  PetscCall(PetscFOpen(comm, path, "r", fp));
  PetscCall(PetscSynchronizedFGets(comm, *fp, char_array_len, line));
  PetscCall(PetscStrToArray(line, ' ', &ndims, &array));
  PetscCheck(ndims == 2, comm, PETSC_ERR_FILE_UNEXPECTED, "Found %d dimensions instead of 2 on the first line of %s", ndims, path);

  for (PetscInt i = 0; i < ndims; i++) dims[i] = atoi(array[i]);
  PetscCall(PetscStrToArrayDestroy(ndims, array));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * @brief Get the number of rows for the PHASTA file at path.
 *
 * Assumes that the first line of the file has the number of rows and columns as the only two entries, separated by a single space.
 *
 * @param[in]  comm  MPI_Comm for the program
 * @param[in]  path  Path to the file
 * @param[out] nrows Number of rows
 */
PetscErrorCode PHASTADatFileGetNRows(const MPI_Comm comm, const char path[PETSC_MAX_PATH_LEN], PetscInt *nrows) {
  const PetscInt char_array_len = 512;
  PetscInt       dims[2];
  FILE          *fp;

  PetscFunctionBeginUser;
  PetscCall(PHASTADatFileOpen(comm, path, char_array_len, dims, &fp));
  *nrows = dims[0];
  PetscCall(PetscFClose(comm, fp));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PHASTADatFileReadToArrayReal(MPI_Comm comm, const char path[PETSC_MAX_PATH_LEN], PetscReal array[]) {
  PetscInt       dims[2];
  int            ndims;
  FILE          *fp;
  const PetscInt char_array_len = 512;
  char           line[char_array_len];
  char         **row_array;
  PetscFunctionBeginUser;

  PetscCall(PHASTADatFileOpen(comm, path, char_array_len, dims, &fp));

  for (PetscInt i = 0; i < dims[0]; i++) {
    PetscCall(PetscSynchronizedFGets(comm, fp, char_array_len, line));
    PetscCall(PetscStrToArray(line, ' ', &ndims, &row_array));
    PetscCheck(ndims == dims[1], comm, PETSC_ERR_FILE_UNEXPECTED,
               "Line %" PetscInt_FMT " of %s does not contain enough columns (%d instead of %" PetscInt_FMT ")", i, path, ndims, dims[1]);

    for (PetscInt j = 0; j < dims[1]; j++) {
      array[i * dims[1] + j] = (PetscReal)atof(row_array[j]);
    }
  }

  PetscCall(PetscFClose(comm, fp));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscLogEvent       FLUIDS_CeedOperatorApply;
PetscLogEvent       FLUIDS_CeedOperatorAssemble;
PetscLogEvent       FLUIDS_CeedOperatorAssembleDiagonal;
PetscLogEvent       FLUIDS_CeedOperatorAssemblePointBlockDiagonal;
static PetscClassId libCEED_classid;

PetscErrorCode RegisterLogEvents() {
  PetscFunctionBeginUser;
  PetscCall(PetscClassIdRegister("libCEED", &libCEED_classid));
  PetscCall(PetscLogEventRegister("CeedOpApply", libCEED_classid, &FLUIDS_CeedOperatorApply));
  PetscCall(PetscLogEventRegister("CeedOpAsm", libCEED_classid, &FLUIDS_CeedOperatorAssemble));
  PetscCall(PetscLogEventRegister("CeedOpAsmD", libCEED_classid, &FLUIDS_CeedOperatorAssembleDiagonal));
  PetscCall(PetscLogEventRegister("CeedOpAsmPBD", libCEED_classid, &FLUIDS_CeedOperatorAssemblePointBlockDiagonal));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Translate array of CeedInt to PetscInt.
    If the types differ, `array_ceed` is freed with `free()` and `array_petsc` is allocated with `malloc()`.
    Caller is responsible for freeing `array_petsc` with `free()`.

  @param[in]      num_entries  Number of array entries
  @param[in,out]  array_ceed   Array of CeedInts
  @param[out]     array_petsc  Array of PetscInts
**/
PetscErrorCode IntArrayC2P(PetscInt num_entries, CeedInt **array_ceed, PetscInt **array_petsc) {
  CeedInt  int_c = 0;
  PetscInt int_p = 0;

  PetscFunctionBeginUser;
  if (sizeof(int_c) == sizeof(int_p)) {
    *array_petsc = (PetscInt *)*array_ceed;
  } else {
    *array_petsc = malloc(num_entries * sizeof(PetscInt));
    for (PetscInt i = 0; i < num_entries; i++) (*array_petsc)[i] = (*array_ceed)[i];
    free(*array_ceed);
  }
  *array_ceed = NULL;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Translate array of PetscInt to CeedInt.
    If the types differ, `array_petsc` is freed with `PetscFree()` and `array_ceed` is allocated with `PetscMalloc1()`.
    Caller is responsible for freeing `array_ceed` with `PetscFree()`.

  @param[in]      num_entries  Number of array entries
  @param[in,out]  array_petsc  Array of PetscInts
  @param[out]     array_ceed   Array of CeedInts
**/
PetscErrorCode IntArrayP2C(PetscInt num_entries, PetscInt **array_petsc, CeedInt **array_ceed) {
  CeedInt  int_c = 0;
  PetscInt int_p = 0;

  PetscFunctionBeginUser;
  if (sizeof(int_c) == sizeof(int_p)) {
    *array_ceed = (CeedInt *)*array_petsc;
  } else {
    PetscCall(PetscMalloc1(num_entries, array_ceed));
    for (PetscInt i = 0; i < num_entries; i++) (*array_ceed)[i] = (*array_petsc)[i];
    PetscCall(PetscFree(*array_petsc));
  }
  *array_petsc = NULL;

  PetscFunctionReturn(PETSC_SUCCESS);
}
