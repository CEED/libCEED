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
#include <petscsf.h>
#include <petscts.h>

#include "../navierstokes.h"
#include "../qfunctions/mass.h"

PetscErrorCode ICs_FixMultiplicity(DM dm, CeedData ceed_data, User user, Vec Q_loc, Vec Q, CeedScalar time) {
  Ceed         ceed = user->ceed;
  CeedVector   mult_vec;
  PetscMemType m_mem_type;
  Vec          Multiplicity, Multiplicity_loc;

  PetscFunctionBeginUser;
  if (user->phys->ics_time_label) PetscCallCeed(ceed, CeedOperatorSetContextDouble(ceed_data->op_ics_ctx->op, user->phys->ics_time_label, &time));
  PetscCall(ApplyCeedOperatorLocalToGlobal(NULL, Q, ceed_data->op_ics_ctx));

  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &mult_vec, NULL));

  // -- Get multiplicity
  PetscCall(DMGetLocalVector(dm, &Multiplicity_loc));
  PetscCall(VecPetscToCeed(Multiplicity_loc, &m_mem_type, mult_vec));
  PetscCallCeed(ceed, CeedElemRestrictionGetMultiplicity(ceed_data->elem_restr_q, mult_vec));
  PetscCall(VecCeedToPetsc(mult_vec, m_mem_type, Multiplicity_loc));

  PetscCall(DMGetGlobalVector(dm, &Multiplicity));
  PetscCall(VecZeroEntries(Multiplicity));
  PetscCall(DMLocalToGlobal(dm, Multiplicity_loc, ADD_VALUES, Multiplicity));

  // -- Fix multiplicity
  PetscCall(VecPointwiseDivide(Q, Q, Multiplicity));
  PetscCall(VecPointwiseDivide(Q_loc, Q_loc, Multiplicity_loc));

  PetscCall(DMRestoreLocalVector(dm, &Multiplicity_loc));
  PetscCall(DMRestoreGlobalVector(dm, &Multiplicity));
  PetscCallCeed(ceed, CeedVectorDestroy(&mult_vec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Record boundary values from initial condition
PetscErrorCode SetBCsFromICs(DM dm, Vec Q, Vec Q_loc) {
  Vec Qbc, boundary_mask;

  PetscFunctionBeginUser;
  PetscCall(DMGetNamedLocalVector(dm, "Qbc", &Qbc));
  PetscCall(VecCopy(Q_loc, Qbc));
  PetscCall(VecZeroEntries(Q_loc));
  PetscCall(DMGlobalToLocal(dm, Q, INSERT_VALUES, Q_loc));
  PetscCall(VecAXPY(Qbc, -1., Q_loc));
  PetscCall(DMRestoreNamedLocalVector(dm, "Qbc", &Qbc));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm, "DMPlexInsertBoundaryValues_C", DMPlexInsertBoundaryValues_FromICs));

  PetscCall(DMGetNamedLocalVector(dm, "boundary mask", &boundary_mask));
  PetscCall(DMGetGlobalVector(dm, &Q));
  PetscCall(VecZeroEntries(boundary_mask));
  PetscCall(VecSet(Q, 1.0));
  PetscCall(DMGlobalToLocal(dm, Q, INSERT_VALUES, boundary_mask));
  PetscCall(DMRestoreNamedLocalVector(dm, "boundary mask", &boundary_mask));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexInsertBoundaryValues_FromICs(DM dm, PetscBool insert_essential, Vec Q_loc, PetscReal time, Vec face_geom_FVM, Vec cell_geom_FVM,
                                                  Vec grad_FVM) {
  Vec Qbc, boundary_mask;

  PetscFunctionBeginUser;
  // Mask (zero) Strong BC entries
  PetscCall(DMGetNamedLocalVector(dm, "boundary mask", &boundary_mask));
  PetscCall(VecPointwiseMult(Q_loc, Q_loc, boundary_mask));
  PetscCall(DMRestoreNamedLocalVector(dm, "boundary mask", &boundary_mask));

  PetscCall(DMGetNamedLocalVector(dm, "Qbc", &Qbc));
  PetscCall(VecAXPY(Q_loc, 1., Qbc));
  PetscCall(DMRestoreNamedLocalVector(dm, "Qbc", &Qbc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BinaryReadIntoInt(PetscViewer viewer, PetscInt *out, PetscDataType file_type) {
  PetscFunctionBeginUser;
  if (file_type == PETSC_INT32) {
    PetscInt32 val;
    PetscCall(PetscViewerBinaryRead(viewer, &val, 1, NULL, PETSC_INT32));
    *out = val;
  } else if (file_type == FLUIDS_FILE_TOKEN_64) {
    PetscInt64 val;
    PetscCall(PetscViewerBinaryRead(viewer, &val, 1, NULL, PETSC_INT64));
    *out = val;
  } else {
    PetscCall(PetscViewerBinaryRead(viewer, &out, 1, NULL, PETSC_INT));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Load vector from binary file, possibly with embedded solution time and step number
PetscErrorCode LoadFluidsBinaryVec(MPI_Comm comm, PetscViewer viewer, Vec Q, PetscReal *time, PetscInt *step_number) {
  PetscInt      file_step_number;
  PetscInt32    token;
  PetscReal     file_time;
  PetscDataType file_type = PETSC_INT;

  PetscFunctionBeginUser;
  PetscCall(PetscViewerBinaryRead(viewer, &token, 1, NULL, PETSC_INT32));
  if (token == FLUIDS_FILE_TOKEN_32 || token == FLUIDS_FILE_TOKEN_64 ||
      token == FLUIDS_FILE_TOKEN) {  // New style format; we're reading a file with step number and time in the header
    if (token == FLUIDS_FILE_TOKEN_32) file_type = PETSC_INT32;
    else if (token == FLUIDS_FILE_TOKEN_64) file_type = PETSC_INT64;
    PetscCall(BinaryReadIntoInt(viewer, &file_step_number, file_type));
    PetscCall(PetscViewerBinaryRead(viewer, &file_time, 1, NULL, PETSC_REAL));
    if (time) *time = file_time;
    if (step_number) *step_number = file_step_number;
  } else if (token == VEC_FILE_CLASSID) {  // Legacy format of just the vector, encoded as [VEC_FILE_CLASSID, length, ]
    PetscInt length, N;
    PetscCall(BinaryReadIntoInt(viewer, &length, file_type));
    PetscCall(VecGetSize(Q, &N));
    PetscCheck(length == N, comm, PETSC_ERR_ARG_INCOMP, "File Vec has length %" PetscInt_FMT " but DM has global Vec size %" PetscInt_FMT, length, N);
    PetscCall(PetscViewerBinarySetSkipHeader(viewer, PETSC_TRUE));
  } else SETERRQ(comm, PETSC_ERR_FILE_UNEXPECTED, "Not a fluids header token or a PETSc Vec in file");

  PetscCall(VecLoad(Q, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Compare reference solution values with current test run for CI
PetscErrorCode RegressionTest(AppCtx app_ctx, Vec Q) {
  Vec         Qref;
  PetscViewer viewer;
  PetscReal   error, Qrefnorm;
  MPI_Comm    comm = PetscObjectComm((PetscObject)Q);

  PetscFunctionBeginUser;
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
PetscErrorCode PrintError(CeedData ceed_data, DM dm, User user, Vec Q, PetscScalar final_time) {
  PetscInt  loc_nodes;
  Vec       Q_exact, Q_exact_loc;
  PetscReal rel_error, norm_error, norm_exact;

  PetscFunctionBeginUser;
  // Get exact solution at final time
  PetscCall(DMGetGlobalVector(dm, &Q_exact));
  PetscCall(DMGetLocalVector(dm, &Q_exact_loc));
  PetscCall(VecGetSize(Q_exact_loc, &loc_nodes));
  PetscCall(ICs_FixMultiplicity(dm, ceed_data, user, Q_exact_loc, Q_exact, final_time));

  // Get |exact solution - obtained solution|
  PetscCall(VecNorm(Q_exact, NORM_1, &norm_exact));
  PetscCall(VecAXPY(Q, -1.0, Q_exact));
  PetscCall(VecNorm(Q, NORM_1, &norm_error));

  rel_error = norm_error / norm_exact;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Relative Error: %g\n", (double)rel_error));
  PetscCall(DMRestoreLocalVector(dm, &Q_exact_loc));
  PetscCall(DMRestoreGlobalVector(dm, &Q_exact));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Post-processing
PetscErrorCode PostProcess(TS ts, CeedData ceed_data, DM dm, ProblemData *problem, User user, Vec Q, PetscScalar final_time) {
  PetscInt          steps;
  TSConvergedReason reason;

  PetscFunctionBeginUser;
  // Print relative error
  if (problem->non_zero_time && user->app_ctx->test_type == TESTTYPE_NONE) {
    PetscCall(PrintError(ceed_data, dm, user, Q, final_time));
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
    PetscCall(RegressionTest(user->app_ctx, Q));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

const PetscInt32 FLUIDS_FILE_TOKEN    = 0xceedf00;  // for backwards compatibility
const PetscInt32 FLUIDS_FILE_TOKEN_32 = 0xceedf32;
const PetscInt32 FLUIDS_FILE_TOKEN_64 = 0xceedf64;

// Gather initial Q values in case of continuation of simulation
PetscErrorCode SetupICsFromBinary(MPI_Comm comm, AppCtx app_ctx, Vec Q) {
  PetscViewer viewer;

  PetscFunctionBeginUser;
  PetscCall(PetscViewerBinaryOpen(comm, app_ctx->cont_file, FILE_MODE_READ, &viewer));
  PetscCall(LoadFluidsBinaryVec(comm, viewer, Q, &app_ctx->cont_time, &app_ctx->cont_steps));
  PetscCall(PetscViewerDestroy(&viewer));
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
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Mass_1, Mass_1_loc, qf));
      break;
    case 5:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Mass_5, Mass_5_loc, qf));
      break;
    case 7:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Mass_7, Mass_7_loc, qf));
      break;
    case 9:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Mass_9, Mass_9_loc, qf));
      break;
    case 22:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Mass_22, Mass_22_loc, qf));
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Could not find mass qfunction of size %d", N);
  }

  PetscCallCeed(ceed, CeedQFunctionAddInput(*qf, "u", N, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(*qf, "qdata", q_data_size, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(*qf, "v", N, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionSetUserFlopsEstimate(*qf, N));
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
PetscErrorCode PhastaDatFileOpen(const MPI_Comm comm, const char path[PETSC_MAX_PATH_LEN], const PetscInt char_array_len, PetscInt dims[2],
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
PetscErrorCode PhastaDatFileGetNRows(const MPI_Comm comm, const char path[PETSC_MAX_PATH_LEN], PetscInt *nrows) {
  const PetscInt char_array_len = 512;
  PetscInt       dims[2];
  FILE          *fp;

  PetscFunctionBeginUser;
  PetscCall(PhastaDatFileOpen(comm, path, char_array_len, dims, &fp));
  *nrows = dims[0];
  PetscCall(PetscFClose(comm, fp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhastaDatFileReadToArrayReal(MPI_Comm comm, const char path[PETSC_MAX_PATH_LEN], PetscReal array[]) {
  PetscInt       dims[2];
  int            ndims;
  FILE          *fp;
  const PetscInt char_array_len = 512;
  char           line[char_array_len];
  char         **row_array;

  PetscFunctionBeginUser;
  PetscCall(PhastaDatFileOpen(comm, path, char_array_len, dims, &fp));

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
PetscLogEvent       FLUIDS_SmartRedis_Init;
PetscLogEvent       FLUIDS_SmartRedis_Meta;
PetscLogEvent       FLUIDS_SmartRedis_Train;
PetscLogEvent       FLUIDS_TrainDataCompute;
PetscLogEvent       FLUIDS_DifferentialFilter;
PetscLogEvent       FLUIDS_VelocityGradientProjection;
static PetscClassId libCEED_classid, onlineTrain_classid, misc_classid;

PetscErrorCode RegisterLogEvents() {
  PetscFunctionBeginUser;
  PetscCall(PetscClassIdRegister("libCEED", &libCEED_classid));
  PetscCall(PetscLogEventRegister("CeedOpApply", libCEED_classid, &FLUIDS_CeedOperatorApply));
  PetscCall(PetscLogEventRegister("CeedOpAsm", libCEED_classid, &FLUIDS_CeedOperatorAssemble));
  PetscCall(PetscLogEventRegister("CeedOpAsmD", libCEED_classid, &FLUIDS_CeedOperatorAssembleDiagonal));
  PetscCall(PetscLogEventRegister("CeedOpAsmPBD", libCEED_classid, &FLUIDS_CeedOperatorAssemblePointBlockDiagonal));

  PetscCall(PetscClassIdRegister("onlineTrain", &onlineTrain_classid));
  PetscCall(PetscLogEventRegister("SmartRedis_Init", onlineTrain_classid, &FLUIDS_SmartRedis_Init));
  PetscCall(PetscLogEventRegister("SmartRedis_Meta", onlineTrain_classid, &FLUIDS_SmartRedis_Meta));
  PetscCall(PetscLogEventRegister("SmartRedis_Train", onlineTrain_classid, &FLUIDS_SmartRedis_Train));
  PetscCall(PetscLogEventRegister("TrainDataCompute", onlineTrain_classid, &FLUIDS_TrainDataCompute));

  PetscCall(PetscClassIdRegister("Miscellaneous", &misc_classid));
  PetscCall(PetscLogEventRegister("DiffFilter", misc_classid, &FLUIDS_DifferentialFilter));
  PetscCall(PetscLogEventRegister("VeloGradProj", misc_classid, &FLUIDS_VelocityGradientProjection));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Print information about the given simulation run
PetscErrorCode PrintRunInfo(User user, Physics phys_ctx, ProblemData *problem, MPI_Comm comm) {
  Ceed ceed = user->ceed;
  PetscFunctionBeginUser;
  // Header and rank
  char        host_name[PETSC_MAX_PATH_LEN];
  PetscMPIInt rank, comm_size;
  PetscCall(PetscGetHostName(host_name, sizeof host_name));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &comm_size));
  PetscCall(PetscPrintf(comm,
                        "\n-- Navier-Stokes solver - libCEED + PETSc --\n"
                        "  MPI:\n"
                        "    Host Name                          : %s\n"
                        "    Total ranks                        : %d\n",
                        host_name, comm_size));

  // Problem specific info
  PetscCall(problem->print_info(user, problem, user->app_ctx));

  // libCEED
  const char *used_resource;
  CeedMemType mem_type_backend;
  PetscCallCeed(ceed, CeedGetResource(user->ceed, &used_resource));
  PetscCallCeed(ceed, CeedGetPreferredMemType(user->ceed, &mem_type_backend));
  PetscCall(PetscPrintf(comm,
                        "  libCEED:\n"
                        "    libCEED Backend                    : %s\n"
                        "    libCEED Backend MemType            : %s\n",
                        used_resource, CeedMemTypes[mem_type_backend]));
  // PETSc
  char box_faces_str[PETSC_MAX_PATH_LEN] = "3,3,3";
  if (problem->dim == 2) box_faces_str[3] = '\0';
  PetscCall(PetscOptionsGetString(NULL, NULL, "-dm_plex_box_faces", box_faces_str, sizeof(box_faces_str), NULL));
  MatType amat_type = user->app_ctx->amat_type, pmat_type;
  VecType vec_type;
  PetscCall(DMGetMatType(user->dm, &pmat_type));
  if (!amat_type) amat_type = pmat_type;
  PetscCall(DMGetVecType(user->dm, &vec_type));
  PetscCall(PetscPrintf(comm,
                        "  PETSc:\n"
                        "    Box Faces                          : %s\n"
                        "    A MatType                          : %s\n"
                        "    P MatType                          : %s\n"
                        "    DM VecType                         : %s\n"
                        "    Time Stepping Scheme               : %s\n",
                        box_faces_str, amat_type, pmat_type, vec_type, phys_ctx->implicit ? "implicit" : "explicit"));
  if (user->app_ctx->cont_steps) {
    PetscCall(PetscPrintf(comm,
                          "  Continue:\n"
                          "    Filename:                          : %s\n"
                          "    Step:                              : %" PetscInt_FMT "\n"
                          "    Time:                              : %g\n",
                          user->app_ctx->cont_file, user->app_ctx->cont_steps, user->app_ctx->cont_time));
  }
  // Mesh
  const PetscInt num_comp_q = 5;
  PetscInt       glob_dofs, owned_dofs, local_dofs;
  const CeedInt  num_P = user->app_ctx->degree + 1, num_Q = num_P + user->app_ctx->q_extra;
  PetscCall(DMGetGlobalVectorInfo(user->dm, &owned_dofs, &glob_dofs, NULL));
  PetscCall(DMGetLocalVectorInfo(user->dm, &local_dofs, NULL, NULL));
  PetscCall(PetscPrintf(comm,
                        "  Mesh:\n"
                        "    Number of 1D Basis Nodes (P)       : %" CeedInt_FMT "\n"
                        "    Number of 1D Quadrature Points (Q) : %" CeedInt_FMT "\n"
                        "    Global DoFs                        : %" PetscInt_FMT "\n"
                        "    DoFs per node                      : %" PetscInt_FMT "\n"
                        "    Global %" PetscInt_FMT "-DoF nodes                 : %" PetscInt_FMT "\n",
                        num_P, num_Q, glob_dofs, num_comp_q, num_comp_q, glob_dofs / num_comp_q));
  // -- Get Partition Statistics
  PetscCall(PetscPrintf(comm, "  Partition:                             (min,max,median,max/median)\n"));
  {
    PetscInt *gather_buffer = NULL;
    PetscInt  part_owned_dofs[3], part_local_dofs[3], part_boundary_dofs[3], part_neighbors[3];
    PetscInt  median_index = comm_size % 2 ? comm_size / 2 : comm_size / 2 - 1;
    if (!rank) PetscCall(PetscMalloc1(comm_size, &gather_buffer));

    PetscCallMPI(MPI_Gather(&owned_dofs, 1, MPIU_INT, gather_buffer, 1, MPIU_INT, 0, comm));
    if (!rank) {
      PetscCall(PetscSortInt(comm_size, gather_buffer));
      part_owned_dofs[0]             = gather_buffer[0];              // min
      part_owned_dofs[1]             = gather_buffer[comm_size - 1];  // max
      part_owned_dofs[2]             = gather_buffer[median_index];   // median
      PetscReal part_owned_dof_ratio = (PetscReal)part_owned_dofs[1] / (PetscReal)part_owned_dofs[2];
      PetscCall(PetscPrintf(
          comm, "    Global Vector %" PetscInt_FMT "-DoF nodes          : %" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ", %f\n", num_comp_q,
          part_owned_dofs[0] / num_comp_q, part_owned_dofs[1] / num_comp_q, part_owned_dofs[2] / num_comp_q, part_owned_dof_ratio));
    }

    PetscCallMPI(MPI_Gather(&local_dofs, 1, MPIU_INT, gather_buffer, 1, MPIU_INT, 0, comm));
    if (!rank) {
      PetscCall(PetscSortInt(comm_size, gather_buffer));
      part_local_dofs[0]             = gather_buffer[0];              // min
      part_local_dofs[1]             = gather_buffer[comm_size - 1];  // max
      part_local_dofs[2]             = gather_buffer[median_index];   // median
      PetscReal part_local_dof_ratio = (PetscReal)part_local_dofs[1] / (PetscReal)part_local_dofs[2];
      PetscCall(PetscPrintf(
          comm, "    Local Vector %" PetscInt_FMT "-DoF nodes           : %" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ", %f\n", num_comp_q,
          part_local_dofs[0] / num_comp_q, part_local_dofs[1] / num_comp_q, part_local_dofs[2] / num_comp_q, part_local_dof_ratio));
    }

    if (comm_size != 1) {
      PetscInt num_remote_roots_total = 0, num_remote_leaves_total = 0, num_ghost_interface_ranks = 0, num_owned_interface_ranks = 0;
      {
        PetscSF            sf;
        PetscInt           nrranks, niranks;
        const PetscInt    *roffset, *rmine, *rremote, *ioffset, *irootloc;
        const PetscMPIInt *rranks, *iranks;
        PetscCall(DMGetSectionSF(user->dm, &sf));
        PetscCall(PetscSFGetRootRanks(sf, &nrranks, &rranks, &roffset, &rmine, &rremote));
        PetscCall(PetscSFGetLeafRanks(sf, &niranks, &iranks, &ioffset, &irootloc));
        for (PetscInt i = 0; i < nrranks; i++) {
          if (rranks[i] == rank) continue;  // Ignore same-part global->local transfers
          num_remote_roots_total += roffset[i + 1] - roffset[i];
          num_ghost_interface_ranks++;
        }
        for (PetscInt i = 0; i < niranks; i++) {
          if (iranks[i] == rank) continue;
          num_remote_leaves_total += ioffset[i + 1] - ioffset[i];
          num_owned_interface_ranks++;
        }
      }
      PetscCallMPI(MPI_Gather(&num_remote_roots_total, 1, MPIU_INT, gather_buffer, 1, MPIU_INT, 0, comm));
      if (!rank) {
        PetscCall(PetscSortInt(comm_size, gather_buffer));
        part_boundary_dofs[0]           = gather_buffer[0];              // min
        part_boundary_dofs[1]           = gather_buffer[comm_size - 1];  // max
        part_boundary_dofs[2]           = gather_buffer[median_index];   // median
        PetscReal part_shared_dof_ratio = (PetscReal)part_boundary_dofs[1] / (PetscReal)part_boundary_dofs[2];
        PetscCall(PetscPrintf(
            comm, "    Ghost Interface %" PetscInt_FMT "-DoF nodes        : %" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ", %f\n",
            num_comp_q, part_boundary_dofs[0] / num_comp_q, part_boundary_dofs[1] / num_comp_q, part_boundary_dofs[2] / num_comp_q,
            part_shared_dof_ratio));
      }

      PetscCallMPI(MPI_Gather(&num_ghost_interface_ranks, 1, MPIU_INT, gather_buffer, 1, MPIU_INT, 0, comm));
      if (!rank) {
        PetscCall(PetscSortInt(comm_size, gather_buffer));
        part_neighbors[0]              = gather_buffer[0];              // min
        part_neighbors[1]              = gather_buffer[comm_size - 1];  // max
        part_neighbors[2]              = gather_buffer[median_index];   // median
        PetscReal part_neighbors_ratio = (PetscReal)part_neighbors[1] / (PetscReal)part_neighbors[2];
        PetscCall(PetscPrintf(comm, "    Ghost Interface Ranks              : %" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ", %f\n",
                              part_neighbors[0], part_neighbors[1], part_neighbors[2], part_neighbors_ratio));
      }

      PetscCallMPI(MPI_Gather(&num_remote_leaves_total, 1, MPIU_INT, gather_buffer, 1, MPIU_INT, 0, comm));
      if (!rank) {
        PetscCall(PetscSortInt(comm_size, gather_buffer));
        part_boundary_dofs[0]           = gather_buffer[0];              // min
        part_boundary_dofs[1]           = gather_buffer[comm_size - 1];  // max
        part_boundary_dofs[2]           = gather_buffer[median_index];   // median
        PetscReal part_shared_dof_ratio = (PetscReal)part_boundary_dofs[1] / (PetscReal)part_boundary_dofs[2];
        PetscCall(PetscPrintf(
            comm, "    Owned Interface %" PetscInt_FMT "-DoF nodes        : %" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ", %f\n",
            num_comp_q, part_boundary_dofs[0] / num_comp_q, part_boundary_dofs[1] / num_comp_q, part_boundary_dofs[2] / num_comp_q,
            part_shared_dof_ratio));
      }

      PetscCallMPI(MPI_Gather(&num_owned_interface_ranks, 1, MPIU_INT, gather_buffer, 1, MPIU_INT, 0, comm));
      if (!rank) {
        PetscCall(PetscSortInt(comm_size, gather_buffer));
        part_neighbors[0]              = gather_buffer[0];              // min
        part_neighbors[1]              = gather_buffer[comm_size - 1];  // max
        part_neighbors[2]              = gather_buffer[median_index];   // median
        PetscReal part_neighbors_ratio = (PetscReal)part_neighbors[1] / (PetscReal)part_neighbors[2];
        PetscCall(PetscPrintf(comm, "    Owned Interface Ranks              : %" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ", %f\n",
                              part_neighbors[0], part_neighbors[1], part_neighbors[2], part_neighbors_ratio));
      }
    }

    if (!rank) PetscCall(PetscFree(gather_buffer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
