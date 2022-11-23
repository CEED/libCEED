// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

//                        libCEED + PETSc Example: CEED BPs 3-6 with Multigrid
//
// This example demonstrates a simple usage of libCEED with PETSc to solve the
// CEED BP benchmark problems, see http://ceed.exascaleproject.org/bps.
//
// The code uses higher level communication protocols in DMPlex.
//
// Build with:
//
//     make multigrid [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     multigrid -problem bp3
//     multigrid -problem bp4
//     multigrid -problem bp5 -ceed /cpu/self
//     multigrid -problem bp6 -ceed /gpu/cuda
//
//TESTARGS -ceed {ceed_resource} -test -problem bp3 -degree 3
//TESTARGS -ceed {ceed_resource} -test -problem bp3 -degree 3 -simplex

/// @file
/// CEED BPs 1-6 multigrid example using PETSc
const char help[] = "Solve CEED BPs using p-multigrid with PETSc and DMPlex\n";

#include <ceed.h>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscksp.h>
#include <petscsys.h>
#include <stdbool.h>
#include <string.h>

#include "bps.h"
#include "include/bpsproblemdata.h"
#include "include/libceedsetup.h"
#include "include/matops.h"
#include "include/petscutils.h"
#include "include/petscversion.h"
#include "include/structs.h"

#if PETSC_VERSION_LT(3, 12, 0)
#ifdef PETSC_HAVE_CUDA
#include <petsccuda.h>
// Note: With PETSc prior to version 3.12.0, providing the source path to
//       include 'cublas_v2.h' will be needed to use 'petsccuda.h'.
#endif
#endif

int main(int argc, char **argv) {
  MPI_Comm comm;
  char     filename[PETSC_MAX_PATH_LEN], ceed_resource[PETSC_MAX_PATH_LEN] = "/cpu/self";
  double   my_rt_start, my_rt, rt_min, rt_max;
  PetscInt degree = 3, q_extra, *l_size, *xl_size, *g_size, dim = 3, fine_level, mesh_elem[3] = {3, 3, 3}, num_comp_u = 1, num_levels = degree,
           *level_degrees;
  PetscScalar          *r;
  PetscScalar           eps = 1.0;
  PetscBool             test_mode, benchmark_mode, read_mesh, write_solution, simplex;
  PetscLogStage         solve_stage;
  PetscLogEvent         assemble_event;
  DM                   *dm, dm_orig;
  KSP                   ksp;
  PC                    pc;
  Mat                  *mat_O, *mat_pr, mat_coarse;
  Vec                  *X, *X_loc, *mult, rhs, rhs_loc;
  PetscMemType          mem_type;
  OperatorApplyContext *op_apply_ctx, op_error_ctx;
  ProlongRestrContext  *pr_restr_ctx;
  Ceed                  ceed;
  CeedData             *ceed_data;
  CeedVector            rhs_ceed, target;
  CeedQFunction         qf_error;
  CeedOperator          op_error;
  BPType                bp_choice;
  CoarsenType           coarsen;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  // Parse command line options
  PetscOptionsBegin(comm, NULL, "CEED BPs in PETSc", NULL);
  bp_choice = CEED_BP3;
  PetscCall(PetscOptionsEnum("-problem", "CEED benchmark problem to solve", NULL, bp_types, (PetscEnum)bp_choice, (PetscEnum *)&bp_choice, NULL));
  num_comp_u = bp_options[bp_choice].num_comp_u;
  test_mode  = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-test", "Testing mode (do not print unless error is large)", NULL, test_mode, &test_mode, NULL));
  benchmark_mode = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-benchmark", "Benchmarking mode (prints benchmark statistics)", NULL, benchmark_mode, &benchmark_mode, NULL));
  write_solution = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-write_solution", "Write solution for visualization", NULL, write_solution, &write_solution, NULL));
  simplex = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-simplex", "Element topology (default:hex)", NULL, simplex, &simplex, NULL));
  if ((bp_choice == CEED_BP5 || bp_choice == CEED_BP6) && (simplex)) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "BP5/6 is not supported with simplex");
  }
  PetscCall(PetscOptionsScalar("-eps", "Epsilon parameter for Kershaw mesh transformation", NULL, eps, &eps, NULL));
  if (eps > 1 || eps <= 0) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "-eps %g must be (0,1]", (double)PetscRealPart(eps));
  degree = test_mode ? 3 : 2;
  PetscCall(PetscOptionsInt("-degree", "Polynomial degree of tensor product basis", NULL, degree, &degree, NULL));
  if (degree < 1) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "-degree %" PetscInt_FMT " must be at least 1", degree);
  q_extra = bp_options[bp_choice].q_extra;
  PetscCall(PetscOptionsInt("-q_extra", "Number of extra quadrature points", NULL, q_extra, &q_extra, NULL));
  PetscCall(PetscOptionsString("-ceed", "CEED resource specifier", NULL, ceed_resource, ceed_resource, sizeof(ceed_resource), NULL));
  coarsen = COARSEN_UNIFORM;
  PetscCall(PetscOptionsEnum("-coarsen", "Coarsening strategy to use", NULL, coarsen_types, (PetscEnum)coarsen, (PetscEnum *)&coarsen, NULL));
  read_mesh = PETSC_FALSE;
  PetscCall(PetscOptionsString("-mesh", "Read mesh from file", NULL, filename, filename, sizeof(filename), &read_mesh));
  if (!read_mesh) {
    PetscInt tmp = dim;
    PetscCall(PetscOptionsIntArray("-cells", "Number of cells per dimension", NULL, mesh_elem, &tmp, NULL));
  }
  PetscOptionsEnd();

  // Set up libCEED
  CeedInit(ceed_resource, &ceed);
  CeedMemType mem_type_backend;
  CeedGetPreferredMemType(ceed, &mem_type_backend);

  // Setup DM
  if (read_mesh) {
    PetscCall(DMPlexCreateFromFile(PETSC_COMM_WORLD, filename, NULL, PETSC_TRUE, &dm_orig));
  } else {
    PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, simplex, mesh_elem, NULL, NULL, NULL, PETSC_TRUE, &dm_orig));
  }

  VecType vec_type;
  switch (mem_type_backend) {
    case CEED_MEM_HOST:
      vec_type = VECSTANDARD;
      break;
    case CEED_MEM_DEVICE: {
      const char *resolved;
      CeedGetResource(ceed, &resolved);
      if (strstr(resolved, "/gpu/cuda")) vec_type = VECCUDA;
      else if (strstr(resolved, "/gpu/hip/occa")) vec_type = VECSTANDARD;  // https://github.com/CEED/libCEED/issues/678
      else if (strstr(resolved, "/gpu/hip")) vec_type = VECHIP;
      else vec_type = VECSTANDARD;
    }
  }
  PetscCall(DMSetVecType(dm_orig, vec_type));
  PetscCall(DMSetFromOptions(dm_orig));
  PetscCall(DMViewFromOptions(dm_orig, NULL, "-dm_view"));

  // Apply Kershaw mesh transformation
  PetscCall(Kershaw(dm_orig, eps));

  // Allocate arrays for PETSc objects for each level
  switch (coarsen) {
    case COARSEN_UNIFORM:
      num_levels = degree;
      break;
    case COARSEN_LOGARITHMIC:
      num_levels = ceil(log(degree) / log(2)) + 1;
      break;
  }
  PetscCall(PetscMalloc1(num_levels, &level_degrees));
  fine_level = num_levels - 1;

  switch (coarsen) {
    case COARSEN_UNIFORM:
      for (int i = 0; i < num_levels; i++) level_degrees[i] = i + 1;
      break;
    case COARSEN_LOGARITHMIC:
      for (int i = 0; i < num_levels - 1; i++) level_degrees[i] = pow(2, i);
      level_degrees[fine_level] = degree;
      break;
  }
  PetscCall(PetscMalloc1(num_levels, &dm));
  PetscCall(PetscMalloc1(num_levels, &X));
  PetscCall(PetscMalloc1(num_levels, &X_loc));
  PetscCall(PetscMalloc1(num_levels, &mult));
  PetscCall(PetscMalloc1(num_levels, &op_apply_ctx));
  PetscCall(PetscMalloc1(num_levels, &pr_restr_ctx));
  PetscCall(PetscMalloc1(num_levels, &mat_O));
  PetscCall(PetscMalloc1(num_levels, &mat_pr));
  PetscCall(PetscMalloc1(num_levels, &l_size));
  PetscCall(PetscMalloc1(num_levels, &xl_size));
  PetscCall(PetscMalloc1(num_levels, &g_size));

  PetscInt c_start, c_end;
  PetscCall(DMPlexGetHeightStratum(dm_orig, 0, &c_start, &c_end));
  DMPolytopeType cell_type;
  PetscCall(DMPlexGetCellType(dm_orig, c_start, &cell_type));
  CeedElemTopology elem_topo = ElemTopologyP2C(cell_type);

  // Setup DM and Operator Mat Shells for each level
  for (CeedInt i = 0; i < num_levels; i++) {
    // Create DM
    PetscCall(DMClone(dm_orig, &dm[i]));
    PetscCall(DMGetVecType(dm_orig, &vec_type));
    PetscCall(DMSetVecType(dm[i], vec_type));
    PetscInt dim;
    PetscCall(DMGetDimension(dm[i], &dim));
    PetscCall(SetupDMByDegree(dm[i], level_degrees[fine_level], q_extra, num_comp_u, dim, bp_options[bp_choice].enforce_bc));

    // Create vectors
    PetscCall(DMCreateGlobalVector(dm[i], &X[i]));
    PetscCall(VecGetLocalSize(X[i], &l_size[i]));
    PetscCall(VecGetSize(X[i], &g_size[i]));
    PetscCall(DMCreateLocalVector(dm[i], &X_loc[i]));
    PetscCall(VecGetSize(X_loc[i], &xl_size[i]));

    // Operator
    PetscCall(PetscMalloc1(1, &op_apply_ctx[i]));
    PetscCall(PetscMalloc1(1, &op_error_ctx));
    PetscCall(MatCreateShell(comm, l_size[i], l_size[i], g_size[i], g_size[i], op_apply_ctx[i], &mat_O[i]));
    PetscCall(MatShellSetOperation(mat_O[i], MATOP_MULT, (void (*)(void))MatMult_Ceed));
    PetscCall(MatShellSetOperation(mat_O[i], MATOP_GET_DIAGONAL, (void (*)(void))MatGetDiag));
    PetscCall(MatShellSetVecType(mat_O[i], vec_type));

    // Level transfers
    if (i > 0) {
      // Interp
      PetscCall(PetscMalloc1(1, &pr_restr_ctx[i]));
      PetscCall(MatCreateShell(comm, l_size[i], l_size[i - 1], g_size[i], g_size[i - 1], pr_restr_ctx[i], &mat_pr[i]));
      PetscCall(MatShellSetOperation(mat_pr[i], MATOP_MULT, (void (*)(void))MatMult_Prolong));
      PetscCall(MatShellSetOperation(mat_pr[i], MATOP_MULT_TRANSPOSE, (void (*)(void))MatMult_Restrict));
      PetscCall(MatShellSetVecType(mat_pr[i], vec_type));
    }
  }
  PetscCall(VecDuplicate(X[fine_level], &rhs));

  // Print global grid information
  if (!test_mode) {
    PetscInt P = degree + 1, Q = P + q_extra;

    const char *used_resource;
    CeedGetResource(ceed, &used_resource);

    PetscCall(VecGetType(X[0], &vec_type));

    PetscCall(PetscPrintf(comm,
                          "\n-- CEED Benchmark Problem %" CeedInt_FMT " -- libCEED + PETSc + PCMG --\n"
                          "  PETSc:\n"
                          "    PETSc Vec Type                          : %s\n"
                          "  libCEED:\n"
                          "    libCEED Backend                         : %s\n"
                          "    libCEED Backend MemType                 : %s\n"
                          "  Mesh:\n"
                          "    Solution Order (P)                      : %" CeedInt_FMT "\n"
                          "    Quadrature  Order (Q)                   : %" CeedInt_FMT "\n"
                          "    Additional quadrature points (q_extra)  : %" CeedInt_FMT "\n"
                          "    Global Nodes                            : %" PetscInt_FMT "\n"
                          "    Owned Nodes                             : %" PetscInt_FMT "\n"
                          "    DoF per node                            : %" PetscInt_FMT "\n"
                          "    Element topology                        : %s\n"
                          "  Multigrid:\n"
                          "    Number of Levels                        : %" CeedInt_FMT "\n",
                          bp_choice + 1, vec_type, used_resource, CeedMemTypes[mem_type_backend], P, Q, q_extra, g_size[fine_level] / num_comp_u,
                          l_size[fine_level] / num_comp_u, num_comp_u, CeedElemTopologies[elem_topo], num_levels));
  }

  // Create RHS vector
  PetscCall(VecDuplicate(X_loc[fine_level], &rhs_loc));
  PetscCall(VecZeroEntries(rhs_loc));
  PetscCall(VecGetArrayAndMemType(rhs_loc, &r, &mem_type));
  CeedVectorCreate(ceed, xl_size[fine_level], &rhs_ceed);
  CeedVectorSetArray(rhs_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, r);

  // Set up libCEED operators on each level
  PetscCall(PetscMalloc1(num_levels, &ceed_data));
  for (PetscInt i = 0; i < num_levels; i++) {
    // Print level information
    if (!test_mode && (i == 0 || i == fine_level)) {
      PetscCall(PetscPrintf(comm,
                            "    Level %" PetscInt_FMT " (%s):\n"
                            "      Solution Order (P)                    : %" CeedInt_FMT "\n"
                            "      Global Nodes                          : %" PetscInt_FMT "\n"
                            "      Owned Nodes                           : %" PetscInt_FMT "\n",
                            i, (i ? "fine" : "coarse"), level_degrees[i] + 1, g_size[i] / num_comp_u, l_size[i] / num_comp_u));
    }
    PetscCall(PetscMalloc1(1, &ceed_data[i]));
    PetscCall(SetupLibceedByDegree(dm[i], ceed, level_degrees[i], dim, q_extra, dim, num_comp_u, g_size[i], xl_size[i], bp_options[bp_choice],
                                   ceed_data[i], i == (fine_level), rhs_ceed, &target));
  }

  // Gather RHS
  CeedVectorTakeArray(rhs_ceed, MemTypeP2C(mem_type), NULL);
  PetscCall(VecRestoreArrayAndMemType(rhs_loc, &r));
  PetscCall(VecZeroEntries(rhs));
  PetscCall(DMLocalToGlobal(dm[fine_level], rhs_loc, ADD_VALUES, rhs));
  CeedVectorDestroy(&rhs_ceed);

  // Create the error QFunction
  CeedQFunctionCreateInterior(ceed, 1, bp_options[bp_choice].error, bp_options[bp_choice].error_loc, &qf_error);
  CeedQFunctionAddInput(qf_error, "u", num_comp_u, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_error, "true_soln", num_comp_u, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_error, "qdata", ceed_data[fine_level]->q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_error, "error", num_comp_u, CEED_EVAL_INTERP);

  // Create the error operator
  CeedOperatorCreate(ceed, qf_error, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_error);
  CeedOperatorSetField(op_error, "u", ceed_data[fine_level]->elem_restr_u, ceed_data[fine_level]->basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_error, "true_soln", ceed_data[fine_level]->elem_restr_u_i, CEED_BASIS_COLLOCATED, target);
  CeedOperatorSetField(op_error, "qdata", ceed_data[fine_level]->elem_restr_qd_i, CEED_BASIS_COLLOCATED, ceed_data[fine_level]->q_data);
  CeedOperatorSetField(op_error, "error", ceed_data[fine_level]->elem_restr_u, ceed_data[fine_level]->basis_u, CEED_VECTOR_ACTIVE);

  // Calculate multiplicity
  for (int i = 0; i < num_levels; i++) {
    PetscScalar *x;

    // CEED vector
    PetscCall(VecZeroEntries(X_loc[i]));
    PetscCall(VecGetArray(X_loc[i], &x));
    CeedVectorSetArray(ceed_data[i]->x_ceed, CEED_MEM_HOST, CEED_USE_POINTER, x);

    // Multiplicity
    CeedElemRestrictionGetMultiplicity(ceed_data[i]->elem_restr_u, ceed_data[i]->x_ceed);
    CeedVectorSyncArray(ceed_data[i]->x_ceed, CEED_MEM_HOST);

    // Restore vector
    PetscCall(VecRestoreArray(X_loc[i], &x));

    // Creat mult vector
    PetscCall(VecDuplicate(X_loc[i], &mult[i]));

    // Local-to-global
    PetscCall(VecZeroEntries(X[i]));
    PetscCall(DMLocalToGlobal(dm[i], X_loc[i], ADD_VALUES, X[i]));
    PetscCall(VecZeroEntries(X_loc[i]));

    // Global-to-local
    PetscCall(DMGlobalToLocal(dm[i], X[i], INSERT_VALUES, mult[i]));
    PetscCall(VecZeroEntries(X[i]));

    // Multiplicity scaling
    PetscCall(VecReciprocal(mult[i]));
  }

  // Set up Mat
  for (int i = 0; i < num_levels; i++) {
    // Set up apply operator context
    PetscCall(SetupApplyOperatorCtx(comm, dm[i], ceed, ceed_data[i], X_loc[i], op_apply_ctx[i]));

    if (i > 0) {
      // Prolongation/Restriction Operator
      PetscCall(CeedLevelTransferSetup(dm[i - 1], ceed, i, num_comp_u, ceed_data, bp_options[bp_choice], mult[i]));
      pr_restr_ctx[i]->comm        = comm;
      pr_restr_ctx[i]->dmf         = dm[i];
      pr_restr_ctx[i]->dmc         = dm[i - 1];
      pr_restr_ctx[i]->loc_vec_c   = X_loc[i - 1];
      pr_restr_ctx[i]->loc_vec_f   = op_apply_ctx[i]->Y_loc;
      pr_restr_ctx[i]->mult_vec    = mult[i];
      pr_restr_ctx[i]->ceed_vec_c  = op_apply_ctx[i - 1]->x_ceed;
      pr_restr_ctx[i]->ceed_vec_f  = op_apply_ctx[i]->y_ceed;
      pr_restr_ctx[i]->op_prolong  = ceed_data[i]->op_prolong;
      pr_restr_ctx[i]->op_restrict = ceed_data[i]->op_restrict;
      pr_restr_ctx[i]->ceed        = ceed;
    }
  }

  // Assemble coarse grid Jacobian for AMG (or other sparse matrix) solve
  PetscCall(DMCreateMatrix(dm[0], &mat_coarse));

  PetscCall(PetscLogEventRegister("AssembleMatrix", MAT_CLASSID, &assemble_event));
  {
    // Assemble matrix analytically
    PetscCount num_entries;
    CeedInt   *rows, *cols;
    CeedVector coo_values;
    CeedOperatorLinearAssembleSymbolic(op_apply_ctx[0]->op, &num_entries, &rows, &cols);
    ISLocalToGlobalMapping ltog_row, ltog_col;
    PetscCall(MatGetLocalToGlobalMapping(mat_coarse, &ltog_row, &ltog_col));
    PetscCall(ISLocalToGlobalMappingApply(ltog_row, num_entries, rows, rows));
    PetscCall(ISLocalToGlobalMappingApply(ltog_col, num_entries, cols, cols));
    PetscCall(MatSetPreallocationCOO(mat_coarse, num_entries, rows, cols));
    free(rows);
    free(cols);
    CeedVectorCreate(ceed, num_entries, &coo_values);
    PetscCall(PetscLogEventBegin(assemble_event, mat_coarse, 0, 0, 0));
    CeedOperatorLinearAssemble(op_apply_ctx[0]->op, coo_values);
    const CeedScalar *values;
    CeedVectorGetArrayRead(coo_values, CEED_MEM_HOST, &values);
    PetscCall(MatSetValuesCOO(mat_coarse, values, ADD_VALUES));
    CeedVectorRestoreArrayRead(coo_values, &values);
    PetscCall(PetscLogEventEnd(assemble_event, mat_coarse, 0, 0, 0));
    CeedVectorDestroy(&coo_values);
  }

  // Set up KSP
  PetscCall(KSPCreate(comm, &ksp));
  {
    PetscCall(KSPSetType(ksp, KSPCG));
    PetscCall(KSPSetNormType(ksp, KSP_NORM_NATURAL));
    PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  }
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetOperators(ksp, mat_O[fine_level], mat_O[fine_level]));

  // Set up PCMG
  PetscCall(KSPGetPC(ksp, &pc));
  PCMGCycleType pcmg_cycle_type = PC_MG_CYCLE_V;
  {
    PetscCall(PCSetType(pc, PCMG));

    // PCMG levels
    PetscCall(PCMGSetLevels(pc, num_levels, NULL));
    for (int i = 0; i < num_levels; i++) {
      // Smoother
      KSP smoother;
      PC  smoother_pc;
      PetscCall(PCMGGetSmoother(pc, i, &smoother));
      PetscCall(KSPSetType(smoother, KSPCHEBYSHEV));
      PetscCall(KSPChebyshevEstEigSet(smoother, 0, 0.1, 0, 1.1));
      PetscCall(KSPChebyshevEstEigSetUseNoisy(smoother, PETSC_TRUE));
      PetscCall(KSPSetOperators(smoother, mat_O[i], mat_O[i]));
      PetscCall(KSPGetPC(smoother, &smoother_pc));
      PetscCall(PCSetType(smoother_pc, PCJACOBI));
      PetscCall(PCJacobiSetType(smoother_pc, PC_JACOBI_DIAGONAL));

      // Work vector
      if (i < num_levels - 1) {
        PetscCall(PCMGSetX(pc, i, X[i]));
      }

      // Level transfers
      if (i > 0) {
        // Interpolation
        PetscCall(PCMGSetInterpolation(pc, i, mat_pr[i]));
      }

      // Coarse solve
      KSP coarse;
      PC  coarse_pc;
      PetscCall(PCMGGetCoarseSolve(pc, &coarse));
      PetscCall(KSPSetType(coarse, KSPPREONLY));
      PetscCall(KSPSetOperators(coarse, mat_coarse, mat_coarse));

      PetscCall(KSPGetPC(coarse, &coarse_pc));
      PetscCall(PCSetType(coarse_pc, PCGAMG));

      PetscCall(KSPSetOptionsPrefix(coarse, "coarse_"));
      PetscCall(PCSetOptionsPrefix(coarse_pc, "coarse_"));
      PetscCall(KSPSetFromOptions(coarse));
      PetscCall(PCSetFromOptions(coarse_pc));
    }

    // PCMG options
    PetscCall(PCMGSetType(pc, PC_MG_MULTIPLICATIVE));
    PetscCall(PCMGSetNumberSmooth(pc, 3));
    PetscCall(PCMGSetCycleType(pc, pcmg_cycle_type));
  }

  // First run, if benchmarking
  if (benchmark_mode) {
    PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 1));
    PetscCall(VecZeroEntries(X[fine_level]));
    my_rt_start = MPI_Wtime();
    PetscCall(KSPSolve(ksp, rhs, X[fine_level]));
    my_rt = MPI_Wtime() - my_rt_start;
    PetscCall(MPI_Allreduce(MPI_IN_PLACE, &my_rt, 1, MPI_DOUBLE, MPI_MIN, comm));
    // Set maxits based on first iteration timing
    if (my_rt > 0.02) {
      PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 5));
    } else {
      PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 20));
    }
  }

  // Timed solve
  PetscCall(VecZeroEntries(X[fine_level]));
  PetscCall(PetscBarrier((PetscObject)ksp));

  // -- Performance logging
  PetscCall(PetscLogStageRegister("Solve Stage", &solve_stage));
  PetscCall(PetscLogStagePush(solve_stage));

  // -- Solve
  my_rt_start = MPI_Wtime();
  PetscCall(KSPSolve(ksp, rhs, X[fine_level]));
  my_rt = MPI_Wtime() - my_rt_start;

  // -- Performance logging
  PetscCall(PetscLogStagePop());

  // Output results
  {
    KSPType            ksp_type;
    PCMGType           pcmg_type;
    KSPConvergedReason reason;
    PetscReal          rnorm;
    PetscInt           its;
    PetscCall(KSPGetType(ksp, &ksp_type));
    PetscCall(KSPGetConvergedReason(ksp, &reason));
    PetscCall(KSPGetIterationNumber(ksp, &its));
    PetscCall(KSPGetResidualNorm(ksp, &rnorm));
    PetscCall(PCMGGetType(pc, &pcmg_type));
    if (!test_mode || reason < 0 || rnorm > 1e-8) {
      PetscCall(PetscPrintf(comm,
                            "  KSP:\n"
                            "    KSP Type                                : %s\n"
                            "    KSP Convergence                         : %s\n"
                            "    Total KSP Iterations                    : %" PetscInt_FMT "\n"
                            "    Final rnorm                             : %e\n",
                            ksp_type, KSPConvergedReasons[reason], its, (double)rnorm));
      PetscCall(PetscPrintf(comm,
                            "  PCMG:\n"
                            "    PCMG Type                               : %s\n"
                            "    PCMG Cycle Type                         : %s\n",
                            PCMGTypes[pcmg_type], PCMGCycleTypes[pcmg_cycle_type]));
    }
    if (!test_mode) {
      PetscCall(PetscPrintf(comm, "  Performance:\n"));
    }
    {
      // Set up error operator context
      PetscCall(SetupErrorOperatorCtx(comm, dm[fine_level], ceed, ceed_data[fine_level], X_loc[fine_level], op_error, op_error_ctx));
      PetscScalar l2_error;
      PetscCall(ComputeL2Error(X[fine_level], &l2_error, op_error_ctx));
      PetscReal tol = 5e-2;
      if (!test_mode || l2_error > tol) {
        PetscCall(MPI_Allreduce(&my_rt, &rt_min, 1, MPI_DOUBLE, MPI_MIN, comm));
        PetscCall(MPI_Allreduce(&my_rt, &rt_max, 1, MPI_DOUBLE, MPI_MAX, comm));
        PetscCall(PetscPrintf(comm,
                              "    L2 Error                                : %e\n"
                              "    CG Solve Time                           : %g (%g) sec\n",
                              (double)l2_error, rt_max, rt_min));
      }
    }
    if (benchmark_mode && (!test_mode)) {
      PetscCall(PetscPrintf(comm, "    DoFs/Sec in CG                            : %g (%g) million\n", 1e-6 * g_size[fine_level] * its / rt_max,
                            1e-6 * g_size[fine_level] * its / rt_min));
    }
  }

  if (write_solution) {
    PetscViewer vtk_viewer_soln;

    PetscCall(PetscViewerCreate(comm, &vtk_viewer_soln));
    PetscCall(PetscViewerSetType(vtk_viewer_soln, PETSCVIEWERVTK));
    PetscCall(PetscViewerFileSetName(vtk_viewer_soln, "solution.vtu"));
    PetscCall(VecView(X[fine_level], vtk_viewer_soln));
    PetscCall(PetscViewerDestroy(&vtk_viewer_soln));
  }

  // Cleanup
  for (int i = 0; i < num_levels; i++) {
    PetscCall(VecDestroy(&X[i]));
    PetscCall(VecDestroy(&X_loc[i]));
    PetscCall(VecDestroy(&mult[i]));
    PetscCall(VecDestroy(&op_apply_ctx[i]->Y_loc));
    PetscCall(MatDestroy(&mat_O[i]));
    PetscCall(PetscFree(op_apply_ctx[i]));
    if (i > 0) {
      PetscCall(MatDestroy(&mat_pr[i]));
      PetscCall(PetscFree(pr_restr_ctx[i]));
    }
    PetscCall(CeedDataDestroy(i, ceed_data[i]));
    PetscCall(DMDestroy(&dm[i]));
  }
  PetscCall(PetscFree(level_degrees));
  PetscCall(PetscFree(dm));
  PetscCall(PetscFree(X));
  PetscCall(PetscFree(X_loc));
  PetscCall(VecDestroy(&op_error_ctx->Y_loc));
  PetscCall(PetscFree(mult));
  PetscCall(PetscFree(mat_O));
  PetscCall(PetscFree(mat_pr));
  PetscCall(PetscFree(ceed_data));
  PetscCall(PetscFree(op_apply_ctx));
  PetscCall(PetscFree(op_error_ctx));
  PetscCall(PetscFree(pr_restr_ctx));
  PetscCall(PetscFree(l_size));
  PetscCall(PetscFree(xl_size));
  PetscCall(PetscFree(g_size));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&rhs_loc));
  PetscCall(MatDestroy(&mat_coarse));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(DMDestroy(&dm_orig));
  CeedVectorDestroy(&target);
  CeedQFunctionDestroy(&qf_error);
  CeedOperatorDestroy(&op_error);
  CeedDestroy(&ceed);
  return PetscFinalize();
}
