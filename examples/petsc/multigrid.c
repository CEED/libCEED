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

/// @file
/// CEED BPs 1-6 multigrid example using PETSc
const char help[] = "Solve CEED BPs using p-multigrid with PETSc and DMPlex\n";

#include <stdbool.h>
#include <string.h>
#include <ceed.h>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscksp.h>
#include <petscsys.h>

#include "bps.h"
#include "include/bpsproblemdata.h"
#include "include/petscutils.h"
#include "include/petscversion.h"
#include "include/matops.h"
#include "include/structs.h"
#include "include/libceedsetup.h"

#if PETSC_VERSION_LT(3,12,0)
#ifdef PETSC_HAVE_CUDA
#include <petsccuda.h>
// Note: With PETSc prior to version 3.12.0, providing the source path to
//       include 'cublas_v2.h' will be needed to use 'petsccuda.h'.
#endif
#endif

int main(int argc, char **argv) {
  PetscInt ierr;
  MPI_Comm comm;
  char filename[PETSC_MAX_PATH_LEN],
       ceed_resource[PETSC_MAX_PATH_LEN] = "/cpu/self";
  double my_rt_start, my_rt, rt_min, rt_max;
  PetscInt degree = 3, q_extra, *l_size, *xl_size, *g_size, dim = 3, fine_level,
           mesh_elem[3] = {3, 3, 3}, num_comp_u = 1, num_levels = degree, *level_degrees;
  PetscScalar *r;
  PetscScalar eps = 1.0;
  PetscBool test_mode, benchmark_mode, read_mesh, write_solution;
  PetscLogStage solve_stage;
  PetscLogEvent assemble_event;
  DM  *dm, dm_orig;
  KSP ksp;
  PC pc;
  Mat *mat_O, *mat_pr, mat_coarse;
  Vec *X, *X_loc, *mult, rhs, rhs_loc;
  PetscMemType mem_type;
  UserO *user_O;
  UserProlongRestr *user_pr;
  Ceed ceed;
  CeedData *ceed_data;
  CeedVector rhs_ceed, target;
  CeedQFunction qf_error, qf_restrict, qf_prolong;
  CeedOperator op_error;
  BPType bp_choice;
  CoarsenType coarsen;

  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;

  // Parse command line options
  PetscOptionsBegin(comm, NULL, "CEED BPs in PETSc", NULL);
  bp_choice = CEED_BP3;
  ierr = PetscOptionsEnum("-problem",
                          "CEED benchmark problem to solve", NULL,
                          bp_types, (PetscEnum)bp_choice, (PetscEnum *)&bp_choice,
                          NULL); CHKERRQ(ierr);
  num_comp_u = bp_options[bp_choice].num_comp_u;
  test_mode = PETSC_FALSE;
  ierr = PetscOptionsBool("-test",
                          "Testing mode (do not print unless error is large)",
                          NULL, test_mode, &test_mode, NULL); CHKERRQ(ierr);
  benchmark_mode = PETSC_FALSE;
  ierr = PetscOptionsBool("-benchmark",
                          "Benchmarking mode (prints benchmark statistics)",
                          NULL, benchmark_mode, &benchmark_mode, NULL);
  CHKERRQ(ierr);
  write_solution = PETSC_FALSE;
  ierr = PetscOptionsBool("-write_solution",
                          "Write solution for visualization",
                          NULL, write_solution, &write_solution, NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-eps",
                            "Epsilon parameter for Kershaw mesh transformation",
                            NULL, eps, &eps, NULL);
  if (eps > 1 || eps <= 0) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
                                     "-eps %g must be (0,1]", (double)PetscRealPart(eps));
  degree = test_mode ? 3 : 2;
  ierr = PetscOptionsInt("-degree", "Polynomial degree of tensor product basis",
                         NULL, degree, &degree, NULL); CHKERRQ(ierr);
  if (degree < 1) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
                            "-degree %" PetscInt_FMT " must be at least 1", degree);
  q_extra = bp_options[bp_choice].q_extra;
  ierr = PetscOptionsInt("-q_extra", "Number of extra quadrature points",
                         NULL, q_extra, &q_extra, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, ceed_resource, ceed_resource,
                            sizeof(ceed_resource), NULL); CHKERRQ(ierr);
  coarsen = COARSEN_UNIFORM;
  ierr = PetscOptionsEnum("-coarsen",
                          "Coarsening strategy to use", NULL,
                          coarsen_types, (PetscEnum)coarsen,
                          (PetscEnum *)&coarsen, NULL); CHKERRQ(ierr);
  read_mesh = PETSC_FALSE;
  ierr = PetscOptionsString("-mesh", "Read mesh from file", NULL,
                            filename, filename, sizeof(filename), &read_mesh);
  CHKERRQ(ierr);
  if (!read_mesh) {
    PetscInt tmp = dim;
    ierr = PetscOptionsIntArray("-cells","Number of cells per dimension", NULL,
                                mesh_elem, &tmp, NULL); CHKERRQ(ierr);
  }
  PetscOptionsEnd();

  // Set up libCEED
  CeedInit(ceed_resource, &ceed);
  CeedMemType mem_type_backend;
  CeedGetPreferredMemType(ceed, &mem_type_backend);

  // Setup DM
  if (read_mesh) {
    ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, filename, NULL, PETSC_TRUE,
                                &dm_orig);
    CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, mesh_elem, NULL,
                               NULL, NULL, PETSC_TRUE, &dm_orig); CHKERRQ(ierr);
  }

  VecType vec_type;
  switch (mem_type_backend) {
  case CEED_MEM_HOST: vec_type = VECSTANDARD; break;
  case CEED_MEM_DEVICE: {
    const char *resolved;
    CeedGetResource(ceed, &resolved);
    if (strstr(resolved, "/gpu/cuda")) vec_type = VECCUDA;
    else if (strstr(resolved, "/gpu/hip")) vec_type = VECHIP;
    else vec_type = VECSTANDARD;
  }
  }
  ierr = DMSetVecType(dm_orig, vec_type); CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm_orig); CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm_orig, NULL, "-dm_view"); CHKERRQ(ierr);

  // Apply Kershaw mesh transformation
  ierr = Kershaw(dm_orig, eps); CHKERRQ(ierr);

  // Allocate arrays for PETSc objects for each level
  switch (coarsen) {
  case COARSEN_UNIFORM:
    num_levels = degree;
    break;
  case COARSEN_LOGARITHMIC:
    num_levels = ceil(log(degree)/log(2)) + 1;
    break;
  }
  ierr = PetscMalloc1(num_levels, &level_degrees); CHKERRQ(ierr);
  fine_level = num_levels - 1;

  switch (coarsen) {
  case COARSEN_UNIFORM:
    for (int i=0; i<num_levels; i++) level_degrees[i] = i + 1;
    break;
  case COARSEN_LOGARITHMIC:
    for (int i=0; i<num_levels - 1; i++) level_degrees[i] = pow(2,i);
    level_degrees[fine_level] = degree;
    break;
  }
  ierr = PetscMalloc1(num_levels, &dm); CHKERRQ(ierr);
  ierr = PetscMalloc1(num_levels, &X); CHKERRQ(ierr);
  ierr = PetscMalloc1(num_levels, &X_loc); CHKERRQ(ierr);
  ierr = PetscMalloc1(num_levels, &mult); CHKERRQ(ierr);
  ierr = PetscMalloc1(num_levels, &user_O); CHKERRQ(ierr);
  ierr = PetscMalloc1(num_levels, &user_pr); CHKERRQ(ierr);
  ierr = PetscMalloc1(num_levels, &mat_O); CHKERRQ(ierr);
  ierr = PetscMalloc1(num_levels, &mat_pr); CHKERRQ(ierr);
  ierr = PetscMalloc1(num_levels, &l_size); CHKERRQ(ierr);
  ierr = PetscMalloc1(num_levels, &xl_size); CHKERRQ(ierr);
  ierr = PetscMalloc1(num_levels, &g_size); CHKERRQ(ierr);

  // Setup DM and Operator Mat Shells for each level
  for (CeedInt i=0; i<num_levels; i++) {
    // Create DM
    ierr = DMClone(dm_orig, &dm[i]); CHKERRQ(ierr);
    ierr = DMGetVecType(dm_orig, &vec_type); CHKERRQ(ierr);
    ierr = DMSetVecType(dm[i], vec_type); CHKERRQ(ierr);
    PetscInt dim;
    ierr = DMGetDimension(dm[i], &dim); CHKERRQ(ierr);
    ierr = SetupDMByDegree(dm[i], level_degrees[i], num_comp_u, dim,
                           bp_options[bp_choice].enforce_bc, bp_options[bp_choice].bc_func);
    CHKERRQ(ierr);

    // Create vectors
    ierr = DMCreateGlobalVector(dm[i], &X[i]); CHKERRQ(ierr);
    ierr = VecGetLocalSize(X[i], &l_size[i]); CHKERRQ(ierr);
    ierr = VecGetSize(X[i], &g_size[i]); CHKERRQ(ierr);
    ierr = DMCreateLocalVector(dm[i], &X_loc[i]); CHKERRQ(ierr);
    ierr = VecGetSize(X_loc[i], &xl_size[i]); CHKERRQ(ierr);

    // Operator
    ierr = PetscMalloc1(1, &user_O[i]); CHKERRQ(ierr);
    ierr = MatCreateShell(comm, l_size[i], l_size[i], g_size[i], g_size[i],
                          user_O[i], &mat_O[i]); CHKERRQ(ierr);
    ierr = MatShellSetOperation(mat_O[i], MATOP_MULT,
                                (void(*)(void))MatMult_Ceed); CHKERRQ(ierr);
    ierr = MatShellSetOperation(mat_O[i], MATOP_GET_DIAGONAL,
                                (void(*)(void))MatGetDiag); CHKERRQ(ierr);
    ierr = MatShellSetVecType(mat_O[i], vec_type); CHKERRQ(ierr);

    // Level transfers
    if (i > 0) {
      // Interp
      ierr = PetscMalloc1(1, &user_pr[i]); CHKERRQ(ierr);
      ierr = MatCreateShell(comm, l_size[i], l_size[i-1], g_size[i], g_size[i-1],
                            user_pr[i], &mat_pr[i]); CHKERRQ(ierr);
      ierr = MatShellSetOperation(mat_pr[i], MATOP_MULT,
                                  (void(*)(void))MatMult_Prolong);
      CHKERRQ(ierr);
      ierr = MatShellSetOperation(mat_pr[i], MATOP_MULT_TRANSPOSE,
                                  (void(*)(void))MatMult_Restrict);
      CHKERRQ(ierr);
      ierr = MatShellSetVecType(mat_pr[i], vec_type); CHKERRQ(ierr);
    }
  }
  ierr = VecDuplicate(X[fine_level], &rhs); CHKERRQ(ierr);

  // Print global grid information
  if (!test_mode) {
    PetscInt P = degree + 1, Q = P + q_extra;

    const char *used_resource;
    CeedGetResource(ceed, &used_resource);

    ierr = VecGetType(X[0], &vec_type); CHKERRQ(ierr);

    ierr = PetscPrintf(comm,
                       "\n-- CEED Benchmark Problem %" CeedInt_FMT " -- libCEED + PETSc + PCMG --\n"
                       "  PETSc:\n"
                       "    PETSc Vec Type                     : %s\n"
                       "  libCEED:\n"
                       "    libCEED Backend                    : %s\n"
                       "    libCEED Backend MemType            : %s\n"
                       "  Mesh:\n"
                       "    Number of 1D Basis Nodes (p)       : %" CeedInt_FMT "\n"
                       "    Number of 1D Quadrature Points (q) : %" CeedInt_FMT "\n"
                       "    Global Nodes                       : %" PetscInt_FMT "\n"
                       "    Owned Nodes                        : %" PetscInt_FMT "\n"
                       "    DoF per node                       : %" PetscInt_FMT "\n"
                       "  Multigrid:\n"
                       "    Number of Levels                   : %" CeedInt_FMT "\n",
                       bp_choice+1, vec_type, used_resource,
                       CeedMemTypes[mem_type_backend],
                       P, Q, g_size[fine_level]/num_comp_u, l_size[fine_level]/num_comp_u,
                       num_comp_u, num_levels); CHKERRQ(ierr);
  }

  // Create RHS vector
  ierr = VecDuplicate(X_loc[fine_level], &rhs_loc); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhs_loc); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(rhs_loc, &r, &mem_type); CHKERRQ(ierr);
  CeedVectorCreate(ceed, xl_size[fine_level], &rhs_ceed);
  CeedVectorSetArray(rhs_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, r);

  // Set up libCEED operators on each level
  ierr = PetscMalloc1(num_levels, &ceed_data); CHKERRQ(ierr);
  for (PetscInt i=0; i<num_levels; i++) {
    // Print level information
    if (!test_mode && (i == 0 || i == fine_level)) {
      ierr = PetscPrintf(comm,"    Level %" PetscInt_FMT " (%s):\n"
                         "      Number of 1D Basis Nodes (p)     : %" CeedInt_FMT "\n"
                         "      Global Nodes                     : %" PetscInt_FMT "\n"
                         "      Owned Nodes                      : %" PetscInt_FMT "\n",
                         i, (i? "fine" : "coarse"), level_degrees[i] + 1,
                         g_size[i]/num_comp_u, l_size[i]/num_comp_u); CHKERRQ(ierr);
    }
    ierr = PetscMalloc1(1, &ceed_data[i]); CHKERRQ(ierr);
    ierr = SetupLibceedByDegree(dm[i], ceed, level_degrees[i], dim, q_extra,
                                dim, num_comp_u, g_size[i], xl_size[i], bp_options[bp_choice],
                                ceed_data[i], i==(fine_level), rhs_ceed, &target);
    CHKERRQ(ierr);
  }

  // Gather RHS
  CeedVectorTakeArray(rhs_ceed, MemTypeP2C(mem_type), NULL);
  ierr = VecRestoreArrayAndMemType(rhs_loc, &r); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhs); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dm[fine_level], rhs_loc, ADD_VALUES, rhs); CHKERRQ(ierr);
  CeedVectorDestroy(&rhs_ceed);

  // Create the restriction/interpolation QFunction
  CeedQFunctionCreateIdentity(ceed, num_comp_u, CEED_EVAL_NONE, CEED_EVAL_INTERP,
                              &qf_restrict);
  CeedQFunctionCreateIdentity(ceed, num_comp_u, CEED_EVAL_INTERP, CEED_EVAL_NONE,
                              &qf_prolong);

  // Set up libCEED level transfer operators
  ierr = CeedLevelTransferSetup(ceed, num_levels, num_comp_u, ceed_data,
                                level_degrees,
                                qf_restrict, qf_prolong); CHKERRQ(ierr);

  // Create the error QFunction
  CeedQFunctionCreateInterior(ceed, 1, bp_options[bp_choice].error,
                              bp_options[bp_choice].error_loc, &qf_error);
  CeedQFunctionAddInput(qf_error, "u", num_comp_u, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_error, "true_soln", num_comp_u, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_error, "error", num_comp_u, CEED_EVAL_NONE);

  // Create the error operator
  CeedOperatorCreate(ceed, qf_error, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_error);
  CeedOperatorSetField(op_error, "u", ceed_data[fine_level]->elem_restr_u,
                       ceed_data[fine_level]->basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_error, "true_soln",
                       ceed_data[fine_level]->elem_restr_u_i,
                       CEED_BASIS_COLLOCATED, target);
  CeedOperatorSetField(op_error, "error", ceed_data[fine_level]->elem_restr_u_i,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Calculate multiplicity
  for (int i=0; i<num_levels; i++) {
    PetscScalar *x;

    // CEED vector
    ierr = VecZeroEntries(X_loc[i]); CHKERRQ(ierr);
    ierr = VecGetArray(X_loc[i], &x); CHKERRQ(ierr);
    CeedVectorSetArray(ceed_data[i]->x_ceed, CEED_MEM_HOST, CEED_USE_POINTER, x);

    // Multiplicity
    CeedElemRestrictionGetMultiplicity(ceed_data[i]->elem_restr_u,
                                       ceed_data[i]->x_ceed);
    CeedVectorSyncArray(ceed_data[i]->x_ceed, CEED_MEM_HOST);

    // Restore vector
    ierr = VecRestoreArray(X_loc[i], &x); CHKERRQ(ierr);

    // Creat mult vector
    ierr = VecDuplicate(X_loc[i], &mult[i]); CHKERRQ(ierr);

    // Local-to-global
    ierr = VecZeroEntries(X[i]); CHKERRQ(ierr);
    ierr = DMLocalToGlobal(dm[i], X_loc[i], ADD_VALUES, X[i]);
    CHKERRQ(ierr);
    ierr = VecZeroEntries(X_loc[i]); CHKERRQ(ierr);

    // Global-to-local
    ierr = DMGlobalToLocal(dm[i], X[i], INSERT_VALUES, mult[i]);
    CHKERRQ(ierr);
    ierr = VecZeroEntries(X[i]); CHKERRQ(ierr);

    // Multiplicity scaling
    ierr = VecReciprocal(mult[i]);
  }

  // Set up Mat
  for (int i=0; i<num_levels; i++) {
    // User Operator
    user_O[i]->comm = comm;
    user_O[i]->dm = dm[i];
    user_O[i]->X_loc = X_loc[i];
    ierr = VecDuplicate(X_loc[i], &user_O[i]->Y_loc); CHKERRQ(ierr);
    user_O[i]->x_ceed = ceed_data[i]->x_ceed;
    user_O[i]->y_ceed = ceed_data[i]->y_ceed;
    user_O[i]->op = ceed_data[i]->op_apply;
    user_O[i]->ceed = ceed;

    if (i > 0) {
      // Prolongation/Restriction Operator
      user_pr[i]->comm = comm;
      user_pr[i]->dmf = dm[i];
      user_pr[i]->dmc = dm[i-1];
      user_pr[i]->loc_vec_c = X_loc[i-1];
      user_pr[i]->loc_vec_f = user_O[i]->Y_loc;
      user_pr[i]->mult_vec = mult[i];
      user_pr[i]->ceed_vec_c = user_O[i-1]->x_ceed;
      user_pr[i]->ceed_vec_f = user_O[i]->y_ceed;
      user_pr[i]->op_prolong = ceed_data[i]->op_prolong;
      user_pr[i]->op_restrict = ceed_data[i]->op_restrict;
      user_pr[i]->ceed = ceed;
    }
  }

  // Assemble coarse grid Jacobian for AMG (or other sparse matrix) solve
  ierr = DMCreateMatrix(dm[0], &mat_coarse); CHKERRQ(ierr);

  ierr = PetscLogEventRegister("AssembleMatrix", MAT_CLASSID, &assemble_event);
  CHKERRQ(ierr);
  {
    // Assemble matrix analytically
    PetscCount num_entries;
    CeedInt *rows, *cols;
    CeedVector coo_values;
    CeedOperatorLinearAssembleSymbolic(user_O[0]->op, &num_entries, &rows, &cols);
    ISLocalToGlobalMapping ltog_row, ltog_col;
    ierr = MatGetLocalToGlobalMapping(mat_coarse, &ltog_row, &ltog_col);
    CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingApply(ltog_row, num_entries, rows, rows);
    CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingApply(ltog_col, num_entries, cols, cols);
    CHKERRQ(ierr);
    ierr = MatSetPreallocationCOO(mat_coarse, num_entries, rows, cols);
    CHKERRQ(ierr);
    free(rows);
    free(cols);
    CeedVectorCreate(ceed, num_entries, &coo_values);
    ierr = PetscLogEventBegin(assemble_event, mat_coarse, 0, 0, 0); CHKERRQ(ierr);
    CeedOperatorLinearAssemble(user_O[0]->op, coo_values);
    const CeedScalar *values;
    CeedVectorGetArrayRead(coo_values, CEED_MEM_HOST, &values);
    ierr = MatSetValuesCOO(mat_coarse, values, ADD_VALUES); CHKERRQ(ierr);
    CeedVectorRestoreArrayRead(coo_values, &values);
    ierr = PetscLogEventEnd(assemble_event, mat_coarse, 0, 0, 0); CHKERRQ(ierr);
    CeedVectorDestroy(&coo_values);
  }

  // Set up KSP
  ierr = KSPCreate(comm, &ksp); CHKERRQ(ierr);
  {
    ierr = KSPSetType(ksp, KSPCG); CHKERRQ(ierr);
    ierr = KSPSetNormType(ksp, KSP_NORM_NATURAL); CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT,
                            PETSC_DEFAULT); CHKERRQ(ierr);
  }
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, mat_O[fine_level], mat_O[fine_level]);
  CHKERRQ(ierr);

  // Set up PCMG
  ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
  PCMGCycleType pcmg_cycle_type = PC_MG_CYCLE_V;
  {
    ierr = PCSetType(pc, PCMG); CHKERRQ(ierr);

    // PCMG levels
    ierr = PCMGSetLevels(pc, num_levels, NULL); CHKERRQ(ierr);
    for (int i=0; i<num_levels; i++) {
      // Smoother
      KSP smoother;
      PC smoother_pc;
      ierr = PCMGGetSmoother(pc, i, &smoother); CHKERRQ(ierr);
      ierr = KSPSetType(smoother, KSPCHEBYSHEV); CHKERRQ(ierr);
      ierr = KSPChebyshevEstEigSet(smoother, 0, 0.1, 0, 1.1); CHKERRQ(ierr);
      ierr = KSPChebyshevEstEigSetUseNoisy(smoother, PETSC_TRUE); CHKERRQ(ierr);
      ierr = KSPSetOperators(smoother, mat_O[i], mat_O[i]); CHKERRQ(ierr);
      ierr = KSPGetPC(smoother, &smoother_pc); CHKERRQ(ierr);
      ierr = PCSetType(smoother_pc, PCJACOBI); CHKERRQ(ierr);
      ierr = PCJacobiSetType(smoother_pc, PC_JACOBI_DIAGONAL); CHKERRQ(ierr);

      // Work vector
      if (i < num_levels - 1) {
        ierr = PCMGSetX(pc, i, X[i]); CHKERRQ(ierr);
      }

      // Level transfers
      if (i > 0) {
        // Interpolation
        ierr = PCMGSetInterpolation(pc, i, mat_pr[i]); CHKERRQ(ierr);
      }

      // Coarse solve
      KSP coarse;
      PC coarse_pc;
      ierr = PCMGGetCoarseSolve(pc, &coarse); CHKERRQ(ierr);
      ierr = KSPSetType(coarse, KSPPREONLY); CHKERRQ(ierr);
      ierr = KSPSetOperators(coarse, mat_coarse, mat_coarse); CHKERRQ(ierr);

      ierr = KSPGetPC(coarse, &coarse_pc); CHKERRQ(ierr);
      ierr = PCSetType(coarse_pc, PCGAMG); CHKERRQ(ierr);

      ierr = KSPSetOptionsPrefix(coarse, "coarse_"); CHKERRQ(ierr);
      ierr = PCSetOptionsPrefix(coarse_pc, "coarse_"); CHKERRQ(ierr);
      ierr = KSPSetFromOptions(coarse); CHKERRQ(ierr);
      ierr = PCSetFromOptions(coarse_pc); CHKERRQ(ierr);
    }

    // PCMG options
    ierr = PCMGSetType(pc, PC_MG_MULTIPLICATIVE); CHKERRQ(ierr);
    ierr = PCMGSetNumberSmooth(pc, 3); CHKERRQ(ierr);
    ierr = PCMGSetCycleType(pc, pcmg_cycle_type); CHKERRQ(ierr);
  }

  // First run, if benchmarking
  if (benchmark_mode) {
    ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 1);
    CHKERRQ(ierr);
    ierr = VecZeroEntries(X[fine_level]); CHKERRQ(ierr);
    my_rt_start = MPI_Wtime();
    ierr = KSPSolve(ksp, rhs, X[fine_level]); CHKERRQ(ierr);
    my_rt = MPI_Wtime() - my_rt_start;
    ierr = MPI_Allreduce(MPI_IN_PLACE, &my_rt, 1, MPI_DOUBLE, MPI_MIN, comm);
    CHKERRQ(ierr);
    // Set maxits based on first iteration timing
    if (my_rt > 0.02) {
      ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 5);
      CHKERRQ(ierr);
    } else {
      ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 20);
      CHKERRQ(ierr);
    }
  }

  // Timed solve
  ierr = VecZeroEntries(X[fine_level]); CHKERRQ(ierr);
  ierr = PetscBarrier((PetscObject)ksp); CHKERRQ(ierr);

  // -- Performance logging
  ierr = PetscLogStageRegister("Solve Stage", &solve_stage); CHKERRQ(ierr);
  ierr = PetscLogStagePush(solve_stage); CHKERRQ(ierr);

  // -- Solve
  my_rt_start = MPI_Wtime();
  ierr = KSPSolve(ksp, rhs, X[fine_level]); CHKERRQ(ierr);
  my_rt = MPI_Wtime() - my_rt_start;


  // -- Performance logging
  ierr = PetscLogStagePop();

  // Output results
  {
    KSPType ksp_type;
    PCMGType pcmg_type;
    KSPConvergedReason reason;
    PetscReal rnorm;
    PetscInt its;
    ierr = KSPGetType(ksp, &ksp_type); CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(ksp, &reason); CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp, &its); CHKERRQ(ierr);
    ierr = KSPGetResidualNorm(ksp, &rnorm); CHKERRQ(ierr);
    ierr = PCMGGetType(pc, &pcmg_type); CHKERRQ(ierr);
    if (!test_mode || reason < 0 || rnorm > 1e-8) {
      ierr = PetscPrintf(comm,
                         "  KSP:\n"
                         "    KSP Type                           : %s\n"
                         "    KSP Convergence                    : %s\n"
                         "    Total KSP Iterations               : %" PetscInt_FMT "\n"
                         "    Final rnorm                        : %e\n",
                         ksp_type, KSPConvergedReasons[reason], its,
                         (double)rnorm); CHKERRQ(ierr);
      ierr = PetscPrintf(comm,
                         "  PCMG:\n"
                         "    PCMG Type                          : %s\n"
                         "    PCMG Cycle Type                    : %s\n",
                         PCMGTypes[pcmg_type],
                         PCMGCycleTypes[pcmg_cycle_type]); CHKERRQ(ierr);
    }
    if (!test_mode) {
      ierr = PetscPrintf(comm,"  Performance:\n"); CHKERRQ(ierr);
    }
    {
      PetscReal max_error;
      ierr = ComputeErrorMax(user_O[fine_level], op_error, X[fine_level], target,
                             &max_error); CHKERRQ(ierr);
      PetscReal tol = 5e-2;
      if (!test_mode || max_error > tol) {
        ierr = MPI_Allreduce(&my_rt, &rt_min, 1, MPI_DOUBLE, MPI_MIN, comm);
        CHKERRQ(ierr);
        ierr = MPI_Allreduce(&my_rt, &rt_max, 1, MPI_DOUBLE, MPI_MAX, comm);
        CHKERRQ(ierr);
        ierr = PetscPrintf(comm,
                           "    Pointwise Error (max)              : %e\n"
                           "    CG Solve Time                      : %g (%g) sec\n",
                           (double)max_error, rt_max, rt_min); CHKERRQ(ierr);
      }
    }
    if (benchmark_mode && (!test_mode)) {
      ierr = PetscPrintf(comm,
                         "    DoFs/Sec in CG                     : %g (%g) million\n",
                         1e-6*g_size[fine_level]*its/rt_max,
                         1e-6*g_size[fine_level]*its/rt_min);
      CHKERRQ(ierr);
    }
  }

  if (write_solution) {
    PetscViewer vtk_viewer_soln;

    ierr = PetscViewerCreate(comm, &vtk_viewer_soln); CHKERRQ(ierr);
    ierr = PetscViewerSetType(vtk_viewer_soln, PETSCVIEWERVTK); CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(vtk_viewer_soln, "solution.vtu"); CHKERRQ(ierr);
    ierr = VecView(X[fine_level], vtk_viewer_soln); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vtk_viewer_soln); CHKERRQ(ierr);
  }

  // Cleanup
  for (int i=0; i<num_levels; i++) {
    ierr = VecDestroy(&X[i]); CHKERRQ(ierr);
    ierr = VecDestroy(&X_loc[i]); CHKERRQ(ierr);
    ierr = VecDestroy(&mult[i]); CHKERRQ(ierr);
    ierr = VecDestroy(&user_O[i]->Y_loc); CHKERRQ(ierr);
    ierr = MatDestroy(&mat_O[i]); CHKERRQ(ierr);
    ierr = PetscFree(user_O[i]); CHKERRQ(ierr);
    if (i > 0) {
      ierr = MatDestroy(&mat_pr[i]); CHKERRQ(ierr);
      ierr = PetscFree(user_pr[i]); CHKERRQ(ierr);
    }
    ierr = CeedDataDestroy(i, ceed_data[i]); CHKERRQ(ierr);
    ierr = DMDestroy(&dm[i]); CHKERRQ(ierr);
  }
  ierr = PetscFree(level_degrees); CHKERRQ(ierr);
  ierr = PetscFree(dm); CHKERRQ(ierr);
  ierr = PetscFree(X); CHKERRQ(ierr);
  ierr = PetscFree(X_loc); CHKERRQ(ierr);
  ierr = PetscFree(mult); CHKERRQ(ierr);
  ierr = PetscFree(mat_O); CHKERRQ(ierr);
  ierr = PetscFree(mat_pr); CHKERRQ(ierr);
  ierr = PetscFree(ceed_data); CHKERRQ(ierr);
  ierr = PetscFree(user_O); CHKERRQ(ierr);
  ierr = PetscFree(user_pr); CHKERRQ(ierr);
  ierr = PetscFree(l_size); CHKERRQ(ierr);
  ierr = PetscFree(xl_size); CHKERRQ(ierr);
  ierr = PetscFree(g_size); CHKERRQ(ierr);
  ierr = VecDestroy(&rhs); CHKERRQ(ierr);
  ierr = VecDestroy(&rhs_loc); CHKERRQ(ierr);
  ierr = MatDestroy(&mat_coarse); CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
  ierr = DMDestroy(&dm_orig); CHKERRQ(ierr);
  CeedVectorDestroy(&target);
  CeedQFunctionDestroy(&qf_error);
  CeedQFunctionDestroy(&qf_restrict);
  CeedQFunctionDestroy(&qf_prolong);
  CeedOperatorDestroy(&op_error);
  CeedDestroy(&ceed);
  return PetscFinalize();
}
