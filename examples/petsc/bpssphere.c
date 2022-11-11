// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

//                        libCEED + PETSc Example: CEED BPs
//
// This example demonstrates a simple usage of libCEED with PETSc to solve the
// CEED BP benchmark problems, see http://ceed.exascaleproject.org/bps,
// on a closed surface, such as the one of a discrete sphere.
//
// The code uses higher level communication protocols in DMPlex.
//
// Build with:
//
//     make bpssphere [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     bpssphere -problem bp1 -degree 3
//     bpssphere -problem bp2 -degree 3
//     bpssphere -problem bp3 -degree 3
//     bpssphere -problem bp4 -degree 3
//     bpssphere -problem bp5 -degree 3 -ceed /cpu/self
//     bpssphere -problem bp6 -degree 3 -ceed /gpu/cuda
//
//TESTARGS -ceed {ceed_resource} -test -problem bp3 -degree 3 -dm_refine 2

/// @file
/// CEED BPs example using PETSc with DMPlex
/// See bps.c for a "raw" implementation using a structured grid.
/// and bpsdmplex.c for an implementation using an unstructured grid.
static const char help[] = "Solve CEED BPs on a sphere using DMPlex in PETSc\n";

#include "bpssphere.h"

#include <ceed.h>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscksp.h>
#include <stdbool.h>
#include <string.h>

#include "include/libceedsetup.h"
#include "include/matops.h"
#include "include/petscutils.h"
#include "include/petscversion.h"
#include "include/sphereproblemdata.h"

#if PETSC_VERSION_LT(3, 12, 0)
#ifdef PETSC_HAVE_CUDA
#include <petsccuda.h>
// Note: With PETSc prior to version 3.12.0, providing the source path to
//       include 'cublas_v2.h' will be needed to use 'petsccuda.h'.
#endif
#endif

int main(int argc, char **argv) {
  MPI_Comm             comm;
  char                 ceed_resource[PETSC_MAX_PATH_LEN] = "/cpu/self", filename[PETSC_MAX_PATH_LEN];
  double               my_rt_start, my_rt, rt_min, rt_max;
  PetscInt             degree = 3, q_extra, l_size, g_size, topo_dim = 2, num_comp_x = 3, num_comp_u = 1, xl_size;
  PetscScalar         *r;
  PetscBool            test_mode, benchmark_mode, read_mesh, write_solution, simplex;
  PetscLogStage        solve_stage;
  Vec                  X, X_loc, rhs, rhs_loc;
  Mat                  mat_O;
  KSP                  ksp;
  DM                   dm;
  OperatorApplyContext op_apply_ctx, op_error_ctx;
  Ceed                 ceed;
  CeedData             ceed_data;
  CeedQFunction        qf_error;
  CeedOperator         op_error;
  CeedVector           rhs_ceed, target;
  BPType               bp_choice;
  VecType              vec_type;
  PetscMemType         mem_type;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  // Read command line options
  PetscOptionsBegin(comm, NULL, "CEED BPs in PETSc", NULL);
  bp_choice = CEED_BP1;
  PetscCall(PetscOptionsEnum("-problem", "CEED benchmark problem to solve", NULL, bp_types, (PetscEnum)bp_choice, (PetscEnum *)&bp_choice, NULL));
  num_comp_u = bp_options[bp_choice].num_comp_u;
  test_mode  = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-test", "Testing mode (do not print unless error is large)", NULL, test_mode, &test_mode, NULL));
  benchmark_mode = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-benchmark", "Benchmarking mode (prints benchmark statistics)", NULL, benchmark_mode, &benchmark_mode, NULL));
  write_solution = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-write_solution", "Write solution for visualization", NULL, write_solution, &write_solution, NULL));
  degree = test_mode ? 3 : 2;
  PetscCall(PetscOptionsInt("-degree", "Polynomial degree of tensor product basis", NULL, degree, &degree, NULL));
  q_extra = bp_options[bp_choice].q_extra;
  PetscCall(PetscOptionsInt("-q_extra", "Number of extra quadrature points", NULL, q_extra, &q_extra, NULL));
  PetscCall(PetscOptionsString("-ceed", "CEED resource specifier", NULL, ceed_resource, ceed_resource, sizeof(ceed_resource), NULL));
  read_mesh = PETSC_FALSE;
  PetscCall(PetscOptionsString("-mesh", "Read mesh from file", NULL, filename, filename, sizeof(filename), &read_mesh));
  simplex = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-simplex", "Use simplices, or tensor product cells", NULL, simplex, &simplex, NULL));
  PetscOptionsEnd();

  // Setup DM
  if (read_mesh) {
    PetscCall(DMPlexCreateFromFile(PETSC_COMM_WORLD, filename, NULL, PETSC_TRUE, &dm));
  } else {
    // Create the mesh as a 0-refined sphere. This will create a cubic surface,
    // not a box, and will snap to the unit sphere upon refinement.
    PetscCall(DMPlexCreateSphereMesh(PETSC_COMM_WORLD, topo_dim, simplex, 1., &dm));
    // Set the object name
    PetscCall(PetscObjectSetName((PetscObject)dm, "Sphere"));
    // Refine DMPlex with uniform refinement using runtime option -dm_refine
    PetscCall(DMPlexSetRefinementUniform(dm, PETSC_TRUE));
  }
  PetscCall(DMSetFromOptions(dm));
  // View DMPlex via runtime option
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  // Create DM
  PetscCall(SetupDMByDegree(dm, degree, q_extra, num_comp_u, topo_dim, false));

  // Create vectors
  PetscCall(DMCreateGlobalVector(dm, &X));
  PetscCall(VecGetLocalSize(X, &l_size));
  PetscCall(VecGetSize(X, &g_size));
  PetscCall(DMCreateLocalVector(dm, &X_loc));
  PetscCall(VecGetSize(X_loc, &xl_size));
  PetscCall(VecDuplicate(X, &rhs));

  // Operator
  PetscCall(PetscMalloc1(1, &op_apply_ctx));
  PetscCall(PetscMalloc1(1, &op_error_ctx));
  PetscCall(MatCreateShell(comm, l_size, l_size, g_size, g_size, op_apply_ctx, &mat_O));
  PetscCall(MatShellSetOperation(mat_O, MATOP_MULT, (void (*)(void))MatMult_Ceed));

  // Set up libCEED
  CeedInit(ceed_resource, &ceed);
  CeedMemType mem_type_backend;
  CeedGetPreferredMemType(ceed, &mem_type_backend);

  PetscCall(DMGetVecType(dm, &vec_type));
  if (!vec_type) {  // Not yet set by user -dm_vec_type
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
    PetscCall(DMSetVecType(dm, vec_type));
  }

  // Print summary
  if (!test_mode) {
    PetscInt    P = degree + 1, Q = P + q_extra;
    const char *used_resource;
    CeedGetResource(ceed, &used_resource);
    PetscCall(PetscPrintf(comm,
                          "\n-- CEED Benchmark Problem %" CeedInt_FMT " on the Sphere -- libCEED + PETSc --\n"
                          "  libCEED:\n"
                          "    libCEED Backend                         : %s\n"
                          "    libCEED Backend MemType                 : %s\n"
                          "  Mesh:\n"
                          "    Solution Order (P)                      : %" CeedInt_FMT "\n"
                          "    Quadrature  Order (Q)                   : %" CeedInt_FMT "\n"
                          "    Additional quadrature points (q_extra)  : %" CeedInt_FMT "\n"
                          "    Global nodes                            : %" PetscInt_FMT "\n",
                          bp_choice + 1, ceed_resource, CeedMemTypes[mem_type_backend], P, Q, q_extra, g_size / num_comp_u));
  }

  // Create RHS vector
  PetscCall(VecDuplicate(X_loc, &rhs_loc));
  PetscCall(VecZeroEntries(rhs_loc));
  PetscCall(VecGetArrayAndMemType(rhs_loc, &r, &mem_type));
  CeedVectorCreate(ceed, xl_size, &rhs_ceed);
  CeedVectorSetArray(rhs_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, r);

  // Setup libCEED's objects
  PetscCall(PetscMalloc1(1, &ceed_data));
  PetscCall(SetupLibceedByDegree(dm, ceed, degree, topo_dim, q_extra, num_comp_x, num_comp_u, g_size, xl_size, bp_options[bp_choice], ceed_data, true,
                                 rhs_ceed, &target));

  // Gather RHS
  CeedVectorTakeArray(rhs_ceed, MemTypeP2C(mem_type), NULL);
  PetscCall(VecRestoreArrayAndMemType(rhs_loc, &r));
  PetscCall(VecZeroEntries(rhs));
  PetscCall(DMLocalToGlobal(dm, rhs_loc, ADD_VALUES, rhs));
  CeedVectorDestroy(&rhs_ceed);

  // Create the error Q-function
  CeedQFunctionCreateInterior(ceed, 1, bp_options[bp_choice].error, bp_options[bp_choice].error_loc, &qf_error);
  CeedQFunctionAddInput(qf_error, "u", num_comp_u, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_error, "true_soln", num_comp_u, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_error, "qdata", ceed_data->q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_error, "error", num_comp_u, CEED_EVAL_INTERP);

  // Create the error operator
  CeedOperatorCreate(ceed, qf_error, NULL, NULL, &op_error);
  CeedOperatorSetField(op_error, "u", ceed_data->elem_restr_u, ceed_data->basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_error, "true_soln", ceed_data->elem_restr_u_i, CEED_BASIS_COLLOCATED, target);
  CeedOperatorSetField(op_error, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_COLLOCATED, ceed_data->q_data);
  CeedOperatorSetField(op_error, "error", ceed_data->elem_restr_u, ceed_data->basis_u, CEED_VECTOR_ACTIVE);

  // Set up apply operator context
  PetscCall(SetupApplyOperatorCtx(comm, dm, ceed, ceed_data, X_loc, op_apply_ctx));

  // Setup solver
  PetscCall(KSPCreate(comm, &ksp));
  {
    PC pc;
    PetscCall(KSPGetPC(ksp, &pc));
    if (bp_choice == CEED_BP1 || bp_choice == CEED_BP2) {
      PetscCall(PCSetType(pc, PCJACOBI));
      PetscCall(PCJacobiSetType(pc, PC_JACOBI_ROWSUM));
    } else {
      PetscCall(PCSetType(pc, PCNONE));
      MatNullSpace nullspace;

      PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace));
      PetscCall(MatSetNullSpace(mat_O, nullspace));
      PetscCall(MatNullSpaceDestroy(&nullspace));
    }
    PetscCall(KSPSetType(ksp, KSPCG));
    PetscCall(KSPSetNormType(ksp, KSP_NORM_NATURAL));
    PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  }
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetOperators(ksp, mat_O, mat_O));

  // First run, if benchmarking
  if (benchmark_mode) {
    PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 1));
    my_rt_start = MPI_Wtime();
    PetscCall(KSPSolve(ksp, rhs, X));
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
  PetscCall(VecZeroEntries(X));
  PetscCall(PetscBarrier((PetscObject)ksp));

  // -- Performance logging
  PetscCall(PetscLogStageRegister("Solve Stage", &solve_stage));
  PetscCall(PetscLogStagePush(solve_stage));

  // -- Solve
  my_rt_start = MPI_Wtime();
  PetscCall(KSPSolve(ksp, rhs, X));
  my_rt = MPI_Wtime() - my_rt_start;

  // -- Performance logging
  PetscCall(PetscLogStagePop());

  // Output results
  {
    KSPType            ksp_type;
    KSPConvergedReason reason;
    PetscReal          rnorm;
    PetscInt           its;
    PetscCall(KSPGetType(ksp, &ksp_type));
    PetscCall(KSPGetConvergedReason(ksp, &reason));
    PetscCall(KSPGetIterationNumber(ksp, &its));
    PetscCall(KSPGetResidualNorm(ksp, &rnorm));
    if (!test_mode || reason < 0 || rnorm > 1e-8) {
      PetscCall(PetscPrintf(comm,
                            "  KSP:\n"
                            "    KSP Type                                : %s\n"
                            "    KSP Convergence                         : %s\n"
                            "    Total KSP Iterations                    : %" PetscInt_FMT "\n"
                            "    Final rnorm                             : %e\n",
                            ksp_type, KSPConvergedReasons[reason], its, (double)rnorm));
    }
    if (!test_mode) {
      PetscCall(PetscPrintf(comm, "  Performance:\n"));
    }
    {
      // Set up error operator context
      PetscCall(SetupErrorOperatorCtx(comm, dm, ceed, ceed_data, X_loc, op_error, op_error_ctx));
      PetscScalar l2_error;
      PetscCall(ComputeL2Error(X, &l2_error, op_error_ctx));
      PetscReal tol = 5e-4;
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
      PetscCall(PetscPrintf(comm, "    DoFs/Sec in CG                            : %g (%g) million\n", 1e-6 * g_size * its / rt_max,
                            1e-6 * g_size * its / rt_min));
    }
  }

  // Output solution
  if (write_solution) {
    PetscViewer vtk_viewer_soln;

    PetscCall(PetscViewerCreate(comm, &vtk_viewer_soln));
    PetscCall(PetscViewerSetType(vtk_viewer_soln, PETSCVIEWERVTK));
    PetscCall(PetscViewerFileSetName(vtk_viewer_soln, "solution.vtu"));
    PetscCall(VecView(X, vtk_viewer_soln));
    PetscCall(PetscViewerDestroy(&vtk_viewer_soln));
  }

  // Cleanup
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&X_loc));
  PetscCall(VecDestroy(&op_apply_ctx->Y_loc));
  PetscCall(VecDestroy(&op_error_ctx->Y_loc));
  PetscCall(MatDestroy(&mat_O));
  PetscCall(PetscFree(op_apply_ctx));
  PetscCall(PetscFree(op_error_ctx));
  PetscCall(CeedDataDestroy(0, ceed_data));
  PetscCall(DMDestroy(&dm));

  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&rhs_loc));
  PetscCall(KSPDestroy(&ksp));
  CeedVectorDestroy(&target);
  CeedQFunctionDestroy(&qf_error);
  CeedOperatorDestroy(&op_error);
  CeedDestroy(&ceed);
  return PetscFinalize();
}
