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

#include <stdbool.h>
#include <string.h>
#include <ceed.h>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscksp.h>

#include "bpssphere.h"
#include "include/sphereproblemdata.h"
#include "include/petscutils.h"
#include "include/petscversion.h"
#include "include/matops.h"
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
  char ceed_resource[PETSC_MAX_PATH_LEN] = "/cpu/self",
      filename[PETSC_MAX_PATH_LEN];
  double my_rt_start, my_rt, rt_min, rt_max;
  PetscInt degree = 3, q_extra, l_size, g_size, topo_dim = 2, num_comp_x = 3,
           num_comp_u = 1, xl_size;
  PetscScalar *r;
  PetscBool test_mode, benchmark_mode, read_mesh, write_solution, simplex;
  PetscLogStage solve_stage;
  Vec X, X_loc, rhs, rhs_loc;
  Mat mat_O;
  KSP ksp;
  DM  dm;
  UserO user_O;
  Ceed ceed;
  CeedData ceed_data;
  CeedQFunction qf_error;
  CeedOperator op_error;
  CeedVector rhs_ceed, target;
  BPType bp_choice;
  VecType vec_type;
  PetscMemType mem_type;

  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;

  // Read command line options
  PetscOptionsBegin(comm, NULL, "CEED BPs in PETSc", NULL);
  bp_choice = CEED_BP1;
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
  degree = test_mode ? 3 : 2;
  ierr = PetscOptionsInt("-degree", "Polynomial degree of tensor product basis",
                         NULL, degree, &degree, NULL); CHKERRQ(ierr);
  q_extra = bp_options[bp_choice].q_extra;
  ierr = PetscOptionsInt("-q_extra", "Number of extra quadrature points",
                         NULL, q_extra, &q_extra, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, ceed_resource, ceed_resource,
                            sizeof(ceed_resource), NULL); CHKERRQ(ierr);
  read_mesh = PETSC_FALSE;
  ierr = PetscOptionsString("-mesh", "Read mesh from file", NULL,
                            filename, filename, sizeof(filename), &read_mesh);
  CHKERRQ(ierr);
  simplex = PETSC_FALSE;
  ierr = PetscOptionsBool("-simplex", "Use simplices, or tensor product cells",
                          NULL, simplex, &simplex, NULL); CHKERRQ(ierr);
  PetscOptionsEnd();

  // Setup DM
  if (read_mesh) {
    ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, filename, NULL, PETSC_TRUE,
                                &dm);
    CHKERRQ(ierr);
  } else {
    // Create the mesh as a 0-refined sphere. This will create a cubic surface,
    // not a box, and will snap to the unit sphere upon refinement.
    ierr = DMPlexCreateSphereMesh(PETSC_COMM_WORLD, topo_dim, simplex, 1., &dm);
    CHKERRQ(ierr);
    // Set the object name
    ierr = PetscObjectSetName((PetscObject)dm, "Sphere"); CHKERRQ(ierr);
    // Refine DMPlex with uniform refinement using runtime option -dm_refine
    ierr = DMPlexSetRefinementUniform(dm, PETSC_TRUE); CHKERRQ(ierr);
  }
  ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
  // View DMPlex via runtime option
  ierr = DMViewFromOptions(dm, NULL, "-dm_view"); CHKERRQ(ierr);

  // Create DM
  ierr = SetupDMByDegree(dm, degree, num_comp_u, topo_dim, false,
                         (BCFunction)NULL);
  CHKERRQ(ierr);

  // Create vectors
  ierr = DMCreateGlobalVector(dm, &X); CHKERRQ(ierr);
  ierr = VecGetLocalSize(X, &l_size); CHKERRQ(ierr);
  ierr = VecGetSize(X, &g_size); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm, &X_loc); CHKERRQ(ierr);
  ierr = VecGetSize(X_loc, &xl_size); CHKERRQ(ierr);
  ierr = VecDuplicate(X, &rhs); CHKERRQ(ierr);

  // Operator
  ierr = PetscMalloc1(1, &user_O); CHKERRQ(ierr);
  ierr = MatCreateShell(comm, l_size, l_size, g_size, g_size,
                        user_O, &mat_O); CHKERRQ(ierr);
  ierr = MatShellSetOperation(mat_O, MATOP_MULT,
                              (void(*)(void))MatMult_Ceed); CHKERRQ(ierr);

  // Set up libCEED
  CeedInit(ceed_resource, &ceed);
  CeedMemType mem_type_backend;
  CeedGetPreferredMemType(ceed, &mem_type_backend);

  ierr = DMGetVecType(dm, &vec_type); CHKERRQ(ierr);
  if (!vec_type) { // Not yet set by user -dm_vec_type
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
    ierr = DMSetVecType(dm, vec_type); CHKERRQ(ierr);
  }

  // Print summary
  if (!test_mode) {
    PetscInt P = degree + 1, Q = P + q_extra;
    const char *used_resource;
    CeedGetResource(ceed, &used_resource);
    ierr = PetscPrintf(comm,
                       "\n-- CEED Benchmark Problem %" CeedInt_FMT
                       " on the Sphere -- libCEED + PETSc --\n"
                       "  libCEED:\n"
                       "    libCEED Backend                    : %s\n"
                       "    libCEED Backend MemType            : %s\n"
                       "  Mesh:\n"
                       "    Number of 1D Basis Nodes (p)       : %" CeedInt_FMT "\n"
                       "    Number of 1D Quadrature Points (q) : %" CeedInt_FMT "\n"
                       "    Global nodes                       : %" PetscInt_FMT "\n",
                       bp_choice+1, ceed_resource, CeedMemTypes[mem_type_backend], P, Q,
                       g_size/num_comp_u); CHKERRQ(ierr);
  }

  // Create RHS vector
  ierr = VecDuplicate(X_loc, &rhs_loc); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhs_loc); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(rhs_loc, &r, &mem_type); CHKERRQ(ierr);
  CeedVectorCreate(ceed, xl_size, &rhs_ceed);
  CeedVectorSetArray(rhs_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, r);

  // Setup libCEED's objects
  ierr = PetscMalloc1(1, &ceed_data); CHKERRQ(ierr);
  ierr = SetupLibceedByDegree(dm, ceed, degree, topo_dim, q_extra, num_comp_x,
                              num_comp_u, g_size, xl_size, bp_options[bp_choice],
                              ceed_data, true, rhs_ceed, &target); CHKERRQ(ierr);

  // Gather RHS
  CeedVectorTakeArray(rhs_ceed, MemTypeP2C(mem_type), NULL);
  ierr = VecRestoreArrayAndMemType(rhs_loc, &r); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhs); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dm, rhs_loc, ADD_VALUES, rhs); CHKERRQ(ierr);
  CeedVectorDestroy(&rhs_ceed);

  // Create the error Q-function
  CeedQFunctionCreateInterior(ceed, 1, bp_options[bp_choice].error,
                              bp_options[bp_choice].error_loc, &qf_error);
  CeedQFunctionAddInput(qf_error, "u", num_comp_u, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_error, "true_soln", num_comp_u, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_error, "error", num_comp_u, CEED_EVAL_NONE);

  // Create the error operator
  CeedOperatorCreate(ceed, qf_error, NULL, NULL, &op_error);
  CeedOperatorSetField(op_error, "u", ceed_data->elem_restr_u,
                       ceed_data->basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_error, "true_soln", ceed_data->elem_restr_u_i,
                       CEED_BASIS_COLLOCATED, target);
  CeedOperatorSetField(op_error, "error", ceed_data->elem_restr_u_i,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Set up Mat
  user_O->comm = comm;
  user_O->dm = dm;
  user_O->X_loc = X_loc;
  ierr = VecDuplicate(X_loc, &user_O->Y_loc); CHKERRQ(ierr);
  user_O->x_ceed = ceed_data->x_ceed;
  user_O->y_ceed = ceed_data->y_ceed;
  user_O->op = ceed_data->op_apply;
  user_O->ceed = ceed;

  // Setup solver
  ierr = KSPCreate(comm, &ksp); CHKERRQ(ierr);
  {
    PC pc;
    ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
    if (bp_choice == CEED_BP1 || bp_choice == CEED_BP2) {
      ierr = PCSetType(pc, PCJACOBI); CHKERRQ(ierr);
      ierr = PCJacobiSetType(pc, PC_JACOBI_ROWSUM); CHKERRQ(ierr);
    } else {
      ierr = PCSetType(pc, PCNONE); CHKERRQ(ierr);
      MatNullSpace nullspace;

      ierr = MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace);
      CHKERRQ(ierr);
      ierr = MatSetNullSpace(mat_O, nullspace); CHKERRQ(ierr);
      ierr = MatNullSpaceDestroy(&nullspace); CHKERRQ(ierr);
    }
    ierr = KSPSetType(ksp, KSPCG); CHKERRQ(ierr);
    ierr = KSPSetNormType(ksp, KSP_NORM_NATURAL); CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT,
                            PETSC_DEFAULT); CHKERRQ(ierr);
  }
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, mat_O, mat_O); CHKERRQ(ierr);

  // First run, if benchmarking
  if (benchmark_mode) {
    ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 1);
    CHKERRQ(ierr);
    my_rt_start = MPI_Wtime();
    ierr = KSPSolve(ksp, rhs, X); CHKERRQ(ierr);
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
  ierr = VecZeroEntries(X); CHKERRQ(ierr);
  ierr = PetscBarrier((PetscObject)ksp); CHKERRQ(ierr);

  // -- Performance logging
  ierr = PetscLogStageRegister("Solve Stage", &solve_stage); CHKERRQ(ierr);
  ierr = PetscLogStagePush(solve_stage); CHKERRQ(ierr);

  // -- Solve
  my_rt_start = MPI_Wtime();
  ierr = KSPSolve(ksp, rhs, X); CHKERRQ(ierr);
  my_rt = MPI_Wtime() - my_rt_start;

  // -- Performance logging
  ierr = PetscLogStagePop();

  // Output results
  {
    KSPType ksp_type;
    KSPConvergedReason reason;
    PetscReal rnorm;
    PetscInt its;
    ierr = KSPGetType(ksp, &ksp_type); CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(ksp, &reason); CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp, &its); CHKERRQ(ierr);
    ierr = KSPGetResidualNorm(ksp, &rnorm); CHKERRQ(ierr);
    if (!test_mode || reason < 0 || rnorm > 1e-8) {
      ierr = PetscPrintf(comm,
                         "  KSP:\n"
                         "    KSP Type                           : %s\n"
                         "    KSP Convergence                    : %s\n"
                         "    Total KSP Iterations               : %" PetscInt_FMT "\n"
                         "    Final rnorm                        : %e\n",
                         ksp_type, KSPConvergedReasons[reason], its,
                         (double)rnorm); CHKERRQ(ierr);
    }
    if (!test_mode) {
      ierr = PetscPrintf(comm,"  Performance:\n"); CHKERRQ(ierr);
    }
    {
      PetscReal max_error;
      ierr = ComputeErrorMax(user_O, op_error, X, target, &max_error);
      CHKERRQ(ierr);
      PetscReal tol = 5e-4;
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
                         1e-6*g_size*its/rt_max, 1e-6*g_size*its/rt_min); CHKERRQ(ierr);
    }
  }

  // Output solution
  if (write_solution) {
    PetscViewer vtk_viewer_soln;

    ierr = PetscViewerCreate(comm, &vtk_viewer_soln); CHKERRQ(ierr);
    ierr = PetscViewerSetType(vtk_viewer_soln, PETSCVIEWERVTK); CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(vtk_viewer_soln, "solution.vtu"); CHKERRQ(ierr);
    ierr = VecView(X, vtk_viewer_soln); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vtk_viewer_soln); CHKERRQ(ierr);
  }

  // Cleanup
  ierr = VecDestroy(&X); CHKERRQ(ierr);
  ierr = VecDestroy(&X_loc); CHKERRQ(ierr);
  ierr = VecDestroy(&user_O->Y_loc); CHKERRQ(ierr);
  ierr = MatDestroy(&mat_O); CHKERRQ(ierr);
  ierr = PetscFree(user_O); CHKERRQ(ierr);
  ierr = CeedDataDestroy(0, ceed_data); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);

  ierr = VecDestroy(&rhs); CHKERRQ(ierr);
  ierr = VecDestroy(&rhs_loc); CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
  CeedVectorDestroy(&target);
  CeedQFunctionDestroy(&qf_error);
  CeedOperatorDestroy(&op_error);
  CeedDestroy(&ceed);
  return PetscFinalize();
}
