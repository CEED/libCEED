// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

//                        libCEED + PETSc Example: CEED BPs
//
// This example demonstrates a simple usage of libCEED with PETSc to solve the
// CEED BP benchmark problems, see http://ceed.exascaleproject.org/bps.
//
// The code uses higher level communication protocols in DMPlex.
//
// Build with:
//
//     make bps [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     ./bps -problem bp1 -degree 3
//     ./bps -problem bp2 -degree 3
//     ./bps -problem bp3 -degree 3
//     ./bps -problem bp4 -degree 3
//     ./bps -problem bp5 -degree 3 -ceed /cpu/self
//     ./bps -problem bp6 -degree 3 -ceed /gpu/cuda
//
//TESTARGS -ceed {ceed_resource} -test -problem bp5 -degree 3 -ksp_max_it_clip 15,15

/// @file
/// CEED BPs example using PETSc with DMPlex
/// See bpsraw.c for a "raw" implementation using a structured grid.
const char help[] = "Solve CEED BPs using PETSc with DMPlex\n";

#include "bps.h"

#include <ceed.h>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscksp.h>
#include <petscsys.h>
#include <stdbool.h>
#include <string.h>

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

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

// Utility function, compute three factors of an integer
static void Split3(PetscInt size, PetscInt m[3], bool reverse) {
  for (PetscInt d = 0, size_left = size; d < 3; d++) {
    PetscInt try = (PetscInt)PetscCeilReal(PetscPowReal(size_left, 1. / (3 - d)));
    while (try * (size_left / try) != size_left) try++;
    m[reverse ? 2 - d : d] = try;
    size_left /= try;
  }
}

static int Max3(const PetscInt a[3]) { return PetscMax(a[0], PetscMax(a[1], a[2])); }

static int Min3(const PetscInt a[3]) { return PetscMin(a[0], PetscMin(a[1], a[2])); }

// -----------------------------------------------------------------------------
// Parameter structure for running problems
// -----------------------------------------------------------------------------
typedef struct RunParams_ *RunParams;
struct RunParams_ {
  MPI_Comm      comm;
  PetscBool     test_mode, read_mesh, user_l_nodes, write_solution;
  char         *filename, *hostname;
  PetscInt      local_nodes, degree, q_extra, dim, num_comp_u, *mesh_elem;
  PetscInt      ksp_max_it_clip[2];
  PetscMPIInt   ranks_per_node;
  BPType        bp_choice;
  PetscLogStage solve_stage;
};

// -----------------------------------------------------------------------------
// Main body of program, called in a loop for performance benchmarking purposes
// -----------------------------------------------------------------------------
static PetscErrorCode RunWithDM(RunParams rp, DM dm, const char *ceed_resource) {
  double        my_rt_start, my_rt, rt_min, rt_max;
  PetscInt      xl_size, l_size, g_size;
  PetscScalar  *r;
  Vec           X, X_loc, rhs, rhs_loc;
  Mat           mat_O;
  KSP           ksp;
  UserO         user_O;
  Ceed          ceed;
  CeedData      ceed_data;
  CeedQFunction qf_error;
  CeedOperator  op_error;
  CeedVector    rhs_ceed, target;
  VecType       vec_type;
  PetscMemType  mem_type;

  PetscFunctionBeginUser;
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

  // Create global and local solution vectors
  PetscCall(DMCreateGlobalVector(dm, &X));
  PetscCall(VecGetLocalSize(X, &l_size));
  PetscCall(VecGetSize(X, &g_size));
  PetscCall(DMCreateLocalVector(dm, &X_loc));
  PetscCall(VecGetSize(X_loc, &xl_size));
  PetscCall(VecDuplicate(X, &rhs));

  // Operator
  PetscCall(PetscMalloc1(1, &user_O));
  PetscCall(MatCreateShell(rp->comm, l_size, l_size, g_size, g_size, user_O, &mat_O));
  PetscCall(MatShellSetOperation(mat_O, MATOP_MULT, (void (*)(void))MatMult_Ceed));
  PetscCall(MatShellSetOperation(mat_O, MATOP_GET_DIAGONAL, (void (*)(void))MatGetDiag));
  PetscCall(MatShellSetVecType(mat_O, vec_type));

  // Print summary
  if (!rp->test_mode) {
    PetscInt    P = rp->degree + 1, Q = P + rp->q_extra;

    const char *used_resource;
    CeedGetResource(ceed, &used_resource);

    VecType vec_type;
    PetscCall(VecGetType(X, &vec_type));

    PetscInt c_start, c_end;
    PetscCall(DMPlexGetHeightStratum(dm, 0, &c_start, &c_end));
    PetscMPIInt comm_size;
    PetscCall(MPI_Comm_size(rp->comm, &comm_size));
    PetscCall(PetscPrintf(rp->comm,
                          "\n-- CEED Benchmark Problem %" CeedInt_FMT " -- libCEED + PETSc --\n"
                          "  MPI:\n"
                          "    Hostname                           : %s\n"
                          "    Total ranks                        : %d\n"
                          "    Ranks per compute node             : %d\n"
                          "  PETSc:\n"
                          "    PETSc Vec Type                     : %s\n"
                          "  libCEED:\n"
                          "    libCEED Backend                    : %s\n"
                          "    libCEED Backend MemType            : %s\n"
                          "  Mesh:\n"
                          "    Number of 1D Basis Nodes (P)       : %" CeedInt_FMT "\n"
                          "    Number of 1D Quadrature Points (Q) : %" CeedInt_FMT "\n"
                          "    Global nodes                       : %" PetscInt_FMT "\n"
                          "    Local Elements                     : %" PetscInt_FMT "\n"
                          "    Owned nodes                        : %" PetscInt_FMT "\n"
                          "    DoF per node                       : %" PetscInt_FMT "\n",
                          rp->bp_choice + 1, rp->hostname, comm_size, rp->ranks_per_node, vec_type, used_resource, CeedMemTypes[mem_type_backend], P,
                          Q, g_size / rp->num_comp_u, c_end - c_start, l_size / rp->num_comp_u, rp->num_comp_u));
  }

  // Create RHS vector
  PetscCall(VecDuplicate(X_loc, &rhs_loc));
  PetscCall(VecZeroEntries(rhs_loc));
  PetscCall(VecGetArrayAndMemType(rhs_loc, &r, &mem_type));
  CeedVectorCreate(ceed, xl_size, &rhs_ceed);
  CeedVectorSetArray(rhs_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, r);

  PetscCall(PetscMalloc1(1, &ceed_data));
  PetscCall(SetupLibceedByDegree(dm, ceed, rp->degree, rp->dim, rp->q_extra, rp->dim, rp->num_comp_u, g_size, xl_size, bp_options[rp->bp_choice],
                                 ceed_data, true, rhs_ceed, &target));

  // Gather RHS
  CeedVectorTakeArray(rhs_ceed, MemTypeP2C(mem_type), NULL);
  PetscCall(VecRestoreArrayAndMemType(rhs_loc, &r));
  PetscCall(VecZeroEntries(rhs));
  PetscCall(DMLocalToGlobal(dm, rhs_loc, ADD_VALUES, rhs));
  CeedVectorDestroy(&rhs_ceed);

  // Create the error QFunction
  CeedQFunctionCreateInterior(ceed, 1, bp_options[rp->bp_choice].error, bp_options[rp->bp_choice].error_loc, &qf_error);
  CeedQFunctionAddInput(qf_error, "u", rp->num_comp_u, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_error, "true_soln", rp->num_comp_u, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_error, "error", rp->num_comp_u, CEED_EVAL_NONE);

  // Create the error operator
  CeedOperatorCreate(ceed, qf_error, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_error);
  CeedOperatorSetField(op_error, "u", ceed_data->elem_restr_u, ceed_data->basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_error, "true_soln", ceed_data->elem_restr_u_i, CEED_BASIS_COLLOCATED, target);
  CeedOperatorSetField(op_error, "error", ceed_data->elem_restr_u_i, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Set up Mat
  user_O->comm  = rp->comm;
  user_O->dm    = dm;
  user_O->X_loc = X_loc;
  PetscCall(VecDuplicate(X_loc, &user_O->Y_loc));
  user_O->x_ceed = ceed_data->x_ceed;
  user_O->y_ceed = ceed_data->y_ceed;
  user_O->op     = ceed_data->op_apply;
  user_O->ceed   = ceed;

  PetscCall(KSPCreate(rp->comm, &ksp));
  {
    PC pc;
    PetscCall(KSPGetPC(ksp, &pc));
    if (rp->bp_choice == CEED_BP1 || rp->bp_choice == CEED_BP2) {
      PetscCall(PCSetType(pc, PCJACOBI));
      PetscCall(PCJacobiSetType(pc, PC_JACOBI_ROWSUM));
    } else {
      PetscCall(PCSetType(pc, PCNONE));
    }
    PetscCall(KSPSetType(ksp, KSPCG));
    PetscCall(KSPSetNormType(ksp, KSP_NORM_NATURAL));
    PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  }
  PetscCall(KSPSetOperators(ksp, mat_O, mat_O));

  // First run's performance log is not considered for benchmarking purposes
  PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 1));
  my_rt_start = MPI_Wtime();
  PetscCall(KSPSolve(ksp, rhs, X));
  my_rt = MPI_Wtime() - my_rt_start;
  PetscCall(MPI_Allreduce(MPI_IN_PLACE, &my_rt, 1, MPI_DOUBLE, MPI_MIN, rp->comm));
  // Set maxits based on first iteration timing
  if (my_rt > 0.02) {
    PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, rp->ksp_max_it_clip[0]));
  } else {
    PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, rp->ksp_max_it_clip[1]));
  }
  PetscCall(KSPSetFromOptions(ksp));

  // Timed solve
  PetscCall(VecZeroEntries(X));
  PetscCall(PetscBarrier((PetscObject)ksp));

  // -- Performance logging
  PetscCall(PetscLogStagePush(rp->solve_stage));

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
    if (!rp->test_mode || reason < 0 || rnorm > 1e-8) {
      PetscCall(PetscPrintf(rp->comm,
                            "  KSP:\n"
                            "    KSP Type                           : %s\n"
                            "    KSP Convergence                    : %s\n"
                            "    Total KSP Iterations               : %" PetscInt_FMT "\n"
                            "    Final rnorm                        : %e\n",
                            ksp_type, KSPConvergedReasons[reason], its, (double)rnorm));
    }
    if (!rp->test_mode) {
      PetscCall(PetscPrintf(rp->comm, "  Performance:\n"));
    }
    {
      PetscReal max_error;
      PetscCall(ComputeErrorMax(user_O, op_error, X, target, &max_error));
      PetscReal tol = 5e-2;
      if (!rp->test_mode || max_error > tol) {
        PetscCall(MPI_Allreduce(&my_rt, &rt_min, 1, MPI_DOUBLE, MPI_MIN, rp->comm));
        PetscCall(MPI_Allreduce(&my_rt, &rt_max, 1, MPI_DOUBLE, MPI_MAX, rp->comm));
        PetscCall(PetscPrintf(rp->comm,
                              "    Pointwise Error (max)              : %e\n"
                              "    CG Solve Time                      : %g (%g) sec\n",
                              (double)max_error, rt_max, rt_min));
      }
    }
    if (!rp->test_mode) {
      PetscCall(PetscPrintf(rp->comm, "    DoFs/Sec in CG                     : %g (%g) million\n", 1e-6 * g_size * its / rt_max,
                            1e-6 * g_size * its / rt_min));
    }
  }

  if (rp->write_solution) {
    PetscViewer vtk_viewer_soln;

    PetscCall(PetscViewerCreate(rp->comm, &vtk_viewer_soln));
    PetscCall(PetscViewerSetType(vtk_viewer_soln, PETSCVIEWERVTK));
    PetscCall(PetscViewerFileSetName(vtk_viewer_soln, "solution.vtu"));
    PetscCall(VecView(X, vtk_viewer_soln));
    PetscCall(PetscViewerDestroy(&vtk_viewer_soln));
  }

  // Cleanup
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&X_loc));
  PetscCall(VecDestroy(&user_O->Y_loc));
  PetscCall(MatDestroy(&mat_O));
  PetscCall(PetscFree(user_O));
  PetscCall(CeedDataDestroy(0, ceed_data));

  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&rhs_loc));
  PetscCall(KSPDestroy(&ksp));
  CeedVectorDestroy(&target);
  CeedQFunctionDestroy(&qf_error);
  CeedOperatorDestroy(&op_error);
  CeedDestroy(&ceed);
  PetscFunctionReturn(0);
}

static PetscErrorCode Run(RunParams rp, PetscInt num_resources, char *const *ceed_resources, PetscInt num_bp_choices, const BPType *bp_choices) {
  DM dm;

  PetscFunctionBeginUser;
  // Setup DM
  if (rp->read_mesh) {
    PetscCall(DMPlexCreateFromFile(PETSC_COMM_WORLD, rp->filename, NULL, PETSC_TRUE, &dm));
  } else {
    if (rp->user_l_nodes) {
      // Find a nicely composite number of elements no less than global nodes
      PetscMPIInt size;
      PetscCall(MPI_Comm_size(rp->comm, &size));
      for (PetscInt g_elem = PetscMax(1, size * rp->local_nodes / PetscPowInt(rp->degree, rp->dim));; g_elem++) {
        Split3(g_elem, rp->mesh_elem, true);
        if (Max3(rp->mesh_elem) / Min3(rp->mesh_elem) <= 2) break;
      }
    }
    PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, rp->dim, PETSC_FALSE, rp->mesh_elem, NULL, NULL, NULL, PETSC_TRUE, &dm));
  }

  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  for (PetscInt b = 0; b < num_bp_choices; b++) {
    DM       dm_deg;
    VecType  vec_type;
    PetscInt q_extra = rp->q_extra;
    rp->bp_choice    = bp_choices[b];
    rp->num_comp_u   = bp_options[rp->bp_choice].num_comp_u;
    rp->q_extra      = q_extra < 0 ? bp_options[rp->bp_choice].q_extra : q_extra;
    PetscCall(DMClone(dm, &dm_deg));
    PetscCall(DMGetVecType(dm, &vec_type));
    PetscCall(DMSetVecType(dm_deg, vec_type));
    // Create DM
    PetscInt dim;
    PetscCall(DMGetDimension(dm_deg, &dim));
    PetscCall(SetupDMByDegree(dm_deg, rp->degree, rp->num_comp_u, dim, bp_options[rp->bp_choice].enforce_bc, bp_options[rp->bp_choice].bc_func));
    for (PetscInt r = 0; r < num_resources; r++) {
      PetscCall(RunWithDM(rp, dm_deg, ceed_resources[r]));
    }
    PetscCall(DMDestroy(&dm_deg));
    rp->q_extra = q_extra;
  }

  PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  PetscInt    comm_size;
  RunParams   rp;
  MPI_Comm    comm;
  char        filename[PETSC_MAX_PATH_LEN];
  char       *ceed_resources[30];
  PetscInt    num_ceed_resources = 30;
  char        hostname[PETSC_MAX_PATH_LEN];

  PetscInt    dim = 3, mesh_elem[3] = {3, 3, 3};
  PetscInt    num_degrees = 30, degree[30] = {}, num_local_nodes = 2, local_nodes[2] = {};
  PetscMPIInt ranks_per_node;
  PetscBool   degree_set;
  BPType      bp_choices[10];
  PetscInt    num_bp_choices = 10;

  // Initialize PETSc
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(MPI_Comm_size(comm, &comm_size));
#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
  {
    MPI_Comm splitcomm;
    PetscCall(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &splitcomm));
    PetscCall(MPI_Comm_size(splitcomm, &ranks_per_node));
    PetscCall(MPI_Comm_free(&splitcomm));
  }
#else
  ranks_per_node = -1;  // Unknown
#endif

  // Setup all parameters needed in Run()
  PetscCall(PetscMalloc1(1, &rp));
  rp->comm = comm;

  // Read command line options
  PetscOptionsBegin(comm, NULL, "CEED BPs in PETSc", NULL);
  {
    PetscBool set;
    PetscCall(PetscOptionsEnumArray("-problem", "CEED benchmark problem to solve", NULL, bp_types, (PetscEnum *)bp_choices, &num_bp_choices, &set));
    if (!set) {
      bp_choices[0]  = CEED_BP1;
      num_bp_choices = 1;
    }
  }
  rp->test_mode = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-test", "Testing mode (do not print unless error is large)", NULL, rp->test_mode, &rp->test_mode, NULL));
  rp->write_solution = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-write_solution", "Write solution for visualization", NULL, rp->write_solution, &rp->write_solution, NULL));
  degree[0] = rp->test_mode ? 3 : 2;
  PetscCall(PetscOptionsIntArray("-degree", "Polynomial degree of tensor product basis", NULL, degree, &num_degrees, &degree_set));
  if (!degree_set) num_degrees = 1;
  rp->q_extra = PETSC_DECIDE;
  PetscCall(PetscOptionsInt("-q_extra", "Number of extra quadrature points (-1 for auto)", NULL, rp->q_extra, &rp->q_extra, NULL));
  {
    PetscBool set;
    PetscCall(PetscOptionsStringArray("-ceed", "CEED resource specifier (comma-separated list)", NULL, ceed_resources, &num_ceed_resources, &set));
    if (!set) {
      PetscCall(PetscStrallocpy("/cpu/self", &ceed_resources[0]));
      num_ceed_resources = 1;
    }
  }
  PetscCall(PetscGetHostName(hostname, sizeof hostname));
  PetscCall(PetscOptionsString("-hostname", "Hostname for output", NULL, hostname, hostname, sizeof(hostname), NULL));
  rp->read_mesh = PETSC_FALSE;
  PetscCall(PetscOptionsString("-mesh", "Read mesh from file", NULL, filename, filename, sizeof(filename), &rp->read_mesh));
  rp->filename = filename;
  if (!rp->read_mesh) {
    PetscInt tmp = dim;
    PetscCall(PetscOptionsIntArray("-cells", "Number of cells per dimension", NULL, mesh_elem, &tmp, NULL));
  }
  local_nodes[0] = 1000;
  PetscCall(PetscOptionsIntArray("-local_nodes",
                                 "Target number of locally owned nodes per "
                                 "process (single value or min,max)",
                                 NULL, local_nodes, &num_local_nodes, &rp->user_l_nodes));
  if (num_local_nodes < 2) local_nodes[1] = 2 * local_nodes[0];
  {
    PetscInt two           = 2;
    rp->ksp_max_it_clip[0] = 5;
    rp->ksp_max_it_clip[1] = 20;
    PetscCall(PetscOptionsIntArray("-ksp_max_it_clip", "Min and max number of iterations to use during benchmarking", NULL, rp->ksp_max_it_clip, &two,
                                   NULL));
  }
  if (!degree_set) {
    PetscInt max_degree = 8;
    PetscCall(PetscOptionsInt("-max_degree", "Range of degrees [1, max_degree] to run with", NULL, max_degree, &max_degree, NULL));
    for (PetscInt i = 0; i < max_degree; i++) degree[i] = i + 1;
    num_degrees = max_degree;
  }
  {
    PetscBool flg;
    PetscInt  p = ranks_per_node;
    PetscCall(PetscOptionsInt("-p", "Number of MPI ranks per node", NULL, p, &p, &flg));
    if (flg) ranks_per_node = p;
  }

  PetscOptionsEnd();

  // Register PETSc logging stage
  PetscCall(PetscLogStageRegister("Solve Stage", &rp->solve_stage));

  rp->hostname       = hostname;
  rp->dim            = dim;
  rp->mesh_elem      = mesh_elem;
  rp->ranks_per_node = ranks_per_node;

  for (PetscInt d = 0; d < num_degrees; d++) {
    PetscInt deg = degree[d];
    for (PetscInt n = local_nodes[0]; n < local_nodes[1]; n *= 2) {
      rp->degree      = deg;
      rp->local_nodes = n;
      PetscCall(Run(rp, num_ceed_resources, ceed_resources, num_bp_choices, bp_choices));
    }
  }
  // Clear memory
  PetscCall(PetscFree(rp));
  for (PetscInt i = 0; i < num_ceed_resources; i++) {
    PetscCall(PetscFree(ceed_resources[i]));
  }
  return PetscFinalize();
}
