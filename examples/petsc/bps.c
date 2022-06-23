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

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

// Utility function, compute three factors of an integer
static void Split3(PetscInt size, PetscInt m[3], bool reverse) {
  for (PetscInt d=0, size_left=size; d<3; d++) {
    PetscInt try = (PetscInt)PetscCeilReal(PetscPowReal(size_left, 1./(3 - d)));
    while (try * (size_left / try) != size_left) try++;
    m[reverse ? 2-d : d] = try;
    size_left /= try;
  }
}

static int Max3(const PetscInt a[3]) {
  return PetscMax(a[0], PetscMax(a[1], a[2]));
}

static int Min3(const PetscInt a[3]) {
  return PetscMin(a[0], PetscMin(a[1], a[2]));
}

// -----------------------------------------------------------------------------
// Parameter structure for running problems
// -----------------------------------------------------------------------------
typedef struct RunParams_ *RunParams;
struct RunParams_ {
  MPI_Comm comm;
  PetscBool test_mode, read_mesh, user_l_nodes, write_solution;
  char *filename, *hostname;
  PetscInt local_nodes, degree, q_extra, dim, num_comp_u, *mesh_elem;
  PetscInt ksp_max_it_clip[2];
  PetscMPIInt ranks_per_node;
  BPType bp_choice;
  PetscLogStage solve_stage;
};

// -----------------------------------------------------------------------------
// Main body of program, called in a loop for performance benchmarking purposes
// -----------------------------------------------------------------------------
static PetscErrorCode RunWithDM(RunParams rp, DM dm,
                                const char *ceed_resource) {
  PetscErrorCode ierr;
  double my_rt_start, my_rt, rt_min, rt_max;
  PetscInt xl_size, l_size, g_size;
  PetscScalar *r;
  Vec X, X_loc, rhs, rhs_loc;
  Mat mat_O;
  KSP ksp;
  UserO user_O;
  Ceed ceed;
  CeedData ceed_data;
  CeedQFunction qf_error;
  CeedOperator op_error;
  CeedVector rhs_ceed, target;
  VecType vec_type;
  PetscMemType mem_type;

  PetscFunctionBeginUser;
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

  // Create global and local solution vectors
  ierr = DMCreateGlobalVector(dm, &X); CHKERRQ(ierr);
  ierr = VecGetLocalSize(X, &l_size); CHKERRQ(ierr);
  ierr = VecGetSize(X, &g_size); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm, &X_loc); CHKERRQ(ierr);
  ierr = VecGetSize(X_loc, &xl_size); CHKERRQ(ierr);
  ierr = VecDuplicate(X, &rhs); CHKERRQ(ierr);

  // Operator
  ierr = PetscMalloc1(1, &user_O); CHKERRQ(ierr);
  ierr = MatCreateShell(rp->comm, l_size, l_size, g_size, g_size,
                        user_O, &mat_O); CHKERRQ(ierr);
  ierr = MatShellSetOperation(mat_O, MATOP_MULT,
                              (void(*)(void))MatMult_Ceed); CHKERRQ(ierr);
  ierr = MatShellSetOperation(mat_O, MATOP_GET_DIAGONAL,
                              (void(*)(void))MatGetDiag); CHKERRQ(ierr);
  ierr = MatShellSetVecType(mat_O, vec_type); CHKERRQ(ierr);

  // Print summary
  if (!rp->test_mode) {
    PetscInt P = rp->degree + 1, Q = P + rp->q_extra;

    const char *used_resource;
    CeedGetResource(ceed, &used_resource);

    VecType vec_type;
    ierr = VecGetType(X, &vec_type); CHKERRQ(ierr);

    PetscInt c_start, c_end;
    ierr = DMPlexGetHeightStratum(dm, 0, &c_start, &c_end); CHKERRQ(ierr);
    PetscMPIInt comm_size;
    ierr = MPI_Comm_size(rp->comm, &comm_size); CHKERRQ(ierr);
    ierr = PetscPrintf(rp->comm,
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
                       rp->bp_choice+1, rp->hostname, comm_size,
                       rp->ranks_per_node, vec_type, used_resource,
                       CeedMemTypes[mem_type_backend],
                       P, Q, g_size/rp->num_comp_u, c_end - c_start, l_size/rp->num_comp_u,
                       rp->num_comp_u);
    CHKERRQ(ierr);
  }

  // Create RHS vector
  ierr = VecDuplicate(X_loc, &rhs_loc); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhs_loc); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(rhs_loc, &r, &mem_type); CHKERRQ(ierr);
  CeedVectorCreate(ceed, xl_size, &rhs_ceed);
  CeedVectorSetArray(rhs_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, r);

  ierr = PetscMalloc1(1, &ceed_data); CHKERRQ(ierr);
  ierr = SetupLibceedByDegree(dm, ceed, rp->degree, rp->dim, rp->q_extra,
                              rp->dim, rp->num_comp_u, g_size, xl_size, bp_options[rp->bp_choice],
                              ceed_data, true, rhs_ceed, &target); CHKERRQ(ierr);

  // Gather RHS
  CeedVectorTakeArray(rhs_ceed, MemTypeP2C(mem_type), NULL);
  ierr = VecRestoreArrayAndMemType(rhs_loc, &r); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhs); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dm, rhs_loc, ADD_VALUES, rhs); CHKERRQ(ierr);
  CeedVectorDestroy(&rhs_ceed);

  // Create the error QFunction
  CeedQFunctionCreateInterior(ceed, 1, bp_options[rp->bp_choice].error,
                              bp_options[rp->bp_choice].error_loc, &qf_error);
  CeedQFunctionAddInput(qf_error, "u", rp->num_comp_u, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_error, "true_soln", rp->num_comp_u, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_error, "error", rp->num_comp_u, CEED_EVAL_NONE);

  // Create the error operator
  CeedOperatorCreate(ceed, qf_error, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_error);
  CeedOperatorSetField(op_error, "u", ceed_data->elem_restr_u,
                       ceed_data->basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_error, "true_soln", ceed_data->elem_restr_u_i,
                       CEED_BASIS_COLLOCATED, target);
  CeedOperatorSetField(op_error, "error", ceed_data->elem_restr_u_i,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Set up Mat
  user_O->comm = rp->comm;
  user_O->dm = dm;
  user_O->X_loc = X_loc;
  ierr = VecDuplicate(X_loc, &user_O->Y_loc); CHKERRQ(ierr);
  user_O->x_ceed = ceed_data->x_ceed;
  user_O->y_ceed = ceed_data->y_ceed;
  user_O->op = ceed_data->op_apply;
  user_O->ceed = ceed;

  ierr = KSPCreate(rp->comm, &ksp); CHKERRQ(ierr);
  {
    PC pc;
    ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
    if (rp->bp_choice == CEED_BP1 || rp->bp_choice == CEED_BP2) {
      ierr = PCSetType(pc, PCJACOBI); CHKERRQ(ierr);
      ierr = PCJacobiSetType(pc, PC_JACOBI_ROWSUM); CHKERRQ(ierr);
    } else {
      ierr = PCSetType(pc, PCNONE); CHKERRQ(ierr);
    }
    ierr = KSPSetType(ksp, KSPCG); CHKERRQ(ierr);
    ierr = KSPSetNormType(ksp, KSP_NORM_NATURAL); CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT,
                            PETSC_DEFAULT); CHKERRQ(ierr);
  }
  ierr = KSPSetOperators(ksp, mat_O, mat_O); CHKERRQ(ierr);

  // First run's performance log is not considered for benchmarking purposes
  ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 1);
  CHKERRQ(ierr);
  my_rt_start = MPI_Wtime();
  ierr = KSPSolve(ksp, rhs, X); CHKERRQ(ierr);
  my_rt = MPI_Wtime() - my_rt_start;
  ierr = MPI_Allreduce(MPI_IN_PLACE, &my_rt, 1, MPI_DOUBLE, MPI_MIN, rp->comm);
  CHKERRQ(ierr);
  // Set maxits based on first iteration timing
  if (my_rt > 0.02) {
    ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT,
                            rp->ksp_max_it_clip[0]);
    CHKERRQ(ierr);
  } else {
    ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT,
                            rp->ksp_max_it_clip[1]);
    CHKERRQ(ierr);
  }
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

  // Timed solve
  ierr = VecZeroEntries(X); CHKERRQ(ierr);
  ierr = PetscBarrier((PetscObject)ksp); CHKERRQ(ierr);

  // -- Performance logging
  ierr = PetscLogStagePush(rp->solve_stage); CHKERRQ(ierr);

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
    if (!rp->test_mode || reason < 0 || rnorm > 1e-8) {
      ierr = PetscPrintf(rp->comm,
                         "  KSP:\n"
                         "    KSP Type                           : %s\n"
                         "    KSP Convergence                    : %s\n"
                         "    Total KSP Iterations               : %" PetscInt_FMT "\n"
                         "    Final rnorm                        : %e\n",
                         ksp_type, KSPConvergedReasons[reason], its,
                         (double)rnorm); CHKERRQ(ierr);
    }
    if (!rp->test_mode) {
      ierr = PetscPrintf(rp->comm,"  Performance:\n"); CHKERRQ(ierr);
    }
    {
      PetscReal max_error;
      ierr = ComputeErrorMax(user_O, op_error, X, target, &max_error);
      CHKERRQ(ierr);
      PetscReal tol = 5e-2;
      if (!rp->test_mode || max_error > tol) {
        ierr = MPI_Allreduce(&my_rt, &rt_min, 1, MPI_DOUBLE, MPI_MIN, rp->comm);
        CHKERRQ(ierr);
        ierr = MPI_Allreduce(&my_rt, &rt_max, 1, MPI_DOUBLE, MPI_MAX, rp->comm);
        CHKERRQ(ierr);
        ierr = PetscPrintf(rp->comm,
                           "    Pointwise Error (max)              : %e\n"
                           "    CG Solve Time                      : %g (%g) sec\n",
                           (double)max_error, rt_max, rt_min); CHKERRQ(ierr);
      }
    }
    if (!rp->test_mode) {
      ierr = PetscPrintf(rp->comm,
                         "    DoFs/Sec in CG                     : %g (%g) million\n",
                         1e-6*g_size*its/rt_max,
                         1e-6*g_size*its/rt_min); CHKERRQ(ierr);
    }
  }

  if (rp->write_solution) {
    PetscViewer vtk_viewer_soln;

    ierr = PetscViewerCreate(rp->comm, &vtk_viewer_soln); CHKERRQ(ierr);
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

  ierr = VecDestroy(&rhs); CHKERRQ(ierr);
  ierr = VecDestroy(&rhs_loc); CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
  CeedVectorDestroy(&target);
  CeedQFunctionDestroy(&qf_error);
  CeedOperatorDestroy(&op_error);
  CeedDestroy(&ceed);
  PetscFunctionReturn(0);
}

static PetscErrorCode Run(RunParams rp, PetscInt num_resources,
                          char *const *ceed_resources, PetscInt num_bp_choices,
                          const BPType *bp_choices) {
  PetscInt ierr;
  DM dm;

  PetscFunctionBeginUser;
  // Setup DM
  if (rp->read_mesh) {
    ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, rp->filename, NULL, PETSC_TRUE,
                                &dm);
    CHKERRQ(ierr);
  } else {
    if (rp->user_l_nodes) {
      // Find a nicely composite number of elements no less than global nodes
      PetscMPIInt size;
      ierr = MPI_Comm_size(rp->comm, &size); CHKERRQ(ierr);
      for (PetscInt g_elem =
             PetscMax(1, size * rp->local_nodes / PetscPowInt(rp->degree, rp->dim));
           ;
           g_elem++) {
        Split3(g_elem, rp->mesh_elem, true);
        if (Max3(rp->mesh_elem) / Min3(rp->mesh_elem) <= 2) break;
      }
    }
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, rp->dim, PETSC_FALSE,
                               rp->mesh_elem,
                               NULL, NULL, NULL, PETSC_TRUE, &dm); CHKERRQ(ierr);
  }

  ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view"); CHKERRQ(ierr);

  for (PetscInt b = 0; b < num_bp_choices; b++) {
    DM dm_deg;
    VecType vec_type;
    PetscInt q_extra = rp->q_extra;
    rp->bp_choice = bp_choices[b];
    rp->num_comp_u = bp_options[rp->bp_choice].num_comp_u;
    rp->q_extra = q_extra < 0 ? bp_options[rp->bp_choice].q_extra : q_extra;
    ierr = DMClone(dm, &dm_deg); CHKERRQ(ierr);
    ierr = DMGetVecType(dm, &vec_type); CHKERRQ(ierr);
    ierr = DMSetVecType(dm_deg, vec_type); CHKERRQ(ierr);
    // Create DM
    PetscInt dim;
    ierr = DMGetDimension(dm_deg, &dim); CHKERRQ(ierr);
    ierr = SetupDMByDegree(dm_deg, rp->degree, rp->num_comp_u, dim,
                           bp_options[rp->bp_choice].enforce_bc,
                           bp_options[rp->bp_choice].bc_func); CHKERRQ(ierr);
    for (PetscInt r = 0; r < num_resources; r++) {
      ierr = RunWithDM(rp, dm_deg, ceed_resources[r]); CHKERRQ(ierr);
    }
    ierr = DMDestroy(&dm_deg); CHKERRQ(ierr);
    rp->q_extra = q_extra;
  }

  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  PetscInt ierr, comm_size;
  RunParams rp;
  MPI_Comm comm;
  char filename[PETSC_MAX_PATH_LEN];
  char *ceed_resources[30];
  PetscInt num_ceed_resources = 30;
  char hostname[PETSC_MAX_PATH_LEN];

  PetscInt dim = 3, mesh_elem[3] = {3, 3, 3};
  PetscInt num_degrees = 30, degree[30] = {}, num_local_nodes = 2,
                                          local_nodes[2] = {};
  PetscMPIInt ranks_per_node;
  PetscBool degree_set;
  BPType bp_choices[10];
  PetscInt num_bp_choices = 10;

  // Initialize PETSc
  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_size(comm, &comm_size);
  if (ierr != MPI_SUCCESS) return ierr;
  #if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
  {
    MPI_Comm splitcomm;
    ierr = MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                               &splitcomm);
    CHKERRQ(ierr);
    ierr = MPI_Comm_size(splitcomm, &ranks_per_node); CHKERRQ(ierr);
    ierr = MPI_Comm_free(&splitcomm); CHKERRQ(ierr);
  }
  #else
  ranks_per_node = -1; // Unknown
  #endif

  // Setup all parameters needed in Run()
  ierr = PetscMalloc1(1, &rp); CHKERRQ(ierr);
  rp->comm = comm;

  // Read command line options
  PetscOptionsBegin(comm, NULL, "CEED BPs in PETSc", NULL);
  {
    PetscBool set;
    ierr = PetscOptionsEnumArray("-problem", "CEED benchmark problem to solve",
                                 NULL,
                                 bp_types, (PetscEnum *)bp_choices, &num_bp_choices, &set);
    CHKERRQ(ierr);
    if (!set) {
      bp_choices[0] = CEED_BP1;
      num_bp_choices = 1;
    }
  }
  rp->test_mode = PETSC_FALSE;
  ierr = PetscOptionsBool("-test",
                          "Testing mode (do not print unless error is large)",
                          NULL, rp->test_mode, &rp->test_mode, NULL); CHKERRQ(ierr);
  rp->write_solution = PETSC_FALSE;
  ierr = PetscOptionsBool("-write_solution", "Write solution for visualization",
                          NULL, rp->write_solution, &rp->write_solution, NULL);
  CHKERRQ(ierr);
  degree[0] = rp->test_mode ? 3 : 2;
  ierr = PetscOptionsIntArray("-degree",
                              "Polynomial degree of tensor product basis", NULL,
                              degree, &num_degrees, &degree_set); CHKERRQ(ierr);
  if (!degree_set)
    num_degrees = 1;
  rp->q_extra = PETSC_DECIDE;
  ierr = PetscOptionsInt("-q_extra",
                         "Number of extra quadrature points (-1 for auto)", NULL,
                         rp->q_extra, &rp->q_extra, NULL); CHKERRQ(ierr);
  {
    PetscBool set;
    ierr = PetscOptionsStringArray("-ceed",
                                   "CEED resource specifier (comma-separated list)", NULL,
                                   ceed_resources, &num_ceed_resources, &set); CHKERRQ(ierr);
    if (!set) {
      ierr = PetscStrallocpy( "/cpu/self", &ceed_resources[0]); CHKERRQ(ierr);
      num_ceed_resources = 1;
    }
  }
  ierr = PetscGetHostName(hostname, sizeof hostname); CHKERRQ(ierr);
  ierr = PetscOptionsString("-hostname", "Hostname for output", NULL, hostname,
                            hostname, sizeof(hostname), NULL); CHKERRQ(ierr);
  rp->read_mesh = PETSC_FALSE;
  ierr = PetscOptionsString("-mesh", "Read mesh from file", NULL, filename,
                            filename, sizeof(filename), &rp->read_mesh);
  CHKERRQ(ierr);
  rp->filename = filename;
  if (!rp->read_mesh) {
    PetscInt tmp = dim;
    ierr = PetscOptionsIntArray("-cells", "Number of cells per dimension", NULL,
                                mesh_elem, &tmp, NULL); CHKERRQ(ierr);
  }
  local_nodes[0] = 1000;
  ierr = PetscOptionsIntArray("-local_nodes",
                              "Target number of locally owned nodes per "
                              "process (single value or min,max)",
                              NULL, local_nodes, &num_local_nodes, &rp->user_l_nodes);
  CHKERRQ(ierr);
  if (num_local_nodes < 2)
    local_nodes[1] = 2 * local_nodes[0];
  {
    PetscInt two = 2;
    rp->ksp_max_it_clip[0] = 5;
    rp->ksp_max_it_clip[1] = 20;
    ierr = PetscOptionsIntArray("-ksp_max_it_clip",
                                "Min and max number of iterations to use during benchmarking",
                                NULL, rp->ksp_max_it_clip, &two, NULL); CHKERRQ(ierr);
  }
  if (!degree_set) {
    PetscInt max_degree = 8;
    ierr = PetscOptionsInt("-max_degree",
                           "Range of degrees [1, max_degree] to run with",
                           NULL, max_degree, &max_degree, NULL);
    CHKERRQ(ierr);
    for (PetscInt i = 0; i < max_degree; i++)
      degree[i] = i + 1;
    num_degrees = max_degree;
  }
  {
    PetscBool flg;
    PetscInt p = ranks_per_node;
    ierr = PetscOptionsInt("-p", "Number of MPI ranks per node", NULL,
                           p, &p, &flg);
    CHKERRQ(ierr);
    if (flg) ranks_per_node = p;
  }

  PetscOptionsEnd();

  // Register PETSc logging stage
  ierr = PetscLogStageRegister("Solve Stage", &rp->solve_stage);
  CHKERRQ(ierr);

  rp->hostname = hostname;
  rp->dim = dim;
  rp->mesh_elem = mesh_elem;
  rp->ranks_per_node = ranks_per_node;

  for (PetscInt d = 0; d < num_degrees; d++) {
    PetscInt deg = degree[d];
    for (PetscInt n = local_nodes[0]; n < local_nodes[1]; n *= 2) {
      rp->degree = deg;
      rp->local_nodes = n;
      ierr = Run(rp, num_ceed_resources, ceed_resources,
                 num_bp_choices, bp_choices); CHKERRQ(ierr);
    }
  }
  // Clear memory
  ierr = PetscFree(rp); CHKERRQ(ierr);
  for (PetscInt i=0; i<num_ceed_resources; i++) {
    ierr = PetscFree(ceed_resources[i]); CHKERRQ(ierr);
  }
  return PetscFinalize();
}
