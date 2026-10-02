// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

//                        libCEED + PETSc Example: CEED BPs
//
// This example demonstrates a simple usage of libCEED with PETSc to solve the CEED BP benchmark problems, see http://ceed.exascaleproject.org/bps, on
// a particle swarm.
//
// The code uses higher level communication protocols in DMPlex and DMSwarm.
//
// Build with:
//
//     make bpsswarm [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     bpsswarm -problem bp1 -degree 3 -points_per_cell_1d 125 -swarm uniform
//     bpsswarm -problem bp2 -degree 3 -points_per_cell_1d 125 -swarm uniform
//     bpsswarm -problem bp3 -degree 3 -points_per_cell_1d 125 -swarm uniform
//
//TESTARGS(name="BP2") -ceed {ceed_resource} -test -degree 2 -problem bp2 -cells 6,6,6 -swarm uniform -points_per_cell_1d 4
//TESTARGS(name="BP3") -ceed {ceed_resource} -test -degree 2 -problem bp3 -cells 4,4,4 -swarm uniform -points_per_cell_1d 4
//TESTARGS(name="BP5") -ceed {ceed_resource} -test -degree 2 -problem bp5 -cells 5,5,5 -swarm gauss -points_per_cell_1d 4

/// @file
/// CEED BP-like example using PETSc with DMPlex & DMSwarm for arbitrarily placed quadrature points
/// See bpsraw.c for a "raw" implementation using a structured grid and bps.c for an implementation using an unstructured grid.
static const char help[]              = "Solve CEED BP-like problems on a particle swarm using DMPlex and DMSwarm in PETSc\n";
const char        DMSwarmPICField_u[] = "u";

#include "bps.h"

#include <ceed.h>
#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscksp.h>
#include <stdbool.h>
#include <string.h>

#include "include/bpsproblemdata.h"
#include "include/libceedsetup.h"
#include "include/matops.h"
#include "include/petscutils.h"
#include "include/petscversion.h"
#include "include/swarmutils.h"

// -----------------------------------------------------------------------------
// Main body of program, called in a loop for performance benchmarking purposes
// -----------------------------------------------------------------------------
static PetscErrorCode RunWithDM(RunParams rp, DM dm_swarm, const char *ceed_resource) {
  double               my_rt_start, my_rt, rt_min, rt_max;
  PetscInt             xl_size, l_size, g_size;
  Vec                  X, X_loc, target, rhs;
  Mat                  mat_O;
  KSP                  ksp;
  OperatorApplyContext op_apply_ctx;
  Ceed                 ceed;
  CeedData             ceed_data;
  VecType              vec_type = VECSTANDARD;
  PetscMemType         mem_type;
  DM                   dm_mesh;

  PetscFunctionBeginUser;
  // Set up libCEED
  CeedInit(ceed_resource, &ceed);
  CeedMemType mem_type_backend;
  CeedGetPreferredMemType(ceed, &mem_type_backend);
  PetscCall(DMSwarmGetCellDM(dm_swarm, &dm_mesh));

  // Set mesh vec_type
  switch (mem_type_backend) {
    case CEED_MEM_HOST:
      vec_type = VECSTANDARD;
      break;
    case CEED_MEM_DEVICE: {
      const char *resolved;

      CeedGetResource(ceed, &resolved);
      if (strstr(resolved, "/gpu/cuda"))
        vec_type = VECCUDA;
      else if (strstr(resolved, "/gpu/hip"))
        vec_type = VECHIP;
      else
        vec_type = VECSTANDARD;
    }
  }
  PetscCall(DMSetVecType(dm_mesh, vec_type));
  PetscCall(DMSetFromOptions(dm_mesh));

  // DMSetFromOptions() may redistribute the mesh and replace the sections.
  PetscBool is_simplex = PETSC_TRUE;
  PetscCall(DMPlexIsSimplex(dm_mesh, &is_simplex));

  // Restore tensor closure permutations for the solution and coordinate DMs.
  if (!is_simplex) {
    DM dm_coord;
    PetscCall(DMGetCoordinateDM(dm_mesh, &dm_coord));
    PetscCall(DMPlexSetClosurePermutationTensor(dm_mesh, PETSC_DETERMINE, NULL));
    PetscCall(DMPlexSetClosurePermutationTensor(dm_coord, PETSC_DETERMINE, NULL));
  }

  // Create global and local solution vectors
  PetscCall(DMCreateGlobalVector(dm_mesh, &X));
  PetscCall(VecGetLocalSize(X, &l_size));
  PetscCall(VecGetSize(X, &g_size));
  PetscCall(DMCreateLocalVector(dm_mesh, &X_loc));
  PetscCall(VecGetSize(X_loc, &xl_size));
  PetscCall(VecDuplicate(X, &rhs));

  // Operator
  PetscCall(PetscMalloc1(1, &op_apply_ctx));
  PetscCall(MatCreateShell(rp->comm, l_size, l_size, g_size, g_size, op_apply_ctx, &mat_O));
  PetscCall(MatShellSetOperation(mat_O, MATOP_MULT, (void (*)(void))MatMult_Ceed));
  PetscCall(MatShellSetOperation(mat_O, MATOP_GET_DIAGONAL, (void (*)(void))MatGetDiag));
  PetscCall(MatShellSetVecType(mat_O, vec_type));

  // Print summary
  if (!rp->test_mode) {
    PetscInt P = rp->degree + 1, Q = P;
    PetscInt num_points_global, num_points_min_max[2], num_cells_global;

    const char *used_resource;
    CeedGetResource(ceed, &used_resource);

    bool is_combined_bp = rp->bp_choice > CEED_BP6;
    char bp_name[6]     = "";

    if (is_combined_bp) {
      PetscCall(PetscSNPrintf(bp_name, 6, "%d + %d", rp->bp_choice % 2 ? 2 : 1, rp->bp_choice - CEED_BP4));
    } else {
      PetscCall(PetscSNPrintf(bp_name, 6, "%d", rp->bp_choice + 1));
    }

    VecType vec_type;
    PetscCall(VecGetType(X, &vec_type));

    PetscInt c_start, c_end;
    PetscCall(DMPlexGetHeightStratum(dm_mesh, 0, &c_start, &c_end));
    DMPolytopeType cell_type;
    PetscCall(DMPlexGetCellType(dm_mesh, c_start, &cell_type));
    CeedElemTopology elem_topo = ElemTopologyP2C(cell_type);
    PetscMPIInt      comm_size;
    PetscCall(MPI_Comm_size(rp->comm, &comm_size));
    PetscCall(DMSwarmGetSize(dm_swarm, &num_points_global));
    {
      PetscInt num_points_local, num_points_min_max_local[2];
      PetscInt num_cells_local = c_end - c_start;

      PetscCall(DMSwarmGetLocalSize(dm_swarm, &num_points_local));
      num_points_min_max_local[0] = num_points_local;
      num_points_min_max_local[1] = -num_points_local;
      PetscCall(MPIU_Reduce(num_points_min_max_local, num_points_min_max, 2, MPIU_INT, MPIU_MIN, 0, rp->comm));
      num_points_min_max[1] *= -1;
      PetscCall(MPIU_Reduce(&num_cells_local, &num_cells_global, 1, MPIU_INT, MPIU_SUM, 0, rp->comm));
    }
    PetscCall(
        PetscPrintf(rp->comm,
                    "\n-- CEED Benchmark Problem Points %s -- libCEED + PETSc --\n"
                    "  MPI:\n"
                    "    Hostname                                : %s\n"
                    "    Total ranks                             : %d\n"
                    "    Ranks per compute node                  : %d\n"
                    "  PETSc:\n"
                    "    PETSc Vec Type                          : %s\n"
                    "  libCEED:\n"
                    "    libCEED Backend                         : %s\n"
                    "    libCEED Backend MemType                 : %s\n"
                    "  Mesh:\n"
                    "    Solution Order (P)                      : %" PetscInt_FMT "\n"
                    "    Quadrature Order (Q)                    : %" PetscInt_FMT "\n"
                    "    Additional points (q_extra)             : %" PetscInt_FMT "\n"
                    "    Global nodes                            : %" PetscInt_FMT "\n"
                    "    Local Elements                          : %" PetscInt_FMT "\n"
                    "    Element topology                        : %s\n"
                    "    Owned nodes                             : %" PetscInt_FMT "\n"
                    "    DoF per node                            : %" PetscInt_FMT "\n"
                    "  Swarm:\n"
                    "    Global points                           : %" PetscInt_FMT "\n"
                    "    Local points                            : %" PetscInt_FMT " (%" PetscInt_FMT ")\n"
                    "    Avg points per cell                     : %" PetscInt_FMT "\n"
                    "    Point distribution                      : %s\n",
                    // bp_choice + 1, hostname, comm_size, ranks_per_node, vec_type, used_resource, CeedMemTypes[mem_type_backend], P, Q, q_extra,
                    // g_size / num_comp_u, num_cells_local, l_size / num_comp_u, num_comp_u, num_points_global, num_points_local,
                    // num_cells_local > 0 ? num_points_local / num_cells_local : 0, point_swarm_types[point_swarm_type]));

                    bp_name, rp->hostname, comm_size, rp->ranks_per_node, vec_type, used_resource, CeedMemTypes[mem_type_backend], P, Q, rp->q_extra,
                    g_size / rp->num_comp_u, c_end - c_start, CeedElemTopologies[elem_topo], l_size / rp->num_comp_u, rp->num_comp_u,
                    num_points_global, num_points_min_max[0], num_points_min_max[1], num_points_global / num_cells_global,
                    PointSwarmTypes[rp->swarm_type]));
  }

  PetscCall(DMCreateLocalVector(dm_swarm, &target));
  PetscCall(PetscMalloc1(1, &ceed_data));
  PetscCall(SetupProblemSwarm(dm_swarm, ceed, bp_options[rp->bp_choice], ceed_data, true, rhs, target));

  // Set up apply operator context
  PetscCall(SetupApplyOperatorCtx(rp->comm, dm_mesh, ceed, ceed_data, X_loc, op_apply_ctx));
  PetscCall(KSPCreate(rp->comm, &ksp));
  {
    PC pc;
    PetscCall(KSPGetPC(ksp, &pc));
    if (rp->bp_choice == CEED_BP1 || rp->bp_choice == CEED_BP2 || rp->bp_choice == CEED_BP13 || rp->bp_choice == CEED_BP24 ||
        rp->bp_choice == CEED_BP15 || rp->bp_choice == CEED_BP26) {
      PetscCall(PCSetType(pc, PCJACOBI));
      if (rp->simplex || rp->bp_choice == CEED_BP13 || rp->bp_choice == CEED_BP24 || rp->bp_choice == CEED_BP15 || rp->bp_choice == CEED_BP26) {
        PetscCall(PCJacobiSetType(pc, PC_JACOBI_DIAGONAL));
      } else {
        PetscCall(PCJacobiSetType(pc, PC_JACOBI_ROWSUM));
      }
    } else {
      PetscCall(PCSetType(pc, PCNONE));
    }
    PetscCall(KSPSetType(ksp, KSPCG));
    PetscCall(KSPSetNormType(ksp, KSP_NORM_NATURAL));
    PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  }
  PetscCall(KSPSetOperators(ksp, mat_O, mat_O));

  // First run's performance log is not considered for benchmarking purposes
  if (!rp->test_mode) {
    PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 1));
    PetscCall(KSPSolve(ksp, rhs, X));
    my_rt_start = MPI_Wtime();
    PetscCall(KSPSolve(ksp, rhs, X));
    my_rt = MPI_Wtime() - my_rt_start;
    PetscCall(MPI_Allreduce(MPI_IN_PLACE, &my_rt, 1, MPI_DOUBLE, MPI_MIN, rp->comm));
    // Set maxits based on first iteration timing
    PetscCall(KSPSetMinimumIterations(ksp, rp->ksp_max_it_clip[0]));
    if (my_rt > 0.02) {
      PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, rp->ksp_max_it_clip[0]));
    } else {
      PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, rp->ksp_max_it_clip[1]));
    }
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
                            "    KSP Type                                : %s\n"
                            "    KSP Convergence                         : %s\n"
                            "    Total KSP Iterations                    : %" PetscInt_FMT "\n"
                            "    Final rnorm                             : %e\n",
                            ksp_type, KSPConvergedReasons[reason], its, (double)rnorm));
    }
    if (!rp->test_mode) {
      PetscCall(PetscPrintf(rp->comm, "  Performance:\n"));
    }

    {
      CeedOperator         op_error;
      OperatorApplyContext op_error_ctx;

      // Set up error operator context
      PetscCall(PetscMalloc1(1, &op_error_ctx));
      PetscCall(SetupErrorOperator(dm_mesh, ceed, bp_options[rp->bp_choice], rp->dim, rp->dim, rp->num_comp_u, &op_error));
      PetscCall(SetupErrorOperatorCtx(rp->comm, dm_mesh, ceed, ceed_data, X_loc, op_error, op_error_ctx));
      PetscScalar l2_error;
      PetscCall(ComputeL2Error(X, &l2_error, op_error_ctx));
      // Tighter tol for BP1, BP2
      // Looser tol for BP3, BP4, BP5, and BP6 with extra for vector valued problems
      // BP1+3 and BP2+4 follow the pattern for BP3 and BP4
      // BP1+5 and BP2+6 follow the pattern for BP5 and BP6
      PetscReal tol = rp->tolerance < 0 ? rp->bp_choice < CEED_BP3 ? 5e-4 : (5e-2 + (rp->bp_choice % 2 == 1 ? 5e-3 : 0)) : rp->tolerance;
      if (!rp->test_mode || l2_error > tol) {
        PetscCall(MPI_Allreduce(&my_rt, &rt_min, 1, MPI_DOUBLE, MPI_MIN, rp->comm));
        PetscCall(MPI_Allreduce(&my_rt, &rt_max, 1, MPI_DOUBLE, MPI_MAX, rp->comm));
        PetscCall(PetscPrintf(rp->comm,
                              "    L2 Error                                : %e\n"
                              "    CG Solve Time                           : %g (%g) sec\n",
                              (double)l2_error, rt_max, rt_min));
      }

      // Cleanup
      PetscCall(VecDestroy(&op_error_ctx->Y_loc));
      PetscCall(PetscFree(op_error_ctx));
      CeedOperatorDestroy(&op_error);
    }
    if (!rp->test_mode) {
      PetscCall(PetscPrintf(rp->comm, "    DoFs/Sec in CG                          : %g (%g) million\n", 1e-6 * g_size * its / rt_max,
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

  if (rp->write_true_solution_swarm) {
    // View true solution at particles
    Vec u_swarm, u_swarm_old;

    PetscCall(DMSwarmSortGetAccess(dm_swarm));
    PetscCall(DMSwarmCreateLocalVectorFromField(dm_swarm, DMSwarmPICField_u, &u_swarm));
    PetscCall(VecDuplicate(u_swarm, &u_swarm_old));
    PetscCall(VecCopy(u_swarm, u_swarm_old));
    PetscCall(VecCopy(target, u_swarm));
    PetscCall(DMSwarmDestroyLocalVectorFromField(dm_swarm, DMSwarmPICField_u, &u_swarm));
    PetscCall(DMSwarmSortRestoreAccess(dm_swarm));

    PetscCall(DMSwarmViewXDMF(dm_swarm, "true_solution_swarm.xmf"));

    PetscCall(DMSwarmSortGetAccess(dm_swarm));
    PetscCall(DMSwarmCreateLocalVectorFromField(dm_swarm, DMSwarmPICField_u, &u_swarm));
    PetscCall(VecCopy(u_swarm_old, u_swarm));
    PetscCall(DMSwarmDestroyLocalVectorFromField(dm_swarm, DMSwarmPICField_u, &u_swarm));
    PetscCall(DMSwarmSortRestoreAccess(dm_swarm));
    PetscCall(VecDestroy(&u_swarm_old));
  }

  // Cleanup
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&X_loc));
  PetscCall(VecDestroy(&op_apply_ctx->Y_loc));
  PetscCall(MatDestroy(&mat_O));
  PetscCall(PetscFree(op_apply_ctx));
  PetscCall(CeedDataDestroy(0, ceed_data));

  PetscCall(VecDestroy(&rhs));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&target));
  CeedDestroy(&ceed);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Run(RunParams rp, PetscInt num_resources, char *const *ceed_resources, PetscInt num_bp_choices, const BPType *bp_choices) {
  DM dm;

  PetscFunctionBeginUser;
  // Setup DM
  PetscCall(CreateDistributedDM(rp, &dm));
  {
    PetscBool is_simplex;

    PetscCall(DMPlexIsSimplex(dm, &is_simplex));
    PetscCheck(!is_simplex, rp->comm, PETSC_ERR_USER, "Only tensor-product background meshes supported");
  }

  for (PetscInt b = 0; b < num_bp_choices; b++) {
    DM       dm_deg;
    DM       dm_swarm;
    VecType  vec_type;
    PetscInt q_extra = rp->q_extra, num_points = rp->swarm_num_points, num_points_per_cell_1d = rp->swarm_num_points_per_cell_1d;
    PetscInt dim;

    rp->bp_choice  = bp_choices[b];
    rp->num_comp_u = bp_options[rp->bp_choice].num_comp_u;
    rp->q_extra    = q_extra < 0 ? bp_options[rp->bp_choice].q_extra : q_extra;
    PetscCall(DMClone(dm, &dm_deg));
    PetscCall(DMGetVecType(dm, &vec_type));
    PetscCall(DMSetVecType(dm_deg, vec_type));
    // Create DM
    PetscCall(DMGetDimension(dm_deg, &dim));
    PetscCall(SetupDMByDegree(dm_deg, rp->degree, 0, rp->num_comp_u, dim, bp_options[rp->bp_choice].enforce_bc));
    // Create particle swarm
    // default to q = p + q_extra
    if (num_points_per_cell_1d < 0) rp->swarm_num_points_per_cell_1d = rp->degree + 1 + rp->q_extra;
    if (num_points < 0) {
      PetscInt c_start, c_end;

      PetscCall(DMPlexGetHeightStratum(dm, 0, &c_start, &c_end));
      rp->swarm_num_points = PetscPowInt(rp->swarm_num_points_per_cell_1d, rp->dim) * (c_end - c_start);
    }
    PetscCall(DMCreate(rp->comm, &dm_swarm));
    PetscCall(DMSetType(dm_swarm, DMSWARM));
    PetscCall(DMSetDimension(dm_swarm, dim));
    PetscCall(DMSwarmSetType(dm_swarm, DMSWARM_PIC));
    PetscCall(DMSwarmSetCellDM(dm_swarm, dm_deg));
    // Swarm field
    PetscCall(DMSwarmRegisterPetscDatatypeField(dm_swarm, DMSwarmPICField_u, rp->num_comp_u, PETSC_SCALAR));
    PetscCall(DMSwarmFinalizeFieldRegister(dm_swarm));
    // Set swarm point locations
    PetscCall(DMSwarmInitalizePointLocations(dm_swarm, rp->swarm_type, rp->swarm_num_points, rp->swarm_num_points_per_cell_1d));
    PetscCall(DMSwarmVectorDefineField(dm_swarm, DMSwarmPICField_u));
    PetscCall(DMSetFromOptions(dm_swarm));
    for (PetscInt r = 0; r < num_resources; r++) {
      PetscCall(RunWithDM(rp, dm_swarm, ceed_resources[r]));
    }
    PetscCall(DMDestroy(&dm_deg));
    PetscCall(DMDestroy(&dm_swarm));
    rp->q_extra                      = q_extra;
    rp->swarm_num_points_per_cell_1d = num_points_per_cell_1d;
    rp->swarm_num_points             = num_points;
  }

  PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv) {
  PetscMPIInt comm_size;
  RunParams   rp;
  MPI_Comm    comm;
  char        filename[PETSC_MAX_PATH_LEN];
  char       *ceed_resources[30];
  PetscInt    num_ceed_resources = 30;
  char        hostname[PETSC_MAX_PATH_LEN];

  PetscInt    dim = 3, mesh_elem[3] = {3, 3, 3};
  PetscInt    num_degrees = 30, degree[30] = {0}, num_local_nodes = 2, local_nodes[2] = {0};
  PetscMPIInt ranks_per_node;
  PetscBool   degree_set, points_per_cell_set, points_set;
  BPType      bp_choices[10];
  PetscInt    num_bp_choices = 10;

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
  rp->tolerance = -1;
  PetscCall(PetscOptionsScalar("-tolerance", "Tolerance for L2 error", NULL, rp->tolerance, &rp->tolerance, NULL));
  rp->write_solution = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-write_solution", "Write solution for visualization", NULL, rp->write_solution, &rp->write_solution, NULL));
  rp->simplex = PETSC_FALSE;
  degree[0]   = rp->test_mode ? 3 : 2;
  PetscCall(PetscOptionsIntArray("-degree", "Polynomial degree of tensor product basis", NULL, degree, &num_degrees, &degree_set));
  if (!degree_set) num_degrees = 1;
  rp->q_extra = PETSC_DECIDE;
  PetscCall(PetscOptionsInt("-q_extra", "Number of extra swarm points in each dimension (-1 for auto)", NULL, rp->q_extra, &rp->q_extra, NULL));
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
  rp->swarm_type = SWARM_GAUSS;
  PetscCall(PetscOptionsEnum("-swarm", "Swarm points distribution", NULL, PointSwarmTypes, (PetscEnum)rp->swarm_type, (PetscEnum *)&rp->swarm_type,
                             NULL));
  rp->swarm_num_points_per_cell_1d = -1;
  PetscCall(PetscOptionsInt("-points_per_cell_1d", "Total number of swarm points in each cell in each dimension", NULL,
                            rp->swarm_num_points_per_cell_1d, &rp->swarm_num_points_per_cell_1d, &points_per_cell_set));
  rp->swarm_num_points = -1;
  PetscCall(PetscOptionsInt("-local_points", "Total number of local swarm points", NULL, rp->swarm_num_points, &rp->swarm_num_points, &points_set));
  PetscCheck(!points_set || !points_per_cell_set, rp->comm, PETSC_ERR_USER, "Only one of -local_points and -points_per_cell can be set");
  PetscCheck(!points_set || rp->swarm_type == SWARM_SINUSOIDAL, rp->comm, PETSC_ERR_USER, "Only sinusoidal point location can use total point count");
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
