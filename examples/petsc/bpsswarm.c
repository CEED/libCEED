// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
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
//     bpssphere -problem bp1 -degree 3
//     bpssphere -problem bp2 -degree 3
//     bpssphere -problem bp3 -degree 3
//
//TESTARGS(name="BP2") -ceed {ceed_resource} -test -problem bp2 -dm_plex_dim 3 -dm_plex_box_faces 3,3,3 -dm_plex_simplex 0 -swarm uniform -points_per_cell 64
//TESTARGS(name="BP3") -ceed {ceed_resource} -test -problem bp3 -dm_plex_dim 3 -dm_plex_box_faces 4,4,4 -dm_plex_simplex 0 -swarm uniform -points_per_cell 64 -tolerance 3e-2
//TESTARGS(name="BP5") -ceed {ceed_resource} -test -problem bp5 -dm_plex_dim 3 -dm_plex_box_faces 3,3,3 -dm_plex_simplex 0 -swarm uniform -points_per_cell 64

/// @file
/// CEED BPs example using PETSc with DMPlex
/// See bpsraw.c for a "raw" implementation using a structured grid and bps.c for an implementation using an unstructured grid.
static const char help[]              = "Solve CEED BPs on a particle swarm using DMPlex and DMSwarm in PETSc\n";
const char        DMSwarmPICField_u[] = "u";

#include "bps.h"

#include <ceed.h>
#include <petscdmplex.h>
#include <petscksp.h>
#include <stdbool.h>
#include <string.h>

#include "include/bpsproblemdata.h"
#include "include/libceedsetup.h"
#include "include/matops.h"
#include "include/petscutils.h"
#include "include/petscversion.h"
#include "include/swarmutils.h"

int main(int argc, char **argv) {
  MPI_Comm             comm;
  char                 ceed_resource[PETSC_MAX_PATH_LEN] = "/cpu/self", filename[PETSC_MAX_PATH_LEN];
  double               my_rt_start, my_rt, rt_min, rt_max;
  PetscScalar          tolerance;
  PetscMPIInt          comm_size;
  PetscInt             degree, q_extra, l_size, g_size, dim = 3, num_comp_u = 1, xl_size, num_points = 1728, num_points_per_cell = 64;
  PetscBool            test_mode, benchmark_mode, read_mesh, write_solution, write_true_solution_swarm;
  PetscLogStage        solve_stage;
  Vec                  X, X_loc, rhs;
  Mat                  mat_O;
  KSP                  ksp;
  DM                   dm_mesh, dm_swarm;
  OperatorApplyContext op_apply_ctx, op_error_ctx;
  Ceed                 ceed;
  CeedData             ceed_data;
  CeedOperator         op_error;
  BPType               bp_choice;
  VecType              vec_type;
  PointSwarmType       point_swarm_type = SWARM_GAUSS;
  PetscMPIInt          ranks_per_node;
  char                 hostname[PETSC_MAX_PATH_LEN];

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
  write_true_solution_swarm = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-write_true_solution_swarm", "Write true solution at swarm points for visualization", NULL, write_true_solution_swarm,
                             &write_true_solution_swarm, NULL));
  degree = 2;
  PetscCall(PetscOptionsInt("-degree", "Polynomial degree of tensor product basis", NULL, degree, &degree, NULL));
  q_extra = bp_options[bp_choice].q_extra;
  PetscCall(PetscOptionsInt("-q_extra", "Number of extra quadrature points", NULL, q_extra, &q_extra, NULL));
  PetscCall(PetscOptionsString("-ceed", "CEED resource specifier", NULL, ceed_resource, ceed_resource, sizeof(ceed_resource), NULL));
  PetscCall(PetscGetHostName(hostname, sizeof hostname));
  PetscCall(PetscOptionsString("-hostname", "Hostname for output", NULL, hostname, hostname, sizeof(hostname), NULL));
  read_mesh = PETSC_FALSE;
  PetscCall(PetscOptionsString("-mesh", "Read mesh from file", NULL, filename, filename, sizeof(filename), &read_mesh));
  tolerance = 1e-2;
  PetscCall(PetscOptionsScalar("-tolerance", "Tolerance for L2 error", NULL, tolerance, &tolerance, NULL));
  PetscCall(PetscOptionsEnum("-swarm", "Swarm points distribution", NULL, point_swarm_types, (PetscEnum)point_swarm_type,
                             (PetscEnum *)&point_swarm_type, NULL));
  {
    PetscBool user_set_num_points_per_cell = PETSC_FALSE;
    PetscInt  num_cells_total = 1, tmp = dim;
    PetscInt  num_cells[] = {1, 1, 1};

    PetscCall(PetscOptionsInt("-points_per_cell", "Total number of swarm points in each cell", NULL, num_points_per_cell, &num_points_per_cell,
                              &user_set_num_points_per_cell));
    PetscCall(PetscOptionsInt("-dm_plex_dim", "Background mesh dimension", NULL, dim, &dim, NULL));
    PetscCall(PetscOptionsIntArray("-dm_plex_box_faces", "Number of cells", NULL, num_cells, &tmp, NULL));

    PetscCheck(tmp == dim, comm, PETSC_ERR_USER, "Number of values for -dm_plex_box_faces must match dimension");

    num_cells_total = num_cells[0] * num_cells[1] * num_cells[2];
    PetscCheck(!user_set_num_points_per_cell || point_swarm_type != SWARM_SINUSOIDAL, comm, PETSC_ERR_USER,
               "Cannot specify points per cell with sinusoidal points locations");
    if (!user_set_num_points_per_cell) {
      PetscCall(PetscOptionsInt("-points", "Total number of swarm points", NULL, num_points, &num_points, NULL));
      num_points_per_cell = PetscCeilInt(num_points, num_cells_total);
    }
    if (point_swarm_type != SWARM_SINUSOIDAL) {
      PetscInt num_points_per_cell_1d = round(cbrt(num_points_per_cell * 1.0));

      num_points_per_cell = 1;
      for (PetscInt i = 0; i < dim; i++) num_points_per_cell *= num_points_per_cell_1d;
    }
    num_points = num_points_per_cell * num_cells_total;
  }
  {
    PetscBool flg;
    PetscInt  p = ranks_per_node;
    PetscCall(PetscOptionsInt("-p", "Number of MPI ranks per node", NULL, p, &p, &flg));
    if (flg) ranks_per_node = p;
  }
  PetscOptionsEnd();

  // Setup DM
  if (read_mesh) {
    PetscCall(DMPlexCreateFromFile(comm, filename, NULL, PETSC_TRUE, &dm_mesh));
  } else {
    PetscCall(DMCreate(comm, &dm_mesh));
    PetscCall(DMSetType(dm_mesh, DMPLEX));
    PetscCall(DMSetFromOptions(dm_mesh));

    // -- Check for tensor product mesh
    {
      PetscBool is_simplex;

      PetscCall(DMPlexIsSimplex(dm_mesh, &is_simplex));
      PetscCheck(!is_simplex, comm, PETSC_ERR_USER, "Only tensor-product background meshes supported");
    }
  }
  PetscCall(DMGetDimension(dm_mesh, &dim));
  PetscCall(SetupDMByDegree(dm_mesh, degree, q_extra, num_comp_u, dim, bp_options[bp_choice].enforce_bc));

  // View mesh
  PetscCall(DMSetOptionsPrefix(dm_mesh, "final_"));
  PetscCall(DMViewFromOptions(dm_mesh, NULL, "-dm_view"));

  // Create particle swarm
  PetscCall(DMCreate(comm, &dm_swarm));
  PetscCall(DMSetType(dm_swarm, DMSWARM));
  PetscCall(DMSetDimension(dm_swarm, dim));
  PetscCall(DMSwarmSetType(dm_swarm, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(dm_swarm, dm_mesh));

  // -- Swarm field
  PetscCall(DMSwarmRegisterPetscDatatypeField(dm_swarm, DMSwarmPICField_u, num_comp_u, PETSC_SCALAR));
  PetscCall(DMSwarmFinalizeFieldRegister(dm_swarm));
  {
    PetscInt c_start, c_end, num_cells_local;
    PetscCall(DMPlexGetHeightStratum(dm_mesh, 0, &c_start, &c_end));
    num_cells_local = c_end - c_start;
    PetscCall(DMSwarmSetLocalSizes(dm_swarm, num_cells_local * num_points_per_cell, 0));
  }
  PetscCall(DMSetFromOptions(dm_swarm));

  // -- Set swarm point locations
  PetscCall(DMSwarmInitalizePointLocations(dm_swarm, point_swarm_type, num_points, num_points_per_cell));
  PetscCall(DMSwarmVectorDefineField(dm_swarm, DMSwarmPICField_u));

  // -- Final particle swarm
  PetscCall(PetscObjectSetName((PetscObject)dm_swarm, "Particle Swarm"));
  PetscCall(DMViewFromOptions(dm_swarm, NULL, "-dm_swarm_view"));

  // Create vectors
  PetscCall(DMCreateGlobalVector(dm_mesh, &X));
  PetscCall(VecGetLocalSize(X, &l_size));
  PetscCall(VecGetSize(X, &g_size));
  PetscCall(DMCreateLocalVector(dm_mesh, &X_loc));
  PetscCall(VecGetSize(X_loc, &xl_size));
  PetscCall(VecDuplicate(X, &rhs));

  // Operator
  PetscCall(PetscMalloc1(1, &op_apply_ctx));
  PetscCall(PetscMalloc1(1, &op_error_ctx));
  PetscCall(MatCreateShell(comm, l_size, l_size, g_size, g_size, op_apply_ctx, &mat_O));
  PetscCall(MatSetDM(mat_O, dm_mesh));
  PetscCall(MatShellSetOperation(mat_O, MATOP_MULT, (void (*)(void))MatMult_Ceed));
  PetscCall(MatShellSetOperation(mat_O, MATOP_GET_DIAGONAL, (void (*)(void))MatGetDiag));

  // Set up libCEED
  CeedInit(ceed_resource, &ceed);
  CeedMemType mem_type_backend;
  CeedGetPreferredMemType(ceed, &mem_type_backend);

  PetscCall(DMGetVecType(dm_mesh, &vec_type));
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
    PetscCall(DMSetVecType(dm_mesh, vec_type));
  }

  // Print summary
  if (!test_mode) {
    PetscInt P = degree + 1, Q = P + q_extra;

    const char *used_resource;
    CeedGetResource(ceed, &used_resource);

    VecType vec_type;
    PetscCall(VecGetType(X, &vec_type));

    PetscInt c_start, c_end, num_cells_local;
    PetscCall(DMPlexGetHeightStratum(dm_mesh, 0, &c_start, &c_end));
    num_cells_local = c_end - c_start;
    DMPolytopeType cell_type;
    PetscCall(DMPlexGetCellType(dm_mesh, c_start, &cell_type));
    PetscMPIInt comm_size;
    PetscCall(MPI_Comm_size(comm, &comm_size));

    PetscInt num_points_local, num_points_global;
    PetscCall(DMSwarmGetLocalSize(dm_swarm, &num_points_local));
    PetscCall(DMSwarmGetSize(dm_swarm, &num_points_global));

    PetscCall(PetscPrintf(comm,
                          "\n-- CEED Benchmark Problem %" CeedInt_FMT " -- libCEED + PETSc --\n"
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
                          "    Quadrature  Order (Q)                   : %" PetscInt_FMT "\n"
                          "    Additional quadrature points (q_extra)  : %" PetscInt_FMT "\n"
                          "    Global nodes                            : %" PetscInt_FMT "\n"
                          "    Local Elements                          : %" PetscInt_FMT "\n"
                          "    Owned nodes                             : %" PetscInt_FMT "\n"
                          "    DoF per node                            : %" PetscInt_FMT "\n"
                          "  Swarm:\n"
                          "    Global points                           : %" PetscInt_FMT "\n"
                          "    Local points                            : %" PetscInt_FMT "\n"
                          "    Avg points per cell                     : %" PetscInt_FMT "\n"
                          "    Point distribution                      : %s\n",
                          bp_choice + 1, hostname, comm_size, ranks_per_node, vec_type, used_resource, CeedMemTypes[mem_type_backend], P, Q, q_extra,
                          g_size / num_comp_u, num_cells_local, l_size / num_comp_u, num_comp_u, num_points_global, num_points_local,
                          num_cells_local > 0 ? num_points_local / num_cells_local : 0, point_swarm_types[point_swarm_type]));
  }

  // Setup libCEED's objects
  Vec target;

  PetscCall(DMCreateLocalVector(dm_swarm, &target));
  PetscCall(PetscMalloc1(1, &ceed_data));
  PetscCall(SetupProblemSwarm(dm_swarm, ceed, bp_options[bp_choice], ceed_data, true, rhs, target));
  PetscCall(SetupErrorOperator(dm_mesh, ceed, bp_options[bp_choice], dim, dim, num_comp_u, &op_error));

  // Set up apply operator context
  PetscCall(SetupApplyOperatorCtx(comm, dm_mesh, ceed, ceed_data, X_loc, op_apply_ctx));

  // Setup solver
  PetscCall(KSPCreate(comm, &ksp));
  {
    PC pc;
    PetscCall(KSPGetPC(ksp, &pc));
    if (bp_choice == CEED_BP1 || bp_choice == CEED_BP2) {
      PetscCall(PCSetType(pc, PCJACOBI));
      PetscCall(PCJacobiSetType(pc, PC_JACOBI_DIAGONAL));
    } else {
      PetscCall(PCSetType(pc, PCNONE));
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

    // View true solution at particles
    if (write_true_solution_swarm) {
      Vec u_swarm, u_swarm_old;
      PetscCall(DMSwarmSortGetAccess(dm_swarm));
      PetscCall(DMSwarmCreateLocalVectorFromField(dm_swarm, DMSwarmPICField_u, &u_swarm));
      PetscCall(VecDuplicate(u_swarm, &u_swarm_old));
      PetscCall(VecCopy(u_swarm, u_swarm_old));
      PetscCall(VecCopy(target, u_swarm));
      PetscCall(DMSwarmDestroyLocalVectorFromField(dm_swarm, DMSwarmPICField_u, &u_swarm));
      PetscCall(DMSwarmSortRestoreAccess(dm_swarm));

      PetscCall(DMSwarmViewXDMF(dm_swarm, "swarm.xmf"));

      PetscCall(DMSwarmSortGetAccess(dm_swarm));
      PetscCall(DMSwarmCreateLocalVectorFromField(dm_swarm, DMSwarmPICField_u, &u_swarm));
      PetscCall(VecCopy(u_swarm_old, u_swarm));
      PetscCall(DMSwarmDestroyLocalVectorFromField(dm_swarm, DMSwarmPICField_u, &u_swarm));
      PetscCall(DMSwarmSortRestoreAccess(dm_swarm));
      PetscCall(VecDestroy(&u_swarm_old));
    }

    // View solution at mesh points
    PetscCall(VecViewFromOptions(X, NULL, "-solution_view"));

    // Compute L2 Error
    {
      // Set up error operator context
      PetscCall(SetupErrorOperatorCtx(comm, dm_mesh, ceed, ceed_data, X_loc, op_error, op_error_ctx));
      PetscScalar l2_error;
      PetscCall(ComputeL2Error(X, &l2_error, op_error_ctx));

      if (!test_mode || l2_error > tolerance) {
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
  PetscCall(DMDestroy(&dm_mesh));
  PetscCall(DMDestroy(&dm_swarm));

  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&target));
  PetscCall(KSPDestroy(&ksp));
  CeedOperatorDestroy(&op_error);
  CeedDestroy(&ceed);
  return PetscFinalize();
}
