// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

//                        libCEED + PETSc DMSwarm Example
//
// This example demonstrates a simple usage of libCEED with DMSwarm.
// This example combines elements of PETSc src/impls/dm/swam/tutorials/ex1.c and src/impls/dm/swarm/tests/ex6.c
//
// Build with:
//
//     make dmswarm [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     ./dmswarm -dm_plex_dim 3 -dm_plex_box_faces 3,3,3 -dm_plex_box_lower -1.0,-1.0,-1.0 -dm_plex_simplex 0 -dm_plex_hash_location true -num_comp 2
//     -gauss_swarm
//
//TESTARGS(name="Uniform swarm, CG projection") -ceed {ceed_resource} -test -dm_plex_dim 3 -dm_plex_box_faces 3,3,3 -dm_plex_box_lower -1.0,-1.0,-1.0 -dm_plex_simplex 0 -dm_plex_hash_location true -num_comp 2 -uniform_swarm
//TESTARGS(name="Gauss swarm, lumped projection") -ceed {ceed_resource} -test -dm_plex_dim 3 -dm_plex_box_faces 3,3,3 -dm_plex_box_lower -1.0,-1.0,-1.0 -dm_plex_simplex 0 -dm_plex_hash_location true -num_comp 2 -gauss_swarm -ksp_type preonly -pc_type jacobi -pc_jacobi_type rowsum -tolerance 9e-2

/// @file
/// libCEED example using PETSc with DMSwarm
const char help[] = "libCEED example using PETSc with DMSwarm\n";

#include <ceed.h>
#include <math.h>
#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscds.h>
#include <petscfe.h>
#include <petscksp.h>
#include <petsc/private/petscfeimpl.h> /* For interpolation */

#include "include/petscutils.h"

typedef PetscErrorCode (*DMFunc)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx);

typedef struct DMSwarmCeedContext_ *DMSwarmCeedContext;
struct DMSwarmCeedContext_ {
  Ceed                ceed;
  CeedVector          u_mesh_loc, v_mesh_loc, u_mesh_elem, u_points_loc, u_points_elem, x_ref_points_loc, x_ref_points_elem;
  CeedElemRestriction restriction_u_mesh, restriction_x_points, restriction_u_points;
  CeedBasis           basis_u;
};

const char DMSwarmPICField_u[] = "u";

PetscErrorCode DMSwarmCeedContextCreate(DM dm_swarm, const char *ceed_resource, DMSwarmCeedContext *ctx);
PetscErrorCode DMSwarmCeedContextDestroy(DMSwarmCeedContext *ctx);

PetscErrorCode VecP2C(Vec X_petsc, PetscMemType *mem_type, CeedVector x_ceed);
PetscErrorCode VecC2P(CeedVector x_ceed, PetscMemType mem_type, Vec X_petsc);
PetscErrorCode VecReadP2C(Vec X_petsc, PetscMemType *mem_type, CeedVector x_ceed);
PetscErrorCode VecReadC2P(CeedVector x_ceed, PetscMemType mem_type, Vec X_petsc);
PetscErrorCode DMSwarmPICFieldP2C(DM dm_swarm, const char *field, CeedVector x_ceed);
PetscErrorCode DMSwarmPICFieldC2P(DM dm_swarm, const char *field, CeedVector x_ceed);

PetscScalar    EvalU_Poly(PetscInt dim, const PetscScalar x[]);
PetscScalar    EvalU_Tanh(PetscInt dim, const PetscScalar x[]);
PetscErrorCode EvalU_Poly_proj(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt num_comp, PetscScalar *u, void *ctx);
PetscErrorCode EvalU_Tanh_proj(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt num_comp, PetscScalar *u, void *ctx);

PetscErrorCode DMSwarmInitalizePointLocations(DM dm_swarm, PetscInt num_points, PetscBool set_gauss_swarm, PetscBool set_uniform_swarm);

PetscErrorCode DMSwarmCreateReferenceCoordinates(DM dm_swarm, IS *is_points, Vec *ref_coords);
PetscErrorCode DMSwarmInterpolateFromCellToSwarm_Petsc(DM dm_swarm, const char *field, Vec U_mesh);
PetscErrorCode DMSwarmInterpolateFromCellToSwarm_Ceed(DM dm_swarm, const char *field, Vec U_mesh);
PetscErrorCode DMSwarmCheckSwarmValues(DM dm_swarm, const char *field, PetscScalar tolerance, DMFunc TrueSolution);

PetscErrorCode DMSwarmCreateProjectionRHS(DM dm_swarm, const char *field, Vec B_mesh);
PetscErrorCode MatMult_SwarmMass(Mat A, Vec U_mesh, Vec V_mesh);

// ------------------------------------------------------------------------------------------------
// main driver
// ------------------------------------------------------------------------------------------------
int main(int argc, char **argv) {
  MPI_Comm  comm;
  char      ceed_resource[PETSC_MAX_PATH_LEN] = "/cpu/self";
  PetscBool test_mode = PETSC_FALSE, set_uniform_swarm = PETSC_FALSE, set_gauss_swarm = PETSC_FALSE, view_petsc_swarm = PETSC_FALSE,
            view_ceed_swarm = PETSC_FALSE, use_polynomial_target = PETSC_FALSE;
  PetscInt           dim = 3, num_comp = 1, num_points = 200, mesh_order = 1, solution_order = 3, q_extra = 3;
  PetscScalar        tolerance = 1E-3;
  DM                 dm_mesh, dm_swarm;
  Vec                U_mesh;
  DMSwarmCeedContext swarm_ceed_context;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  // Read command line options
  PetscOptionsBegin(comm, NULL, "libCEED example using PETSc with DMSwarm", NULL);

  PetscCall(PetscOptionsBool("-test", "Testing mode (do not print unless error is large)", NULL, test_mode, &test_mode, NULL));
  PetscCall(PetscOptionsBool("-gauss_swarm", "Use gauss points for coordinates in swarm", NULL, set_gauss_swarm, &set_gauss_swarm, NULL));
  PetscCall(PetscOptionsBool("-uniform_swarm", "Use uniform coordinates in swarm", NULL, set_uniform_swarm, &set_uniform_swarm, NULL));
  PetscCall(
      PetscOptionsBool("-u_petsc_swarm_view", "View XDMF of swarm values interpolated by PETSc", NULL, view_petsc_swarm, &view_petsc_swarm, NULL));
  PetscCall(
      PetscOptionsBool("-u_ceed_swarm_view", "View XDMF of swarm values interpolated by libCEED", NULL, view_ceed_swarm, &view_ceed_swarm, NULL));
  PetscCall(PetscOptionsBool("-polynomial_target", "Use polynomial target function in field instead of tanh()", NULL, use_polynomial_target,
                             &use_polynomial_target, NULL));
  PetscCall(PetscOptionsInt("-solution_order", "Order of mesh solution space", NULL, solution_order, &solution_order, NULL));
  PetscCall(PetscOptionsInt("-mesh_order", "Order of mesh coordinate space", NULL, mesh_order, &mesh_order, NULL));
  PetscCall(PetscOptionsInt("-q_extra", "Number of extra quadrature points", NULL, q_extra, &q_extra, NULL));
  PetscCall(PetscOptionsInt("-num_comp", "Number of components in solution", NULL, num_comp, &num_comp, NULL));
  {
    PetscInt dim = 3, num_cells[] = {1, 1, 1};

    PetscCall(PetscOptionsInt("-dm_plex_dim", "Background mesh dimension", NULL, dim, &dim, NULL));
    PetscCall(PetscOptionsIntArray("-dm_plex_box_faces", "Number of cells", NULL, num_cells, &dim, NULL));
    PetscInt total_num_cells = num_cells[0] * num_cells[1] * num_cells[2];
    PetscInt points_per_cell = 1;

    for (PetscInt i = 0; i < dim; i++) points_per_cell *= solution_order + 2;
    num_points = 0;
    for (PetscInt i = 0; i < total_num_cells; i++) num_points += points_per_cell;
  }
  PetscCall(PetscOptionsInt("-points", "Total number of swarm points", NULL, num_points, &num_points, NULL));
  PetscCall(PetscOptionsScalar("-tolerance", "Tolerance for swarm point values and projection relative L2 error", NULL, tolerance, &tolerance, NULL));
  PetscCall(PetscOptionsString("-ceed", "CEED resource specifier", NULL, ceed_resource, ceed_resource, sizeof(ceed_resource), NULL));

  PetscOptionsEnd();

  // Create background mesh
  {
    PetscCall(DMCreate(comm, &dm_mesh));
    PetscCall(DMSetType(dm_mesh, DMPLEX));
    PetscCall(DMSetFromOptions(dm_mesh));

    // -- Check for tensor product mesh
    {
      PetscBool is_simplex;

      PetscCall(DMPlexIsSimplex(dm_mesh, &is_simplex));
      PetscCheck(!is_simplex, comm, PETSC_ERR_USER, "Only tensor-product background meshes supported");
    }

    // -- Mesh FE space
    PetscCall(DMGetDimension(dm_mesh, &dim));
    {
      PetscFE fe;

      PetscCall(DMGetDimension(dm_mesh, &dim));
      PetscCall(PetscFECreateLagrange(comm, dim, num_comp, PETSC_FALSE, solution_order, solution_order + q_extra, &fe));
      PetscCall(DMAddField(dm_mesh, NULL, (PetscObject)fe));
      PetscCall(PetscFEDestroy(&fe));
    }
    PetscCall(DMCreateDS(dm_mesh));

    // -- Coordinate FE space
    {
      PetscFE fe_coord;

      PetscCall(PetscFECreateLagrange(comm, dim, dim, PETSC_FALSE, mesh_order, solution_order + q_extra, &fe_coord));
      PetscCall(DMProjectCoordinates(dm_mesh, fe_coord));
      PetscCall(PetscFEDestroy(&fe_coord));
    }

    // -- Set tensor permutation
    {
      DM dm_coord;

      PetscCall(DMGetCoordinateDM(dm_mesh, &dm_coord));
      PetscCall(DMPlexSetClosurePermutationTensor(dm_mesh, PETSC_DETERMINE, NULL));
      PetscCall(DMPlexSetClosurePermutationTensor(dm_coord, PETSC_DETERMINE, NULL));
    }

    // -- Final background mesh
    PetscCall(PetscObjectSetName((PetscObject)dm_mesh, "Background Mesh"));
    PetscCall(DMViewFromOptions(dm_mesh, NULL, "-dm_mesh_view"));
  }

  // Create particle swarm
  {
    PetscCall(DMCreate(comm, &dm_swarm));
    PetscCall(DMSetType(dm_swarm, DMSWARM));
    PetscCall(DMSetDimension(dm_swarm, dim));
    PetscCall(DMSwarmSetType(dm_swarm, DMSWARM_PIC));
    PetscCall(DMSwarmSetCellDM(dm_swarm, dm_mesh));

    // -- Swarm field
    PetscCall(DMSwarmRegisterPetscDatatypeField(dm_swarm, DMSwarmPICField_u, num_comp, PETSC_SCALAR));
    PetscCall(DMSwarmFinalizeFieldRegister(dm_swarm));
    PetscCall(DMSwarmSetLocalSizes(dm_swarm, num_points, 0));
    PetscCall(DMSetFromOptions(dm_swarm));

    // -- Set swarm point locations
    PetscCall(DMSwarmInitalizePointLocations(dm_swarm, num_points, set_gauss_swarm, set_uniform_swarm));

    // -- Final particle swarm
    PetscCall(PetscObjectSetName((PetscObject)dm_swarm, "Particle Swarm"));
    PetscCall(DMViewFromOptions(dm_swarm, NULL, "-dm_swarm_view"));
  }

  // Set field values on background mesh
  PetscCall(DMCreateGlobalVector(dm_mesh, &U_mesh));
  {
    DMFunc mesh_solution[1] = {use_polynomial_target ? EvalU_Poly_proj : EvalU_Tanh_proj};

    PetscCall(DMProjectFunction(dm_mesh, 0.0, mesh_solution, NULL, INSERT_VALUES, U_mesh));
  }

  // Visualize background mesh
  PetscCall(VecViewFromOptions(U_mesh, NULL, "-u_mesh_view"));

  // libCEED objects for swarm and background mesh
  PetscCall(DMSwarmCeedContextCreate(dm_swarm, ceed_resource, &swarm_ceed_context));

  // Interpolate from mesh to points via PETSc
  {
    PetscCall(DMSwarmInterpolateFromCellToSwarm_Petsc(dm_swarm, DMSwarmPICField_u, U_mesh));
    if (view_petsc_swarm) PetscCall(DMSwarmViewXDMF(dm_swarm, "swarm_petsc.xmf"));
    PetscCall(DMSwarmCheckSwarmValues(dm_swarm, DMSwarmPICField_u, tolerance, use_polynomial_target ? EvalU_Poly_proj : EvalU_Tanh_proj));
  }

  // Interpolate from mesh to points via libCEED
  {
    PetscCall(DMSwarmInterpolateFromCellToSwarm_Ceed(dm_swarm, DMSwarmPICField_u, U_mesh));
    if (view_ceed_swarm) PetscCall(DMSwarmViewXDMF(dm_swarm, "swarm_ceed.xmf"));
    PetscCall(DMSwarmCheckSwarmValues(dm_swarm, DMSwarmPICField_u, tolerance, use_polynomial_target ? EvalU_Poly_proj : EvalU_Tanh_proj));
  }

  // Project from points to mesh via libCEED
  {
    Vec B_mesh, U_projected;
    Mat M;
    KSP ksp;

    PetscCall(VecDuplicate(U_mesh, &B_mesh));
    PetscCall(VecDuplicate(U_mesh, &U_projected));

    // -- Setup "mass matrix"
    {
      PetscInt l_size, g_size;

      PetscCall(VecGetLocalSize(U_mesh, &l_size));
      PetscCall(VecGetSize(U_mesh, &g_size));
      PetscCall(MatCreateShell(comm, l_size, l_size, g_size, g_size, swarm_ceed_context, &M));
      PetscCall(MatSetDM(M, dm_mesh));
      PetscCall(MatShellSetOperation(M, MATOP_MULT, (void (*)(void))MatMult_SwarmMass));
    }

    // -- Setup KSP
    {
      PC pc;

      PetscCall(KSPCreate(comm, &ksp));
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCSetType(pc, PCJACOBI));
      PetscCall(PCJacobiSetType(pc, PC_JACOBI_ROWSUM));
      PetscCall(KSPSetType(ksp, KSPCG));
      PetscCall(KSPSetNormType(ksp, KSP_NORM_NATURAL));
      PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
      PetscCall(KSPSetOperators(ksp, M, M));
      PetscCall(KSPSetFromOptions(ksp));
      PetscCall(PetscObjectSetName((PetscObject)ksp, "Swarm-to-Mesh Projection"));
      PetscCall(KSPViewFromOptions(ksp, NULL, "-ksp_projection_view"));
    }

    // -- Setup RHS
    PetscCall(DMSwarmCreateProjectionRHS(dm_swarm, DMSwarmPICField_u, B_mesh));

    // -- Solve
    PetscCall(VecZeroEntries(U_projected));
    PetscCall(KSPSolve(ksp, B_mesh, U_projected));

    // -- KSP summary
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
                              "Swarm-to-Mesh Projection KSP Solve:\n"
                              "  KSP type: %s\n"
                              "  KSP convergence: %s\n"
                              "  Total KSP iterations: %" PetscInt_FMT "\n"
                              "  Final rnorm: %e\n",
                              ksp_type, KSPConvergedReasons[reason], its, (double)rnorm));
      }
    }

    // -- Check error
    PetscCall(KSPViewFromOptions(ksp, NULL, "-ksp_view"));
    PetscCall(VecAXPY(U_projected, -1.0, U_mesh));
    PetscCall(VecViewFromOptions(U_projected, NULL, "-u_error_view"));
    {
      PetscScalar error, norm_u_mesh;

      PetscCall(VecNorm(U_projected, NORM_2, &error));
      PetscCall(VecNorm(U_mesh, NORM_2, &norm_u_mesh));
      PetscCheck(error / norm_u_mesh < tolerance, comm, PETSC_ERR_USER, "Projection error too high: %e\n", error / norm_u_mesh);
      if (!test_mode) PetscCall(PetscPrintf(comm, "  Projection error: %e\n", error / norm_u_mesh));
    }

    // -- Cleanup
    PetscCall(VecDestroy(&B_mesh));
    PetscCall(VecDestroy(&U_projected));
    PetscCall(MatDestroy(&M));
    PetscCall(KSPDestroy(&ksp));
  }

  // Cleanup
  PetscCall(DMSwarmCeedContextDestroy(&swarm_ceed_context));
  PetscCall(DMDestroy(&dm_swarm));
  PetscCall(DMDestroy(&dm_mesh));
  PetscCall(VecDestroy(&U_mesh));
  return PetscFinalize();
}

// ------------------------------------------------------------------------------------------------
// Context utilities
// ------------------------------------------------------------------------------------------------
PetscErrorCode DMSwarmCeedContextCreate(DM dm_swarm, const char *ceed_resource, DMSwarmCeedContext *ctx) {
  DM dm_mesh;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(ctx));
  PetscCall(DMSwarmGetCellDM(dm_swarm, &dm_mesh));

  CeedInit(ceed_resource, &(*ctx)->ceed);
  // Background mesh objects
  {
    CeedInt elem_size, num_comp;
    BPData  bp_data = {.q_mode = CEED_GAUSS};

    PetscCall(CreateBasisFromPlex((*ctx)->ceed, dm_mesh, NULL, 0, 0, 0, bp_data, &(*ctx)->basis_u));
    PetscCall(CreateRestrictionFromPlex((*ctx)->ceed, dm_mesh, 0, NULL, 0, &(*ctx)->restriction_u_mesh));

    // -- U vector
    CeedElemRestrictionCreateVector((*ctx)->restriction_u_mesh, &(*ctx)->u_mesh_loc, NULL);
    CeedElemRestrictionCreateVector((*ctx)->restriction_u_mesh, &(*ctx)->v_mesh_loc, NULL);
    CeedElemRestrictionGetElementSize((*ctx)->restriction_u_mesh, &elem_size);
    CeedElemRestrictionGetNumComponents((*ctx)->restriction_u_mesh, &num_comp);
    CeedVectorCreate((*ctx)->ceed, elem_size * num_comp, &(*ctx)->u_mesh_elem);
  }
  // Swarm objects
  {
    PetscInt        dim;
    const PetscInt *cell_points;
    IS              is_points;
    Vec             X_ref;
    CeedInt         num_elem, num_comp, max_points_in_cell;

    PetscCall(DMSwarmCreateReferenceCoordinates(dm_swarm, &is_points, &X_ref));
    PetscCall(DMGetDimension(dm_mesh, &dim));
    CeedElemRestrictionGetNumElements((*ctx)->restriction_u_mesh, &num_elem);
    CeedElemRestrictionGetNumComponents((*ctx)->restriction_u_mesh, &num_comp);

    PetscCall(ISGetIndices(is_points, &cell_points));
    PetscInt num_points = cell_points[num_elem + 1] - num_elem - 2;
    CeedInt  offsets[num_elem + 1 + num_points];

    for (PetscInt i = 0; i < num_elem + 1; i++) offsets[i] = cell_points[i + 1] - 1;
    for (PetscInt i = num_elem + 1; i < num_points + num_elem + 1; i++) offsets[i] = cell_points[i + 1];
    PetscCall(ISRestoreIndices(is_points, &cell_points));

    // -- Points restrictions
    CeedElemRestrictionCreateAtPoints((*ctx)->ceed, num_elem, num_points, num_comp, num_points * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES, offsets,
                                      &(*ctx)->restriction_u_points);
    CeedElemRestrictionCreateAtPoints((*ctx)->ceed, num_elem, num_points, dim, num_points * dim, CEED_MEM_HOST, CEED_COPY_VALUES, offsets,
                                      &(*ctx)->restriction_x_points);

    // -- U vector
    CeedElemRestrictionGetMaxPointsInElement((*ctx)->restriction_u_points, &max_points_in_cell);
    CeedElemRestrictionCreateVector((*ctx)->restriction_u_points, &(*ctx)->u_points_loc, NULL);
    CeedVectorCreate((*ctx)->ceed, max_points_in_cell * num_comp, &(*ctx)->u_points_elem);

    // -- Ref coordinates
    {
      PetscMemType       X_mem_type;
      const PetscScalar *x;

      CeedVectorCreate((*ctx)->ceed, num_points * dim, &(*ctx)->x_ref_points_loc);
      CeedVectorCreate((*ctx)->ceed, max_points_in_cell * dim, &(*ctx)->x_ref_points_elem);

      PetscCall(VecGetArrayReadAndMemType(X_ref, (const PetscScalar **)&x, &X_mem_type));
      CeedVectorSetArray((*ctx)->x_ref_points_loc, MemTypeP2C(X_mem_type), CEED_COPY_VALUES, (CeedScalar *)x);
      PetscCall(VecRestoreArrayReadAndMemType(X_ref, (const PetscScalar **)&x));
    }

    // -- Cleanup
    PetscCall(ISDestroy(&is_points));
    PetscCall(VecDestroy(&X_ref));
  }

  PetscCall(DMSetApplicationContext(dm_mesh, (void *)(*ctx)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmCeedContextDestroy(DMSwarmCeedContext *ctx) {
  PetscFunctionBeginUser;
  CeedDestroy(&(*ctx)->ceed);
  CeedVectorDestroy(&(*ctx)->u_mesh_loc);
  CeedVectorDestroy(&(*ctx)->v_mesh_loc);
  CeedVectorDestroy(&(*ctx)->u_mesh_elem);
  CeedVectorDestroy(&(*ctx)->u_points_loc);
  CeedVectorDestroy(&(*ctx)->u_points_elem);
  CeedVectorDestroy(&(*ctx)->x_ref_points_loc);
  CeedVectorDestroy(&(*ctx)->x_ref_points_elem);
  CeedElemRestrictionDestroy(&(*ctx)->restriction_u_mesh);
  CeedElemRestrictionDestroy(&(*ctx)->restriction_x_points);
  CeedElemRestrictionDestroy(&(*ctx)->restriction_u_points);
  CeedBasisDestroy(&(*ctx)->basis_u);
  PetscCall(PetscFree(*ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ------------------------------------------------------------------------------------------------
// PETSc-libCEED memory space utilities
// ------------------------------------------------------------------------------------------------
PetscErrorCode VecP2C(Vec X_petsc, PetscMemType *mem_type, CeedVector x_ceed) {
  PetscScalar *x;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayAndMemType(X_petsc, &x, mem_type));
  CeedVectorSetArray(x_ceed, MemTypeP2C(*mem_type), CEED_USE_POINTER, x);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecC2P(CeedVector x_ceed, PetscMemType mem_type, Vec X_petsc) {
  PetscScalar *x;

  PetscFunctionBeginUser;
  CeedVectorTakeArray(x_ceed, MemTypeP2C(mem_type), &x);
  PetscCall(VecRestoreArrayAndMemType(X_petsc, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecReadP2C(Vec X_petsc, PetscMemType *mem_type, CeedVector x_ceed) {
  PetscScalar *x;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayReadAndMemType(X_petsc, (const PetscScalar **)&x, mem_type));
  CeedVectorSetArray(x_ceed, MemTypeP2C(*mem_type), CEED_USE_POINTER, x);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecReadC2P(CeedVector x_ceed, PetscMemType mem_type, Vec X_petsc) {
  PetscScalar *x;

  PetscFunctionBeginUser;
  CeedVectorTakeArray(x_ceed, MemTypeP2C(mem_type), &x);
  PetscCall(VecRestoreArrayReadAndMemType(X_petsc, (const PetscScalar **)&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmPICFieldP2C(DM dm_swarm, const char *field, CeedVector x_ceed) {
  PetscScalar *x;

  PetscFunctionBeginUser;
  PetscCall(DMSwarmGetField(dm_swarm, field, NULL, NULL, (void **)&x));
  CeedVectorSetArray(x_ceed, CEED_MEM_HOST, CEED_USE_POINTER, (CeedScalar *)x);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmPICFieldC2P(DM dm_swarm, const char *field, CeedVector x_ceed) {
  PetscScalar *x;

  PetscFunctionBeginUser;
  CeedVectorTakeArray(x_ceed, CEED_MEM_HOST, (CeedScalar **)&x);
  PetscCall(DMSwarmRestoreField(dm_swarm, field, NULL, NULL, (void **)&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ------------------------------------------------------------------------------------------------
// Solution functions
// ------------------------------------------------------------------------------------------------
PetscScalar EvalU_Poly(PetscInt dim, const PetscScalar x[]) {
  PetscScalar       result = 0.0;
  const PetscScalar p[5]   = {3, 1, 4, 1, 5};

  for (PetscInt d = 0; d < dim; d++) {
    PetscScalar result_1d = 1.0;

    for (PetscInt i = 4; i >= 0; i--) result_1d = result_1d * x[d] + p[i];
    result += result_1d;
  }
  return result * 1E-3;
}

PetscScalar EvalU_Tanh(PetscInt dim, const PetscScalar x[]) {
  PetscScalar result = 1.0, center = 0.1;

  for (PetscInt d = 0; d < dim; d++) {
    result *= tanh(x[d] - center);
    center += 0.1;
  }
  return result;
}

PetscErrorCode EvalU_Poly_proj(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt num_comp, PetscScalar *u, void *ctx) {
  PetscFunctionBeginUser;

  const PetscScalar f_x = EvalU_Poly(dim, x);

  for (PetscInt c = 0; c < num_comp; c++) u[c] = (c + 1.0) * f_x;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EvalU_Tanh_proj(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt num_comp, PetscScalar *u, void *ctx) {
  PetscFunctionBeginUser;

  const PetscScalar f_x = EvalU_Tanh(dim, x);

  for (PetscInt c = 0; c < num_comp; c++) u[c] = (c + 1.0) * f_x;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ------------------------------------------------------------------------------------------------
// Swarm point location utility
// ------------------------------------------------------------------------------------------------
PetscErrorCode DMSwarmInitalizePointLocations(DM dm_swarm, PetscInt num_points, PetscBool set_gauss_swarm, PetscBool set_uniform_swarm) {
  PetscFunctionBeginUser;

  if (set_gauss_swarm || set_uniform_swarm) {
    // -- Set gauss quadrature point locations in each cell
    PetscBool user_set_points_per_cell = PETSC_FALSE;
    PetscInt  dim = 3, points_per_cell = num_points, points_per_cell_1d = num_points;
    PetscInt  num_cells[] = {1, 1, 1};
    DM        dm_mesh;

    PetscCall(DMSwarmGetCellDM(dm_swarm, &dm_mesh));
    PetscCall(DMGetDimension(dm_mesh, &dim));

    PetscOptionsBegin(PetscObjectComm((PetscObject)dm_swarm), NULL, "libCEED example using PETSc with DMSwarm", NULL);
    PetscCall(PetscOptionsInt("-points_per_cell", "Total number of swarm points in each cell", NULL, points_per_cell, &points_per_cell,
                              &user_set_points_per_cell));
    PetscCall(PetscOptionsIntArray("-dm_plex_box_faces", "Number of cells", NULL, num_cells, &dim, NULL));
    PetscOptionsEnd();

    if (!user_set_points_per_cell) {
      PetscInt total_num_cells = num_cells[0] * num_cells[1] * num_cells[2];

      points_per_cell = PetscCeilInt(num_points, total_num_cells);
    }
    points_per_cell_1d = ceil(cbrt(points_per_cell * 1.0));
    points_per_cell = 1;
    for (PetscInt i = 0; i < dim; i++) points_per_cell *= points_per_cell_1d;

    PetscScalar point_coords[points_per_cell * 3];
    CeedScalar  points_1d[points_per_cell_1d], weights_1d[points_per_cell_1d];

    if (set_gauss_swarm) {
      PetscCall(CeedGaussQuadrature(points_per_cell_1d, points_1d, weights_1d));
    } else {
      for (PetscInt i = 0; i < points_per_cell_1d; i++) points_1d[i] = 2.0 * (PetscReal)(i + 1) / (PetscReal)(points_per_cell_1d + 1) - 1;
    }
    for (PetscInt i = 0; i < points_per_cell_1d; i++) {
      for (PetscInt j = 0; j < points_per_cell_1d; j++) {
        for (PetscInt k = 0; k < points_per_cell_1d; k++) {
          PetscInt p = (i * points_per_cell_1d + j) * points_per_cell_1d + k;

          point_coords[p * dim + 0] = points_1d[i];
          point_coords[p * dim + 1] = points_1d[j];
          point_coords[p * dim + 2] = points_1d[k];
        }
      }
    }
    PetscCall(DMSwarmSetPointCoordinatesCellwise(dm_swarm, points_per_cell_1d * points_per_cell_1d * points_per_cell_1d, point_coords));
  } else {
    // -- Set points distributed per sinusoidal functions
    PetscInt     dim = 3;
    PetscScalar *point_coords;
    DM           dm_mesh;

    PetscCall(DMSwarmGetCellDM(dm_swarm, &dm_mesh));
    PetscCall(DMGetDimension(dm_mesh, &dim));

    PetscCall(DMSwarmGetField(dm_swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&point_coords));
    for (PetscInt p = 0; p < num_points; p++) {
      point_coords[p * dim + 0] = -PetscCosReal((PetscReal)(p + 1) / (PetscReal)(num_points + 1) * PETSC_PI);
      if (dim > 1) point_coords[p * dim + 1] = -PetscSinReal((PetscReal)(p + 1) / (PetscReal)(num_points + 1) * PETSC_PI);
      if (dim > 2) point_coords[p * dim + 2] = PetscSinReal((PetscReal)(p + 1) / (PetscReal)(num_points + 1) * PETSC_PI);
    }
    PetscCall(DMSwarmRestoreField(dm_swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&point_coords));
  }
  PetscCall(DMSwarmMigrate(dm_swarm, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMSwarmCreateReferenceCoordinates - Compute the cell reference coordinates for local DMSwarm points.

  Collective

  Input Parameter:
. dm_swarm  - the `DMSwarm`

  Output Parameters:
+ is_points    - The IS object for indexing into points per cell
- X_points_ref - Vec holding the cell reference coordinates for local DMSwarm points

The index set contains ranges of indices for each local cell. This range contains the indices of every point in the cell.

```
total_num_cells
cell_0_start_index
cell_1_start_index
cell_2_start_index
...
cell_n_start_index
cell_n_stop_index
cell_0_point_0
cell_0_point_0
...
cell_n_point_m
```

  Level: beginner

.seealso: `DMSwarm`
@*/
PetscErrorCode DMSwarmCreateReferenceCoordinates(DM dm_swarm, IS *is_points, Vec *X_points_ref) {
  PetscInt           cell_start, cell_end, num_cells_local, dim, num_points_local, *cell_points, points_offset;
  PetscScalar       *coords_points_ref;
  const PetscScalar *coords_points_true;
  DM                 dm_mesh;

  PetscFunctionBeginUser;
  PetscCall(DMSwarmGetCellDM(dm_swarm, &dm_mesh));

  // Create vector to hold reference coordinates
  {
    Vec X_points_true;

    PetscCall(DMSwarmCreateLocalVectorFromField(dm_swarm, DMSwarmPICField_coor, &X_points_true));
    PetscCall(VecDuplicate(X_points_true, X_points_ref));
    PetscCall(DMSwarmDestroyLocalVectorFromField(dm_swarm, DMSwarmPICField_coor, &X_points_true));
  }

  // Allocate index set array
  PetscCall(DMPlexGetHeightStratum(dm_mesh, 0, &cell_start, &cell_end));
  num_cells_local = cell_end - cell_start;
  points_offset   = num_cells_local + 2;
  PetscCall(VecGetLocalSize(*X_points_ref, &num_points_local));
  PetscCall(DMGetDimension(dm_mesh, &dim));
  num_points_local /= dim;
  PetscCall(PetscMalloc1(num_points_local + num_cells_local + 2, &cell_points));
  cell_points[0] = num_cells_local;

  // Get reference coordinates for each swarm point wrt the elements in the background mesh
  PetscCall(DMSwarmSortGetAccess(dm_swarm));
  PetscCall(DMSwarmGetField(dm_swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords_points_true));
  PetscCall(VecGetArray(*X_points_ref, &coords_points_ref));
  for (PetscInt cell = cell_start, num_points_processed = 0; cell < cell_end; cell++) {
    PetscInt *points_in_cell, num_points_in_cell, local_cell = cell - cell_start;
    PetscReal v[3], J[9], invJ[9], detJ, v0_ref[3] = {-1.0, -1.0, -1.0};

    PetscCall(DMSwarmSortGetPointsPerCell(dm_swarm, cell, &num_points_in_cell, &points_in_cell));
    // -- Reference coordinates for swarm points in background mesh element
    PetscCall(DMPlexComputeCellGeometryFEM(dm_mesh, cell, NULL, v, J, invJ, &detJ));
    cell_points[local_cell + 1] = num_points_processed + points_offset;
    for (PetscInt p = 0; p < num_points_in_cell; p++) {
      const CeedInt point = points_in_cell[p];

      cell_points[points_offset + (num_points_processed++)] = point;
      CoordinatesRealToRef(dim, dim, v0_ref, v, invJ, &coords_points_true[point * dim], &coords_points_ref[point * dim]);
    }

    // -- Cleanup
    PetscCall(PetscFree(points_in_cell));
  }
  cell_points[points_offset - 1] = num_points_local + points_offset;

  // Cleanup
  PetscCall(DMSwarmRestoreField(dm_swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords_points_true));
  PetscCall(VecRestoreArray(*X_points_ref, &coords_points_ref));
  PetscCall(DMSwarmSortRestoreAccess(dm_swarm));

  // Create index set
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, num_points_local + points_offset, cell_points, PETSC_OWN_POINTER, is_points));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ------------------------------------------------------------------------------------------------
// Projection via PETSc
// ------------------------------------------------------------------------------------------------
PetscErrorCode DMSwarmInterpolateFromCellToSwarm_Petsc(DM dm_swarm, const char *field, Vec U_mesh) {
  PetscInt           dim, num_comp, cell_start, cell_end;
  PetscScalar       *u_points;
  const PetscScalar *coords_points;
  const PetscReal    v0_ref[3] = {-1.0, -1.0, -1.0};
  DM                 dm_mesh;
  PetscSection       section_u_mesh_loc;
  PetscDS            ds;
  PetscFE            fe;
  PetscFEGeom        fe_geometry;
  PetscQuadrature    quadrature;
  Vec                U_loc;

  PetscFunctionBeginUser;
  // Get mesh DM
  PetscCall(DMSwarmGetCellDM(dm_swarm, &dm_mesh));
  PetscCall(DMGetDimension(dm_mesh, &dim));
  {
    PetscSection section_u_mesh_loc_closure_permutation;

    PetscCall(DMGetLocalSection(dm_mesh, &section_u_mesh_loc_closure_permutation));
    PetscCall(PetscSectionClone(section_u_mesh_loc_closure_permutation, &section_u_mesh_loc));
    PetscCall(PetscSectionResetClosurePermutation(section_u_mesh_loc));
  }

  // Get local mesh values
  PetscCall(DMGetLocalVector(dm_mesh, &U_loc));
  PetscCall(VecZeroEntries(U_loc));
  PetscCall(DMGlobalToLocal(dm_mesh, U_mesh, INSERT_VALUES, U_loc));

  // Get local swarm data
  PetscCall(DMSwarmSortGetAccess(dm_swarm));
  PetscCall(DMPlexGetHeightStratum(dm_mesh, 0, &cell_start, &cell_end));
  PetscCall(DMSwarmGetField(dm_swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords_points));
  PetscCall(DMSwarmGetField(dm_swarm, field, &num_comp, NULL, (void **)&u_points));

  // Interpolate values to each swarm point, one element in the background mesh at a time
  PetscCall(DMGetDS(dm_mesh, &ds));
  PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject *)&fe));
  for (PetscInt cell = cell_start; cell < cell_end; cell++) {
    PetscTabulation tabulation;
    PetscScalar    *u_cell = NULL, *coords_points_cell_true, *coords_points_cell_ref;
    PetscReal       v[dim], J[dim * dim], invJ[dim * dim], detJ;
    PetscInt       *points_cell;
    PetscInt        num_points_in_cell;

    PetscCall(DMSwarmSortGetPointsPerCell(dm_swarm, cell, &num_points_in_cell, &points_cell));
    PetscCall(DMGetWorkArray(dm_mesh, num_points_in_cell * dim, MPIU_REAL, &coords_points_cell_true));
    PetscCall(DMGetWorkArray(dm_mesh, num_points_in_cell * dim, MPIU_REAL, &coords_points_cell_ref));
    // -- Reference coordinates for swarm points in background mesh element
    for (PetscInt p = 0; p < num_points_in_cell; p++) {
      for (PetscInt d = 0; d < dim; d++) coords_points_cell_true[p * dim + d] = coords_points[points_cell[p] * dim + d];
    }
    PetscCall(DMPlexComputeCellGeometryFEM(dm_mesh, cell, NULL, v, J, invJ, &detJ));
    for (PetscInt p = 0; p < num_points_in_cell; p++) {
      CoordinatesRealToRef(dim, dim, v0_ref, v, invJ, &coords_points_cell_true[p * dim], &coords_points_cell_ref[p * dim]);
    }
    // -- Interpolate values from current element in background mesh to swarm points
    PetscCall(PetscFECreateTabulation(fe, 1, num_points_in_cell, coords_points_cell_ref, 1, &tabulation));
    PetscCall(DMPlexVecGetClosure(dm_mesh, section_u_mesh_loc, U_loc, cell, NULL, &u_cell));
    PetscCall(PetscFEGetQuadrature(fe, &quadrature));
    PetscCall(PetscFECreateCellGeometry(fe, quadrature, &fe_geometry));
    for (PetscInt p = 0; p < num_points_in_cell; p++) {
      PetscCall(PetscFEInterpolateAtPoints_Static(fe, tabulation, u_cell, &fe_geometry, p, &u_points[points_cell[p] * num_comp]));
    }

    // -- Cleanup
    PetscCall(PetscFEDestroyCellGeometry(fe, &fe_geometry));
    PetscCall(DMPlexVecRestoreClosure(dm_mesh, section_u_mesh_loc, U_loc, cell, NULL, &u_cell));
    PetscCall(DMRestoreWorkArray(dm_mesh, num_points_in_cell * dim, MPIU_REAL, &coords_points_cell_true));
    PetscCall(DMRestoreWorkArray(dm_mesh, num_points_in_cell * dim, MPIU_REAL, &coords_points_cell_ref));
    PetscCall(PetscTabulationDestroy(&tabulation));
    PetscCall(PetscFree(points_cell));
  }

  // Cleanup
  PetscCall(DMSwarmRestoreField(dm_swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords_points));
  PetscCall(DMSwarmRestoreField(dm_swarm, field, NULL, NULL, (void **)&u_points));
  PetscCall(DMSwarmSortRestoreAccess(dm_swarm));
  PetscCall(DMRestoreLocalVector(dm_mesh, &U_loc));
  PetscCall(PetscSectionDestroy(&section_u_mesh_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ------------------------------------------------------------------------------------------------
// Projection via libCEED
// ------------------------------------------------------------------------------------------------
PetscErrorCode DMSwarmInterpolateFromCellToSwarm_Ceed(DM dm_swarm, const char *field, Vec U_mesh) {
  PetscInt           num_elem;
  PetscMemType       U_mem_type;
  DM                 dm_mesh;
  Vec                U_mesh_loc;
  DMSwarmCeedContext swarm_ceed_context;

  PetscFunctionBeginUser;
  // Get mesh DM
  PetscCall(DMSwarmGetCellDM(dm_swarm, &dm_mesh));
  PetscCall(DMGetApplicationContext(dm_mesh, (void *)&swarm_ceed_context));

  // Get mesh values
  PetscCall(DMGetLocalVector(dm_mesh, &U_mesh_loc));
  PetscCall(VecZeroEntries(U_mesh_loc));
  PetscCall(DMGlobalToLocal(dm_mesh, U_mesh, INSERT_VALUES, U_mesh_loc));
  PetscCall(VecReadP2C(U_mesh_loc, &U_mem_type, swarm_ceed_context->u_mesh_loc));

  // Get swarm access
  PetscCall(DMSwarmSortGetAccess(dm_swarm));
  PetscCall(DMSwarmPICFieldP2C(dm_swarm, field, swarm_ceed_context->u_points_loc));

  // Interpolate values to each swarm point, one element in the background mesh at a time
  CeedElemRestrictionGetNumElements(swarm_ceed_context->restriction_u_mesh, &num_elem);
  for (PetscInt e = 0; e < num_elem; e++) {
    PetscInt num_points_in_elem;

    CeedElemRestrictionGetNumPointsInElement(swarm_ceed_context->restriction_u_points, e, &num_points_in_elem);

    // -- Reference coordinates for swarm points in background mesh element
    CeedElemRestrictionApplyAtPointsInElement(swarm_ceed_context->restriction_x_points, e, CEED_NOTRANSPOSE, swarm_ceed_context->x_ref_points_loc,
                                              swarm_ceed_context->x_ref_points_elem, CEED_REQUEST_IMMEDIATE);

    // -- Interpolate values from current element in background mesh to swarm points
    // Note: This will only work for CPU backends at this time, as only CPU backends support ApplyBlock and ApplyAtPoints
    CeedElemRestrictionApplyBlock(swarm_ceed_context->restriction_u_mesh, e, CEED_NOTRANSPOSE, swarm_ceed_context->u_mesh_loc,
                                  swarm_ceed_context->u_mesh_elem, CEED_REQUEST_IMMEDIATE);
    CeedBasisApplyAtPoints(swarm_ceed_context->basis_u, num_points_in_elem, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, swarm_ceed_context->x_ref_points_elem,
                           swarm_ceed_context->u_mesh_elem, swarm_ceed_context->u_points_elem);

    // -- Insert result back into local vector
    CeedElemRestrictionApplyAtPointsInElement(swarm_ceed_context->restriction_u_points, e, CEED_TRANSPOSE, swarm_ceed_context->u_points_elem,
                                              swarm_ceed_context->u_points_loc, CEED_REQUEST_IMMEDIATE);
  }

  // Cleanup
  PetscCall(DMSwarmPICFieldC2P(dm_swarm, field, swarm_ceed_context->u_points_loc));
  PetscCall(DMSwarmSortRestoreAccess(dm_swarm));
  PetscCall(VecReadC2P(swarm_ceed_context->u_mesh_loc, U_mem_type, U_mesh_loc));
  PetscCall(DMRestoreLocalVector(dm_mesh, &U_mesh_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ------------------------------------------------------------------------------------------------
// Error checking utility
// ------------------------------------------------------------------------------------------------
PetscErrorCode DMSwarmCheckSwarmValues(DM dm_swarm, const char *field, PetscScalar tolerance, DMFunc TrueSolution) {
  PetscBool          within_tolerance = PETSC_TRUE;
  PetscInt           dim, num_comp, cell_start, cell_end;
  const PetscScalar *u_points, *coords_points;
  DM                 dm_mesh;

  PetscFunctionBeginUser;
  PetscCall(DMSwarmSortGetAccess(dm_swarm));
  PetscCall(DMSwarmGetCellDM(dm_swarm, &dm_mesh));
  PetscCall(DMGetDimension(dm_mesh, &dim));
  PetscCall(DMPlexGetHeightStratum(dm_mesh, 0, &cell_start, &cell_end));
  PetscCall(DMSwarmGetField(dm_swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords_points));
  PetscCall(DMSwarmGetField(dm_swarm, field, &num_comp, NULL, (void **)&u_points));

  // Interpolate values to each swarm point, one element in the background mesh at a time
  for (PetscInt cell = cell_start; cell < cell_end; cell++) {
    PetscInt *points;
    PetscInt  num_points_in_cell;

    PetscCall(DMSwarmSortGetPointsPerCell(dm_swarm, cell, &num_points_in_cell, &points));
    // -- Reference coordinates for swarm points in background mesh element
    for (PetscInt p = 0; p < num_points_in_cell; p++) {
      PetscScalar x[dim], u_true[num_comp];

      for (PetscInt d = 0; d < dim; d++) x[d] = coords_points[points[p] * dim + d];
      PetscCall(TrueSolution(dim, 0.0, x, num_comp, u_true, NULL));
      for (PetscInt i = 0; i < num_comp; i++) {
        if (PetscAbs(u_points[points[p] * num_comp + i] - u_true[i]) > tolerance) {
          within_tolerance = PETSC_FALSE;
          PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm_swarm),
                                "Incorrect interpolated value, cell %" PetscInt_FMT " point %" PetscInt_FMT " component %" PetscInt_FMT
                                ", found %f expected %f\n",
                                cell, p, i, u_points[points[p] * num_comp + i], u_true[i]));
        }
      }
    }

    // -- Cleanup
    PetscCall(PetscFree(points));
  }

  // Cleanup
  PetscCall(DMSwarmRestoreField(dm_swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords_points));
  PetscCall(DMSwarmRestoreField(dm_swarm, field, NULL, NULL, (void **)&u_points));
  PetscCall(DMSwarmSortRestoreAccess(dm_swarm));
  PetscCheck(within_tolerance, PetscObjectComm((PetscObject)dm_swarm), PETSC_ERR_USER, "Interpolation to swarm points not within tolerance");
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ------------------------------------------------------------------------------------------------
// RHS for Swarm to Mesh projection
// ------------------------------------------------------------------------------------------------
PetscErrorCode DMSwarmCreateProjectionRHS(DM dm_swarm, const char *field, Vec B_mesh) {
  PetscInt           num_elem;
  PetscMemType       B_mem_type;
  DM                 dm_mesh;
  Vec                B_mesh_loc;
  DMSwarmCeedContext swarm_ceed_context;

  PetscFunctionBeginUser;
  // Get mesh DM
  PetscCall(DMSwarmGetCellDM(dm_swarm, &dm_mesh));
  PetscCall(DMGetApplicationContext(dm_mesh, (void *)&swarm_ceed_context));

  // Get mesh values
  PetscCall(DMGetLocalVector(dm_mesh, &B_mesh_loc));
  PetscCall(VecZeroEntries(B_mesh_loc));
  PetscCall(VecP2C(B_mesh_loc, &B_mem_type, swarm_ceed_context->v_mesh_loc));

  // Get swarm access
  PetscCall(DMSwarmSortGetAccess(dm_swarm));
  PetscCall(DMSwarmPICFieldP2C(dm_swarm, field, swarm_ceed_context->u_points_loc));

  // Interpolate values to each swarm point, one element in the background mesh at a time
  CeedElemRestrictionGetNumElements(swarm_ceed_context->restriction_u_mesh, &num_elem);
  for (PetscInt e = 0; e < num_elem; e++) {
    PetscInt num_points_in_elem;

    CeedElemRestrictionGetNumPointsInElement(swarm_ceed_context->restriction_u_points, e, &num_points_in_elem);

    // -- Reference coordinates for swarm points in background mesh element
    CeedElemRestrictionApplyAtPointsInElement(swarm_ceed_context->restriction_x_points, e, CEED_NOTRANSPOSE, swarm_ceed_context->x_ref_points_loc,
                                              swarm_ceed_context->x_ref_points_elem, CEED_REQUEST_IMMEDIATE);

    // -- Interpolate values from current element in background mesh to swarm points
    // Note: This will only work for CPU backends at this time, as only CPU backends support ApplyBlock and ApplyAtPoints
    CeedElemRestrictionApplyAtPointsInElement(swarm_ceed_context->restriction_u_points, e, CEED_NOTRANSPOSE, swarm_ceed_context->u_points_loc,
                                              swarm_ceed_context->u_points_elem, CEED_REQUEST_IMMEDIATE);
    CeedBasisApplyAtPoints(swarm_ceed_context->basis_u, num_points_in_elem, CEED_TRANSPOSE, CEED_EVAL_INTERP, swarm_ceed_context->x_ref_points_elem,
                           swarm_ceed_context->u_points_elem, swarm_ceed_context->u_mesh_elem);

    // -- Insert result back into local vector
    CeedElemRestrictionApplyBlock(swarm_ceed_context->restriction_u_mesh, e, CEED_TRANSPOSE, swarm_ceed_context->u_mesh_elem,
                                  swarm_ceed_context->v_mesh_loc, CEED_REQUEST_IMMEDIATE);
  }

  // Restore PETSc Vecs and Local to Global
  PetscCall(VecC2P(swarm_ceed_context->v_mesh_loc, B_mem_type, B_mesh_loc));
  PetscCall(VecZeroEntries(B_mesh));
  PetscCall(DMLocalToGlobal(dm_mesh, B_mesh_loc, ADD_VALUES, B_mesh));

  // Cleanup
  PetscCall(DMSwarmPICFieldC2P(dm_swarm, field, swarm_ceed_context->u_points_loc));
  PetscCall(DMSwarmSortRestoreAccess(dm_swarm));
  PetscCall(DMRestoreLocalVector(dm_mesh, &B_mesh_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ------------------------------------------------------------------------------------------------
// Swarm "mass matrix"
// ------------------------------------------------------------------------------------------------
PetscErrorCode MatMult_SwarmMass(Mat A, Vec U_mesh, Vec V_mesh) {
  PetscInt           num_elem;
  PetscMemType       U_mem_type, V_mem_type;
  DM                 dm_mesh;
  Vec                U_mesh_loc, V_mesh_loc;
  DMSwarmCeedContext swarm_ceed_context;

  PetscFunctionBeginUser;
  // Get mesh DM
  PetscCall(MatGetDM(A, &dm_mesh));
  PetscCall(DMGetApplicationContext(dm_mesh, (void *)&swarm_ceed_context));

  // Global to Local and get PETSc Vec access
  PetscCall(DMGetLocalVector(dm_mesh, &U_mesh_loc));
  PetscCall(VecZeroEntries(U_mesh_loc));
  PetscCall(DMGlobalToLocal(dm_mesh, U_mesh, INSERT_VALUES, U_mesh_loc));
  PetscCall(VecReadP2C(U_mesh_loc, &U_mem_type, swarm_ceed_context->u_mesh_loc));
  PetscCall(DMGetLocalVector(dm_mesh, &V_mesh_loc));
  PetscCall(VecZeroEntries(V_mesh_loc));
  PetscCall(VecP2C(V_mesh_loc, &V_mem_type, swarm_ceed_context->v_mesh_loc));

  // Interpolate values to each swarm point, one element in the background mesh at a time
  CeedElemRestrictionGetNumElements(swarm_ceed_context->restriction_u_mesh, &num_elem);
  for (PetscInt e = 0; e < num_elem; e++) {
    PetscInt num_points_in_elem;

    CeedElemRestrictionGetNumPointsInElement(swarm_ceed_context->restriction_u_points, e, &num_points_in_elem);

    // -- Reference coordinates for swarm points in background mesh element
    CeedElemRestrictionApplyAtPointsInElement(swarm_ceed_context->restriction_x_points, e, CEED_NOTRANSPOSE, swarm_ceed_context->x_ref_points_loc,
                                              swarm_ceed_context->x_ref_points_elem, CEED_REQUEST_IMMEDIATE);

    // -- Interpolate values from current element in background mesh to swarm points
    // Note: This will only work for CPU backends at this time, as only CPU backends support ApplyBlock and ApplyAtPoints
    CeedElemRestrictionApplyBlock(swarm_ceed_context->restriction_u_mesh, e, CEED_NOTRANSPOSE, swarm_ceed_context->u_mesh_loc,
                                  swarm_ceed_context->u_mesh_elem, CEED_REQUEST_IMMEDIATE);
    CeedBasisApplyAtPoints(swarm_ceed_context->basis_u, num_points_in_elem, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, swarm_ceed_context->x_ref_points_elem,
                           swarm_ceed_context->u_mesh_elem, swarm_ceed_context->u_points_elem);

    // -- Interpolate transpose back from swarm points to mesh
    CeedBasisApplyAtPoints(swarm_ceed_context->basis_u, num_points_in_elem, CEED_TRANSPOSE, CEED_EVAL_INTERP, swarm_ceed_context->x_ref_points_elem,
                           swarm_ceed_context->u_points_elem, swarm_ceed_context->u_mesh_elem);
    CeedElemRestrictionApplyBlock(swarm_ceed_context->restriction_u_mesh, e, CEED_TRANSPOSE, swarm_ceed_context->u_mesh_elem,
                                  swarm_ceed_context->v_mesh_loc, CEED_REQUEST_IMMEDIATE);
  }

  // Restore PETSc Vecs and Local to Global
  PetscCall(VecReadC2P(swarm_ceed_context->u_mesh_loc, U_mem_type, U_mesh_loc));
  PetscCall(VecC2P(swarm_ceed_context->v_mesh_loc, V_mem_type, V_mesh_loc));
  PetscCall(VecZeroEntries(V_mesh));
  PetscCall(DMLocalToGlobal(dm_mesh, V_mesh_loc, ADD_VALUES, V_mesh));

  // Cleanup
  PetscCall(DMRestoreLocalVector(dm_mesh, &U_mesh_loc));
  PetscCall(DMRestoreLocalVector(dm_mesh, &V_mesh_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}
