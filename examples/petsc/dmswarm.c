// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
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
//  ./dmswarm -dm_plex_dim 3 -dm_plex_box_faces 3,3,3 -dm_plex_box_lower -1.0,-1.0,-1.0 -dm_plex_simplex 0 -num_comp 2 -swarm gauss
//
//TESTARGS(name="Uniform swarm, CG projection") -ceed {ceed_resource} -test -dm_plex_dim 3 -dm_plex_box_faces 3,3,3 -dm_plex_box_lower -1.0,-1.0,-1.0 -dm_plex_simplex 0 -dm_plex_hash_location true -num_comp 2 -swarm uniform -solution_order 3 -points_per_cell 125
//TESTARGS(name="Gauss swarm, lumped projection") -ceed {ceed_resource} -test -dm_plex_dim 3 -dm_plex_box_faces 3,3,3 -dm_plex_box_lower -1.0,-1.0,-1.0 -dm_plex_simplex 0 -dm_plex_hash_location true -num_comp 2 -swarm gauss -ksp_type preonly -pc_type jacobi -pc_jacobi_type rowsum -tolerance 9e-2

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
#include "include/petscversion.h"
#include "include/swarmutils.h"

const char DMSwarmPICField_u[] = "u";

// Target functions
typedef enum { TARGET_TANH = 0, TARGET_POLYNOMIAL = 1, TARGET_SPHERE = 2 } TargetType;
static const char *const target_types[] = {"tanh", "polynomial", "sphere", "TargetType", "TARGET", 0};

typedef PetscErrorCode (*TargetFunc)(PetscInt dim, const PetscScalar x[]);
typedef PetscErrorCode (*TargetFuncProj)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx);

PetscScalar    EvalU_Tanh(PetscInt dim, const PetscScalar x[]);
PetscScalar    EvalU_Poly(PetscInt dim, const PetscScalar x[]);
PetscScalar    EvalU_Sphere(PetscInt dim, const PetscScalar x[]);
PetscErrorCode EvalU_Tanh_proj(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt num_comp, PetscScalar *u, void *ctx);
PetscErrorCode EvalU_Poly_proj(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt num_comp, PetscScalar *u, void *ctx);
PetscErrorCode EvalU_Sphere_proj(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt num_comp, PetscScalar *u, void *ctx);

// Swarm to mesh and mesh to swarm
PetscErrorCode DMSwarmInterpolateFromCellToSwarm_Petsc(DM dm_swarm, const char *field, Vec U_mesh);
PetscErrorCode DMSwarmInterpolateFromCellToSwarm_Ceed(DM dm_swarm, const char *field, Vec U_mesh);
PetscErrorCode DMSwarmCheckSwarmValues(DM dm_swarm, const char *field, PetscScalar tolerance, TargetFuncProj TrueSolution);

// ------------------------------------------------------------------------------------------------
// main driver
// ------------------------------------------------------------------------------------------------
int main(int argc, char **argv) {
  MPI_Comm           comm;
  char               ceed_resource[PETSC_MAX_PATH_LEN] = "/cpu/self";
  PetscBool          test_mode = PETSC_FALSE, view_petsc_swarm = PETSC_FALSE, view_ceed_swarm = PETSC_FALSE;
  PetscInt           dim = 3, num_comp = 1, num_points = 1728, num_points_per_cell = 64, mesh_order = 1, solution_order = 2, q_extra = 3;
  PetscScalar        tolerance = 1E-3;
  DM                 dm_mesh, dm_swarm;
  Vec                U_mesh;
  DMSwarmCeedContext swarm_ceed_context;
  PointSwarmType     point_swarm_type     = SWARM_UNIFORM;
  TargetType         target_type          = TARGET_TANH;
  TargetFuncProj     target_function_proj = EvalU_Tanh_proj;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  // Read command line options
  PetscOptionsBegin(comm, NULL, "libCEED example using PETSc with DMSwarm", NULL);

  PetscCall(PetscOptionsBool("-test", "Testing mode (do not print unless error is large)", NULL, test_mode, &test_mode, NULL));
  PetscCall(
      PetscOptionsBool("-u_petsc_swarm_view", "View XDMF of swarm values interpolated by PETSc", NULL, view_petsc_swarm, &view_petsc_swarm, NULL));
  PetscCall(
      PetscOptionsBool("-u_ceed_swarm_view", "View XDMF of swarm values interpolated by libCEED", NULL, view_ceed_swarm, &view_ceed_swarm, NULL));
  PetscCall(PetscOptionsEnum("-target", "Target field function", NULL, target_types, (PetscEnum)target_type, (PetscEnum *)&target_type, NULL));
  PetscCall(PetscOptionsInt("-solution_order", "Order of mesh solution space", NULL, solution_order, &solution_order, NULL));
  PetscCall(PetscOptionsInt("-mesh_order", "Order of mesh coordinate space", NULL, mesh_order, &mesh_order, NULL));
  PetscCall(PetscOptionsInt("-q_extra", "Number of extra quadrature points", NULL, q_extra, &q_extra, NULL));
  PetscCall(PetscOptionsInt("-num_comp", "Number of components in solution", NULL, num_comp, &num_comp, NULL));
  PetscCall(PetscOptionsEnum("-swarm", "Swarm points distribution", NULL, point_swarm_types, (PetscEnum)point_swarm_type,
                             (PetscEnum *)&point_swarm_type, NULL));
  {
    PetscBool user_set_num_points_per_cell = PETSC_FALSE;
    PetscInt  dim = 3, num_cells_total = 1;
    PetscInt  num_cells[] = {1, 1, 1};

    PetscCall(PetscOptionsInt("-points_per_cell", "Total number of swarm points in each cell", NULL, num_points_per_cell, &num_points_per_cell,
                              &user_set_num_points_per_cell));
    PetscCall(PetscOptionsInt("-dm_plex_dim", "Background mesh dimension", NULL, dim, &dim, NULL));
    PetscCall(PetscOptionsIntArray("-dm_plex_box_faces", "Number of cells", NULL, num_cells, &dim, NULL));

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
      PetscCall(DMSetCoordinateDisc(dm_mesh, fe_coord, PETSC_TRUE));
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
    PetscCall(DMSwarmInitalizePointLocations(dm_swarm, point_swarm_type, num_points, num_points_per_cell));

    // -- Final particle swarm
    PetscCall(PetscObjectSetName((PetscObject)dm_swarm, "Particle Swarm"));
    PetscCall(DMViewFromOptions(dm_swarm, NULL, "-dm_swarm_view"));
  }

  // Set field values on background mesh
  PetscCall(DMCreateGlobalVector(dm_mesh, &U_mesh));
  switch (target_type) {
    case TARGET_TANH:
      target_function_proj = EvalU_Tanh_proj;
      break;
    case TARGET_POLYNOMIAL:
      target_function_proj = EvalU_Poly_proj;
      break;
    case TARGET_SPHERE:
      target_function_proj = EvalU_Sphere_proj;
      break;
  }
  {
    TargetFuncProj mesh_solution[1] = {target_function_proj};

    PetscCall(DMProjectFunction(dm_mesh, 0.0, mesh_solution, NULL, INSERT_VALUES, U_mesh));
  }

  // Visualize background mesh
  PetscCall(PetscObjectSetName((PetscObject)U_mesh, "U on Background Mesh"));
  PetscCall(VecViewFromOptions(U_mesh, NULL, "-u_mesh_view"));

  // libCEED objects for swarm and background mesh
  PetscCall(DMSwarmCeedContextCreate(dm_swarm, ceed_resource, &swarm_ceed_context));

  // Interpolate from mesh to points via PETSc
  {
    PetscCall(DMSwarmInterpolateFromCellToSwarm_Petsc(dm_swarm, DMSwarmPICField_u, U_mesh));
    if (view_petsc_swarm) PetscCall(DMSwarmViewXDMF(dm_swarm, "swarm_petsc.xmf"));
    PetscCall(DMSwarmCheckSwarmValues(dm_swarm, DMSwarmPICField_u, tolerance, target_function_proj));
  }

  // Interpolate from mesh to points via libCEED
  {
    PetscCall(DMSwarmInterpolateFromCellToSwarm_Ceed(dm_swarm, DMSwarmPICField_u, U_mesh));
    if (view_ceed_swarm) PetscCall(DMSwarmViewXDMF(dm_swarm, "swarm_ceed.xmf"));
    PetscCall(DMSwarmCheckSwarmValues(dm_swarm, DMSwarmPICField_u, tolerance, target_function_proj));
  }

  // Project from points to mesh via libCEED
  {
    Vec U_projected;

    PetscCall(VecDuplicate(U_mesh, &U_projected));
    PetscCall(DMSwarmProjectFromSwarmToCells(dm_swarm, DMSwarmPICField_u, NULL, U_projected));

    PetscCall(PetscObjectSetName((PetscObject)U_projected, "U projected to Background Mesh"));
    PetscCall(VecViewFromOptions(U_projected, NULL, "-u_projected_view"));
    PetscCall(VecAXPY(U_projected, -1.0, U_mesh));
    PetscCall(PetscObjectSetName((PetscObject)U_projected, "U Projection Error"));
    PetscCall(VecViewFromOptions(U_projected, NULL, "-u_error_view"));

    // -- Check error
    {
      PetscScalar error, norm_u_mesh;

      PetscCall(VecNorm(U_projected, NORM_2, &error));
      PetscCall(VecNorm(U_mesh, NORM_2, &norm_u_mesh));
      PetscCheck(error / norm_u_mesh < tolerance, comm, PETSC_ERR_USER, "Projection error too high: %e\n", error / norm_u_mesh);
      if (!test_mode) PetscCall(PetscPrintf(comm, "  Projection error: %e\n", error / norm_u_mesh));
    }

    PetscCall(VecDestroy(&U_projected));
  }
  // Cleanup
  PetscCall(DMSwarmCeedContextDestroy(&swarm_ceed_context));
  PetscCall(DMDestroy(&dm_swarm));
  PetscCall(DMDestroy(&dm_mesh));
  PetscCall(VecDestroy(&U_mesh));
  return PetscFinalize();
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

PetscScalar EvalU_Sphere(PetscInt dim, const PetscScalar x[]) {
  PetscScalar distance = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);

  return distance < 1.0 ? 1.0 : 0.1;
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

PetscErrorCode EvalU_Sphere_proj(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt num_comp, PetscScalar *u, void *ctx) {
  PetscFunctionBeginUser;
  const PetscScalar f_x = EvalU_Sphere(dim, x);

  for (PetscInt c = 0; c < num_comp; c++) u[c] = (c + 1.0) * f_x;
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
  PetscCall(VecReadP2C(U_mesh_loc, &U_mem_type, swarm_ceed_context->u_mesh));

  // Get swarm access
  PetscCall(DMSwarmSortGetAccess(dm_swarm));
  PetscCall(DMSwarmPICFieldP2C(dm_swarm, field, swarm_ceed_context->u_points));

  // Interpolate field from mesh to swarm points
  CeedOperatorApply(swarm_ceed_context->op_mesh_to_points, swarm_ceed_context->u_mesh, swarm_ceed_context->u_points, CEED_REQUEST_IMMEDIATE);

  // Cleanup
  PetscCall(DMSwarmPICFieldC2P(dm_swarm, field, swarm_ceed_context->u_points));
  PetscCall(DMSwarmSortRestoreAccess(dm_swarm));
  PetscCall(VecReadC2P(swarm_ceed_context->u_mesh, U_mem_type, U_mesh_loc));
  PetscCall(DMRestoreLocalVector(dm_mesh, &U_mesh_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ------------------------------------------------------------------------------------------------
// Error checking utility
// ------------------------------------------------------------------------------------------------
PetscErrorCode DMSwarmCheckSwarmValues(DM dm_swarm, const char *field, PetscScalar tolerance, TargetFuncProj TrueSolution) {
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
