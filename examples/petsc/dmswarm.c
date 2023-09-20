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
//     ./dmswarm
///
//TESTARGS -ceed {ceed_resource} -test -tolerance 1e-3 -dm_plex_dim 3 -dm_plex_box_faces 3,3,3 -dm_plex_box_lower -1.0,-1.0,-1.0 -dm_plex_simplex 0 -dm_plex_hash_location true -uniform_swarm -num_comp 4
//TESTARGS -ceed {ceed_resource} -test -tolerance 1e-3 -dm_plex_dim 3 -dm_plex_box_faces 3,3,3 -dm_plex_box_lower -1.0,-1.0,-1.0 -dm_plex_simplex 0 -dm_plex_hash_location true -num_comp 2

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
  CeedElemRestriction restriction_u_mesh, restriction_x_points, restriction_u_points;
  CeedBasis           basis;
};

const char DMSwarmPICField_u[] = "u";

PetscErrorCode DMSwarmCeedContextCreate(const char *ceed_resource, DMSwarmCeedContext *ctx);
PetscErrorCode DMSwarmCeedContextDestroy(DMSwarmCeedContext *ctx);

PetscErrorCode VecReadP2C(Vec X_petsc, PetscMemType *mem_type, CeedVector x_ceed);
PetscErrorCode VecReadC2P(CeedVector x_ceed, PetscMemType mem_type, Vec X_petsc);
PetscErrorCode DMSwarmPICFieldP2C(DM dm_swarm, const char *field, CeedVector x_ceed);
PetscErrorCode DMSwarmPICFieldC2P(DM dm_swarm, const char *field, CeedVector x_ceed);

PetscScalar    EvalU(PetscInt dim, const PetscScalar x[]);
PetscErrorCode EvalU_proj(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt num_comp, PetscScalar *u, void *ctx);

PetscErrorCode DMSwarmCreateReferenceCoordinates(DM dm_swarm, IS *is_points, Vec *ref_coords);
PetscErrorCode DMSwarmInterpolateFromCellToSwarm_Petsc(DM dm_swarm, const char *field, Vec U_mesh);
PetscErrorCode DMSwarmInterpolateFromCellToSwarm_Ceed(DM dm_swarm, const char *field, Vec U_mesh);
PetscErrorCode DMSwarmCheckSwarmValues(DM dm_swarm, const char *field, PetscScalar tolerance, DMFunc TrueSolution);

int main(int argc, char **argv) {
  MPI_Comm           comm;
  char               ceed_resource[PETSC_MAX_PATH_LEN] = "/cpu/self";
  PetscBool          test_mode = PETSC_FALSE, set_uniform_swarm = PETSC_FALSE;
  PetscInt           dim = 3, num_comp = 1, num_points = 200, geometry_order = 1, mesh_order = 3, q_extra = 3;
  PetscScalar        tolerance = 1E-3;
  DM                 dm_mesh, dm_swarm;
  Vec                U_mesh;
  DMSwarmCeedContext swarm_ceed_context;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  // Read command line options
  PetscOptionsBegin(comm, NULL, "libCEED example using PETSc with DMSwarm", NULL);

  PetscCall(PetscOptionsBool("-test", "Testing mode (do not print unless error is large)", NULL, test_mode, &test_mode, NULL));
  PetscCall(PetscOptionsBool("-uniform_swarm", "Use uniform coordinates in swarm", NULL, set_uniform_swarm, &set_uniform_swarm, NULL));
  PetscCall(PetscOptionsInt("-order", "Order of mesh solution space", NULL, mesh_order, &mesh_order, NULL));
  PetscCall(PetscOptionsInt("-mesh_order", "Order of mesh coordinate space", NULL, geometry_order, &geometry_order, NULL));
  PetscCall(PetscOptionsInt("-q_extra", "Number of extra quadrature points", NULL, q_extra, &q_extra, NULL));
  PetscCall(PetscOptionsInt("-num_comp", "Number of components in solution", NULL, num_comp, &num_comp, NULL));
  PetscCall(PetscOptionsInt("-points", "Number of swarm points", NULL, num_points, &num_points, NULL));
  PetscCall(PetscOptionsScalar("-tolerance", "Absolute tolerance for swarm point values", NULL, tolerance, &tolerance, NULL));
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
      PetscCall(PetscFECreateLagrange(comm, dim, num_comp, PETSC_FALSE, mesh_order, mesh_order + q_extra, &fe));
      PetscCall(DMAddField(dm_mesh, NULL, (PetscObject)fe));
      PetscCall(PetscFEDestroy(&fe));
    }
    PetscCall(DMCreateDS(dm_mesh));

    // -- Coordinate FE space
    {
      PetscFE fe_coord;

      PetscCall(PetscFECreateLagrange(comm, dim, dim, PETSC_FALSE, geometry_order, mesh_order + q_extra, &fe_coord));
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

  // -- libCEED objects from background mesh
  PetscCall(DMSwarmCeedContextCreate(ceed_resource, &swarm_ceed_context));
  {
    BPData bp_data = {.q_mode = CEED_GAUSS};

    PetscCall(CreateBasisFromPlex(swarm_ceed_context->ceed, dm_mesh, NULL, 0, 0, 0, bp_data, &swarm_ceed_context->basis));
    PetscCall(CreateRestrictionFromPlex(swarm_ceed_context->ceed, dm_mesh, 0, NULL, 0, &swarm_ceed_context->restriction_u_mesh));
    PetscCall(DMSetApplicationContext(dm_mesh, (void *)swarm_ceed_context));
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
    if (set_uniform_swarm) {
      // ---- Set uniform point locations in each cell
      PetscInt dim_cells   = dim;
      PetscInt num_cells[] = {1, 1, 1};

      PetscOptionsBegin(comm, NULL, "libCEED example using PETSc with DMSwarm", NULL);
      PetscCall(PetscOptionsIntArray("-dm_plex_box_faces", "Number of cells", NULL, num_cells, &dim_cells, NULL));
      PetscOptionsEnd();

      PetscInt total_num_cells     = num_cells[0] * num_cells[1] * num_cells[2];
      PetscInt points_per_cell     = PetscCeilInt(num_points, total_num_cells);
      PetscInt points_per_cell_dim = ceil(cbrt(points_per_cell * 1.0));
      points_per_cell              = 1;
      for (PetscInt i = 0; i < dim; i++) points_per_cell *= points_per_cell_dim;

      PetscInt    num_points[] = {points_per_cell_dim, points_per_cell_dim, points_per_cell_dim};
      PetscScalar point_coords[points_per_cell * 3];

      for (PetscInt i = 0; i < num_points[0]; i++) {
        for (PetscInt j = 0; j < num_points[1]; j++) {
          for (PetscInt k = 0; k < num_points[2]; k++) {
            PetscInt p = (i * num_points[1] + j) * num_points[2] + k;

            point_coords[p * dim + 0] = 2.0 * (PetscReal)(i + 1) / (PetscReal)(num_points[0] + 1) - 1;
            point_coords[p * dim + 1] = 2.0 * (PetscReal)(j + 1) / (PetscReal)(num_points[1] + 1) - 1;
            point_coords[p * dim + 2] = 2.0 * (PetscReal)(k + 1) / (PetscReal)(num_points[2] + 1) - 1;
          }
        }
      }
      PetscCall(DMSwarmSetPointCoordinatesCellwise(dm_swarm, num_points[0] * num_points[1] * num_points[2], point_coords));
    } else {
      // ---- Set points distributed per sinusoidal functions
      PetscScalar *point_coords;

      PetscCall(DMSwarmGetField(dm_swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&point_coords));
      for (PetscInt p = 0; p < num_points; p++) {
        point_coords[p * dim + 0] = -PetscCosReal((PetscReal)(p + 1) / (PetscReal)(num_points + 1) * PETSC_PI);
        if (dim > 1) point_coords[p * dim + 1] = -PetscSinReal((PetscReal)(p + 1) / (PetscReal)(num_points + 1) * PETSC_PI);
        if (dim > 2) point_coords[p * dim + 2] = PetscSinReal((PetscReal)(p + 1) / (PetscReal)(num_points + 1) * PETSC_PI);
      }
      PetscCall(DMSwarmRestoreField(dm_swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&point_coords));
    }
    PetscCall(DMSwarmMigrate(dm_swarm, PETSC_TRUE));

    // -- Final particle swarm
    PetscCall(PetscObjectSetName((PetscObject)dm_swarm, "Particle Swarm"));
    PetscCall(DMViewFromOptions(dm_swarm, NULL, "-dm_swarm_view"));
  }

  // Set field values on background mesh
  PetscCall(DMCreateGlobalVector(dm_mesh, &U_mesh));
  {
    DMFunc mesh_solution[1] = {EvalU_proj};

    PetscCall(DMProjectFunction(dm_mesh, 0.0, mesh_solution, NULL, INSERT_VALUES, U_mesh));
  }

  // Visualize background mesh
  PetscCall(VecViewFromOptions(U_mesh, NULL, "-u_mesh_view"));

  // Interpolate from mesh to points via PETSc
  {
    PetscCall(DMSwarmInterpolateFromCellToSwarm_Petsc(dm_swarm, DMSwarmPICField_u, U_mesh));
    if (!test_mode) PetscCall(DMSwarmViewXDMF(dm_swarm, "swarm_petsc.xmf"));
    PetscCall(DMSwarmCheckSwarmValues(dm_swarm, DMSwarmPICField_u, tolerance, EvalU_proj));
  }

  // Interpolate from mesh to points via libCEED
  {
    PetscCall(DMSwarmInterpolateFromCellToSwarm_Ceed(dm_swarm, DMSwarmPICField_u, U_mesh));
    if (!test_mode) PetscCall(DMSwarmViewXDMF(dm_swarm, "swarm_ceed.xmf"));
    PetscCall(DMSwarmCheckSwarmValues(dm_swarm, DMSwarmPICField_u, tolerance, EvalU_proj));
  }

  // Cleanup
  PetscCall(DMSwarmCeedContextDestroy(&swarm_ceed_context));
  PetscCall(DMDestroy(&dm_swarm));
  PetscCall(DMDestroy(&dm_mesh));
  PetscCall(VecDestroy(&U_mesh));
  return PetscFinalize();
}

// Context utilities
PetscErrorCode DMSwarmCeedContextCreate(const char *ceed_resource, DMSwarmCeedContext *ctx) {
  PetscFunctionBeginUser;
  PetscCall(PetscNew(ctx));
  CeedInit(ceed_resource, &(*ctx)->ceed);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmCeedContextDestroy(DMSwarmCeedContext *ctx) {
  PetscFunctionBeginUser;
  CeedDestroy(&(*ctx)->ceed);
  CeedElemRestrictionDestroy(&(*ctx)->restriction_u_mesh);
  CeedElemRestrictionDestroy(&(*ctx)->restriction_x_points);
  CeedElemRestrictionDestroy(&(*ctx)->restriction_u_points);
  CeedBasisDestroy(&(*ctx)->basis);
  PetscCall(PetscFree(*ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// PETSc-libCEED memory space utilities
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

// Solution functions
PetscScalar EvalU(PetscInt dim, const PetscScalar x[]) {
  PetscScalar result = 1, center = 0.1;

  for (PetscInt d = 0; d < dim; d++) {
    result *= tanh(x[d] - center);
    center += 0.1;
  }
  return result;
}

PetscErrorCode EvalU_proj(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt num_comp, PetscScalar *u, void *ctx) {
  PetscFunctionBeginUser;
  for (PetscInt c = 0; c < num_comp; c++) u[c] = (c + 1.0) * EvalU(dim, x);
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

// Projection via PETSc
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

// Projection via libCEED
PetscErrorCode DMSwarmInterpolateFromCellToSwarm_Ceed(DM dm_swarm, const char *field, Vec U_mesh) {
  PetscInt           dim, num_elem, num_comp;
  PetscMemType       U_mem_type, X_mem_type;
  DM                 dm_mesh;
  IS                 is_points;
  Vec                U_loc, X_ref;
  CeedVector         u_l_vec, u_l_vec_points, u_cell, u_points_cell, x_l_vec_points, x_points_cell;
  DMSwarmCeedContext swarm_ceed_context;

  PetscFunctionBeginUser;
  // Get mesh DM
  PetscCall(DMSwarmGetCellDM(dm_swarm, &dm_mesh));
  PetscCall(DMGetDimension(dm_mesh, &dim));
  PetscCall(DMGetApplicationContext(dm_mesh, (void *)&swarm_ceed_context));

  // Get mesh values
  {
    PetscCall(DMGetLocalVector(dm_mesh, &U_loc));
    PetscCall(VecZeroEntries(U_loc));
    PetscCall(DMGlobalToLocal(dm_mesh, U_mesh, INSERT_VALUES, U_loc));

    CeedElemRestrictionCreateVector(swarm_ceed_context->restriction_u_mesh, &u_l_vec, NULL);
    PetscCall(VecReadP2C(U_loc, &U_mem_type, u_l_vec));
  }
  {
    CeedInt elem_size;

    CeedElemRestrictionGetElementSize(swarm_ceed_context->restriction_u_mesh, &elem_size);
    CeedElemRestrictionGetNumComponents(swarm_ceed_context->restriction_u_mesh, &num_comp);
    CeedVectorCreate(swarm_ceed_context->ceed, elem_size * num_comp, &u_cell);
  }

  // Setup current swarm restriction;
  PetscCall(DMSwarmCreateReferenceCoordinates(dm_swarm, &is_points, &X_ref));
  CeedElemRestrictionGetNumElements(swarm_ceed_context->restriction_u_mesh, &num_elem);
  {
    const PetscInt *cell_points;

    PetscCall(ISGetIndices(is_points, &cell_points));
    PetscInt num_points = cell_points[num_elem + 1] - num_elem - 2;
    CeedInt  offsets[num_elem + 1 + num_points];

    for (PetscInt i = 0; i < num_elem + 1; i++) offsets[i] = cell_points[i + 1] - 1;
    for (PetscInt i = num_elem + 1; i < num_points + num_elem + 1; i++) offsets[i] = cell_points[i + 1];
    PetscCall(ISRestoreIndices(is_points, &cell_points));

    CeedElemRestrictionCreateAtPoints(swarm_ceed_context->ceed, num_elem, num_points, num_comp, num_points * num_comp, CEED_MEM_HOST,
                                      CEED_COPY_VALUES, offsets, &swarm_ceed_context->restriction_u_points);
    CeedElemRestrictionCreateAtPoints(swarm_ceed_context->ceed, num_elem, num_points, dim, num_points * dim, CEED_MEM_HOST, CEED_COPY_VALUES, offsets,
                                      &swarm_ceed_context->restriction_x_points);
  }

  // Setup libCEED swarm vectors
  {
    CeedInt max_points_in_cell;

    CeedElemRestrictionGetMaxPointsInElement(swarm_ceed_context->restriction_u_points, &max_points_in_cell);

    // -- U vector
    CeedElemRestrictionCreateVector(swarm_ceed_context->restriction_u_points, &u_l_vec_points, NULL);
    CeedVectorCreate(swarm_ceed_context->ceed, max_points_in_cell * num_comp, &u_points_cell);
    PetscCall(DMSwarmPICFieldP2C(dm_swarm, field, u_l_vec_points));
    CeedVectorSetValue(u_l_vec_points, 0.0);

    // -- X vector
    CeedElemRestrictionCreateVector(swarm_ceed_context->restriction_x_points, &x_l_vec_points, NULL);
    CeedVectorCreate(swarm_ceed_context->ceed, max_points_in_cell * dim, &x_points_cell);
    PetscCall(VecReadP2C(X_ref, &X_mem_type, x_l_vec_points));
  }

  // Interpolate values to each swarm point, one element in the background mesh at a time
  for (PetscInt e = 0; e < num_elem; e++) {
    PetscInt num_points_in_elem;

    CeedElemRestrictionGetNumPointsInElement(swarm_ceed_context->restriction_u_points, e, &num_points_in_elem);

    // -- Reference coordinates for swarm points in background mesh element
    CeedElemRestrictionApplyAtPointsInElement(swarm_ceed_context->restriction_x_points, e, CEED_NOTRANSPOSE, x_l_vec_points, x_points_cell,
                                              CEED_REQUEST_IMMEDIATE);

    // -- Interpolate values from current element in background mesh to swarm points
    // Note: This will only work for CPU backends at this time, as only CPU backends support ApplyBlock and ApplyAtPoints
    CeedElemRestrictionApplyBlock(swarm_ceed_context->restriction_u_mesh, e, CEED_NOTRANSPOSE, u_l_vec, u_cell, CEED_REQUEST_IMMEDIATE);
    CeedBasisApplyAtPoints(swarm_ceed_context->basis, num_points_in_elem, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_points_cell, u_cell, u_points_cell);

    // -- Insert result back into local vector
    CeedElemRestrictionApplyAtPointsInElement(swarm_ceed_context->restriction_u_points, e, CEED_TRANSPOSE, u_points_cell, u_l_vec_points,
                                              CEED_REQUEST_IMMEDIATE);
  }

  // Cleanup
  PetscCall(ISDestroy(&is_points));
  PetscCall(DMSwarmPICFieldC2P(dm_swarm, field, u_l_vec_points));
  PetscCall(VecReadC2P(x_l_vec_points, X_mem_type, X_ref));
  PetscCall(VecDestroy(&X_ref));
  PetscCall(VecReadC2P(u_l_vec, U_mem_type, U_loc));
  PetscCall(DMRestoreLocalVector(dm_mesh, &U_loc));
  CeedVectorDestroy(&u_l_vec);
  CeedVectorDestroy(&u_l_vec_points);
  CeedVectorDestroy(&u_cell);
  CeedVectorDestroy(&u_points_cell);
  CeedVectorDestroy(&x_l_vec_points);
  CeedVectorDestroy(&x_points_cell);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Error checking utility
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
          PetscPrintf(PetscObjectComm((PetscObject)dm_swarm),
                      "Incorrect interpolated value, cell %" PetscInt_FMT " point %" PetscInt_FMT " component %" PetscInt_FMT
                      ", found %f expected %f\n",
                      cell, p, i, u_points[points[p] * num_comp + i], u_true[i]);
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
