// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../include/swarmutils.h"
#include "../include/matops.h"
#include "../qfunctions/swarm/swarmmass.h"

// ------------------------------------------------------------------------------------------------
// Context utilities
// ------------------------------------------------------------------------------------------------
PetscErrorCode DMSwarmCeedContextCreate(DM dm_swarm, const char *ceed_resource, DMSwarmCeedContext *ctx) {
  DM                  dm_mesh, dm_coord;
  CeedElemRestriction elem_restr_u_mesh, elem_restr_x_mesh, elem_restr_x_points, elem_restr_u_points, elem_restr_q_data_points;
  CeedBasis           basis_u, basis_x;
  CeedVector          x_ref_points, q_data_points;
  CeedInt             num_comp;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(ctx));
  PetscCall(DMSwarmGetCellDM(dm_swarm, &dm_mesh));
  PetscCall(DMGetCoordinateDM(dm_mesh, &dm_coord));

  CeedInit(ceed_resource, &(*ctx)->ceed);
  // Background mesh objects
  {
    BPData bp_data = {.q_mode = CEED_GAUSS};

    PetscCall(CreateBasisFromPlex((*ctx)->ceed, dm_mesh, NULL, 0, 0, 0, bp_data, &basis_u));
    PetscCall(CreateBasisFromPlex((*ctx)->ceed, dm_coord, NULL, 0, 0, 0, bp_data, &basis_x));
    PetscCall(CreateRestrictionFromPlex((*ctx)->ceed, dm_mesh, 0, NULL, 0, &elem_restr_u_mesh));
    PetscCall(CreateRestrictionFromPlex((*ctx)->ceed, dm_coord, 0, NULL, 0, &elem_restr_x_mesh));

    // -- Mesh vectors
    CeedElemRestrictionCreateVector(elem_restr_u_mesh, &(*ctx)->u_mesh, NULL);
    CeedElemRestrictionCreateVector(elem_restr_u_mesh, &(*ctx)->v_mesh, NULL);
  }
  // Swarm objects
  {
    PetscInt        dim;
    const PetscInt *cell_points;
    IS              is_points;
    Vec             X_ref;
    CeedInt         num_elem;

    PetscCall(DMSwarmCreateReferenceCoordinates(dm_swarm, &is_points, &X_ref));
    PetscCall(DMGetDimension(dm_mesh, &dim));
    CeedElemRestrictionGetNumElements(elem_restr_u_mesh, &num_elem);
    CeedElemRestrictionGetNumComponents(elem_restr_u_mesh, &num_comp);

    PetscCall(ISGetIndices(is_points, &cell_points));
    PetscInt num_points = cell_points[num_elem + 1] - num_elem - 2;
    CeedInt  offsets[num_elem + 1 + num_points];

    for (PetscInt i = 0; i < num_elem + 1; i++) offsets[i] = cell_points[i + 1] - 1;
    for (PetscInt i = num_elem + 1; i < num_points + num_elem + 1; i++) offsets[i] = cell_points[i + 1];
    PetscCall(ISRestoreIndices(is_points, &cell_points));

    // -- Points restrictions
    CeedElemRestrictionCreateAtPoints((*ctx)->ceed, num_elem, num_points, num_comp, num_points * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES, offsets,
                                      &elem_restr_u_points);
    CeedElemRestrictionCreateAtPoints((*ctx)->ceed, num_elem, num_points, dim, num_points * dim, CEED_MEM_HOST, CEED_COPY_VALUES, offsets,
                                      &elem_restr_x_points);
    CeedElemRestrictionCreateAtPoints((*ctx)->ceed, num_elem, num_points, 1, num_points, CEED_MEM_HOST, CEED_COPY_VALUES, offsets,
                                      &elem_restr_q_data_points);

    // -- Points vectors
    CeedElemRestrictionCreateVector(elem_restr_u_points, &(*ctx)->u_points, NULL);
    CeedElemRestrictionCreateVector(elem_restr_q_data_points, &q_data_points, NULL);

    // -- Ref coordinates
    {
      PetscMemType       X_mem_type;
      const PetscScalar *x;

      CeedVectorCreate((*ctx)->ceed, num_points * dim, &x_ref_points);

      PetscCall(VecGetArrayReadAndMemType(X_ref, (const PetscScalar **)&x, &X_mem_type));
      CeedVectorSetArray(x_ref_points, MemTypeP2C(X_mem_type), CEED_COPY_VALUES, (CeedScalar *)x);
      PetscCall(VecRestoreArrayReadAndMemType(X_ref, (const PetscScalar **)&x));
    }

    // Create Q data
    {
      CeedQFunction qf_setup;
      CeedOperator  op_setup;
      CeedVector    x_coord;

      {
        Vec                X_loc;
        CeedInt            len;
        const PetscScalar *x;

        PetscCall(DMGetCoordinatesLocal(dm_mesh, &X_loc));
        PetscCall(VecGetLocalSize(X_loc, &len));
        CeedVectorCreate((*ctx)->ceed, len, &x_coord);

        PetscCall(VecGetArrayRead(X_loc, &x));
        CeedVectorSetArray(x_coord, CEED_MEM_HOST, CEED_COPY_VALUES, (CeedScalar *)x);
        PetscCall(VecRestoreArrayRead(X_loc, &x));
      }

      // Setup geometric scaling
      CeedQFunctionCreateInterior((*ctx)->ceed, 1, SetupMass, SetupMass_loc, &qf_setup);
      CeedQFunctionAddInput(qf_setup, "x", dim * dim, CEED_EVAL_GRAD);
      CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
      CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);

      CeedOperatorCreateAtPoints((*ctx)->ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup);
      CeedOperatorSetField(op_setup, "x", elem_restr_x_mesh, basis_x, CEED_VECTOR_ACTIVE);
      CeedOperatorSetField(op_setup, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
      CeedOperatorSetField(op_setup, "rho", elem_restr_q_data_points, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
      CeedOperatorAtPointsSetPoints(op_setup, elem_restr_x_points, x_ref_points);

      CeedOperatorApply(op_setup, x_coord, q_data_points, CEED_REQUEST_IMMEDIATE);

      // Cleanup
      CeedVectorDestroy(&x_coord);
      CeedQFunctionDestroy(&qf_setup);
      CeedOperatorDestroy(&op_setup);
    }

    // -- Cleanup
    PetscCall(ISDestroy(&is_points));
    PetscCall(VecDestroy(&X_ref));
  }

  PetscCall(DMSetApplicationContext(dm_mesh, (void *)(*ctx)));

  // Create operators
  // Mesh to points interpolation operator
  {
    CeedQFunction qf_mesh_to_points;

    // -- Create operator
    CeedQFunctionCreateIdentity((*ctx)->ceed, num_comp, CEED_EVAL_INTERP, CEED_EVAL_NONE, &qf_mesh_to_points);

    CeedOperatorCreateAtPoints((*ctx)->ceed, qf_mesh_to_points, NULL, NULL, &(*ctx)->op_mesh_to_points);
    CeedOperatorSetField((*ctx)->op_mesh_to_points, "input", elem_restr_u_mesh, basis_u, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField((*ctx)->op_mesh_to_points, "output", elem_restr_u_points, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
    CeedOperatorAtPointsSetPoints((*ctx)->op_mesh_to_points, elem_restr_x_points, x_ref_points);

    // -- Cleanup
    CeedQFunctionDestroy(&qf_mesh_to_points);
  }

  // RHS operator
  {
    CeedQFunction        qf_pts_to_mesh;
    CeedQFunctionContext qf_ctx;

    // -- Mass QFunction
    CeedQFunctionCreateInterior((*ctx)->ceed, 1, Mass, Mass_loc, &qf_pts_to_mesh);
    CeedQFunctionAddInput(qf_pts_to_mesh, "q data", 1, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qf_pts_to_mesh, "u", num_comp, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qf_pts_to_mesh, "v", num_comp, CEED_EVAL_INTERP);

    // -- QFunction context
    CeedQFunctionContextCreate((*ctx)->ceed, &qf_ctx);
    CeedQFunctionContextSetData(qf_ctx, CEED_MEM_HOST, CEED_COPY_VALUES, sizeof(num_comp), &num_comp);
    CeedQFunctionSetContext(qf_pts_to_mesh, qf_ctx);

    // -- Mass Operator
    CeedOperatorCreateAtPoints((*ctx)->ceed, qf_pts_to_mesh, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &(*ctx)->op_points_to_mesh);
    CeedOperatorSetField((*ctx)->op_points_to_mesh, "q data", elem_restr_q_data_points, CEED_BASIS_NONE, q_data_points);
    CeedOperatorSetField((*ctx)->op_points_to_mesh, "u", elem_restr_u_points, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField((*ctx)->op_points_to_mesh, "v", elem_restr_u_mesh, basis_u, CEED_VECTOR_ACTIVE);
    CeedOperatorAtPointsSetPoints((*ctx)->op_points_to_mesh, elem_restr_x_points, x_ref_points);

    // -- Cleanup
    CeedQFunctionContextDestroy(&qf_ctx);
    CeedQFunctionDestroy(&qf_pts_to_mesh);
  }

  // Mass operator
  {
    CeedQFunction        qf_mass;
    CeedQFunctionContext ctx_mass;

    // -- Mass QFunction
    CeedQFunctionCreateInterior((*ctx)->ceed, 1, Mass, Mass_loc, &qf_mass);
    CeedQFunctionAddInput(qf_mass, "q data", 1, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qf_mass, "u", num_comp, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_mass, "v", num_comp, CEED_EVAL_INTERP);

    // -- QFunction context
    CeedQFunctionContextCreate((*ctx)->ceed, &ctx_mass);
    CeedQFunctionContextSetData(ctx_mass, CEED_MEM_HOST, CEED_COPY_VALUES, sizeof(num_comp), &num_comp);
    CeedQFunctionSetContext(qf_mass, ctx_mass);

    // -- Mass Operator
    CeedOperatorCreateAtPoints((*ctx)->ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &(*ctx)->op_mass);
    CeedOperatorSetField((*ctx)->op_mass, "q data", elem_restr_q_data_points, CEED_BASIS_NONE, q_data_points);
    CeedOperatorSetField((*ctx)->op_mass, "u", elem_restr_u_mesh, basis_u, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField((*ctx)->op_mass, "v", elem_restr_u_mesh, basis_u, CEED_VECTOR_ACTIVE);
    CeedOperatorAtPointsSetPoints((*ctx)->op_mass, elem_restr_x_points, x_ref_points);

    // -- Cleanup
    CeedQFunctionContextDestroy(&ctx_mass);
    CeedQFunctionDestroy(&qf_mass);
  }

  // Cleanup
  CeedElemRestrictionDestroy(&elem_restr_u_mesh);
  CeedElemRestrictionDestroy(&elem_restr_x_mesh);
  CeedElemRestrictionDestroy(&elem_restr_u_points);
  CeedElemRestrictionDestroy(&elem_restr_x_points);
  CeedElemRestrictionDestroy(&elem_restr_q_data_points);
  CeedBasisDestroy(&basis_u);
  CeedBasisDestroy(&basis_x);
  CeedVectorDestroy(&x_ref_points);
  CeedVectorDestroy(&q_data_points);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmCeedContextDestroy(DMSwarmCeedContext *ctx) {
  PetscFunctionBeginUser;
  CeedDestroy(&(*ctx)->ceed);
  CeedVectorDestroy(&(*ctx)->u_mesh);
  CeedVectorDestroy(&(*ctx)->v_mesh);
  CeedVectorDestroy(&(*ctx)->u_points);
  CeedOperatorDestroy(&(*ctx)->op_mesh_to_points);
  CeedOperatorDestroy(&(*ctx)->op_points_to_mesh);
  CeedOperatorDestroy(&(*ctx)->op_mass);
  PetscCall(PetscFree(*ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ------------------------------------------------------------------------------------------------
// PETSc-libCEED memory space utilities
// ------------------------------------------------------------------------------------------------
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
// Swarm point location utility
// ------------------------------------------------------------------------------------------------
PetscErrorCode DMSwarmInitalizePointLocations(DM dm_swarm, PointSwarmType point_swarm_type, PetscInt num_points, PetscInt num_points_per_cell) {
  PetscFunctionBeginUser;
  switch (point_swarm_type) {
    case SWARM_GAUSS:
    case SWARM_UNIFORM: {
      // -- Set gauss or uniform point locations in each cell
      PetscInt    num_points_per_cell_1d = round(cbrt(num_points_per_cell * 1.0)), dim = 3;
      PetscScalar point_coords[num_points_per_cell * 3];
      CeedScalar  points_1d[num_points_per_cell_1d], weights_1d[num_points_per_cell_1d];

      if (point_swarm_type == SWARM_GAUSS) {
        PetscCall(CeedGaussQuadrature(num_points_per_cell_1d, points_1d, weights_1d));
      } else {
        for (PetscInt i = 0; i < num_points_per_cell_1d; i++) points_1d[i] = 2.0 * (PetscReal)(i + 1) / (PetscReal)(num_points_per_cell_1d + 1) - 1;
      }
      for (PetscInt i = 0; i < num_points_per_cell_1d; i++) {
        for (PetscInt j = 0; j < num_points_per_cell_1d; j++) {
          for (PetscInt k = 0; k < num_points_per_cell_1d; k++) {
            PetscInt p = (i * num_points_per_cell_1d + j) * num_points_per_cell_1d + k;

            point_coords[p * dim + 0] = points_1d[i];
            point_coords[p * dim + 1] = points_1d[j];
            point_coords[p * dim + 2] = points_1d[k];
          }
        }
      }
      PetscCall(DMSwarmSetPointCoordinatesCellwise(dm_swarm, num_points_per_cell_1d * num_points_per_cell_1d * num_points_per_cell_1d, point_coords));
    } break;
    case SWARM_CELL_RANDOM: {
      // -- Set points randomly in each cell
      PetscInt     dim = 3, num_cells_total = 1, num_cells[] = {1, 1, 1};
      PetscScalar *point_coords;
      PetscRandom  rng;

      PetscOptionsBegin(PetscObjectComm((PetscObject)dm_swarm), NULL, "libCEED example using PETSc with DMSwarm", NULL);

      PetscCall(PetscOptionsInt("-dm_plex_dim", "Background mesh dimension", NULL, dim, &dim, NULL));
      PetscCall(PetscOptionsIntArray("-dm_plex_box_faces", "Number of cells", NULL, num_cells, &dim, NULL));

      PetscOptionsEnd();

      PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)dm_swarm), &rng));

      num_cells_total = num_cells[0] * num_cells[1] * num_cells[2];
      PetscCall(DMSwarmGetField(dm_swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&point_coords));
      for (PetscInt c = 0; c < num_cells_total; c++) {
        PetscInt cell_index[3] = {c % num_cells[0], (c / num_cells[0]) % num_cells[1], (c / num_cells[0] / num_cells[1]) % num_cells[2]};

        for (PetscInt p = 0; p < num_points_per_cell; p++) {
          PetscInt    point_index = c * num_points_per_cell + p;
          PetscScalar random_value;

          for (PetscInt i = 0; i < dim; i++) {
            PetscCall(PetscRandomGetValue(rng, &random_value));
            point_coords[point_index * dim + i] = -1.0 + cell_index[i] * 2.0 / (num_cells[i] + 1.0) + random_value;
          }
        }
      }
      PetscCall(DMSwarmRestoreField(dm_swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&point_coords));
      PetscCall(PetscRandomDestroy(&rng));
    } break;
    case SWARM_SINUSOIDAL: {
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
    } break;
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
// RHS for Swarm to Mesh projection
// ------------------------------------------------------------------------------------------------
PetscErrorCode DMSwarmCreateProjectionRHS(DM dm_swarm, const char *field, Vec U_points, Vec B_mesh) {
  PetscMemType       B_mem_type, U_mem_type;
  DM                 dm_mesh;
  Vec                B_mesh_loc;
  PetscBool          has_u_points;
  DMSwarmCeedContext swarm_ceed_context;

  PetscFunctionBeginUser;
  // Get mesh DM
  PetscCall(DMSwarmGetCellDM(dm_swarm, &dm_mesh));
  PetscCall(DMGetApplicationContext(dm_mesh, (void *)&swarm_ceed_context));

  // Get swarm access
  has_u_points = !!U_points;
  if (!has_u_points) {
    PetscCall(DMSwarmSortGetAccess(dm_swarm));
    PetscCall(DMSwarmCreateLocalVectorFromField(dm_swarm, field, &U_points));
  }

  // Get mesh values
  PetscCall(VecReadP2C(U_points, &U_mem_type, swarm_ceed_context->u_points));
  PetscCall(DMGetLocalVector(dm_mesh, &B_mesh_loc));
  PetscCall(VecZeroEntries(B_mesh_loc));
  PetscCall(VecP2C(B_mesh_loc, &B_mem_type, swarm_ceed_context->v_mesh));

  // Interpolate field from swarm points to mesh
  CeedOperatorApply(swarm_ceed_context->op_points_to_mesh, swarm_ceed_context->u_points, swarm_ceed_context->v_mesh, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc Vecs and Local to Global
  PetscCall(VecReadC2P(swarm_ceed_context->u_points, U_mem_type, U_points));
  PetscCall(VecC2P(swarm_ceed_context->v_mesh, B_mem_type, B_mesh_loc));
  PetscCall(VecZeroEntries(B_mesh));
  PetscCall(DMLocalToGlobal(dm_mesh, B_mesh_loc, ADD_VALUES, B_mesh));

  // Restore swarm access
  if (!has_u_points) {
    PetscCall(DMSwarmDestroyLocalVectorFromField(dm_swarm, field, &U_points));
    PetscCall(DMSwarmSortRestoreAccess(dm_swarm));
  }

  // Cleanup
  PetscCall(DMRestoreLocalVector(dm_mesh, &B_mesh_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ------------------------------------------------------------------------------------------------
// Swarm "mass matrix"
// ------------------------------------------------------------------------------------------------
PetscErrorCode MatMult_SwarmMass(Mat A, Vec U_mesh, Vec V_mesh) {
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
  PetscCall(VecReadP2C(U_mesh_loc, &U_mem_type, swarm_ceed_context->u_mesh));
  PetscCall(DMGetLocalVector(dm_mesh, &V_mesh_loc));
  PetscCall(VecZeroEntries(V_mesh_loc));
  PetscCall(VecP2C(V_mesh_loc, &V_mem_type, swarm_ceed_context->v_mesh));

  // Apply swarm mass operator
  CeedOperatorApply(swarm_ceed_context->op_mass, swarm_ceed_context->u_mesh, swarm_ceed_context->v_mesh, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc Vecs and Local to Global
  PetscCall(VecReadC2P(swarm_ceed_context->u_mesh, U_mem_type, U_mesh_loc));
  PetscCall(VecC2P(swarm_ceed_context->v_mesh, V_mem_type, V_mesh_loc));
  PetscCall(VecZeroEntries(V_mesh));
  PetscCall(DMLocalToGlobal(dm_mesh, V_mesh_loc, ADD_VALUES, V_mesh));

  // Cleanup
  PetscCall(DMRestoreLocalVector(dm_mesh, &U_mesh_loc));
  PetscCall(DMRestoreLocalVector(dm_mesh, &V_mesh_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ------------------------------------------------------------------------------------------------
// Swarm to mesh projection
// ------------------------------------------------------------------------------------------------
PetscErrorCode DMSwarmProjectFromSwarmToCells(DM dm_swarm, const char *field, Vec U_points, Vec U_mesh) {
  PetscBool          test_mode;
  Vec                B_mesh;
  Mat                M;
  KSP                ksp;
  DM                 dm_mesh;
  DMSwarmCeedContext swarm_ceed_context;
  MPI_Comm           comm;

  PetscFunctionBeginUser;
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Swarm-to-Mesh Projection Options", NULL);
  PetscCall(PetscOptionsBool("-test", "Testing mode (do not print unless error is large)", NULL, test_mode, &test_mode, NULL));
  PetscOptionsEnd();

  comm = PetscObjectComm((PetscObject)dm_swarm);
  PetscCall(DMSwarmGetCellDM(dm_swarm, &dm_mesh));
  PetscCall(DMGetApplicationContext(dm_mesh, (void *)&swarm_ceed_context));
  PetscCall(VecDuplicate(U_mesh, &B_mesh));

  // Setup "mass matrix"
  {
    PetscInt l_size, g_size;

    PetscCall(VecGetLocalSize(U_mesh, &l_size));
    PetscCall(VecGetSize(U_mesh, &g_size));
    PetscCall(MatCreateShell(comm, l_size, l_size, g_size, g_size, swarm_ceed_context, &M));
    PetscCall(MatSetDM(M, dm_mesh));
    PetscCall(MatShellSetOperation(M, MATOP_MULT, (void (*)(void))MatMult_SwarmMass));
  }

  // Setup KSP
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

  // Setup RHS
  PetscCall(DMSwarmCreateProjectionRHS(dm_swarm, field, U_points, B_mesh));

  // Solve
  PetscCall(VecZeroEntries(U_mesh));
  PetscCall(KSPSolve(ksp, B_mesh, U_mesh));

  // KSP summary
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

  // Optional viewing
  PetscCall(KSPViewFromOptions(ksp, NULL, "-ksp_view"));

  // Cleanup
  PetscCall(VecDestroy(&B_mesh));
  PetscCall(MatDestroy(&M));
  PetscCall(KSPDestroy(&ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ------------------------------------------------------------------------------------------------
// BP setup
// ------------------------------------------------------------------------------------------------
PetscErrorCode SetupProblemSwarm(DM dm_swarm, Ceed ceed, BPData bp_data, CeedData data, PetscBool setup_rhs, Vec rhs, Vec target) {
  DM                  dm_mesh, dm_coord;
  CeedElemRestriction elem_restr_u_mesh, elem_restr_x_mesh, elem_restr_x_points, elem_restr_u_points, elem_restr_q_data_points;
  CeedBasis           basis_u, basis_x;
  CeedVector          x_coord, x_ref_points, q_data_points;
  CeedInt             num_comp, q_data_size = bp_data.q_data_size, dim, X_loc_len;
  CeedScalar          R = 1;                         // radius of the sphere
  CeedScalar          l = 1.0 / PetscSqrtReal(3.0);  // half edge of the inscribed cube
  Vec                 X_loc;
  PetscMemType        X_mem_type;

  PetscFunctionBeginUser;
  PetscCall(DMSwarmGetCellDM(dm_swarm, &dm_mesh));
  PetscCall(DMGetCoordinateDM(dm_mesh, &dm_coord));

  // Get coordinates
  PetscCall(DMGetCoordinatesLocal(dm_mesh, &X_loc));
  PetscCall(VecGetLocalSize(X_loc, &X_loc_len));
  CeedVectorCreate(ceed, X_loc_len, &x_coord);
  PetscCall(VecReadP2C(X_loc, &X_mem_type, x_coord));

  // Background mesh objects
  PetscCall(CreateBasisFromPlex(ceed, dm_mesh, NULL, 0, 0, 0, bp_data, &basis_u));
  PetscCall(CreateBasisFromPlex(ceed, dm_coord, NULL, 0, 0, 0, bp_data, &basis_x));
  PetscCall(CreateRestrictionFromPlex(ceed, dm_mesh, 0, NULL, 0, &elem_restr_u_mesh));
  PetscCall(CreateRestrictionFromPlex(ceed, dm_coord, 0, NULL, 0, &elem_restr_x_mesh));

  CeedElemRestrictionCreateVector(elem_restr_u_mesh, &data->x_ceed, NULL);
  CeedElemRestrictionCreateVector(elem_restr_u_mesh, &data->y_ceed, NULL);

  // Swarm objects
  {
    const PetscInt *cell_points;
    IS              is_points;
    Vec             X_ref;
    CeedInt         num_elem;

    PetscCall(DMSwarmCreateReferenceCoordinates(dm_swarm, &is_points, &X_ref));
    PetscCall(DMGetDimension(dm_mesh, &dim));
    CeedElemRestrictionGetNumElements(elem_restr_u_mesh, &num_elem);
    CeedElemRestrictionGetNumComponents(elem_restr_u_mesh, &num_comp);

    PetscCall(ISGetIndices(is_points, &cell_points));
    PetscInt num_points = cell_points[num_elem + 1] - num_elem - 2;
    CeedInt  offsets[num_elem + 1 + num_points];

    for (PetscInt i = 0; i < num_elem + 1; i++) offsets[i] = cell_points[i + 1] - 1;
    for (PetscInt i = num_elem + 1; i < num_points + num_elem + 1; i++) offsets[i] = cell_points[i + 1];
    PetscCall(ISRestoreIndices(is_points, &cell_points));

    // -- Points restrictions
    CeedElemRestrictionCreateAtPoints(ceed, num_elem, num_points, num_comp, num_points * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES, offsets,
                                      &elem_restr_u_points);
    CeedElemRestrictionCreateAtPoints(ceed, num_elem, num_points, dim, num_points * dim, CEED_MEM_HOST, CEED_COPY_VALUES, offsets,
                                      &elem_restr_x_points);
    CeedElemRestrictionCreateAtPoints(ceed, num_elem, num_points, q_data_size, num_points * q_data_size, CEED_MEM_HOST, CEED_COPY_VALUES, offsets,
                                      &elem_restr_q_data_points);

    // -- Points vectors
    CeedElemRestrictionCreateVector(elem_restr_q_data_points, &q_data_points, NULL);

    // -- Ref coordinates
    {
      PetscMemType       X_mem_type;
      const PetscScalar *x;

      CeedVectorCreate(ceed, num_points * dim, &x_ref_points);

      PetscCall(VecGetArrayReadAndMemType(X_ref, (const PetscScalar **)&x, &X_mem_type));
      CeedVectorSetArray(x_ref_points, MemTypeP2C(X_mem_type), CEED_COPY_VALUES, (CeedScalar *)x);
      PetscCall(VecRestoreArrayReadAndMemType(X_ref, (const PetscScalar **)&x));
    }

    // Create Q data
    {
      CeedQFunction qf_setup;
      CeedOperator  op_setup;

      // Setup geometric scaling
      CeedQFunctionCreateInterior(ceed, 1, bp_data.setup_geo, bp_data.setup_geo_loc, &qf_setup);
      CeedQFunctionAddInput(qf_setup, "x", dim, CEED_EVAL_INTERP);
      CeedQFunctionAddInput(qf_setup, "dx", dim * dim, CEED_EVAL_GRAD);
      CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
      CeedQFunctionAddOutput(qf_setup, "qdata", q_data_size, CEED_EVAL_NONE);

      CeedOperatorCreateAtPoints(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup);
      CeedOperatorSetField(op_setup, "x", elem_restr_x_mesh, basis_x, CEED_VECTOR_ACTIVE);
      CeedOperatorSetField(op_setup, "dx", elem_restr_x_mesh, basis_x, CEED_VECTOR_ACTIVE);
      CeedOperatorSetField(op_setup, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
      CeedOperatorSetField(op_setup, "qdata", elem_restr_q_data_points, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
      CeedOperatorAtPointsSetPoints(op_setup, elem_restr_x_points, x_ref_points);

      CeedOperatorApply(op_setup, x_coord, q_data_points, CEED_REQUEST_IMMEDIATE);

      // Cleanup
      CeedQFunctionDestroy(&qf_setup);
      CeedOperatorDestroy(&op_setup);
    }

    // Cleanup
    PetscCall(ISDestroy(&is_points));
    PetscCall(VecDestroy(&X_ref));
  }

  // Set up PDE operator

  CeedQFunction qf_apply;
  CeedOperator  op_apply;
  CeedInt       in_scale  = bp_data.in_mode == CEED_EVAL_GRAD ? dim : 1;
  CeedInt       out_scale = bp_data.out_mode == CEED_EVAL_GRAD ? dim : 1;

  CeedQFunctionCreateInterior(ceed, 1, bp_data.apply, bp_data.apply_loc, &qf_apply);
  CeedQFunctionAddInput(qf_apply, "u", num_comp * in_scale, bp_data.in_mode);
  CeedQFunctionAddInput(qf_apply, "qdata", q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_apply, "v", num_comp * out_scale, bp_data.out_mode);

  // Create the mass or diff operator
  CeedOperatorCreateAtPoints(ceed, qf_apply, NULL, NULL, &op_apply);
  CeedOperatorSetField(op_apply, "u", elem_restr_u_mesh, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "qdata", elem_restr_q_data_points, CEED_BASIS_NONE, q_data_points);
  CeedOperatorSetField(op_apply, "v", elem_restr_u_mesh, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorAtPointsSetPoints(op_apply, elem_restr_x_points, x_ref_points);

  // Set up RHS if needed
  if (setup_rhs) {
    CeedQFunction qf_setup_rhs;
    CeedOperator  op_setup_rhs;
    Vec           rhs_loc;
    CeedVector    rhs_ceed, target_ceed;
    PetscMemType  rhs_mem_type, target_mem_type;

    // Create RHS vector
    PetscCall(DMCreateLocalVector(dm_mesh, &rhs_loc));

    CeedElemRestrictionCreateVector(elem_restr_u_points, &target_ceed, NULL);
    CeedElemRestrictionCreateVector(elem_restr_u_mesh, &rhs_ceed, NULL);

    // Create the q-function that sets up the RHS and true solution
    CeedQFunctionCreateInterior(ceed, 1, bp_data.setup_rhs, bp_data.setup_rhs_loc, &qf_setup_rhs);
    CeedQFunctionAddInput(qf_setup_rhs, "x", dim, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_setup_rhs, "qdata", q_data_size, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qf_setup_rhs, "true solution", num_comp, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qf_setup_rhs, "rhs", num_comp, CEED_EVAL_INTERP);

    // Create the operator that builds the RHS and true solution
    CeedOperatorCreateAtPoints(ceed, qf_setup_rhs, NULL, NULL, &op_setup_rhs);
    CeedOperatorSetField(op_setup_rhs, "x", elem_restr_x_mesh, basis_x, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_setup_rhs, "qdata", elem_restr_q_data_points, CEED_BASIS_NONE, q_data_points);
    CeedOperatorSetField(op_setup_rhs, "true solution", elem_restr_u_points, CEED_BASIS_NONE, target_ceed);
    CeedOperatorSetField(op_setup_rhs, "rhs", elem_restr_u_mesh, basis_u, CEED_VECTOR_ACTIVE);
    CeedOperatorAtPointsSetPoints(op_setup_rhs, elem_restr_x_points, x_ref_points);

    // Set up the libCEED context
    CeedQFunctionContext ctx_rhs_setup;
    CeedQFunctionContextCreate(ceed, &ctx_rhs_setup);
    CeedScalar rhs_setup_data[2] = {R, l};
    CeedQFunctionContextSetData(ctx_rhs_setup, CEED_MEM_HOST, CEED_COPY_VALUES, sizeof rhs_setup_data, &rhs_setup_data);
    CeedQFunctionSetContext(qf_setup_rhs, ctx_rhs_setup);
    CeedQFunctionContextDestroy(&ctx_rhs_setup);

    // Setup RHS and target
    PetscCall(VecP2C(rhs_loc, &rhs_mem_type, rhs_ceed));
    PetscCall(VecP2C(target, &target_mem_type, target_ceed));
    CeedOperatorApply(op_setup_rhs, x_coord, rhs_ceed, CEED_REQUEST_IMMEDIATE);
    PetscCall(VecC2P(rhs_ceed, rhs_mem_type, rhs_loc));
    PetscCall(VecC2P(target_ceed, target_mem_type, target));

    // Local-to-global
    PetscCall(VecZeroEntries(rhs));
    PetscCall(DMLocalToGlobal(dm_mesh, rhs_loc, ADD_VALUES, rhs));

    PetscCall(VecViewFromOptions(rhs, NULL, "-rhs_view"));

    // Cleanup
    PetscCall(DMRestoreLocalVector(dm_mesh, &rhs_loc));
    CeedVectorDestroy(&rhs_ceed);
    CeedVectorDestroy(&target_ceed);
    CeedQFunctionDestroy(&qf_setup_rhs);
    CeedOperatorDestroy(&op_setup_rhs);
  }

  // Save libCEED data
  data->basis_x         = basis_x;
  data->basis_u         = basis_u;
  data->elem_restr_x    = elem_restr_x_mesh;
  data->elem_restr_u    = elem_restr_u_mesh;
  data->elem_restr_u_i  = elem_restr_u_points;
  data->elem_restr_qd_i = elem_restr_q_data_points;
  data->qf_apply        = qf_apply;
  data->op_apply        = op_apply;
  data->q_data          = q_data_points;
  data->q_data_size     = q_data_size;

  // Cleanup
  PetscCall(VecReadC2P(x_coord, X_mem_type, X_loc));
  CeedVectorDestroy(&x_coord);
  CeedVectorDestroy(&x_ref_points);
  CeedElemRestrictionDestroy(&elem_restr_x_points);

  PetscFunctionReturn(PETSC_SUCCESS);
}
