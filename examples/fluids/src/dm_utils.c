// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utilities for setting up DM and PetscFE

#include <ceed.h>
#include <petscdmplex.h>
#include <petscds.h>

#include "../navierstokes.h"

/**
  @brief Convert `DM` field to `DS` field.

  @param[in]   dm            `DM` holding mesh
  @param[in]   domain_label  Label for `DM` domain
  @param[in]   dm_field      Index of `DM` field
  @param[out]  ds_field      Index of `DS` field

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode DMFieldToDSField(DM dm, DMLabel domain_label, PetscInt dm_field, PetscInt *ds_field) {
  PetscDS         ds;
  IS              field_is;
  const PetscInt *fields;
  PetscInt        num_fields;

  PetscFunctionBeginUser;
  PetscCall(DMGetRegionDS(dm, domain_label, &field_is, &ds, NULL));
  PetscCall(ISGetIndices(field_is, &fields));
  PetscCall(ISGetSize(field_is, &num_fields));
  for (PetscInt i = 0; i < num_fields; i++) {
    if (dm_field == fields[i]) {
      *ds_field = i;
      break;
    }
  }
  PetscCall(ISRestoreIndices(field_is, &fields));

  PetscCheck(*ds_field != -1, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Could not find dm_field %" PetscInt_FMT " in DS", dm_field);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Create `CeedElemRestriction` from `DMPlex`.

  Not collective across MPI processes.

  @param[in]   ceed          `Ceed` context
  @param[in]   dm            `DMPlex` holding mesh
  @param[in]   domain_label  `DMLabel` for `DMPlex` domain
  @param[in]   label_value   Stratum value
  @param[in]   height        Height of `DMPlex` topology
  @param[in]   dm_field      Index of `DMPlex` field
  @param[out]  restriction   `CeedElemRestriction` for `DMPlex`

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode DMPlexCeedElemRestrictionCreate(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height, PetscInt dm_field,
                                               CeedElemRestriction *restriction) {
  PetscInt num_elem, elem_size, num_dof, num_comp, *restriction_offsets_petsc;
  CeedInt *restriction_offsets_ceed = NULL;

  PetscFunctionBeginUser;
  PetscCall(
      DMPlexGetLocalOffsets(dm, domain_label, label_value, height, dm_field, &num_elem, &elem_size, &num_comp, &num_dof, &restriction_offsets_petsc));
  PetscCall(IntArrayPetscToCeed(num_elem * elem_size, &restriction_offsets_petsc, &restriction_offsets_ceed));
  PetscCallCeed(ceed, CeedElemRestrictionCreate(ceed, num_elem, elem_size, num_comp, 1, num_dof, CEED_MEM_HOST, CEED_COPY_VALUES,
                                                restriction_offsets_ceed, restriction));
  PetscCall(PetscFree(restriction_offsets_ceed));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Create `CeedElemRestriction` from `DMPlex` domain for mesh coordinates.

  Not collective across MPI processes.

  @param[in]   ceed          `Ceed` context
  @param[in]   dm            `DMPlex` holding mesh
  @param[in]   domain_label  Label for `DMPlex` domain
  @param[in]   label_value   Stratum value
  @param[in]   height        Height of `DMPlex` topology
  @param[out]  restriction   `CeedElemRestriction` for mesh

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode DMPlexCeedElemRestrictionCoordinateCreate(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height,
                                                         CeedElemRestriction *restriction) {
  DM dm_coord;

  PetscFunctionBeginUser;
  PetscCall(DMGetCellCoordinateDM(dm, &dm_coord));
  if (!dm_coord) {
    PetscCall(DMGetCoordinateDM(dm, &dm_coord));
  }
  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, dm_coord, domain_label, label_value, height, 0, restriction));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Create `CeedElemRestriction` from `DMPlex` domain for auxilury `QFunction` data.

  Not collective across MPI processes.

  @param[in]   ceed           `Ceed` context
  @param[in]   dm             `DMPlex` holding mesh
  @param[in]   domain_label   Label for `DMPlex` domain
  @param[in]   label_value    Stratum value
  @param[in]   height         Height of `DMPlex` topology
  @param[in]   q_data_size    Number of components for `QFunction` data
  @param[in]   is_collocated  Boolean flag indicating if the data is collocated on the nodes (`PETSC_TRUE`) or on quadrature points (`PETSC_FALSE`)
  @param[out]  restriction    Strided `CeedElemRestriction` for `QFunction` data

  @return An error code: 0 - success, otherwise - failure
**/
static PetscErrorCode DMPlexCeedElemRestrictionStridedCreate(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height,
                                                             PetscInt q_data_size, PetscBool is_collocated, CeedElemRestriction *restriction) {
  PetscInt num_elem, num_qpts, dm_field = 0;

  PetscFunctionBeginUser;
  {  // Get number of elements
    PetscInt depth;
    DMLabel  depth_label;
    IS       point_is, depth_is;

    PetscCall(DMPlexGetDepth(dm, &depth));
    PetscCall(DMPlexGetDepthLabel(dm, &depth_label));
    PetscCall(DMLabelGetStratumIS(depth_label, depth - height, &depth_is));
    if (domain_label) {
      IS domain_is;

      PetscCall(DMLabelGetStratumIS(domain_label, label_value, &domain_is));
      if (domain_is) {
        PetscCall(ISIntersect(depth_is, domain_is, &point_is));
        PetscCall(ISDestroy(&domain_is));
      } else {
        point_is = NULL;
      }
      PetscCall(ISDestroy(&depth_is));
    } else {
      point_is = depth_is;
    }
    if (point_is) {
      PetscCall(ISGetLocalSize(point_is, &num_elem));
    } else {
      num_elem = 0;
    }
    PetscCall(ISDestroy(&point_is));
  }

  {  // Get number of quadrature points
    PetscDS  ds;
    PetscFE  fe;
    PetscInt ds_field = -1;

    PetscCall(DMGetRegionDS(dm, domain_label, NULL, &ds, NULL));
    PetscCall(DMFieldToDSField(dm, domain_label, dm_field, &ds_field));
    PetscCall(PetscDSGetDiscretization(ds, ds_field, (PetscObject *)&fe));
    PetscCall(PetscFEGetHeightSubspace(fe, height, &fe));
    if (is_collocated) {
      PetscDualSpace dual_space;
      PetscInt       num_dual_basis_vectors, dim, num_comp;

      PetscCall(PetscFEGetSpatialDimension(fe, &dim));
      PetscCall(PetscFEGetNumComponents(fe, &num_comp));
      PetscCall(PetscFEGetDualSpace(fe, &dual_space));
      PetscCall(PetscDualSpaceGetDimension(dual_space, &num_dual_basis_vectors));
      num_qpts = num_dual_basis_vectors / num_comp;
    } else {
      PetscQuadrature quadrature;

      PetscCall(DMGetRegionDS(dm, domain_label, NULL, &ds, NULL));
      PetscCall(DMFieldToDSField(dm, domain_label, dm_field, &ds_field));
      PetscCall(PetscDSGetDiscretization(ds, ds_field, (PetscObject *)&fe));
      PetscCall(PetscFEGetHeightSubspace(fe, height, &fe));
      PetscCall(PetscFEGetQuadrature(fe, &quadrature));
      PetscCall(PetscQuadratureGetData(quadrature, NULL, NULL, &num_qpts, NULL, NULL));
    }
  }

  // Create the restriction
  PetscCallCeed(ceed, CeedElemRestrictionCreateStrided(ceed, num_elem, num_qpts, q_data_size, num_elem * num_qpts * q_data_size, CEED_STRIDES_BACKEND,
                                                       restriction));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Create `CeedElemRestriction` from `DMPlex` domain for auxilury `QFunction` data.

  Not collective across MPI processes.

  @param[in]   ceed           `Ceed` context
  @param[in]   dm             `DMPlex` holding mesh
  @param[in]   domain_label   Label for `DMPlex` domain
  @param[in]   label_value    Stratum value
  @param[in]   height         Height of `DMPlex` topology
  @param[in]   q_data_size    Number of components for `QFunction` data
  @param[out]  restriction    Strided `CeedElemRestriction` for `QFunction` data

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode DMPlexCeedElemRestrictionQDataCreate(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height,
                                                    PetscInt q_data_size, CeedElemRestriction *restriction) {
  PetscFunctionBeginUser;
  PetscCall(DMPlexCeedElemRestrictionStridedCreate(ceed, dm, domain_label, label_value, height, q_data_size, PETSC_FALSE, restriction));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Create `CeedElemRestriction` from `DMPlex` domain for nodally collocated auxilury `QFunction` data.

  Not collective across MPI processes.

  @param[in]   ceed           `Ceed` context
  @param[in]   dm             `DMPlex` holding mesh
  @param[in]   domain_label   Label for `DMPlex` domain
  @param[in]   label_value    Stratum value
  @param[in]   height         Height of `DMPlex` topology
  @param[in]   q_data_size    Number of components for `QFunction` data
  @param[out]  restriction    Strided `CeedElemRestriction` for `QFunction` data

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode DMPlexCeedElemRestrictionCollocatedCreate(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height,
                                                         PetscInt q_data_size, CeedElemRestriction *restriction) {
  PetscFunctionBeginUser;
  PetscCall(DMPlexCeedElemRestrictionStridedCreate(ceed, dm, domain_label, label_value, height, q_data_size, PETSC_TRUE, restriction));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// -----------------------------------------------------------------------------
// Utility function - convert from DMPolytopeType to CeedElemTopology
// -----------------------------------------------------------------------------
static CeedElemTopology ElemTopologyP2C(DMPolytopeType cell_type) {
  switch (cell_type) {
    case DM_POLYTOPE_TRIANGLE:
      return CEED_TOPOLOGY_TRIANGLE;
    case DM_POLYTOPE_QUADRILATERAL:
      return CEED_TOPOLOGY_QUAD;
    case DM_POLYTOPE_TETRAHEDRON:
      return CEED_TOPOLOGY_TET;
    case DM_POLYTOPE_HEXAHEDRON:
      return CEED_TOPOLOGY_HEX;
    default:
      return 0;
  }
}

// -----------------------------------------------------------------------------
// Create libCEED Basis from PetscTabulation
// -----------------------------------------------------------------------------
PetscErrorCode BasisCreateFromTabulation(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height, PetscInt face, PetscFE fe,
                                         PetscTabulation basis_tabulation, PetscQuadrature quadrature, CeedBasis *basis) {
  PetscInt           first_point;
  PetscInt           ids[1] = {label_value};
  DMLabel            depth_label;
  DMPolytopeType     cell_type;
  CeedElemTopology   elem_topo;
  PetscScalar       *q_points, *interp, *grad;
  const PetscScalar *q_weights;
  PetscDualSpace     dual_space;
  PetscInt           num_dual_basis_vectors;
  PetscInt           dim, num_comp, P, Q;

  PetscFunctionBeginUser;
  PetscCall(PetscFEGetSpatialDimension(fe, &dim));
  PetscCall(PetscFEGetNumComponents(fe, &num_comp));
  PetscCall(PetscFEGetDualSpace(fe, &dual_space));
  PetscCall(PetscDualSpaceGetDimension(dual_space, &num_dual_basis_vectors));
  P = num_dual_basis_vectors / num_comp;

  // Use depth label if no domain label present
  if (!domain_label) {
    PetscInt depth;

    PetscCall(DMPlexGetDepth(dm, &depth));
    PetscCall(DMPlexGetDepthLabel(dm, &depth_label));
    ids[0] = depth - height;
  }

  // Get cell interp, grad, and quadrature data
  PetscCall(DMGetFirstLabeledPoint(dm, dm, domain_label ? domain_label : depth_label, 1, ids, height, &first_point, NULL));
  PetscCall(DMPlexGetCellType(dm, first_point, &cell_type));
  elem_topo = ElemTopologyP2C(cell_type);
  PetscCheck(elem_topo, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "DMPlex topology not supported");
  {
    size_t             q_points_size;
    const PetscScalar *q_points_petsc;
    PetscInt           q_dim;

    PetscCall(PetscQuadratureGetData(quadrature, &q_dim, NULL, &Q, &q_points_petsc, &q_weights));
    q_points_size = Q * dim * sizeof(CeedScalar);
    PetscCall(PetscCalloc(q_points_size, &q_points));
    for (PetscInt q = 0; q < Q; q++) {
      for (PetscInt d = 0; d < q_dim; d++) q_points[q * dim + d] = q_points_petsc[q * q_dim + d];
    }
  }

  {  // Convert to libCEED orientation
    PetscBool       is_simplex  = PETSC_FALSE;
    IS              permutation = NULL;
    const PetscInt *permutation_indices;

    PetscCall(DMPlexIsSimplex(dm, &is_simplex));
    if (!is_simplex) {
      PetscSection section;

      // -- Get permutation
      PetscCall(DMGetLocalSection(dm, &section));
      PetscCall(PetscSectionGetClosurePermutation(section, (PetscObject)dm, dim, num_comp * P, &permutation));
      PetscCall(ISGetIndices(permutation, &permutation_indices));
    }

    // -- Copy interp, grad matrices
    PetscCall(PetscCalloc(P * Q * sizeof(CeedScalar), &interp));
    PetscCall(PetscCalloc(P * Q * dim * sizeof(CeedScalar), &grad));
    const CeedInt c = 0;
    for (CeedInt q = 0; q < Q; q++) {
      for (CeedInt p_ceed = 0; p_ceed < P; p_ceed++) {
        CeedInt p_petsc = is_simplex ? (p_ceed * num_comp) : permutation_indices[p_ceed * num_comp];

        interp[q * P + p_ceed] = basis_tabulation->T[0][((face * Q + q) * P * num_comp + p_petsc) * num_comp + c];
        for (CeedInt d = 0; d < dim; d++) {
          grad[(d * Q + q) * P + p_ceed] = basis_tabulation->T[1][(((face * Q + q) * P * num_comp + p_petsc) * num_comp + c) * dim + d];
        }
      }
    }

    // -- Cleanup
    if (permutation) PetscCall(ISRestoreIndices(permutation, &permutation_indices));
    PetscCall(ISDestroy(&permutation));
  }

  // Finally, create libCEED basis
  PetscCallCeed(ceed, CeedBasisCreateH1(ceed, elem_topo, num_comp, P, Q, interp, grad, q_points, q_weights, basis));
  PetscCall(PetscFree(q_points));
  PetscCall(PetscFree(interp));
  PetscCall(PetscFree(grad));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// -----------------------------------------------------------------------------
// Get CEED Basis from DMPlex
// -----------------------------------------------------------------------------
PetscErrorCode CreateBasisFromPlex(Ceed ceed, DM dm, DMLabel domain_label, CeedInt label_value, CeedInt height, CeedInt dm_field, CeedBasis *basis) {
  PetscDS         ds;
  PetscFE         fe;
  PetscQuadrature quadrature;
  PetscBool       is_simplex = PETSC_TRUE;
  PetscInt        ds_field   = -1;

  PetscFunctionBeginUser;
  // Get element information
  PetscCall(DMGetRegionDS(dm, domain_label, NULL, &ds, NULL));
  PetscCall(DMFieldToDSField(dm, domain_label, dm_field, &ds_field));
  PetscCall(PetscDSGetDiscretization(ds, ds_field, (PetscObject *)&fe));
  PetscCall(PetscFEGetHeightSubspace(fe, height, &fe));
  PetscCall(PetscFEGetQuadrature(fe, &quadrature));

  // Check if simplex or tensor-product mesh
  PetscCall(DMPlexIsSimplex(dm, &is_simplex));

  // Build libCEED basis
  if (is_simplex) {
    PetscTabulation basis_tabulation;
    PetscInt        num_derivatives = 1, face = 0;

    PetscCall(PetscFEGetCellTabulation(fe, num_derivatives, &basis_tabulation));
    PetscCall(BasisCreateFromTabulation(ceed, dm, domain_label, label_value, height, face, fe, basis_tabulation, quadrature, basis));
  } else {
    PetscDualSpace dual_space;
    PetscInt       num_dual_basis_vectors;
    PetscInt       dim, num_comp, P, Q;

    PetscCall(PetscFEGetSpatialDimension(fe, &dim));
    PetscCall(PetscFEGetNumComponents(fe, &num_comp));
    PetscCall(PetscFEGetDualSpace(fe, &dual_space));
    PetscCall(PetscDualSpaceGetDimension(dual_space, &num_dual_basis_vectors));
    P = num_dual_basis_vectors / num_comp;
    PetscCall(PetscQuadratureGetData(quadrature, NULL, NULL, &Q, NULL, NULL));

    CeedInt P_1d = (CeedInt)round(pow(P, 1.0 / dim));
    CeedInt Q_1d = (CeedInt)round(pow(Q, 1.0 / dim));

    PetscCallCeed(ceed, CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp, P_1d, Q_1d, CEED_GAUSS, basis));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Setup `DM` with FE space of appropriate degree

  Must be followed by `DMSetupByOrderEnd_FEM`

  @param[in]   setup_faces     Flag to setup face geometry
  @param[in]   setup_coords    Flag to setup coordinate spaces
  @param[in]   degree          Polynomial orders of field
  @param[in]   coord_order     Polynomial order of coordinate basis, or `PETSC_DECIDE` for default
  @param[in]   q_extra         Additional quadrature order
  @param[in]   num_fields      Number of fields in solution vector
  @param[in]   field_sizes     Array of number of components for each field
  @param[out]  dm              `DM` to setup

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode DMSetupByOrderBegin_FEM(PetscBool setup_faces, PetscBool setup_coords, PetscInt degree, PetscInt coord_order, PetscInt q_extra,
                                       PetscInt num_fields, const PetscInt *field_sizes, DM dm) {
  PetscInt  dim, q_order = degree + q_extra;
  PetscBool is_simplex = PETSC_TRUE;
  PetscFE   fe;
  MPI_Comm  comm = PetscObjectComm((PetscObject)dm);

  PetscFunctionBeginUser;
  PetscCall(DMPlexIsSimplex(dm, &is_simplex));

  // Setup DM
  PetscCall(DMGetDimension(dm, &dim));
  for (PetscInt i = 0; i < num_fields; i++) {
    PetscFE  fe_face;
    PetscInt q_order = degree + q_extra;

    PetscCall(PetscFECreateLagrange(comm, dim, field_sizes[i], is_simplex, degree, q_order, &fe));
    if (setup_faces) PetscCall(PetscFEGetHeightSubspace(fe, 1, &fe_face));
    PetscCall(DMAddField(dm, NULL, (PetscObject)fe));
    PetscCall(PetscFEDestroy(&fe));
  }
  PetscCall(DMCreateDS(dm));

  // Project coordinates to enrich quadrature space
  if (setup_coords) {
    DM             dm_coord;
    PetscDS        ds_coord;
    PetscFE        fe_coord_current, fe_coord_new, fe_coord_face_new;
    PetscDualSpace fe_coord_dual_space;
    PetscInt       fe_coord_order, num_comp_coord;

    PetscCall(DMGetCoordinateDM(dm, &dm_coord));
    PetscCall(DMGetCoordinateDim(dm, &num_comp_coord));
    PetscCall(DMGetRegionDS(dm_coord, NULL, NULL, &ds_coord, NULL));
    PetscCall(PetscDSGetDiscretization(ds_coord, 0, (PetscObject *)&fe_coord_current));
    PetscCall(PetscFEGetDualSpace(fe_coord_current, &fe_coord_dual_space));
    PetscCall(PetscDualSpaceGetOrder(fe_coord_dual_space, &fe_coord_order));

    // Create FE for coordinates
    if (coord_order != PETSC_DECIDE) fe_coord_order = coord_order;
    PetscCall(PetscFECreateLagrange(comm, dim, num_comp_coord, is_simplex, fe_coord_order, q_order, &fe_coord_new));
    if (setup_faces) PetscCall(PetscFEGetHeightSubspace(fe_coord_new, 1, &fe_coord_face_new));
    PetscCall(DMSetCoordinateDisc(dm, fe_coord_new, PETSC_TRUE));
    PetscCall(DMLocalizeCoordinates(dm));  // Update CellCoordinateDM with projected coordinates
    PetscCall(PetscFEDestroy(&fe_coord_new));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Finish setting up `DM`

  Must be called after `DMSetupByOrderBegin_FEM` and all strong BCs have be declared via `DMAddBoundaries`.

  @param[in]   setup_coords    Flag to setup coordinate spaces
  @param[out]  dm              `DM` to setup

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode DMSetupByOrderEnd_FEM(PetscBool setup_coords, DM dm) {
  PetscBool is_simplex;

  PetscFunctionBeginUser;
  PetscCall(DMPlexIsSimplex(dm, &is_simplex));
  // Set tensor permutation if needed
  if (!is_simplex) {
    PetscCall(DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL));
    if (setup_coords) {
      DM dm_coord;

      PetscCall(DMGetCoordinateDM(dm, &dm_coord));
      PetscCall(DMPlexSetClosurePermutationTensor(dm_coord, PETSC_DETERMINE, NULL));
      PetscCall(DMGetCellCoordinateDM(dm, &dm_coord));
      if (dm_coord) PetscCall(DMPlexSetClosurePermutationTensor(dm_coord, PETSC_DETERMINE, NULL));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Setup `DM` with FE space of appropriate degree with no boundary conditions

  Calls `DMSetupByOrderBegin_FEM` and `DMSetupByOrderEnd_FEM` successively

  @param[in]   setup_faces     Flag to setup face geometry
  @param[in]   setup_coords    Flag to setup coordinate spaces
  @param[in]   degree          Polynomial orders of field
  @param[in]   coord_order     Polynomial order of coordinate basis, or `PETSC_DECIDE` for default
  @param[in]   q_extra         Additional quadrature order
  @param[in]   num_fields      Number of fields in solution vector
  @param[in]   field_sizes     Array of number of components for each field
  @param[out]  dm              `DM` to setup

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode DMSetupByOrder_FEM(PetscBool setup_faces, PetscBool setup_coords, PetscInt degree, PetscInt coord_order, PetscInt q_extra,
                                  PetscInt num_fields, const PetscInt *field_sizes, DM dm) {
  PetscFunctionBeginUser;
  PetscCall(DMSetupByOrderBegin_FEM(setup_faces, setup_coords, degree, coord_order, q_extra, num_fields, field_sizes, dm));
  PetscCall(DMSetupByOrderEnd_FEM(setup_coords, dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}
