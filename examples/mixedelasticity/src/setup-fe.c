#include "../include/setup-fe.h"

#include "petscerror.h"

// -----------------------------------------------------------------------------
// Convert PETSc MemType to libCEED MemType
// -----------------------------------------------------------------------------
CeedMemType MemTypeP2C(PetscMemType mem_type) { return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST; }

// ---------------------------------------------------------------------------
// Set-up FE for H1 space
// ---------------------------------------------------------------------------
PetscErrorCode SetupFEByOrder(AppCtx app_ctx, ProblemData problem_data, DM dm) {
  // FE space for displacement and pressure fields
  PetscFE fe_u, fe_p;
  // number of quadrature points
  app_ctx->q_order     = app_ctx->u_order + app_ctx->q_extra;
  PetscInt  q_order    = app_ctx->q_order;
  PetscInt  u_order    = app_ctx->u_order;
  PetscInt  p_order    = app_ctx->p_order;
  PetscBool is_simplex = PETSC_TRUE;
  PetscInt  dim;
  PetscFunctionBeginUser;

  // Check if simplex or tensor-product element
  PetscCall(DMPlexIsSimplex(dm, &is_simplex));
  // Create FE space
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscFECreateLagrange(app_ctx->comm, dim, dim, is_simplex, u_order, q_order, &fe_u));
  PetscCall(DMAddField(dm, NULL, (PetscObject)fe_u));
  PetscCall(PetscFECreateLagrange(app_ctx->comm, dim, 1, is_simplex, p_order, q_order, &fe_p));
  PetscCall(DMAddField(dm, NULL, (PetscObject)fe_p));
  PetscCall(DMCreateDS(dm));
  {
    // create FE field for coordinates
    PetscFE  fe_coords;
    PetscInt num_comp_coord;
    PetscCall(DMGetCoordinateDim(dm, &num_comp_coord));
    PetscCall(PetscFECreateLagrange(app_ctx->comm, dim, num_comp_coord, is_simplex, 1, q_order, &fe_coords));
    PetscCall(DMProjectCoordinates(dm, fe_coords));
    PetscCall(PetscFEDestroy(&fe_coords));
  }

  // Setup boundary
  DMAddBoundariesDirichlet(dm);

  if (!is_simplex) {
    DM dm_coord;
    PetscCall(DMGetCoordinateDM(dm, &dm_coord));
    PetscCall(DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL));
    PetscCall(DMPlexSetClosurePermutationTensor(dm_coord, PETSC_DETERMINE, NULL));
  }
  PetscCall(PetscFEDestroy(&fe_u));
  PetscCall(PetscFEDestroy(&fe_p));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Get CEED restriction data from DMPlex
// -----------------------------------------------------------------------------
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm, DMLabel domain_label, CeedInt value, CeedInt height, PetscInt dm_field,
                                         CeedElemRestriction *elem_restr) {
  PetscInt num_elem, elem_size, num_dof, num_comp, *elem_restr_offsets;

  PetscFunctionBeginUser;

  PetscCall(DMPlexGetLocalOffsets(dm, domain_label, value, height, dm_field, &num_elem, &elem_size, &num_comp, &num_dof, &elem_restr_offsets));

  CeedElemRestrictionCreate(ceed, num_elem, elem_size, num_comp, 1, num_dof, CEED_MEM_HOST, CEED_COPY_VALUES, elem_restr_offsets, elem_restr);
  PetscCall(PetscFree(elem_restr_offsets));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Utility function - convert from DMPolytopeType to CeedElemTopology
// -----------------------------------------------------------------------------
CeedElemTopology ElemTopologyP2C(DMPolytopeType cell_type) {
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
// Convert DM field to DS field
// -----------------------------------------------------------------------------
PetscErrorCode DMFieldToDSField(DM dm, DMLabel domain_label, PetscInt dm_field, PetscInt *ds_field) {
  PetscDS         ds;
  IS              field_is;
  const PetscInt *fields;
  PetscInt        num_fields;

  PetscFunctionBeginUser;

  // Translate dm_field to ds_field
  PetscCall(DMGetRegionDS(dm, domain_label, &field_is, &ds));
  PetscCall(ISGetIndices(field_is, &fields));
  PetscCall(ISGetSize(field_is, &num_fields));
  for (PetscInt i = 0; i < num_fields; i++) {
    if (dm_field == fields[i]) {
      *ds_field = i;
      break;
    }
  }
  PetscCall(ISRestoreIndices(field_is, &fields));

  if (*ds_field == -1) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Could not find dm_field %" PetscInt_FMT " in DS", dm_field);

  PetscFunctionReturn(0);
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

  // General basis information
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
  if (!elem_topo) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "DMPlex topology not supported");
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

  // Convert to libCEED orientation
  {
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
  CeedBasisCreateH1(ceed, elem_topo, num_comp, P, Q, interp, grad, q_points, q_weights, basis);
  PetscCall(PetscFree(q_points));
  PetscCall(PetscFree(interp));
  PetscCall(PetscFree(grad));

  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// Get CEED Basis from DMPlex
// -----------------------------------------------------------------------------
PetscErrorCode CreateBasisFromPlex(Ceed ceed, DM dm, DMLabel domain_label, CeedInt label_value, CeedInt height, CeedInt dm_field,
                                   ProblemData problem_data, CeedBasis *basis) {
  PetscDS         ds;
  PetscFE         fe;
  PetscQuadrature quadrature;
  PetscBool       is_simplex = PETSC_TRUE;
  PetscInt        ds_field   = -1;

  PetscFunctionBeginUser;

  // Get element information
  PetscCall(DMGetRegionDS(dm, domain_label, NULL, &ds));
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

    CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp, P_1d, Q_1d, problem_data->quadrature_mode, basis);
  }

  PetscFunctionReturn(0);
}
