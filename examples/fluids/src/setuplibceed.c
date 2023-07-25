// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Setup libCEED for Navier-Stokes example using PETSc

#include <ceed.h>
#include <petscdmplex.h>

#include <petscds.h>
#include "../navierstokes.h"

// Utility function to create local CEED restriction
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt height, DMLabel domain_label, CeedInt label_value, PetscInt dm_field,
                                         CeedElemRestriction *elem_restr) {
  PetscInt num_elem, elem_size, num_dof, num_comp, *elem_restr_offsets_petsc;
  CeedInt *elem_restr_offsets_ceed;

  PetscFunctionBeginUser;
  PetscCall(
      DMPlexGetLocalOffsets(dm, domain_label, label_value, height, dm_field, &num_elem, &elem_size, &num_comp, &num_dof, &elem_restr_offsets_petsc));

  PetscCall(IntArrayP2C(num_elem * elem_size, &elem_restr_offsets_petsc, &elem_restr_offsets_ceed));
  CeedElemRestrictionCreate(ceed, num_elem, elem_size, num_comp, 1, num_dof, CEED_MEM_HOST, CEED_COPY_VALUES, elem_restr_offsets_ceed, elem_restr);
  PetscCall(PetscFree(elem_restr_offsets_ceed));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Utility function to get Ceed Restriction for each domain
PetscErrorCode GetRestrictionForDomain(Ceed ceed, DM dm, CeedInt height, DMLabel domain_label, PetscInt label_value, PetscInt dm_field, CeedInt Q,
                                       CeedInt q_data_size, CeedElemRestriction *elem_restr_q, CeedElemRestriction *elem_restr_x,
                                       CeedElemRestriction *elem_restr_qd_i) {
  DM                  dm_coord;
  CeedInt             loc_num_elem;
  PetscInt            dim;
  CeedElemRestriction elem_restr_tmp;
  PetscFunctionBeginUser;

  PetscCall(DMGetDimension(dm, &dim));
  dim -= height;
  PetscCall(CreateRestrictionFromPlex(ceed, dm, height, domain_label, label_value, dm_field, &elem_restr_tmp));
  if (elem_restr_q) *elem_restr_q = elem_restr_tmp;
  if (elem_restr_x) {
    PetscCall(DMGetCellCoordinateDM(dm, &dm_coord));
    if (!dm_coord) {
      PetscCall(DMGetCoordinateDM(dm, &dm_coord));
    }
    PetscCall(DMPlexSetClosurePermutationTensor(dm_coord, PETSC_DETERMINE, NULL));
    PetscCall(CreateRestrictionFromPlex(ceed, dm_coord, height, domain_label, label_value, dm_field, elem_restr_x));
  }
  if (elem_restr_qd_i) {
    CeedElemRestrictionGetNumElements(elem_restr_tmp, &loc_num_elem);
    CeedElemRestrictionCreateStrided(ceed, loc_num_elem, Q, q_data_size, q_data_size * loc_num_elem * Q, CEED_STRIDES_BACKEND, elem_restr_qd_i);
  }
  if (!elem_restr_q) CeedElemRestrictionDestroy(&elem_restr_tmp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AddBCSubOperator(Ceed ceed, DM dm, CeedData ceed_data, DMLabel domain_label, PetscInt label_value, CeedInt height, CeedInt Q_sur,
                                CeedInt q_data_size_sur, CeedInt jac_data_size_sur, CeedQFunction qf_apply_bc, CeedQFunction qf_apply_bc_jacobian,
                                CeedOperator *op_apply, CeedOperator *op_apply_ijacobian) {
  CeedVector          q_data_sur, jac_data_sur;
  CeedOperator        op_setup_sur, op_apply_bc, op_apply_bc_jacobian = NULL;
  CeedElemRestriction elem_restr_x_sur, elem_restr_q_sur, elem_restr_qd_i_sur, elem_restr_jd_i_sur;
  CeedInt             num_qpts_sur;
  PetscFunctionBeginUser;

  // --- Get number of quadrature points for the boundaries
  CeedBasisGetNumQuadraturePoints(ceed_data->basis_q_sur, &num_qpts_sur);

  // ---- CEED Restriction
  PetscCall(GetRestrictionForDomain(ceed, dm, height, domain_label, label_value, 0, num_qpts_sur, q_data_size_sur, &elem_restr_q_sur,
                                    &elem_restr_x_sur, &elem_restr_qd_i_sur));
  if (jac_data_size_sur > 0) {
    // State-dependent data will be passed from residual to Jacobian. This will be collocated.
    PetscCall(
        GetRestrictionForDomain(ceed, dm, height, domain_label, label_value, 0, num_qpts_sur, jac_data_size_sur, NULL, NULL, &elem_restr_jd_i_sur));
    CeedElemRestrictionCreateVector(elem_restr_jd_i_sur, &jac_data_sur, NULL);
  } else {
    elem_restr_jd_i_sur = NULL;
    jac_data_sur        = NULL;
  }

  // ---- CEED Vector
  CeedInt loc_num_elem_sur;
  CeedElemRestrictionGetNumElements(elem_restr_q_sur, &loc_num_elem_sur);
  CeedVectorCreate(ceed, q_data_size_sur * loc_num_elem_sur * num_qpts_sur, &q_data_sur);

  // ---- CEED Operator
  // ----- CEED Operator for Setup (geometric factors)
  CeedOperatorCreate(ceed, ceed_data->qf_setup_sur, NULL, NULL, &op_setup_sur);
  CeedOperatorSetField(op_setup_sur, "dx", elem_restr_x_sur, ceed_data->basis_x_sur, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_sur, "weight", CEED_ELEMRESTRICTION_NONE, ceed_data->basis_x_sur, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_sur, "surface qdata", elem_restr_qd_i_sur, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // ----- CEED Operator for Physics
  CeedOperatorCreate(ceed, qf_apply_bc, NULL, NULL, &op_apply_bc);
  CeedOperatorSetField(op_apply_bc, "q", elem_restr_q_sur, ceed_data->basis_q_sur, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply_bc, "Grad_q", elem_restr_q_sur, ceed_data->basis_q_sur, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply_bc, "surface qdata", elem_restr_qd_i_sur, CEED_BASIS_COLLOCATED, q_data_sur);
  CeedOperatorSetField(op_apply_bc, "x", elem_restr_x_sur, ceed_data->basis_x_sur, ceed_data->x_coord);
  CeedOperatorSetField(op_apply_bc, "v", elem_restr_q_sur, ceed_data->basis_q_sur, CEED_VECTOR_ACTIVE);
  if (elem_restr_jd_i_sur) CeedOperatorSetField(op_apply_bc, "surface jacobian data", elem_restr_jd_i_sur, CEED_BASIS_COLLOCATED, jac_data_sur);

  if (qf_apply_bc_jacobian) {
    CeedOperatorCreate(ceed, qf_apply_bc_jacobian, NULL, NULL, &op_apply_bc_jacobian);
    CeedOperatorSetField(op_apply_bc_jacobian, "dq", elem_restr_q_sur, ceed_data->basis_q_sur, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_apply_bc_jacobian, "Grad_dq", elem_restr_q_sur, ceed_data->basis_q_sur, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_apply_bc_jacobian, "surface qdata", elem_restr_qd_i_sur, CEED_BASIS_COLLOCATED, q_data_sur);
    CeedOperatorSetField(op_apply_bc_jacobian, "x", elem_restr_x_sur, ceed_data->basis_x_sur, ceed_data->x_coord);
    CeedOperatorSetField(op_apply_bc_jacobian, "surface jacobian data", elem_restr_jd_i_sur, CEED_BASIS_COLLOCATED, jac_data_sur);
    CeedOperatorSetField(op_apply_bc_jacobian, "v", elem_restr_q_sur, ceed_data->basis_q_sur, CEED_VECTOR_ACTIVE);
  }

  // ----- Apply CEED operator for Setup
  CeedOperatorApply(op_setup_sur, ceed_data->x_coord, q_data_sur, CEED_REQUEST_IMMEDIATE);

  // ----- Apply Sub-Operator for Physics
  CeedCompositeOperatorAddSub(*op_apply, op_apply_bc);
  if (op_apply_bc_jacobian) CeedCompositeOperatorAddSub(*op_apply_ijacobian, op_apply_bc_jacobian);

  // ----- Cleanup
  CeedVectorDestroy(&q_data_sur);
  CeedVectorDestroy(&jac_data_sur);
  CeedElemRestrictionDestroy(&elem_restr_q_sur);
  CeedElemRestrictionDestroy(&elem_restr_x_sur);
  CeedElemRestrictionDestroy(&elem_restr_qd_i_sur);
  CeedElemRestrictionDestroy(&elem_restr_jd_i_sur);
  CeedOperatorDestroy(&op_setup_sur);
  CeedOperatorDestroy(&op_apply_bc);
  CeedOperatorDestroy(&op_apply_bc_jacobian);

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Utility function to create CEED Composite Operator for the entire domain
PetscErrorCode CreateOperatorForDomain(Ceed ceed, DM dm, SimpleBC bc, CeedData ceed_data, Physics phys, CeedOperator op_apply_vol,
                                       CeedOperator op_apply_ijacobian_vol, CeedInt height, CeedInt P_sur, CeedInt Q_sur, CeedInt q_data_size_sur,
                                       CeedInt jac_data_size_sur, CeedOperator *op_apply, CeedOperator *op_apply_ijacobian) {
  DMLabel domain_label;
  PetscFunctionBeginUser;

  // Create Composite Operaters
  CeedCompositeOperatorCreate(ceed, op_apply);
  if (op_apply_ijacobian) CeedCompositeOperatorCreate(ceed, op_apply_ijacobian);

  // --Apply Sub-Operator for the volume
  CeedCompositeOperatorAddSub(*op_apply, op_apply_vol);
  if (op_apply_ijacobian) CeedCompositeOperatorAddSub(*op_apply_ijacobian, op_apply_ijacobian_vol);

  // -- Create Sub-Operator for in/outflow BCs
  if (phys->has_neumann || 1) {
    // --- Setup
    PetscCall(DMGetLabel(dm, "Face Sets", &domain_label));

    // --- Create Sub-Operator for inflow boundaries
    for (CeedInt i = 0; i < bc->num_inflow; i++) {
      PetscCall(AddBCSubOperator(ceed, dm, ceed_data, domain_label, bc->inflows[i], height, Q_sur, q_data_size_sur, jac_data_size_sur,
                                 ceed_data->qf_apply_inflow, ceed_data->qf_apply_inflow_jacobian, op_apply, op_apply_ijacobian));
    }
    // --- Create Sub-Operator for outflow boundaries
    for (CeedInt i = 0; i < bc->num_outflow; i++) {
      PetscCall(AddBCSubOperator(ceed, dm, ceed_data, domain_label, bc->outflows[i], height, Q_sur, q_data_size_sur, jac_data_size_sur,
                                 ceed_data->qf_apply_outflow, ceed_data->qf_apply_outflow_jacobian, op_apply, op_apply_ijacobian));
    }
    // --- Create Sub-Operator for freestream boundaries
    for (CeedInt i = 0; i < bc->num_freestream; i++) {
      PetscCall(AddBCSubOperator(ceed, dm, ceed_data, domain_label, bc->freestreams[i], height, Q_sur, q_data_size_sur, jac_data_size_sur,
                                 ceed_data->qf_apply_freestream, ceed_data->qf_apply_freestream_jacobian, op_apply, op_apply_ijacobian));
    }
  }

  // ----- Get Context Labels for Operator
  CeedOperatorGetContextFieldLabel(*op_apply, "solution time", &phys->solution_time_label);
  CeedOperatorGetContextFieldLabel(*op_apply, "timestep size", &phys->timestep_size_label);

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetupBCQFunctions(Ceed ceed, PetscInt dim_sur, PetscInt num_comp_x, PetscInt num_comp_q, PetscInt q_data_size_sur,
                                 PetscInt jac_data_size_sur, ProblemQFunctionSpec apply_bc, ProblemQFunctionSpec apply_bc_jacobian,
                                 CeedQFunction *qf_apply_bc, CeedQFunction *qf_apply_bc_jacobian) {
  PetscFunctionBeginUser;

  if (apply_bc.qfunction) {
    CeedQFunctionCreateInterior(ceed, 1, apply_bc.qfunction, apply_bc.qfunction_loc, qf_apply_bc);
    CeedQFunctionSetContext(*qf_apply_bc, apply_bc.qfunction_context);
    CeedQFunctionAddInput(*qf_apply_bc, "q", num_comp_q, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(*qf_apply_bc, "Grad_q", num_comp_q * dim_sur, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(*qf_apply_bc, "surface qdata", q_data_size_sur, CEED_EVAL_NONE);
    CeedQFunctionAddInput(*qf_apply_bc, "x", num_comp_x, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(*qf_apply_bc, "v", num_comp_q, CEED_EVAL_INTERP);
    if (jac_data_size_sur) CeedQFunctionAddOutput(*qf_apply_bc, "surface jacobian data", jac_data_size_sur, CEED_EVAL_NONE);
  }
  if (apply_bc_jacobian.qfunction) {
    CeedQFunctionCreateInterior(ceed, 1, apply_bc_jacobian.qfunction, apply_bc_jacobian.qfunction_loc, qf_apply_bc_jacobian);
    CeedQFunctionSetContext(*qf_apply_bc_jacobian, apply_bc_jacobian.qfunction_context);
    CeedQFunctionAddInput(*qf_apply_bc_jacobian, "dq", num_comp_q, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(*qf_apply_bc_jacobian, "Grad_dq", num_comp_q * dim_sur, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(*qf_apply_bc_jacobian, "surface qdata", q_data_size_sur, CEED_EVAL_NONE);
    CeedQFunctionAddInput(*qf_apply_bc_jacobian, "x", num_comp_x, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(*qf_apply_bc_jacobian, "surface jacobian data", jac_data_size_sur, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(*qf_apply_bc_jacobian, "v", num_comp_q, CEED_EVAL_INTERP);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
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

  if (*ds_field == -1) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Could not find dm_field %" PetscInt_FMT " in DS", dm_field);

  PetscFunctionReturn(PETSC_SUCCESS);
}

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
  CeedBasisCreateH1(ceed, elem_topo, num_comp, P, Q, interp, grad, q_points, q_weights, basis);
  PetscCall(PetscFree(q_points));
  PetscCall(PetscFree(interp));
  PetscCall(PetscFree(grad));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// -----------------------------------------------------------------------------
// Get CEED Basis from DMPlex
// -----------------------------------------------------------------------------
PetscErrorCode CreateBasisFromPlex(Ceed ceed, DM dm, DMLabel domain_label, CeedInt label_value, CeedInt height, CeedInt dm_field,
                                   CeedQuadMode quadMode, CeedBasis *basis) {
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

    CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp, P_1d, Q_1d, quadMode, basis);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetupLibceed(Ceed ceed, CeedData ceed_data, DM dm, User user, AppCtx app_ctx, ProblemData *problem, SimpleBC bc) {
  PetscFunctionBeginUser;

  // *****************************************************************************
  // Set up CEED objects for the interior domain (volume)
  // *****************************************************************************
  const PetscInt num_comp_q = 5;
  const CeedInt  dim = problem->dim, num_comp_x = problem->dim, q_data_size_vol = problem->q_data_size_vol, jac_data_size_vol = num_comp_q + 6 + 3;
  CeedElemRestriction elem_restr_jd_i;
  CeedVector          jac_data;
  CeedInt             num_qpts;

  // -----------------------------------------------------------------------------
  // CEED Bases
  // -----------------------------------------------------------------------------
  DM dm_coord;
  PetscCall(DMGetCoordinateDM(dm, &dm_coord));

  PetscCall(CreateBasisFromPlex(ceed, dm, 0, 0, 0, 0, CEED_GAUSS, &ceed_data->basis_q));
  PetscCall(CreateBasisFromPlex(ceed, dm_coord, 0, 0, 0, 0, CEED_GAUSS, &ceed_data->basis_x));
  PetscCall(CeedBasisCreateProjection(ceed_data->basis_x, ceed_data->basis_q, &ceed_data->basis_xc));
  CeedBasisGetNumQuadraturePoints(ceed_data->basis_q, &num_qpts);

  // -----------------------------------------------------------------------------
  // CEED Restrictions
  // -----------------------------------------------------------------------------
  // -- Create restriction
  PetscCall(GetRestrictionForDomain(ceed, dm, 0, 0, 0, 0, num_qpts, q_data_size_vol, &ceed_data->elem_restr_q, &ceed_data->elem_restr_x,
                                    &ceed_data->elem_restr_qd_i));

  PetscCall(GetRestrictionForDomain(ceed, dm, 0, 0, 0, 0, num_qpts, jac_data_size_vol, NULL, NULL, &elem_restr_jd_i));
  // -- Create E vectors
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &user->q_ceed, NULL);
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &user->q_dot_ceed, NULL);
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &user->g_ceed, NULL);

  // -----------------------------------------------------------------------------
  // CEED QFunctions
  // -----------------------------------------------------------------------------
  // -- Create QFunction for quadrature data
  CeedQFunctionCreateInterior(ceed, 1, problem->setup_vol.qfunction, problem->setup_vol.qfunction_loc, &ceed_data->qf_setup_vol);
  if (problem->setup_vol.qfunction_context) {
    CeedQFunctionSetContext(ceed_data->qf_setup_vol, problem->setup_vol.qfunction_context);
  }
  CeedQFunctionAddInput(ceed_data->qf_setup_vol, "dx", num_comp_x * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(ceed_data->qf_setup_vol, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(ceed_data->qf_setup_vol, "qdata", q_data_size_vol, CEED_EVAL_NONE);

  // -- Create QFunction for ICs
  CeedQFunctionCreateInterior(ceed, 1, problem->ics.qfunction, problem->ics.qfunction_loc, &ceed_data->qf_ics);
  CeedQFunctionSetContext(ceed_data->qf_ics, problem->ics.qfunction_context);
  CeedQFunctionAddInput(ceed_data->qf_ics, "x", num_comp_x, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(ceed_data->qf_ics, "dx", num_comp_x * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(ceed_data->qf_ics, "q0", num_comp_q, CEED_EVAL_NONE);

  // -- Create QFunction for RHS
  if (problem->apply_vol_rhs.qfunction) {
    CeedQFunctionCreateInterior(ceed, 1, problem->apply_vol_rhs.qfunction, problem->apply_vol_rhs.qfunction_loc, &ceed_data->qf_rhs_vol);
    CeedQFunctionSetContext(ceed_data->qf_rhs_vol, problem->apply_vol_rhs.qfunction_context);
    CeedQFunctionAddInput(ceed_data->qf_rhs_vol, "q", num_comp_q, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(ceed_data->qf_rhs_vol, "Grad_q", num_comp_q * dim, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(ceed_data->qf_rhs_vol, "qdata", q_data_size_vol, CEED_EVAL_NONE);
    CeedQFunctionAddInput(ceed_data->qf_rhs_vol, "x", num_comp_x, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(ceed_data->qf_rhs_vol, "v", num_comp_q, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(ceed_data->qf_rhs_vol, "Grad_v", num_comp_q * dim, CEED_EVAL_GRAD);
  }

  // -- Create QFunction for IFunction
  if (problem->apply_vol_ifunction.qfunction) {
    CeedQFunctionCreateInterior(ceed, 1, problem->apply_vol_ifunction.qfunction, problem->apply_vol_ifunction.qfunction_loc,
                                &ceed_data->qf_ifunction_vol);
    CeedQFunctionSetContext(ceed_data->qf_ifunction_vol, problem->apply_vol_ifunction.qfunction_context);
    CeedQFunctionAddInput(ceed_data->qf_ifunction_vol, "q", num_comp_q, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(ceed_data->qf_ifunction_vol, "Grad_q", num_comp_q * dim, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(ceed_data->qf_ifunction_vol, "q dot", num_comp_q, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(ceed_data->qf_ifunction_vol, "qdata", q_data_size_vol, CEED_EVAL_NONE);
    CeedQFunctionAddInput(ceed_data->qf_ifunction_vol, "x", num_comp_x, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(ceed_data->qf_ifunction_vol, "v", num_comp_q, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(ceed_data->qf_ifunction_vol, "Grad_v", num_comp_q * dim, CEED_EVAL_GRAD);
    CeedQFunctionAddOutput(ceed_data->qf_ifunction_vol, "jac_data", jac_data_size_vol, CEED_EVAL_NONE);
  }

  CeedQFunction qf_ijacobian_vol = NULL;
  if (problem->apply_vol_ijacobian.qfunction) {
    CeedQFunctionCreateInterior(ceed, 1, problem->apply_vol_ijacobian.qfunction, problem->apply_vol_ijacobian.qfunction_loc, &qf_ijacobian_vol);
    CeedQFunctionSetContext(qf_ijacobian_vol, problem->apply_vol_ijacobian.qfunction_context);
    CeedQFunctionAddInput(qf_ijacobian_vol, "dq", num_comp_q, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_ijacobian_vol, "Grad_dq", num_comp_q * dim, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(qf_ijacobian_vol, "qdata", q_data_size_vol, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qf_ijacobian_vol, "x", num_comp_x, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_ijacobian_vol, "jac_data", jac_data_size_vol, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qf_ijacobian_vol, "v", num_comp_q, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_ijacobian_vol, "Grad_v", num_comp_q * dim, CEED_EVAL_GRAD);
  }

  // ---------------------------------------------------------------------------
  // Element coordinates
  // ---------------------------------------------------------------------------
  // -- Create CEED vector
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_x, &ceed_data->x_coord, NULL);

  // -- Copy PETSc vector in CEED vector
  Vec X_loc;
  {
    DM cdm;
    PetscCall(DMGetCellCoordinateDM(dm, &cdm));
    if (cdm) {
      PetscCall(DMGetCellCoordinatesLocal(dm, &X_loc));
    } else {
      PetscCall(DMGetCoordinatesLocal(dm, &X_loc));
    }
  }
  PetscCall(VecScale(X_loc, problem->dm_scale));
  PetscCall(VecCopyP2C(X_loc, ceed_data->x_coord));

  // -----------------------------------------------------------------------------
  // CEED vectors
  // -----------------------------------------------------------------------------
  // -- Create CEED vector for geometric data
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_qd_i, &ceed_data->q_data, NULL);
  CeedElemRestrictionCreateVector(elem_restr_jd_i, &jac_data, NULL);

  // -----------------------------------------------------------------------------
  // CEED Operators
  // -----------------------------------------------------------------------------
  // -- Create CEED operator for quadrature data
  CeedOperatorCreate(ceed, ceed_data->qf_setup_vol, NULL, NULL, &ceed_data->op_setup_vol);
  CeedOperatorSetField(ceed_data->op_setup_vol, "dx", ceed_data->elem_restr_x, ceed_data->basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(ceed_data->op_setup_vol, "weight", CEED_ELEMRESTRICTION_NONE, ceed_data->basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(ceed_data->op_setup_vol, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // -- Create CEED operator for ICs
  CeedOperator op_ics;
  CeedOperatorCreate(ceed, ceed_data->qf_ics, NULL, NULL, &op_ics);
  CeedOperatorSetField(op_ics, "x", ceed_data->elem_restr_x, ceed_data->basis_xc, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_ics, "dx", ceed_data->elem_restr_x, ceed_data->basis_xc, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_ics, "q0", ceed_data->elem_restr_q, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedOperatorGetContextFieldLabel(op_ics, "evaluation time", &user->phys->ics_time_label);
  PetscCall(OperatorApplyContextCreate(NULL, dm, user->ceed, op_ics, ceed_data->x_coord, NULL, NULL, user->Q_loc, &ceed_data->op_ics_ctx));
  CeedOperatorDestroy(&op_ics);

  // Create CEED operator for RHS
  if (ceed_data->qf_rhs_vol) {
    CeedOperator op;
    CeedOperatorCreate(ceed, ceed_data->qf_rhs_vol, NULL, NULL, &op);
    CeedOperatorSetField(op, "q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "Grad_q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_COLLOCATED, ceed_data->q_data);
    CeedOperatorSetField(op, "x", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord);
    CeedOperatorSetField(op, "v", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "Grad_v", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE);
    user->op_rhs_vol = op;
  }

  // -- CEED operator for IFunction
  if (ceed_data->qf_ifunction_vol) {
    CeedOperator op;
    CeedOperatorCreate(ceed, ceed_data->qf_ifunction_vol, NULL, NULL, &op);
    CeedOperatorSetField(op, "q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "Grad_q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "q dot", ceed_data->elem_restr_q, ceed_data->basis_q, user->q_dot_ceed);
    CeedOperatorSetField(op, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_COLLOCATED, ceed_data->q_data);
    CeedOperatorSetField(op, "x", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord);
    CeedOperatorSetField(op, "v", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "Grad_v", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "jac_data", elem_restr_jd_i, CEED_BASIS_COLLOCATED, jac_data);

    user->op_ifunction_vol = op;
  }

  CeedOperator op_ijacobian_vol = NULL;
  if (qf_ijacobian_vol) {
    CeedOperator op;
    CeedOperatorCreate(ceed, qf_ijacobian_vol, NULL, NULL, &op);
    CeedOperatorSetField(op, "dq", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "Grad_dq", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_COLLOCATED, ceed_data->q_data);
    CeedOperatorSetField(op, "x", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord);
    CeedOperatorSetField(op, "jac_data", elem_restr_jd_i, CEED_BASIS_COLLOCATED, jac_data);
    CeedOperatorSetField(op, "v", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "Grad_v", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE);
    op_ijacobian_vol = op;
    CeedQFunctionDestroy(&qf_ijacobian_vol);
  }

  // *****************************************************************************
  // Set up CEED objects for the exterior domain (surface)
  // *****************************************************************************
  CeedInt       height = 1, dim_sur = dim - height, P_sur = app_ctx->degree + 1, Q_sur = P_sur + app_ctx->q_extra;
  const CeedInt q_data_size_sur = problem->q_data_size_sur, jac_data_size_sur = problem->jac_data_size_sur;

  // -----------------------------------------------------------------------------
  // CEED Bases
  // -----------------------------------------------------------------------------

  DMLabel  label   = 0;
  PetscInt face_id = 0;
  PetscInt field   = 0;  // Still want the normal, default field
  PetscCall(CreateBasisFromPlex(ceed, dm, label, face_id, height, field, CEED_GAUSS, &ceed_data->basis_q_sur));
  PetscCall(CreateBasisFromPlex(ceed, dm_coord, label, face_id, height, field, CEED_GAUSS, &ceed_data->basis_x_sur));
  PetscCall(CeedBasisCreateProjection(ceed_data->basis_x_sur, ceed_data->basis_q_sur, &ceed_data->basis_xc_sur));

  // -----------------------------------------------------------------------------
  // CEED QFunctions
  // -----------------------------------------------------------------------------
  // -- Create QFunction for quadrature data
  CeedQFunctionCreateInterior(ceed, 1, problem->setup_sur.qfunction, problem->setup_sur.qfunction_loc, &ceed_data->qf_setup_sur);
  if (problem->setup_sur.qfunction_context) {
    CeedQFunctionSetContext(ceed_data->qf_setup_sur, problem->setup_sur.qfunction_context);
  }
  CeedQFunctionAddInput(ceed_data->qf_setup_sur, "dx", num_comp_x * dim_sur, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(ceed_data->qf_setup_sur, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(ceed_data->qf_setup_sur, "surface qdata", q_data_size_sur, CEED_EVAL_NONE);

  PetscCall(SetupBCQFunctions(ceed, dim_sur, num_comp_x, num_comp_q, q_data_size_sur, jac_data_size_sur, problem->apply_inflow,
                              problem->apply_inflow_jacobian, &ceed_data->qf_apply_inflow, &ceed_data->qf_apply_inflow_jacobian));
  PetscCall(SetupBCQFunctions(ceed, dim_sur, num_comp_x, num_comp_q, q_data_size_sur, jac_data_size_sur, problem->apply_outflow,
                              problem->apply_outflow_jacobian, &ceed_data->qf_apply_outflow, &ceed_data->qf_apply_outflow_jacobian));
  PetscCall(SetupBCQFunctions(ceed, dim_sur, num_comp_x, num_comp_q, q_data_size_sur, jac_data_size_sur, problem->apply_freestream,
                              problem->apply_freestream_jacobian, &ceed_data->qf_apply_freestream, &ceed_data->qf_apply_freestream_jacobian));

  // *****************************************************************************
  // CEED Operator Apply
  // *****************************************************************************
  // -- Apply CEED Operator for the geometric data
  CeedOperatorApply(ceed_data->op_setup_vol, ceed_data->x_coord, ceed_data->q_data, CEED_REQUEST_IMMEDIATE);

  // -- Create and apply CEED Composite Operator for the entire domain
  if (!user->phys->implicit) {  // RHS
    CeedOperator op_rhs;
    PetscCall(CreateOperatorForDomain(ceed, dm, bc, ceed_data, user->phys, user->op_rhs_vol, NULL, height, P_sur, Q_sur, q_data_size_sur, 0, &op_rhs,
                                      NULL));
    PetscCall(OperatorApplyContextCreate(dm, dm, ceed, op_rhs, user->q_ceed, user->g_ceed, user->Q_loc, NULL, &user->op_rhs_ctx));
    CeedOperatorDestroy(&op_rhs);
  } else {  // IFunction
    PetscCall(CreateOperatorForDomain(ceed, dm, bc, ceed_data, user->phys, user->op_ifunction_vol, op_ijacobian_vol, height, P_sur, Q_sur,
                                      q_data_size_sur, jac_data_size_sur, &user->op_ifunction, op_ijacobian_vol ? &user->op_ijacobian : NULL));
    if (user->op_ijacobian) {
      CeedOperatorGetContextFieldLabel(user->op_ijacobian, "ijacobian time shift", &user->phys->ijacobian_time_shift_label);
    }
    if (problem->use_strong_bc_ceed) PetscCall(SetupStrongBC_Ceed(ceed, ceed_data, dm, user, problem, bc, Q_sur, q_data_size_sur));
    if (app_ctx->sgs_model_type == SGS_MODEL_DATA_DRIVEN) PetscCall(SGS_DD_ModelSetup(ceed, user, ceed_data, problem));
  }

  if (app_ctx->turb_spanstats_enable) PetscCall(TurbulenceStatisticsSetup(ceed, user, ceed_data, problem));
  if (app_ctx->diff_filter_monitor) PetscCall(DifferentialFilterSetup(ceed, user, ceed_data, problem));

  CeedElemRestrictionDestroy(&elem_restr_jd_i);
  CeedOperatorDestroy(&op_ijacobian_vol);
  CeedVectorDestroy(&jac_data);
  PetscFunctionReturn(PETSC_SUCCESS);
}
