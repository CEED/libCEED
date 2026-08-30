// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../include/libceedsetup.h"

#include <stdio.h>

#include "../include/libceedsetup.h"
#include "../include/petscutils.h"

// -----------------------------------------------------------------------------
// Destroy libCEED operator objects
// -----------------------------------------------------------------------------
PetscErrorCode CeedDataDestroy(CeedInt i, CeedData data) {
  PetscFunctionBeginUser;
  CeedVectorDestroy(&data->q_data);
  CeedVectorDestroy(&data->x_ceed);
  CeedVectorDestroy(&data->y_ceed);
  CeedBasisDestroy(&data->basis_x);
  CeedBasisDestroy(&data->basis_u);
  CeedElemRestrictionDestroy(&data->elem_restr_u);
  CeedElemRestrictionDestroy(&data->elem_restr_x);
  CeedElemRestrictionDestroy(&data->elem_restr_u_i);
  CeedElemRestrictionDestroy(&data->elem_restr_qd_i);
  CeedQFunctionDestroy(&data->qf_apply);
  CeedOperatorDestroy(&data->op_apply);
  if (i > 0) {
    CeedOperatorDestroy(&data->op_prolong);
    CeedOperatorDestroy(&data->op_restrict);
  }
  PetscCall(PetscFree(data));
  PetscFunctionReturn(PETSC_SUCCESS);
};

// -----------------------------------------------------------------------------
// Destroy libCEED BDDC objects
// -----------------------------------------------------------------------------
PetscErrorCode CeedDataBDDCDestroy(CeedDataBDDC data) {
  PetscFunctionBeginUser;

  CeedBasisDestroy(&data->basis_Pi);
  CeedBasisDestroy(&data->basis_Pi_r);
  CeedElemRestrictionDestroy(&data->elem_restr_Pi);
  CeedElemRestrictionDestroy(&data->elem_restr_Pi_r);
  CeedElemRestrictionDestroy(&data->elem_restr_r);
  CeedOperatorDestroy(&data->op_Pi_r);
  CeedOperatorDestroy(&data->op_r_Pi);
  CeedOperatorDestroy(&data->op_Pi_Pi);
  CeedOperatorDestroy(&data->op_r_r);
  CeedOperatorDestroy(&data->op_r_r_inv);
  CeedOperatorDestroy(&data->op_inject_Pi);
  CeedOperatorDestroy(&data->op_inject_Pi_r);
  CeedOperatorDestroy(&data->op_inject_r);
  CeedOperatorDestroy(&data->op_restrict_Pi);
  CeedOperatorDestroy(&data->op_restrict_Pi_r);
  CeedOperatorDestroy(&data->op_restrict_r);
  CeedVectorDestroy(&data->x_Pi_ceed);
  CeedVectorDestroy(&data->y_Pi_ceed);
  CeedVectorDestroy(&data->x_Pi_r_ceed);
  CeedVectorDestroy(&data->y_Pi_r_ceed);
  CeedVectorDestroy(&data->x_r_ceed);
  CeedVectorDestroy(&data->y_r_ceed);
  CeedVectorDestroy(&data->z_r_ceed);
  CeedVectorDestroy(&data->mult_ceed);
  CeedVectorDestroy(&data->mask_r_ceed);
  CeedVectorDestroy(&data->mask_Gamma_ceed);
  CeedVectorDestroy(&data->mask_I_ceed);
  PetscCall(PetscFree(data));

  PetscFunctionReturn(PETSC_SUCCESS);
};

// -----------------------------------------------------------------------------
// Set up libCEED for a given degree
// -----------------------------------------------------------------------------
PetscErrorCode SetupLibceedByDegree(DM dm, Ceed ceed, CeedInt degree, CeedInt topo_dim, CeedInt q_extra, PetscInt num_comp_x, PetscInt num_comp_u,
                                    PetscInt g_size, PetscInt xl_size, BPData bp_data, CeedData data, PetscBool setup_rhs, PetscBool is_fine_level,
                                    CeedVector rhs_ceed, CeedVector *target) {
  DM                  dm_coord;
  Vec                 coords;
  const PetscScalar  *coord_array;
  CeedBasis           basis_x, basis_u;
  CeedElemRestriction elem_restr_x, elem_restr_u, elem_restr_u_i, elem_restr_qd_i;
  CeedQFunction       qf_setup_geo = NULL, qf_apply = NULL;
  CeedOperator        op_setup_geo, op_apply;
  CeedVector          x_coord, q_data, x_ceed, y_ceed;
  PetscInt            c_start, c_end, num_elem;
  CeedInt             num_qpts, q_data_size = bp_data.q_data_size;
  CeedScalar          R = 1;                         // radius of the sphere
  CeedScalar          l = 1.0 / PetscSqrtReal(3.0);  // half edge of the inscribed cube

  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinateDM(dm, &dm_coord));

  // CEED bases
  PetscCall(CreateBasisFromPlex(ceed, dm_coord, 0, 0, 0, 0, bp_data, &basis_x));
  PetscCall(CreateBasisFromPlex(ceed, dm, 0, 0, 0, 0, bp_data, &basis_u));

  // CEED restrictions
  PetscCall(CreateRestrictionFromPlex(ceed, dm_coord, 0, 0, 0, &elem_restr_x));
  PetscCall(CreateRestrictionFromPlex(ceed, dm, 0, 0, 0, &elem_restr_u));

  PetscCall(DMPlexGetHeightStratum(dm, 0, &c_start, &c_end));
  num_elem = c_end - c_start;
  CeedBasisGetNumQuadraturePoints(basis_u, &num_qpts);

  CeedElemRestrictionCreateStrided(ceed, num_elem, num_qpts, num_comp_u, num_comp_u * num_elem * num_qpts, CEED_STRIDES_BACKEND, &elem_restr_u_i);
  CeedElemRestrictionCreateStrided(ceed, num_elem, num_qpts, q_data_size, q_data_size * num_elem * num_qpts, CEED_STRIDES_BACKEND, &elem_restr_qd_i);

  // Element coordinates
  PetscCall(DMGetCoordinatesLocal(dm, &coords));
  PetscCall(VecGetArrayRead(coords, &coord_array));

  CeedElemRestrictionCreateVector(elem_restr_x, &x_coord, NULL);
  CeedVectorSetArray(x_coord, CEED_MEM_HOST, CEED_COPY_VALUES, (PetscScalar *)coord_array);
  PetscCall(VecRestoreArrayRead(coords, &coord_array));

  // Create the persistent vectors that will be needed in setup and apply
  CeedVectorCreate(ceed, q_data_size * num_elem * num_qpts, &q_data);
  CeedVectorCreate(ceed, xl_size, &x_ceed);
  CeedVectorCreate(ceed, xl_size, &y_ceed);

  if (is_fine_level) {
    // Create the QFunction that builds the context data
    CeedQFunctionCreateInterior(ceed, 1, bp_data.setup_geo, bp_data.setup_geo_loc, &qf_setup_geo);
    CeedQFunctionAddInput(qf_setup_geo, "x", num_comp_x, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_setup_geo, "dx", num_comp_x * topo_dim, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(qf_setup_geo, "weight", 1, CEED_EVAL_WEIGHT);
    CeedQFunctionAddOutput(qf_setup_geo, "qdata", q_data_size, CEED_EVAL_NONE);

    // Create the operator that builds the quadrature data
    CeedOperatorCreate(ceed, qf_setup_geo, NULL, NULL, &op_setup_geo);
    CeedOperatorSetField(op_setup_geo, "x", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_setup_geo, "dx", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_setup_geo, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
    CeedOperatorSetField(op_setup_geo, "qdata", elem_restr_qd_i, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);

    // Setup q_data
    CeedOperatorApply(op_setup_geo, x_coord, q_data, CEED_REQUEST_IMMEDIATE);

    // Set up PDE operator
    PetscBool is_interp = bp_data.in_mode == CEED_EVAL_INTERP;
    CeedInt   in_scale  = bp_data.in_mode == CEED_EVAL_GRAD ? topo_dim : 1;
    CeedInt   out_scale = bp_data.out_mode == CEED_EVAL_GRAD ? topo_dim : 1;

    CeedQFunctionCreateInterior(ceed, 1, bp_data.apply, bp_data.apply_loc, &qf_apply);
    if (bp_data.in_mode == CEED_EVAL_INTERP + CEED_EVAL_GRAD) {
      CeedQFunctionAddInput(qf_apply, "u", num_comp_u, CEED_EVAL_INTERP);
      CeedQFunctionAddInput(qf_apply, "du", num_comp_u * topo_dim, CEED_EVAL_GRAD);
    } else {
      CeedQFunctionAddInput(qf_apply, is_interp ? "u" : "du", num_comp_u * in_scale, bp_data.in_mode);
    }
    CeedQFunctionAddInput(qf_apply, "qdata", q_data_size, CEED_EVAL_NONE);
    if (bp_data.out_mode == CEED_EVAL_INTERP + CEED_EVAL_GRAD) {
      CeedQFunctionAddOutput(qf_apply, "v", num_comp_u, CEED_EVAL_INTERP);
      CeedQFunctionAddOutput(qf_apply, "dv", num_comp_u * topo_dim, CEED_EVAL_GRAD);
    } else {
      CeedQFunctionAddOutput(qf_apply, is_interp ? "v" : "dv", num_comp_u * out_scale, bp_data.out_mode);
    }

    // Create the mass or diff operator
    CeedOperatorCreate(ceed, qf_apply, NULL, NULL, &op_apply);
    if (bp_data.in_mode == CEED_EVAL_INTERP + CEED_EVAL_GRAD) {
      CeedOperatorSetField(op_apply, "u", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);
      CeedOperatorSetField(op_apply, "du", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);
    } else {
      CeedOperatorSetField(op_apply, is_interp ? "u" : "du", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);
    }
    CeedOperatorSetField(op_apply, "qdata", elem_restr_qd_i, CEED_BASIS_NONE, q_data);
    if (bp_data.out_mode == CEED_EVAL_INTERP + CEED_EVAL_GRAD) {
      CeedOperatorSetField(op_apply, "v", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);
      CeedOperatorSetField(op_apply, "dv", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);
    } else {
      CeedOperatorSetField(op_apply, is_interp ? "v" : "dv", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);
    }

    // Cleanup
    CeedQFunctionDestroy(&qf_setup_geo);
    CeedOperatorDestroy(&op_setup_geo);
  }

  // Set up RHS if needed
  if (setup_rhs) {
    CeedQFunction qf_setup_rhs;
    CeedOperator  op_setup_rhs;
    CeedVectorCreate(ceed, num_elem * num_qpts * num_comp_u, target);
    // Create the q-function that sets up the RHS and true solution
    CeedQFunctionCreateInterior(ceed, 1, bp_data.setup_rhs, bp_data.setup_rhs_loc, &qf_setup_rhs);
    CeedQFunctionAddInput(qf_setup_rhs, "x", num_comp_x, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_setup_rhs, "qdata", q_data_size, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qf_setup_rhs, "true solution", num_comp_u, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qf_setup_rhs, "rhs", num_comp_u, CEED_EVAL_INTERP);

    // Create the operator that builds the RHS and true solution
    CeedOperatorCreate(ceed, qf_setup_rhs, NULL, NULL, &op_setup_rhs);
    CeedOperatorSetField(op_setup_rhs, "x", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_setup_rhs, "qdata", elem_restr_qd_i, CEED_BASIS_NONE, q_data);
    CeedOperatorSetField(op_setup_rhs, "true solution", elem_restr_u_i, CEED_BASIS_NONE, *target);
    CeedOperatorSetField(op_setup_rhs, "rhs", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);

    // Set up the libCEED context
    CeedQFunctionContext ctx_rhs_setup;
    CeedQFunctionContextCreate(ceed, &ctx_rhs_setup);
    CeedScalar rhs_setup_data[2] = {R, l};
    CeedQFunctionContextSetData(ctx_rhs_setup, CEED_MEM_HOST, CEED_COPY_VALUES, sizeof rhs_setup_data, &rhs_setup_data);
    CeedQFunctionSetContext(qf_setup_rhs, ctx_rhs_setup);
    CeedQFunctionContextDestroy(&ctx_rhs_setup);

    // Setup RHS and target
    CeedOperatorApply(op_setup_rhs, x_coord, rhs_ceed, CEED_REQUEST_IMMEDIATE);

    // Cleanup
    CeedQFunctionDestroy(&qf_setup_rhs);
    CeedOperatorDestroy(&op_setup_rhs);
  }
  // Cleanup
  CeedVectorDestroy(&x_coord);

  // Save libCEED data required for level
  data->ceed            = ceed;
  data->basis_x         = basis_x;
  data->basis_u         = basis_u;
  data->elem_restr_x    = elem_restr_x;
  data->elem_restr_u    = elem_restr_u;
  data->elem_restr_u_i  = elem_restr_u_i;
  data->elem_restr_qd_i = elem_restr_qd_i;
  data->qf_apply        = qf_apply;
  data->op_apply        = op_apply;
  data->q_data          = q_data;
  data->x_ceed          = x_ceed;
  data->y_ceed          = y_ceed;
  data->q_data_size     = q_data_size;
  PetscFunctionReturn(PETSC_SUCCESS);
};

// -----------------------------------------------------------------------------
// Setup libCEED level transfer operator objects
// -----------------------------------------------------------------------------
PetscErrorCode CeedLevelTransferSetup(DM dm, Ceed ceed, CeedInt level, CeedInt num_comp_u, CeedData *data, BPData bp_data, Vec fine_mult) {
  PetscFunctionBeginUser;

  // Restriction - Fine to corse
  CeedOperator op_restrict;
  // Interpolation - Corse to fine
  CeedOperator op_prolong;
  // Coarse grid operator
  CeedOperator op_apply;
  // Basis
  CeedBasis basis_u;
  PetscCall(CreateBasisFromPlex(ceed, dm, 0, 0, 0, 0, bp_data, &basis_u));

  // ---------------------------------------------------------------------------
  // Coarse Grid, Prolongation, and Restriction Operators
  // ---------------------------------------------------------------------------
  // Create the Operators that compute the prolongation and
  //   restriction between the p-multigrid levels and the coarse grid eval.
  // ---------------------------------------------------------------------------
  // Place in libCEED array
  PetscMemType m_mem_type;
  PetscCall(VecReadP2C(fine_mult, &m_mem_type, data[level]->x_ceed));

  CeedOperatorMultigridLevelCreate(data[level]->op_apply, data[level]->x_ceed, data[level - 1]->elem_restr_u, basis_u, &op_apply, &op_prolong,
                                   &op_restrict);

  // Restore PETSc vector
  PetscCall(VecReadC2P(data[level]->x_ceed, m_mem_type, fine_mult));
  PetscCall(VecZeroEntries(fine_mult));
  // -- Save libCEED data
  data[level - 1]->op_apply = op_apply;
  data[level]->op_prolong   = op_prolong;
  data[level]->op_restrict  = op_restrict;

  CeedBasisDestroy(&basis_u);
  PetscFunctionReturn(PETSC_SUCCESS);
};

// -----------------------------------------------------------------------------
// Set up libCEED for BDDC interface vertices
// -----------------------------------------------------------------------------
PetscErrorCode SetupLibceedBDDC(DM dm_Pi, CeedData data_fine, CeedDataBDDC data_bddc, PetscInt g_vertex_size, PetscInt xl_vertex_size,
                                BPData bp_data) {
  Ceed                ceed = data_fine->ceed;
  CeedBasis           basis_Pi, basis_Pi_r, basis_u = data_fine->basis_u;
  CeedElemRestriction elem_restr_Pi, elem_restr_Pi_r, elem_restr_r;
  CeedOperator        op_Pi_r, op_r_Pi, op_Pi_Pi, op_r_r, op_r_r_inv, op_inject_Pi, op_inject_Pi_r, op_inject_r, op_restrict_Pi, op_restrict_Pi_r,
      op_restrict_r;
  CeedVector x_Pi_ceed, y_Pi_ceed, x_Pi_r_ceed, y_Pi_r_ceed, mult_ceed, x_r_ceed, y_r_ceed, z_r_ceed, mask_r_ceed, mask_Gamma_ceed, mask_I_ceed;
  CeedInt    topo_dim, num_comp_u, P, P_Pi = 2, Q, num_qpts, num_elem, elem_size;

  PetscFunctionBeginUser;

  // CEED basis
  // -- Basis for interface vertices
  CeedBasisGetDimension(basis_u, &topo_dim);
  CeedBasisGetNumComponents(basis_u, &num_comp_u);
  CeedBasisGetNumNodes1D(basis_u, &P);
  elem_size = CeedIntPow(P, topo_dim);
  CeedBasisGetNumQuadraturePoints1D(basis_u, &Q);
  CeedBasisGetNumQuadraturePoints(basis_u, &num_qpts);
  CeedScalar *interp_1d, *grad_1d, *q_ref_1d, *q_weight_1d;
  interp_1d = calloc(2 * Q, sizeof(CeedScalar));
  CeedScalar const *temp;
  CeedBasisGetInterp1D(basis_u, &temp);
  memcpy(interp_1d, temp, Q * sizeof(CeedScalar));
  memcpy(&interp_1d[1 * Q], &temp[(P - 1) * Q], Q * sizeof(CeedScalar));
  grad_1d = calloc(2 * Q, sizeof(CeedScalar));
  CeedBasisGetGrad1D(basis_u, &temp);
  memcpy(grad_1d, temp, Q * sizeof(CeedScalar));
  memcpy(&grad_1d[1 * Q], &temp[(P - 1) * Q], Q * sizeof(CeedScalar));
  q_ref_1d = calloc(Q, sizeof(CeedScalar));
  CeedBasisGetQRef(basis_u, &temp);
  memcpy(q_ref_1d, temp, Q * sizeof(CeedScalar));
  q_weight_1d = calloc(Q, sizeof(CeedScalar));
  CeedBasisGetQWeights(basis_u, &temp);
  memcpy(q_weight_1d, temp, Q * sizeof(CeedScalar));
  CeedBasisCreateTensorH1(ceed, topo_dim, num_comp_u, P_Pi, Q, interp_1d, grad_1d, q_ref_1d, q_weight_1d, &basis_Pi);
  free(interp_1d);
  free(grad_1d);
  free(q_ref_1d);
  free(q_weight_1d);
  // -- Basis for injection/restriction
  interp_1d            = calloc(2 * P, sizeof(CeedScalar));
  interp_1d[0]         = 1;
  interp_1d[2 * P - 1] = 1;  // Pick off corner vertices
  grad_1d              = calloc(2 * P, sizeof(CeedScalar));
  q_ref_1d             = calloc(2, sizeof(CeedScalar));
  q_weight_1d          = calloc(2, sizeof(CeedScalar));
  CeedBasisCreateTensorH1(ceed, topo_dim, num_comp_u, P, P_Pi, interp_1d, grad_1d, q_ref_1d, q_weight_1d, &basis_Pi_r);
  free(interp_1d);
  free(grad_1d);
  free(q_ref_1d);
  free(q_weight_1d);

  // CEED restrictions
  // -- Interface vertex restriction
  PetscCall(CreateRestrictionFromPlex(ceed, dm_Pi, 0, 0, 0, &elem_restr_Pi));

  // -- Subdomain restriction
  CeedElemRestrictionGetNumElements(elem_restr_Pi, &num_elem);
  CeedInt strides_r[3] = {num_comp_u, 1, num_comp_u * elem_size};
  CeedElemRestrictionCreateStrided(ceed, num_elem, elem_size, num_comp_u, num_comp_u * num_elem * elem_size, strides_r, &elem_restr_r);

  // -- Broken subdomain restriction
  CeedInt strides_Pi_r[3] = {num_comp_u, 1, num_comp_u * 8};
  CeedElemRestrictionCreateStrided(ceed, num_elem, 8, num_comp_u, num_comp_u * num_elem * 8, strides_Pi_r, &elem_restr_Pi_r);

  // Create the persistent vectors that will be needed
  CeedVectorCreate(ceed, xl_vertex_size, &x_Pi_ceed);
  CeedVectorCreate(ceed, xl_vertex_size, &y_Pi_ceed);
  CeedVectorCreate(ceed, 8 * num_elem, &x_Pi_r_ceed);
  CeedVectorCreate(ceed, 8 * num_elem, &y_Pi_r_ceed);
  CeedVectorCreate(ceed, num_comp_u * elem_size * num_elem, &mult_ceed);
  CeedVectorCreate(ceed, num_comp_u * elem_size * num_elem, &x_r_ceed);
  CeedVectorCreate(ceed, num_comp_u * elem_size * num_elem, &y_r_ceed);
  CeedVectorCreate(ceed, num_comp_u * elem_size * num_elem, &z_r_ceed);
  CeedVectorCreate(ceed, num_comp_u * elem_size * num_elem, &mask_r_ceed);
  CeedVectorCreate(ceed, num_comp_u * elem_size * num_elem, &mask_Gamma_ceed);
  CeedVectorCreate(ceed, num_comp_u * elem_size * num_elem, &mask_I_ceed);

  // -- Masks for subdomains
  CeedScalar *mask_r_array, *mask_Gamma_array, *mask_I_array;
  CeedVectorGetArrayWrite(mask_r_ceed, CEED_MEM_HOST, &mask_r_array);
  CeedVectorGetArrayWrite(mask_Gamma_ceed, CEED_MEM_HOST, &mask_Gamma_array);
  CeedVectorGetArrayWrite(mask_I_ceed, CEED_MEM_HOST, &mask_I_array);
  for (CeedInt e = 0; e < num_elem; e++) {
    for (CeedInt n = 0; n < elem_size; n++) {
      PetscBool r = n != 0 * (P - 1) && n != 1 * P - 1 && n != P * (P - 1) && n != P * P - 1 && n != P * P * (P - 1) + 0 &&
                    n != P * P * (P - 1) + 1 * P - 1 && n != P * P * (P - 1) + P * (P - 1) && n != P * P * (P - 1) + P * P - 1;
      PetscBool Gamma =
          n % P == 0 || n % P == P - 1 || (n / P) % P == 0 || (n / P) % P == P - 1 || (n / (P * P)) % P == 0 || (n / (P * P)) % P == P - 1;
      for (CeedInt c = 0; c < num_comp_u; c++) {
        CeedInt index           = strides_r[0] * n + strides_r[1] * c + strides_r[2] * e;
        mask_r_array[index]     = r ? 1.0 : 0.0;
        mask_Gamma_array[index] = Gamma ? 1.0 : 0.0;
        mask_I_array[index]     = !Gamma ? 1.0 : 0.0;
      }
    }
  }
  CeedVectorRestoreArray(mask_r_ceed, &mask_r_array);
  CeedVectorRestoreArray(mask_Gamma_ceed, &mask_Gamma_array);
  CeedVectorRestoreArray(mask_I_ceed, &mask_I_array);

  // Create the mass or diff operator
  PetscBool is_interp = bp_data.in_mode == CEED_EVAL_INTERP;

  // -- Interface vertices
  CeedOperatorCreate(ceed, data_fine->qf_apply, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_Pi_Pi);
  CeedOperatorSetName(op_Pi_Pi, "BDDC Pi, Pi operator");
  CeedOperatorSetField(op_Pi_Pi, is_interp ? "u" : "du", elem_restr_Pi, basis_Pi, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_Pi_Pi, "qdata", data_fine->elem_restr_qd_i, CEED_BASIS_NONE, data_fine->q_data);
  CeedOperatorSetField(op_Pi_Pi, is_interp ? "v" : "dv", elem_restr_Pi, basis_Pi, CEED_VECTOR_ACTIVE);
  // -- Subdomains to interface vertices
  CeedOperatorCreate(ceed, data_fine->qf_apply, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_Pi_r);
  CeedOperatorSetName(op_Pi_r, "BDDC Pi, r operator");
  CeedOperatorSetField(op_Pi_r, is_interp ? "u" : "du", elem_restr_r, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_Pi_r, "qdata", data_fine->elem_restr_qd_i, CEED_BASIS_NONE, data_fine->q_data);
  CeedOperatorSetField(op_Pi_r, is_interp ? "v" : "dv", elem_restr_Pi, basis_Pi, CEED_VECTOR_ACTIVE);
  // -- Interface vertices to subdomains
  CeedOperatorCreate(ceed, data_fine->qf_apply, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_r_Pi);
  CeedOperatorSetName(op_r_Pi, "BDDC r, Pi operator");
  CeedOperatorSetField(op_r_Pi, is_interp ? "u" : "du", elem_restr_Pi, basis_Pi, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_r_Pi, "qdata", data_fine->elem_restr_qd_i, CEED_BASIS_NONE, data_fine->q_data);
  CeedOperatorSetField(op_r_Pi, is_interp ? "v" : "dv", elem_restr_r, basis_u, CEED_VECTOR_ACTIVE);
  // -- Subdomains
  CeedOperatorCreate(ceed, data_fine->qf_apply, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_r_r);
  CeedOperatorSetName(op_r_r, "BDDC r, r operator");
  CeedOperatorSetField(op_r_r, is_interp ? "u" : "du", elem_restr_r, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_r_r, "qdata", data_fine->elem_restr_qd_i, CEED_BASIS_NONE, data_fine->q_data);
  CeedOperatorSetField(op_r_r, is_interp ? "v" : "dv", elem_restr_r, basis_u, CEED_VECTOR_ACTIVE);
  // -- Subdomain FDM inverse
  CeedOperatorCreateFDMElementInverse(op_r_r, &op_r_r_inv, CEED_REQUEST_IMMEDIATE);

  // Injection and restriction operators
  CeedQFunction qf_identity, qf_inject_Pi, qf_restrict_Pi;
  CeedQFunctionCreateIdentity(ceed, num_comp_u, CEED_EVAL_NONE, CEED_EVAL_NONE, &qf_identity);
  CeedQFunctionCreateIdentity(ceed, num_comp_u, CEED_EVAL_INTERP, CEED_EVAL_NONE, &qf_inject_Pi);
  CeedQFunctionCreateIdentity(ceed, num_comp_u, CEED_EVAL_NONE, CEED_EVAL_INTERP, &qf_restrict_Pi);
  // -- Injection to interface vertices
  CeedOperatorCreate(ceed, qf_inject_Pi, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_inject_Pi);
  CeedOperatorSetName(op_inject_Pi, "BDDC Pi injection operator");
  CeedOperatorSetField(op_inject_Pi, "input", elem_restr_r, basis_Pi_r, CEED_VECTOR_ACTIVE);  // Note: from r to Pi
  CeedOperatorSetField(op_inject_Pi, "output", elem_restr_Pi, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
  // -- Injection to broken interface vertices
  CeedOperatorCreate(ceed, qf_restrict_Pi, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_inject_Pi_r);
  CeedOperatorSetName(op_inject_Pi_r, "BDDC Pi, r injection operator");
  CeedOperatorSetField(op_inject_Pi_r, "input", elem_restr_Pi_r, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);  // Note: from r to Pi_r
  CeedOperatorSetField(op_inject_Pi_r, "output", elem_restr_r, basis_Pi_r, CEED_VECTOR_ACTIVE);
  // -- Injection to subdomains
  CeedOperatorCreate(ceed, qf_identity, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_inject_r);
  CeedOperatorSetName(op_inject_r, "BDDC r injection operator");
  CeedOperatorSetField(op_inject_r, "input", data_fine->elem_restr_u, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_inject_r, "output", elem_restr_r, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
  // -- Restriction from interface vertices
  CeedOperatorCreate(ceed, qf_restrict_Pi, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_restrict_Pi);
  CeedOperatorSetName(op_restrict_Pi, "BDDC Pi restriction operator");
  CeedOperatorSetField(op_restrict_Pi, "input", elem_restr_Pi, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_restrict_Pi, "output", elem_restr_r, basis_Pi_r, CEED_VECTOR_ACTIVE);  // Note: from Pi to r
  // -- Restriction from interface vertices
  CeedOperatorCreate(ceed, qf_inject_Pi, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_restrict_Pi_r);
  CeedOperatorSetName(op_restrict_Pi_r, "BDDC Pi, r restriction operator");
  CeedOperatorSetField(op_restrict_Pi_r, "input", elem_restr_r, basis_Pi_r, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_restrict_Pi_r, "output", elem_restr_Pi_r, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);  // Note: from Pi_r to r
  // -- Restriction from subdomains
  CeedOperatorCreate(ceed, qf_identity, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_restrict_r);
  CeedOperatorSetName(op_restrict_r, "BDDC r restriction operator");
  CeedOperatorSetField(op_restrict_r, "input", elem_restr_r, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_restrict_r, "output", data_fine->elem_restr_u, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
  // -- Cleanup
  CeedQFunctionDestroy(&qf_identity);
  CeedQFunctionDestroy(&qf_inject_Pi);
  CeedQFunctionDestroy(&qf_restrict_Pi);

  // Save libCEED data required for level
  data_bddc->basis_Pi         = basis_Pi;
  data_bddc->basis_Pi_r       = basis_Pi_r;
  data_bddc->elem_restr_Pi    = elem_restr_Pi;
  data_bddc->elem_restr_Pi_r  = elem_restr_Pi_r;
  data_bddc->elem_restr_r     = elem_restr_r;
  data_bddc->op_Pi_r          = op_Pi_r;
  data_bddc->op_r_Pi          = op_r_Pi;
  data_bddc->op_Pi_Pi         = op_Pi_Pi;
  data_bddc->op_r_r           = op_r_r;
  data_bddc->op_r_r_inv       = op_r_r_inv;
  data_bddc->op_inject_r      = op_inject_r;
  data_bddc->op_inject_Pi     = op_inject_Pi;
  data_bddc->op_inject_Pi_r   = op_inject_Pi_r;
  data_bddc->op_restrict_r    = op_restrict_r;
  data_bddc->op_restrict_Pi   = op_restrict_Pi;
  data_bddc->op_restrict_Pi_r = op_restrict_Pi_r;
  data_bddc->x_Pi_ceed        = x_Pi_ceed;
  data_bddc->y_Pi_ceed        = y_Pi_ceed;
  data_bddc->x_Pi_r_ceed      = x_Pi_r_ceed;
  data_bddc->y_Pi_r_ceed      = y_Pi_r_ceed;
  data_bddc->mult_ceed        = mult_ceed;
  data_bddc->x_r_ceed         = x_r_ceed;
  data_bddc->y_r_ceed         = y_r_ceed;
  data_bddc->z_r_ceed         = z_r_ceed;
  data_bddc->mask_r_ceed      = mask_r_ceed;
  data_bddc->mask_Gamma_ceed  = mask_Gamma_ceed;
  data_bddc->mask_I_ceed      = mask_I_ceed;
  data_bddc->x_ceed           = data_fine->x_ceed;
  data_bddc->y_ceed           = data_fine->y_ceed;

  PetscFunctionReturn(PETSC_SUCCESS);
};

// -----------------------------------------------------------------------------
// Set up libCEED error operator
// -----------------------------------------------------------------------------
PetscErrorCode SetupErrorOperator(DM dm, Ceed ceed, BPData bp_data, CeedInt topo_dim, PetscInt num_comp_x, PetscInt num_comp_u,
                                  CeedOperator *op_error) {
  DM                  dm_coord;
  Vec                 coords;
  const PetscScalar  *coord_array;
  CeedBasis           basis_x, basis_u;
  CeedElemRestriction elem_restr_x, elem_restr_u, elem_restr_u_i, elem_restr_qd_i;
  CeedQFunction       qf_setup_geo, qf_setup_rhs, qf_error;
  CeedOperator        op_setup_geo, op_setup_rhs;
  CeedVector          x_coord, q_data, target, rhs;
  PetscInt            c_start, c_end, num_elem;
  CeedInt             num_qpts, q_data_size = bp_data.q_data_size;
  CeedScalar          R = 1;                         // radius of the sphere
  CeedScalar          l = 1.0 / PetscSqrtReal(3.0);  // half edge of the inscribed cube

  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinateDM(dm, &dm_coord));

  // CEED bases
  PetscCall(CreateBasisFromPlex(ceed, dm_coord, 0, 0, 0, 0, bp_data, &basis_x));
  PetscCall(CreateBasisFromPlex(ceed, dm, 0, 0, 0, 0, bp_data, &basis_u));

  // CEED restrictions
  PetscCall(CreateRestrictionFromPlex(ceed, dm_coord, 0, 0, 0, &elem_restr_x));
  PetscCall(CreateRestrictionFromPlex(ceed, dm, 0, 0, 0, &elem_restr_u));

  PetscCall(DMPlexGetHeightStratum(dm, 0, &c_start, &c_end));
  num_elem = c_end - c_start;
  CeedBasisGetNumQuadraturePoints(basis_u, &num_qpts);

  CeedElemRestrictionCreateStrided(ceed, num_elem, num_qpts, num_comp_u, num_comp_u * num_elem * num_qpts, CEED_STRIDES_BACKEND, &elem_restr_u_i);
  CeedElemRestrictionCreateStrided(ceed, num_elem, num_qpts, q_data_size, q_data_size * num_elem * num_qpts, CEED_STRIDES_BACKEND, &elem_restr_qd_i);

  // Element coordinates
  PetscCall(DMGetCoordinatesLocal(dm, &coords));
  PetscCall(VecGetArrayRead(coords, &coord_array));

  CeedElemRestrictionCreateVector(elem_restr_x, &x_coord, NULL);
  CeedVectorSetArray(x_coord, CEED_MEM_HOST, CEED_COPY_VALUES, (PetscScalar *)coord_array);
  PetscCall(VecRestoreArrayRead(coords, &coord_array));

  // Create the persistent vectors that will be needed in setup and apply
  CeedVectorCreate(ceed, q_data_size * num_elem * num_qpts, &q_data);

  // Create the QFunction that builds the context data
  CeedQFunctionCreateInterior(ceed, 1, bp_data.setup_geo, bp_data.setup_geo_loc, &qf_setup_geo);
  CeedQFunctionAddInput(qf_setup_geo, "x", num_comp_x, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_setup_geo, "dx", num_comp_x * topo_dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup_geo, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup_geo, "qdata", q_data_size, CEED_EVAL_NONE);

  // Create the operator that builds the quadrature data
  CeedOperatorCreate(ceed, qf_setup_geo, NULL, NULL, &op_setup_geo);
  CeedOperatorSetField(op_setup_geo, "x", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_geo, "dx", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_geo, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_geo, "qdata", elem_restr_qd_i, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);

  // Setup q_data
  CeedOperatorApply(op_setup_geo, x_coord, q_data, CEED_REQUEST_IMMEDIATE);

  // Set up target vector
  CeedElemRestrictionCreateVector(elem_restr_u, &rhs, NULL);
  CeedVectorCreate(ceed, num_elem * num_qpts * num_comp_u, &target);
  // Create the q-function that sets up the RHS and true solution
  CeedQFunctionCreateInterior(ceed, 1, bp_data.setup_rhs, bp_data.setup_rhs_loc, &qf_setup_rhs);
  CeedQFunctionAddInput(qf_setup_rhs, "x", num_comp_x, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_setup_rhs, "qdata", q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_setup_rhs, "true solution", num_comp_u, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_setup_rhs, "rhs", num_comp_u, CEED_EVAL_INTERP);

  // Create the operator that builds the RHS and true solution
  CeedOperatorCreate(ceed, qf_setup_rhs, NULL, NULL, &op_setup_rhs);
  CeedOperatorSetField(op_setup_rhs, "x", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_rhs, "qdata", elem_restr_qd_i, CEED_BASIS_NONE, q_data);
  CeedOperatorSetField(op_setup_rhs, "true solution", elem_restr_u_i, CEED_BASIS_NONE, target);
  CeedOperatorSetField(op_setup_rhs, "rhs", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);

  // Set up the libCEED context
  CeedQFunctionContext ctx_rhs_setup;
  CeedQFunctionContextCreate(ceed, &ctx_rhs_setup);
  CeedScalar rhs_setup_data[2] = {R, l};
  CeedQFunctionContextSetData(ctx_rhs_setup, CEED_MEM_HOST, CEED_COPY_VALUES, sizeof rhs_setup_data, &rhs_setup_data);
  CeedQFunctionSetContext(qf_setup_rhs, ctx_rhs_setup);
  CeedQFunctionContextDestroy(&ctx_rhs_setup);

  // Setup RHS and target
  CeedOperatorApply(op_setup_rhs, x_coord, rhs, CEED_REQUEST_IMMEDIATE);

  // Set up error operator
  // Create the error QFunction
  CeedQFunctionCreateInterior(ceed, 1, bp_data.error, bp_data.error_loc, &qf_error);
  CeedQFunctionAddInput(qf_error, "u", num_comp_u, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_error, "true_soln", num_comp_u, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_error, "qdata", q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_error, "error", num_comp_u, CEED_EVAL_INTERP);

  // Create the error operator
  CeedOperatorCreate(ceed, qf_error, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, op_error);
  CeedOperatorSetField(*op_error, "u", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(*op_error, "true_soln", elem_restr_u_i, CEED_BASIS_NONE, target);
  CeedOperatorSetField(*op_error, "qdata", elem_restr_qd_i, CEED_BASIS_NONE, q_data);
  CeedOperatorSetField(*op_error, "error", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);

  // Cleanup
  CeedQFunctionDestroy(&qf_setup_rhs);
  CeedOperatorDestroy(&op_setup_rhs);
  CeedQFunctionDestroy(&qf_setup_geo);
  CeedOperatorDestroy(&op_setup_geo);
  CeedQFunctionDestroy(&qf_error);
  CeedVectorDestroy(&x_coord);
  CeedVectorDestroy(&rhs);
  CeedVectorDestroy(&target);
  CeedVectorDestroy(&q_data);
  CeedElemRestrictionDestroy(&elem_restr_u_i);
  CeedElemRestrictionDestroy(&elem_restr_qd_i);
  CeedElemRestrictionDestroy(&elem_restr_x);
  CeedElemRestrictionDestroy(&elem_restr_u);
  CeedBasisDestroy(&basis_x);
  CeedBasisDestroy(&basis_u);

  PetscFunctionReturn(PETSC_SUCCESS);
};

// -----------------------------------------------------------------------------
