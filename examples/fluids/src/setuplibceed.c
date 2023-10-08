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

#include "../navierstokes.h"

PetscErrorCode AddBCSubOperator(Ceed ceed, DM dm, CeedData ceed_data, DMLabel domain_label, PetscInt label_value, CeedInt height, CeedInt Q_sur,
                                CeedInt q_data_size_sur, CeedInt jac_data_size_sur, CeedQFunction qf_apply_bc, CeedQFunction qf_apply_bc_jacobian,
                                CeedOperator *op_apply, CeedOperator *op_apply_ijacobian) {
  CeedVector          q_data_sur, jac_data_sur = NULL;
  CeedOperator        op_setup_sur, op_apply_bc, op_apply_bc_jacobian = NULL;
  CeedElemRestriction elem_restr_x_sur, elem_restr_q_sur, elem_restr_qd_i_sur, elem_restr_jd_i_sur = NULL;
  CeedInt             num_qpts_sur, dm_field = 0;

  PetscFunctionBeginUser;
  // --- Get number of quadrature points for the boundaries
  PetscCallCeed(ceed, CeedBasisGetNumQuadraturePoints(ceed_data->basis_q_sur, &num_qpts_sur));

  // ---- CEED Restriction
  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, dm, domain_label, label_value, height, dm_field, &elem_restr_q_sur));
  PetscCall(DMPlexCeedElemRestrictionCoordinateCreate(ceed, dm, domain_label, label_value, height, &elem_restr_x_sur));
  PetscCall(DMPlexCeedElemRestrictionQDataCreate(ceed, dm, domain_label, label_value, height, q_data_size_sur, &elem_restr_qd_i_sur));
  if (jac_data_size_sur > 0) {
    // State-dependent data will be passed from residual to Jacobian. This will be collocated.
    PetscCall(DMPlexCeedElemRestrictionQDataCreate(ceed, dm, domain_label, label_value, height, jac_data_size_sur, &elem_restr_jd_i_sur));
    PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_jd_i_sur, &jac_data_sur, NULL));
  }

  // ---- CEED Vector
  CeedInt loc_num_elem_sur;
  PetscCallCeed(ceed, CeedElemRestrictionGetNumElements(elem_restr_q_sur, &loc_num_elem_sur));
  PetscCallCeed(ceed, CeedVectorCreate(ceed, q_data_size_sur * loc_num_elem_sur * num_qpts_sur, &q_data_sur));

  // ---- CEED Operator
  // ----- CEED Operator for Setup (geometric factors)
  PetscCallCeed(ceed, CeedOperatorCreate(ceed, ceed_data->qf_setup_sur, NULL, NULL, &op_setup_sur));
  PetscCallCeed(ceed, CeedOperatorSetField(op_setup_sur, "dx", elem_restr_x_sur, ceed_data->basis_x_sur, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_setup_sur, "weight", CEED_ELEMRESTRICTION_NONE, ceed_data->basis_x_sur, CEED_VECTOR_NONE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_setup_sur, "surface qdata", elem_restr_qd_i_sur, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  // ----- CEED Operator for Physics
  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_apply_bc, NULL, NULL, &op_apply_bc));
  PetscCallCeed(ceed, CeedOperatorSetField(op_apply_bc, "q", elem_restr_q_sur, ceed_data->basis_q_sur, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_apply_bc, "Grad_q", elem_restr_q_sur, ceed_data->basis_q_sur, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_apply_bc, "surface qdata", elem_restr_qd_i_sur, CEED_BASIS_NONE, q_data_sur));
  PetscCallCeed(ceed, CeedOperatorSetField(op_apply_bc, "x", elem_restr_x_sur, ceed_data->basis_x_sur, ceed_data->x_coord));
  PetscCallCeed(ceed, CeedOperatorSetField(op_apply_bc, "v", elem_restr_q_sur, ceed_data->basis_q_sur, CEED_VECTOR_ACTIVE));
  if (elem_restr_jd_i_sur)
    PetscCallCeed(ceed, CeedOperatorSetField(op_apply_bc, "surface jacobian data", elem_restr_jd_i_sur, CEED_BASIS_NONE, jac_data_sur));

  if (qf_apply_bc_jacobian) {
    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_apply_bc_jacobian, NULL, NULL, &op_apply_bc_jacobian));
    PetscCallCeed(ceed, CeedOperatorSetField(op_apply_bc_jacobian, "dq", elem_restr_q_sur, ceed_data->basis_q_sur, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_apply_bc_jacobian, "Grad_dq", elem_restr_q_sur, ceed_data->basis_q_sur, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_apply_bc_jacobian, "surface qdata", elem_restr_qd_i_sur, CEED_BASIS_NONE, q_data_sur));
    PetscCallCeed(ceed, CeedOperatorSetField(op_apply_bc_jacobian, "x", elem_restr_x_sur, ceed_data->basis_x_sur, ceed_data->x_coord));
    PetscCallCeed(ceed, CeedOperatorSetField(op_apply_bc_jacobian, "surface jacobian data", elem_restr_jd_i_sur, CEED_BASIS_NONE, jac_data_sur));
    PetscCallCeed(ceed, CeedOperatorSetField(op_apply_bc_jacobian, "v", elem_restr_q_sur, ceed_data->basis_q_sur, CEED_VECTOR_ACTIVE));
  }

  // ----- Apply CEED operator for Setup
  PetscCallCeed(ceed, CeedOperatorApply(op_setup_sur, ceed_data->x_coord, q_data_sur, CEED_REQUEST_IMMEDIATE));

  // ----- Apply Sub-Operator for Physics
  PetscCallCeed(ceed, CeedCompositeOperatorAddSub(*op_apply, op_apply_bc));
  if (op_apply_bc_jacobian) PetscCallCeed(ceed, CeedCompositeOperatorAddSub(*op_apply_ijacobian, op_apply_bc_jacobian));

  // ----- Cleanup
  PetscCallCeed(ceed, CeedVectorDestroy(&q_data_sur));
  PetscCallCeed(ceed, CeedVectorDestroy(&jac_data_sur));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_q_sur));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_x_sur));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd_i_sur));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_jd_i_sur));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_setup_sur));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_apply_bc));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_apply_bc_jacobian));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Utility function to create CEED Composite Operator for the entire domain
PetscErrorCode CreateOperatorForDomain(Ceed ceed, DM dm, SimpleBC bc, CeedData ceed_data, Physics phys, CeedOperator op_apply_vol,
                                       CeedOperator op_apply_ijacobian_vol, CeedInt height, CeedInt P_sur, CeedInt Q_sur, CeedInt q_data_size_sur,
                                       CeedInt jac_data_size_sur, CeedOperator *op_apply, CeedOperator *op_apply_ijacobian) {
  DMLabel domain_label;

  PetscFunctionBeginUser;
  // Create Composite Operaters
  PetscCallCeed(ceed, CeedCompositeOperatorCreate(ceed, op_apply));
  if (op_apply_ijacobian) PetscCallCeed(ceed, CeedCompositeOperatorCreate(ceed, op_apply_ijacobian));

  // --Apply Sub-Operator for the volume
  PetscCallCeed(ceed, CeedCompositeOperatorAddSub(*op_apply, op_apply_vol));
  if (op_apply_ijacobian) PetscCallCeed(ceed, CeedCompositeOperatorAddSub(*op_apply_ijacobian, op_apply_ijacobian_vol));

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
  PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(*op_apply, "solution time", &phys->solution_time_label));
  PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(*op_apply, "timestep size", &phys->timestep_size_label));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetupBCQFunctions(Ceed ceed, PetscInt dim_sur, PetscInt num_comp_x, PetscInt num_comp_q, PetscInt q_data_size_sur,
                                 PetscInt jac_data_size_sur, ProblemQFunctionSpec apply_bc, ProblemQFunctionSpec apply_bc_jacobian,
                                 CeedQFunction *qf_apply_bc, CeedQFunction *qf_apply_bc_jacobian) {
  PetscFunctionBeginUser;
  if (apply_bc.qfunction) {
    PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, apply_bc.qfunction, apply_bc.qfunction_loc, qf_apply_bc));
    PetscCallCeed(ceed, CeedQFunctionSetContext(*qf_apply_bc, apply_bc.qfunction_context));
    PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_apply_bc, "q", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_apply_bc, "Grad_q", num_comp_q * dim_sur, CEED_EVAL_GRAD));
    PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_apply_bc, "surface qdata", q_data_size_sur, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_apply_bc, "x", num_comp_x, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(*qf_apply_bc, "v", num_comp_q, CEED_EVAL_INTERP));
    if (jac_data_size_sur) PetscCallCeed(ceed, CeedQFunctionAddOutput(*qf_apply_bc, "surface jacobian data", jac_data_size_sur, CEED_EVAL_NONE));
  }
  if (apply_bc_jacobian.qfunction) {
    PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, apply_bc_jacobian.qfunction, apply_bc_jacobian.qfunction_loc, qf_apply_bc_jacobian));
    PetscCallCeed(ceed, CeedQFunctionSetContext(*qf_apply_bc_jacobian, apply_bc_jacobian.qfunction_context));
    PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_apply_bc_jacobian, "dq", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_apply_bc_jacobian, "Grad_dq", num_comp_q * dim_sur, CEED_EVAL_GRAD));
    PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_apply_bc_jacobian, "surface qdata", q_data_size_sur, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_apply_bc_jacobian, "x", num_comp_x, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_apply_bc_jacobian, "surface jacobian data", jac_data_size_sur, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(*qf_apply_bc_jacobian, "v", num_comp_q, CEED_EVAL_INTERP));
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
  DMLabel             domain_label = NULL;
  PetscInt            label_value = 0, height = 0, dm_field = 0;

  // -----------------------------------------------------------------------------
  // CEED Bases
  // -----------------------------------------------------------------------------
  DM dm_coord;
  PetscCall(DMGetCoordinateDM(dm, &dm_coord));

  PetscCall(CreateBasisFromPlex(ceed, dm, domain_label, label_value, height, dm_field, &ceed_data->basis_q));
  PetscCall(CreateBasisFromPlex(ceed, dm_coord, domain_label, label_value, height, dm_field, &ceed_data->basis_x));
  PetscCallCeed(ceed, CeedBasisCreateProjection(ceed_data->basis_x, ceed_data->basis_q, &ceed_data->basis_xc));
  PetscCallCeed(ceed, CeedBasisGetNumQuadraturePoints(ceed_data->basis_q, &num_qpts));

  // -----------------------------------------------------------------------------
  // CEED Restrictions
  // -----------------------------------------------------------------------------
  // -- Create restriction
  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, dm, domain_label, label_value, height, 0, &ceed_data->elem_restr_q));
  PetscCall(DMPlexCeedElemRestrictionCoordinateCreate(ceed, dm, domain_label, label_value, height, &ceed_data->elem_restr_x));
  PetscCall(DMPlexCeedElemRestrictionQDataCreate(ceed, dm, domain_label, label_value, height, q_data_size_vol, &ceed_data->elem_restr_qd_i));
  PetscCall(DMPlexCeedElemRestrictionQDataCreate(ceed, dm, domain_label, label_value, height, jac_data_size_vol, &elem_restr_jd_i));
  // -- Create E vectors
  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &user->q_ceed, NULL));
  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &user->q_dot_ceed, NULL));
  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &user->g_ceed, NULL));

  // -----------------------------------------------------------------------------
  // CEED QFunctions
  // -----------------------------------------------------------------------------
  // -- Create QFunction for quadrature data
  PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, problem->setup_vol.qfunction, problem->setup_vol.qfunction_loc, &ceed_data->qf_setup_vol));
  if (problem->setup_vol.qfunction_context) {
    PetscCallCeed(ceed, CeedQFunctionSetContext(ceed_data->qf_setup_vol, problem->setup_vol.qfunction_context));
  }
  PetscCallCeed(ceed, CeedQFunctionAddInput(ceed_data->qf_setup_vol, "dx", num_comp_x * dim, CEED_EVAL_GRAD));
  PetscCallCeed(ceed, CeedQFunctionAddInput(ceed_data->qf_setup_vol, "weight", 1, CEED_EVAL_WEIGHT));
  PetscCallCeed(ceed, CeedQFunctionAddInput(ceed_data->qf_setup_vol, "x", num_comp_x, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(ceed_data->qf_setup_vol, "qdata", q_data_size_vol, CEED_EVAL_NONE));

  // -- Create QFunction for ICs
  PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, problem->ics.qfunction, problem->ics.qfunction_loc, &ceed_data->qf_ics));
  PetscCallCeed(ceed, CeedQFunctionSetContext(ceed_data->qf_ics, problem->ics.qfunction_context));
  PetscCallCeed(ceed, CeedQFunctionAddInput(ceed_data->qf_ics, "x", num_comp_x, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(ceed_data->qf_ics, "dx", num_comp_x * dim, CEED_EVAL_GRAD));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(ceed_data->qf_ics, "q0", num_comp_q, CEED_EVAL_NONE));

  // -- Create QFunction for RHS
  if (problem->apply_vol_rhs.qfunction) {
    PetscCallCeed(
        ceed, CeedQFunctionCreateInterior(ceed, 1, problem->apply_vol_rhs.qfunction, problem->apply_vol_rhs.qfunction_loc, &ceed_data->qf_rhs_vol));
    PetscCallCeed(ceed, CeedQFunctionSetContext(ceed_data->qf_rhs_vol, problem->apply_vol_rhs.qfunction_context));
    PetscCallCeed(ceed, CeedQFunctionAddInput(ceed_data->qf_rhs_vol, "q", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddInput(ceed_data->qf_rhs_vol, "Grad_q", num_comp_q * dim, CEED_EVAL_GRAD));
    PetscCallCeed(ceed, CeedQFunctionAddInput(ceed_data->qf_rhs_vol, "qdata", q_data_size_vol, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(ceed_data->qf_rhs_vol, "v", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(ceed_data->qf_rhs_vol, "Grad_v", num_comp_q * dim, CEED_EVAL_GRAD));
  }

  // -- Create QFunction for IFunction
  if (problem->apply_vol_ifunction.qfunction) {
    PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, problem->apply_vol_ifunction.qfunction, problem->apply_vol_ifunction.qfunction_loc,
                                                    &ceed_data->qf_ifunction_vol));
    PetscCallCeed(ceed, CeedQFunctionSetContext(ceed_data->qf_ifunction_vol, problem->apply_vol_ifunction.qfunction_context));
    PetscCallCeed(ceed, CeedQFunctionAddInput(ceed_data->qf_ifunction_vol, "q", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddInput(ceed_data->qf_ifunction_vol, "Grad_q", num_comp_q * dim, CEED_EVAL_GRAD));
    PetscCallCeed(ceed, CeedQFunctionAddInput(ceed_data->qf_ifunction_vol, "q dot", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddInput(ceed_data->qf_ifunction_vol, "qdata", q_data_size_vol, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(ceed_data->qf_ifunction_vol, "v", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(ceed_data->qf_ifunction_vol, "Grad_v", num_comp_q * dim, CEED_EVAL_GRAD));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(ceed_data->qf_ifunction_vol, "jac_data", jac_data_size_vol, CEED_EVAL_NONE));
  }

  CeedQFunction qf_ijacobian_vol = NULL;
  if (problem->apply_vol_ijacobian.qfunction) {
    PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, problem->apply_vol_ijacobian.qfunction, problem->apply_vol_ijacobian.qfunction_loc,
                                                    &qf_ijacobian_vol));
    PetscCallCeed(ceed, CeedQFunctionSetContext(qf_ijacobian_vol, problem->apply_vol_ijacobian.qfunction_context));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ijacobian_vol, "dq", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ijacobian_vol, "Grad_dq", num_comp_q * dim, CEED_EVAL_GRAD));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ijacobian_vol, "qdata", q_data_size_vol, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ijacobian_vol, "jac_data", jac_data_size_vol, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_ijacobian_vol, "v", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_ijacobian_vol, "Grad_v", num_comp_q * dim, CEED_EVAL_GRAD));
  }

  // ---------------------------------------------------------------------------
  // Element coordinates
  // ---------------------------------------------------------------------------
  // -- Create CEED vector
  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(ceed_data->elem_restr_x, &ceed_data->x_coord, NULL));

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
  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(ceed_data->elem_restr_qd_i, &ceed_data->q_data, NULL));
  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_jd_i, &jac_data, NULL));

  // -----------------------------------------------------------------------------
  // CEED Operators
  // -----------------------------------------------------------------------------
  // -- Create CEED operator for quadrature data
  PetscCallCeed(ceed, CeedOperatorCreate(ceed, ceed_data->qf_setup_vol, NULL, NULL, &ceed_data->op_setup_vol));
  PetscCallCeed(ceed, CeedOperatorSetField(ceed_data->op_setup_vol, "dx", ceed_data->elem_restr_x, ceed_data->basis_x, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(ceed_data->op_setup_vol, "weight", CEED_ELEMRESTRICTION_NONE, ceed_data->basis_x, CEED_VECTOR_NONE));
  PetscCallCeed(ceed, CeedOperatorSetField(ceed_data->op_setup_vol, "x", ceed_data->elem_restr_x, ceed_data->basis_xc, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(ceed_data->op_setup_vol, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  // -- Create CEED operator for ICs
  CeedOperator op_ics;
  PetscCallCeed(ceed, CeedOperatorCreate(ceed, ceed_data->qf_ics, NULL, NULL, &op_ics));
  PetscCallCeed(ceed, CeedOperatorSetField(op_ics, "x", ceed_data->elem_restr_x, ceed_data->basis_xc, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_ics, "dx", ceed_data->elem_restr_x, ceed_data->basis_xc, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_ics, "q0", ceed_data->elem_restr_q, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(op_ics, "evaluation time", &user->phys->ics_time_label));
  PetscCall(OperatorApplyContextCreate(NULL, dm, user->ceed, op_ics, ceed_data->x_coord, NULL, NULL, user->Q_loc, &ceed_data->op_ics_ctx));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_ics));

  // Create CEED operator for RHS
  if (ceed_data->qf_rhs_vol) {
    CeedOperator op;
    PetscCallCeed(ceed, CeedOperatorCreate(ceed, ceed_data->qf_rhs_vol, NULL, NULL, &op));
    PetscCallCeed(ceed, CeedOperatorSetField(op, "q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op, "Grad_q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_NONE, ceed_data->q_data));
    PetscCallCeed(ceed, CeedOperatorSetField(op, "v", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op, "Grad_v", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    user->op_rhs_vol = op;
  }

  // -- CEED operator for IFunction
  if (ceed_data->qf_ifunction_vol) {
    CeedOperator op;
    PetscCallCeed(ceed, CeedOperatorCreate(ceed, ceed_data->qf_ifunction_vol, NULL, NULL, &op));
    PetscCallCeed(ceed, CeedOperatorSetField(op, "q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op, "Grad_q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op, "q dot", ceed_data->elem_restr_q, ceed_data->basis_q, user->q_dot_ceed));
    PetscCallCeed(ceed, CeedOperatorSetField(op, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_NONE, ceed_data->q_data));
    PetscCallCeed(ceed, CeedOperatorSetField(op, "v", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op, "Grad_v", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op, "jac_data", elem_restr_jd_i, CEED_BASIS_NONE, jac_data));

    user->op_ifunction_vol = op;
  }

  CeedOperator op_ijacobian_vol = NULL;
  if (qf_ijacobian_vol) {
    CeedOperator op;
    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_ijacobian_vol, NULL, NULL, &op));
    PetscCallCeed(ceed, CeedOperatorSetField(op, "dq", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op, "Grad_dq", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_NONE, ceed_data->q_data));
    PetscCallCeed(ceed, CeedOperatorSetField(op, "jac_data", elem_restr_jd_i, CEED_BASIS_NONE, jac_data));
    PetscCallCeed(ceed, CeedOperatorSetField(op, "v", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op, "Grad_v", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    op_ijacobian_vol = op;
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_ijacobian_vol));
  }

  // *****************************************************************************
  // Set up CEED objects for the exterior domain (surface)
  // *****************************************************************************
  height                = 1;
  CeedInt       dim_sur = dim - height, P_sur = app_ctx->degree + 1, Q_sur = P_sur + app_ctx->q_extra;
  const CeedInt q_data_size_sur = problem->q_data_size_sur, jac_data_size_sur = problem->jac_data_size_sur;

  // -----------------------------------------------------------------------------
  // CEED Bases
  // -----------------------------------------------------------------------------

  DMLabel  label   = 0;
  PetscInt face_id = 0;
  PetscInt field   = 0;  // Still want the normal, default field
  PetscCall(CreateBasisFromPlex(ceed, dm, label, face_id, height, field, &ceed_data->basis_q_sur));
  PetscCall(CreateBasisFromPlex(ceed, dm_coord, label, face_id, height, field, &ceed_data->basis_x_sur));
  PetscCallCeed(ceed, CeedBasisCreateProjection(ceed_data->basis_x_sur, ceed_data->basis_q_sur, &ceed_data->basis_xc_sur));

  // -----------------------------------------------------------------------------
  // CEED QFunctions
  // -----------------------------------------------------------------------------
  // -- Create QFunction for quadrature data
  PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, problem->setup_sur.qfunction, problem->setup_sur.qfunction_loc, &ceed_data->qf_setup_sur));
  if (problem->setup_sur.qfunction_context) {
    PetscCallCeed(ceed, CeedQFunctionSetContext(ceed_data->qf_setup_sur, problem->setup_sur.qfunction_context));
  }
  PetscCallCeed(ceed, CeedQFunctionAddInput(ceed_data->qf_setup_sur, "dx", num_comp_x * dim_sur, CEED_EVAL_GRAD));
  PetscCallCeed(ceed, CeedQFunctionAddInput(ceed_data->qf_setup_sur, "weight", 1, CEED_EVAL_WEIGHT));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(ceed_data->qf_setup_sur, "surface qdata", q_data_size_sur, CEED_EVAL_NONE));

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
  PetscCallCeed(ceed, CeedOperatorApply(ceed_data->op_setup_vol, ceed_data->x_coord, ceed_data->q_data, CEED_REQUEST_IMMEDIATE));

  // -- Create and apply CEED Composite Operator for the entire domain
  if (!user->phys->implicit) {  // RHS
    CeedOperator op_rhs;
    PetscCall(CreateOperatorForDomain(ceed, dm, bc, ceed_data, user->phys, user->op_rhs_vol, NULL, height, P_sur, Q_sur, q_data_size_sur, 0, &op_rhs,
                                      NULL));
    PetscCall(OperatorApplyContextCreate(dm, dm, ceed, op_rhs, user->q_ceed, user->g_ceed, user->Q_loc, NULL, &user->op_rhs_ctx));
    PetscCallCeed(ceed, CeedOperatorDestroy(&op_rhs));
  } else {  // IFunction
    PetscCall(CreateOperatorForDomain(ceed, dm, bc, ceed_data, user->phys, user->op_ifunction_vol, op_ijacobian_vol, height, P_sur, Q_sur,
                                      q_data_size_sur, jac_data_size_sur, &user->op_ifunction, op_ijacobian_vol ? &user->op_ijacobian : NULL));
    if (user->op_ijacobian) {
      PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(user->op_ijacobian, "ijacobian time shift", &user->phys->ijacobian_time_shift_label));
    }
    if (problem->use_strong_bc_ceed) PetscCall(SetupStrongBC_Ceed(ceed, ceed_data, dm, user, problem, bc));
    if (app_ctx->sgs_model_type == SGS_MODEL_DATA_DRIVEN) PetscCall(SgsDDModelSetup(ceed, user, ceed_data, problem));
  }

  if (app_ctx->turb_spanstats_enable) PetscCall(TurbulenceStatisticsSetup(ceed, user, ceed_data, problem));
  if (app_ctx->diff_filter_monitor) PetscCall(DifferentialFilterSetup(ceed, user, ceed_data, problem));

  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_jd_i));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_ijacobian_vol));
  PetscCallCeed(ceed, CeedVectorDestroy(&jac_data));
  PetscFunctionReturn(PETSC_SUCCESS);
}
