// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
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

// @brief Create CeedOperator for unstabilized mass KSP for explicit timestepping
static PetscErrorCode CreateKSPMassOperator_Unstabilized(User user, CeedOperator *op_mass) {
  Ceed                ceed = user->ceed;
  CeedInt             num_comp_q, q_data_size;
  CeedQFunction       qf_mass;
  CeedElemRestriction elem_restr_q, elem_restr_qd_i;
  CeedBasis           basis_q;
  CeedVector          q_data;

  PetscFunctionBeginUser;
  {  // Get restriction and basis from the RHS function
    CeedOperator     *sub_ops;
    CeedOperatorField field;
    PetscInt          sub_op_index = 0;  // will be 0 for the volume op

    PetscCallCeed(ceed, CeedCompositeOperatorGetSubList(user->op_rhs_ctx->op, &sub_ops));
    PetscCallCeed(ceed, CeedOperatorGetFieldByName(sub_ops[sub_op_index], "q", &field));
    PetscCallCeed(ceed, CeedOperatorFieldGetElemRestriction(field, &elem_restr_q));
    PetscCallCeed(ceed, CeedOperatorFieldGetBasis(field, &basis_q));

    PetscCallCeed(ceed, CeedOperatorGetFieldByName(sub_ops[sub_op_index], "qdata", &field));
    PetscCallCeed(ceed, CeedOperatorFieldGetElemRestriction(field, &elem_restr_qd_i));
    PetscCallCeed(ceed, CeedOperatorFieldGetVector(field, &q_data));
  }

  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_q, &num_comp_q));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_qd_i, &q_data_size));

  PetscCall(CreateMassQFunction(ceed, num_comp_q, q_data_size, &qf_mass));
  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_mass, NULL, NULL, op_mass));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_mass, "u", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_mass, "qdata", elem_restr_qd_i, CEED_BASIS_NONE, q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_mass, "v", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));

  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_mass));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Create KSP to solve the inverse mass operator for explicit time stepping schemes
static PetscErrorCode CreateKSPMass(User user, ProblemData problem) {
  Ceed         ceed = user->ceed;
  DM           dm   = user->dm;
  CeedOperator op_mass;

  PetscFunctionBeginUser;
  if (problem->create_mass_operator) PetscCall(problem->create_mass_operator(user, &op_mass));
  else PetscCall(CreateKSPMassOperator_Unstabilized(user, &op_mass));

  {  // -- Setup KSP for mass operator
    Mat      mat_mass;
    Vec      Zeros_loc;
    MPI_Comm comm = PetscObjectComm((PetscObject)dm);

    PetscCall(DMCreateLocalVector(dm, &Zeros_loc));
    PetscCall(VecZeroEntries(Zeros_loc));
    PetscCall(MatCeedCreate(dm, dm, op_mass, NULL, &mat_mass));
    PetscCall(MatCeedSetLocalVectors(mat_mass, Zeros_loc, NULL));

    PetscCall(KSPCreate(comm, &user->mass_ksp));
    PetscCall(KSPSetOptionsPrefix(user->mass_ksp, "mass_"));
    {  // lumped by default
      PC pc;
      PetscCall(KSPGetPC(user->mass_ksp, &pc));
      PetscCall(PCSetType(pc, PCJACOBI));
      PetscCall(PCJacobiSetType(pc, PC_JACOBI_ROWSUM));
      PetscCall(KSPSetType(user->mass_ksp, KSPPREONLY));
    }
    PetscCall(KSPSetFromOptions_WithMatCeed(user->mass_ksp, mat_mass));
    PetscCall(KSPSetFromOptions(user->mass_ksp));
    PetscCall(VecDestroy(&Zeros_loc));
    PetscCall(MatDestroy(&mat_mass));
  }

  PetscCallCeed(ceed, CeedOperatorDestroy(&op_mass));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AddBCSubOperator(Ceed ceed, DM dm, CeedData ceed_data, DMLabel domain_label, PetscInt label_value, CeedInt height,
                                       CeedInt Q_sur, CeedInt q_data_size_sur, CeedInt jac_data_size_sur, CeedBasis basis_q_sur,
                                       CeedBasis basis_x_sur, CeedQFunction qf_apply_bc, CeedQFunction qf_apply_bc_jacobian, CeedOperator op_apply,
                                       CeedOperator op_apply_ijacobian) {
  CeedVector          q_data_sur, jac_data_sur          = NULL;
  CeedOperator        op_apply_bc, op_apply_bc_jacobian = NULL;
  CeedElemRestriction elem_restr_x_sur, elem_restr_q_sur, elem_restr_qd_i_sur, elem_restr_jd_i_sur = NULL;
  PetscInt            dm_field = 0;

  PetscFunctionBeginUser;
  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, dm, domain_label, label_value, height, dm_field, &elem_restr_q_sur));
  PetscCall(DMPlexCeedElemRestrictionCoordinateCreate(ceed, dm, domain_label, label_value, height, &elem_restr_x_sur));
  if (jac_data_size_sur > 0) {
    // State-dependent data will be passed from residual to Jacobian. This will be collocated.
    PetscCall(DMPlexCeedElemRestrictionQDataCreate(ceed, dm, domain_label, label_value, height, jac_data_size_sur, &elem_restr_jd_i_sur));
    PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_jd_i_sur, &jac_data_sur, NULL));
  }

  PetscCall(QDataBoundaryGet(ceed, dm, domain_label, label_value, elem_restr_x_sur, basis_x_sur, ceed_data->x_coord, &elem_restr_qd_i_sur,
                             &q_data_sur, &q_data_size_sur));

  // CEED Operator for Physics
  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_apply_bc, NULL, NULL, &op_apply_bc));
  PetscCallCeed(ceed, CeedOperatorSetField(op_apply_bc, "q", elem_restr_q_sur, basis_q_sur, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_apply_bc, "Grad_q", elem_restr_q_sur, basis_q_sur, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_apply_bc, "surface qdata", elem_restr_qd_i_sur, CEED_BASIS_NONE, q_data_sur));
  PetscCallCeed(ceed, CeedOperatorSetField(op_apply_bc, "x", elem_restr_x_sur, basis_x_sur, ceed_data->x_coord));
  PetscCallCeed(ceed, CeedOperatorSetField(op_apply_bc, "v", elem_restr_q_sur, basis_q_sur, CEED_VECTOR_ACTIVE));
  if (elem_restr_jd_i_sur)
    PetscCallCeed(ceed, CeedOperatorSetField(op_apply_bc, "surface jacobian data", elem_restr_jd_i_sur, CEED_BASIS_NONE, jac_data_sur));

  if (qf_apply_bc_jacobian && elem_restr_jd_i_sur) {
    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_apply_bc_jacobian, NULL, NULL, &op_apply_bc_jacobian));
    PetscCallCeed(ceed, CeedOperatorSetField(op_apply_bc_jacobian, "dq", elem_restr_q_sur, basis_q_sur, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_apply_bc_jacobian, "Grad_dq", elem_restr_q_sur, basis_q_sur, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_apply_bc_jacobian, "surface qdata", elem_restr_qd_i_sur, CEED_BASIS_NONE, q_data_sur));
    PetscCallCeed(ceed, CeedOperatorSetField(op_apply_bc_jacobian, "x", elem_restr_x_sur, basis_x_sur, ceed_data->x_coord));
    PetscCallCeed(ceed, CeedOperatorSetField(op_apply_bc_jacobian, "surface jacobian data", elem_restr_jd_i_sur, CEED_BASIS_NONE, jac_data_sur));
    PetscCallCeed(ceed, CeedOperatorSetField(op_apply_bc_jacobian, "v", elem_restr_q_sur, basis_q_sur, CEED_VECTOR_ACTIVE));
  }

  // Apply Sub-Operator for Physics
  PetscCallCeed(ceed, CeedCompositeOperatorAddSub(op_apply, op_apply_bc));
  if (op_apply_bc_jacobian) PetscCallCeed(ceed, CeedCompositeOperatorAddSub(op_apply_ijacobian, op_apply_bc_jacobian));

  PetscCallCeed(ceed, CeedVectorDestroy(&q_data_sur));
  PetscCallCeed(ceed, CeedVectorDestroy(&jac_data_sur));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_q_sur));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_x_sur));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd_i_sur));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_jd_i_sur));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_apply_bc));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_apply_bc_jacobian));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupBCQFunctions(Ceed ceed, PetscInt dim_sur, PetscInt num_comp_x, PetscInt num_comp_q, PetscInt q_data_size_sur,
                                        PetscInt jac_data_size_sur, ProblemQFunctionSpec apply_bc, ProblemQFunctionSpec apply_bc_jacobian,
                                        CeedQFunction *qf_apply_bc, CeedQFunction *qf_apply_bc_jacobian) {
  PetscFunctionBeginUser;
  if (apply_bc.qfunction) {
    PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, apply_bc.qfunction, apply_bc.qfunction_loc, qf_apply_bc));
    PetscCallCeed(ceed, CeedQFunctionSetContext(*qf_apply_bc, apply_bc.qfunction_context));
    PetscCallCeed(ceed, CeedQFunctionSetUserFlopsEstimate(*qf_apply_bc, 0));
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
    PetscCallCeed(ceed, CeedQFunctionSetUserFlopsEstimate(*qf_apply_bc_jacobian, 0));
    PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_apply_bc_jacobian, "dq", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_apply_bc_jacobian, "Grad_dq", num_comp_q * dim_sur, CEED_EVAL_GRAD));
    PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_apply_bc_jacobian, "surface qdata", q_data_size_sur, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_apply_bc_jacobian, "x", num_comp_x, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_apply_bc_jacobian, "surface jacobian data", jac_data_size_sur, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(*qf_apply_bc_jacobian, "v", num_comp_q, CEED_EVAL_INTERP));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Utility function to add boundary operators to the composite operator
static PetscErrorCode AddBCSubOperators(User user, Ceed ceed, DM dm, SimpleBC bc, ProblemData problem, CeedData ceed_data, CeedOperator op_apply,
                                        CeedOperator op_apply_ijacobian) {
  CeedInt       height = 1, num_comp_q, num_comp_x;
  CeedInt       P_sur = user->app_ctx->degree + 1, Q_sur = P_sur + user->app_ctx->q_extra, dim_sur, q_data_size_sur;
  const CeedInt jac_data_size_sur = user->phys->implicit ? problem->jac_data_size_sur : 0;
  PetscInt      dim;
  DMLabel       face_sets_label;
  CeedBasis     basis_q_sur, basis_x_sur;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(QDataBoundaryGetNumComponents(dm, &q_data_size_sur));
  dim_sur = dim - height;
  {  // Get number of components and coordinate dimension from op_apply
    CeedOperator       *sub_ops;
    CeedOperatorField   field;
    PetscInt            sub_op_index = 0;  // will be 0 for the volume op
    CeedElemRestriction elem_restr_q, elem_restr_x;

    PetscCallCeed(ceed, CeedCompositeOperatorGetSubList(op_apply, &sub_ops));
    PetscCallCeed(ceed, CeedOperatorGetFieldByName(sub_ops[sub_op_index], "q", &field));
    PetscCallCeed(ceed, CeedOperatorFieldGetElemRestriction(field, &elem_restr_q));
    PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_q, &num_comp_q));

    PetscCallCeed(ceed, CeedOperatorGetFieldByName(sub_ops[sub_op_index], "x", &field));
    PetscCallCeed(ceed, CeedOperatorFieldGetElemRestriction(field, &elem_restr_x));
    PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_x, &num_comp_x));
  }

  {  // Get bases
    DM dm_coord;

    PetscCall(DMGetCoordinateDM(dm, &dm_coord));
    DMLabel  label       = NULL;
    PetscInt label_value = 0;
    PetscInt field       = 0;
    PetscCall(CreateBasisFromPlex(ceed, dm, label, label_value, height, field, &basis_q_sur));
    PetscCall(CreateBasisFromPlex(ceed, dm_coord, label, label_value, height, field, &basis_x_sur));
  }

  PetscCall(DMGetLabel(dm, "Face Sets", &face_sets_label));

  {  // --- Create Sub-Operator for inflow boundaries
    CeedQFunction qf_apply_inflow = NULL, qf_apply_inflow_jacobian = NULL;

    PetscCall(SetupBCQFunctions(ceed, dim_sur, num_comp_x, num_comp_q, q_data_size_sur, jac_data_size_sur, problem->apply_inflow,
                                problem->apply_inflow_jacobian, &qf_apply_inflow, &qf_apply_inflow_jacobian));
    for (CeedInt i = 0; i < bc->num_inflow; i++) {
      PetscCall(AddBCSubOperator(ceed, dm, ceed_data, face_sets_label, bc->inflows[i], height, Q_sur, q_data_size_sur, jac_data_size_sur, basis_q_sur,
                                 basis_x_sur, qf_apply_inflow, qf_apply_inflow_jacobian, op_apply, op_apply_ijacobian));
    }
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_apply_inflow));
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_apply_inflow_jacobian));
  }

  {  // --- Create Sub-Operator for outflow boundaries
    CeedQFunction qf_apply_outflow = NULL, qf_apply_outflow_jacobian = NULL;

    PetscCall(SetupBCQFunctions(ceed, dim_sur, num_comp_x, num_comp_q, q_data_size_sur, jac_data_size_sur, problem->apply_outflow,
                                problem->apply_outflow_jacobian, &qf_apply_outflow, &qf_apply_outflow_jacobian));
    for (CeedInt i = 0; i < bc->num_outflow; i++) {
      PetscCall(AddBCSubOperator(ceed, dm, ceed_data, face_sets_label, bc->outflows[i], height, Q_sur, q_data_size_sur, jac_data_size_sur,
                                 basis_q_sur, basis_x_sur, qf_apply_outflow, qf_apply_outflow_jacobian, op_apply, op_apply_ijacobian));
    }
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_apply_outflow));
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_apply_outflow_jacobian));
  }

  {  // --- Create Sub-Operator for freestream boundaries
    CeedQFunction qf_apply_freestream = NULL, qf_apply_freestream_jacobian = NULL;

    PetscCall(SetupBCQFunctions(ceed, dim_sur, num_comp_x, num_comp_q, q_data_size_sur, jac_data_size_sur, problem->apply_freestream,
                                problem->apply_freestream_jacobian, &qf_apply_freestream, &qf_apply_freestream_jacobian));
    for (CeedInt i = 0; i < bc->num_freestream; i++) {
      PetscCall(AddBCSubOperator(ceed, dm, ceed_data, face_sets_label, bc->freestreams[i], height, Q_sur, q_data_size_sur, jac_data_size_sur,
                                 basis_q_sur, basis_x_sur, qf_apply_freestream, qf_apply_freestream_jacobian, op_apply, op_apply_ijacobian));
    }
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_apply_freestream));
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_apply_freestream_jacobian));
  }

  {  // --- Create Sub-Operator for slip boundaries
    CeedQFunction qf_apply_slip = NULL, qf_apply_slip_jacobian = NULL;

    PetscCall(SetupBCQFunctions(ceed, dim_sur, num_comp_x, num_comp_q, q_data_size_sur, jac_data_size_sur, problem->apply_slip,
                                problem->apply_slip_jacobian, &qf_apply_slip, &qf_apply_slip_jacobian));
    for (CeedInt i = 0; i < bc->num_slip; i++) {
      PetscCall(AddBCSubOperator(ceed, dm, ceed_data, face_sets_label, bc->slips[i], height, Q_sur, q_data_size_sur, jac_data_size_sur, basis_q_sur,
                                 basis_x_sur, qf_apply_slip, qf_apply_slip_jacobian, op_apply, op_apply_ijacobian));
    }
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_apply_slip));
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_apply_slip_jacobian));
  }

  PetscCallCeed(ceed, CeedBasisDestroy(&basis_q_sur));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_x_sur));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetupLibceed(Ceed ceed, CeedData ceed_data, DM dm, User user, AppCtx app_ctx, ProblemData problem, SimpleBC bc) {
  const PetscInt      num_comp_q = 5;
  const CeedInt       dim = problem->dim, num_comp_x = problem->dim;
  CeedInt             jac_data_size_vol = num_comp_q + 6 + 3;
  CeedElemRestriction elem_restr_jd_i;
  CeedVector          jac_data;
  CeedOperator        op_ifunction_vol = NULL, op_rhs_vol = NULL, op_ijacobian_vol = NULL;

  PetscFunctionBeginUser;

  if (problem->apply_vol_ifunction.qfunction && problem->uses_newtonian) {
    NewtonianIdealGasContext gas;
    PetscCallCeed(ceed, CeedQFunctionContextGetDataRead(problem->apply_vol_ifunction.qfunction_context, CEED_MEM_HOST, &gas));
    jac_data_size_vol += (gas->idl_enable ? 1 : 0);
    PetscCallCeed(ceed, CeedQFunctionContextRestoreDataRead(problem->apply_vol_ifunction.qfunction_context, &gas));
  }

  {  // Create bases and element restrictions
    DMLabel  domain_label = NULL;
    PetscInt label_value = 0, height = 0, dm_field = 0;
    DM       dm_coord;

    PetscCall(DMGetCoordinateDM(dm, &dm_coord));
    PetscCall(CreateBasisFromPlex(ceed, dm, domain_label, label_value, height, dm_field, &ceed_data->basis_q));
    PetscCall(CreateBasisFromPlex(ceed, dm_coord, domain_label, label_value, height, dm_field, &ceed_data->basis_x));

    PetscCall(DMPlexCeedElemRestrictionCreate(ceed, dm, domain_label, label_value, height, 0, &ceed_data->elem_restr_q));
    PetscCall(DMPlexCeedElemRestrictionCoordinateCreate(ceed, dm, domain_label, label_value, height, &ceed_data->elem_restr_x));
    PetscCall(DMPlexCeedElemRestrictionQDataCreate(ceed, dm, domain_label, label_value, height, jac_data_size_vol, &elem_restr_jd_i));

    PetscCallCeed(ceed, CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &user->q_ceed, NULL));
    PetscCallCeed(ceed, CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &user->q_dot_ceed, NULL));
    PetscCallCeed(ceed, CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &user->g_ceed, NULL));
    PetscCallCeed(ceed, CeedElemRestrictionCreateVector(ceed_data->elem_restr_x, &ceed_data->x_coord, NULL));
    PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_jd_i, &jac_data, NULL));

    {  // -- Copy PETSc coordinate vector into CEED vector
      Vec X_loc;
      DM  cdm;

      PetscCall(DMGetCellCoordinateDM(dm, &cdm));
      if (cdm) {
        PetscCall(DMGetCellCoordinatesLocal(dm, &X_loc));
      } else {
        PetscCall(DMGetCoordinatesLocal(dm, &X_loc));
      }
      PetscCall(VecScale(X_loc, problem->dm_scale));
      PetscCall(VecCopyPetscToCeed(X_loc, ceed_data->x_coord));
    }

    PetscCall(QDataGet(ceed, dm, domain_label, label_value, ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord,
                       &ceed_data->elem_restr_qd_i, &ceed_data->q_data, &problem->q_data_size_vol));
  }

  {  // -- Create QFunction for ICs
    CeedBasis     basis_xc;
    CeedQFunction qf_ics;
    CeedOperator  op_ics;

    PetscCallCeed(ceed, CeedBasisCreateProjection(ceed_data->basis_x, ceed_data->basis_q, &basis_xc));
    PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, problem->ics.qfunction, problem->ics.qfunction_loc, &qf_ics));
    PetscCallCeed(ceed, CeedQFunctionSetContext(qf_ics, problem->ics.qfunction_context));
    PetscCallCeed(ceed, CeedQFunctionSetUserFlopsEstimate(qf_ics, 0));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ics, "x", num_comp_x, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ics, "dx", num_comp_x * dim, CEED_EVAL_GRAD));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_ics, "q0", num_comp_q, CEED_EVAL_NONE));

    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_ics, NULL, NULL, &op_ics));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ics, "x", ceed_data->elem_restr_x, basis_xc, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ics, "dx", ceed_data->elem_restr_x, basis_xc, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ics, "q0", ceed_data->elem_restr_q, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(op_ics, "evaluation time", &user->phys->ics_time_label));
    PetscCall(OperatorApplyContextCreate(NULL, dm, user->ceed, op_ics, ceed_data->x_coord, NULL, NULL, user->Q_loc, &ceed_data->op_ics_ctx));

    PetscCallCeed(ceed, CeedBasisDestroy(&basis_xc));
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_ics));
    PetscCallCeed(ceed, CeedOperatorDestroy(&op_ics));
  }

  if (problem->apply_vol_rhs.qfunction) {
    CeedQFunction qf_rhs_vol;

    PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, problem->apply_vol_rhs.qfunction, problem->apply_vol_rhs.qfunction_loc, &qf_rhs_vol));
    PetscCallCeed(ceed, CeedQFunctionSetContext(qf_rhs_vol, problem->apply_vol_rhs.qfunction_context));
    PetscCallCeed(ceed, CeedQFunctionSetUserFlopsEstimate(qf_rhs_vol, 0));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_vol, "q", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_vol, "Grad_q", num_comp_q * dim, CEED_EVAL_GRAD));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_vol, "qdata", problem->q_data_size_vol, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_vol, "x", num_comp_x, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_rhs_vol, "v", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_rhs_vol, "Grad_v", num_comp_q * dim, CEED_EVAL_GRAD));

    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_rhs_vol, NULL, NULL, &op_rhs_vol));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_vol, "q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_vol, "Grad_q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_vol, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_NONE, ceed_data->q_data));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_vol, "x", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_vol, "v", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_vol, "Grad_v", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));

    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_rhs_vol));
  }

  if (problem->apply_vol_ifunction.qfunction) {
    CeedQFunction qf_ifunction_vol;

    PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, problem->apply_vol_ifunction.qfunction, problem->apply_vol_ifunction.qfunction_loc,
                                                    &qf_ifunction_vol));
    PetscCallCeed(ceed, CeedQFunctionSetContext(qf_ifunction_vol, problem->apply_vol_ifunction.qfunction_context));
    PetscCallCeed(ceed, CeedQFunctionSetUserFlopsEstimate(qf_ifunction_vol, 0));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ifunction_vol, "q", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ifunction_vol, "Grad_q", num_comp_q * dim, CEED_EVAL_GRAD));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ifunction_vol, "q dot", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ifunction_vol, "qdata", problem->q_data_size_vol, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ifunction_vol, "x", num_comp_x, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_ifunction_vol, "v", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_ifunction_vol, "Grad_v", num_comp_q * dim, CEED_EVAL_GRAD));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_ifunction_vol, "jac_data", jac_data_size_vol, CEED_EVAL_NONE));

    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_ifunction_vol, NULL, NULL, &op_ifunction_vol));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ifunction_vol, "q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ifunction_vol, "Grad_q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ifunction_vol, "q dot", ceed_data->elem_restr_q, ceed_data->basis_q, user->q_dot_ceed));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ifunction_vol, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_NONE, ceed_data->q_data));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ifunction_vol, "x", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ifunction_vol, "v", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ifunction_vol, "Grad_v", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ifunction_vol, "jac_data", elem_restr_jd_i, CEED_BASIS_NONE, jac_data));

    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_ifunction_vol));
  }

  if (problem->apply_vol_ijacobian.qfunction) {
    CeedQFunction qf_ijacobian_vol;

    PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, problem->apply_vol_ijacobian.qfunction, problem->apply_vol_ijacobian.qfunction_loc,
                                                    &qf_ijacobian_vol));
    PetscCallCeed(ceed, CeedQFunctionSetContext(qf_ijacobian_vol, problem->apply_vol_ijacobian.qfunction_context));
    PetscCallCeed(ceed, CeedQFunctionSetUserFlopsEstimate(qf_ijacobian_vol, 0));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ijacobian_vol, "dq", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ijacobian_vol, "Grad_dq", num_comp_q * dim, CEED_EVAL_GRAD));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ijacobian_vol, "qdata", problem->q_data_size_vol, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ijacobian_vol, "jac_data", jac_data_size_vol, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_ijacobian_vol, "v", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_ijacobian_vol, "Grad_v", num_comp_q * dim, CEED_EVAL_GRAD));

    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_ijacobian_vol, NULL, NULL, &op_ijacobian_vol));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ijacobian_vol, "dq", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ijacobian_vol, "Grad_dq", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ijacobian_vol, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_NONE, ceed_data->q_data));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ijacobian_vol, "jac_data", elem_restr_jd_i, CEED_BASIS_NONE, jac_data));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ijacobian_vol, "v", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ijacobian_vol, "Grad_v", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));

    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_ijacobian_vol));
  }

  // -- Create and apply CEED Composite Operator for the entire domain
  if (!user->phys->implicit) {  // RHS
    CeedOperator op_rhs;

    PetscCallCeed(ceed, CeedCompositeOperatorCreate(ceed, &op_rhs));
    PetscCallCeed(ceed, CeedCompositeOperatorAddSub(op_rhs, op_rhs_vol));
    PetscCall(AddBCSubOperators(user, ceed, dm, bc, problem, ceed_data, op_rhs, NULL));

    PetscCall(OperatorApplyContextCreate(dm, dm, ceed, op_rhs, user->q_ceed, user->g_ceed, user->Q_loc, NULL, &user->op_rhs_ctx));

    // ----- Get Context Labels for Operator
    PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(op_rhs, "solution time", &user->phys->solution_time_label));
    PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(op_rhs, "timestep size", &user->phys->timestep_size_label));

    PetscCallCeed(ceed, CeedOperatorDestroy(&op_rhs));
    PetscCall(CreateKSPMass(user, problem));
    PetscCheck(app_ctx->sgs_model_type == SGS_MODEL_NONE, user->comm, PETSC_ERR_SUP, "SGS modeling not implemented for explicit timestepping");
  } else {  // IFunction
    CeedOperator op_ijacobian = NULL;

    // Create Composite Operaters
    PetscCallCeed(ceed, CeedCompositeOperatorCreate(ceed, &user->op_ifunction));
    PetscCallCeed(ceed, CeedCompositeOperatorAddSub(user->op_ifunction, op_ifunction_vol));
    if (op_ijacobian_vol) {
      PetscCallCeed(ceed, CeedCompositeOperatorCreate(ceed, &op_ijacobian));
      PetscCallCeed(ceed, CeedCompositeOperatorAddSub(op_ijacobian, op_ijacobian_vol));
    }
    PetscCall(AddBCSubOperators(user, ceed, dm, bc, problem, ceed_data, user->op_ifunction, op_ijacobian));

    // ----- Get Context Labels for Operator
    PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(user->op_ifunction, "solution time", &user->phys->solution_time_label));
    PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(user->op_ifunction, "timestep size", &user->phys->timestep_size_label));

    if (op_ijacobian) {
      PetscCall(MatCeedCreate(user->dm, user->dm, op_ijacobian, NULL, &user->mat_ijacobian));
      PetscCall(MatCeedSetLocalVectors(user->mat_ijacobian, user->Q_dot_loc, NULL));
      PetscCallCeed(ceed, CeedOperatorDestroy(&op_ijacobian));
    }
    if (app_ctx->sgs_model_type == SGS_MODEL_DATA_DRIVEN) PetscCall(SgsDDSetup(ceed, user, ceed_data, problem));
  }

  if (problem->use_strong_bc_ceed) PetscCall(SetupStrongBC_Ceed(ceed, ceed_data, dm, user, problem, bc));
  if (app_ctx->turb_spanstats_enable) PetscCall(TurbulenceStatisticsSetup(ceed, user, ceed_data, problem));
  if (app_ctx->diff_filter_monitor && !user->diff_filter) PetscCall(DifferentialFilterSetup(ceed, user, ceed_data, problem));
  if (app_ctx->sgs_train_enable) PetscCall(SGS_DD_TrainingSetup(ceed, user, ceed_data, problem));

  PetscCallCeed(ceed, CeedVectorDestroy(&jac_data));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_jd_i));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_ijacobian_vol));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_ifunction_vol));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_rhs_vol));
  PetscFunctionReturn(PETSC_SUCCESS);
}
