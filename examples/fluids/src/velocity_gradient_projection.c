// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
/// @file
/// Functions for setting up and projecting the velocity gradient

#include "../qfunctions/velocity_gradient_projection.h"

#include <petscdmplex.h>

#include "../navierstokes.h"

PetscErrorCode VelocityGradientProjectionCreateDM(NodalProjectionData grad_velo_proj, User user, PetscInt degree) {
  PetscSection section;

  PetscFunctionBeginUser;
  grad_velo_proj->num_comp = 9;  // 9 velocity gradient

  PetscCall(DMClone(user->dm, &grad_velo_proj->dm));
  PetscCall(PetscObjectSetName((PetscObject)grad_velo_proj->dm, "Velocity Gradient Projection"));

  PetscCall(
      DMSetupByOrder_FEM(PETSC_TRUE, PETSC_TRUE, user->app_ctx->degree, 1, user->app_ctx->q_extra, 1, &grad_velo_proj->num_comp, grad_velo_proj->dm));

  PetscCall(DMGetLocalSection(grad_velo_proj->dm, &section));
  PetscCall(PetscSectionSetFieldName(section, 0, ""));
  PetscCall(PetscSectionSetComponentName(section, 0, 0, "VelocityGradientXX"));
  PetscCall(PetscSectionSetComponentName(section, 0, 1, "VelocityGradientXY"));
  PetscCall(PetscSectionSetComponentName(section, 0, 2, "VelocityGradientXZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 3, "VelocityGradientYX"));
  PetscCall(PetscSectionSetComponentName(section, 0, 4, "VelocityGradientYY"));
  PetscCall(PetscSectionSetComponentName(section, 0, 5, "VelocityGradientYZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 6, "VelocityGradientZX"));
  PetscCall(PetscSectionSetComponentName(section, 0, 7, "VelocityGradientZY"));
  PetscCall(PetscSectionSetComponentName(section, 0, 8, "VelocityGradientZZ"));
  PetscFunctionReturn(PETSC_SUCCESS);
};

PetscErrorCode VelocityGradientProjectionSetup(Ceed ceed, User user, CeedData ceed_data, ProblemData problem, StateVariable state_var_input,
                                               CeedElemRestriction elem_restr_input, CeedBasis basis_input, NodalProjectionData *pgrad_velo_proj) {
  NodalProjectionData grad_velo_proj;
  CeedOperator        op_rhs_assemble, op_mass;
  CeedQFunction       qf_rhs_assemble, qf_mass;
  CeedBasis           basis_grad_velo;
  CeedElemRestriction elem_restr_grad_velo;
  PetscInt            dim, label_value = 0, height = 0, dm_field = 0;
  CeedInt             num_comp_x, num_comp_input, q_data_size;
  DMLabel             domain_label = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&grad_velo_proj));

  PetscCall(VelocityGradientProjectionCreateDM(grad_velo_proj, user, user->app_ctx->degree));

  // -- Get Pre-requisite things
  PetscCall(DMGetDimension(grad_velo_proj->dm, &dim));
  PetscCallCeed(ceed, CeedBasisGetNumComponents(ceed_data->basis_x, &num_comp_x));
  PetscCallCeed(ceed, CeedBasisGetNumComponents(ceed_data->basis_q, &num_comp_input));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_qd_i, &q_data_size));
  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, grad_velo_proj->dm, domain_label, label_value, height, dm_field, &elem_restr_grad_velo));

  PetscCall(CreateBasisFromPlex(ceed, grad_velo_proj->dm, domain_label, label_value, height, dm_field, &basis_grad_velo));

  // -- Build RHS operator
  switch (state_var_input) {
    case STATEVAR_PRIMITIVE:
      PetscCallCeed(
          ceed, CeedQFunctionCreateInterior(ceed, 1, VelocityGradientProjectionRHS_Prim, VelocityGradientProjectionRHS_Prim_loc, &qf_rhs_assemble));
      break;
    case STATEVAR_CONSERVATIVE:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, VelocityGradientProjectionRHS_Conserv, VelocityGradientProjectionRHS_Conserv_loc,
                                                      &qf_rhs_assemble));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)user->dm), PETSC_ERR_SUP, "No velocity gradient projection QFunction for chosen state variable");
  }

  PetscCallCeed(ceed, CeedQFunctionSetContext(qf_rhs_assemble, problem->apply_vol_ifunction.qfunction_context));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_assemble, "q", num_comp_input, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_assemble, "Grad_q", num_comp_input * dim, CEED_EVAL_GRAD));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_assemble, "qdata", q_data_size, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_assemble, "x", num_comp_x, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_rhs_assemble, "velocity gradient", grad_velo_proj->num_comp, CEED_EVAL_INTERP));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_rhs_assemble, NULL, NULL, &op_rhs_assemble));
  PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_assemble, "q", elem_restr_input, basis_input, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_assemble, "Grad_q", elem_restr_input, basis_input, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_assemble, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_NONE, ceed_data->q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_assemble, "x", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord));
  PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_assemble, "velocity gradient", elem_restr_grad_velo, basis_grad_velo, CEED_VECTOR_ACTIVE));

  PetscCall(OperatorApplyContextCreate(user->dm, grad_velo_proj->dm, ceed, op_rhs_assemble, NULL, NULL, NULL, NULL, &grad_velo_proj->l2_rhs_ctx));

  // -- Build Mass operator
  PetscCall(CreateMassQFunction(ceed, grad_velo_proj->num_comp, q_data_size, &qf_mass));
  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "u", elem_restr_grad_velo, basis_grad_velo, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_NONE, ceed_data->q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "v", elem_restr_grad_velo, basis_grad_velo, CEED_VECTOR_ACTIVE));

  {  // -- Setup KSP for L^2 projection with lumped mass operator
    Mat      mat_mass;
    MPI_Comm comm = PetscObjectComm((PetscObject)grad_velo_proj->dm);

    PetscCall(MatCeedCreate(grad_velo_proj->dm, grad_velo_proj->dm, op_mass, NULL, &mat_mass));

    PetscCall(KSPCreate(comm, &grad_velo_proj->ksp));
    PetscCall(KSPSetOptionsPrefix(grad_velo_proj->ksp, "velocity_gradient_projection_"));
    {
      PC pc;
      PetscCall(KSPGetPC(grad_velo_proj->ksp, &pc));
      PetscCall(PCSetType(pc, PCJACOBI));
      PetscCall(PCJacobiSetType(pc, PC_JACOBI_ROWSUM));
      PetscCall(KSPSetType(grad_velo_proj->ksp, KSPPREONLY));
    }
    PetscCall(KSPSetFromOptions_WithMatCeed(grad_velo_proj->ksp, mat_mass));
    PetscCall(MatDestroy(&mat_mass));
  }

  *pgrad_velo_proj = grad_velo_proj;

  PetscCallCeed(ceed, CeedBasisDestroy(&basis_grad_velo));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_grad_velo));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_rhs_assemble));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_mass));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_rhs_assemble));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_mass));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VelocityGradientProjectionApply(NodalProjectionData grad_velo_proj, Vec Q_loc, Vec VelocityGradient) {
  OperatorApplyContext l2_rhs_ctx = grad_velo_proj->l2_rhs_ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(FLUIDS_VelocityGradientProjection, Q_loc, VelocityGradient, 0, 0));
  PetscCall(ApplyCeedOperatorLocalToGlobal(Q_loc, VelocityGradient, l2_rhs_ctx));

  PetscCall(KSPSolve(grad_velo_proj->ksp, VelocityGradient, VelocityGradient));
  PetscCall(PetscLogEventEnd(FLUIDS_VelocityGradientProjection, Q_loc, VelocityGradient, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}
