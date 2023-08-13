// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
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
  PetscFE      fe;
  PetscSection section;
  PetscInt     dim;

  PetscFunctionBeginUser;
  grad_velo_proj->num_comp = 9;  // 9 velocity gradient

  PetscCall(DMClone(user->dm, &grad_velo_proj->dm));
  PetscCall(DMGetDimension(grad_velo_proj->dm, &dim));
  PetscCall(PetscObjectSetName((PetscObject)grad_velo_proj->dm, "Velocity Gradient Projection"));

  PetscInt q_order = user->app_ctx->degree + user->app_ctx->q_extra;
  PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, dim, grad_velo_proj->num_comp, PETSC_FALSE, degree, q_order, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "Velocity Gradient Projection"));
  PetscCall(DMAddField(grad_velo_proj->dm, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(grad_velo_proj->dm));
  PetscCall(DMPlexSetClosurePermutationTensor(grad_velo_proj->dm, PETSC_DETERMINE, NULL));

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

  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(PETSC_SUCCESS);
};

PetscErrorCode VelocityGradientProjectionSetup(Ceed ceed, User user, CeedData ceed_data, ProblemData *problem) {
  OperatorApplyContext mass_matop_ctx;
  CeedOperator         op_rhs_assemble, op_mass;
  CeedQFunction        qf_rhs_assemble, qf_mass;
  CeedBasis            basis_grad_velo;
  CeedElemRestriction  elem_restr_grad_velo;
  PetscInt             dim;
  CeedInt              num_comp_x, num_comp_q, q_data_size;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&user->grad_velo_proj));
  NodalProjectionData grad_velo_proj = user->grad_velo_proj;

  PetscCall(VelocityGradientProjectionCreateDM(grad_velo_proj, user, user->app_ctx->degree));

  // -- Get Pre-requisite things
  PetscCall(DMGetDimension(grad_velo_proj->dm, &dim));
  PetscCallCeed(ceed, CeedBasisGetNumComponents(ceed_data->basis_x, &num_comp_x));
  PetscCallCeed(ceed, CeedBasisGetNumComponents(ceed_data->basis_q, &num_comp_q));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_qd_i, &q_data_size));
  PetscCall(GetRestrictionForDomain(ceed, grad_velo_proj->dm, 0, 0, 0, 0, -1, 0, &elem_restr_grad_velo, NULL, NULL));

  PetscCall(CreateBasisFromPlex(ceed, grad_velo_proj->dm, 0, 0, 0, 0, &basis_grad_velo));

  // -- Build RHS operator
  switch (user->phys->state_var) {
    case STATEVAR_PRIMITIVE:
      PetscCallCeed(
          ceed, CeedQFunctionCreateInterior(ceed, 1, VelocityGradientProjectionRHS_Prim, VelocityGradientProjectionRHS_Prim_loc, &qf_rhs_assemble));
      break;
    case STATEVAR_CONSERVATIVE:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, VelocityGradientProjectionRHS_Conserv, VelocityGradientProjectionRHS_Conserv_loc,
                                                      &qf_rhs_assemble));
      break;
    case STATEVAR_ENTROPY:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, VelocityGradientProjectionRHS_Entropy, VelocityGradientProjectionRHS_Entropy_loc,
                                                      &qf_rhs_assemble));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)user->dm), PETSC_ERR_SUP, "No velocity gradient projection QFunction for chosen state variable");
  }

  PetscCallCeed(ceed, CeedQFunctionSetContext(qf_rhs_assemble, problem->apply_vol_ifunction.qfunction_context));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_assemble, "q", num_comp_q, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_assemble, "Grad_q", num_comp_q * dim, CEED_EVAL_GRAD));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_assemble, "qdata", q_data_size, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_assemble, "x", num_comp_x, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_rhs_assemble, "velocity gradient", grad_velo_proj->num_comp, CEED_EVAL_INTERP));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_rhs_assemble, NULL, NULL, &op_rhs_assemble));
  PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_assemble, "q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_assemble, "Grad_q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_assemble, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_COLLOCATED, ceed_data->q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_assemble, "x", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord));
  PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_assemble, "velocity gradient", elem_restr_grad_velo, basis_grad_velo, CEED_VECTOR_ACTIVE));

  PetscCall(OperatorApplyContextCreate(user->dm, grad_velo_proj->dm, ceed, op_rhs_assemble, NULL, NULL, NULL, NULL, &grad_velo_proj->l2_rhs_ctx));

  // -- Build Mass operator
  PetscCall(CreateMassQFunction(ceed, grad_velo_proj->num_comp, q_data_size, &qf_mass));
  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "u", elem_restr_grad_velo, basis_grad_velo, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_COLLOCATED, ceed_data->q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "v", elem_restr_grad_velo, basis_grad_velo, CEED_VECTOR_ACTIVE));

  {  // -- Setup KSP for L^2 projection with lumped mass operator
    Mat      mat_mass;
    MPI_Comm comm = PetscObjectComm((PetscObject)grad_velo_proj->dm);

    PetscCall(OperatorApplyContextCreate(grad_velo_proj->dm, grad_velo_proj->dm, ceed, op_mass, NULL, NULL, NULL, NULL, &mass_matop_ctx));
    PetscCall(CreateMatShell_Ceed(mass_matop_ctx, &mat_mass));

    PetscCall(KSPCreate(comm, &grad_velo_proj->ksp));
    PetscCall(KSPSetOptionsPrefix(grad_velo_proj->ksp, "velocity_gradient_projection_"));
    {
      PC pc;
      PetscCall(KSPGetPC(grad_velo_proj->ksp, &pc));
      PetscCall(PCSetType(pc, PCJACOBI));
      PetscCall(PCJacobiSetType(pc, PC_JACOBI_ROWSUM));
      PetscCall(KSPSetType(grad_velo_proj->ksp, KSPPREONLY));
      // TODO Not sure if the option below are necessary
      PetscCall(KSPSetConvergenceTest(grad_velo_proj->ksp, KSPConvergedSkip, NULL, NULL));
    }
    PetscCall(KSPSetOperators(grad_velo_proj->ksp, mat_mass, mat_mass));
    PetscCall(KSPSetFromOptions(grad_velo_proj->ksp));
  }

  PetscCallCeed(ceed, CeedBasisDestroy(&basis_grad_velo));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_grad_velo));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_rhs_assemble));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_mass));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_rhs_assemble));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_mass));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VelocityGradientProjectionApply(User user, Vec Q_loc, Vec VelocityGradient) {
  NodalProjectionData  grad_velo_proj = user->grad_velo_proj;
  OperatorApplyContext l2_rhs_ctx     = grad_velo_proj->l2_rhs_ctx;

  PetscFunctionBeginUser;
  PetscCall(ApplyCeedOperatorLocalToGlobal(Q_loc, VelocityGradient, l2_rhs_ctx));

  PetscCall(KSPSolve(grad_velo_proj->ksp, VelocityGradient, VelocityGradient));

  PetscFunctionReturn(PETSC_SUCCESS);
}
