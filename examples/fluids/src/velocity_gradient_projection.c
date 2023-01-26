// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
/// @file
/// Functions for setting up and projecting the velocity gradient

#include "../qfunctions/velocity_gradient_projection.h"

#include "../navierstokes.h"

PetscErrorCode VelocityGradientProjectionCreateDM(NodalProjectionData grad_velo_proj, User user, ProblemData *problem, PetscInt degree) {
  PetscFE      fe;
  PetscSection section;
  PetscInt     dim;

  PetscFunctionBeginUser;
  grad_velo_proj->num_comp = 9;  // 9 velocity gradient

  PetscCall(DMClone(user->dm, &grad_velo_proj->dm));
  PetscCall(DMGetDimension(grad_velo_proj->dm, &dim));
  PetscCall(PetscObjectSetName((PetscObject)grad_velo_proj->dm, "Velocity Gradient Projection"));

  PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, dim, grad_velo_proj->num_comp, PETSC_FALSE, degree, PETSC_DECIDE, &fe));
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
  PetscFunctionReturn(0);
};

PetscErrorCode VelocityGradientProjectionSetup(Ceed ceed, User user, CeedData ceed_data, ProblemData *problem) {
  MatopApplyContext   mass_matop_ctx;
  CeedOperator        op_rhs_assemble, op_mass;
  CeedQFunction       qf_rhs_assemble, qf_mass;
  CeedBasis           basis_grad_velo;
  CeedVector          q_ceed, rhs_ceed, mass_output;
  CeedElemRestriction elem_restr_grad_velo;
  PetscInt            dim, num_comp_x, num_comp_q, q_data_size, num_qpts_1d, num_nodes_1d;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&user->grad_velo_proj));
  NodalProjectionData grad_velo_proj = user->grad_velo_proj;

  PetscCall(VelocityGradientProjectionCreateDM(grad_velo_proj, user, problem, user->app_ctx->degree));

  // -- Get Pre-requisite things
  PetscCall(DMGetDimension(grad_velo_proj->dm, &dim));
  CeedBasisGetNumComponents(ceed_data->basis_x, &num_comp_x);
  CeedBasisGetNumComponents(ceed_data->basis_q, &num_comp_q);
  CeedBasisGetNumQuadraturePoints1D(ceed_data->basis_q, &num_qpts_1d);
  CeedBasisGetNumNodes1D(ceed_data->basis_q, &num_nodes_1d);
  CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_qd_i, &q_data_size);

  PetscCall(GetRestrictionForDomain(ceed, grad_velo_proj->dm, 0, 0, 0, num_qpts_1d, q_data_size, &elem_restr_grad_velo, NULL, NULL));

  CeedBasisCreateTensorH1Lagrange(ceed, dim, grad_velo_proj->num_comp, num_nodes_1d, num_qpts_1d, CEED_GAUSS, &basis_grad_velo);

  // -- Build RHS operator
  switch (user->phys->state_var) {
    case STATEVAR_PRIMITIVE:
      CeedQFunctionCreateInterior(ceed, 1, VelocityGradientProjection_Prim, VelocityGradientProjection_Prim_loc, &qf_rhs_assemble);
      break;
    case STATEVAR_CONSERVATIVE:
      CeedQFunctionCreateInterior(ceed, 1, VelocityGradientProjection_Conserv, VelocityGradientProjection_Conserv_loc, &qf_rhs_assemble);
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)user->dm), PETSC_ERR_SUP, "No velocity gradient projection QFunction for chosen state variable");
  }

  CeedQFunctionSetContext(qf_rhs_assemble, problem->apply_vol_ifunction.qfunction_context);
  CeedQFunctionAddInput(qf_rhs_assemble, "q", num_comp_q, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_rhs_assemble, "Grad_q", num_comp_q * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_rhs_assemble, "qdata", q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_rhs_assemble, "x", num_comp_x, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_rhs_assemble, "velocity gradient", grad_velo_proj->num_comp, CEED_EVAL_INTERP);

  CeedOperatorCreate(ceed, qf_rhs_assemble, NULL, NULL, &op_rhs_assemble);
  CeedOperatorSetField(op_rhs_assemble, "q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_rhs_assemble, "Grad_q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_rhs_assemble, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_COLLOCATED, ceed_data->q_data);
  CeedOperatorSetField(op_rhs_assemble, "x", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord);
  CeedOperatorSetField(op_rhs_assemble, "velocity gradient", elem_restr_grad_velo, basis_grad_velo, CEED_VECTOR_ACTIVE);

  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &q_ceed, NULL);
  CeedElemRestrictionCreateVector(elem_restr_grad_velo, &rhs_ceed, NULL);

  PetscCall(MatopApplyContextCreate(user->dm, grad_velo_proj->dm, ceed, op_rhs_assemble, q_ceed, rhs_ceed, NULL, NULL, &grad_velo_proj->l2_rhs_ctx));

  // -- Build Lumped Mass Matrix
  PetscCall(CreateMassQFunction(ceed, grad_velo_proj->num_comp, q_data_size, &qf_mass));
  CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass);
  CeedOperatorSetField(op_mass, "u", elem_restr_grad_velo, basis_grad_velo, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_COLLOCATED, ceed_data->q_data);
  CeedOperatorSetField(op_mass, "v", elem_restr_grad_velo, basis_grad_velo, CEED_VECTOR_ACTIVE);

  {  // -- Setup KSP for L^2 projection with lumped mass operator
    PetscInt l_size, g_size;
    Mat      mat_mass;
    VecType  vec_type;
    Vec      M_inv;
    MPI_Comm comm = PetscObjectComm((PetscObject)grad_velo_proj->dm);

    PetscCall(DMGetGlobalVector(grad_velo_proj->dm, &M_inv));
    PetscCall(VecGetLocalSize(M_inv, &l_size));
    PetscCall(VecGetSize(M_inv, &g_size));
    PetscCall(VecGetType(M_inv, &vec_type));
    PetscCall(DMRestoreGlobalVector(grad_velo_proj->dm, &M_inv));

    CeedElemRestrictionCreateVector(elem_restr_grad_velo, &mass_output, NULL);
    PetscCall(MatopApplyContextCreate(grad_velo_proj->dm, grad_velo_proj->dm, ceed, op_mass, rhs_ceed, mass_output, NULL, NULL, &mass_matop_ctx));
    CeedVectorDestroy(&mass_output);

    PetscCall(MatCreateShell(comm, l_size, l_size, g_size, g_size, mass_matop_ctx, &mat_mass));
    PetscCall(MatShellSetContextDestroy(mat_mass, (PetscErrorCode(*)(void *))MatopApplyContextDestroy));
    PetscCall(MatShellSetOperation(mat_mass, MATOP_MULT, (void (*)(void))MatMult_Ceed));
    PetscCall(MatShellSetOperation(mat_mass, MATOP_GET_DIAGONAL, (void (*)(void))MatGetDiag_Ceed));
    PetscCall(MatShellSetVecType(mat_mass, vec_type));

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

  CeedVectorDestroy(&q_ceed);
  CeedVectorDestroy(&rhs_ceed);
  CeedBasisDestroy(&basis_grad_velo);
  CeedElemRestrictionDestroy(&elem_restr_grad_velo);
  CeedQFunctionDestroy(&qf_rhs_assemble);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_rhs_assemble);
  CeedOperatorDestroy(&op_mass);
  PetscFunctionReturn(0);
}

PetscErrorCode VelocityGradientProjectionApply(User user, Vec Q_loc, Vec *VelocityGradient) {
  NodalProjectionData grad_velo_proj = user->grad_velo_proj;
  MatopApplyContext   l2_rhs_ctx     = grad_velo_proj->l2_rhs_ctx;
  PetscScalar        *x, *y;
  PetscMemType        x_mem_type, y_mem_type;
  Vec                 Y_loc;
  PetscFunctionBeginUser;

  // -- Create global output vector
  PetscCall(DMGetGlobalVector(grad_velo_proj->dm, VelocityGradient));
  PetscCall(VecZeroEntries(*VelocityGradient));

  // -- Get RHS of projection problem
  PetscCall(VecGetArrayReadAndMemType(Q_loc, (const PetscScalar **)&x, &x_mem_type));
  CeedVectorSetArray(l2_rhs_ctx->x_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER, x);

  if (l2_rhs_ctx->Y_loc) Y_loc = l2_rhs_ctx->Y_loc;
  else PetscCall(DMGetLocalVector(l2_rhs_ctx->dm_y, &Y_loc));

  PetscCall(VecGetArrayAndMemType(Y_loc, &y, &y_mem_type));
  CeedVectorSetArray(l2_rhs_ctx->y_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER, y);

  // Apply libCEED operator
  CeedOperatorApply(l2_rhs_ctx->op, l2_rhs_ctx->x_ceed, l2_rhs_ctx->y_ceed, CEED_REQUEST_IMMEDIATE);

  CeedVectorTakeArray(l2_rhs_ctx->x_ceed, MemTypeP2C(x_mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(Q_loc, (const PetscScalar **)&x));

  CeedVectorTakeArray(l2_rhs_ctx->y_ceed, MemTypeP2C(y_mem_type), NULL);
  PetscCall(VecRestoreArrayAndMemType(Y_loc, &y));
  PetscCall(VecZeroEntries(*VelocityGradient));
  PetscCall(DMLocalToGlobal(l2_rhs_ctx->dm_y, Y_loc, ADD_VALUES, *VelocityGradient));
  if (!l2_rhs_ctx->Y_loc) PetscCall(DMRestoreLocalVector(l2_rhs_ctx->dm_y, &Y_loc));

  // -- Solve L^2 projection via lumped mass matrix
  PetscCall(KSPSolve(grad_velo_proj->ksp, *VelocityGradient, *VelocityGradient));

  PetscFunctionReturn(0);
}
