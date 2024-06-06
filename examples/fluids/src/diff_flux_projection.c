// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
/// @file
/// Functions for setting up and projecting the divergence of the diffusive flux

#include "../qfunctions/diff_flux_projection.h"

#include <petscdmplex.h>

#include "../navierstokes.h"

PetscErrorCode DiffFluxProjectionInitialize(User user, CeedElemRestriction *elem_restr_diff_flux, CeedBasis *basis_diff_flux) {
  PetscSection        section;
  PetscInt            label_value = 0, height = 0, dm_field = 0;
  DMLabel             domain_label   = NULL;
  NodalProjectionData diff_flux_proj = user->diff_flux_proj;

  PetscFunctionBeginUser;
  diff_flux_proj->num_comp = 4;

  PetscCall(DMClone(user->dm, &diff_flux_proj->dm));
  // PetscCall(PetscObjectSetName((PetscObject)diff_flux_proj->dm, "Divergence of Diffusive Flux Projection"));
  PetscCall(PetscObjectSetName((PetscObject)diff_flux_proj->dm, "DivDiffFluxProj"));

  PetscCall(
      DMSetupByOrder_FEM(PETSC_TRUE, PETSC_TRUE, user->app_ctx->degree, 1, user->app_ctx->q_extra, 1, &diff_flux_proj->num_comp, diff_flux_proj->dm));

  PetscCall(DMGetLocalSection(diff_flux_proj->dm, &section));
  PetscCall(PetscSectionSetFieldName(section, 0, ""));
  // PetscCall(PetscSectionSetComponentName(section, 0, 0, "DivergenceDiffusiveFlux_MomentumX"));
  // PetscCall(PetscSectionSetComponentName(section, 0, 1, "DivergenceDiffusiveFlux_MomentumY"));
  // PetscCall(PetscSectionSetComponentName(section, 0, 2, "DivergenceDiffusiveFlux_MomentumZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 0, "DivDiffusiveFlux_MomentumX"));
  PetscCall(PetscSectionSetComponentName(section, 0, 1, "DivDiffusiveFlux_MomentumY"));
  PetscCall(PetscSectionSetComponentName(section, 0, 2, "DivDiffusiveFlux_MomentumZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 3, "DivergenceDiffusiveFlux_Energy"));

  PetscCall(DMPlexCeedElemRestrictionCreate(user->ceed, diff_flux_proj->dm, domain_label, label_value, height, dm_field, elem_restr_diff_flux));
  PetscCallCeed(user->ceed, CeedElemRestrictionCreateVector(*elem_restr_diff_flux, &user->divFdiff_ceed, NULL));

  PetscCall(CreateBasisFromPlex(user->ceed, diff_flux_proj->dm, domain_label, label_value, height, dm_field, basis_diff_flux));
  PetscFunctionReturn(PETSC_SUCCESS);
};

PetscErrorCode DivDiffFluxProjectionSetup(Ceed ceed, User user, CeedData ceed_data, ProblemData problem) {
  NodalProjectionData diff_flux_proj;
  CeedOperator        op_rhs_assemble, op_mass;
  CeedQFunction       qf_rhs_assemble, qf_mass;
  CeedBasis           basis_diff_flux;
  CeedElemRestriction elem_restr_diff_flux;
  CeedInt             num_comp_q, qdata_size;
  PetscInt            dim;

  PetscFunctionBeginUser;
  diff_flux_proj = user->diff_flux_proj;

  // -- Get Pre-requisite things
  PetscCall(DMGetDimension(diff_flux_proj->dm, &dim));
  PetscCallCeed(ceed, CeedBasisGetNumComponents(ceed_data->basis_q, &num_comp_q));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_qd_i, &qdata_size));

  {  // Get elem_restr_diff_flux and basis_diff_flux
    CeedOperator     *sub_ops;
    CeedOperatorField op_field;
    PetscInt          sub_op_index = 0;  // will be 0 for the volume op

    PetscCallCeed(ceed, CeedCompositeOperatorGetSubList(user->op_ifunction, &sub_ops));
    PetscCallCeed(ceed, CeedOperatorGetFieldByName(sub_ops[sub_op_index], "div F_diff", &op_field));
    PetscCallCeed(ceed, CeedOperatorFieldGetElemRestriction(op_field, &elem_restr_diff_flux));
    PetscCallCeed(ceed, CeedOperatorFieldGetBasis(op_field, &basis_diff_flux));
  }

  // -- Build RHS operator
  switch (user->phys->state_var) {
    case STATEVAR_PRIMITIVE:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DivDiffusiveFluxRHS_Prim, DivDiffusiveFluxRHS_Prim_loc, &qf_rhs_assemble));
      break;
    case STATEVAR_CONSERVATIVE:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DivDiffusiveFluxRHS_Conserv, DivDiffusiveFluxRHS_Conserv_loc, &qf_rhs_assemble));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)user->dm), PETSC_ERR_SUP, "No velocity gradient projection QFunction for chosen state variable");
  }

  PetscCallCeed(ceed, CeedQFunctionSetContext(qf_rhs_assemble, problem->apply_vol_ifunction.qfunction_context));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_assemble, "q", num_comp_q, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_assemble, "Grad_q", num_comp_q * dim, CEED_EVAL_GRAD));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_assemble, "qdata", qdata_size, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_rhs_assemble, "diffusive flux RHS", diff_flux_proj->num_comp * dim, CEED_EVAL_GRAD));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_rhs_assemble, NULL, NULL, &op_rhs_assemble));
  PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_assemble, "q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_assemble, "Grad_q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_assemble, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_NONE, ceed_data->q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_assemble, "diffusive flux RHS", elem_restr_diff_flux, basis_diff_flux, CEED_VECTOR_ACTIVE));

  PetscCall(OperatorApplyContextCreate(user->dm, diff_flux_proj->dm, ceed, op_rhs_assemble, NULL, NULL, NULL, NULL, &diff_flux_proj->l2_rhs_ctx));

  // -- Build Mass operator
  PetscCall(CreateMassQFunction(ceed, diff_flux_proj->num_comp, qdata_size, &qf_mass));
  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "u", elem_restr_diff_flux, basis_diff_flux, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_NONE, ceed_data->q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "v", elem_restr_diff_flux, basis_diff_flux, CEED_VECTOR_ACTIVE));

  {  // -- Setup KSP for L^2 projection
    Mat      mat_mass;
    MPI_Comm comm = PetscObjectComm((PetscObject)diff_flux_proj->dm);

    PetscCall(MatCeedCreate(diff_flux_proj->dm, diff_flux_proj->dm, op_mass, NULL, &mat_mass));

    PetscCall(KSPCreate(comm, &diff_flux_proj->ksp));
    PetscCall(KSPSetOptionsPrefix(diff_flux_proj->ksp, "div_diff_flux_projection_"));
    {  // lumped by default
      PC pc;
      PetscCall(KSPGetPC(diff_flux_proj->ksp, &pc));
      PetscCall(PCSetType(pc, PCJACOBI));
      PetscCall(PCJacobiSetType(pc, PC_JACOBI_ROWSUM));
      PetscCall(KSPSetType(diff_flux_proj->ksp, KSPPREONLY));
    }
    PetscCall(KSPSetFromOptions_WithMatCeed(diff_flux_proj->ksp, mat_mass));
    PetscCall(MatDestroy(&mat_mass));
  }

  PetscCallCeed(ceed, CeedBasisDestroy(&basis_diff_flux));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_diff_flux));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_rhs_assemble));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_mass));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_rhs_assemble));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_mass));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DiffFluxProjectionApply(NodalProjectionData diff_flux_proj, Vec Q_loc, Vec DivDiffFlux) {
  OperatorApplyContext l2_rhs_ctx = diff_flux_proj->l2_rhs_ctx;

  PetscFunctionBeginUser;
  PetscCall(ApplyCeedOperatorLocalToGlobal(Q_loc, DivDiffFlux, l2_rhs_ctx));
  PetscCall(VecViewFromOptions(DivDiffFlux, NULL, "-div_diff_flux_proj_rhs_view"));

  PetscCall(KSPSolve(diff_flux_proj->ksp, DivDiffFlux, DivDiffFlux));
  PetscCall(VecViewFromOptions(DivDiffFlux, NULL, "-div_diff_flux_proj_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}
