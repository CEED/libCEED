// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../qfunctions/strong_boundary_conditions.h"

#include <ceed.h>
#include <petscdmplex.h>

#include "../navierstokes.h"
#include "../problems/stg_shur14.h"

PetscErrorCode SetupStrongSTG_Ceed(Ceed ceed, CeedData ceed_data, DM dm, ProblemData *problem, SimpleBC bc, Physics phys, CeedOperator op_strong_bc) {
  CeedInt num_comp_x = problem->dim, num_comp_q = 5, num_elem, elem_size, stg_data_size = 1, dim_boundary = 2, dXdx_size = num_comp_x * dim_boundary;
  CeedVector          multiplicity, x_stored, scale_stored, stg_data, dXdx;
  CeedBasis           basis_x_to_q_sur;
  CeedElemRestriction elem_restr_x_sur, elem_restr_q_sur, elem_restr_x_stored, elem_restr_scale, elem_restr_stgdata, elem_restr_dXdx;
  CeedQFunction       qf_setup, qf_strongbc, qf_stgdata;
  CeedOperator        op_setup, op_strong_bc_sub, op_stgdata;
  DMLabel             domain_label;

  PetscFunctionBeginUser;
  PetscCall(DMGetLabel(dm, "Face Sets", &domain_label));

  // Basis
  CeedInt height = 1;
  PetscCallCeed(ceed, CeedBasisCreateProjection(ceed_data->basis_x_sur, ceed_data->basis_q_sur, &basis_x_to_q_sur));
  // --- Get number of quadrature points for the boundaries
  CeedInt num_qpts_sur;
  PetscCallCeed(ceed, CeedBasisGetNumQuadraturePoints(ceed_data->basis_q_sur, &num_qpts_sur));

  // Setup QFunction
  PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, SetupStrongBC, SetupStrongBC_loc, &qf_setup));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_setup, "x", num_comp_x, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_setup, "dxdX", num_comp_x * dim_boundary, CEED_EVAL_GRAD));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_setup, "multiplicity", num_comp_q, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_setup, "x stored", num_comp_x, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_setup, "scale", 1, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_setup, "dXdx", dXdx_size, CEED_EVAL_NONE));

  // Setup STG Setup QFunction
  PetscCall(SetupStrongSTG_PreProcessing(ceed, problem, num_comp_x, stg_data_size, dXdx_size, &qf_stgdata));

  // Compute contribution on each boundary face
  for (CeedInt i = 0; i < bc->num_inflow; i++) {
    // -- Restrictions
    PetscCall(GetRestrictionForDomain(ceed, dm, height, domain_label, bc->inflows[i], 0, -1, -1, &elem_restr_q_sur, &elem_restr_x_sur, NULL));
    PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_q_sur, &multiplicity, NULL));
    PetscCallCeed(ceed, CeedElemRestrictionGetMultiplicity(elem_restr_q_sur, multiplicity));
    PetscCallCeed(ceed, CeedElemRestrictionGetNumElements(elem_restr_q_sur, &num_elem));
    PetscCallCeed(ceed, CeedElemRestrictionGetElementSize(elem_restr_q_sur, &elem_size));

    PetscCallCeed(ceed, CeedElemRestrictionCreateStrided(ceed, num_elem, elem_size, num_comp_x, num_elem * elem_size * num_comp_x,
                                                         CEED_STRIDES_BACKEND, &elem_restr_x_stored));
    PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_x_stored, &x_stored, NULL));

    PetscCallCeed(ceed,
                  CeedElemRestrictionCreateStrided(ceed, num_elem, elem_size, 1, num_elem * elem_size, CEED_STRIDES_BACKEND, &elem_restr_scale));
    PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_scale, &scale_stored, NULL));

    PetscCallCeed(ceed, CeedElemRestrictionCreateStrided(ceed, num_elem, elem_size, stg_data_size, num_elem * elem_size * stg_data_size,
                                                         CEED_STRIDES_BACKEND, &elem_restr_stgdata));
    PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_stgdata, &stg_data, NULL));
    PetscCallCeed(ceed, CeedElemRestrictionCreateStrided(ceed, num_elem, elem_size, dXdx_size, num_elem * elem_size * dXdx_size, CEED_STRIDES_BACKEND,
                                                         &elem_restr_dXdx));
    PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_dXdx, &dXdx, NULL));

    // -- Setup Operator
    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_setup, NULL, NULL, &op_setup));
    PetscCallCeed(ceed, CeedOperatorSetName(op_setup, "Precomputed data for strong boundary conditions"));
    PetscCallCeed(ceed, CeedOperatorSetField(op_setup, "x", elem_restr_x_sur, basis_x_to_q_sur, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_setup, "dxdX", elem_restr_x_sur, basis_x_to_q_sur, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_setup, "multiplicity", elem_restr_q_sur, CEED_BASIS_COLLOCATED, multiplicity));
    PetscCallCeed(ceed, CeedOperatorSetField(op_setup, "x stored", elem_restr_x_stored, CEED_BASIS_COLLOCATED, x_stored));
    PetscCallCeed(ceed, CeedOperatorSetField(op_setup, "scale", elem_restr_scale, CEED_BASIS_COLLOCATED, scale_stored));
    PetscCallCeed(ceed, CeedOperatorSetField(op_setup, "dXdx", elem_restr_dXdx, CEED_BASIS_COLLOCATED, dXdx));

    // -- Compute geometric factors
    PetscCallCeed(ceed, CeedOperatorApply(op_setup, ceed_data->x_coord, CEED_VECTOR_NONE, CEED_REQUEST_IMMEDIATE));

    // -- Compute STGData
    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_stgdata, NULL, NULL, &op_stgdata));
    PetscCallCeed(ceed, CeedOperatorSetField(op_stgdata, "dXdx", elem_restr_dXdx, CEED_BASIS_COLLOCATED, dXdx));
    PetscCallCeed(ceed, CeedOperatorSetField(op_stgdata, "x", elem_restr_x_stored, CEED_BASIS_COLLOCATED, x_stored));
    PetscCallCeed(ceed, CeedOperatorSetField(op_stgdata, "stg data", elem_restr_stgdata, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));

    PetscCallCeed(ceed, CeedOperatorApply(op_stgdata, CEED_VECTOR_NONE, stg_data, CEED_REQUEST_IMMEDIATE));

    // -- Setup BC QFunctions
    SetupStrongSTG_QF(ceed, problem, num_comp_x, num_comp_q, stg_data_size, dXdx_size, &qf_strongbc);
    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_strongbc, NULL, NULL, &op_strong_bc_sub));
    PetscCallCeed(ceed, CeedOperatorSetName(op_strong_bc_sub, "Strong STG"));

    PetscCallCeed(ceed, CeedOperatorSetField(op_strong_bc_sub, "dXdx", elem_restr_dXdx, CEED_BASIS_COLLOCATED, dXdx));
    PetscCallCeed(ceed, CeedOperatorSetField(op_strong_bc_sub, "x", elem_restr_x_stored, CEED_BASIS_COLLOCATED, x_stored));
    PetscCallCeed(ceed, CeedOperatorSetField(op_strong_bc_sub, "scale", elem_restr_scale, CEED_BASIS_COLLOCATED, scale_stored));
    PetscCallCeed(ceed, CeedOperatorSetField(op_strong_bc_sub, "stg data", elem_restr_stgdata, CEED_BASIS_COLLOCATED, stg_data));
    PetscCallCeed(ceed, CeedOperatorSetField(op_strong_bc_sub, "q", elem_restr_q_sur, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));

    // -- Add to composite operator
    PetscCallCeed(ceed, CeedCompositeOperatorAddSub(op_strong_bc, op_strong_bc_sub));

    PetscCallCeed(ceed, CeedVectorDestroy(&multiplicity));
    PetscCallCeed(ceed, CeedVectorDestroy(&x_stored));
    PetscCallCeed(ceed, CeedVectorDestroy(&scale_stored));
    PetscCallCeed(ceed, CeedVectorDestroy(&stg_data));
    PetscCallCeed(ceed, CeedVectorDestroy(&dXdx));
    PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_x_sur));
    PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_q_sur));
    PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_x_stored));
    PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_scale));
    PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_stgdata));
    PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_dXdx));
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_strongbc));
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_stgdata));
    PetscCallCeed(ceed, CeedOperatorDestroy(&op_strong_bc_sub));
    PetscCallCeed(ceed, CeedOperatorDestroy(&op_setup));
    PetscCallCeed(ceed, CeedOperatorDestroy(&op_stgdata));
  }

  PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(op_strong_bc, "solution time", &phys->stg_solution_time_label));

  PetscCallCeed(ceed, CeedBasisDestroy(&basis_x_to_q_sur));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_setup));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexInsertBoundaryValues_StrongBCCeed(DM dm, PetscBool insert_essential, Vec Q_loc, PetscReal time, Vec face_geom_FVM,
                                                       Vec cell_geom_FVM, Vec grad_FVM) {
  Vec  boundary_mask;
  User user;

  PetscFunctionBeginUser;
  PetscCall(DMGetApplicationContext(dm, &user));

  if (user->phys->stg_solution_time_label) {
    PetscCallCeed(user->ceed, CeedOperatorSetContextDouble(user->op_strong_bc_ctx->op, user->phys->stg_solution_time_label, &time));
  }

  // Mask Strong BC entries
  PetscCall(DMGetNamedLocalVector(dm, "boundary mask", &boundary_mask));
  PetscCall(VecPointwiseMult(Q_loc, Q_loc, boundary_mask));
  PetscCall(DMRestoreNamedLocalVector(dm, "boundary mask", &boundary_mask));

  PetscCall(ApplyAddCeedOperatorLocalToLocal(NULL, Q_loc, user->op_strong_bc_ctx));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetupStrongBC_Ceed(Ceed ceed, CeedData ceed_data, DM dm, User user, ProblemData *problem, SimpleBC bc) {
  CeedOperator op_strong_bc;

  PetscFunctionBeginUser;
  {
    Vec boundary_mask, global_vec;

    PetscCall(DMGetNamedLocalVector(dm, "boundary mask", &boundary_mask));
    PetscCall(DMGetGlobalVector(dm, &global_vec));
    PetscCall(VecZeroEntries(boundary_mask));
    PetscCall(VecSet(global_vec, 1.0));
    PetscCall(DMGlobalToLocal(dm, global_vec, INSERT_VALUES, boundary_mask));
    PetscCall(DMRestoreNamedLocalVector(dm, "boundary mask", &boundary_mask));
    PetscCall(DMRestoreGlobalVector(dm, &global_vec));
  }

  PetscCallCeed(ceed, CeedCompositeOperatorCreate(ceed, &op_strong_bc));
  {
    PetscBool use_strongstg = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-stg_strong", &use_strongstg, NULL));

    if (use_strongstg) {
      PetscCall(SetupStrongSTG_Ceed(ceed, ceed_data, dm, problem, bc, user->phys, op_strong_bc));
    }
  }

  PetscCall(OperatorApplyContextCreate(NULL, NULL, ceed, op_strong_bc, CEED_VECTOR_NONE, NULL, NULL, NULL, &user->op_strong_bc_ctx));

  PetscCall(PetscObjectComposeFunction((PetscObject)dm, "DMPlexInsertBoundaryValues_C", DMPlexInsertBoundaryValues_StrongBCCeed));
  PetscFunctionReturn(PETSC_SUCCESS);
}
