// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../navierstokes.h"
#include "../problems/stg_shur14.h"
#include "../qfunctions/dirichlet_boundary.h"

PetscErrorCode SetupStrongSTG_Ceed(Ceed ceed, CeedData ceed_data, DM dm, AppCtx app_ctx, ProblemData *problem, SimpleBC bc, Physics phys,
                                   CeedInt Q_sur, CeedInt q_data_size_sur, CeedOperator op_dirichlet) {
  CeedInt             num_comp_x = problem->dim, num_comp_q = 5, num_elem, elem_size, stg_data_size = 1;
  CeedVector          multiplicity, x_stored, scale_stored, q_data_sur, stg_data;
  CeedBasis           basis_x_to_q_sur;
  CeedElemRestriction elem_restr_x_sur, elem_restr_q_sur, elem_restr_x_stored, elem_restr_scale, elem_restr_qd_sur, elem_restr_stgdata;
  CeedQFunction       qf_setup, qf_strongbc, qf_stgdata;
  CeedOperator        op_setup, op_dirichlet_sub, op_setup_sur, op_stgdata;
  PetscFunctionBeginUser;

  DMLabel domain_label;
  PetscCall(DMGetLabel(dm, "Face Sets", &domain_label));

  // Basis
  CeedInt height = 1;
  PetscCall(CeedBasisCreateProjection(ceed_data->basis_x_sur, ceed_data->basis_q_sur, &basis_x_to_q_sur));

  // Setup QFunction
  CeedQFunctionCreateInterior(ceed, 1, SetupDirichletBC, SetupDirichletBC_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "x", num_comp_x, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_setup, "multiplicity", num_comp_q, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_setup, "x stored", num_comp_x, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_setup, "scale", 1, CEED_EVAL_NONE);

  // Setup STG Setup QFunction
  PetscCall(SetupStrongSTG_PreProcessing(ceed, problem, num_comp_x, stg_data_size, q_data_size_sur, &qf_stgdata));

  // Compute contribution on each boundary face
  for (CeedInt i = 0; i < bc->num_inflow; i++) {
    // -- Restrictions
    PetscCall(GetRestrictionForDomain(ceed, dm, height, domain_label, bc->inflows[i], Q_sur, q_data_size_sur, &elem_restr_q_sur, &elem_restr_x_sur,
                                      &elem_restr_qd_sur));
    CeedElemRestrictionCreateVector(elem_restr_q_sur, &multiplicity, NULL);
    CeedElemRestrictionGetMultiplicity(elem_restr_q_sur, multiplicity);
    CeedElemRestrictionGetNumElements(elem_restr_q_sur, &num_elem);
    CeedElemRestrictionGetElementSize(elem_restr_q_sur, &elem_size);
    CeedElemRestrictionCreateStrided(ceed, num_elem, elem_size, num_comp_x, num_elem * elem_size * num_comp_x, CEED_STRIDES_BACKEND,
                                     &elem_restr_x_stored);
    CeedElemRestrictionCreateVector(elem_restr_x_stored, &x_stored, NULL);

    CeedElemRestrictionCreateStrided(ceed, num_elem, elem_size, 1, num_elem * elem_size, CEED_STRIDES_BACKEND, &elem_restr_scale);
    CeedElemRestrictionCreateVector(elem_restr_scale, &scale_stored, NULL);

    CeedElemRestrictionCreateStrided(ceed, num_elem, elem_size, stg_data_size, num_elem * elem_size, CEED_STRIDES_BACKEND, &elem_restr_stgdata);
    CeedElemRestrictionCreateVector(elem_restr_stgdata, &stg_data, NULL);

    CeedVectorCreate(ceed, q_data_size_sur * num_elem * elem_size, &q_data_sur);

    // -- Setup Operator
    CeedOperatorCreate(ceed, qf_setup, NULL, NULL, &op_setup);
    CeedOperatorSetName(op_setup, "surface geometric data");
    CeedOperatorSetField(op_setup, "x", elem_restr_x_sur, basis_x_to_q_sur, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_setup, "multiplicity", elem_restr_q_sur, CEED_BASIS_COLLOCATED, multiplicity);
    CeedOperatorSetField(op_setup, "x stored", elem_restr_x_stored, CEED_BASIS_COLLOCATED, x_stored);
    CeedOperatorSetField(op_setup, "scale", elem_restr_scale, CEED_BASIS_COLLOCATED, scale_stored);

    // -- Compute geometric factors
    CeedOperatorApply(op_setup, ceed_data->x_coord, CEED_VECTOR_NONE, CEED_REQUEST_IMMEDIATE);

    // -- Compute QData for the surface
    CeedOperatorCreate(ceed, ceed_data->qf_setup_sur, NULL, NULL, &op_setup_sur);
    CeedOperatorSetField(op_setup_sur, "dx", elem_restr_x_sur, ceed_data->basis_xc_sur, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_setup_sur, "weight", CEED_ELEMRESTRICTION_NONE, ceed_data->basis_xc_sur, CEED_VECTOR_NONE);
    CeedOperatorSetField(op_setup_sur, "surface qdata", elem_restr_qd_sur, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

    CeedOperatorApply(op_setup_sur, ceed_data->x_coord, q_data_sur, CEED_REQUEST_IMMEDIATE);

    // -- Compute STGData
    CeedOperatorCreate(ceed, qf_stgdata, NULL, NULL, &op_stgdata);
    CeedOperatorSetField(op_stgdata, "surface qdata", elem_restr_qd_sur, CEED_BASIS_COLLOCATED, q_data_sur);
    CeedOperatorSetField(op_stgdata, "x", elem_restr_x_stored, CEED_BASIS_COLLOCATED, x_stored);
    CeedOperatorSetField(op_stgdata, "stg data", elem_restr_stgdata, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
    CeedOperatorSetNumQuadraturePoints(op_stgdata, elem_size);

    CeedOperatorApply(op_stgdata, NULL, stg_data, CEED_REQUEST_IMMEDIATE);

    // -- Setup BC QFunctions
    SetupStrongSTG_QF(ceed, problem, num_comp_x, num_comp_q, stg_data_size, q_data_size_sur, &qf_strongbc);
    CeedOperatorCreate(ceed, qf_strongbc, NULL, NULL, &op_dirichlet_sub);
    CeedOperatorSetName(op_dirichlet_sub, "Strong STG");

    CeedOperatorSetField(op_dirichlet_sub, "surface qdata", elem_restr_qd_sur, CEED_BASIS_COLLOCATED, q_data_sur);
    CeedOperatorSetField(op_dirichlet_sub, "x", elem_restr_x_stored, CEED_BASIS_COLLOCATED, x_stored);
    CeedOperatorSetField(op_dirichlet_sub, "scale", elem_restr_scale, CEED_BASIS_COLLOCATED, scale_stored);
    CeedOperatorSetField(op_dirichlet_sub, "stg data", elem_restr_stgdata, CEED_BASIS_COLLOCATED, stg_data);
    CeedOperatorSetField(op_dirichlet_sub, "q", elem_restr_q_sur, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
    CeedOperatorSetNumQuadraturePoints(op_dirichlet_sub, elem_size);

    // -- Add to composite operator
    CeedCompositeOperatorAddSub(op_dirichlet, op_dirichlet_sub);

    CeedVectorDestroy(&q_data_sur);
    CeedVectorDestroy(&multiplicity);
    CeedVectorDestroy(&x_stored);
    CeedVectorDestroy(&scale_stored);
    CeedVectorDestroy(&stg_data);
    CeedElemRestrictionDestroy(&elem_restr_x_sur);
    CeedElemRestrictionDestroy(&elem_restr_q_sur);
    CeedElemRestrictionDestroy(&elem_restr_qd_sur);
    CeedElemRestrictionDestroy(&elem_restr_x_stored);
    CeedElemRestrictionDestroy(&elem_restr_scale);
    CeedElemRestrictionDestroy(&elem_restr_stgdata);
    CeedQFunctionDestroy(&qf_strongbc);
    CeedQFunctionDestroy(&qf_stgdata);
    CeedOperatorDestroy(&op_setup_sur);
    CeedOperatorDestroy(&op_dirichlet_sub);
    CeedOperatorDestroy(&op_setup);
    CeedOperatorDestroy(&op_stgdata);
  }

  CeedOperatorContextGetFieldLabel(op_dirichlet, "solution time", &phys->stg_solution_time_label);

  CeedBasisDestroy(&basis_x_to_q_sur);
  CeedQFunctionDestroy(&qf_setup);

  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexInsertBoundaryValues_StrongBCCeed(DM dm, PetscBool insert_essential, Vec Q_loc, PetscReal time, Vec face_geom_FVM,
                                                       Vec cell_geom_FVM, Vec grad_FVM) {
  Vec          boundary_mask;
  User         user;
  PetscScalar *q;
  PetscMemType q_mem_type;
  PetscFunctionBeginUser;

  PetscCall(DMGetApplicationContext(dm, &user));

  if (user->phys->stg_solution_time_label) {
    CeedOperatorContextSetDouble(user->op_dirichlet, user->phys->stg_solution_time_label, &time);
  }

  // Mask Dirichlet entries
  PetscCall(DMGetNamedLocalVector(dm, "boundary mask", &boundary_mask));
  PetscCall(VecPointwiseMult(Q_loc, Q_loc, boundary_mask));
  PetscCall(DMRestoreNamedLocalVector(dm, "boundary mask", &boundary_mask));

  // Setup libCEED vector
  PetscCall(VecGetArrayAndMemType(Q_loc, &q, &q_mem_type));
  CeedVectorSetArray(user->q_ceed, MemTypeP2C(q_mem_type), CEED_USE_POINTER, q);

  // Apply libCEED operator
  CeedOperatorApplyAdd(user->op_dirichlet, CEED_VECTOR_NONE, user->q_ceed, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(user->q_ceed, MemTypeP2C(q_mem_type), NULL);
  PetscCall(VecRestoreArrayAndMemType(Q_loc, &q));

  PetscFunctionReturn(0);
}

PetscErrorCode SetupStrongBC_Ceed(Ceed ceed, CeedData ceed_data, DM dm, User user, AppCtx app_ctx, ProblemData *problem, SimpleBC bc, CeedInt Q_sur,
                                  CeedInt q_data_size_sur) {
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

  CeedCompositeOperatorCreate(ceed, &user->op_dirichlet);
  {
    PetscBool use_strongstg = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-stg_strong", &use_strongstg, NULL));

    if (use_strongstg) {
      PetscCall(SetupStrongSTG_Ceed(ceed, ceed_data, dm, app_ctx, problem, bc, user->phys, Q_sur, q_data_size_sur, user->op_dirichlet));
    }
  }

  PetscCall(PetscObjectComposeFunction((PetscObject)dm, "DMPlexInsertBoundaryValues_C", DMPlexInsertBoundaryValues_StrongBCCeed));
  PetscFunctionReturn(0);
}
