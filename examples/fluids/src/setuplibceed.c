// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Setup libCEED for Navier-Stokes example using PETSc

#include "../navierstokes.h"

// Utility function - essential BC dofs are encoded in closure indices as -(i+1).
PetscInt Involute(PetscInt i) {
  return i >= 0 ? i : -(i+1);
}

// Utility function to create local CEED restriction
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt height,
    DMLabel domain_label, CeedInt value, CeedElemRestriction *elem_restr) {
  PetscInt num_elem, elem_size, num_dof, num_comp, *elem_restr_offsets;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexGetLocalOffsets(dm, domain_label, value, height, 0, &num_elem,
                               &elem_size, &num_comp, &num_dof, &elem_restr_offsets);
  CHKERRQ(ierr);

  CeedElemRestrictionCreate(ceed, num_elem, elem_size, num_comp,
                            1, num_dof, CEED_MEM_HOST, CEED_COPY_VALUES,
                            elem_restr_offsets, elem_restr);
  ierr = PetscFree(elem_restr_offsets); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Utility function to get Ceed Restriction for each domain
PetscErrorCode GetRestrictionForDomain(Ceed ceed, DM dm, CeedInt height,
                                       DMLabel domain_label, PetscInt value,
                                       CeedInt Q, CeedInt q_data_size,
                                       CeedElemRestriction *elem_restr_q,
                                       CeedElemRestriction *elem_restr_x,
                                       CeedElemRestriction *elem_restr_qd_i) {
  DM             dm_coord;
  CeedInt        dim, loc_num_elem;
  CeedInt        Q_dim;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  dim -= height;
  Q_dim = CeedIntPow(Q, dim);
  ierr = DMGetCoordinateDM(dm, &dm_coord); CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dm_coord, PETSC_DETERMINE, NULL);
  CHKERRQ(ierr);
  ierr = CreateRestrictionFromPlex(ceed, dm, height, domain_label, value,
                                   elem_restr_q);
  CHKERRQ(ierr);
  ierr = CreateRestrictionFromPlex(ceed, dm_coord, height, domain_label, value,
                                   elem_restr_x);
  CHKERRQ(ierr);
  CeedElemRestrictionGetNumElements(*elem_restr_q, &loc_num_elem);
  CeedElemRestrictionCreateStrided(ceed, loc_num_elem, Q_dim,
                                   q_data_size, q_data_size*loc_num_elem*Q_dim,
                                   CEED_STRIDES_BACKEND, elem_restr_qd_i);
  PetscFunctionReturn(0);
}

// Utility function to create CEED Composite Operator for the entire domain
PetscErrorCode CreateOperatorForDomain(Ceed ceed, DM dm, SimpleBC bc,
                                       CeedData ceed_data, Physics phys,
                                       CeedOperator op_apply_vol, CeedInt height,
                                       CeedInt P_sur, CeedInt Q_sur, CeedInt q_data_size_sur,
                                       CeedOperator *op_apply) {
  //CeedInt        dim;
  DMLabel        domain_label;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // Create Composite Operaters
  CeedCompositeOperatorCreate(ceed, op_apply);

  // --Apply Sub-Operator for the volume
  CeedCompositeOperatorAddSub(*op_apply, op_apply_vol);

  // -- Create Sub-Operator for in/outflow BCs
  if (phys->has_neumann || 1) {
    // --- Setup
    ierr = DMGetLabel(dm, "Face Sets", &domain_label); CHKERRQ(ierr);
    //ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

    // --- Get number of quadrature points for the boundaries
    CeedInt num_qpts_sur;
    CeedBasisGetNumQuadraturePoints(ceed_data->basis_q_sur, &num_qpts_sur);

    // --- Create Sub-Operator for inflow boundaries
    for (CeedInt i=0; i < bc->num_inflow; i++) {
      CeedVector          q_data_sur;
      CeedOperator        op_setup_sur, op_apply_inflow;
      CeedElemRestriction elem_restr_x_sur, elem_restr_q_sur, elem_restr_qd_i_sur;

      // ---- CEED Restriction
      ierr = GetRestrictionForDomain(ceed, dm, height, domain_label, bc->inflows[i],
                                     Q_sur, q_data_size_sur, &elem_restr_q_sur, &elem_restr_x_sur,
                                     &elem_restr_qd_i_sur);
      CHKERRQ(ierr);

      // ---- CEED Vector
      PetscInt loc_num_elem_sur;
      CeedElemRestrictionGetNumElements(elem_restr_q_sur, &loc_num_elem_sur);
      CeedVectorCreate(ceed, q_data_size_sur*loc_num_elem_sur*num_qpts_sur,
                       &q_data_sur);

      // ---- CEED Operator
      // ----- CEED Operator for Setup (geometric factors)
      CeedOperatorCreate(ceed, ceed_data->qf_setup_sur, NULL, NULL, &op_setup_sur);
      CeedOperatorSetField(op_setup_sur, "dx", elem_restr_x_sur,
                           ceed_data->basis_x_sur, CEED_VECTOR_ACTIVE);
      CeedOperatorSetField(op_setup_sur, "weight", CEED_ELEMRESTRICTION_NONE,
                           ceed_data->basis_x_sur, CEED_VECTOR_NONE);
      CeedOperatorSetField(op_setup_sur, "surface qdata", elem_restr_qd_i_sur,
                           CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

      // ----- CEED Operator for Physics
      CeedOperatorCreate(ceed, ceed_data->qf_apply_inflow, NULL, NULL,
                         &op_apply_inflow);
      CeedOperatorSetField(op_apply_inflow, "q", elem_restr_q_sur,
                           ceed_data->basis_q_sur, CEED_VECTOR_ACTIVE);
      CeedOperatorSetField(op_apply_inflow, "surface qdata", elem_restr_qd_i_sur,
                           CEED_BASIS_COLLOCATED, q_data_sur);
      CeedOperatorSetField(op_apply_inflow, "x", elem_restr_x_sur,
                           ceed_data->basis_x_sur, ceed_data->x_coord);
      CeedOperatorSetField(op_apply_inflow, "v", elem_restr_q_sur,
                           ceed_data->basis_q_sur, CEED_VECTOR_ACTIVE);

      // ----- Apply CEED operator for Setup
      CeedOperatorApply(op_setup_sur, ceed_data->x_coord, q_data_sur,
                        CEED_REQUEST_IMMEDIATE);

      // ----- Apply Sub-Operator for Physics
      CeedCompositeOperatorAddSub(*op_apply, op_apply_inflow);

      // ----- Cleanup
      CeedVectorDestroy(&q_data_sur);
      CeedElemRestrictionDestroy(&elem_restr_q_sur);
      CeedElemRestrictionDestroy(&elem_restr_x_sur);
      CeedElemRestrictionDestroy(&elem_restr_qd_i_sur);
      CeedOperatorDestroy(&op_setup_sur);
      CeedOperatorDestroy(&op_apply_inflow);
    }

    // --- Create Sub-Operator for outflow boundaries
    for (CeedInt i=0; i < bc->num_outflow; i++) {
      CeedVector          q_data_sur;
      CeedOperator        op_setup_sur, op_apply_outflow;
      CeedElemRestriction elem_restr_x_sur, elem_restr_q_sur, elem_restr_qd_i_sur;

      // ---- CEED Restriction
      ierr = GetRestrictionForDomain(ceed, dm, height, domain_label, bc->outflows[i],
                                     Q_sur, q_data_size_sur, &elem_restr_q_sur, &elem_restr_x_sur,
                                     &elem_restr_qd_i_sur);
      CHKERRQ(ierr);

      // ---- CEED Vector
      PetscInt loc_num_elem_sur;
      CeedElemRestrictionGetNumElements(elem_restr_q_sur, &loc_num_elem_sur);
      CeedVectorCreate(ceed, q_data_size_sur*loc_num_elem_sur*num_qpts_sur,
                       &q_data_sur);

      // ---- CEED Operator
      // ----- CEED Operator for Setup (geometric factors)
      CeedOperatorCreate(ceed, ceed_data->qf_setup_sur, NULL, NULL, &op_setup_sur);
      CeedOperatorSetField(op_setup_sur, "dx", elem_restr_x_sur,
                           ceed_data->basis_x_sur, CEED_VECTOR_ACTIVE);
      CeedOperatorSetField(op_setup_sur, "weight", CEED_ELEMRESTRICTION_NONE,
                           ceed_data->basis_x_sur, CEED_VECTOR_NONE);
      CeedOperatorSetField(op_setup_sur, "surface qdata", elem_restr_qd_i_sur,
                           CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

      // ----- CEED Operator for Physics
      CeedOperatorCreate(ceed, ceed_data->qf_apply_outflow, NULL, NULL,
                         &op_apply_outflow);
      CeedOperatorSetField(op_apply_outflow, "q", elem_restr_q_sur,
                           ceed_data->basis_q_sur, CEED_VECTOR_ACTIVE);
      CeedOperatorSetField(op_apply_outflow, "surface qdata", elem_restr_qd_i_sur,
                           CEED_BASIS_COLLOCATED, q_data_sur);
      CeedOperatorSetField(op_apply_outflow, "x", elem_restr_x_sur,
                           ceed_data->basis_x_sur, ceed_data->x_coord);
      CeedOperatorSetField(op_apply_outflow, "v", elem_restr_q_sur,
                           ceed_data->basis_q_sur, CEED_VECTOR_ACTIVE);

      // ----- Apply CEED operator for Setup
      CeedOperatorApply(op_setup_sur, ceed_data->x_coord, q_data_sur,
                        CEED_REQUEST_IMMEDIATE);

      // ----- Apply Sub-Operator for Physics
      CeedCompositeOperatorAddSub(*op_apply, op_apply_outflow);

      // ----- Cleanup
      CeedVectorDestroy(&q_data_sur);
      CeedElemRestrictionDestroy(&elem_restr_q_sur);
      CeedElemRestrictionDestroy(&elem_restr_x_sur);
      CeedElemRestrictionDestroy(&elem_restr_qd_i_sur);
      CeedOperatorDestroy(&op_setup_sur);
      CeedOperatorDestroy(&op_apply_outflow);
    }
  }

  // ----- Get Context Labels for Operator
  CeedOperatorContextGetFieldLabel(*op_apply, "solution time",
                                   &phys->solution_time_label);
  CeedOperatorContextGetFieldLabel(*op_apply, "timestep size",
                                   &phys->timestep_size_label);

  PetscFunctionReturn(0);
}

PetscErrorCode SetupLibceed(Ceed ceed, CeedData ceed_data, DM dm, User user,
                            AppCtx app_ctx, ProblemData *problem, SimpleBC bc, SetupContext setup_ctx) {
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // *****************************************************************************
  // Set up CEED objects for the interior domain (volume)
  // *****************************************************************************
  const PetscInt num_comp_q      = 5;
  const CeedInt  dim             = problem->dim,
                 num_comp_x      = problem->dim,
                 q_data_size_vol = problem->q_data_size_vol,
                 P               = app_ctx->degree + 1,
                 Q               = P + app_ctx->q_extra;

  // -----------------------------------------------------------------------------
  // CEED Bases
  // -----------------------------------------------------------------------------
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_q, P, Q, CEED_GAUSS,
                                  &ceed_data->basis_q);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_x, 2, Q, CEED_GAUSS,
                                  &ceed_data->basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_x, 2, P,
                                  CEED_GAUSS_LOBATTO, &ceed_data->basis_xc);

  // -----------------------------------------------------------------------------
  // CEED Restrictions
  // -----------------------------------------------------------------------------
  // -- Create restriction
  ierr = GetRestrictionForDomain(ceed, dm, 0, 0, 0, Q, q_data_size_vol,
                                 &ceed_data->elem_restr_q, &ceed_data->elem_restr_x,
                                 &ceed_data->elem_restr_qd_i); CHKERRQ(ierr);
  // -- Create E vectors
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &user->q_ceed, NULL);
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &user->q_dot_ceed,
                                  NULL);
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &user->g_ceed, NULL);

  // -----------------------------------------------------------------------------
  // CEED QFunctions
  // -----------------------------------------------------------------------------
  // -- Create QFunction for quadrature data
  CeedQFunctionCreateInterior(ceed, 1, problem->setup_vol, problem->setup_vol_loc,
                              &ceed_data->qf_setup_vol);
  CeedQFunctionAddInput(ceed_data->qf_setup_vol, "dx", num_comp_x*dim,
                        CEED_EVAL_GRAD);
  CeedQFunctionAddInput(ceed_data->qf_setup_vol, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(ceed_data->qf_setup_vol, "qdata", q_data_size_vol,
                         CEED_EVAL_NONE);

  // -- Create QFunction for ICs
  CeedQFunctionCreateInterior(ceed, 1, problem->ics, problem->ics_loc,
                              &ceed_data->qf_ics);
  CeedQFunctionAddInput(ceed_data->qf_ics, "x", num_comp_x, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(ceed_data->qf_ics, "q0", num_comp_q, CEED_EVAL_NONE);

  // -- Create QFunction for RHS
  if (problem->apply_vol_rhs) {
    CeedQFunctionCreateInterior(ceed, 1, problem->apply_vol_rhs,
                                problem->apply_vol_rhs_loc, &ceed_data->qf_rhs_vol);
    CeedQFunctionAddInput(ceed_data->qf_rhs_vol, "q", num_comp_q, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(ceed_data->qf_rhs_vol, "dq", num_comp_q*dim,
                          CEED_EVAL_GRAD);
    CeedQFunctionAddInput(ceed_data->qf_rhs_vol, "qdata", q_data_size_vol,
                          CEED_EVAL_NONE);
    CeedQFunctionAddInput(ceed_data->qf_rhs_vol, "x", num_comp_x, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(ceed_data->qf_rhs_vol, "v", num_comp_q,
                           CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(ceed_data->qf_rhs_vol, "dv", num_comp_q*dim,
                           CEED_EVAL_GRAD);
  }

  // -- Create QFunction for IFunction
  if (problem->apply_vol_ifunction) {
    CeedQFunctionCreateInterior(ceed, 1, problem->apply_vol_ifunction,
                                problem->apply_vol_ifunction_loc, &ceed_data->qf_ifunction_vol);
    CeedQFunctionAddInput(ceed_data->qf_ifunction_vol, "q", num_comp_q,
                          CEED_EVAL_INTERP);
    CeedQFunctionAddInput(ceed_data->qf_ifunction_vol, "dq", num_comp_q*dim,
                          CEED_EVAL_GRAD);
    CeedQFunctionAddInput(ceed_data->qf_ifunction_vol, "q dot", num_comp_q,
                          CEED_EVAL_INTERP);
    CeedQFunctionAddInput(ceed_data->qf_ifunction_vol, "qdata", q_data_size_vol,
                          CEED_EVAL_NONE);
    CeedQFunctionAddInput(ceed_data->qf_ifunction_vol, "x", num_comp_x,
                          CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(ceed_data->qf_ifunction_vol, "v", num_comp_q,
                           CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(ceed_data->qf_ifunction_vol, "dv", num_comp_q*dim,
                           CEED_EVAL_GRAD);
  }

  // ---------------------------------------------------------------------------
  // Element coordinates
  // ---------------------------------------------------------------------------
  // -- Create CEED vector
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_x, &ceed_data->x_coord,
                                  NULL);

  // -- Copy PETSc vector in CEED vector
  Vec               X_loc;
  const PetscScalar *X_loc_array;
  ierr = DMGetCoordinatesLocal(dm, &X_loc); CHKERRQ(ierr);
  ierr = VecScale(X_loc, problem->dm_scale); CHKERRQ(ierr);
  ierr = VecGetArrayRead(X_loc, &X_loc_array); CHKERRQ(ierr);
  CeedVectorSetArray(ceed_data->x_coord, CEED_MEM_HOST, CEED_COPY_VALUES,
                     (PetscScalar *)X_loc_array);
  ierr = VecRestoreArrayRead(X_loc, &X_loc_array); CHKERRQ(ierr);

  // -----------------------------------------------------------------------------
  // CEED vectors
  // -----------------------------------------------------------------------------
  // -- Create CEED vector for geometric data
  CeedInt  num_qpts_vol;
  PetscInt loc_num_elem_vol;
  CeedBasisGetNumQuadraturePoints(ceed_data->basis_q, &num_qpts_vol);
  CeedElemRestrictionGetNumElements(ceed_data->elem_restr_q, &loc_num_elem_vol);
  CeedVectorCreate(ceed, q_data_size_vol*loc_num_elem_vol*num_qpts_vol,
                   &ceed_data->q_data);

  // -----------------------------------------------------------------------------
  // CEED Operators
  // -----------------------------------------------------------------------------
  // -- Create CEED operator for quadrature data
  CeedOperatorCreate(ceed, ceed_data->qf_setup_vol, NULL, NULL,
                     &ceed_data->op_setup_vol);
  CeedOperatorSetField(ceed_data->op_setup_vol, "dx", ceed_data->elem_restr_x,
                       ceed_data->basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(ceed_data->op_setup_vol, "weight",
                       CEED_ELEMRESTRICTION_NONE, ceed_data->basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(ceed_data->op_setup_vol, "qdata",
                       ceed_data->elem_restr_qd_i, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // -- Create CEED operator for ICs
  CeedOperatorCreate(ceed, ceed_data->qf_ics, NULL, NULL, &ceed_data->op_ics);
  CeedOperatorSetField(ceed_data->op_ics, "x", ceed_data->elem_restr_x,
                       ceed_data->basis_xc, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(ceed_data->op_ics, "q0", ceed_data->elem_restr_q,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Create CEED operator for RHS
  if (ceed_data->qf_rhs_vol) {
    CeedOperator op;
    CeedOperatorCreate(ceed, ceed_data->qf_rhs_vol, NULL, NULL, &op);
    CeedOperatorSetField(op, "q", ceed_data->elem_restr_q, ceed_data->basis_q,
                         CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dq", ceed_data->elem_restr_q, ceed_data->basis_q,
                         CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "qdata", ceed_data->elem_restr_qd_i,
                         CEED_BASIS_COLLOCATED, ceed_data->q_data);
    CeedOperatorSetField(op, "x", ceed_data->elem_restr_x, ceed_data->basis_x,
                         ceed_data->x_coord);
    CeedOperatorSetField(op, "v", ceed_data->elem_restr_q, ceed_data->basis_q,
                         CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dv", ceed_data->elem_restr_q, ceed_data->basis_q,
                         CEED_VECTOR_ACTIVE);
    user->op_rhs_vol = op;
  }

  // -- CEED operator for IFunction
  if (ceed_data->qf_ifunction_vol) {
    CeedOperator op;
    CeedOperatorCreate(ceed, ceed_data->qf_ifunction_vol, NULL, NULL, &op);
    CeedOperatorSetField(op, "q", ceed_data->elem_restr_q, ceed_data->basis_q,
                         CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dq", ceed_data->elem_restr_q, ceed_data->basis_q,
                         CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "q dot", ceed_data->elem_restr_q, ceed_data->basis_q,
                         user->q_dot_ceed);
    CeedOperatorSetField(op, "qdata", ceed_data->elem_restr_qd_i,
                         CEED_BASIS_COLLOCATED, ceed_data->q_data);
    CeedOperatorSetField(op, "x", ceed_data->elem_restr_x, ceed_data->basis_x,
                         ceed_data->x_coord);
    CeedOperatorSetField(op, "v", ceed_data->elem_restr_q, ceed_data->basis_q,
                         CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dv", ceed_data->elem_restr_q, ceed_data->basis_q,
                         CEED_VECTOR_ACTIVE);
    user->op_ifunction_vol = op;
  }

  // *****************************************************************************
  // Set up CEED objects for the exterior domain (surface)
  // *****************************************************************************
  CeedInt height  = 1,
          dim_sur = dim - height,
          P_sur   = app_ctx->degree + 1,
          Q_sur   = P_sur + app_ctx->q_extra;
  const CeedInt q_data_size_sur = problem->q_data_size_sur;

  // -----------------------------------------------------------------------------
  // CEED Bases
  // -----------------------------------------------------------------------------
  CeedBasisCreateTensorH1Lagrange(ceed, dim_sur, num_comp_q, P_sur, Q_sur,
                                  CEED_GAUSS, &ceed_data->basis_q_sur);
  CeedBasisCreateTensorH1Lagrange(ceed, dim_sur, num_comp_x, 2, Q_sur, CEED_GAUSS,
                                  &ceed_data->basis_x_sur);

  // -----------------------------------------------------------------------------
  // CEED QFunctions
  // -----------------------------------------------------------------------------
  // -- Create QFunction for quadrature data
  CeedQFunctionCreateInterior(ceed, 1, problem->setup_sur, problem->setup_sur_loc,
                              &ceed_data->qf_setup_sur);
  CeedQFunctionAddInput(ceed_data->qf_setup_sur, "dx", num_comp_x*dim_sur,
                        CEED_EVAL_GRAD);
  CeedQFunctionAddInput(ceed_data->qf_setup_sur, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(ceed_data->qf_setup_sur, "surface qdata",
                         q_data_size_sur, CEED_EVAL_NONE);

  // -- Creat QFunction for inflow boundaries
  if (problem->apply_inflow) {
    CeedQFunctionCreateInterior(ceed, 1, problem->apply_inflow,
                                problem->apply_inflow_loc, &ceed_data->qf_apply_inflow);
    CeedQFunctionAddInput(ceed_data->qf_apply_inflow, "q", num_comp_q,
                          CEED_EVAL_INTERP);
    CeedQFunctionAddInput(ceed_data->qf_apply_inflow, "surface qdata",
                          q_data_size_sur, CEED_EVAL_NONE);
    CeedQFunctionAddInput(ceed_data->qf_apply_inflow, "x", num_comp_x,
                          CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(ceed_data->qf_apply_inflow, "v", num_comp_q,
                           CEED_EVAL_INTERP);
  }

  // -- Creat QFunction for outflow boundaries
  if (problem->apply_outflow) {
    CeedQFunctionCreateInterior(ceed, 1, problem->apply_outflow,
                                problem->apply_outflow_loc, &ceed_data->qf_apply_outflow);
    CeedQFunctionAddInput(ceed_data->qf_apply_outflow, "q", num_comp_q,
                          CEED_EVAL_INTERP);
    CeedQFunctionAddInput(ceed_data->qf_apply_outflow, "surface qdata",
                          q_data_size_sur, CEED_EVAL_NONE);
    CeedQFunctionAddInput(ceed_data->qf_apply_outflow, "x", num_comp_x,
                          CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(ceed_data->qf_apply_outflow, "v", num_comp_q,
                           CEED_EVAL_INTERP);
  }

  // *****************************************************************************
  // CEED Operator Apply
  // *****************************************************************************
  // -- Apply CEED Operator for the geometric data
  CeedOperatorApply(ceed_data->op_setup_vol, ceed_data->x_coord,
                    ceed_data->q_data, CEED_REQUEST_IMMEDIATE);

  // -- Set up context for QFunctions
  ierr = problem->setup_ctx(ceed, ceed_data, app_ctx, setup_ctx, user->phys);
  CHKERRQ(ierr);

  // -- Create and apply CEED Composite Operator for the entire domain
  if (!user->phys->implicit) { // RHS
    ierr = CreateOperatorForDomain(ceed, dm, bc, ceed_data, user->phys,
                                   user->op_rhs_vol, height, P_sur, Q_sur,
                                   q_data_size_sur, &user->op_rhs); CHKERRQ(ierr);
  } else { // IFunction
    ierr = CreateOperatorForDomain(ceed, dm, bc, ceed_data, user->phys,
                                   user->op_ifunction_vol, height, P_sur, Q_sur,
                                   q_data_size_sur, &user->op_ifunction); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
