// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
/// @file
/// Functions for setting up and performing differential filtering

#include "../qfunctions//differential_filter.h"

#include <petscdmplex.h>

#include "../navierstokes.h"

// @brief Create RHS and LHS operators for differential filtering
PetscErrorCode DifferentialFilterCreateOperators(Ceed ceed, User user, CeedData ceed_data, CeedQFunctionContext diff_filter_qfctx) {
  DiffFilterData      diff_filter     = user->diff_filter;
  DM                  dm_filter       = diff_filter->dm_filter;
  CeedInt             num_comp_filter = diff_filter->num_field_components[0];
  CeedInt             num_comp_q, num_comp_qd, dim, num_qpts_1d, num_nodes_1d, num_comp_x;
  CeedElemRestriction elem_restr_filter;
  CeedBasis           basis_filter;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(user->dm, &dim));
  CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_x, &num_comp_x);
  CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_q, &num_comp_q);
  CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_qd_i, &num_comp_qd);
  CeedBasisGetNumQuadraturePoints1D(ceed_data->basis_q, &num_qpts_1d);
  CeedBasisGetNumNodes1D(ceed_data->basis_q, &num_nodes_1d);

  PetscCall(GetRestrictionForDomain(ceed, dm_filter, 0, 0, 0, 0, num_qpts_1d, 0, &elem_restr_filter, NULL, NULL));
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_filter, num_nodes_1d, num_qpts_1d, CEED_GAUSS, &basis_filter);

  {  // -- Create RHS MatopApplyContext
    CeedQFunction qf_rhs;
    CeedOperator  op_rhs;
    switch (user->phys->state_var) {
      case STATEVAR_PRIMITIVE:
        CeedQFunctionCreateInterior(ceed, 1, DifferentialFilter_RHS_Prim, DifferentialFilter_RHS_Prim_loc, &qf_rhs);
        break;
      case STATEVAR_CONSERVATIVE:
        CeedQFunctionCreateInterior(ceed, 1, DifferentialFilter_RHS_Conserv, DifferentialFilter_RHS_Conserv_loc, &qf_rhs);
        break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)user->dm), PETSC_ERR_SUP, "Differential filtering not available for chosen state variable");
    }
    if (diff_filter->do_mms_test) {
      CeedQFunctionDestroy(&qf_rhs);
      CeedQFunctionCreateInterior(ceed, 1, DifferentialFilter_MMS_RHS, DifferentialFilter_MMS_RHS_loc, &qf_rhs);
    }

    CeedQFunctionSetContext(qf_rhs, diff_filter_qfctx);
    CeedQFunctionAddInput(qf_rhs, "q", num_comp_q, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_rhs, "qdata", num_comp_qd, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qf_rhs, "x", num_comp_x, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_rhs, "v", num_comp_filter, CEED_EVAL_INTERP);

    CeedOperatorCreate(ceed, qf_rhs, NULL, NULL, &op_rhs);
    CeedOperatorSetField(op_rhs, "q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_rhs, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_COLLOCATED, ceed_data->q_data);
    CeedOperatorSetField(op_rhs, "x", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord);
    CeedOperatorSetField(op_rhs, "v", elem_restr_filter, basis_filter, CEED_VECTOR_ACTIVE);

    PetscCall(OperatorApplyContextCreate(user->dm, dm_filter, ceed, op_rhs, NULL, NULL, user->Q_loc, NULL, &diff_filter->op_rhs_ctx));

    CeedQFunctionDestroy(&qf_rhs);
    CeedOperatorDestroy(&op_rhs);
  }

  {  // Setup LHS Operator and KSP for the differential filtering solve
    CeedQFunction        qf_lhs;
    CeedOperator         op_lhs;
    OperatorApplyContext mat_ctx;
    Mat                  mat_lhs;
    CeedInt              num_comp_qd, dim, num_comp_grid_aniso;
    CeedElemRestriction  elem_restr_grid_aniso;
    CeedVector           grid_aniso_ceed;

    PetscCall(DMGetDimension(user->dm, &dim));
    CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_qd_i, &num_comp_qd);

    switch (num_comp_filter) {
      case 1:
        CeedQFunctionCreateInterior(ceed, 1, DifferentialFilter_LHS_1, DifferentialFilter_LHS_1_loc, &qf_lhs);
        break;
      case 5:
        CeedQFunctionCreateInterior(ceed, 1, DifferentialFilter_LHS_5, DifferentialFilter_LHS_5_loc, &qf_lhs);
        break;
      case 11:
        CeedQFunctionCreateInterior(ceed, 1, DifferentialFilter_LHS_11, DifferentialFilter_LHS_11_loc, &qf_lhs);
        break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)user->dm), PETSC_ERR_SUP, "Differential filtering not available for (%d) components", num_comp_filter);
    }

    // -- Get Grid anisotropy tensor
    PetscCall(GridAnisotropyTensorCalculateCollocatedVector(ceed, user, ceed_data, &elem_restr_grid_aniso, &grid_aniso_ceed, &num_comp_grid_aniso));

    CeedQFunctionSetContext(qf_lhs, diff_filter_qfctx);
    CeedQFunctionAddInput(qf_lhs, "q", num_comp_filter, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_lhs, "Grad_q", num_comp_filter * dim, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(qf_lhs, "anisotropy tensor", num_comp_grid_aniso, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qf_lhs, "x", num_comp_x, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_lhs, "qdata", num_comp_qd, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qf_lhs, "v", num_comp_filter, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_lhs, "Grad_v", num_comp_filter * dim, CEED_EVAL_GRAD);

    CeedOperatorCreate(ceed, qf_lhs, NULL, NULL, &op_lhs);
    CeedOperatorSetField(op_lhs, "q", elem_restr_filter, basis_filter, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_lhs, "Grad_q", elem_restr_filter, basis_filter, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_lhs, "anisotropy tensor", elem_restr_grid_aniso, CEED_BASIS_COLLOCATED, grid_aniso_ceed);
    CeedOperatorSetField(op_lhs, "x", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord);
    CeedOperatorSetField(op_lhs, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_COLLOCATED, ceed_data->q_data);
    CeedOperatorSetField(op_lhs, "v", elem_restr_filter, basis_filter, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_lhs, "Grad_v", elem_restr_filter, basis_filter, CEED_VECTOR_ACTIVE);

    PetscCall(OperatorApplyContextCreate(dm_filter, dm_filter, ceed, op_lhs, NULL, NULL, NULL, NULL, &mat_ctx));
    PetscCall(CreateMatShell_Ceed(mat_ctx, &mat_lhs));

    PetscCall(KSPCreate(PetscObjectComm((PetscObject)dm_filter), &diff_filter->ksp));
    PetscCall(KSPSetOptionsPrefix(diff_filter->ksp, "diff_filter_"));
    {
      PC pc;
      PetscCall(KSPGetPC(diff_filter->ksp, &pc));
      PetscCall(PCSetType(pc, PCJACOBI));
      PetscCall(PCJacobiSetType(pc, PC_JACOBI_DIAGONAL));
      PetscCall(KSPSetType(diff_filter->ksp, KSPCG));
      PetscCall(KSPSetNormType(diff_filter->ksp, KSP_NORM_NATURAL));
      PetscCall(KSPSetTolerances(diff_filter->ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    }
    PetscCall(KSPSetOperators(diff_filter->ksp, mat_lhs, mat_lhs));
    PetscCall(KSPSetFromOptions(diff_filter->ksp));

    CeedQFunctionDestroy(&qf_lhs);
    CeedOperatorDestroy(&op_lhs);
  }

  CeedElemRestrictionDestroy(&elem_restr_filter);
  CeedBasisDestroy(&basis_filter);
  PetscFunctionReturn(0);
}

// @brief Setup DM, operators, contexts, etc. for performing differential filtering
PetscErrorCode DifferentialFilterSetup(Ceed ceed, User user, CeedData ceed_data, ProblemData *problem) {
  MPI_Comm                  comm = user->comm;
  NewtonianIdealGasContext  gas;
  DifferentialFilterContext diff_filter_ctx;
  CeedQFunctionContext      diff_filter_qfctx;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&user->diff_filter));
  DiffFilterData diff_filter = user->diff_filter;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-diff_filter_mms", &diff_filter->do_mms_test, NULL));

  {  // Create DM for filtered quantities
    PetscFE      fe;
    PetscSection section;
    PetscInt     dim;

    PetscCall(DMClone(user->dm, &diff_filter->dm_filter));
    PetscCall(DMGetDimension(diff_filter->dm_filter, &dim));
    PetscCall(PetscObjectSetName((PetscObject)diff_filter->dm_filter, "Differential Filtering"));

    diff_filter->num_filtered_fields = diff_filter->do_mms_test ? 1 : 1;
    PetscCall(PetscMalloc1(diff_filter->num_filtered_fields, &diff_filter->num_field_components));

    if (diff_filter->do_mms_test) {
      diff_filter->num_field_components[0] = 1;
      PetscCall(
          PetscFECreateLagrange(PETSC_COMM_SELF, dim, diff_filter->num_field_components[0], PETSC_FALSE, user->app_ctx->degree, PETSC_DECIDE, &fe));
      PetscCall(PetscObjectSetName((PetscObject)fe, "Differential Filtering - MMS"));
      PetscCall(DMAddField(diff_filter->dm_filter, NULL, (PetscObject)fe));
      PetscCall(PetscFEDestroy(&fe));

      PetscCall(DMGetLocalSection(diff_filter->dm_filter, &section));
      PetscCall(PetscSectionSetFieldName(section, 0, ""));
      PetscCall(PetscSectionSetComponentName(section, 0, 0, "FilteredPhi"));
    } else {
      diff_filter->num_field_components[0] = DIFF_FILTER_NUM_COMPONENTS;
      PetscCall(
          PetscFECreateLagrange(PETSC_COMM_SELF, dim, diff_filter->num_field_components[0], PETSC_FALSE, user->app_ctx->degree, PETSC_DECIDE, &fe));
      PetscCall(PetscObjectSetName((PetscObject)fe, "Differential Filtering - Primitive State Variables"));
      PetscCall(DMAddField(diff_filter->dm_filter, NULL, (PetscObject)fe));
      PetscCall(PetscFEDestroy(&fe));

      PetscCall(DMGetLocalSection(diff_filter->dm_filter, &section));
      PetscCall(PetscSectionSetFieldName(section, 0, ""));
      PetscCall(PetscSectionSetComponentName(section, 0, DIFF_FILTER_PRESSURE, "FilteredPressure"));
      PetscCall(PetscSectionSetComponentName(section, 0, DIFF_FILTER_VELOCITY_X, "FilteredVelocityX"));
      PetscCall(PetscSectionSetComponentName(section, 0, DIFF_FILTER_VELOCITY_Y, "FilteredVelocityY"));
      PetscCall(PetscSectionSetComponentName(section, 0, DIFF_FILTER_VELOCITY_Z, "FilteredVelocityZ"));
      PetscCall(PetscSectionSetComponentName(section, 0, DIFF_FILTER_TEMPERATURE, "FilteredTemperature"));
      PetscCall(PetscSectionSetComponentName(section, 0, DIFF_FILTER_VELOCITY_SQUARED_XX, "FilteredVelocitySquaredXX"));
      PetscCall(PetscSectionSetComponentName(section, 0, DIFF_FILTER_VELOCITY_SQUARED_YY, "FilteredVelocitySquaredYY"));
      PetscCall(PetscSectionSetComponentName(section, 0, DIFF_FILTER_VELOCITY_SQUARED_ZZ, "FilteredVelocitySquaredZZ"));
      PetscCall(PetscSectionSetComponentName(section, 0, DIFF_FILTER_VELOCITY_SQUARED_YZ, "FilteredVelocitySquaredYZ"));
      PetscCall(PetscSectionSetComponentName(section, 0, DIFF_FILTER_VELOCITY_SQUARED_XZ, "FilteredVelocitySquaredXZ"));
      PetscCall(PetscSectionSetComponentName(section, 0, DIFF_FILTER_VELOCITY_SQUARED_XY, "FilteredVelocitySquaredXY"));
    }

    PetscCall(DMPlexSetClosurePermutationTensor(diff_filter->dm_filter, PETSC_DETERMINE, NULL));
    PetscCall(DMCreateDS(diff_filter->dm_filter));
  }

  PetscCall(PetscNew(&diff_filter_ctx));
  diff_filter_ctx->grid_based_width = false;
  for (int i = 0; i < 3; i++) diff_filter_ctx->width_scaling[i] = 1;
  diff_filter_ctx->kernel_scaling   = 0.1;
  diff_filter_ctx->damping_function = DIFF_FILTER_DAMP_NONE;
  diff_filter_ctx->friction_length  = 0;
  diff_filter_ctx->damping_constant = 25;

  PetscOptionsBegin(comm, NULL, "Differential Filtering Options", NULL);
  PetscInt narray = 3;
  PetscCall(PetscOptionsBool("-diff_filter_grid_based_width", "Use filter width based on the grid size", NULL, diff_filter_ctx->grid_based_width,
                             (PetscBool *)&diff_filter_ctx->grid_based_width, NULL));
  PetscCall(PetscOptionsRealArray("-diff_filter_width_scaling", "Anisotropic scaling of filter width tensor", NULL, diff_filter_ctx->width_scaling,
                                  &narray, NULL));
  PetscCall(PetscOptionsReal("-diff_filter_kernel_scaling", "Scaling to make differential kernel size \"equivalent\" to other filter kernels", NULL,
                             diff_filter_ctx->kernel_scaling, &diff_filter_ctx->kernel_scaling, NULL));
  PetscCall(PetscOptionsEnum("-diff_filter_wall_damping_function", "Damping function to use at the wall", NULL, DifferentialFilterDampingFunctions,
                             (PetscEnum)(diff_filter_ctx->damping_function), (PetscEnum *)&diff_filter_ctx->damping_function, NULL));
  PetscCall(PetscOptionsReal("-diff_filter_wall_damping_constant", "Contant for the wall-damping function", NULL, diff_filter_ctx->damping_constant,
                             &diff_filter_ctx->damping_constant, NULL));
  PetscCall(PetscOptionsReal("-diff_filter_friction_length", "Friction length associated with the flow, \\delta_\\nu. For wall-damping functions",
                             NULL, diff_filter_ctx->friction_length, &diff_filter_ctx->friction_length, NULL));
  PetscOptionsEnd();

  Units units = user->units;
  for (int i = 0; i < 3; i++) diff_filter_ctx->width_scaling[i] *= units->meter;
  diff_filter_ctx->kernel_scaling *= units->meter;
  diff_filter_ctx->friction_length *= units->meter;

  // -- Create QFContext
  CeedQFunctionContextGetDataRead(problem->apply_vol_ifunction.qfunction_context, CEED_MEM_HOST, &gas);
  diff_filter_ctx->gas = *gas;
  CeedQFunctionContextRestoreDataRead(problem->apply_vol_ifunction.qfunction_context, &gas);

  CeedQFunctionContextCreate(ceed, &diff_filter_qfctx);
  CeedQFunctionContextSetData(diff_filter_qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*diff_filter_ctx), diff_filter_ctx);
  CeedQFunctionContextSetDataDestroy(diff_filter_qfctx, CEED_MEM_HOST, FreeContextPetsc);

  // -- Setup Operators
  PetscCall(DifferentialFilterCreateOperators(ceed, user, ceed_data, diff_filter_qfctx));

  CeedQFunctionContextDestroy(&diff_filter_qfctx);
  PetscFunctionReturn(0);
}

// @brief Apply differential filter to the solution given by Q
PetscErrorCode DifferentialFilterApply(User user, const PetscReal solution_time, const Vec Q, Vec Filtered_Solution) {
  DiffFilterData diff_filter = user->diff_filter;

  PetscFunctionBeginUser;
  PetscCall(UpdateBoundaryValues(user, diff_filter->op_rhs_ctx->X_loc, solution_time));
  ApplyCeedOperatorGlobalToGlobal(Q, Filtered_Solution, diff_filter->op_rhs_ctx);
  PetscCall(VecViewFromOptions(Filtered_Solution, NULL, "-diff_filter_rhs_view"));

  PetscCall(KSPSolve(diff_filter->ksp, Filtered_Solution, Filtered_Solution));

  PetscFunctionReturn(0);
}

// @brief TSMonitor for just applying differential filtering to the simulation
// This runs every time step and is primarily for testing purposes
PetscErrorCode TSMonitor_DifferentialFilter(TS ts, PetscInt steps, PetscReal solution_time, Vec Q, void *ctx) {
  User           user        = (User)ctx;
  DiffFilterData diff_filter = user->diff_filter;
  Vec            Filtered_Field;

  PetscFunctionBeginUser;
  PetscCall(DMGetGlobalVector(diff_filter->dm_filter, &Filtered_Field));

  PetscCall(DifferentialFilterApply(user, solution_time, Q, Filtered_Field));
  PetscCall(VecViewFromOptions(Filtered_Field, NULL, "-diff_filter_view"));
  if (user->app_ctx->test_type == TESTTYPE_DIFF_FILTER) PetscCall(RegressionTests_NS(user->app_ctx, Filtered_Field));

  PetscCall(DMRestoreGlobalVector(diff_filter->dm_filter, &Filtered_Field));

  PetscFunctionReturn(0);
}

PetscErrorCode DifferentialFilterDataDestroy(DiffFilterData diff_filter) {
  PetscFunctionBeginUser;
  if (!diff_filter) PetscFunctionReturn(0);

  OperatorApplyContextDestroy(diff_filter->op_rhs_ctx);
  PetscCall(DMDestroy(&diff_filter->dm_filter));
  PetscCall(KSPDestroy(&diff_filter->ksp));

  PetscCall(PetscFree(diff_filter->num_field_components));
  PetscCall(PetscFree(diff_filter));

  PetscFunctionReturn(0);
}

PetscErrorCode DifferentialFilter_MMS_ICSetup(ProblemData *problem) {
  PetscFunctionBeginUser;
  problem->ics.qfunction     = DifferentialFilter_MMS_IC;
  problem->ics.qfunction_loc = DifferentialFilter_MMS_IC_loc;

  PetscFunctionReturn(0);
}
