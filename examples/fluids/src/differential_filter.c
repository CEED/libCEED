// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
/// @file
/// Functions for setting up and performing differential filtering

#include "../qfunctions//differential_filter.h"
#include <ceed.h>

#include <petscdmplex.h>

#include "../navierstokes.h"

// @brief Create RHS and LHS operators for differential filtering
PetscErrorCode DifferentialFilterCreateOperators(Ceed ceed, User user, CeedData ceed_data, CeedQFunctionContext diff_filter_qfctx) {
  DiffFilterData diff_filter = user->diff_filter;
  DM             dm_filter   = diff_filter->dm_filter;
  CeedInt        num_comp_q, num_comp_qd, num_comp_x;
  PetscInt       dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(user->dm, &dim));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_x, &num_comp_x));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_q, &num_comp_q));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_qd_i, &num_comp_qd));

  {  // -- Create RHS MatopApplyContext
    CeedQFunction qf_rhs;
    CeedOperator  op_rhs;
    switch (user->phys->state_var) {
      case STATEVAR_PRIMITIVE:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DifferentialFilter_RHS_Prim, DifferentialFilter_RHS_Prim_loc, &qf_rhs));
        break;
      case STATEVAR_CONSERVATIVE:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DifferentialFilter_RHS_Conserv, DifferentialFilter_RHS_Conserv_loc, &qf_rhs));
        break;
      case STATEVAR_ENTROPY:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DifferentialFilter_RHS_Entropy, DifferentialFilter_RHS_Entropy_loc, &qf_rhs));
        break;
    }
    if (diff_filter->do_mms_test) {
      PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_rhs));
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DifferentialFilter_MMS_RHS, DifferentialFilter_MMS_RHS_loc, &qf_rhs));
    }

    PetscCallCeed(ceed, CeedQFunctionSetContext(qf_rhs, diff_filter_qfctx));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs, "q", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs, "qdata", num_comp_qd, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs, "x", num_comp_x, CEED_EVAL_INTERP));
    for (PetscInt i = 0; i < diff_filter->num_filtered_fields; i++) {
      char field_name[PETSC_MAX_PATH_LEN];
      PetscCall(PetscSNPrintf(field_name, PETSC_MAX_PATH_LEN, "v%" PetscInt_FMT, i));
      PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_rhs, field_name, diff_filter->num_field_components[i], CEED_EVAL_INTERP));
    }

    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_rhs, NULL, NULL, &op_rhs));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs, "q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_NONE, ceed_data->q_data));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs, "x", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord));
    for (PetscInt dm_field = 0; dm_field < diff_filter->num_filtered_fields; dm_field++) {
      char                field_name[PETSC_MAX_PATH_LEN];
      CeedElemRestriction elem_restr_filter;
      CeedBasis           basis_filter;
      DMLabel             domain_label = NULL;
      PetscInt            label_value = 0, height = 0;
      PetscCall(DMPlexCeedElemRestrictionCreate(ceed, dm_filter, domain_label, label_value, height, dm_field, &elem_restr_filter));
      PetscCall(CreateBasisFromPlex(ceed, dm_filter, domain_label, label_value, height, dm_field, &basis_filter));

      PetscCall(PetscSNPrintf(field_name, PETSC_MAX_PATH_LEN, "v%" PetscInt_FMT, dm_field));
      PetscCallCeed(ceed, CeedOperatorSetField(op_rhs, field_name, elem_restr_filter, basis_filter, CEED_VECTOR_ACTIVE));

      PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_filter));
      PetscCallCeed(ceed, CeedBasisDestroy(&basis_filter));
    }

    PetscCall(OperatorApplyContextCreate(user->dm, dm_filter, ceed, op_rhs, NULL, NULL, user->Q_loc, NULL, &diff_filter->op_rhs_ctx));

    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_rhs));
    PetscCallCeed(ceed, CeedOperatorDestroy(&op_rhs));
  }

  {  // Setup LHS Operator and KSP for the differential filtering solve
    CeedOperator        op_lhs;
    Mat                 mat_lhs;
    CeedInt             num_comp_qd;
    PetscInt            dim, num_comp_grid_aniso;
    CeedElemRestriction elem_restr_grid_aniso;
    CeedVector          grid_aniso_ceed;

    PetscCall(DMGetDimension(user->dm, &dim));
    PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_qd_i, &num_comp_qd));

    // -- Get Grid anisotropy tensor
    PetscCall(GridAnisotropyTensorCalculateCollocatedVector(ceed, user, ceed_data, &elem_restr_grid_aniso, &grid_aniso_ceed, &num_comp_grid_aniso));

    PetscCallCeed(ceed, CeedCompositeOperatorCreate(ceed, &op_lhs));
    for (PetscInt i = 0; i < diff_filter->num_filtered_fields; i++) {
      CeedQFunction       qf_lhs;
      PetscInt            num_comp_filter = diff_filter->num_field_components[i];
      CeedOperator        op_lhs_sub;
      CeedElemRestriction elem_restr_filter;
      CeedBasis           basis_filter;

      switch (num_comp_filter) {
        case 1:
          PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DifferentialFilter_LHS_1, DifferentialFilter_LHS_1_loc, &qf_lhs));
          break;
        case 5:
          PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DifferentialFilter_LHS_5, DifferentialFilter_LHS_5_loc, &qf_lhs));
          break;
        case 6:
          PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DifferentialFilter_LHS_6, DifferentialFilter_LHS_6_loc, &qf_lhs));
          break;
        case 11:
          PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DifferentialFilter_LHS_11, DifferentialFilter_LHS_11_loc, &qf_lhs));
          break;
        default:
          SETERRQ(PetscObjectComm((PetscObject)user->dm), PETSC_ERR_SUP, "Differential filtering not available for (%" PetscInt_FMT ") components",
                  num_comp_filter);
      }

      PetscCallCeed(ceed, CeedQFunctionSetContext(qf_lhs, diff_filter_qfctx));
      PetscCallCeed(ceed, CeedQFunctionSetUserFlopsEstimate(qf_lhs, 0));
      PetscCallCeed(ceed, CeedQFunctionAddInput(qf_lhs, "q", num_comp_filter, CEED_EVAL_INTERP));
      PetscCallCeed(ceed, CeedQFunctionAddInput(qf_lhs, "Grad_q", num_comp_filter * dim, CEED_EVAL_GRAD));
      PetscCallCeed(ceed, CeedQFunctionAddInput(qf_lhs, "anisotropy tensor", num_comp_grid_aniso, CEED_EVAL_NONE));
      PetscCallCeed(ceed, CeedQFunctionAddInput(qf_lhs, "x", num_comp_x, CEED_EVAL_INTERP));
      PetscCallCeed(ceed, CeedQFunctionAddInput(qf_lhs, "qdata", num_comp_qd, CEED_EVAL_NONE));
      PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_lhs, "v", num_comp_filter, CEED_EVAL_INTERP));
      PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_lhs, "Grad_v", num_comp_filter * dim, CEED_EVAL_GRAD));

      {
        CeedOperatorField op_field;
        char              field_name[PETSC_MAX_PATH_LEN];
        PetscCall(PetscSNPrintf(field_name, PETSC_MAX_PATH_LEN, "v%" PetscInt_FMT, i));
        PetscCallCeed(ceed, CeedOperatorGetFieldByName(diff_filter->op_rhs_ctx->op, field_name, &op_field));
        PetscCallCeed(ceed, CeedOperatorFieldGetData(op_field, NULL, &elem_restr_filter, &basis_filter, NULL));
      }

      PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_lhs, NULL, NULL, &op_lhs_sub));
      PetscCallCeed(ceed, CeedOperatorSetField(op_lhs_sub, "q", elem_restr_filter, basis_filter, CEED_VECTOR_ACTIVE));
      PetscCallCeed(ceed, CeedOperatorSetField(op_lhs_sub, "Grad_q", elem_restr_filter, basis_filter, CEED_VECTOR_ACTIVE));
      PetscCallCeed(ceed, CeedOperatorSetField(op_lhs_sub, "anisotropy tensor", elem_restr_grid_aniso, CEED_BASIS_NONE, grid_aniso_ceed));
      PetscCallCeed(ceed, CeedOperatorSetField(op_lhs_sub, "x", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord));
      PetscCallCeed(ceed, CeedOperatorSetField(op_lhs_sub, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_NONE, ceed_data->q_data));
      PetscCallCeed(ceed, CeedOperatorSetField(op_lhs_sub, "v", elem_restr_filter, basis_filter, CEED_VECTOR_ACTIVE));
      PetscCallCeed(ceed, CeedOperatorSetField(op_lhs_sub, "Grad_v", elem_restr_filter, basis_filter, CEED_VECTOR_ACTIVE));

      PetscCallCeed(ceed, CeedCompositeOperatorAddSub(op_lhs, op_lhs_sub));
      PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_filter));
      PetscCallCeed(ceed, CeedBasisDestroy(&basis_filter));
      PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_lhs));
      PetscCallCeed(ceed, CeedOperatorDestroy(&op_lhs_sub));
    }
    PetscCallCeed(ceed, CeedVectorDestroy(&grid_aniso_ceed));
    PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_grid_aniso));

    PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(op_lhs, "filter width scaling", &diff_filter->filter_width_scaling_label));
    PetscCall(MatCreateCeed(dm_filter, dm_filter, op_lhs, NULL, &mat_lhs));

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
    PetscCall(KSPSetFromOptions_WithMatCeed(diff_filter->ksp, mat_lhs));

    PetscCall(MatDestroy(&mat_lhs));
    PetscCallCeed(ceed, CeedOperatorDestroy(&op_lhs));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Setup DM, operators, contexts, etc. for performing differential filtering
PetscErrorCode DifferentialFilterSetup(Ceed ceed, User user, CeedData ceed_data, ProblemData problem) {
  MPI_Comm                  comm = user->comm;
  NewtonianIdealGasContext  gas;
  DifferentialFilterContext diff_filter_ctx;
  CeedQFunctionContext      diff_filter_qfctx;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&user->diff_filter));
  DiffFilterData diff_filter = user->diff_filter;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-diff_filter_mms", &diff_filter->do_mms_test, NULL));

  {  // Create DM for filtered quantities
    PetscSection section;

    PetscCall(DMClone(user->dm, &diff_filter->dm_filter));
    PetscCall(PetscObjectSetName((PetscObject)diff_filter->dm_filter, "Differential Filtering"));

    diff_filter->num_filtered_fields = diff_filter->do_mms_test ? 1 : 2;
    PetscCall(PetscMalloc1(diff_filter->num_filtered_fields, &diff_filter->num_field_components));

    if (diff_filter->do_mms_test) {
      PetscInt field_components;
      diff_filter->num_field_components[0] = field_components = 1;
      PetscCall(DMSetupByOrder_FEM(PETSC_TRUE, PETSC_TRUE, user->app_ctx->degree, 1, user->app_ctx->q_extra, diff_filter->num_filtered_fields,
                                   &field_components, diff_filter->dm_filter));

      PetscCall(DMGetLocalSection(diff_filter->dm_filter, &section));
      PetscCall(PetscSectionSetFieldName(section, 0, ""));
      PetscCall(PetscSectionSetComponentName(section, 0, 0, "FilteredPhi"));
    } else {
      PetscInt field_components[2];
      diff_filter->num_field_components[0] = field_components[0] = DIFF_FILTER_STATE_NUM;
      diff_filter->num_field_components[1] = field_components[1] = DIFF_FILTER_VELOCITY_SQUARED_NUM;
      PetscCall(DMSetupByOrder_FEM(PETSC_TRUE, PETSC_TRUE, user->app_ctx->degree, 1, user->app_ctx->q_extra, diff_filter->num_filtered_fields,
                                   field_components, diff_filter->dm_filter));

      diff_filter->field_prim_state = 0;
      diff_filter->field_velo_prod  = 1;
      PetscCall(DMGetLocalSection(diff_filter->dm_filter, &section));
      PetscCall(PetscSectionSetFieldName(section, diff_filter->field_prim_state, "Filtered Primitive State Variables"));
      PetscCall(PetscSectionSetComponentName(section, 0, DIFF_FILTER_PRESSURE, "FilteredPressure"));
      PetscCall(PetscSectionSetComponentName(section, 0, DIFF_FILTER_VELOCITY_X, "FilteredVelocityX"));
      PetscCall(PetscSectionSetComponentName(section, 0, DIFF_FILTER_VELOCITY_Y, "FilteredVelocityY"));
      PetscCall(PetscSectionSetComponentName(section, 0, DIFF_FILTER_VELOCITY_Z, "FilteredVelocityZ"));
      PetscCall(PetscSectionSetComponentName(section, 0, DIFF_FILTER_TEMPERATURE, "FilteredTemperature"));
      PetscCall(PetscSectionSetFieldName(section, diff_filter->field_velo_prod, "Filtered Velocity Products"));
      PetscCall(PetscSectionSetComponentName(section, 1, DIFF_FILTER_VELOCITY_SQUARED_XX, "FilteredVelocitySquaredXX"));
      PetscCall(PetscSectionSetComponentName(section, 1, DIFF_FILTER_VELOCITY_SQUARED_YY, "FilteredVelocitySquaredYY"));
      PetscCall(PetscSectionSetComponentName(section, 1, DIFF_FILTER_VELOCITY_SQUARED_ZZ, "FilteredVelocitySquaredZZ"));
      PetscCall(PetscSectionSetComponentName(section, 1, DIFF_FILTER_VELOCITY_SQUARED_YZ, "FilteredVelocitySquaredYZ"));
      PetscCall(PetscSectionSetComponentName(section, 1, DIFF_FILTER_VELOCITY_SQUARED_XZ, "FilteredVelocitySquaredXZ"));
      PetscCall(PetscSectionSetComponentName(section, 1, DIFF_FILTER_VELOCITY_SQUARED_XY, "FilteredVelocitySquaredXY"));
    }
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
  PetscCallCeed(ceed, CeedQFunctionContextGetDataRead(problem->apply_vol_ifunction.qfunction_context, CEED_MEM_HOST, &gas));
  diff_filter_ctx->gas = *gas;
  PetscCallCeed(ceed, CeedQFunctionContextRestoreDataRead(problem->apply_vol_ifunction.qfunction_context, &gas));

  PetscCallCeed(ceed, CeedQFunctionContextCreate(ceed, &diff_filter_qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetData(diff_filter_qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*diff_filter_ctx), diff_filter_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(diff_filter_qfctx, CEED_MEM_HOST, FreeContextPetsc));
  PetscCallCeed(ceed, CeedQFunctionContextRegisterDouble(
                          diff_filter_qfctx, "filter width scaling", offsetof(struct DifferentialFilterContext_, width_scaling),
                          sizeof(diff_filter_ctx->width_scaling) / sizeof(diff_filter_ctx->width_scaling[0]), "Filter width scaling"));

  // -- Setup Operators
  PetscCall(DifferentialFilterCreateOperators(ceed, user, ceed_data, diff_filter_qfctx));

  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&diff_filter_qfctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Apply differential filter to the solution given by Q
PetscErrorCode DifferentialFilterApply(User user, const PetscReal solution_time, const Vec Q, Vec Filtered_Solution) {
  DiffFilterData   diff_filter = user->diff_filter;
  PetscObjectState X_loc_state;
  Vec              RHS;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(FLUIDS_DifferentialFilter, Q, Filtered_Solution, 0, 0));
  PetscCall(DMGetNamedGlobalVector(diff_filter->dm_filter, "RHS", &RHS));
  PetscCall(UpdateBoundaryValues(user, diff_filter->op_rhs_ctx->X_loc, solution_time));
  PetscCall(VecGetState(diff_filter->op_rhs_ctx->X_loc, &X_loc_state));
  if (X_loc_state != diff_filter->X_loc_state) {
    PetscCall(ApplyCeedOperatorGlobalToGlobal(Q, RHS, diff_filter->op_rhs_ctx));
    PetscCall(VecGetState(diff_filter->op_rhs_ctx->X_loc, &X_loc_state));
    diff_filter->X_loc_state = X_loc_state;
  }
  PetscCall(VecViewFromOptions(RHS, NULL, "-diff_filter_rhs_view"));

  PetscCall(KSPSolve(diff_filter->ksp, RHS, Filtered_Solution));
  PetscCall(DMRestoreNamedGlobalVector(diff_filter->dm_filter, "RHS", &RHS));
  PetscCall(PetscLogEventEnd(FLUIDS_DifferentialFilter, Q, Filtered_Solution, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  if (user->app_ctx->test_type == TESTTYPE_DIFF_FILTER) PetscCall(RegressionTest(user->app_ctx, Filtered_Field));

  PetscCall(DMRestoreGlobalVector(diff_filter->dm_filter, &Filtered_Field));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DifferentialFilterDataDestroy(DiffFilterData diff_filter) {
  PetscFunctionBeginUser;
  if (!diff_filter) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(OperatorApplyContextDestroy(diff_filter->op_rhs_ctx));
  PetscCall(DMDestroy(&diff_filter->dm_filter));
  PetscCall(KSPDestroy(&diff_filter->ksp));

  PetscCall(PetscFree(diff_filter->num_field_components));
  PetscCall(PetscFree(diff_filter));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DifferentialFilterMmsICSetup(ProblemData problem) {
  PetscFunctionBeginUser;
  problem->ics.qfunction     = DifferentialFilter_MMS_IC;
  problem->ics.qfunction_loc = DifferentialFilter_MMS_IC_loc;
  PetscFunctionReturn(PETSC_SUCCESS);
}
