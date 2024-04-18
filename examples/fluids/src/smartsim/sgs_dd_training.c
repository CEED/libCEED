// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../../qfunctions/sgs_dd_training.h"

#include <petscdmplex.h>

#include "../../include/smartsim.h"
#include "../../navierstokes.h"

typedef struct {
  CeedElemRestriction  elem_restr_grid_aniso;
  CeedVector           grid_aniso_ceed;
  CeedQFunctionContext sgs_dd_train_qfctx;
} *SGS_DD_TrainingSetupData;

static PetscErrorCode SGS_DD_TrainingSetupDataDestroy(SGS_DD_TrainingSetupData sgs_dd_train_setup_data) {
  Ceed ceed;

  PetscFunctionBeginUser;
  PetscCall(CeedElemRestrictionGetCeed(sgs_dd_train_setup_data->elem_restr_grid_aniso, &ceed));

  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&sgs_dd_train_setup_data->elem_restr_grid_aniso));
  PetscCallCeed(ceed, CeedVectorDestroy(&sgs_dd_train_setup_data->grid_aniso_ceed));
  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&sgs_dd_train_setup_data->sgs_dd_train_qfctx));
  PetscCall(PetscFree(sgs_dd_train_setup_data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Create DM for storing data-drive SGS model inputs
static PetscErrorCode SGS_DD_TrainingCreateDM(DM dm_source, DM *dm_dd_training, PetscInt degree, PetscInt q_extra, PetscInt *num_components) {
  PetscSection section;

  PetscFunctionBeginUser;
  *num_components = 12;

  PetscCall(DMClone(dm_source, dm_dd_training));
  PetscCall(PetscObjectSetName((PetscObject)*dm_dd_training, "Data-Driven SGS Training Data"));

  PetscCall(DMSetupByOrder_FEM(PETSC_TRUE, PETSC_TRUE, degree, 1, q_extra, 1, num_components, *dm_dd_training));

  PetscCall(DMGetLocalSection(*dm_dd_training, &section));
  PetscCall(PetscSectionSetFieldName(section, 0, "Data-Driven SGS Training Data"));
  PetscCall(PetscSectionSetComponentName(section, 0, 0, "SGSInput1"));
  PetscCall(PetscSectionSetComponentName(section, 0, 1, "SGSInput2"));
  PetscCall(PetscSectionSetComponentName(section, 0, 2, "SGSInput3"));
  PetscCall(PetscSectionSetComponentName(section, 0, 3, "SGSInput4"));
  PetscCall(PetscSectionSetComponentName(section, 0, 4, "SGSInput5"));
  PetscCall(PetscSectionSetComponentName(section, 0, 5, "SGSInput6"));
  PetscCall(PetscSectionSetComponentName(section, 0, 6, "FilteredSGSXX"));
  PetscCall(PetscSectionSetComponentName(section, 0, 7, "FilteredSGSYY"));
  PetscCall(PetscSectionSetComponentName(section, 0, 8, "FilteredSGSZZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 9, "FilteredSGSYZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 10, "FilteredSGSXZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 11, "FilteredSGSXY"));
  PetscFunctionReturn(PETSC_SUCCESS);
};

// @brief Create CeedOperator to calculate training data for data-drive SGS model at nodes
static PetscErrorCode SetupTrainingDataCalculation(Ceed ceed, User user, CeedData ceed_data, ProblemData *problem,
                                                   SGS_DD_TrainingSetupData sgs_dd_train_setup_data) {
  SGS_DD_TrainingData sgs_dd_train = user->sgs_dd_train;
  CeedQFunction       qf_sgs_dd_train;
  CeedOperator        op_sgs_dd_train;
  CeedInt             num_comp_grad_velo, num_comp_grid_aniso;
  CeedVector          inv_multiplicity, filtered_fields;
  CeedElemRestriction elem_restr_inv_multiplicity, elem_restr_grad_velo, elem_restr_sgs_train;
  DMLabel             domain_label = NULL;
  PetscInt            label_value = 0, height = 0, dm_field = 0;

  PetscFunctionBeginUser;
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(sgs_dd_train_setup_data->elem_restr_grid_aniso, &num_comp_grid_aniso));

  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, sgs_dd_train->dm_dd_training, domain_label, label_value, height, dm_field, &elem_restr_sgs_train));
  PetscCall(GetInverseMultiplicity(ceed, sgs_dd_train->dm_dd_training, domain_label, label_value, height, dm_field, PETSC_TRUE,
                                   &elem_restr_inv_multiplicity, &inv_multiplicity));

  CeedElemRestriction elem_restr_filtered_state;
  CeedInt             num_comp_filtered_state;
  {  // -- Setup filtered velocity gradient projection
    CeedBasis         basis_filtered_state;
    CeedOperatorField op_field;
    PetscCallCeed(ceed, CeedOperatorGetFieldByName(user->diff_filter->op_rhs_ctx->op, "v0", &op_field));
    PetscCallCeed(ceed, CeedOperatorFieldGetElemRestriction(op_field, &elem_restr_filtered_state));
    PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_filtered_state, &num_comp_filtered_state));
    PetscCallCeed(ceed, CeedOperatorFieldGetBasis(op_field, &basis_filtered_state));
    PetscCall(VelocityGradientProjectionSetup(ceed, user, ceed_data, problem, STATEVAR_PRIMITIVE, elem_restr_filtered_state, basis_filtered_state,
                                              &sgs_dd_train->filtered_grad_velo_proj));
    // Get velocity gradient information
    PetscCallCeed(ceed, CeedOperatorGetFieldByName(sgs_dd_train->filtered_grad_velo_proj->l2_rhs_ctx->op, "velocity gradient", &op_field));
    PetscCallCeed(ceed, CeedOperatorFieldGetElemRestriction(op_field, &elem_restr_grad_velo));
    PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_grad_velo, &num_comp_grad_velo));
  }

  CeedElemRestriction elem_restr_filtered_velo_prod;
  CeedInt             num_comp_filtered_velo_prod;
  {  // Get filtered velocity product information
    CeedOperatorField op_field;
    PetscCallCeed(ceed, CeedOperatorGetFieldByName(user->diff_filter->op_rhs_ctx->op, "v1", &op_field));
    PetscCallCeed(ceed, CeedOperatorFieldGetElemRestriction(op_field, &elem_restr_filtered_velo_prod));
    PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_filtered_velo_prod, &num_comp_filtered_velo_prod));
  }

  // -- Create operator for generating training data at nodes
  // Differential Filter only provides filtered primitive variables
  PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, ComputeSGS_DDAnisotropicTrainingDataNodal_Prim,
                                                  ComputeSGS_DDAnisotropicTrainingDataNodal_Prim_loc, &qf_sgs_dd_train));

  PetscCallCeed(ceed, CeedQFunctionSetContext(qf_sgs_dd_train, sgs_dd_train_setup_data->sgs_dd_train_qfctx));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_train, "q", num_comp_filtered_state, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_train, "velocity product", num_comp_filtered_velo_prod, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_train, "gradient velocity", num_comp_grad_velo, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_train, "anisotropy tensor", num_comp_grid_aniso, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_train, "inverse multiplicity", 1, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_sgs_dd_train, "training data", sgs_dd_train->num_comp_dd_inputs, CEED_EVAL_NONE));

  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_filtered_state, &filtered_fields, NULL));
  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_sgs_dd_train, NULL, NULL, &op_sgs_dd_train));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_train, "q", elem_restr_filtered_state, CEED_BASIS_NONE, filtered_fields));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_train, "velocity product", elem_restr_filtered_velo_prod, CEED_BASIS_NONE, filtered_fields));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_train, "gradient velocity", elem_restr_grad_velo, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_train, "anisotropy tensor", sgs_dd_train_setup_data->elem_restr_grid_aniso, CEED_BASIS_NONE,
                                           sgs_dd_train_setup_data->grid_aniso_ceed));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_train, "inverse multiplicity", elem_restr_inv_multiplicity, CEED_BASIS_NONE, inv_multiplicity));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_train, "training data", elem_restr_sgs_train, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  PetscCall(OperatorApplyContextCreate(sgs_dd_train->filtered_grad_velo_proj->dm, sgs_dd_train->dm_dd_training, ceed, op_sgs_dd_train, NULL, NULL,
                                       NULL, NULL, &sgs_dd_train->op_training_data_calc_ctx));

  PetscCallCeed(ceed, CeedVectorDestroy(&inv_multiplicity));
  PetscCallCeed(ceed, CeedVectorDestroy(&filtered_fields));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_inv_multiplicity));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_sgs_dd_train));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_sgs_dd_train));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SGS_DD_TrainingSetup(Ceed ceed, User user, CeedData ceed_data, ProblemData *problem) {
  SGS_DDTrainingContext    sgsdd_train_qfctx;
  SGS_DD_TrainingSetupData sgs_dd_train_setup_data;

  PetscFunctionBeginUser;
  if (!user->diff_filter) PetscCall(DifferentialFilterSetup(ceed, user, ceed_data, problem));
  if (!user->smartsim) PetscCall(SmartSimSetup(user));

  PetscCall(PetscNew(&sgsdd_train_qfctx));
  PetscCall(PetscNew(&sgs_dd_train_setup_data));
  PetscCall(PetscNew(&user->sgs_dd_train));
  SGS_DD_TrainingData sgs_dd_train = user->sgs_dd_train;

  sgs_dd_train->overwrite_training_data = PETSC_TRUE;
  sgs_dd_train->write_data_interval     = 1;
  sgs_dd_train->num_filter_widths       = sizeof(sgs_dd_train->filter_widths) / sizeof(sgs_dd_train->filter_widths[0]);
  PetscOptionsBegin(user->comm, NULL, "SGS Data-Driven Training Options", NULL);
  PetscCall(PetscOptionsInt("-sgs_train_write_data_interval", "Number of timesteps between writing data into database", NULL,
                            sgs_dd_train->write_data_interval, &sgs_dd_train->write_data_interval, NULL));
  PetscCall(PetscOptionsBool("-sgs_train_overwrite_data", "Overwrite old training data in the database", NULL, sgs_dd_train->overwrite_training_data,
                             &sgs_dd_train->overwrite_training_data, NULL));
  PetscCall(PetscOptionsRealArray("-sgs_train_filter_width_scales", "Scales of each filter width put into training database", NULL,
                                  sgs_dd_train->filter_widths, &sgs_dd_train->num_filter_widths, NULL));
  PetscOptionsEnd();

  // -- Create DM for storing training data
  PetscCall(SGS_DD_TrainingCreateDM(user->dm, &sgs_dd_train->dm_dd_training, user->app_ctx->degree, user->app_ctx->q_extra,
                                    &sgs_dd_train->num_comp_dd_inputs));

  {  // -- Create QFunction Context
    NewtonianIdealGasContext gas;
    PetscCallCeed(ceed, CeedQFunctionContextGetDataRead(problem->apply_vol_ifunction.qfunction_context, CEED_MEM_HOST, &gas));
    sgsdd_train_qfctx->gas = *gas;
    PetscCallCeed(ceed, CeedQFunctionContextRestoreDataRead(problem->apply_vol_ifunction.qfunction_context, &gas));
    PetscCallCeed(ceed, CeedQFunctionContextCreate(user->ceed, &sgs_dd_train_setup_data->sgs_dd_train_qfctx));
    PetscCallCeed(ceed, CeedQFunctionContextSetData(sgs_dd_train_setup_data->sgs_dd_train_qfctx, CEED_MEM_HOST, CEED_USE_POINTER,
                                                    sizeof(*sgsdd_train_qfctx), sgsdd_train_qfctx));
    PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(sgs_dd_train_setup_data->sgs_dd_train_qfctx, CEED_MEM_HOST, FreeContextPetsc));
  }

  {  // -- Send training data array info to SmartRedis database
    PetscMPIInt  rank, num_ranks;
    SmartSimData smartsim = user->smartsim;
    PetscCallMPI(MPI_Comm_rank(user->comm, &rank));
    PetscCallMPI(MPI_Comm_size(user->comm, &num_ranks));

    {
      PetscSection global_section;
      PetscInt     num_dofs, num_comps, local_min_max[2] = {0.}, global_min_max[2] = {0.};

      PetscCall(DMGetGlobalSection(sgs_dd_train->dm_dd_training, &global_section));
      PetscCall(DMGetGlobalVectorInfo(sgs_dd_train->dm_dd_training, &num_dofs, NULL, NULL));
      PetscCall(PetscSectionGetFieldComponents(global_section, 0, &num_comps));
      local_min_max[0] = num_dofs;
      PetscCall(PetscGlobalMinMaxInt(user->comm, local_min_max, global_min_max));

      sgs_dd_train->training_data_array_dims[0] = global_min_max[0] / num_comps;
      sgs_dd_train->training_data_array_dims[1] = num_comps;
    }

    if (rank % smartsim->collocated_database_num_ranks == 0) {
      {  // Communicate info on simulation size
        const char tensor_name[]  = "sizeInfo";
        size_t     array_info_dim = 6;
        PetscInt64 array_info[6] = {0}, num_features = 6;

        array_info[0] = sgs_dd_train->training_data_array_dims[0];
        array_info[1] = sgs_dd_train->training_data_array_dims[1];
        array_info[2] = num_features;
        array_info[3] = num_ranks;
        array_info[4] = smartsim->collocated_database_num_ranks;
        array_info[5] = rank;

        PetscCall(PetscLogEventBegin(FLUIDS_SmartRedis_Meta, 0, 0, 0, 0));
        PetscCallSmartRedis(
            put_tensor(smartsim->client, tensor_name, strlen(tensor_name), array_info, &array_info_dim, 1, SRTensorTypeInt64, SRMemLayoutContiguous));
        PetscCall(SmartRedisVerifyPutTensor(smartsim->client, tensor_name, strlen(tensor_name)));
        PetscCall(PetscLogEventEnd(FLUIDS_SmartRedis_Meta, 0, 0, 0, 0));
      }

      {  // Send array that communicates if tensors are overwritten in database
        const char tensor_name[]       = "tensor-ow";
        PetscInt64 tensor_overwrite[2] = {sgs_dd_train->overwrite_training_data};
        size_t     dim_2[1]            = {2};

        PetscCall(PetscLogEventBegin(FLUIDS_SmartRedis_Meta, 0, 0, 0, 0));
        PetscCallSmartRedis(
            put_tensor(smartsim->client, tensor_name, strlen(tensor_name), tensor_overwrite, dim_2, 1, SRTensorTypeInt64, SRMemLayoutContiguous));
        PetscCall(SmartRedisVerifyPutTensor(smartsim->client, tensor_name, strlen(tensor_name)));
        PetscCall(PetscLogEventEnd(FLUIDS_SmartRedis_Meta, 0, 0, 0, 0));
      }

      {  // Communicate number of filter widths used
        const char tensor_name[]     = "num_filter_widths";
        PetscInt64 num_filter_widths = sgs_dd_train->num_filter_widths;
        size_t     dim_2             = 1;

        PetscCall(PetscLogEventBegin(FLUIDS_SmartRedis_Meta, 0, 0, 0, 0));
        PetscCallSmartRedis(
            put_tensor(smartsim->client, tensor_name, strlen(tensor_name), &num_filter_widths, &dim_2, 1, SRTensorTypeInt64, SRMemLayoutContiguous));
        PetscCall(SmartRedisVerifyPutTensor(smartsim->client, tensor_name, strlen(tensor_name)));
        PetscCall(PetscLogEventEnd(FLUIDS_SmartRedis_Meta, 0, 0, 0, 0));
      }
    }
  }

  // -- Compute and store anisotropy tensor
  PetscCall(GridAnisotropyTensorProjectionSetupApply(ceed, user, ceed_data, &sgs_dd_train_setup_data->elem_restr_grid_aniso,
                                                     &sgs_dd_train_setup_data->grid_aniso_ceed));

  // -- Create Nodal Evaluation Operator
  PetscCall(SetupTrainingDataCalculation(ceed, user, ceed_data, problem, sgs_dd_train_setup_data));

  PetscCall(SGS_DD_TrainingSetupDataDestroy(sgs_dd_train_setup_data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSMonitor_SGS_DD_Training(TS ts, PetscInt step_num, PetscReal solution_time, Vec Q, void *ctx) {
  User                user         = (User)ctx;
  Ceed                ceed         = user->ceed;
  SGS_DD_TrainingData sgs_dd_train = user->sgs_dd_train;
  SmartSimData        smartsim     = user->smartsim;
  Vec                 TrainingData;
  PetscMPIInt         rank;

  PetscFunctionBeginUser;

  PetscCallMPI(MPI_Comm_rank(user->comm, &rank));

  if (step_num % sgs_dd_train->write_data_interval != 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMGetGlobalVector(sgs_dd_train->dm_dd_training, &TrainingData));

  for (PetscInt filter_index = 0; filter_index < sgs_dd_train->num_filter_widths; filter_index++) {
    PetscCall(PetscLogEventBegin(FLUIDS_TrainDataCompute, 0, 0, 0, 0));
    {  // -- Compute and assemble training data
      Vec          FilteredVelocityGradient, FilteredFields, FilteredFields_loc;
      PetscMemType filtered_fields_mem_type;
      CeedVector   filtered_fields;

      {  // Set filter width for the current solve
        double       filter_width_scaling[3];
        CeedOperator op_mat;
        Mat          A_mat;

        for (int j = 0; j < 3; j++) filter_width_scaling[j] = sgs_dd_train->filter_widths[filter_index];
        PetscCall(KSPGetOperators(user->diff_filter->ksp, &A_mat, NULL));
        PetscCall(MatCeedGetCeedOperators(A_mat, &op_mat, NULL));
        PetscCall(CeedOperatorSetContextDouble(op_mat, user->diff_filter->filter_width_scaling_label, filter_width_scaling));
      }

      PetscCall(DMGetGlobalVector(user->diff_filter->dm_filter, &FilteredFields));
      PetscCall(DMGetLocalVector(user->diff_filter->dm_filter, &FilteredFields_loc));

      PetscCall(DifferentialFilterApply(user, solution_time, Q, FilteredFields));
      PetscCall(DMGlobalToLocal(user->diff_filter->dm_filter, FilteredFields, INSERT_VALUES, FilteredFields_loc));

      PetscCall(DMGetGlobalVector(sgs_dd_train->filtered_grad_velo_proj->dm, &FilteredVelocityGradient));
      PetscCall(VelocityGradientProjectionApply(sgs_dd_train->filtered_grad_velo_proj, FilteredFields_loc, FilteredVelocityGradient));

      {
        CeedOperatorField op_field;

        PetscCallCeed(ceed, CeedOperatorGetFieldByName(sgs_dd_train->op_training_data_calc_ctx->op, "q", &op_field));
        PetscCallCeed(ceed, CeedOperatorFieldGetVector(op_field, &filtered_fields));
      }

      PetscCall(VecPetscToCeed(FilteredFields_loc, &filtered_fields_mem_type, filtered_fields));  // filtered_fields is an implicit input
      PetscCall(ApplyCeedOperatorGlobalToGlobal(FilteredVelocityGradient, TrainingData, sgs_dd_train->op_training_data_calc_ctx));
      PetscCall(VecCeedToPetsc(filtered_fields, filtered_fields_mem_type, FilteredFields_loc));

      PetscCall(DMRestoreGlobalVector(sgs_dd_train->filtered_grad_velo_proj->dm, &FilteredVelocityGradient));
      PetscCall(DMRestoreGlobalVector(user->diff_filter->dm_filter, &FilteredFields));
      PetscCall(DMRestoreLocalVector(user->diff_filter->dm_filter, &FilteredFields_loc));
    }
    PetscCall(PetscLogEventEnd(FLUIDS_TrainDataCompute, 0, 0, 0, 0));

    {  // -- Send training data to SmartSim
      char   array_key[PETSC_MAX_PATH_LEN];
      size_t array_key_len;

      if (sgs_dd_train->overwrite_training_data) {
        PetscCall(PetscSNPrintf(array_key, sizeof array_key, "%s.%" PetscInt_FMT, smartsim->rank_id_name, filter_index));
      } else {
        PetscCall(PetscSNPrintf(array_key, sizeof array_key, "%s.%" PetscInt_FMT "%" PetscInt_FMT, smartsim->rank_id_name, step_num, filter_index));
      }
      PetscCall(PetscStrlen(array_key, &array_key_len));

      {
        const PetscScalar *training_data;
        PetscCall(VecGetArrayRead(TrainingData, &training_data));
        PetscCall(PetscLogEventBegin(FLUIDS_SmartRedis_Train, 0, 0, 0, 0));
        PetscCallSmartRedis(put_tensor(smartsim->client, array_key, array_key_len, (void *)training_data, sgs_dd_train->training_data_array_dims, 2,
                                       SRTensorTypeDouble, SRMemLayoutContiguous));
        PetscCall(PetscLogEventEnd(FLUIDS_SmartRedis_Train, 0, 0, 0, 0));
        PetscCall(VecRestoreArrayRead(TrainingData, &training_data));
      }
    }
  }

  if (rank % smartsim->collocated_database_num_ranks == 0) {
    const char tensor_name[] = "step";
    size_t     dim_2[1]      = {2};
    PetscInt64 step_array[2] = {step_num, step_num};

    PetscCall(PetscLogEventBegin(FLUIDS_SmartRedis_Meta, 0, 0, 0, 0));
    PetscCallSmartRedis(
        put_tensor(smartsim->client, tensor_name, strlen(tensor_name), step_array, dim_2, 1, SRTensorTypeInt64, SRMemLayoutContiguous));
    PetscCall(PetscLogEventEnd(FLUIDS_SmartRedis_Meta, 0, 0, 0, 0));
  }

  PetscCall(DMRestoreGlobalVector(user->sgs_dd_train->dm_dd_training, &TrainingData));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSPostStep_SGS_DD_Training(TS ts) {
  User         user;
  const char   check_run_key[]   = "check-run";
  PetscReal    check_run[2]      = {1};
  const size_t check_run_dims[1] = {2};
  size_t       check_run_key_size;

  PetscFunctionBeginUser;
  PetscCall(PetscStrlen(check_run_key, &check_run_key_size));
  PetscCall(TSGetApplicationContext(ts, &user));
  SmartSimData smartsim = user->smartsim;

  PetscCall(PetscLogEventBegin(FLUIDS_SmartRedis_Meta, 0, 0, 0, 0));
  PetscCallSmartRedis(
      unpack_tensor(smartsim->client, check_run_key, check_run_key_size, check_run, check_run_dims, 1, SRTensorTypeDouble, SRMemLayoutContiguous));
  PetscCall(PetscLogEventEnd(FLUIDS_SmartRedis_Meta, 0, 0, 0, 0));
  if (check_run[0] == 0) {
    PetscCall(PetscPrintf(user->comm, "-- Simulation stopped by 'check-run' tensor in Redis database\n"));
    PetscCall(TSSetConvergedReason(ts, TS_CONVERGED_USER));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SGS_DD_TrainingDataDestroy(SGS_DD_TrainingData sgs_dd_train) {
  PetscFunctionBeginUser;
  if (!sgs_dd_train) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(OperatorApplyContextDestroy(sgs_dd_train->op_training_data_calc_ctx));
  PetscCall(NodalProjectionDataDestroy(sgs_dd_train->filtered_grad_velo_proj));
  PetscCall(DMDestroy(&sgs_dd_train->dm_dd_training));
  PetscCall(PetscFree(sgs_dd_train));

  PetscFunctionReturn(PETSC_SUCCESS);
}
