// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
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

  {  // -- Create inverse multiplicity for correcting nodal assembly
    CeedVector    multiplicity;
    CeedQFunction qf_multiplicity;
    CeedOperator  op_multiplicity;
    CeedInt       num_comp_q;

    PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_q, &num_comp_q));
    PetscCallCeed(ceed, CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &multiplicity, NULL));
    PetscCallCeed(ceed, CeedElemRestrictionGetMultiplicity(ceed_data->elem_restr_q, multiplicity));
    PetscCall(DMPlexCeedElemRestrictionCollocatedCreate(ceed, sgs_dd_train->dm_dd_training, domain_label, label_value, height, 1,
                                                        &elem_restr_inv_multiplicity));
    PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_inv_multiplicity, &inv_multiplicity, NULL));

    PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, InverseMultiplicity, InverseMultiplicity_loc, &qf_multiplicity));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_multiplicity, "multiplicity", num_comp_q, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_multiplicity, "inverse multiplicity", 1, CEED_EVAL_NONE));

    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_multiplicity, NULL, NULL, &op_multiplicity));
    PetscCallCeed(ceed, CeedOperatorSetName(op_multiplicity, "SGS DD Training Inputs - Create Multiplicity Scaling"));
    PetscCallCeed(ceed, CeedOperatorSetField(op_multiplicity, "multiplicity", ceed_data->elem_restr_q, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));
    PetscCallCeed(
        ceed, CeedOperatorSetField(op_multiplicity, "inverse multiplicity", elem_restr_inv_multiplicity, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));

    PetscCallCeed(ceed, CeedOperatorApply(op_multiplicity, multiplicity, inv_multiplicity, CEED_REQUEST_IMMEDIATE));

    PetscCallCeed(ceed, CeedVectorDestroy(&multiplicity));
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_multiplicity));
    PetscCallCeed(ceed, CeedOperatorDestroy(&op_multiplicity));
  }

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
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_train, "q", elem_restr_filtered_state, CEED_BASIS_COLLOCATED, filtered_fields));
  PetscCallCeed(ceed,
                CeedOperatorSetField(op_sgs_dd_train, "velocity product", elem_restr_filtered_velo_prod, CEED_BASIS_COLLOCATED, filtered_fields));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_train, "gradient velocity", elem_restr_grad_velo, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_train, "anisotropy tensor", sgs_dd_train_setup_data->elem_restr_grid_aniso,
                                           CEED_BASIS_COLLOCATED, sgs_dd_train_setup_data->grid_aniso_ceed));
  PetscCallCeed(ceed,
                CeedOperatorSetField(op_sgs_dd_train, "inverse multiplicity", elem_restr_inv_multiplicity, CEED_BASIS_COLLOCATED, inv_multiplicity));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_train, "training data", elem_restr_sgs_train, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));

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
  PetscOptionsBegin(user->comm, NULL, "SGS Data-Driven Training Options", NULL);
  PetscCall(PetscOptionsInt("-sgs_train_write_data_interval", "Number of timesteps between writing data into database", NULL,
                            sgs_dd_train->write_data_interval, &sgs_dd_train->write_data_interval, NULL));
  PetscCall(PetscOptionsBool("-sgs_train_overwrite_data", "Overwrite old training data in the database", NULL, sgs_dd_train->overwrite_training_data,
                             &sgs_dd_train->overwrite_training_data, NULL));
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
      PetscInt     num_dofs, num_comps;
      PetscCall(DMGetGlobalSection(sgs_dd_train->dm_dd_training, &global_section));
      PetscCall(DMGetGlobalVectorInfo(sgs_dd_train->dm_dd_training, &num_dofs, NULL, NULL));
      PetscCall(PetscSectionGetFieldComponents(global_section, 0, &num_comps));
      sgs_dd_train->training_data_array_dims[0] = num_dofs / num_comps;
      sgs_dd_train->training_data_array_dims[1] = num_comps;
    }

    if (rank % smartsim->collocated_database_num_ranks == 0) {
      size_t   array_info_dim = 6;
      PetscInt array_info[6] = {0}, num_features = 6;

      array_info[0] = sgs_dd_train->training_data_array_dims[0];
      array_info[1] = sgs_dd_train->training_data_array_dims[1];
      array_info[2] = num_features;
      array_info[3] = num_ranks;
      array_info[4] = smartsim->collocated_database_num_ranks;
      array_info[5] = rank;

      PetscCall(PetscLogEventBegin(SmartRedis_Meta, 0, 0, 0, 0));
      PetscSmartRedisCall(put_tensor(smartsim->client, "sizeInfo", 8, array_info, &array_info_dim, 1, SRTensorTypeInt64, SRMemLayoutContiguous));
      PetscCall(SmartRedisVerifyPutTensor(smartsim->client, "sizeInfo", 8));
      PetscCall(PetscLogEventEnd(SmartRedis_Meta, 0, 0, 0, 0));

      // -- Send array that communicates if tensors are overwritten in database
      PetscInt tensor_overwrite[2] = {sgs_dd_train->overwrite_training_data};
      size_t   dim_2[1]            = {2};
      PetscCall(PetscLogEventBegin(SmartRedis_Meta, 0, 0, 0, 0));
      PetscSmartRedisCall(put_tensor(smartsim->client, "tensor-ow", 9, tensor_overwrite, dim_2, 1, SRTensorTypeInt64, SRMemLayoutContiguous));
      PetscCall(SmartRedisVerifyPutTensor(smartsim->client, "tensor-ow", 9));
      PetscCall(PetscLogEventEnd(SmartRedis_Meta, 0, 0, 0, 0));
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

  PetscFunctionBeginUser;
  if (step_num % sgs_dd_train->write_data_interval != 0) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(DMGetGlobalVector(sgs_dd_train->dm_dd_training, &TrainingData));

  PetscCall(PetscLogEventBegin(TrainDataCompute, 0, 0, 0, 0));
  {  // -- Compute and assemble training data
    Vec          FilteredVelocityGradient, FilteredFields, FilteredFields_loc;
    PetscMemType filtered_fields_mem_type;
    CeedVector   filtered_fields;

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
    PetscCall(VecP2C(FilteredFields_loc, &filtered_fields_mem_type, filtered_fields));  // filtered_fields is an implicit input

    PetscCall(ApplyCeedOperatorGlobalToGlobal(FilteredVelocityGradient, TrainingData, sgs_dd_train->op_training_data_calc_ctx));

    PetscCall(VecC2P(filtered_fields, filtered_fields_mem_type, FilteredFields_loc));

    PetscCall(DMRestoreGlobalVector(sgs_dd_train->filtered_grad_velo_proj->dm, &FilteredVelocityGradient));
    PetscCall(DMRestoreGlobalVector(user->diff_filter->dm_filter, &FilteredFields));
    PetscCall(DMRestoreLocalVector(user->diff_filter->dm_filter, &FilteredFields_loc));
  }
  PetscCall(PetscLogEventEnd(TrainDataCompute, 0, 0, 0, 0));

  {  // -- Send training data to SmartSim
    char        array_key[PETSC_MAX_PATH_LEN];
    size_t      array_key_len;
    PetscMPIInt rank;

    PetscCallMPI(MPI_Comm_rank(user->comm, &rank));

    if (sgs_dd_train->overwrite_training_data) {
      PetscCall(PetscSNPrintf(array_key, sizeof array_key, "%s", smartsim->rank_id_name));
    } else {
      PetscCall(PetscSNPrintf(array_key, sizeof array_key, "%s.%" PetscInt_FMT, smartsim->rank_id_name, step_num));
    }
    PetscCall(PetscStrlen(array_key, &array_key_len));

    {
      const PetscScalar *training_data;
      PetscCall(VecGetArrayRead(TrainingData, &training_data));
      PetscCall(PetscLogEventBegin(SmartRedis_Train, 0, 0, 0, 0));
      PetscSmartRedisCall(put_tensor(smartsim->client, array_key, array_key_len, (void *)training_data, sgs_dd_train->training_data_array_dims, 2,
                                SRTensorTypeDouble, SRMemLayoutContiguous));
      PetscCall(PetscLogEventEnd(SmartRedis_Train, 0, 0, 0, 0));
      PetscCall(VecRestoreArrayRead(TrainingData, &training_data));
    }
    //PetscCall(SmartRedisVerifyPutTensor(smartsim->client, array_key, array_key_len));

    if (rank % smartsim->collocated_database_num_ranks == 0) {
      size_t   dim_2[1]      = {2};
      PetscInt step_array[2] = {step_num, step_num};
      PetscCall(PetscLogEventBegin(SmartRedis_Meta, 0, 0, 0, 0));
      PetscSmartRedisCall(put_tensor(smartsim->client, "step", 4, step_array, dim_2, 1, SRTensorTypeInt64, SRMemLayoutContiguous));
      PetscCall(PetscLogEventEnd(SmartRedis_Meta, 0, 0, 0, 0));
    }
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

  PetscCall(PetscLogEventBegin(SmartRedis_Meta, 0, 0, 0, 0));
  PetscSmartRedisCall(
      unpack_tensor(smartsim->client, check_run_key, check_run_key_size, check_run, check_run_dims, 1, SRTensorTypeDouble, SRMemLayoutContiguous));
  PetscCall(PetscLogEventEnd(SmartRedis_Meta, 0, 0, 0, 0));
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
