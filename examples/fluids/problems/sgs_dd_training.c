// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../qfunctions/sgs_dd_training.h"

#include <petscdmplex.h>

#include "../include/smartsim.h"
#include "../navierstokes.h"

typedef struct {
  CeedElemRestriction  elem_restr_grid_aniso;
  CeedVector           grid_aniso_ceed;
  CeedQFunctionContext sgs_dd_train_qfctx;
} *SGS_DD_TrainingSetupData;

static PetscErrorCode SGS_DD_TrainingSetupDataDestroy(SGS_DD_TrainingSetupData sgs_dd_train_setup_data) {
  PetscFunctionBeginUser;
  CeedElemRestrictionDestroy(&sgs_dd_train_setup_data->elem_restr_grid_aniso);
  CeedVectorDestroy(&sgs_dd_train_setup_data->grid_aniso_ceed);
  CeedQFunctionContextDestroy(&sgs_dd_train_setup_data->sgs_dd_train_qfctx);

  PetscCall(PetscFree(sgs_dd_train_setup_data));
  PetscFunctionReturn(0);
}

// @brief Create DM for storing data-drive SGS model inputs
static PetscErrorCode SGS_DD_TrainingCreateDM(DM dm_source, DM *dm_dd_training, PetscInt degree, PetscInt *num_components) {
  PetscFE      fe;
  PetscInt     dim;
  PetscSection section;

  PetscFunctionBeginUser;
  *num_components = 12;

  PetscCall(DMClone(dm_source, dm_dd_training));
  PetscCall(DMGetDimension(*dm_dd_training, &dim));
  PetscCall(PetscObjectSetName((PetscObject)*dm_dd_training, "Data-Driven SGS Model Inputs"));

  PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, dim, *num_components, PETSC_FALSE, degree, PETSC_DECIDE, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "Data-Driven SGS Training Data"));
  PetscCall(DMAddField(*dm_dd_training, NULL, (PetscObject)fe));
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
  PetscCall(DMCreateDS(*dm_dd_training));
  PetscCall(DMPlexSetClosurePermutationTensor(*dm_dd_training, PETSC_DETERMINE, NULL));

  PetscCall(PetscFEDestroy(&fe));

  PetscFunctionReturn(0);
};

// @brief Create CeedOperator to calculate training data for data-drive SGS model at nodes
static PetscErrorCode SetupTrainingDataCalculation(Ceed ceed, User user, CeedData ceed_data, ProblemData *problem,
                                                   SGS_DD_TrainingSetupData sgs_dd_train_setup_data) {
  SGS_DD_TrainingData sgs_dd_train = user->sgs_dd_train;
  CeedQFunction       qf_multiplicity, qf_sgs_dd_train;
  CeedOperator        op_multiplicity, op_sgs_dd_train;
  CeedInt             num_elem, elem_size, num_comp_q, num_qpts_1d, num_comp_grad_velo, num_comp_x, num_comp_grid_aniso;
  CeedVector          inv_multiplicity;
  CeedElemRestriction elem_restr_inv_multiplicity, elem_restr_grad_velo, elem_restr_sgs_train;

  PetscFunctionBeginUser;
  CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_x, &num_comp_x);
  CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_q, &num_comp_q);
  CeedElemRestrictionGetNumComponents(sgs_dd_train_setup_data->elem_restr_grid_aniso, &num_comp_grid_aniso);
  CeedElemRestrictionGetNumElements(ceed_data->elem_restr_q, &num_elem);
  CeedElemRestrictionGetElementSize(ceed_data->elem_restr_q, &elem_size);
  CeedBasisGetNumQuadraturePoints1D(ceed_data->basis_q, &num_qpts_1d);

  PetscCall(GetRestrictionForDomain(ceed, sgs_dd_train->dm_dd_training, 0, 0, 0, 0, num_qpts_1d, 0, &elem_restr_sgs_train, NULL, NULL));

  {  // -- Create inverse multiplicity for correcting nodal assembly
    CeedVector multiplicity;
    CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &multiplicity, NULL);
    CeedElemRestrictionGetMultiplicity(ceed_data->elem_restr_q, multiplicity);
    CeedElemRestrictionCreateStrided(ceed, num_elem, elem_size, 1, num_elem * elem_size, CEED_STRIDES_BACKEND, &elem_restr_inv_multiplicity);
    CeedElemRestrictionCreateVector(elem_restr_inv_multiplicity, &inv_multiplicity, NULL);

    CeedQFunctionCreateInterior(ceed, 1, InverseMultiplicity, InverseMultiplicity_loc, &qf_multiplicity);
    CeedQFunctionAddInput(qf_multiplicity, "multiplicity", num_comp_q, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qf_multiplicity, "inverse multiplicity", 1, CEED_EVAL_NONE);

    CeedOperatorCreate(ceed, qf_multiplicity, NULL, NULL, &op_multiplicity);
    CeedOperatorSetName(op_multiplicity, "SGS DD Training Inputs - Create Multiplicity Scaling");
    CeedOperatorSetField(op_multiplicity, "multiplicity", ceed_data->elem_restr_q, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_multiplicity, "inverse multiplicity", elem_restr_inv_multiplicity, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
    CeedOperatorSetNumQuadraturePoints(op_multiplicity, elem_size);

    CeedOperatorApply(op_multiplicity, multiplicity, inv_multiplicity, CEED_REQUEST_IMMEDIATE);

    CeedVectorDestroy(&multiplicity);
  }

  CeedElemRestriction elem_restr_filtered_state;
  CeedInt             num_comp_filtered_state;
  {  // -- Setup filtered velocity gradient projection
    CeedBasis         basis_filtered_state;
    CeedOperatorField op_field;
    CeedOperatorGetFieldByName(user->diff_filter->op_rhs_ctx->op, "v0", &op_field);
    CeedOperatorFieldGetElemRestriction(op_field, &elem_restr_filtered_state);
    CeedElemRestrictionGetNumComponents(elem_restr_filtered_state, &num_comp_filtered_state);
    CeedOperatorFieldGetBasis(op_field, &basis_filtered_state);
    PetscCall(VelocityGradientProjectionSetup(ceed, user, ceed_data, problem, STATEVAR_PRIMITIVE, elem_restr_filtered_state, basis_filtered_state,
                                              &sgs_dd_train->filtered_grad_velo_proj));
    // Get velocity gradient information
    CeedOperatorGetFieldByName(sgs_dd_train->filtered_grad_velo_proj->l2_rhs_ctx->op, "velocity gradient", &op_field);
    CeedOperatorFieldGetElemRestriction(op_field, &elem_restr_grad_velo);
    CeedElemRestrictionGetNumComponents(elem_restr_grad_velo, &num_comp_grad_velo);
  }

  // -- Create operator for generating training data at nodes
  // Differential Filter only provides filtered primitive variables
  CeedQFunctionCreateInterior(ceed, 1, ComputeSGS_DDAnisotropicTrainingDataNodal_Prim, ComputeSGS_DDAnisotropicTrainingDataNodal_Prim_loc,
                              &qf_sgs_dd_train);

  // Mesh/geometry order and solution basis order may differ, therefore must interpolate
  CeedBasis basis_x_to_q;
  PetscCall(CeedBasisCreateProjection(ceed_data->basis_x, ceed_data->basis_q, &basis_x_to_q));

  CeedElemRestriction elem_restr_filtered_velo_prod;
  CeedInt             num_comp_filtered_velo_prod;
  {  // Get filtered velocity product information
    CeedOperatorField op_field;
    CeedOperatorGetFieldByName(user->diff_filter->op_rhs_ctx->op, "v1", &op_field);
    CeedOperatorFieldGetElemRestriction(op_field, &elem_restr_filtered_velo_prod);
    CeedElemRestrictionGetNumComponents(elem_restr_filtered_velo_prod, &num_comp_filtered_velo_prod);
  }

  CeedQFunctionSetContext(qf_sgs_dd_train, sgs_dd_train_setup_data->sgs_dd_train_qfctx);
  CeedQFunctionAddInput(qf_sgs_dd_train, "q", num_comp_filtered_state, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_sgs_dd_train, "velocity product", num_comp_filtered_velo_prod, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_sgs_dd_train, "x", num_comp_x, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_sgs_dd_train, "gradient velocity", num_comp_grad_velo, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_sgs_dd_train, "anisotropy tensor", num_comp_grid_aniso, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_sgs_dd_train, "inverse multiplicity", 1, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_sgs_dd_train, "training data", sgs_dd_train->num_comp_dd_inputs, CEED_EVAL_NONE);

  CeedVector filtered_fields;
  CeedElemRestrictionCreateVector(elem_restr_filtered_state, &filtered_fields, NULL);
  CeedOperatorCreate(ceed, qf_sgs_dd_train, NULL, NULL, &op_sgs_dd_train);
  CeedOperatorSetField(op_sgs_dd_train, "q", elem_restr_filtered_state, CEED_BASIS_COLLOCATED, filtered_fields);
  CeedOperatorSetField(op_sgs_dd_train, "velocity product", elem_restr_filtered_velo_prod, CEED_BASIS_COLLOCATED, filtered_fields);
  CeedOperatorSetField(op_sgs_dd_train, "x", ceed_data->elem_restr_x, basis_x_to_q, ceed_data->x_coord);
  CeedOperatorSetField(op_sgs_dd_train, "gradient velocity", elem_restr_grad_velo, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_sgs_dd_train, "anisotropy tensor", sgs_dd_train_setup_data->elem_restr_grid_aniso, CEED_BASIS_COLLOCATED,
                       sgs_dd_train_setup_data->grid_aniso_ceed);
  CeedOperatorSetField(op_sgs_dd_train, "inverse multiplicity", elem_restr_inv_multiplicity, CEED_BASIS_COLLOCATED, inv_multiplicity);
  CeedOperatorSetField(op_sgs_dd_train, "training data", elem_restr_sgs_train, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  PetscCall(OperatorApplyContextCreate(sgs_dd_train->filtered_grad_velo_proj->dm, sgs_dd_train->dm_dd_training, ceed, op_sgs_dd_train, NULL, NULL,
                                       NULL, NULL, &sgs_dd_train->op_training_data_calc_ctx));

  CeedVectorDestroy(&inv_multiplicity);
  CeedVectorDestroy(&filtered_fields);
  CeedBasisDestroy(&basis_x_to_q);
  CeedElemRestrictionDestroy(&elem_restr_inv_multiplicity);
  CeedQFunctionDestroy(&qf_multiplicity);
  CeedQFunctionDestroy(&qf_sgs_dd_train);
  CeedOperatorDestroy(&op_multiplicity);
  CeedOperatorDestroy(&op_sgs_dd_train);
  PetscFunctionReturn(0);
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
  PetscCall(SGS_DD_TrainingCreateDM(user->dm, &sgs_dd_train->dm_dd_training, user->app_ctx->degree, &sgs_dd_train->num_comp_dd_inputs));

  {  // -- Create QFunction Context
    NewtonianIdealGasContext gas;
    CeedQFunctionContextGetDataRead(problem->apply_vol_ifunction.qfunction_context, CEED_MEM_HOST, &gas);
    sgsdd_train_qfctx->gas = *gas;
    CeedQFunctionContextRestoreDataRead(problem->apply_vol_ifunction.qfunction_context, &gas);
    CeedQFunctionContextCreate(user->ceed, &sgs_dd_train_setup_data->sgs_dd_train_qfctx);
    CeedQFunctionContextSetData(sgs_dd_train_setup_data->sgs_dd_train_qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(sgsdd_train_qfctx),
                                sgsdd_train_qfctx);
    CeedQFunctionContextSetDataDestroy(sgs_dd_train_setup_data->sgs_dd_train_qfctx, CEED_MEM_HOST, FreeContextPetsc);
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

      SmartRedisCall(put_tensor(smartsim->client, "sizeInfo", 8, array_info, &array_info_dim, 1, SRTensorTypeInt32, SRMemLayoutContiguous));
      PetscCall(SmartRedisVerifyPutTensor(smartsim->client, "sizeInfo", 8));

      // -- Send array that communicates if tensors are overwritten in database
      PetscInt tensor_overwrite[2] = {sgs_dd_train->overwrite_training_data};
      size_t   dim_2[1]            = {2};
      SmartRedisCall(put_tensor(smartsim->client, "tensor-ow", 9, tensor_overwrite, dim_2, 1, SRTensorTypeInt32, SRMemLayoutContiguous));
      PetscCall(SmartRedisVerifyPutTensor(smartsim->client, "tensor-ow", 9));
    }
  }

  // -- Compute and store anisotropy tensor
  PetscCall(GridAnisotropyTensorProjectionSetupApply(ceed, user, ceed_data, &sgs_dd_train_setup_data->elem_restr_grid_aniso,
                                                     &sgs_dd_train_setup_data->grid_aniso_ceed));

  // -- Create Nodal Evaluation Operator
  PetscCall(SetupTrainingDataCalculation(ceed, user, ceed_data, problem, sgs_dd_train_setup_data));

  PetscCall(SGS_DD_TrainingSetupDataDestroy(sgs_dd_train_setup_data));
  PetscFunctionReturn(0);
}

PetscErrorCode TSMonitor_SGS_DD_Training(TS ts, PetscInt step_num, PetscReal solution_time, Vec Q, void *ctx) {
  User                user         = (User)ctx;
  SGS_DD_TrainingData sgs_dd_train = user->sgs_dd_train;
  SmartSimData        smartsim     = user->smartsim;
  Vec                 TrainingData;

  PetscFunctionBeginUser;
  if (step_num % sgs_dd_train->write_data_interval != 0) PetscFunctionReturn(0);
  PetscCall(DMGetGlobalVector(sgs_dd_train->dm_dd_training, &TrainingData));

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

    // Compute Training Data
    {
      CeedOperatorField op_field;
      CeedOperatorGetFieldByName(sgs_dd_train->op_training_data_calc_ctx->op, "q", &op_field);
      CeedOperatorFieldGetVector(op_field, &filtered_fields);
    }
    PetscCall(VecP2C(FilteredFields_loc, &filtered_fields_mem_type, filtered_fields));  // filtered_fields is an implicit input

    PetscCall(ApplyCeedOperatorGlobalToGlobal(FilteredVelocityGradient, TrainingData, sgs_dd_train->op_training_data_calc_ctx));

    PetscCall(VecC2P(filtered_fields, filtered_fields_mem_type, FilteredFields_loc));

    PetscCall(DMRestoreGlobalVector(sgs_dd_train->filtered_grad_velo_proj->dm, &FilteredVelocityGradient));
    PetscCall(DMRestoreGlobalVector(user->diff_filter->dm_filter, &FilteredFields));
  }

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
    printf("put_array with key '%s'\n", array_key);

    {
      const PetscScalar *training_data;
      PetscCall(VecGetArrayRead(TrainingData, &training_data));
      SmartRedisCall(put_tensor(smartsim->client, array_key, array_key_len, (void *)training_data, sgs_dd_train->training_data_array_dims, 2,
                                SRTensorTypeDouble, SRMemLayoutContiguous));
      PetscCall(VecRestoreArrayRead(TrainingData, &training_data));
    }
    PetscCall(SmartRedisVerifyPutTensor(smartsim->client, array_key, array_key_len));

    if (rank % smartsim->collocated_database_num_ranks == 0) {
      size_t   dim_2[1]      = {2};
      PetscInt step_array[2] = {step_num, step_num};
      SmartRedisCall(put_tensor(smartsim->client, "step", 4, step_array, dim_2, 1, SRTensorTypeInt32, SRMemLayoutContiguous));
    }
  }

  PetscCall(DMRestoreGlobalVector(user->sgs_dd_train->dm_dd_training, &TrainingData));
  PetscFunctionReturn(0);
}

PetscErrorCode SGS_DD_TrainingDataDestroy(SGS_DD_TrainingData sgs_dd_train) {
  PetscFunctionBeginUser;
  if (!sgs_dd_train) PetscFunctionReturn(0);

  PetscCall(OperatorApplyContextDestroy(sgs_dd_train->op_training_data_calc_ctx));
  PetscCall(DMDestroy(&sgs_dd_train->dm_dd_training));
  PetscCall(PetscFree(sgs_dd_train));

  PetscFunctionReturn(0);
}
