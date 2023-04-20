// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../qfunctions/sgs_dd_training.h"

#include <petscdmplex.h>

#include "../navierstokes.h"

typedef struct {
  CeedElemRestriction  elem_restr_grid_aniso, elem_restr_dd_inputs;
  CeedVector           grid_aniso_ceed;
  CeedQFunctionContext sgs_dd_train_qfctx;
} *SGS_DD_TrainingSetupData;

static PetscErrorCode SGS_DD_TrainingSetupDataDestroy(SGS_DD_TrainingSetupData sgs_dd_train_setup_data) {
  PetscFunctionBeginUser;
  CeedElemRestrictionDestroy(&sgs_dd_train_setup_data->elem_restr_grid_aniso);
  CeedElemRestrictionDestroy(&sgs_dd_train_setup_data->elem_restr_dd_inputs);
  CeedVectorDestroy(&sgs_dd_train_setup_data->grid_aniso_ceed);
  CeedQFunctionContextDestroy(&sgs_dd_train_setup_data->sgs_dd_train_qfctx);

  PetscCall(PetscFree(sgs_dd_train_setup_data));
  PetscFunctionReturn(0);
}

// @brief Create DM for storing data-drive SGS model inputs
PetscErrorCode SGS_DD_TrainingCreateDM(DM dm_source, DM *dm_dd_inputs, PetscInt degree, PetscInt *num_components) {
  PetscFE  fe;
  PetscInt dim;

  PetscFunctionBeginUser;
  *num_components = 6;

  PetscCall(DMClone(dm_source, dm_dd_inputs));
  PetscCall(DMGetDimension(*dm_dd_inputs, &dim));
  PetscCall(PetscObjectSetName((PetscObject)*dm_dd_inputs, "Data-Driven SGS Model Inputs"));

  PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, dim, *num_components, PETSC_FALSE, degree, PETSC_DECIDE, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "Data-Driven SGS Model Inputs"));
  PetscCall(DMAddField(*dm_dd_inputs, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(*dm_dd_inputs));
  PetscCall(DMPlexSetClosurePermutationTensor(*dm_dd_inputs, PETSC_DETERMINE, NULL));

  PetscCall(PetscFEDestroy(&fe));

  PetscFunctionReturn(0);
};

// @brief Create IS's for VecISCopy to copy training data to unified array
PetscErrorCode SGS_DD_TrainingCreateIS(User user) {
  SGS_DD_TrainingData sgs_dd_train         = user->sgs_dd_train;
  const PetscInt      num_training_sources = 2;
  PetscSection        local_sections[num_training_sources];          // sections to get the local Vec info from
  PetscInt            field_ids[num_training_sources];               // fields of the section that contains the training data subset
  PetscInt            num_comps[num_training_sources];               // number of components of the field
  PetscInt            local_storage_sizes[num_training_sources];     // size of the local Vecs containing training data subset
  PetscInt            training_array_offsets[num_training_sources];  // offset in the training data array for the training data subset
  PetscInt            pStart, pEnd, num_comps_training_data = 0;
  IS                  IS_vec_copies[num_training_sources];

  PetscFunctionBeginUser;
  // TODO: Possibly use PetscSectionCreateGlobalSection to create global section with local offsets and including constraints.
  // This gives negative sizes and offsets to points not owned by this process.
  PetscCall(DMGetLocalSection(sgs_dd_train->dm_dd_inputs, &local_sections[0]));
  field_ids[0] = 0;
  PetscCall(DMGetLocalSection(user->diff_filter->dm_filter, &local_sections[1]));
  field_ids[1] = user->diff_filter->field_velo_prod;

  // -- Get local Vec information and verify compatability
  PetscInt tot_num_nodes[num_training_sources];
  for (PetscInt i = 0; i < num_training_sources; i++) {
    PetscCall(PetscSectionGetFieldComponents(local_sections[i], field_ids[i], &num_comps[i]));
    PetscCall(PetscSectionGetStorageSize(local_sections[i], &local_storage_sizes[i]));
    num_comps_training_data += num_comps[i];
    if (i == 0) training_array_offsets[i] = 0;
    else training_array_offsets[i] = training_array_offsets[i - 1] + num_comps[i - 1];

    // Verify section properties/assumptions
    PetscInt num_comps_section = 0, num_fields_section, num_comps_field;
    PetscCall(PetscSectionGetNumFields(local_sections[i], &num_fields_section));
    for (PetscInt j = 0; j < num_fields_section; j++) {
      PetscCall(PetscSectionGetFieldComponents(local_sections[i], j, &num_comps_field));
      num_comps_section += num_comps_field;
    }
    PetscCheck(local_storage_sizes[i] % num_comps_section == 0, PETSC_COMM_SELF, -1,
               "The %dth local section size (%" PetscInt_FMT ") is not evenly divided by the number of components (%" PetscInt_FMT ").", i,
               local_storage_sizes[i], num_comps[i]);
    tot_num_nodes[i] = local_storage_sizes[i] / num_comps_section;
    for (PetscInt j = 0; j < i; j++)
      PetscCheck(tot_num_nodes[i] == tot_num_nodes[j], PETSC_COMM_SELF, -1,
                 "Total number of local nodes for the %dth (%" PetscInt_FMT ") and %dth (%" PetscInt_FMT ") sections are not equal", i,
                 tot_num_nodes[i], j, tot_num_nodes[j]);
  }
  sgs_dd_train->training_data_array_dims[0] = tot_num_nodes[0];
  sgs_dd_train->training_data_array_dims[1] = num_comps_training_data;

  // -- Create index arrays
  for (PetscInt i = 0; i < num_training_sources; i++) {
    PetscInt *index;
    PetscCall(PetscMalloc1(local_storage_sizes[i], &index));
    for (PetscInt j = 0; j < local_storage_sizes[i]; j++) index[j] = -1;

    PetscInt offset, num_dofs, node_id = 0;
    PetscCall(PetscSectionGetChart(local_sections[i], &pStart, &pEnd));
    for (PetscInt p = pStart; p < pEnd; p++) {
      PetscCall(PetscSectionGetFieldDof(local_sections[i], p, field_ids[i], &num_dofs));
      PetscCall(PetscSectionGetFieldOffset(local_sections[i], p, field_ids[i], &offset));
      PetscInt num_nodes = num_dofs / num_comps[i];
      for (PetscInt node = 0; node < num_nodes; node++) {
        for (PetscInt comp = 0; comp < num_comps[i]; comp++) {
          index[offset + node * num_comps[i] + comp] = node_id * num_comps_training_data + training_array_offsets[i] + comp;
        }
        node_id++;
      }
    }
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, local_storage_sizes[i], index, PETSC_OWN_POINTER, &IS_vec_copies[i]));
  }
  sgs_dd_train->is_dd_inputs         = IS_vec_copies[0];
  sgs_dd_train->is_velocity_products = IS_vec_copies[1];
  PetscCall(ISViewFromOptions(IS_vec_copies[1], NULL, "-sgs_training_is_view"));

  {  // -- Verify that copying vecs using the resulting IS works correctly
    Vec      TrainingData, FilteredVec_loc, DDModelInputs_loc;
    PetscInt training_data_array_size = sgs_dd_train->training_data_array_dims[0] * sgs_dd_train->training_data_array_dims[1];

    PetscCall(VecCreate(PETSC_COMM_SELF, &TrainingData));
    PetscCall(VecSetType(TrainingData, DMReturnVecType(user->diff_filter->dm_filter)));
    PetscCall(VecSetSizes(TrainingData, training_data_array_size, training_data_array_size));
    PetscCall(VecSet(TrainingData, 1));

    PetscCall(DMGetLocalVector(user->diff_filter->dm_filter, &FilteredVec_loc));
    PetscCall(DMGetLocalVector(sgs_dd_train->dm_dd_inputs, &DDModelInputs_loc));

    PetscCall(VecZeroEntries(FilteredVec_loc));
    PetscCall(VecZeroEntries(DDModelInputs_loc));

    PetscCall(VecISCopy(TrainingData, sgs_dd_train->is_dd_inputs, SCATTER_FORWARD, DDModelInputs_loc));
    PetscCall(VecISCopy(TrainingData, sgs_dd_train->is_velocity_products, SCATTER_FORWARD, FilteredVec_loc));

    PetscScalar norm;
    PetscCall(VecNorm(TrainingData, NORM_MAX, &norm));
    PetscCheck(norm == 0.0, PETSC_COMM_SELF, -1, "TrainingData Vec was not completely overwritten in testing VecISCopy. Norm max returned: %.16e",
               norm);

    PetscCall(DMRestoreLocalVector(user->diff_filter->dm_filter, &FilteredVec_loc));
    PetscCall(DMRestoreLocalVector(sgs_dd_train->dm_dd_inputs, &DDModelInputs_loc));
    PetscCall(VecDestroy(&TrainingData));
  }

  PetscFunctionReturn(0);
}

// @brief Create CeedOperator to calculate inputs to the data-drive SGS model at nodes (for online ML training)
static PetscErrorCode SGS_DD_TrainingSetupNodalInputEvaluation(Ceed ceed, User user, CeedData ceed_data, ProblemData *problem,
                                                               SGS_DD_TrainingSetupData sgs_dd_train_setup_data) {
  SGS_DD_TrainingData sgs_dd_train = user->sgs_dd_train;
  CeedQFunction       qf_multiplicity, qf_sgs_dd_nodal;
  CeedOperator        op_multiplicity, op_sgs_dd_nodal;
  CeedInt             num_elem, elem_size, num_comp_q, dim, num_qpts_1d, num_comp_grad_velo, num_comp_x, num_comp_grid_aniso;
  CeedVector          multiplicity, inv_multiplicity;
  CeedElemRestriction elem_restr_inv_multiplicity, elem_restr_grad_velo, elem_restr_sgs;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(user->dm, &dim));
  CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_x, &num_comp_x);
  CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_q, &num_comp_q);
  CeedElemRestrictionGetNumComponents(sgs_dd_train_setup_data->elem_restr_grid_aniso, &num_comp_grid_aniso);
  CeedElemRestrictionGetNumElements(ceed_data->elem_restr_q, &num_elem);
  CeedElemRestrictionGetElementSize(ceed_data->elem_restr_q, &elem_size);
  CeedBasisGetNumQuadraturePoints1D(ceed_data->basis_q, &num_qpts_1d);

  {  // Get velocity gradient information
    CeedOperatorField op_field;
    CeedOperatorGetFieldByName(user->grad_velo_proj->l2_rhs_ctx->op, "velocity gradient", &op_field);
    CeedOperatorFieldGetElemRestriction(op_field, &elem_restr_grad_velo);
    CeedElemRestrictionGetNumComponents(elem_restr_grad_velo, &num_comp_grad_velo);
  }
  PetscCall(GetRestrictionForDomain(ceed, sgs_dd_train->dm_dd_inputs, 0, 0, 0, 0, num_qpts_1d, 0, &elem_restr_sgs, NULL, NULL));

  // -- Create inverse multiplicity for correcting nodal assembly
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

  // -- Create operator for SGS DD model nodal evaluation
  // Differential Filter only provides filtered primitive variables
  CeedQFunctionCreateInterior(ceed, 1, ComputeSGS_DDAnisotropicInputsNodal_Prim, ComputeSGS_DDAnisotropicInputsNodal_Prim_loc, &qf_sgs_dd_nodal);

  // Mesh/geometry order and solution basis order may differ, therefore must interpolate
  CeedBasis basis_x_to_q;
  PetscCall(CeedBasisCreateProjection(ceed_data->basis_x, ceed_data->basis_q, &basis_x_to_q));

  CeedQFunctionSetContext(qf_sgs_dd_nodal, sgs_dd_train_setup_data->sgs_dd_train_qfctx);
  CeedQFunctionAddInput(qf_sgs_dd_nodal, "q", num_comp_q, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_sgs_dd_nodal, "x", num_comp_x, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_sgs_dd_nodal, "gradient velocity", num_comp_grad_velo, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_sgs_dd_nodal, "anisotropy tensor", num_comp_grid_aniso, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_sgs_dd_nodal, "inverse multiplicity", 1, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_sgs_dd_nodal, "inputs", sgs_dd_train->num_comp_dd_inputs, CEED_EVAL_NONE);

  CeedVector filtered_state;
  CeedElemRestrictionCreateVector(elem_restr_filtered_state, &filtered_state, NULL);
  CeedOperatorCreate(ceed, qf_sgs_dd_nodal, NULL, NULL, &op_sgs_dd_nodal);
  CeedOperatorSetField(op_sgs_dd_nodal, "q", elem_restr_filtered_state, CEED_BASIS_COLLOCATED, filtered_state);
  CeedOperatorSetField(op_sgs_dd_nodal, "x", ceed_data->elem_restr_x, basis_x_to_q, ceed_data->x_coord);
  CeedOperatorSetField(op_sgs_dd_nodal, "gradient velocity", elem_restr_grad_velo, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_sgs_dd_nodal, "anisotropy tensor", sgs_dd_train_setup_data->elem_restr_grid_aniso, CEED_BASIS_COLLOCATED,
                       sgs_dd_train_setup_data->grid_aniso_ceed);
  CeedOperatorSetField(op_sgs_dd_nodal, "inverse multiplicity", elem_restr_inv_multiplicity, CEED_BASIS_COLLOCATED, inv_multiplicity);
  CeedOperatorSetField(op_sgs_dd_nodal, "inputs", elem_restr_sgs, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  PetscCall(OperatorApplyContextCreate(sgs_dd_train->filtered_grad_velo_proj->dm, sgs_dd_train->dm_dd_inputs, ceed, op_sgs_dd_nodal, NULL, NULL, NULL,
                                       NULL, &sgs_dd_train->op_nodal_input_evaluation_ctx));

  sgs_dd_train_setup_data->elem_restr_dd_inputs = elem_restr_sgs;

  CeedVectorDestroy(&multiplicity);
  CeedVectorDestroy(&inv_multiplicity);
  CeedBasisDestroy(&basis_x_to_q);
  CeedElemRestrictionDestroy(&elem_restr_inv_multiplicity);
  CeedQFunctionDestroy(&qf_multiplicity);
  CeedQFunctionDestroy(&qf_sgs_dd_nodal);
  CeedOperatorDestroy(&op_multiplicity);
  CeedOperatorDestroy(&op_sgs_dd_nodal);
  PetscFunctionReturn(0);
}

PetscErrorCode SGS_DD_TrainingSetup(Ceed ceed, User user, CeedData ceed_data, ProblemData *problem) {
  SGS_DDTrainingContext    sgsdd_train_ctx;
  SGS_DD_TrainingSetupData sgs_dd_train_setup_data;

  PetscFunctionBeginUser;
  if (!user->diff_filter) PetscCall(DifferentialFilterSetup(ceed, user, ceed_data, problem));
  if (!user->smartsim) PetscCall(SmartSimSetup(user));

  PetscCall(PetscNew(&sgsdd_train_ctx));
  PetscCall(PetscNew(&sgs_dd_train_setup_data));

  PetscOptionsBegin(user->comm, NULL, "SGS Data-Driven Training Options", NULL);

  PetscOptionsEnd();

  // -- Create DM for storing training data
  PetscCall(PetscNew(&user->sgs_dd_train));
  PetscCall(SGS_DD_TrainingCreateDM(user->dm, &user->sgs_dd_train->dm_dd_inputs, user->app_ctx->degree, &user->sgs_dd_train->num_comp_dd_inputs));
  PetscCall(SGS_DD_TrainingCreateIS(user));

  {  // -- Create QFunction Context
    NewtonianIdealGasContext gas;
    CeedQFunctionContextGetDataRead(problem->apply_vol_ifunction.qfunction_context, CEED_MEM_HOST, &gas);
    sgsdd_train_ctx->gas = *gas;
    CeedQFunctionContextRestoreDataRead(problem->apply_vol_ifunction.qfunction_context, &gas);
    CeedQFunctionContextCreate(user->ceed, &sgs_dd_train_setup_data->sgs_dd_train_qfctx);
    CeedQFunctionContextSetData(sgs_dd_train_setup_data->sgs_dd_train_qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(sgsdd_train_ctx),
                                sgsdd_train_ctx);
    CeedQFunctionContextSetDataDestroy(sgs_dd_train_setup_data->sgs_dd_train_qfctx, CEED_MEM_HOST, FreeContextPetsc);
  }

  // -- Compute and store anisotropy tensor
  PetscCall(GridAnisotropyTensorProjectionSetupApply(ceed, user, ceed_data, &sgs_dd_train_setup_data->elem_restr_grid_aniso,
                                                     &sgs_dd_train_setup_data->grid_aniso_ceed));

  // -- Create Nodal Evaluation Operator
  PetscCall(SGS_DD_TrainingSetupNodalInputEvaluation(ceed, user, ceed_data, problem, sgs_dd_train_setup_data));

  PetscCall(SGS_DD_TrainingSetupDataDestroy(sgs_dd_train_setup_data));
  PetscFunctionReturn(0);
}

// @brief Calculate and add data-driven SGS residual to the global residual
PetscErrorCode SGS_DD_TrainingGetModelInputs(User user, Vec FilteredState_loc, Vec Inputs_loc) {
  SGS_DD_TrainingData sgs_dd_train = user->sgs_dd_train;
  Vec                 FilteredVelocityGradient, SGSNodalInputs_loc;
  PetscMemType        filtered_state_mem_type;
  CeedVector          filtered_state;

  PetscFunctionBeginUser;
  PetscCall(DMGetGlobalVector(sgs_dd_train->filtered_grad_velo_proj->dm, &FilteredVelocityGradient));
  PetscCall(VelocityGradientProjectionApply(sgs_dd_train->filtered_grad_velo_proj, FilteredState_loc, FilteredVelocityGradient));

  // -- Compute Nodal SGS tensor
  PetscCall(DMGetLocalVector(sgs_dd_train->dm_dd_inputs, &SGSNodalInputs_loc));
  {
    CeedOperatorField op_field;
    CeedOperatorGetFieldByName(sgs_dd_train->op_nodal_input_evaluation_ctx->op, "q", &op_field);
    CeedOperatorFieldGetVector(op_field, &filtered_state);
  }
  PetscCall(VecP2C(FilteredState_loc, &filtered_state_mem_type, filtered_state));  // q_ceed is an implicit input

  PetscCall(ApplyCeedOperatorGlobalToLocal(FilteredVelocityGradient, SGSNodalInputs_loc, sgs_dd_train->op_nodal_input_evaluation_ctx));

  PetscCall(VecC2P(filtered_state, filtered_state_mem_type, FilteredState_loc));

  PetscCall(DMRestoreLocalVector(sgs_dd_train->dm_dd_inputs, &SGSNodalInputs_loc));
  PetscCall(DMRestoreGlobalVector(sgs_dd_train->filtered_grad_velo_proj->dm, &FilteredVelocityGradient));

  PetscFunctionReturn(0);
}

PetscErrorCode TSMonitor_SGS_DD_Training(TS ts, PetscInt step_num, PetscReal solution_time, Vec Q, void *ctx) {
  User                user                     = (User)ctx;
  SGS_DD_TrainingData sgs_dd_train             = user->sgs_dd_train;
  PetscInt            training_data_array_size = sgs_dd_train->training_data_array_dims[0] * sgs_dd_train->training_data_array_dims[1];
  Vec                 FilteredFields, FilteredFields_loc, DDModelInputs;

  PetscFunctionBeginUser;
  PetscCall(DMGetGlobalVector(user->diff_filter->dm_filter, &FilteredFields));
  PetscCall(DMGetGlobalVector(user->sgs_dd_train->dm_dd_inputs, &DDModelInputs));

  PetscCall(DifferentialFilterApply(user, solution_time, Q, FilteredFields));
  PetscCall(DMGetLocalVector(user->diff_filter->dm_filter, &FilteredFields_loc));
  PetscCall(DMGlobalToLocal(user->diff_filter->dm_filter, FilteredFields, INSERT_VALUES, FilteredFields_loc));
  PetscCall(SGS_DD_TrainingGetModelInputs(user, FilteredFields_loc, DDModelInputs));

  {  // -- Send training data to SmartSim
    Vec TrainingData, DDModelInputs_loc;

    PetscCall(VecCreate(PETSC_COMM_SELF, &TrainingData));
    PetscCall(VecSetType(TrainingData, DMReturnVecType(user->diff_filter->dm_filter)));
    PetscCall(VecSetSizes(TrainingData, training_data_array_size, training_data_array_size));

    PetscCall(DMGetLocalVector(sgs_dd_train->dm_dd_inputs, &DDModelInputs_loc));
    PetscCall(DMGlobalToLocal(sgs_dd_train->dm_dd_inputs, DDModelInputs, INSERT_VALUES, DDModelInputs_loc));

    PetscCall(VecISCopy(TrainingData, sgs_dd_train->is_dd_inputs, SCATTER_FORWARD, DDModelInputs_loc));
    PetscCall(VecISCopy(TrainingData, sgs_dd_train->is_velocity_products, SCATTER_FORWARD, FilteredFields_loc));

    // Send Data (ie. put tensor)

    PetscCall(DMRestoreLocalVector(sgs_dd_train->dm_dd_inputs, &DDModelInputs_loc));
    PetscCall(VecDestroy(&TrainingData));
  }

  PetscCall(DMRestoreGlobalVector(user->diff_filter->dm_filter, &FilteredFields));
  PetscCall(DMRestoreGlobalVector(user->sgs_dd_train->dm_dd_inputs, &DDModelInputs));
  PetscFunctionReturn(0);
}

PetscErrorCode SGS_DD_TrainingDataDestroy(SGS_DD_TrainingData sgs_dd_train) {
  PetscFunctionBeginUser;
  if (!sgs_dd_train) PetscFunctionReturn(0);

  PetscCall(OperatorApplyContextDestroy(sgs_dd_train->op_nodal_input_evaluation_ctx));
  PetscCall(DMDestroy(&sgs_dd_train->dm_dd_inputs));
  PetscCall(PetscFree(sgs_dd_train));

  PetscFunctionReturn(0);
}
