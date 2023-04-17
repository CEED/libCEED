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

// @brief Create CeedOperator to calculate inputs to the data-drive SGS model at nodes (for online ML training)
static PetscErrorCode SGS_DD_TrainingSetupNodalInputEvaluation(Ceed ceed, User user, CeedData ceed_data,
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

  // -- Create operator for SGS DD model nodal evaluation
  switch (user->phys->state_var) {
    case STATEVAR_PRIMITIVE:
      CeedQFunctionCreateInterior(ceed, 1, ComputeSGS_DDAnisotropicInputsNodal_Prim, ComputeSGS_DDAnisotropicInputsNodal_Prim_loc, &qf_sgs_dd_nodal);
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)user->dm), PETSC_ERR_SUP,
              "Anisotropic data-driven model inputs nodal evaluation not available for chosen state variable");
  }

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

  CeedOperatorCreate(ceed, qf_sgs_dd_nodal, NULL, NULL, &op_sgs_dd_nodal);
  CeedOperatorSetField(op_sgs_dd_nodal, "q", ceed_data->elem_restr_q, CEED_BASIS_COLLOCATED, user->q_ceed);
  CeedOperatorSetField(op_sgs_dd_nodal, "x", ceed_data->elem_restr_x, basis_x_to_q, ceed_data->x_coord);
  CeedOperatorSetField(op_sgs_dd_nodal, "gradient velocity", elem_restr_grad_velo, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_sgs_dd_nodal, "anisotropy tensor", sgs_dd_train_setup_data->elem_restr_grid_aniso, CEED_BASIS_COLLOCATED,
                       sgs_dd_train_setup_data->grid_aniso_ceed);
  CeedOperatorSetField(op_sgs_dd_nodal, "inverse multiplicity", elem_restr_inv_multiplicity, CEED_BASIS_COLLOCATED, inv_multiplicity);
  CeedOperatorSetField(op_sgs_dd_nodal, "inputs", elem_restr_sgs, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  PetscCall(OperatorApplyContextCreate(user->grad_velo_proj->dm, sgs_dd_train->dm_dd_inputs, ceed, op_sgs_dd_nodal, NULL, NULL, NULL, NULL,
                                       &sgs_dd_train->op_nodal_input_evaluation_ctx));

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

// @brief Calculate and add data-driven SGS residual to the global residual
PetscErrorCode SGS_DD_TrainingGetModelInputs(User user, const Vec Q_loc, Vec Inputs_loc) {
  SGS_DD_TrainingData sgs_dd_train = user->sgs_dd_train;
  Vec                 VelocityGradient, SGSNodalInputs_loc;
  PetscMemType        q_mem_type;

  PetscFunctionBeginUser;
  // TODO: This should be replaced by a grad_velo_proj defined for the filtered fields
  PetscCall(DMGetGlobalVector(user->grad_velo_proj->dm, &VelocityGradient));
  PetscCall(VelocityGradientProjectionApply(user->grad_velo_proj, Q_loc, VelocityGradient));

  // -- Compute Nodal SGS tensor
  PetscCall(DMGetLocalVector(sgs_dd_train->dm_dd_inputs, &SGSNodalInputs_loc));
  PetscCall(VecP2C(Q_loc, &q_mem_type, user->q_ceed));  // q_ceed is an implicit input

  PetscCall(ApplyCeedOperatorGlobalToLocal(VelocityGradient, SGSNodalInputs_loc, sgs_dd_train->op_nodal_input_evaluation_ctx));

  PetscCall(VecC2P(user->q_ceed, q_mem_type, Q_loc));

  PetscCall(DMRestoreLocalVector(sgs_dd_train->dm_dd_inputs, &SGSNodalInputs_loc));
  PetscCall(DMRestoreGlobalVector(user->grad_velo_proj->dm, &VelocityGradient));

  PetscFunctionReturn(0);
}

PetscErrorCode SGS_DD_TrainingSetup(Ceed ceed, User user, CeedData ceed_data, ProblemData *problem) {
  SGS_DDTrainingContext    sgsdd_train_ctx;
  MPI_Comm                 comm = user->comm;
  SGS_DD_TrainingSetupData sgs_dd_train_setup_data;

  PetscFunctionBeginUser;
  // if (!user->grad_velo_proj) PetscCall(VelocityGradientProjectionSetup(ceed, user, ceed_data, problem));
  if (!user->diff_filter) PetscCall(DifferentialFilterSetup(ceed, user, ceed_data, problem));

  PetscCall(PetscNew(&sgsdd_train_ctx));
  PetscCall(PetscNew(&sgs_dd_train_setup_data));

  PetscOptionsBegin(comm, NULL, "SGS Data-Driven Training Options", NULL);

  PetscOptionsEnd();

  // -- Create DM for storing SGS tensor at nodes
  PetscCall(PetscNew(&user->sgs_dd_train));
  PetscCall(SGS_DD_TrainingCreateDM(user->dm, &user->sgs_dd_train->dm_dd_inputs, user->app_ctx->degree, &user->sgs_dd_train->num_comp_dd_inputs));

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
  PetscCall(SGS_DD_TrainingSetupNodalInputEvaluation(ceed, user, ceed_data, sgs_dd_train_setup_data));

  PetscCall(SGS_DD_TrainingSetupDataDestroy(sgs_dd_train_setup_data));
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
