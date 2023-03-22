// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../qfunctions/sgs_dd_model.h"

#include <petscdmplex.h>

#include "../navierstokes.h"

typedef struct {
  CeedElemRestriction  elem_restr_grid_aniso, elem_restr_sgs;
  CeedVector           grid_aniso_ceed;
  CeedQFunctionContext sgsdd_qfctx;
} *SGS_DD_ModelSetupData;

PetscErrorCode SGS_DD_ModelSetupDataDestroy(SGS_DD_ModelSetupData sgs_dd_setup_data) {
  PetscFunctionBeginUser;
  CeedElemRestrictionDestroy(&sgs_dd_setup_data->elem_restr_grid_aniso);
  CeedElemRestrictionDestroy(&sgs_dd_setup_data->elem_restr_sgs);
  CeedVectorDestroy(&sgs_dd_setup_data->grid_aniso_ceed);
  CeedQFunctionContextDestroy(&sgs_dd_setup_data->sgsdd_qfctx);

  PetscCall(PetscFree(sgs_dd_setup_data));
  PetscFunctionReturn(0);
}

// @brief Create DM for storing subgrid stress at nodes
PetscErrorCode SGS_DD_ModelCreateDM(DM dm_source, DM *dm_sgs, PetscInt degree, PetscInt *num_components) {
  PetscFE      fe;
  PetscSection section;
  PetscInt     dim;

  PetscFunctionBeginUser;
  *num_components = 6;

  PetscCall(DMClone(dm_source, dm_sgs));
  PetscCall(DMGetDimension(*dm_sgs, &dim));
  PetscCall(PetscObjectSetName((PetscObject)*dm_sgs, "Subgrid Stress Projection"));

  PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, dim, *num_components, PETSC_FALSE, degree, PETSC_DECIDE, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "Subgrid Stress Projection"));
  PetscCall(DMAddField(*dm_sgs, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(*dm_sgs));
  PetscCall(DMPlexSetClosurePermutationTensor(*dm_sgs, PETSC_DETERMINE, NULL));

  PetscCall(DMGetLocalSection(*dm_sgs, &section));
  PetscCall(PetscSectionSetFieldName(section, 0, ""));
  PetscCall(PetscSectionSetComponentName(section, 0, 0, "KMSubgridStressXX"));
  PetscCall(PetscSectionSetComponentName(section, 0, 1, "KMSubgridStressYY"));
  PetscCall(PetscSectionSetComponentName(section, 0, 2, "KMSubgridStressZZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 3, "KMSubgridStressYZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 4, "KMSubgridStressXZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 5, "KMSubgridStressXY"));

  PetscCall(PetscFEDestroy(&fe));

  PetscFunctionReturn(0);
};

// @brief Create CeedOperator to calculate data-drive SGS at nodes
PetscErrorCode SGS_DD_ModelSetupNodalEvaluation(Ceed ceed, User user, CeedData ceed_data, SGS_DD_ModelSetupData sgs_dd_setup_data) {
  SGS_DD_Data         sgs_dd_data = user->sgs_dd_data;
  CeedQFunction       qf_multiplicity, qf_sgs_dd_nodal;
  CeedOperator        op_multiplicity, op_sgs_dd_nodal;
  CeedInt             num_elem, elem_size, num_comp_q, dim, num_qpts_1d, num_comp_grad_velo, num_comp_x, num_comp_grid_aniso;
  CeedVector          multiplicity, inv_multiplicity, grad_velo_ceed;
  CeedElemRestriction elem_restr_inv_multiplicity, elem_restr_grad_velo, elem_restr_sgs;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(user->dm, &dim));
  CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_x, &num_comp_x);
  CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_q, &num_comp_q);
  CeedElemRestrictionGetNumComponents(sgs_dd_setup_data->elem_restr_grid_aniso, &num_comp_grid_aniso);
  CeedElemRestrictionGetNumElements(ceed_data->elem_restr_q, &num_elem);
  CeedElemRestrictionGetElementSize(ceed_data->elem_restr_q, &elem_size);
  CeedBasisGetNumQuadraturePoints1D(ceed_data->basis_q, &num_qpts_1d);

  {  // Get velocity gradient information
    CeedOperatorField op_field;
    CeedOperatorGetFieldByName(user->grad_velo_proj->l2_rhs_ctx->op, "velocity gradient", &op_field);
    CeedOperatorFieldGetElemRestriction(op_field, &elem_restr_grad_velo);
    CeedElemRestrictionGetNumComponents(elem_restr_grad_velo, &num_comp_grad_velo);
    CeedElemRestrictionCreateVector(elem_restr_grad_velo, &grad_velo_ceed, NULL);
  }

  PetscCall(GetRestrictionForDomain(ceed, sgs_dd_data->dm_sgs, 0, 0, 0, num_qpts_1d, 0, &elem_restr_sgs, NULL, NULL));
  CeedElemRestrictionCreateVector(elem_restr_sgs, &sgs_dd_data->sgs_nodal_ceed, NULL);

  // -- Create inverse multiplicity for correcting nodal assembly
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &multiplicity, NULL);
  CeedElemRestrictionGetMultiplicity(ceed_data->elem_restr_q, multiplicity);
  CeedElemRestrictionCreateStrided(ceed, num_elem, elem_size, 1, num_elem * elem_size, CEED_STRIDES_BACKEND, &elem_restr_inv_multiplicity);
  CeedElemRestrictionCreateVector(elem_restr_inv_multiplicity, &inv_multiplicity, NULL);

  CeedQFunctionCreateInterior(ceed, 1, InverseMultiplicity, InverseMultiplicity_loc, &qf_multiplicity);
  CeedQFunctionAddInput(qf_multiplicity, "multiplicity", num_comp_q, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_multiplicity, "inverse multiplicity", 1, CEED_EVAL_NONE);

  CeedOperatorCreate(ceed, qf_multiplicity, NULL, NULL, &op_multiplicity);
  CeedOperatorSetName(op_multiplicity, "SGS DD Model - Create Multiplicity Scaling");
  CeedOperatorSetField(op_multiplicity, "multiplicity", ceed_data->elem_restr_q, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_multiplicity, "inverse multiplicity", elem_restr_inv_multiplicity, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedOperatorSetNumQuadraturePoints(op_multiplicity, elem_size);

  CeedOperatorApply(op_multiplicity, multiplicity, inv_multiplicity, CEED_REQUEST_IMMEDIATE);

  // -- Create operator for SGS DD model nodal evaluation
  switch (user->phys->state_var) {
    case STATEVAR_PRIMITIVE:
      CeedQFunctionCreateInterior(ceed, 1, ComputeSGS_DDAnisotropicNodal_Prim, ComputeSGS_DDAnisotropicNodal_Prim_loc, &qf_sgs_dd_nodal);
      break;
    case STATEVAR_CONSERVATIVE:
      CeedQFunctionCreateInterior(ceed, 1, ComputeSGS_DDAnisotropicNodal_Conserv, ComputeSGS_DDAnisotropicNodal_Conserv_loc, &qf_sgs_dd_nodal);
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)user->dm), PETSC_ERR_SUP,
              "Anisotropic data-driven SGS nodal evaluation not available for chosen state variable");
  }

  // Mesh/geometry order and solution basis order may differ, therefore must interpolate
  CeedBasis basis_x_to_q;
  PetscCall(CeedBasisCreateProjection(ceed_data->basis_x, ceed_data->basis_q, &basis_x_to_q));

  CeedQFunctionSetContext(qf_sgs_dd_nodal, sgs_dd_setup_data->sgsdd_qfctx);
  CeedQFunctionAddInput(qf_sgs_dd_nodal, "q", num_comp_q, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_sgs_dd_nodal, "x", num_comp_x, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_sgs_dd_nodal, "gradient velocity", num_comp_grad_velo, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_sgs_dd_nodal, "anisotropy tensor", num_comp_grid_aniso, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_sgs_dd_nodal, "inverse multiplicity", 1, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_sgs_dd_nodal, "km_sgs", sgs_dd_data->num_comp_sgs, CEED_EVAL_NONE);

  CeedOperatorCreate(ceed, qf_sgs_dd_nodal, NULL, NULL, &op_sgs_dd_nodal);
  CeedOperatorSetField(op_sgs_dd_nodal, "q", ceed_data->elem_restr_q, CEED_BASIS_COLLOCATED, user->q_ceed);
  CeedOperatorSetField(op_sgs_dd_nodal, "x", ceed_data->elem_restr_x, basis_x_to_q, ceed_data->x_coord);
  CeedOperatorSetField(op_sgs_dd_nodal, "gradient velocity", elem_restr_grad_velo, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_sgs_dd_nodal, "anisotropy tensor", sgs_dd_setup_data->elem_restr_grid_aniso, CEED_BASIS_COLLOCATED,
                       sgs_dd_setup_data->grid_aniso_ceed);
  CeedOperatorSetField(op_sgs_dd_nodal, "inverse multiplicity", elem_restr_inv_multiplicity, CEED_BASIS_COLLOCATED, inv_multiplicity);
  CeedOperatorSetField(op_sgs_dd_nodal, "km_sgs", elem_restr_sgs, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  PetscCall(OperatorApplyContextCreate(user->grad_velo_proj->dm, sgs_dd_data->dm_sgs, ceed, op_sgs_dd_nodal, grad_velo_ceed,
                                       sgs_dd_data->sgs_nodal_ceed, NULL, NULL, &sgs_dd_data->op_nodal_evaluation_ctx));

  sgs_dd_setup_data->elem_restr_sgs = elem_restr_sgs;

  CeedVectorDestroy(&multiplicity);
  CeedVectorDestroy(&inv_multiplicity);
  CeedVectorDestroy(&grad_velo_ceed);
  CeedBasisDestroy(&basis_x_to_q);
  CeedElemRestrictionDestroy(&elem_restr_inv_multiplicity);
  CeedQFunctionDestroy(&qf_multiplicity);
  CeedQFunctionDestroy(&qf_sgs_dd_nodal);
  CeedOperatorDestroy(&op_multiplicity);
  CeedOperatorDestroy(&op_sgs_dd_nodal);
  PetscFunctionReturn(0);
}

// @brief B = A^T, A is NxM, B is MxN
PetscErrorCode TransposeMatrix(const PetscScalar *A, PetscScalar *B, const PetscInt N, const PetscInt M) {
  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < N; i++) {
    for (PetscInt j = 0; j < M; j++) {
      B[j * N + i] = A[i * M + j];
    }
  }
  PetscFunctionReturn(0);
}

// @brief Read neural network coefficients from file and put into context struct
PetscErrorCode SGS_DD_ModelContextFill(MPI_Comm comm, char data_dir[PETSC_MAX_PATH_LEN], SGS_DDModelContext *psgsdd_ctx) {
  SGS_DDModelContext sgsdd_ctx;
  PetscInt           num_inputs = (*psgsdd_ctx)->num_inputs, num_outputs = (*psgsdd_ctx)->num_outputs, num_neurons = (*psgsdd_ctx)->num_neurons;
  char               file_path[PETSC_MAX_PATH_LEN];
  PetscScalar       *temp;

  PetscFunctionBeginUser;
  {
    SGS_DDModelContext sgsdd_temp;
    PetscCall(PetscNew(&sgsdd_temp));
    *sgsdd_temp                     = **psgsdd_ctx;
    sgsdd_temp->offsets.bias1       = 0;
    sgsdd_temp->offsets.bias2       = sgsdd_temp->offsets.bias1 + num_neurons;
    sgsdd_temp->offsets.weight1     = sgsdd_temp->offsets.bias2 + num_neurons;
    sgsdd_temp->offsets.weight2     = sgsdd_temp->offsets.weight1 + num_neurons * num_inputs;
    sgsdd_temp->offsets.out_scaling = sgsdd_temp->offsets.weight2 + num_inputs * num_neurons;
    PetscInt total_num_scalars      = sgsdd_temp->offsets.out_scaling + 2 * num_outputs;
    sgsdd_temp->total_bytes         = sizeof(*sgsdd_ctx) + total_num_scalars * sizeof(sgsdd_ctx->data[0]);
    PetscCall(PetscMalloc(sgsdd_temp->total_bytes, &sgsdd_ctx));
    *sgsdd_ctx = *sgsdd_temp;
    PetscCall(PetscFree(sgsdd_temp));
  }

  PetscCall(PetscSNPrintf(file_path, sizeof file_path, "%s/%s", data_dir, "b1.dat"));
  PetscCall(PHASTADatFileReadToArrayReal(comm, file_path, &sgsdd_ctx->data[sgsdd_ctx->offsets.bias1]));
  PetscCall(PetscSNPrintf(file_path, sizeof file_path, "%s/%s", data_dir, "b2.dat"));
  PetscCall(PHASTADatFileReadToArrayReal(comm, file_path, &sgsdd_ctx->data[sgsdd_ctx->offsets.bias2]));
  PetscCall(PetscSNPrintf(file_path, sizeof file_path, "%s/%s", data_dir, "OutScaling.dat"));
  PetscCall(PHASTADatFileReadToArrayReal(comm, file_path, &sgsdd_ctx->data[sgsdd_ctx->offsets.out_scaling]));

  {
    PetscCall(PetscMalloc1(num_inputs * num_neurons, &temp));
    PetscCall(PetscSNPrintf(file_path, sizeof file_path, "%s/%s", data_dir, "w1.dat"));
    PetscCall(PHASTADatFileReadToArrayReal(comm, file_path, temp));
    PetscCall(TransposeMatrix(temp, &sgsdd_ctx->data[sgsdd_ctx->offsets.weight1], num_inputs, num_neurons));
    PetscCall(PetscFree(temp));
  }
  {
    PetscCall(PetscMalloc1(num_outputs * num_neurons, &temp));
    PetscCall(PetscSNPrintf(file_path, sizeof file_path, "%s/%s", data_dir, "w2.dat"));
    PetscCall(PHASTADatFileReadToArrayReal(comm, file_path, temp));
    PetscCall(TransposeMatrix(temp, &sgsdd_ctx->data[sgsdd_ctx->offsets.weight2], num_neurons, num_outputs));
    PetscCall(PetscFree(temp));
  }

  PetscCall(PetscFree(*psgsdd_ctx));
  *psgsdd_ctx = sgsdd_ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode SGS_DD_ModelSetup(Ceed ceed, User user, CeedData ceed_data, ProblemData *problem) {
  PetscReal                alpha = 0;
  SGS_DDModelContext       sgsdd_ctx;
  MPI_Comm                 comm                           = user->comm;
  char                     sgs_dd_dir[PETSC_MAX_PATH_LEN] = "./dd_sgs_parameters";
  SGS_DD_ModelSetupData    sgs_dd_setup_data;
  NewtonianIdealGasContext gas;
  PetscFunctionBeginUser;

  PetscCall(VelocityGradientProjectionSetup(ceed, user, ceed_data, problem));

  PetscCall(PetscNew(&sgsdd_ctx));

  PetscOptionsBegin(comm, NULL, "SGS Data-Drive Model Options", NULL);
  PetscCall(PetscOptionsReal("-sgs_model_dd_leakyrelu_alpha", "Slope parameter for Leaky ReLU activation function", NULL, alpha, &alpha, NULL));
  PetscCall(PetscOptionsString("-sgs_model_dd_parameter_dir", "Path to directory with model parameters (weights, biases, etc.)", NULL, sgs_dd_dir,
                               sgs_dd_dir, sizeof(sgs_dd_dir), NULL));
  PetscOptionsEnd();

  sgsdd_ctx->num_layers  = 2;
  sgsdd_ctx->num_inputs  = 6;
  sgsdd_ctx->num_outputs = 6;
  sgsdd_ctx->num_neurons = 20;
  sgsdd_ctx->alpha       = alpha;

  // PetscCall(SGS_DD_ModelContextFill(comm, sgs_dd_dir, &sgsdd_ctx));

  // -- Create DM for storing SGS tensor at nodes
  PetscCall(PetscNew(&user->sgs_dd_data));
  PetscCall(SGS_DD_ModelCreateDM(user->dm, &user->sgs_dd_data->dm_sgs, user->app_ctx->degree, &user->sgs_dd_data->num_comp_sgs));

  PetscCall(PetscNew(&sgs_dd_setup_data));

  CeedQFunctionContextGetDataRead(problem->apply_vol_ifunction.qfunction_context, CEED_MEM_HOST, &gas);
  sgsdd_ctx->gas = *gas;
  CeedQFunctionContextRestoreDataRead(problem->apply_vol_ifunction.qfunction_context, &gas);
  CeedQFunctionContextCreate(user->ceed, &sgs_dd_setup_data->sgsdd_qfctx);
  CeedQFunctionContextSetData(sgs_dd_setup_data->sgsdd_qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sgsdd_ctx->total_bytes, sgsdd_ctx);
  CeedQFunctionContextSetDataDestroy(sgs_dd_setup_data->sgsdd_qfctx, CEED_MEM_HOST, FreeContextPetsc);

  // -- Compute and store anisotropy tensor
  PetscCall(GridAnisotropyTensorProjectionSetupApply(ceed, user, ceed_data, &sgs_dd_setup_data->elem_restr_grid_aniso,
                                                     &sgs_dd_setup_data->grid_aniso_ceed));

  // -- Create Nodal Evaluation Operator
  PetscCall(SGS_DD_ModelSetupNodalEvaluation(ceed, user, ceed_data, sgs_dd_setup_data));

  PetscCall(SGS_DD_ModelSetupDataDestroy(sgs_dd_setup_data));
  PetscFunctionReturn(0);
}

PetscErrorCode SGS_DD_DataDestroy(SGS_DD_Data sgs_dd_data) {
  PetscFunctionBeginUser;
  if (!sgs_dd_data) PetscFunctionReturn(0);

  CeedVectorDestroy(&sgs_dd_data->sgs_nodal_ceed);
  PetscCall(OperatorApplyContextDestroy(sgs_dd_data->op_nodal_evaluation_ctx));
  PetscCall(DMDestroy(&sgs_dd_data->dm_sgs));
  PetscCall(PetscFree(sgs_dd_data));

  PetscFunctionReturn(0);
}
