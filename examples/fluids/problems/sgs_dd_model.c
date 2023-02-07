// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../qfunctions/sgs_dd_model.h"

#include "../navierstokes.h"

typedef struct {
  CeedElemRestriction  elem_restr_grid_aniso, elem_restr_sgs;
  CeedBasis            basis_grid_aniso;
  CeedVector           grid_aniso_ceed;
  CeedQFunctionContext sgsdd_qfctx;
} *SGS_DD_ModelSetupData;

PetscErrorCode SGS_DD_ModelSetupDataDestroy(SGS_DD_ModelSetupData sgs_dd_setup_data) {
  PetscFunctionBeginUser;

  CeedElemRestrictionDestroy(&sgs_dd_setup_data->elem_restr_grid_aniso);
  CeedElemRestrictionDestroy(&sgs_dd_setup_data->elem_restr_sgs);
  CeedBasisDestroy(&sgs_dd_setup_data->basis_grid_aniso);
  CeedVectorDestroy(&sgs_dd_setup_data->grid_aniso_ceed);
  CeedQFunctionContextDestroy(&sgs_dd_setup_data->sgsdd_qfctx);

  PetscCall(PetscFree(sgs_dd_setup_data));
  PetscFunctionReturn(0);
}

PetscErrorCode SGS_DD_ModelCreateDM(User user, ProblemData *problem, PetscInt degree, SimpleBC bc) {
  SGS_DD_Data  sgs_dd_data;
  PetscFE      fe;
  PetscSection section;
  PetscInt     dim;
  PetscFunctionBeginUser;

  PetscCall(PetscNew(&sgs_dd_data));

  // -- Create DM for storing subgrid stress at nodes
  sgs_dd_data->num_comp_sgs = 6;

  PetscCall(DMClone(user->dm, &sgs_dd_data->dm_sgs));
  PetscCall(DMGetDimension(sgs_dd_data->dm_sgs, &dim));
  PetscCall(PetscObjectSetName((PetscObject)sgs_dd_data->dm_sgs, "Subgrid Stress Projection"));

  PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, dim, sgs_dd_data->num_comp_sgs, PETSC_FALSE, degree, PETSC_DECIDE, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "Subgrid Stress Projection"));
  PetscCall(DMAddField(sgs_dd_data->dm_sgs, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(sgs_dd_data->dm_sgs));
  PetscCall(DMPlexSetClosurePermutationTensor(sgs_dd_data->dm_sgs, PETSC_DETERMINE, NULL));

  PetscCall(DMGetLocalSection(sgs_dd_data->dm_sgs, &section));
  PetscCall(PetscSectionSetFieldName(section, 0, ""));
  PetscCall(PetscSectionSetComponentName(section, 0, 0, "KMSubgridStressXX"));
  PetscCall(PetscSectionSetComponentName(section, 0, 1, "KMSubgridStressYY"));
  PetscCall(PetscSectionSetComponentName(section, 0, 2, "KMSubgridStressZZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 3, "KMSubgridStressYZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 4, "KMSubgridStressXZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 5, "KMSubgridStressXY"));

  PetscCall(PetscFEDestroy(&fe));

  user->sgs_dd_data = sgs_dd_data;

  PetscFunctionReturn(0);
};

PetscErrorCode SGS_DD_ModelSetupNodalEvaluation(Ceed ceed, User user, CeedData ceed_data, SGS_DD_ModelSetupData sgs_dd_setup_data) {
  SGS_DD_Data         sgs_dd_data = user->sgs_dd_data;
  CeedQFunction       qf_multiplicity, qf_sgs_dd_nodal;
  CeedOperator        op_multiplicity, op_sgs_dd_nodal;
  CeedInt             num_elem, elem_size, num_comp_q, dim, num_qpts_1d, num_nodes_1d, num_comp_grad_velo, num_comp_x, num_comp_grid_aniso;
  CeedVector          multiplicity, scale_stored;
  CeedElemRestriction elem_restr_scale, elem_restr_grad_velo, elem_restr_sgs;
  CeedBasis           basis_grad_velo;
  CeedOperatorField   op_field;
  PetscFunctionBeginUser;

  PetscCall(DMGetDimension(user->dm, &dim));
  CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_x, &num_comp_x);
  CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_q, &num_comp_q);
  CeedElemRestrictionGetNumComponents(sgs_dd_setup_data->elem_restr_grid_aniso, &num_comp_grid_aniso);
  CeedElemRestrictionGetNumElements(ceed_data->elem_restr_q, &num_elem);
  CeedElemRestrictionGetElementSize(ceed_data->elem_restr_q, &elem_size);
  CeedBasisGetNumQuadraturePoints1D(ceed_data->basis_q, &num_qpts_1d);
  CeedBasisGetNumNodes1D(ceed_data->basis_q, &num_nodes_1d);

  CeedOperatorGetFieldByName(user->grad_velo_proj->l2_rhs_ctx->op, "velocity gradient", &op_field);
  CeedOperatorFieldGetElemRestriction(op_field, &elem_restr_grad_velo);
  CeedElemRestrictionGetNumComponents(elem_restr_grad_velo, &num_comp_grad_velo);

  PetscCall(GetRestrictionForDomain(ceed, sgs_dd_data->dm_sgs, 0, 0, 0, num_qpts_1d, 0, &elem_restr_sgs, NULL, NULL));
  CeedElemRestrictionCreateVector(elem_restr_sgs, &sgs_dd_data->sgs_nodal_ceed, NULL);

  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_grad_velo, num_nodes_1d, num_qpts_1d, CEED_GAUSS_LOBATTO, &basis_grad_velo);

  // -- Create multiplicity scale for correcting nodal assembly
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &multiplicity, NULL);
  CeedElemRestrictionGetMultiplicity(ceed_data->elem_restr_q, multiplicity);
  CeedElemRestrictionCreateStrided(ceed, num_elem, elem_size, 1, num_elem * elem_size, CEED_STRIDES_BACKEND, &elem_restr_scale);
  CeedElemRestrictionCreateVector(elem_restr_scale, &scale_stored, NULL);

  CeedQFunctionCreateInterior(ceed, 1, InverseMultiplicity, InverseMultiplicity_loc, &qf_multiplicity);
  CeedQFunctionAddInput(qf_multiplicity, "multiplicity", num_comp_q, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_multiplicity, "scale", 1, CEED_EVAL_NONE);

  CeedOperatorCreate(ceed, qf_multiplicity, NULL, NULL, &op_multiplicity);
  CeedOperatorSetName(op_multiplicity, "SGS DD Model - Create Multiplicity Scaling");
  CeedOperatorSetField(op_multiplicity, "multiplicity", ceed_data->elem_restr_q, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_multiplicity, "scale", elem_restr_scale, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedOperatorSetNumQuadraturePoints(op_multiplicity, elem_size);

  CeedOperatorApply(op_multiplicity, multiplicity, scale_stored, CEED_REQUEST_IMMEDIATE);

  // -- Create operator for SGS DD model nodal evaluation
  switch (user->phys->state_var) {
    case STATEVAR_PRIMITIVE:
      CeedQFunctionCreateInterior(ceed, 1, ComputeSGS_DDAnisotropicNodal_Prim, ComputeSGS_DDAnisotropicNodal_Prim_loc, &qf_sgs_dd_nodal);
      break;
    case STATEVAR_CONSERVATIVE:
      CeedQFunctionCreateInterior(ceed, 1, ComputeSGS_DDAnisotropicNodal_Conserv, ComputeSGS_DDAnisotropicNodal_Conserv_loc, &qf_sgs_dd_nodal);
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)user->dm), PETSC_ERR_SUP, "No statisics collection available for chosen state variable");
  }
  CeedQFunctionSetContext(qf_sgs_dd_nodal, sgs_dd_setup_data->sgsdd_qfctx);
  CeedQFunctionAddInput(qf_sgs_dd_nodal, "q", num_comp_q, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_sgs_dd_nodal, "x", num_comp_x, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_sgs_dd_nodal, "gradient velocity", num_comp_grad_velo, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_sgs_dd_nodal, "anisotropy tensor", num_comp_grid_aniso, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_sgs_dd_nodal, "scale", 1, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_sgs_dd_nodal, "km_sgs", sgs_dd_data->num_comp_sgs, CEED_EVAL_NONE);

  // WARNING These CeedBasis objects should be going to CEED_GAUSS_LOBATTO points, not CEED_GAUSS
  // Evidence, grad_velo uses CEED_GAUSS_LOBATTO for it's basis evaluation
  CeedOperatorCreate(ceed, qf_sgs_dd_nodal, NULL, NULL, &op_sgs_dd_nodal);
  CeedOperatorSetField(op_sgs_dd_nodal, "q", ceed_data->elem_restr_q, ceed_data->basis_q, user->q_ceed);
  CeedOperatorSetField(op_sgs_dd_nodal, "x", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord);
  CeedOperatorSetField(op_sgs_dd_nodal, "gradient velocity", elem_restr_grad_velo, basis_grad_velo, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_sgs_dd_nodal, "anisotropy tensor", sgs_dd_setup_data->elem_restr_grid_aniso, sgs_dd_setup_data->basis_grid_aniso,
                       sgs_dd_setup_data->grid_aniso_ceed);
  CeedOperatorSetField(op_sgs_dd_nodal, "scale", elem_restr_scale, CEED_BASIS_COLLOCATED, scale_stored);
  CeedOperatorSetField(op_sgs_dd_nodal, "km_sgs", elem_restr_sgs, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  sgs_dd_data->op_nodal_evaluation  = op_sgs_dd_nodal;
  sgs_dd_setup_data->elem_restr_sgs = elem_restr_sgs;

  CeedVectorDestroy(&multiplicity);
  CeedVectorDestroy(&scale_stored);
  CeedElemRestrictionDestroy(&elem_restr_scale);
  CeedQFunctionDestroy(&qf_multiplicity);
  CeedQFunctionDestroy(&qf_sgs_dd_nodal);
  CeedOperatorDestroy(&op_multiplicity);
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

  PetscCall(SGS_DD_ModelContextFill(comm, sgs_dd_dir, &sgsdd_ctx));

  PetscCall(PetscNew(&sgs_dd_setup_data));

  CeedQFunctionContextGetDataRead(problem->apply_vol_ifunction.qfunction_context, CEED_MEM_HOST, &gas);
  sgsdd_ctx->gas = *gas;
  CeedQFunctionContextRestoreDataRead(problem->apply_vol_ifunction.qfunction_context, &gas);
  CeedQFunctionContextCreate(user->ceed, &sgs_dd_setup_data->sgsdd_qfctx);
  CeedQFunctionContextSetData(sgs_dd_setup_data->sgsdd_qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sgsdd_ctx->total_bytes, sgsdd_ctx);
  CeedQFunctionContextSetDataDestroy(sgs_dd_setup_data->sgsdd_qfctx, CEED_MEM_HOST, FreeContextPetsc);

  // -- Compute and store anisotropy tensor
  PetscCall(GridAnisotropyTensorProjectionSetupApply(ceed, user, ceed_data, problem, &sgs_dd_setup_data->elem_restr_grid_aniso, &sgs_dd_setup_data->basis_grid_aniso, &sgs_dd_setup_data->grid_aniso_ceed));

  // -- Create Nodal Evaluation Operator
  PetscCall(SGS_DD_ModelSetupNodalEvaluation(ceed, user, ceed_data, sgs_dd_setup_data));

  PetscCall(SGS_DD_ModelSetupDataDestroy(sgs_dd_setup_data));
  PetscFunctionReturn(0);
}

PetscErrorCode SGS_DD_DataDestroy(SGS_DD_Data sgs_dd_data) {
  PetscFunctionBeginUser;

  if (!sgs_dd_data) PetscFunctionReturn(0);

  CeedVectorDestroy(&sgs_dd_data->sgs_nodal_ceed);

  CeedOperatorDestroy(&sgs_dd_data->op_nodal_evaluation);

  PetscCall(DMDestroy(&sgs_dd_data->dm_sgs));

  PetscCall(PetscFree(sgs_dd_data));

  PetscFunctionReturn(0);
}
