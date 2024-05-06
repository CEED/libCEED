// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../qfunctions/sgs_dd_model.h"

#include <petscdmplex.h>

#include "../include/libtorch.h"
#include "../navierstokes.h"

typedef struct {
  CeedElemRestriction  elem_restr_grid_aniso, elem_restr_sgs;
  CeedVector           grid_aniso_ceed;
  CeedQFunctionContext sgsdd_qfctx, ifunction_qfctx;
} *SgsDDSetupData;

PetscErrorCode SgsDDSetupDataDestroy(SgsDDSetupData sgs_dd_setup_data) {
  Ceed ceed;

  PetscFunctionBeginUser;
  PetscCall(CeedElemRestrictionGetCeed(sgs_dd_setup_data->elem_restr_sgs, &ceed));

  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&sgs_dd_setup_data->elem_restr_grid_aniso));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&sgs_dd_setup_data->elem_restr_sgs));
  PetscCallCeed(ceed, CeedVectorDestroy(&sgs_dd_setup_data->grid_aniso_ceed));
  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&sgs_dd_setup_data->sgsdd_qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&sgs_dd_setup_data->ifunction_qfctx));
  PetscCall(PetscFree(sgs_dd_setup_data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Create DM for storing subgrid stress at nodes
static PetscErrorCode SgsDDCreateDM(DM dm_source, DM *dm_sgs, PetscInt degree, PetscInt q_extra, PetscInt *num_components) {
  PetscSection section;

  PetscFunctionBeginUser;
  *num_components = 6;

  PetscCall(DMClone(dm_source, dm_sgs));
  PetscCall(PetscObjectSetName((PetscObject)*dm_sgs, "Subgrid Stress Projection"));

  PetscCall(DMSetupByOrder_FEM(PETSC_TRUE, PETSC_TRUE, degree, 1, q_extra, 1, num_components, *dm_sgs));

  PetscCall(DMGetLocalSection(*dm_sgs, &section));
  PetscCall(PetscSectionSetFieldName(section, 0, ""));
  PetscCall(PetscSectionSetComponentName(section, 0, 0, "KMSubgridStressXX"));
  PetscCall(PetscSectionSetComponentName(section, 0, 1, "KMSubgridStressYY"));
  PetscCall(PetscSectionSetComponentName(section, 0, 2, "KMSubgridStressZZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 3, "KMSubgridStressYZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 4, "KMSubgridStressXZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 5, "KMSubgridStressXY"));
  PetscFunctionReturn(PETSC_SUCCESS);
};

// @brief Evaluate data-driven SGS using fused method
static PetscErrorCode SgsDDNodalStressEval_Fused(User user, Vec Q_loc, Vec VelocityGradient, Vec SGSNodal_loc) {
  SgsDDData    sgs_dd_data = user->sgs_dd_data;
  PetscMemType q_mem_type;

  PetscFunctionBeginUser;
  PetscCall(VecPetscToCeed(Q_loc, &q_mem_type, user->q_ceed));  // q_ceed is an implicit input

  PetscCall(ApplyCeedOperatorGlobalToLocal(VelocityGradient, SGSNodal_loc, sgs_dd_data->op_nodal_evaluation_ctx));

  PetscCall(VecCeedToPetsc(user->q_ceed, q_mem_type, Q_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Create CeedOperator to calculate data-drive SGS at nodes using fused operator
static PetscErrorCode SgsDDSetupNodalEvaluation_Fused(Ceed ceed, User user, CeedData ceed_data, SgsDDSetupData sgs_dd_setup_data) {
  SgsDDData           sgs_dd_data = user->sgs_dd_data;
  CeedQFunction       qf_sgs_dd_nodal;
  CeedOperator        op_sgs_dd_nodal;
  CeedInt             num_comp_q, num_comp_grad_velo, num_comp_x, num_comp_grid_aniso;
  PetscInt            dim;
  CeedVector          inv_multiplicity;
  CeedElemRestriction elem_restr_inv_multiplicity, elem_restr_grad_velo, elem_restr_sgs;
  DMLabel             domain_label = NULL;
  PetscInt            label_value = 0, height = 0, dm_field = 0;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(user->dm, &dim));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_x, &num_comp_x));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_q, &num_comp_q));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(sgs_dd_setup_data->elem_restr_grid_aniso, &num_comp_grid_aniso));

  {  // Get velocity gradient information
    CeedOperatorField op_field;
    PetscCallCeed(ceed, CeedOperatorGetFieldByName(user->grad_velo_proj->l2_rhs_ctx->op, "velocity gradient", &op_field));
    PetscCallCeed(ceed, CeedOperatorFieldGetElemRestriction(op_field, &elem_restr_grad_velo));
    PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_grad_velo, &num_comp_grad_velo));
  }
  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, sgs_dd_data->dm_sgs, domain_label, label_value, height, dm_field, &elem_restr_sgs));
  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_sgs, &sgs_dd_data->sgs_nodal_ceed, NULL));

  PetscCall(GetInverseMultiplicity(ceed, sgs_dd_data->dm_sgs, domain_label, label_value, height, dm_field, PETSC_FALSE, &elem_restr_inv_multiplicity,
                                   &inv_multiplicity));

  // -- Create operator for SGS DD model nodal evaluation
  switch (user->phys->state_var) {
    case STATEVAR_PRIMITIVE:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, ComputeSgsDDNodal_Prim, ComputeSgsDDNodal_Prim_loc, &qf_sgs_dd_nodal));
      break;
    case STATEVAR_CONSERVATIVE:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, ComputeSgsDDNodal_Conserv, ComputeSgsDDNodal_Conserv_loc, &qf_sgs_dd_nodal));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)user->dm), PETSC_ERR_SUP, "Data-driven SGS nodal evaluation not available for chosen state variable");
  }

  // Mesh/geometry order and solution basis order may differ, therefore must interpolate
  CeedBasis basis_x_to_q;
  PetscCallCeed(ceed, CeedBasisCreateProjection(ceed_data->basis_x, ceed_data->basis_q, &basis_x_to_q));

  PetscCallCeed(ceed, CeedQFunctionSetContext(qf_sgs_dd_nodal, sgs_dd_setup_data->sgsdd_qfctx));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_nodal, "q", num_comp_q, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_nodal, "x", num_comp_x, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_nodal, "gradient velocity", num_comp_grad_velo, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_nodal, "anisotropy tensor", num_comp_grid_aniso, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_nodal, "inverse multiplicity", 1, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_sgs_dd_nodal, "km_sgs", sgs_dd_data->num_comp_sgs, CEED_EVAL_NONE));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_sgs_dd_nodal, NULL, NULL, &op_sgs_dd_nodal));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_nodal, "q", ceed_data->elem_restr_q, CEED_BASIS_NONE, user->q_ceed));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_nodal, "x", ceed_data->elem_restr_x, basis_x_to_q, ceed_data->x_coord));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_nodal, "gradient velocity", elem_restr_grad_velo, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_nodal, "anisotropy tensor", sgs_dd_setup_data->elem_restr_grid_aniso, CEED_BASIS_NONE,
                                           sgs_dd_setup_data->grid_aniso_ceed));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_nodal, "inverse multiplicity", elem_restr_inv_multiplicity, CEED_BASIS_NONE, inv_multiplicity));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_nodal, "km_sgs", elem_restr_sgs, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  PetscCall(OperatorApplyContextCreate(user->grad_velo_proj->dm, sgs_dd_data->dm_sgs, ceed, op_sgs_dd_nodal, NULL, sgs_dd_data->sgs_nodal_ceed, NULL,
                                       NULL, &sgs_dd_data->op_nodal_evaluation_ctx));

  sgs_dd_setup_data->elem_restr_sgs = elem_restr_sgs;
  sgs_dd_data->sgs_nodal_eval       = SgsDDNodalStressEval_Fused;

  PetscCallCeed(ceed, CeedVectorDestroy(&inv_multiplicity));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_x_to_q));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_inv_multiplicity));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_sgs_dd_nodal));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_sgs_dd_nodal));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Setup data-driven model inference using internal (libCEED native) implementation
static PetscErrorCode SgsDDSetupNodalEvaluation_Sequential_Internal(Ceed ceed, SgsDDData sgs_dd_data, SgsDDSetupData sgs_dd_setup_data,
                                                                    CeedElemRestriction elem_restr_dd_inputs,
                                                                    CeedElemRestriction elem_restr_dd_outputs,
                                                                    CeedElemRestriction elem_restr_inv_multiplicity, CeedVector inv_multiplicity,
                                                                    void **ctx) {
  CeedQFunction         qf_sgs_dd_inference;
  CeedOperator          op_sgs_dd_inference;
  OperatorApplyContext *op_context = (OperatorApplyContext *)ctx;

  PetscFunctionBeginUser;
  PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, ComputeSgsDDNodal_Sequential_Inference, ComputeSgsDDNodal_Sequential_Inference_loc,
                                                  &qf_sgs_dd_inference));

  PetscCallCeed(ceed, CeedQFunctionSetContext(qf_sgs_dd_inference, sgs_dd_setup_data->sgsdd_qfctx));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_inference, "model inputs", sgs_dd_data->num_comp_inputs, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_inference, "inverse multiplicity", 1, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_sgs_dd_inference, "model outputs", sgs_dd_data->num_comp_outputs, CEED_EVAL_NONE));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_sgs_dd_inference, NULL, NULL, &op_sgs_dd_inference));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_inference, "model inputs", elem_restr_dd_inputs, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed,
                CeedOperatorSetField(op_sgs_dd_inference, "inverse multiplicity", elem_restr_inv_multiplicity, CEED_BASIS_NONE, inv_multiplicity));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_inference, "model outputs", elem_restr_dd_outputs, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  PetscCall(OperatorApplyContextCreate(sgs_dd_data->dm_dd_inputs, sgs_dd_data->dm_dd_outputs, ceed, op_sgs_dd_inference, NULL, NULL, NULL, NULL,
                                       op_context));
  sgs_dd_data->sgs_nodal_inference_ctx_destroy = (PetscErrorCode(*)(void *))OperatorApplyContextDestroy;

  PetscCallCeed(ceed, CeedOperatorDestroy(&op_sgs_dd_inference));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_sgs_dd_inference));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Perform data-driven model inference using internal (libCEED native) implementation
PetscErrorCode SgsDDNodalStressEval_Sequential_Internal(Vec DD_Inputs_loc, Vec DD_Outputs_loc, void *ctx) {
  OperatorApplyContext op_context = *(OperatorApplyContext *)ctx;

  PetscFunctionBeginUser;
  PetscCall(ApplyCeedOperatorLocalToLocal(DD_Inputs_loc, DD_Outputs_loc, op_context));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Setup data-driven model inference using libtorch
static PetscErrorCode SgsDDSetupNodalEvaluation_Sequential_LibTorch(Ceed ceed, SgsDDData sgs_dd_data, SgsDDSetupData sgs_dd_setup_data,
                                                                    CeedElemRestriction elem_restr_dd_inputs,
                                                                    CeedElemRestriction elem_restr_dd_outputs,
                                                                    CeedElemRestriction elem_restr_inv_multiplicity, CeedVector inv_multiplicity,
                                                                    void **ctx) {
  const char     *ceed_resource;
  TorchDeviceType model_device_type;

  PetscFunctionBeginUser;
  PetscCallCeed(ceed, CeedGetResource(ceed, &ceed_resource));
  if (strstr(ceed_resource, "/gpu/cuda")) model_device_type = TORCH_DEVICE_CUDA;
  else if (strstr(ceed_resource, "/gpu/hip")) model_device_type = TORCH_DEVICE_HIP;
  else if (strstr(ceed_resource, "/gpu/sycl")) model_device_type = TORCH_DEVICE_XPU;
  else model_device_type = TORCH_DEVICE_CPU;

  PetscCall(LoadModel_LibTorch("./examples/fluids/createPyTorchModel/NNModel_HIT_fp64_jit.pt", model_device_type));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Perform data-driven model inference using libtorch
PetscErrorCode SgsDDNodalStressEval_Sequential_LibTorch(Vec DD_Inputs_loc, Vec DD_Outputs_loc, void *ctx) {
  PetscFunctionBeginUser;

  PetscCall(ModelInference_LibTorch(DD_Inputs_loc, DD_Outputs_loc));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Evaluate data-driven SGS using sequential method
PetscErrorCode SgsDDNodalStressEval_Sequential(User user, Vec Q_loc, Vec VelocityGradient, Vec SGSNodal_loc) {
  SgsDDData    sgs_dd_data = user->sgs_dd_data;
  PetscMemType q_mem_type;
  Vec          DD_Inputs_loc, DD_Outputs_loc;

  PetscFunctionBeginUser;
  PetscCall(DMGetLocalVector(sgs_dd_data->dm_dd_inputs, &DD_Inputs_loc));
  PetscCall(DMGetLocalVector(sgs_dd_data->dm_dd_outputs, &DD_Outputs_loc));
  PetscCall(VecPetscToCeed(Q_loc, &q_mem_type, user->q_ceed));  // q_ceed is an implicit input

  PetscCall(ApplyCeedOperatorGlobalToLocal(VelocityGradient, DD_Inputs_loc, sgs_dd_data->op_nodal_dd_inputs_ctx));
  PetscCall(sgs_dd_data->sgs_nodal_inference(DD_Inputs_loc, DD_Outputs_loc, &sgs_dd_data->sgs_nodal_inference_ctx));
  PetscCall(ApplyCeedOperatorLocalToLocal(DD_Outputs_loc, SGSNodal_loc, sgs_dd_data->op_nodal_dd_outputs_ctx));

  PetscCall(VecCeedToPetsc(user->q_ceed, q_mem_type, Q_loc));
  PetscCall(DMRestoreLocalVector(sgs_dd_data->dm_dd_inputs, &DD_Inputs_loc));
  PetscCall(DMRestoreLocalVector(sgs_dd_data->dm_dd_outputs, &DD_Outputs_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Create CeedOperator to calculate data-drive SGS at nodes using sequentially-applied operators
static PetscErrorCode SgsDDSetupNodalEvaluation_Sequential(Ceed ceed, User user, CeedData ceed_data, SgsDDSetupData sgs_dd_setup_data) {
  SgsDDData           sgs_dd_data = user->sgs_dd_data;
  CeedInt             num_comp_q, num_comp_grad_velo, num_comp_x, num_comp_grid_aniso, num_comp_eigvec = 9 + 1;
  PetscInt            dim;
  CeedVector          inv_multiplicity, eigvec;
  CeedElemRestriction elem_restr_inv_multiplicity, elem_restr_grad_velo, elem_restr_sgs, elem_restr_eigvec, elem_restr_dd_inputs,
      elem_restr_dd_outputs;
  DMLabel  domain_label = NULL;
  PetscInt label_value = 0, height = 0, dm_field = 0;

  PetscFunctionBeginUser;
  {  // Create DMs for data-driven input and output values
    PetscSection section;
    PetscInt     degree, q_extra;
    {  // Get degree and number of quadrature points from dm_sgs
      PetscFE         fe;
      PetscSpace      basis;
      PetscQuadrature quadrature;
      PetscInt        num_qpnts;
      PetscCall(DMGetField(sgs_dd_data->dm_sgs, 0, NULL, (PetscObject *)&fe));
      PetscCall(PetscFEGetBasisSpace(fe, &basis));
      PetscCall(PetscSpaceGetDegree(basis, &degree, NULL));
      PetscCall(PetscFEGetQuadrature(fe, &quadrature));
      PetscCall(PetscQuadratureGetOrder(quadrature, &num_qpnts));
      q_extra = degree - num_qpnts;
    }

    PetscCall(DMClone(sgs_dd_data->dm_sgs, &sgs_dd_data->dm_dd_inputs));
    PetscCall(PetscObjectSetName((PetscObject)sgs_dd_data->dm_dd_inputs, "Data-Driven Model Inputs"));
    PetscCall(DMSetupByOrder_FEM(PETSC_TRUE, PETSC_TRUE, degree, 1, q_extra, 1, &sgs_dd_data->num_comp_inputs, sgs_dd_data->dm_dd_inputs));
    PetscCall(DMGetLocalSection(sgs_dd_data->dm_dd_inputs, &section));
    PetscCall(PetscSectionSetFieldName(section, 0, ""));
    for (CeedInt i = 0; i < sgs_dd_data->num_comp_inputs; i++) {
      char component_name[PETSC_MAX_PATH_LEN];

      PetscCall(PetscSNPrintf(component_name, sizeof component_name, "DataDrivenInput%" CeedInt_FMT, i + 1));
      PetscCall(PetscSectionSetComponentName(section, 0, i, component_name));
    }

    PetscCall(DMClone(sgs_dd_data->dm_sgs, &sgs_dd_data->dm_dd_outputs));
    PetscCall(PetscObjectSetName((PetscObject)sgs_dd_data->dm_dd_outputs, "Data-Driven Model Outputs"));
    PetscCall(DMSetupByOrder_FEM(PETSC_TRUE, PETSC_TRUE, degree, 1, q_extra, 1, &sgs_dd_data->num_comp_outputs, sgs_dd_data->dm_dd_outputs));
    PetscCall(DMGetLocalSection(sgs_dd_data->dm_dd_outputs, &section));
    PetscCall(PetscSectionSetFieldName(section, 0, ""));
    for (CeedInt i = 0; i < sgs_dd_data->num_comp_outputs; i++) {
      char component_name[PETSC_MAX_PATH_LEN];

      PetscCall(PetscSNPrintf(component_name, sizeof component_name, "DataDrivenOutput%" CeedInt_FMT, i + 1));
      PetscCall(PetscSectionSetComponentName(section, 0, i, component_name));
    }
  }

  PetscCall(DMGetDimension(user->dm, &dim));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_x, &num_comp_x));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_q, &num_comp_q));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(sgs_dd_setup_data->elem_restr_grid_aniso, &num_comp_grid_aniso));

  {  // Get velocity gradient information
    CeedOperatorField op_field;
    PetscCallCeed(ceed, CeedOperatorGetFieldByName(user->grad_velo_proj->l2_rhs_ctx->op, "velocity gradient", &op_field));
    PetscCallCeed(ceed, CeedOperatorFieldGetElemRestriction(op_field, &elem_restr_grad_velo));
    PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_grad_velo, &num_comp_grad_velo));
    PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_grad_velo, &sgs_dd_data->grad_velo_ceed, NULL));
  }

  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, sgs_dd_data->dm_sgs, domain_label, label_value, height, dm_field, &elem_restr_sgs));
  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_sgs, &sgs_dd_data->sgs_nodal_ceed, NULL));
  PetscCall(
      DMPlexCeedElemRestrictionCollocatedCreate(ceed, sgs_dd_data->dm_sgs, domain_label, label_value, height, num_comp_eigvec, &elem_restr_eigvec));
  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_eigvec, &eigvec, NULL));

  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, sgs_dd_data->dm_dd_inputs, domain_label, label_value, height, dm_field, &elem_restr_dd_inputs));
  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, sgs_dd_data->dm_dd_outputs, domain_label, label_value, height, dm_field, &elem_restr_dd_outputs));

  PetscCall(GetInverseMultiplicity(ceed, sgs_dd_data->dm_sgs, domain_label, label_value, height, dm_field, PETSC_FALSE, &elem_restr_inv_multiplicity,
                                   &inv_multiplicity));

  {  // Create operator for data-driven input evaluation
    CeedQFunction qf_sgs_dd_inputs;
    CeedOperator  op_sgs_dd_inputs;

    switch (user->phys->state_var) {
      case STATEVAR_PRIMITIVE:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, ComputeSgsDDNodal_Sequential_Inputs_Prim,
                                                        ComputeSgsDDNodal_Sequential_Inputs_Prim_loc, &qf_sgs_dd_inputs));
        break;
      case STATEVAR_CONSERVATIVE:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, ComputeSgsDDNodal_Sequential_Inputs_Conserv,
                                                        ComputeSgsDDNodal_Sequential_Inputs_Conserv_loc, &qf_sgs_dd_inputs));
        break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)user->dm), PETSC_ERR_SUP,
                "Data-driven SGS nodal input evaluation not available for chosen state variable");
    }

    PetscCallCeed(ceed, CeedQFunctionSetContext(qf_sgs_dd_inputs, sgs_dd_setup_data->sgsdd_qfctx));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_inputs, "q", num_comp_q, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_inputs, "gradient velocity", num_comp_grad_velo, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_inputs, "anisotropy tensor", num_comp_grid_aniso, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_inputs, "inverse multiplicity", 1, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_sgs_dd_inputs, "eigenvectors", num_comp_eigvec, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_sgs_dd_inputs, "model inputs", sgs_dd_data->num_comp_inputs, CEED_EVAL_NONE));

    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_sgs_dd_inputs, NULL, NULL, &op_sgs_dd_inputs));
    PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_inputs, "q", ceed_data->elem_restr_q, CEED_BASIS_NONE, user->q_ceed));
    PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_inputs, "gradient velocity", elem_restr_grad_velo, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_inputs, "anisotropy tensor", sgs_dd_setup_data->elem_restr_grid_aniso, CEED_BASIS_NONE,
                                             sgs_dd_setup_data->grid_aniso_ceed));
    PetscCallCeed(ceed,
                  CeedOperatorSetField(op_sgs_dd_inputs, "inverse multiplicity", elem_restr_inv_multiplicity, CEED_BASIS_NONE, inv_multiplicity));
    PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_inputs, "eigenvectors", elem_restr_eigvec, CEED_BASIS_NONE, eigvec));
    PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_inputs, "model inputs", elem_restr_dd_inputs, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

    PetscCall(OperatorApplyContextCreate(user->grad_velo_proj->dm, sgs_dd_data->dm_dd_inputs, ceed, op_sgs_dd_inputs, NULL, NULL, NULL, NULL,
                                         &sgs_dd_data->op_nodal_dd_inputs_ctx));
    PetscCallCeed(ceed, CeedOperatorDestroy(&op_sgs_dd_inputs));
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_sgs_dd_inputs));
  }

  {  // Create operator for data-driven output handling
    CeedQFunction qf_sgs_dd_outputs;
    CeedOperator  op_sgs_dd_outputs;

    PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, ComputeSgsDDNodal_Sequential_Outputs, ComputeSgsDDNodal_Sequential_Outputs_loc,
                                                    &qf_sgs_dd_outputs));
    PetscCallCeed(ceed, CeedQFunctionSetContext(qf_sgs_dd_outputs, sgs_dd_setup_data->sgsdd_qfctx));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_outputs, "model outputs", sgs_dd_data->num_comp_outputs, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_outputs, "anisotropy tensor", num_comp_grid_aniso, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_outputs, "inverse multiplicity", 1, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_outputs, "eigenvectors", num_comp_eigvec, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_sgs_dd_outputs, "km_sgs", sgs_dd_data->num_comp_sgs, CEED_EVAL_NONE));

    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_sgs_dd_outputs, NULL, NULL, &op_sgs_dd_outputs));
    PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_outputs, "model outputs", elem_restr_dd_outputs, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_outputs, "anisotropy tensor", sgs_dd_setup_data->elem_restr_grid_aniso, CEED_BASIS_NONE,
                                             sgs_dd_setup_data->grid_aniso_ceed));
    PetscCallCeed(ceed,
                  CeedOperatorSetField(op_sgs_dd_outputs, "inverse multiplicity", elem_restr_inv_multiplicity, CEED_BASIS_NONE, inv_multiplicity));
    PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_outputs, "eigenvectors", elem_restr_eigvec, CEED_BASIS_NONE, eigvec));
    PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_outputs, "km_sgs", elem_restr_sgs, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

    PetscCall(OperatorApplyContextCreate(sgs_dd_data->dm_dd_outputs, sgs_dd_data->dm_sgs, ceed, op_sgs_dd_outputs, NULL, sgs_dd_data->sgs_nodal_ceed,
                                         NULL, NULL, &sgs_dd_data->op_nodal_dd_outputs_ctx));
    PetscCallCeed(ceed, CeedOperatorDestroy(&op_sgs_dd_outputs));
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_sgs_dd_outputs));
  }

  sgs_dd_data->sgs_nodal_eval = SgsDDNodalStressEval_Sequential;

  if (false) {
    sgs_dd_data->sgs_nodal_inference = SgsDDNodalStressEval_Sequential_Internal;
    PetscCall(SgsDDSetupNodalEvaluation_Sequential_Internal(ceed, sgs_dd_data, sgs_dd_setup_data, elem_restr_dd_inputs, elem_restr_dd_outputs,
                                                            elem_restr_inv_multiplicity, inv_multiplicity, &sgs_dd_data->sgs_nodal_inference_ctx));
  } else {
    sgs_dd_data->sgs_nodal_inference = SgsDDNodalStressEval_Sequential_LibTorch;
    PetscCall(SgsDDSetupNodalEvaluation_Sequential_LibTorch(ceed, sgs_dd_data, sgs_dd_setup_data, elem_restr_dd_inputs, elem_restr_dd_outputs,
                                                            elem_restr_inv_multiplicity, inv_multiplicity, &sgs_dd_data->sgs_nodal_inference_ctx));
  }

  sgs_dd_setup_data->elem_restr_sgs = elem_restr_sgs;

  PetscCallCeed(ceed, CeedVectorDestroy(&inv_multiplicity));
  PetscCallCeed(ceed, CeedVectorDestroy(&eigvec));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_inv_multiplicity));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_eigvec));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_dd_inputs));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_dd_outputs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Create CeedOperator to compute SGS contribution to the residual
static PetscErrorCode SgsSetupNodalIFunction(Ceed ceed, User user, CeedData ceed_data, SgsDDSetupData sgs_dd_setup_data) {
  SgsDDData     sgs_dd_data = user->sgs_dd_data;
  CeedInt       num_comp_q, num_comp_qd, num_comp_x;
  PetscInt      dim;
  CeedQFunction qf_sgs_apply;
  CeedOperator  op_sgs_apply;
  CeedBasis     basis_sgs;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(user->dm, &dim));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_q, &num_comp_q));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_qd_i, &num_comp_qd));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_x, &num_comp_x));

  PetscCall(CreateBasisFromPlex(ceed, sgs_dd_data->dm_sgs, 0, 0, 0, 0, &basis_sgs));

  switch (user->phys->state_var) {
    case STATEVAR_PRIMITIVE:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, IFunction_NodalSgs_Prim, IFunction_NodalSgs_Prim_loc, &qf_sgs_apply));
      break;
    case STATEVAR_CONSERVATIVE:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, IFunction_NodalSgs_Conserv, IFunction_NodalSgs_Conserv_loc, &qf_sgs_apply));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)user->dm), PETSC_ERR_SUP, "Nodal SGS evaluation not available for chosen state variable");
  }

  PetscCallCeed(ceed, CeedQFunctionSetContext(qf_sgs_apply, sgs_dd_setup_data->ifunction_qfctx));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_apply, "q", num_comp_q, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_apply, "qdata", num_comp_qd, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_apply, "km_sgs", sgs_dd_data->num_comp_sgs, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_sgs_apply, "Grad_v", num_comp_q * dim, CEED_EVAL_GRAD));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_sgs_apply, NULL, NULL, &op_sgs_apply));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_apply, "q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_apply, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_NONE, ceed_data->q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_apply, "km_sgs", sgs_dd_setup_data->elem_restr_sgs, basis_sgs, sgs_dd_data->sgs_nodal_ceed));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_apply, "Grad_v", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));

  PetscCall(
      OperatorApplyContextCreate(user->dm, user->dm, ceed, op_sgs_apply, user->q_ceed, user->g_ceed, NULL, NULL, &sgs_dd_data->op_sgs_apply_ctx));

  PetscCallCeed(ceed, CeedOperatorDestroy(&op_sgs_apply));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_sgs_apply));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Calculate and add data-driven SGS residual to the global residual
PetscErrorCode SgsDDApplyIFunction(User user, const Vec Q_loc, Vec G_loc) {
  SgsDDData    sgs_dd_data = user->sgs_dd_data;
  Vec          VelocityGradient, SGSNodal_loc;
  PetscMemType sgs_nodal_mem_type;

  PetscFunctionBeginUser;
  PetscCall(DMGetGlobalVector(user->grad_velo_proj->dm, &VelocityGradient));
  PetscCall(VelocityGradientProjectionApply(user->grad_velo_proj, Q_loc, VelocityGradient));

  // -- Compute Nodal SGS tensor
  PetscCall(DMGetLocalVector(sgs_dd_data->dm_sgs, &SGSNodal_loc));
  PetscCall(sgs_dd_data->sgs_nodal_eval(user, Q_loc, VelocityGradient, SGSNodal_loc));

  // -- Compute contribution of the SGS stress
  PetscCall(VecPetscToCeed(SGSNodal_loc, &sgs_nodal_mem_type, sgs_dd_data->sgs_nodal_ceed));  // sgs_nodal_ceed is an implicit input
  PetscCall(ApplyAddCeedOperatorLocalToLocal(Q_loc, G_loc, sgs_dd_data->op_sgs_apply_ctx));

  // -- Return local SGS vector
  PetscCall(VecCeedToPetsc(sgs_dd_data->sgs_nodal_ceed, sgs_nodal_mem_type, SGSNodal_loc));
  PetscCall(DMRestoreLocalVector(sgs_dd_data->dm_sgs, &SGSNodal_loc));
  PetscCall(DMRestoreGlobalVector(user->grad_velo_proj->dm, &VelocityGradient));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief B = A^T, A is NxM, B is MxN
static PetscErrorCode TransposeMatrix(const PetscScalar *A, PetscScalar *B, const PetscInt N, const PetscInt M) {
  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < N; i++) {
    for (PetscInt j = 0; j < M; j++) {
      B[j * N + i] = A[i * M + j];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Read neural network coefficients from file and put into context struct
static PetscErrorCode SgsDDContextFill(MPI_Comm comm, char data_dir[PETSC_MAX_PATH_LEN], SgsDDContext *psgsdd_ctx) {
  SgsDDContext sgsdd_ctx;
  PetscInt     num_inputs = (*psgsdd_ctx)->num_inputs, num_outputs = (*psgsdd_ctx)->num_outputs, num_neurons = (*psgsdd_ctx)->num_neurons;
  char         file_path[PETSC_MAX_PATH_LEN];
  PetscScalar *temp;

  PetscFunctionBeginUser;
  {
    SgsDDContext sgsdd_temp;
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
  PetscCall(PhastaDatFileReadToArrayReal(comm, file_path, &sgsdd_ctx->data[sgsdd_ctx->offsets.bias1]));
  PetscCall(PetscSNPrintf(file_path, sizeof file_path, "%s/%s", data_dir, "b2.dat"));
  PetscCall(PhastaDatFileReadToArrayReal(comm, file_path, &sgsdd_ctx->data[sgsdd_ctx->offsets.bias2]));
  PetscCall(PetscSNPrintf(file_path, sizeof file_path, "%s/%s", data_dir, "OutScaling.dat"));
  PetscCall(PhastaDatFileReadToArrayReal(comm, file_path, &sgsdd_ctx->data[sgsdd_ctx->offsets.out_scaling]));

  {
    PetscCall(PetscMalloc1(num_inputs * num_neurons, &temp));
    PetscCall(PetscSNPrintf(file_path, sizeof file_path, "%s/%s", data_dir, "w1.dat"));
    PetscCall(PhastaDatFileReadToArrayReal(comm, file_path, temp));
    PetscCall(TransposeMatrix(temp, &sgsdd_ctx->data[sgsdd_ctx->offsets.weight1], num_inputs, num_neurons));
    PetscCall(PetscFree(temp));
  }
  {
    PetscCall(PetscMalloc1(num_outputs * num_neurons, &temp));
    PetscCall(PetscSNPrintf(file_path, sizeof file_path, "%s/%s", data_dir, "w2.dat"));
    PetscCall(PhastaDatFileReadToArrayReal(comm, file_path, temp));
    PetscCall(TransposeMatrix(temp, &sgsdd_ctx->data[sgsdd_ctx->offsets.weight2], num_neurons, num_outputs));
    PetscCall(PetscFree(temp));
  }

  PetscCall(PetscFree(*psgsdd_ctx));
  *psgsdd_ctx = sgsdd_ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SgsDDSetup(Ceed ceed, User user, CeedData ceed_data, ProblemData problem) {
  PetscReal                alpha = 0;
  SgsDDContext             sgsdd_ctx;
  MPI_Comm                 comm                           = user->comm;
  char                     sgs_dd_dir[PETSC_MAX_PATH_LEN] = "./dd_sgs_parameters";
  SgsDDSetupData           sgs_dd_setup_data;
  PetscBool                use_fused;
  NewtonianIdealGasContext gas;

  PetscFunctionBeginUser;
  PetscCall(VelocityGradientProjectionSetup(ceed, user, ceed_data, problem, user->phys->state_var, ceed_data->elem_restr_q, ceed_data->basis_q,
                                            &user->grad_velo_proj));

  PetscCall(PetscNew(&user->sgs_dd_data));
  user->sgs_dd_data->num_comp_inputs  = 6;
  user->sgs_dd_data->num_comp_outputs = 6;

  use_fused = PETSC_TRUE;
  PetscOptionsBegin(comm, NULL, "SGS Data-Driven Model Options", NULL);
  PetscCall(PetscOptionsReal("-sgs_model_dd_leakyrelu_alpha", "Slope parameter for Leaky ReLU activation function", NULL, alpha, &alpha, NULL));
  PetscCall(PetscOptionsString("-sgs_model_dd_parameter_dir", "Path to directory with model parameters (weights, biases, etc.)", NULL, sgs_dd_dir,
                               sgs_dd_dir, sizeof(sgs_dd_dir), NULL));
  PetscCall(
      PetscOptionsBool("-sgs_model_dd_use_fused", "Use the fused SGS DD model evaluation instead of sequential", NULL, use_fused, &use_fused, NULL));
  PetscOptionsEnd();

  PetscCall(PetscNew(&sgsdd_ctx));
  sgsdd_ctx->num_layers  = 1;
  sgsdd_ctx->num_inputs  = 6;
  sgsdd_ctx->num_outputs = 6;
  sgsdd_ctx->num_neurons = 20;
  sgsdd_ctx->alpha       = alpha;

  PetscCall(SgsDDContextFill(comm, sgs_dd_dir, &sgsdd_ctx));

  // -- Create DM for storing SGS tensor at nodes
  PetscCall(SgsDDCreateDM(user->dm, &user->sgs_dd_data->dm_sgs, user->app_ctx->degree, user->app_ctx->q_extra, &user->sgs_dd_data->num_comp_sgs));

  PetscCall(PetscNew(&sgs_dd_setup_data));

  PetscCallCeed(ceed, CeedQFunctionContextGetDataRead(problem->apply_vol_ifunction.qfunction_context, CEED_MEM_HOST, &gas));
  sgsdd_ctx->gas = *gas;
  PetscCallCeed(ceed, CeedQFunctionContextRestoreDataRead(problem->apply_vol_ifunction.qfunction_context, &gas));
  PetscCallCeed(ceed, CeedQFunctionContextCreate(user->ceed, &sgs_dd_setup_data->sgsdd_qfctx));
  PetscCallCeed(ceed,
                CeedQFunctionContextSetData(sgs_dd_setup_data->sgsdd_qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sgsdd_ctx->total_bytes, sgsdd_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(sgs_dd_setup_data->sgsdd_qfctx, CEED_MEM_HOST, FreeContextPetsc));

  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(problem->apply_vol_ifunction.qfunction_context, &sgs_dd_setup_data->ifunction_qfctx));

  // -- Compute and store anisotropy tensor
  PetscCall(GridAnisotropyTensorProjectionSetupApply(ceed, user, ceed_data, &sgs_dd_setup_data->elem_restr_grid_aniso,
                                                     &sgs_dd_setup_data->grid_aniso_ceed));

  // -- Create Nodal Evaluation Operator
  if (use_fused) PetscCall(SgsDDSetupNodalEvaluation_Fused(ceed, user, ceed_data, sgs_dd_setup_data));
  else PetscCall(SgsDDSetupNodalEvaluation_Sequential(ceed, user, ceed_data, sgs_dd_setup_data));

  // -- Create Operator to evalutate residual of SGS stress
  PetscCall(SgsSetupNodalIFunction(ceed, user, ceed_data, sgs_dd_setup_data));

  PetscCall(SgsDDSetupDataDestroy(sgs_dd_setup_data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SgsDDDataDestroy(SgsDDData sgs_dd_data) {
  PetscFunctionBeginUser;
  if (!sgs_dd_data) PetscFunctionReturn(PETSC_SUCCESS);
  Ceed ceed = sgs_dd_data->op_sgs_apply_ctx->ceed;

  PetscCallCeed(ceed, CeedVectorDestroy(&sgs_dd_data->sgs_nodal_ceed));
  PetscCallCeed(ceed, CeedVectorDestroy(&sgs_dd_data->grad_velo_ceed));
  PetscCall(OperatorApplyContextDestroy(sgs_dd_data->op_nodal_evaluation_ctx));
  PetscCall(OperatorApplyContextDestroy(sgs_dd_data->op_sgs_apply_ctx));
  PetscCall(OperatorApplyContextDestroy(sgs_dd_data->op_nodal_dd_inputs_ctx));
  PetscCall(OperatorApplyContextDestroy(sgs_dd_data->op_nodal_dd_outputs_ctx));
  PetscCall(DMDestroy(&sgs_dd_data->dm_sgs));
  PetscCall(DMDestroy(&sgs_dd_data->dm_dd_inputs));
  PetscCall(DMDestroy(&sgs_dd_data->dm_dd_outputs));
  if (sgs_dd_data->sgs_nodal_inference_ctx) PetscCall(sgs_dd_data->sgs_nodal_inference_ctx_destroy(sgs_dd_data->sgs_nodal_inference_ctx));
  PetscCall(PetscFree(sgs_dd_data));
  PetscFunctionReturn(PETSC_SUCCESS);
}
