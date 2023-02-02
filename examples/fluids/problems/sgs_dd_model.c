// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../qfunctions/sgs_dd_model.h"

#include "../navierstokes.h"

typedef struct {
  CeedElemRestriction elem_restr_anisotropy;
  CeedBasis           basis_anisotropy;
  CeedVector          anisotropy_ceed;
} *SGS_DD_ModelSetupData;

PetscErrorCode SGS_DD_ModelSetupDataDestroy(SGS_DD_ModelSetupData sgs_dd_setup_data) {
  PetscFunctionBeginUser;

  CeedElemRestrictionDestroy(&sgs_dd_setup_data->elem_restr_anisotropy);
  CeedBasisDestroy(&sgs_dd_setup_data->basis_anisotropy);
  CeedVectorDestroy(&sgs_dd_setup_data->anisotropy_ceed);

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

  // -- Create DM for Anisotropic tensor L^2 projection
  sgs_dd_data->num_comp_aniso = 7;

  PetscCall(DMClone(user->dm, &sgs_dd_data->dm_anisotropy));
  PetscCall(DMGetDimension(sgs_dd_data->dm_anisotropy, &dim));
  PetscCall(PetscObjectSetName((PetscObject)sgs_dd_data->dm_anisotropy, "Anisotropy Tensor Projection"));

  PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, dim, sgs_dd_data->num_comp_aniso, PETSC_FALSE, degree, PETSC_DECIDE, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "Anisotropy Tensor Projection"));
  PetscCall(DMAddField(sgs_dd_data->dm_anisotropy, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(sgs_dd_data->dm_anisotropy));
  PetscCall(DMPlexSetClosurePermutationTensor(sgs_dd_data->dm_anisotropy, PETSC_DETERMINE, NULL));

  PetscCall(DMGetLocalSection(sgs_dd_data->dm_anisotropy, &section));
  PetscCall(PetscSectionSetFieldName(section, 0, ""));
  PetscCall(PetscSectionSetComponentName(section, 0, 0, "KMAnisotropyTensorXX"));
  PetscCall(PetscSectionSetComponentName(section, 0, 1, "KMAnisotropyTensorYY"));
  PetscCall(PetscSectionSetComponentName(section, 0, 2, "KMAnisotropyTensorZZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 3, "KMAnisotropyTensorYZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 4, "KMAnisotropyTensorXZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 5, "KMAnisotropyTensorXY"));
  PetscCall(PetscSectionSetComponentName(section, 0, 6, "AnisotropyTensorFrobNorm"));

  PetscCall(PetscFEDestroy(&fe));

  user->sgs_dd_data = sgs_dd_data;

  PetscFunctionReturn(0);
};

PetscErrorCode AnisotropyTensorProjectionSetupApply(Ceed ceed, User user, CeedData ceed_data, ProblemData *problem,
                                                    SGS_DD_ModelSetupData sgs_dd_setup_data) {
  SGS_DD_Data         sgs_dd_data = user->sgs_dd_data;
  MatopApplyContext   mass_matop_ctx, l2_rhs_ctx;
  CeedOperator        op_rhs_assemble, op_mass;
  CeedQFunction       qf_rhs_assemble, qf_mass;
  CeedBasis           basis_anisotropy;
  CeedVector          q_ceed, rhs_ceed, mass_output;
  CeedElemRestriction elem_restr_anisotropy;
  PetscInt            dim, q_data_size, num_qpts_1d, num_nodes_1d;
  MPI_Comm            comm = PetscObjectComm((PetscObject)sgs_dd_data->dm_anisotropy);
  KSP                 ksp;
  PetscFunctionBeginUser;

  // -- Get Pre-requisite things
  PetscCall(DMGetDimension(sgs_dd_data->dm_anisotropy, &dim));
  CeedBasisGetNumQuadraturePoints1D(ceed_data->basis_q, &num_qpts_1d);
  CeedBasisGetNumNodes1D(ceed_data->basis_q, &num_nodes_1d);
  CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_qd_i, &q_data_size);

  PetscCall(GetRestrictionForDomain(ceed, sgs_dd_data->dm_anisotropy, 0, 0, 0, num_qpts_1d, 0, &elem_restr_anisotropy, NULL, NULL));

  CeedBasisCreateTensorH1Lagrange(ceed, dim, sgs_dd_data->num_comp_aniso, num_nodes_1d, num_qpts_1d, CEED_GAUSS, &basis_anisotropy);

  // -- Build RHS operator
  CeedQFunctionCreateInterior(ceed, 1, AnisotropyTensorProjection, AnisotropyTensorProjection_loc, &qf_rhs_assemble);

  CeedQFunctionAddInput(qf_rhs_assemble, "qdata", q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_rhs_assemble, "v", sgs_dd_data->num_comp_aniso, CEED_EVAL_INTERP);

  CeedOperatorCreate(ceed, qf_rhs_assemble, NULL, NULL, &op_rhs_assemble);
  CeedOperatorSetField(op_rhs_assemble, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_COLLOCATED, ceed_data->q_data);
  CeedOperatorSetField(op_rhs_assemble, "v", elem_restr_anisotropy, basis_anisotropy, CEED_VECTOR_ACTIVE);

  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &q_ceed, NULL);
  CeedElemRestrictionCreateVector(elem_restr_anisotropy, &rhs_ceed, NULL);

  PetscCall(
      MatopApplyContextCreate(user->dm, sgs_dd_data->dm_anisotropy, ceed, op_rhs_assemble, ceed_data->q_data, rhs_ceed, NULL, NULL, &l2_rhs_ctx));

  // -- Build Mass Operator
  PetscCall(CreateMassQFunction(ceed, sgs_dd_data->num_comp_aniso, q_data_size, &qf_mass));
  CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass);
  CeedOperatorSetField(op_mass, "u", elem_restr_anisotropy, basis_anisotropy, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_COLLOCATED, ceed_data->q_data);
  CeedOperatorSetField(op_mass, "v", elem_restr_anisotropy, basis_anisotropy, CEED_VECTOR_ACTIVE);

  {  // -- Setup KSP for L^2 projection with consistent mass operator
    PetscInt l_size, g_size;
    Mat      mat_mass;
    VecType  vec_type;
    Vec      M_inv;

    PetscCall(DMGetGlobalVector(sgs_dd_data->dm_anisotropy, &M_inv));
    PetscCall(VecGetLocalSize(M_inv, &l_size));
    PetscCall(VecGetSize(M_inv, &g_size));
    PetscCall(VecGetType(M_inv, &vec_type));
    PetscCall(DMRestoreGlobalVector(sgs_dd_data->dm_anisotropy, &M_inv));

    CeedElemRestrictionCreateVector(elem_restr_anisotropy, &mass_output, NULL);
    PetscCall(MatopApplyContextCreate(sgs_dd_data->dm_anisotropy, sgs_dd_data->dm_anisotropy, ceed, op_mass, rhs_ceed, mass_output, NULL, NULL,
                                      &mass_matop_ctx));
    CeedVectorDestroy(&mass_output);

    PetscCall(MatCreateShell(comm, l_size, l_size, g_size, g_size, mass_matop_ctx, &mat_mass));
    PetscCall(MatShellSetContextDestroy(mat_mass, (PetscErrorCode(*)(void *))MatopApplyContextDestroy));
    PetscCall(MatShellSetOperation(mat_mass, MATOP_MULT, (void (*)(void))MatMult_Ceed));
    PetscCall(MatShellSetOperation(mat_mass, MATOP_GET_DIAGONAL, (void (*)(void))MatGetDiag_Ceed));
    PetscCall(MatShellSetVecType(mat_mass, vec_type));

    PetscCall(KSPCreate(comm, &ksp));
    PetscCall(KSPSetOptionsPrefix(ksp, "anisotropy_tensor_projection_"));
    {
      PC pc;
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCSetType(pc, PCJACOBI));
      PetscCall(PCJacobiSetType(pc, PC_JACOBI_DIAGONAL));
      PetscCall(KSPSetType(ksp, KSPCG));
      PetscCall(KSPSetNormType(ksp, KSP_NORM_NATURAL));
      PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    }
    PetscCall(KSPSetOperators(ksp, mat_mass, mat_mass));
    PetscCall(KSPSetFromOptions(ksp));
  }

  {  // -- Project anisotropy data and store in CeedVector
    Vec                Anisotropy, anisotropy_loc;
    PetscMemType       mem_type;
    const PetscScalar *anisotropy;

    // Get L^2 Projection RHS
    PetscCall(DMGetGlobalVector(sgs_dd_data->dm_anisotropy, &Anisotropy));
    PetscCall(DMGetLocalVector(sgs_dd_data->dm_anisotropy, &anisotropy_loc));

    PetscCall(VecP2C(Anisotropy, &mem_type, l2_rhs_ctx->y_ceed));
    CeedOperatorApply(l2_rhs_ctx->op, CEED_VECTOR_NONE, l2_rhs_ctx->y_ceed, CEED_REQUEST_IMMEDIATE);
    PetscCall(VecC2P(l2_rhs_ctx->y_ceed, mem_type, Anisotropy));

    // Solve projection problem
    PetscCall(KSPSolve(ksp, Anisotropy, Anisotropy));

    // Get anisotropy tensor data into CeedVector
    CeedElemRestrictionCreateVector(elem_restr_anisotropy, &sgs_dd_setup_data->anisotropy_ceed, NULL);
    PetscCall(DMGlobalToLocal(sgs_dd_data->dm_anisotropy, Anisotropy, INSERT_VALUES, anisotropy_loc));
    PetscCall(VecGetArrayReadAndMemType(anisotropy_loc, &anisotropy, &mem_type));
    CeedVectorSetArray(sgs_dd_setup_data->anisotropy_ceed, MemTypeP2C(mem_type), CEED_COPY_VALUES, (CeedScalar *)anisotropy);

    PetscCall(VecRestoreArrayReadAndMemType(anisotropy_loc, &anisotropy));
    PetscCall(DMRestoreLocalVector(sgs_dd_data->dm_anisotropy, &anisotropy_loc));
    PetscCall(DMRestoreGlobalVector(sgs_dd_data->dm_anisotropy, &Anisotropy));
  }

  CeedElemRestrictionReferenceCopy(elem_restr_anisotropy, &sgs_dd_setup_data->elem_restr_anisotropy);
  CeedBasisReferenceCopy(basis_anisotropy, &sgs_dd_setup_data->basis_anisotropy);

  // -- Cleanup
  PetscCall(MatopApplyContextDestroy(l2_rhs_ctx));
  CeedVectorDestroy(&q_ceed);
  CeedVectorDestroy(&rhs_ceed);
  CeedBasisDestroy(&basis_anisotropy);
  CeedElemRestrictionDestroy(&elem_restr_anisotropy);
  CeedQFunctionDestroy(&qf_rhs_assemble);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_rhs_assemble);
  CeedOperatorDestroy(&op_mass);
  PetscCall(KSPDestroy(&ksp));
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
  PetscReal             alpha;
  SGS_DDModelContext    sgsdd_ctx;
  MPI_Comm              comm                           = user->comm;
  char                  sgs_dd_dir[PETSC_MAX_PATH_LEN] = "./dd_sgs_data";
  SGS_DD_ModelSetupData sgs_dd_setup_data;
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

  // -- Compute and store anisotropy tensor
  PetscCall(AnisotropyTensorProjectionSetupApply(ceed, user, ceed_data, problem, sgs_dd_setup_data));

  PetscCall(SGS_DD_ModelSetupDataDestroy(sgs_dd_setup_data));
  PetscFunctionReturn(0);
}

PetscErrorCode SGS_DD_DataDestroy(SGS_DD_Data sgs_dd_data) {
  PetscFunctionBeginUser;

  if (!sgs_dd_data) PetscFunctionReturn(0);

  PetscCall(DMDestroy(&sgs_dd_data->dm_sgs));
  PetscCall(DMDestroy(&sgs_dd_data->dm_anisotropy));

  PetscCall(PetscFree(sgs_dd_data));

  PetscFunctionReturn(0);
}
