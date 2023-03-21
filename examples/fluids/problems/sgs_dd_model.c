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
} *SGS_DD_ModelSetupData;

PetscErrorCode SGS_DD_ModelSetupDataDestroy(SGS_DD_ModelSetupData sgs_dd_setup_data) {
  PetscFunctionBeginUser;

  CeedElemRestrictionDestroy(&sgs_dd_setup_data->elem_restr_grid_aniso);
  CeedElemRestrictionDestroy(&sgs_dd_setup_data->elem_restr_sgs);
  CeedBasisDestroy(&sgs_dd_setup_data->basis_grid_aniso);
  CeedVectorDestroy(&sgs_dd_setup_data->grid_aniso_ceed);

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
  PetscCall(GridAnisotropyTensorProjectionSetupApply(ceed, user, ceed_data, problem, &sgs_dd_setup_data->elem_restr_grid_aniso, &sgs_dd_setup_data->basis_grid_aniso, &sgs_dd_setup_data->grid_aniso_ceed));

  PetscCall(SGS_DD_ModelSetupDataDestroy(sgs_dd_setup_data));
  PetscFunctionReturn(0);
}

PetscErrorCode SGS_DD_DataDestroy(SGS_DD_Data sgs_dd_data) {
  PetscFunctionBeginUser;

  if (!sgs_dd_data) PetscFunctionReturn(0);

  PetscCall(DMDestroy(&sgs_dd_data->dm_sgs));

  PetscCall(PetscFree(sgs_dd_data));

  PetscFunctionReturn(0);
}
