// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../qfunctions/grid_anisotropy_tensor.h"

#include <petscdmplex.h>

#include "../navierstokes.h"

PetscErrorCode GridAnisotropyTensorProjectionSetupApply(Ceed ceed, User user, CeedData ceed_data, CeedElemRestriction *elem_restr_grid_aniso,
                                                        CeedVector *grid_aniso_vector) {
  NodalProjectionData  grid_aniso_proj;
  OperatorApplyContext l2_rhs_ctx;
  CeedOperator         op_rhs_assemble, op_mass;
  CeedQFunction        qf_rhs_assemble, qf_mass;
  CeedBasis            basis_grid_aniso;
  CeedInt              q_data_size;
  MPI_Comm             comm = PetscObjectComm((PetscObject)user->dm);
  KSP                  ksp;
  DMLabel              domain_label = NULL;
  PetscInt             label_value = 0, height = 0, dm_field = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&grid_aniso_proj));

  // -- Create DM for Anisotropic tensor L^2 projection
  grid_aniso_proj->num_comp = 7;
  PetscCall(DMClone(user->dm, &grid_aniso_proj->dm));
  PetscCall(PetscObjectSetName((PetscObject)grid_aniso_proj->dm, "Grid Anisotropy Tensor Projection"));

  {  // -- Setup DM
    PetscSection section;
    PetscCall(DMSetupByOrder_FEM(PETSC_TRUE, PETSC_TRUE, user->app_ctx->degree, 1, user->app_ctx->q_extra, 1, &grid_aniso_proj->num_comp,
                                 grid_aniso_proj->dm));

    PetscCall(DMGetLocalSection(grid_aniso_proj->dm, &section));
    PetscCall(PetscSectionSetFieldName(section, 0, ""));
    PetscCall(PetscSectionSetComponentName(section, 0, 0, "KMGridAnisotropyTensorXX"));
    PetscCall(PetscSectionSetComponentName(section, 0, 1, "KMGridAnisotropyTensorYY"));
    PetscCall(PetscSectionSetComponentName(section, 0, 2, "KMGridAnisotropyTensorZZ"));
    PetscCall(PetscSectionSetComponentName(section, 0, 3, "KMGridAnisotropyTensorYZ"));
    PetscCall(PetscSectionSetComponentName(section, 0, 4, "KMGridAnisotropyTensorXZ"));
    PetscCall(PetscSectionSetComponentName(section, 0, 5, "KMGridAnisotropyTensorXY"));
    PetscCall(PetscSectionSetComponentName(section, 0, 6, "GridAnisotropyTensorFrobNorm"));
  }

  // -- Get Pre-requisite things
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_qd_i, &q_data_size));

  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, grid_aniso_proj->dm, domain_label, label_value, height, dm_field, elem_restr_grid_aniso));
  PetscCall(CreateBasisFromPlex(ceed, grid_aniso_proj->dm, domain_label, label_value, height, dm_field, &basis_grid_aniso));

  // -- Build RHS operator
  PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, AnisotropyTensorProjection, AnisotropyTensorProjection_loc, &qf_rhs_assemble));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_assemble, "qdata", q_data_size, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_rhs_assemble, "v", grid_aniso_proj->num_comp, CEED_EVAL_INTERP));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_rhs_assemble, NULL, NULL, &op_rhs_assemble));
  PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_assemble, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_NONE, ceed_data->q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_assemble, "v", *elem_restr_grid_aniso, basis_grid_aniso, CEED_VECTOR_ACTIVE));

  PetscCall(OperatorApplyContextCreate(user->dm, grid_aniso_proj->dm, ceed, op_rhs_assemble, CEED_VECTOR_NONE, NULL, NULL, NULL, &l2_rhs_ctx));

  // -- Build Mass Operator
  PetscCall(CreateMassQFunction(ceed, grid_aniso_proj->num_comp, q_data_size, &qf_mass));
  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "u", *elem_restr_grid_aniso, basis_grid_aniso, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_NONE, ceed_data->q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "v", *elem_restr_grid_aniso, basis_grid_aniso, CEED_VECTOR_ACTIVE));

  {  // -- Setup KSP for L^2 projection
    Mat mat_mass;

    PetscCall(MatCeedCreate(grid_aniso_proj->dm, grid_aniso_proj->dm, op_mass, NULL, &mat_mass));

    PetscCall(KSPCreate(comm, &ksp));
    PetscCall(KSPSetOptionsPrefix(ksp, "grid_anisotropy_tensor_projection_"));
    {
      PC pc;
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCSetType(pc, PCJACOBI));
      PetscCall(PCJacobiSetType(pc, PC_JACOBI_DIAGONAL));
      PetscCall(KSPSetType(ksp, KSPCG));
      PetscCall(KSPSetNormType(ksp, KSP_NORM_NATURAL));
      PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    }
    PetscCall(KSPSetFromOptions_WithMatCeed(ksp, mat_mass));
    PetscCall(MatDestroy(&mat_mass));
  }

  {  // -- Project anisotropy data and store in CeedVector
    Vec Grid_Anisotropy, grid_anisotropy_loc;

    // Get L^2 Projection RHS
    PetscCall(DMGetGlobalVector(grid_aniso_proj->dm, &Grid_Anisotropy));

    PetscCall(ApplyCeedOperatorLocalToGlobal(NULL, Grid_Anisotropy, l2_rhs_ctx));

    // Solve projection problem
    PetscCall(KSPSolve(ksp, Grid_Anisotropy, Grid_Anisotropy));

    // Copy anisotropy tensor data to CeedVector
    PetscCall(DMGetLocalVector(grid_aniso_proj->dm, &grid_anisotropy_loc));
    PetscCallCeed(ceed, CeedElemRestrictionCreateVector(*elem_restr_grid_aniso, grid_aniso_vector, NULL));
    PetscCall(DMGlobalToLocal(grid_aniso_proj->dm, Grid_Anisotropy, INSERT_VALUES, grid_anisotropy_loc));
    PetscCall(VecCopyPetscToCeed(grid_anisotropy_loc, *grid_aniso_vector));
    PetscCall(DMRestoreLocalVector(grid_aniso_proj->dm, &grid_anisotropy_loc));
    PetscCall(DMRestoreGlobalVector(grid_aniso_proj->dm, &Grid_Anisotropy));
  }

  // -- Cleanup
  PetscCall(NodalProjectionDataDestroy(grid_aniso_proj));
  PetscCall(OperatorApplyContextDestroy(l2_rhs_ctx));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_rhs_assemble));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_mass));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_grid_aniso));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_rhs_assemble));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_mass));
  PetscCall(KSPDestroy(&ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GridAnisotropyTensorCalculateCollocatedVector(Ceed ceed, User user, CeedData ceed_data, CeedElemRestriction *elem_restr_grid_aniso,
                                                             CeedVector *aniso_colloc_ceed, PetscInt *num_comp_aniso) {
  CeedInt       q_data_size, num_nodes;
  CeedQFunction qf_colloc;
  CeedOperator  op_colloc;
  DMLabel       domain_label = NULL;
  PetscInt      label_value = 0, height = 0;

  PetscFunctionBeginUser;
  *num_comp_aniso = 7;
  PetscCallCeed(ceed, CeedBasisGetNumNodes(ceed_data->basis_q, &num_nodes));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_qd_i, &q_data_size));
  PetscCall(DMPlexCeedElemRestrictionQDataCreate(ceed, user->dm, domain_label, label_value, height, *num_comp_aniso, elem_restr_grid_aniso));

  // -- Build collocation operator
  PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, AnisotropyTensorCollocate, AnisotropyTensorCollocate_loc, &qf_colloc));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_colloc, "qdata", q_data_size, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_colloc, "v", *num_comp_aniso, CEED_EVAL_NONE));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_colloc, NULL, NULL, &op_colloc));
  PetscCallCeed(ceed, CeedOperatorSetField(op_colloc, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_NONE, ceed_data->q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_colloc, "v", *elem_restr_grid_aniso, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(*elem_restr_grid_aniso, aniso_colloc_ceed, NULL));

  PetscCallCeed(ceed, CeedOperatorApply(op_colloc, CEED_VECTOR_NONE, *aniso_colloc_ceed, CEED_REQUEST_IMMEDIATE));

  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_colloc));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_colloc));
  PetscFunctionReturn(PETSC_SUCCESS);
}
