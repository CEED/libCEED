// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../qfunctions/grid_anisotropy_tensor.h"

#include "../navierstokes.h"

PetscErrorCode GridAnisotropyTensorProjection_CreateDM(User user, ProblemData *problem, PetscInt degree, SimpleBC bc) {
  PetscFunctionBeginUser;

  PetscFunctionReturn(0);
};

PetscErrorCode GridAnisotropyTensorProjectionSetupApply(Ceed ceed, User user, CeedData ceed_data, ProblemData *problem, CeedElemRestriction *elem_restr_grid_aniso, CeedBasis *basis_grid_aniso, CeedVector *grid_aniso_vector) {
  NodalProjectionData grid_aniso_proj;
  MatopApplyContext   mass_matop_ctx, l2_rhs_ctx;
  CeedOperator        op_rhs_assemble, op_mass;
  CeedQFunction       qf_rhs_assemble, qf_mass;
  CeedVector          q_ceed, rhs_ceed, mass_output;
  PetscInt            dim, q_data_size, num_qpts_1d, num_nodes_1d;
  MPI_Comm            comm = PetscObjectComm((PetscObject)user->dm);
  KSP                 ksp;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&grid_aniso_proj));

  // -- Create DM for Anisotropic tensor L^2 projection
  grid_aniso_proj->num_comp = 7;

  {
    PetscFE      fe;
    PetscSection section;
    PetscCall(DMClone(user->dm, &grid_aniso_proj->dm));
    PetscCall(DMGetDimension(grid_aniso_proj->dm, &dim));
    PetscCall(PetscObjectSetName((PetscObject)grid_aniso_proj->dm, "Grid Anisotropy Tensor Projection"));

    PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, dim, grid_aniso_proj->num_comp, PETSC_FALSE, user->app_ctx->degree, PETSC_DECIDE, &fe));
    PetscCall(PetscObjectSetName((PetscObject)fe, "Grid Anisotropy Tensor Projection"));
    PetscCall(DMAddField(grid_aniso_proj->dm, NULL, (PetscObject)fe));
    PetscCall(DMCreateDS(grid_aniso_proj->dm));
    PetscCall(DMPlexSetClosurePermutationTensor(grid_aniso_proj->dm, PETSC_DETERMINE, NULL));

    PetscCall(DMGetLocalSection(grid_aniso_proj->dm, &section));
    PetscCall(PetscSectionSetFieldName(section, 0, ""));
    PetscCall(PetscSectionSetComponentName(section, 0, 0, "KMGridAnisotropyTensorXX"));
    PetscCall(PetscSectionSetComponentName(section, 0, 1, "KMGridAnisotropyTensorYY"));
    PetscCall(PetscSectionSetComponentName(section, 0, 2, "KMGridAnisotropyTensorZZ"));
    PetscCall(PetscSectionSetComponentName(section, 0, 3, "KMGridAnisotropyTensorYZ"));
    PetscCall(PetscSectionSetComponentName(section, 0, 4, "KMGridAnisotropyTensorXZ"));
    PetscCall(PetscSectionSetComponentName(section, 0, 5, "KMGridAnisotropyTensorXY"));
    PetscCall(PetscSectionSetComponentName(section, 0, 6, "GridAnisotropyTensorFrobNorm"));

    PetscCall(PetscFEDestroy(&fe));
  }

  // -- Get Pre-requisite things
  PetscCall(DMGetDimension(grid_aniso_proj->dm, &dim));
  CeedBasisGetNumQuadraturePoints1D(ceed_data->basis_q, &num_qpts_1d);
  CeedBasisGetNumNodes1D(ceed_data->basis_q, &num_nodes_1d);
  CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_qd_i, &q_data_size);

  CeedElemRestriction elem_restr_grid_aniso_colloc;
  PetscCall(GetRestrictionForDomain(ceed, grid_aniso_proj->dm, 0, 0, 0, num_qpts_1d, grid_aniso_proj->num_comp, elem_restr_grid_aniso, NULL, &elem_restr_grid_aniso_colloc));

  CeedBasis basis_x_to_grid_aniso;
  CeedBasisCreateTensorH1Lagrange(ceed, dim, grid_aniso_proj->num_comp, num_nodes_1d, num_qpts_1d, CEED_GAUSS_LOBATTO, basis_grid_aniso);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 3, 2, num_qpts_1d, CEED_GAUSS_LOBATTO, &basis_x_to_grid_aniso);

  // -- Build RHS operator
  CeedQFunctionCreateInterior(ceed, 1, AnisotropyTensorProjection, AnisotropyTensorProjection_loc, &qf_rhs_assemble);


  CeedQFunctionAddInput(qf_rhs_assemble, "qdata", q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_rhs_assemble, "x", 3, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_rhs_assemble, "v", grid_aniso_proj->num_comp, CEED_EVAL_INTERP);

  CeedOperatorCreate(ceed, qf_rhs_assemble, NULL, NULL, &op_rhs_assemble);
  CeedOperatorSetField(op_rhs_assemble, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_COLLOCATED, ceed_data->q_data);
  CeedOperatorSetField(op_rhs_assemble, "x", ceed_data->elem_restr_x, basis_x_to_grid_aniso, ceed_data->x_coord);
  CeedOperatorSetField(op_rhs_assemble, "v", *elem_restr_grid_aniso, *basis_grid_aniso, CEED_VECTOR_ACTIVE);

  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &q_ceed, NULL);
  CeedElemRestrictionCreateVector(*elem_restr_grid_aniso, &rhs_ceed, NULL);

  PetscCall(
      MatopApplyContextCreate(user->dm, grid_aniso_proj->dm, ceed, op_rhs_assemble, ceed_data->q_data, rhs_ceed, NULL, NULL, &l2_rhs_ctx));

  // -- Build Mass Operator
  PetscCall(CreateMassQFunction(ceed, grid_aniso_proj->num_comp, q_data_size, &qf_mass));
  CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass);
  CeedOperatorSetField(op_mass, "u", *elem_restr_grid_aniso, *basis_grid_aniso, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_COLLOCATED, ceed_data->q_data);
  CeedOperatorSetField(op_mass, "v", *elem_restr_grid_aniso, *basis_grid_aniso, CEED_VECTOR_ACTIVE);

  {  // -- Setup KSP for L^2 projection with consistent mass operator
    PetscInt l_size, g_size;
    Mat      mat_mass;
    VecType  vec_type;
    Vec      M_inv;

    PetscCall(DMGetGlobalVector(grid_aniso_proj->dm, &M_inv));
    PetscCall(VecGetLocalSize(M_inv, &l_size));
    PetscCall(VecGetSize(M_inv, &g_size));
    PetscCall(VecGetType(M_inv, &vec_type));
    PetscCall(DMRestoreGlobalVector(grid_aniso_proj->dm, &M_inv));

    CeedElemRestrictionCreateVector(*elem_restr_grid_aniso, &mass_output, NULL);
    PetscCall(MatopApplyContextCreate(grid_aniso_proj->dm, grid_aniso_proj->dm, ceed, op_mass, rhs_ceed, mass_output, NULL, NULL,
                                      &mass_matop_ctx));
    CeedVectorDestroy(&mass_output);

    PetscCall(MatCreateShell(comm, l_size, l_size, g_size, g_size, mass_matop_ctx, &mat_mass));
    PetscCall(MatShellSetContextDestroy(mat_mass, (PetscErrorCode(*)(void *))MatopApplyContextDestroy));
    PetscCall(MatShellSetOperation(mat_mass, MATOP_MULT, (void (*)(void))MatMult_Ceed));
    PetscCall(MatShellSetOperation(mat_mass, MATOP_GET_DIAGONAL, (void (*)(void))MatGetDiag_Ceed));
    PetscCall(MatShellSetVecType(mat_mass, vec_type));

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
    PetscCall(KSPSetOperators(ksp, mat_mass, mat_mass));
    PetscCall(KSPSetFromOptions(ksp));
  }

  {  // -- Project anisotropy data and store in CeedVector
    Vec                Grid_Anisotropy, grid_anisotropy_loc;
    PetscMemType       mem_type;
    const PetscScalar *grid_anisotropy;

    // Get L^2 Projection RHS
    PetscCall(DMGetGlobalVector(grid_aniso_proj->dm, &Grid_Anisotropy));
    PetscCall(DMGetLocalVector(grid_aniso_proj->dm, &grid_anisotropy_loc));

    PetscCall(VecP2C(Grid_Anisotropy, &mem_type, l2_rhs_ctx->y_ceed));
    CeedOperatorApply(l2_rhs_ctx->op, CEED_VECTOR_NONE, l2_rhs_ctx->y_ceed, CEED_REQUEST_IMMEDIATE);
    PetscCall(VecC2P(l2_rhs_ctx->y_ceed, mem_type, Grid_Anisotropy));

    // Solve projection problem
    PetscCall(KSPSolve(ksp, Grid_Anisotropy, Grid_Anisotropy));

    // Get anisotropy tensor data into CeedVector
    CeedElemRestrictionCreateVector(*elem_restr_grid_aniso, grid_aniso_vector, NULL);
    PetscCall(DMGlobalToLocal(grid_aniso_proj->dm, Grid_Anisotropy, INSERT_VALUES, grid_anisotropy_loc));
    PetscCall(VecGetArrayReadAndMemType(grid_anisotropy_loc, &grid_anisotropy, &mem_type));
    CeedVectorSetArray(*grid_aniso_vector, MemTypeP2C(mem_type), CEED_COPY_VALUES, (CeedScalar *)grid_anisotropy);

    PetscCall(VecRestoreArrayReadAndMemType(grid_anisotropy_loc, &grid_anisotropy));
    PetscCall(DMRestoreLocalVector(grid_aniso_proj->dm, &grid_anisotropy_loc));
    PetscCall(DMRestoreGlobalVector(grid_aniso_proj->dm, &Grid_Anisotropy));
  }

  // -- Cleanup
  PetscCall(MatopApplyContextDestroy(l2_rhs_ctx));
  CeedVectorDestroy(&q_ceed);
  CeedVectorDestroy(&rhs_ceed);
  CeedQFunctionDestroy(&qf_rhs_assemble);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_rhs_assemble);
  CeedOperatorDestroy(&op_mass);
  PetscCall(KSPDestroy(&ksp));
  PetscFunctionReturn(0);
}

