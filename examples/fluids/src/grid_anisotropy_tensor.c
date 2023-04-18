// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
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
  OperatorApplyContext mass_matop_ctx, l2_rhs_ctx;
  CeedOperator         op_rhs_assemble, op_mass;
  CeedQFunction        qf_rhs_assemble, qf_mass;
  CeedBasis            basis_grid_aniso;
  PetscInt             dim, q_data_size, num_qpts_1d, num_nodes_1d;
  MPI_Comm             comm = PetscObjectComm((PetscObject)user->dm);
  KSP                  ksp;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&grid_aniso_proj));

  // -- Create DM for Anisotropic tensor L^2 projection
  grid_aniso_proj->num_comp = 7;
  PetscCall(DMClone(user->dm, &grid_aniso_proj->dm));
  PetscCall(DMGetDimension(grid_aniso_proj->dm, &dim));
  PetscCall(PetscObjectSetName((PetscObject)grid_aniso_proj->dm, "Grid Anisotropy Tensor Projection"));

  {  // -- Setup DM
    PetscFE      fe;
    PetscSection section;
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
  CeedBasisGetNumQuadraturePoints1D(ceed_data->basis_q, &num_qpts_1d);
  CeedBasisGetNumNodes1D(ceed_data->basis_q, &num_nodes_1d);
  CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_qd_i, &q_data_size);

  PetscCall(
      GetRestrictionForDomain(ceed, grid_aniso_proj->dm, 0, 0, 0, 0, num_qpts_1d, grid_aniso_proj->num_comp, elem_restr_grid_aniso, NULL, NULL));
  CeedBasisCreateTensorH1Lagrange(ceed, dim, grid_aniso_proj->num_comp, num_nodes_1d, num_qpts_1d, CEED_GAUSS, &basis_grid_aniso);

  // -- Build RHS operator
  CeedQFunctionCreateInterior(ceed, 1, AnisotropyTensorProjection, AnisotropyTensorProjection_loc, &qf_rhs_assemble);
  CeedQFunctionAddInput(qf_rhs_assemble, "qdata", q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_rhs_assemble, "v", grid_aniso_proj->num_comp, CEED_EVAL_INTERP);

  CeedOperatorCreate(ceed, qf_rhs_assemble, NULL, NULL, &op_rhs_assemble);
  CeedOperatorSetField(op_rhs_assemble, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_COLLOCATED, ceed_data->q_data);
  CeedOperatorSetField(op_rhs_assemble, "v", *elem_restr_grid_aniso, basis_grid_aniso, CEED_VECTOR_ACTIVE);

  PetscCall(OperatorApplyContextCreate(user->dm, grid_aniso_proj->dm, ceed, op_rhs_assemble, CEED_VECTOR_NONE, NULL, NULL, NULL, &l2_rhs_ctx));

  // -- Build Mass Operator
  PetscCall(CreateMassQFunction(ceed, grid_aniso_proj->num_comp, q_data_size, &qf_mass));
  CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass);
  CeedOperatorSetField(op_mass, "u", *elem_restr_grid_aniso, basis_grid_aniso, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_COLLOCATED, ceed_data->q_data);
  CeedOperatorSetField(op_mass, "v", *elem_restr_grid_aniso, basis_grid_aniso, CEED_VECTOR_ACTIVE);

  {  // -- Setup KSP for L^2 projection
    Mat mat_mass;
    PetscCall(OperatorApplyContextCreate(grid_aniso_proj->dm, grid_aniso_proj->dm, ceed, op_mass, NULL, NULL, NULL, NULL, &mass_matop_ctx));
    PetscCall(CreateMatShell_Ceed(mass_matop_ctx, &mat_mass));

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
    Vec Grid_Anisotropy, grid_anisotropy_loc;

    // Get L^2 Projection RHS
    PetscCall(DMGetGlobalVector(grid_aniso_proj->dm, &Grid_Anisotropy));

    PetscCall(ApplyCeedOperatorLocalToGlobal(NULL, Grid_Anisotropy, l2_rhs_ctx));

    // Solve projection problem
    PetscCall(KSPSolve(ksp, Grid_Anisotropy, Grid_Anisotropy));

    // Copy anisotropy tensor data to CeedVector
    PetscCall(DMGetLocalVector(grid_aniso_proj->dm, &grid_anisotropy_loc));
    CeedElemRestrictionCreateVector(*elem_restr_grid_aniso, grid_aniso_vector, NULL);
    PetscCall(DMGlobalToLocal(grid_aniso_proj->dm, Grid_Anisotropy, INSERT_VALUES, grid_anisotropy_loc));
    PetscCall(VecCopyP2C(grid_anisotropy_loc, *grid_aniso_vector));
    PetscCall(DMRestoreLocalVector(grid_aniso_proj->dm, &grid_anisotropy_loc));
    PetscCall(DMRestoreGlobalVector(grid_aniso_proj->dm, &Grid_Anisotropy));
  }

  // -- Cleanup
  PetscCall(NodalProjectionDataDestroy(grid_aniso_proj));
  PetscCall(OperatorApplyContextDestroy(l2_rhs_ctx));
  CeedQFunctionDestroy(&qf_rhs_assemble);
  CeedQFunctionDestroy(&qf_mass);
  CeedBasisDestroy(&basis_grid_aniso);
  CeedOperatorDestroy(&op_rhs_assemble);
  CeedOperatorDestroy(&op_mass);
  PetscCall(KSPDestroy(&ksp));
  PetscFunctionReturn(0);
}

PetscErrorCode GridAnisotropyTensorCalculateCollocatedVector(Ceed ceed, User user, CeedData ceed_data, CeedElemRestriction *elem_restr_grid_aniso,
                                                             CeedVector *aniso_colloc_ceed, PetscInt *num_comp_grid_aniso) {
  PetscInt      dim, q_data_size, num_qpts_1d, num_nodes_1d, loc_num_elem;
  CeedQFunction qf_colloc;
  CeedOperator  op_colloc;
  CeedBasis     basis_grid_aniso;

  PetscFunctionBeginUser;
  // -- Get Pre-requisite things
  *num_comp_grid_aniso = 7;
  PetscCall(DMGetDimension(user->dm, &dim));
  CeedBasisGetNumQuadraturePoints1D(ceed_data->basis_q, &num_qpts_1d);
  CeedBasisGetNumNodes1D(ceed_data->basis_q, &num_nodes_1d);
  CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_qd_i, &q_data_size);

  PetscCall(GetRestrictionForDomain(ceed, user->dm, 0, 0, 0, 0, num_qpts_1d, *num_comp_grid_aniso, NULL, NULL, elem_restr_grid_aniso));

  CeedInt Q_dim = CeedIntPow(num_qpts_1d, dim);
  CeedElemRestrictionGetNumElements(ceed_data->elem_restr_q, &loc_num_elem);
  CeedElemRestrictionCreateStrided(ceed, loc_num_elem, Q_dim, *num_comp_grid_aniso, *num_comp_grid_aniso * loc_num_elem * Q_dim, CEED_STRIDES_BACKEND,
                                   elem_restr_grid_aniso);

  CeedBasisCreateTensorH1Lagrange(ceed, dim, *num_comp_grid_aniso, num_nodes_1d, num_qpts_1d, CEED_GAUSS, &basis_grid_aniso);

  // -- Build collocation operator
  CeedQFunctionCreateInterior(ceed, 1, AnisotropyTensorCollocate, AnisotropyTensorCollocate_loc, &qf_colloc);
  CeedQFunctionAddInput(qf_colloc, "qdata", q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_colloc, "v", *num_comp_grid_aniso, CEED_EVAL_NONE);

  CeedOperatorCreate(ceed, qf_colloc, NULL, NULL, &op_colloc);
  CeedOperatorSetField(op_colloc, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_COLLOCATED, ceed_data->q_data);
  CeedOperatorSetField(op_colloc, "v", *elem_restr_grid_aniso, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedOperatorSetNumQuadraturePoints(op_colloc, CeedIntPow(num_qpts_1d, dim));

  CeedElemRestrictionCreateVector(*elem_restr_grid_aniso, aniso_colloc_ceed, NULL);

  CeedOperatorApply(op_colloc, CEED_VECTOR_NONE, *aniso_colloc_ceed, CEED_REQUEST_IMMEDIATE);

  PetscFunctionReturn(0);
}
