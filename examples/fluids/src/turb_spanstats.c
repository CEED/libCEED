// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
/// @file
/// Functions for setting up and performing statistics collection

#include "../qfunctions/turb_spanstats.h"

#include <petscsf.h>

#include "../include/matops.h"
#include "../navierstokes.h"
#include "ceed/ceed.h"
#include "petscerror.h"
#include "petscmat.h"
#include "petscsys.h"
#include "petscvec.h"

PetscErrorCode CreateStatsDM(User user, ProblemData *problem, PetscInt degree, SimpleBC bc) {
  user->spanstats.num_comp_stats = 22;
  PetscReal domain_min[3], domain_max[3];
  PetscFunctionBeginUser;

  // Get spanwise length
  PetscCall(DMGetBoundingBox(user->dm, domain_min, domain_max));
  user->spanstats.span_width = domain_max[2] - domain_min[1];

  // Get DM from surface
  {
    DMLabel label;
    PetscCall(DMGetLabel(user->dm, "Face Sets", &label));
    PetscCall(DMPlexLabelComplete(user->dm, label));
    PetscCall(DMPlexFilter(user->dm, label, 1, &user->spanstats.dm));
    PetscCall(DMProjectCoordinates(user->spanstats.dm, NULL));  // Ensure that a coordinate FE exists
  }

  PetscCall(PetscObjectSetName((PetscObject)user->spanstats.dm, "Spanwise_Stats"));
  PetscCall(DMSetOptionsPrefix(user->spanstats.dm, "spanstats_"));
  PetscCall(DMSetFromOptions(user->spanstats.dm));
  PetscCall(DMViewFromOptions(user->spanstats.dm, NULL, "-dm_view"));  // -spanstats_dm_view
  {
    PetscFE fe;
    DMLabel label;

    PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, problem->dim - 1, user->spanstats.num_comp_stats, PETSC_FALSE, degree, PETSC_DECIDE, &fe));
    PetscCall(PetscObjectSetName((PetscObject)fe, "stats"));
    PetscCall(DMAddField(user->spanstats.dm, NULL, (PetscObject)fe));
    PetscCall(DMCreateDS(user->spanstats.dm));
    PetscCall(DMGetLabel(user->spanstats.dm, "Face Sets", &label));

    PetscCall(DMPlexSetClosurePermutationTensor(user->spanstats.dm, PETSC_DETERMINE, NULL));
    PetscCall(PetscFEDestroy(&fe));
  }

  PetscSection section;
  PetscCall(DMGetLocalSection(user->spanstats.dm, &section));
  PetscCall(PetscSectionSetFieldName(section, 0, ""));
  PetscCall(PetscSectionSetComponentName(section, 0, 0, "Mean Density"));
  PetscCall(PetscSectionSetComponentName(section, 0, 1, "Mean Pressure"));
  PetscCall(PetscSectionSetComponentName(section, 0, 2, "Mean Pressure Squared"));
  PetscCall(PetscSectionSetComponentName(section, 0, 3, "Mean Pressure Velocity X"));
  PetscCall(PetscSectionSetComponentName(section, 0, 4, "Mean Pressure Velocity Y"));
  PetscCall(PetscSectionSetComponentName(section, 0, 5, "Mean Pressure Velocity Z"));
  PetscCall(PetscSectionSetComponentName(section, 0, 6, "Mean Density Temperature"));
  PetscCall(PetscSectionSetComponentName(section, 0, 7, "Mean Density Temperature Flux X"));
  PetscCall(PetscSectionSetComponentName(section, 0, 8, "Mean Density Temperature Flux Y"));
  PetscCall(PetscSectionSetComponentName(section, 0, 9, "Mean Density Temperature Flux Z"));
  PetscCall(PetscSectionSetComponentName(section, 0, 10, "Mean Momentum X"));
  PetscCall(PetscSectionSetComponentName(section, 0, 11, "Mean Momentum Y"));
  PetscCall(PetscSectionSetComponentName(section, 0, 12, "Mean Momentum Z"));
  PetscCall(PetscSectionSetComponentName(section, 0, 13, "Mean Momentum Flux XX"));
  PetscCall(PetscSectionSetComponentName(section, 0, 14, "Mean Momentum Flux YY"));
  PetscCall(PetscSectionSetComponentName(section, 0, 15, "Mean Momentum Flux ZZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 16, "Mean Momentum Flux YZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 17, "Mean Momentum Flux XZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 18, "Mean Momentum Flux XY"));
  PetscCall(PetscSectionSetComponentName(section, 0, 19, "Mean Velocity X"));
  PetscCall(PetscSectionSetComponentName(section, 0, 20, "Mean Velocity Y"));
  PetscCall(PetscSectionSetComponentName(section, 0, 21, "Mean Velocity Z"));

  PetscFunctionReturn(0);
}

// Create CeedElemRestriction for collocated data based on associated CeedBasis and CeedElemRestriction
// Number of quadrature points is used from the CeedBasis, and number of elements is used from the CeedElemRestriction
PetscErrorCode CreateElemRestrColloc(Ceed ceed, CeedInt num_comp, CeedBasis basis, CeedElemRestriction elem_restr_base,
                                     CeedElemRestriction *elem_restr_collocated, CeedVector *l_vec, CeedVector *e_vec) {
  CeedInt num_elem_qpts, loc_num_elem;
  PetscFunctionBeginUser;

  CeedBasisGetNumQuadraturePoints(basis, &num_elem_qpts);
  CeedElemRestrictionGetNumElements(elem_restr_base, &loc_num_elem);

  const CeedInt strides[] = {num_comp, 1, num_elem_qpts * num_comp};
  CeedElemRestrictionCreateStrided(ceed, loc_num_elem, num_elem_qpts, num_comp, num_comp * loc_num_elem * num_elem_qpts, strides,
                                   elem_restr_collocated);
  CeedElemRestrictionCreateVector(*elem_restr_collocated, l_vec, e_vec);
  PetscFunctionReturn(0);
}

// Get coordinates of quadrature points
PetscErrorCode GetQuadratureCoords(Ceed ceed, DM dm, CeedElemRestriction elem_restr_x, CeedBasis basis_x, CeedVector x_coords, CeedVector *qx_coords,
                                   PetscInt *total_nqpnts) {
  CeedQFunction       qf_quad_coords;
  CeedOperator        op_quad_coords;
  PetscInt            num_comp_x, loc_num_elem, num_elem_qpts;
  CeedElemRestriction elem_restr_qx;
  PetscFunctionBeginUser;

  // Create Element Restriction and CeedVector for quadrature coordinates
  CeedBasisGetNumQuadraturePoints(basis_x, &num_elem_qpts);
  CeedElemRestrictionGetNumElements(elem_restr_x, &loc_num_elem);
  CeedElemRestrictionGetNumComponents(elem_restr_x, &num_comp_x);
  *total_nqpnts = num_elem_qpts * loc_num_elem;
  PetscCall(CreateElemRestrColloc(ceed, num_comp_x, basis_x, elem_restr_x, &elem_restr_qx, qx_coords, NULL));

  // Create QFunction
  CeedQFunctionCreateIdentity(ceed, num_comp_x, CEED_EVAL_INTERP, CEED_EVAL_NONE, &qf_quad_coords);

  // Create Operator
  CeedOperatorCreate(ceed, qf_quad_coords, NULL, NULL, &op_quad_coords);
  CeedOperatorSetField(op_quad_coords, "input", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_quad_coords, "output", elem_restr_qx, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedOperatorApply(op_quad_coords, x_coords, *qx_coords, CEED_REQUEST_IMMEDIATE);

  CeedQFunctionDestroy(&qf_quad_coords);
  CeedOperatorDestroy(&op_quad_coords);
  PetscFunctionReturn(0);
}

// Create PetscSF for child-to-parent communication
PetscErrorCode CreateStatsSF(Ceed ceed, CeedData ceed_data, DM parentdm, DM childdm, PetscSF statssf) {
  PetscInt   child_num_qpnts, parent_num_qpnts, num_comp_x;
  CeedVector child_qx_coords, parent_qx_coords;
  PetscReal *child_coords, *parent_coords;
  PetscFunctionBeginUser;

  // Assume that child and parent have the same number of components
  CeedBasisGetNumComponents(ceed_data->basis_x, &num_comp_x);
  const PetscInt num_comp_sf = num_comp_x - 1;  // Number of coord components used in the creation of the SF

  // Get quad_coords for child DM
  PetscCall(GetQuadratureCoords(ceed, childdm, ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord, &child_qx_coords, &child_num_qpnts));

  // Get quad_coords for parent DM
  PetscCall(GetQuadratureCoords(ceed, parentdm, ceed_data->spanstats.elem_restr_parent_x, ceed_data->spanstats.basis_x, ceed_data->spanstats.x_coord,
                                &parent_qx_coords, &parent_num_qpnts));

  // Remove z component of coordinates for matching
  {
    const PetscReal *child_quad_coords, *parent_quad_coords;

    CeedVectorGetArrayRead(child_qx_coords, CEED_MEM_HOST, &child_quad_coords);
    CeedVectorGetArrayRead(parent_qx_coords, CEED_MEM_HOST, &parent_quad_coords);

    PetscCall(PetscMalloc2(child_num_qpnts * 2, &child_coords, parent_num_qpnts * 2, &parent_coords));
    for (int i = 0; i < child_num_qpnts; i++) {
      child_coords[0 + i * num_comp_sf] = child_quad_coords[0 + i * num_comp_x];
      child_coords[1 + i * num_comp_sf] = child_quad_coords[1 + i * num_comp_x];
    }
    for (int i = 0; i < parent_num_qpnts; i++) {
      parent_coords[0 + i * num_comp_sf] = parent_quad_coords[0 + i * num_comp_x];
      parent_coords[1 + i * num_comp_sf] = parent_quad_coords[1 + i * num_comp_x];
    }
    CeedVectorRestoreArrayRead(child_qx_coords, &child_quad_coords);
    CeedVectorRestoreArrayRead(parent_qx_coords, &parent_quad_coords);
  }

  PetscCall(PetscSFSetGraphFromCoordinates(statssf, parent_num_qpnts, child_num_qpnts, num_comp_sf, 1e-12, parent_coords, child_coords));

  PetscCall(PetscSFViewFromOptions(statssf, NULL, "-spanstats_sf_view"));

  PetscCall(PetscFree2(child_coords, parent_coords));
  CeedVectorDestroy(&child_qx_coords);
  CeedVectorDestroy(&parent_qx_coords);
  PetscFunctionReturn(0);
}

// Compute mass matrix for statistics projection
PetscErrorCode SetupL2ProjectionStats(Ceed ceed, User user, CeedData ceed_data) {
  CeedQFunction qf_mass;
  CeedOperator  op_mass;
  CeedInt       num_comp_q, q_data_size;
  PetscFunctionBeginUser;

  // CEED Restriction
  CeedElemRestrictionGetNumComponents(ceed_data->spanstats.elem_restr_parent_stats, &num_comp_q);
  CeedElemRestrictionGetNumComponents(ceed_data->spanstats.elem_restr_parent_qd, &q_data_size);

  // Create Mass CeedOperator
  PetscCall(CreateMassQFunction(ceed, num_comp_q, q_data_size, &qf_mass));
  CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass);
  CeedOperatorSetField(op_mass, "q", ceed_data->spanstats.elem_restr_parent_stats, ceed_data->spanstats.basis_stats, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "qdata", ceed_data->spanstats.elem_restr_parent_qd, CEED_BASIS_COLLOCATED, ceed_data->spanstats.q_data);
  CeedOperatorSetField(op_mass, "v", ceed_data->spanstats.elem_restr_parent_stats, ceed_data->spanstats.basis_stats, CEED_VECTOR_ACTIVE);

  // Setup KSP for L^2 projection
  {
    MatopApplyContext M_ctx;
    PetscInt          l_size, g_size;
    Mat               mat_mass;
    VecType           vec_type;
    KSP               ksp;
    Vec               ones, M_inv;
    CeedVector        x_ceed, y_ceed;

    PetscCall(DMCreateGlobalVector(user->spanstats.dm, &M_inv));
    PetscCall(VecGetLocalSize(M_inv, &l_size));
    PetscCall(VecGetSize(M_inv, &g_size));
    PetscCall(VecGetType(M_inv, &vec_type));

    PetscCall(PetscMalloc1(1, &M_ctx));
    PetscCall(MatCreateShell(PETSC_COMM_WORLD, l_size, l_size, g_size, g_size, M_ctx, &mat_mass));
    PetscCall(MatShellSetOperation(mat_mass, MATOP_MULT, (void (*)(void))MatMult_Ceed));
    PetscCall(MatShellSetOperation(mat_mass, MATOP_GET_DIAGONAL, (void (*)(void))MatGetDiag_Ceed));
    PetscCall(MatShellSetVecType(mat_mass, vec_type));

    CeedElemRestrictionCreateVector(ceed_data->spanstats.elem_restr_parent_stats, &x_ceed, NULL);
    CeedElemRestrictionCreateVector(ceed_data->spanstats.elem_restr_parent_stats, &y_ceed, NULL);

    PetscCall(SetupMatopApplyCtx(PETSC_COMM_WORLD, user->spanstats.dm, user->ceed, op_mass, x_ceed, y_ceed, NULL, M_ctx));
    user->spanstats.M_ctx = M_ctx;

    // Create lumped mass matrix inverse
    PetscCall(DMGetGlobalVector(user->spanstats.dm, &ones));
    PetscCall(VecZeroEntries(M_inv));
    PetscCall(VecSet(ones, 1));
    PetscCall(MatMult(mat_mass, ones, M_inv));
    PetscCall(VecReciprocal(M_inv));
    user->spanstats.M_inv = M_inv;
    PetscCall(DMRestoreGlobalVector(user->spanstats.dm, &ones));

    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOptionsPrefix(ksp, "spanstats_"));
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
    user->spanstats.ksp = ksp;
  }

  // Cleanup
  CeedQFunctionDestroy(&qf_mass);
  PetscFunctionReturn(0);
}

// Create CeedOperators and KSP for the statistics collection and processing
PetscErrorCode CreateStatisticsOperators(Ceed ceed, User user, CeedData ceed_data, ProblemData *problem) {
  CeedInt      num_comp_stats = user->spanstats.num_comp_stats, num_comp_x = problem->dim, num_comp_q;
  CeedOperator op_setup_sur;
  PetscFunctionBeginUser;
  CeedBasisGetNumComponents(ceed_data->basis_q, &num_comp_q);

  // Create Operator for statistics collection
  switch (user->phys->state_var) {
    case STATEVAR_PRIMITIVE:
      CeedQFunctionCreateInterior(ceed, 1, ChildStatsCollection_Prim, ChildStatsCollection_Prim_loc, &ceed_data->spanstats.qf_stats_collect);
      break;
    case STATEVAR_CONSERVATIVE:
      CeedQFunctionCreateInterior(ceed, 1, ChildStatsCollection_Conserv, ChildStatsCollection_Conserv_loc, &ceed_data->spanstats.qf_stats_collect);
      break;
  }

  if (user->app_ctx->stats_test) {
    CeedQFunctionDestroy(&ceed_data->spanstats.qf_stats_collect);
    CeedQFunctionCreateInterior(ceed, 1, ChildStatsCollectionTest, ChildStatsCollectionTest_loc, &ceed_data->spanstats.qf_stats_collect);
  }

  CeedQFunctionSetContext(ceed_data->spanstats.qf_stats_collect, problem->apply_vol_ifunction.qfunction_context);
  CeedQFunctionAddInput(ceed_data->spanstats.qf_stats_collect, "q", num_comp_q, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(ceed_data->spanstats.qf_stats_collect, "q_data", problem->q_data_size_vol, CEED_EVAL_NONE);
  CeedQFunctionAddInput(ceed_data->spanstats.qf_stats_collect, "x", num_comp_x, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(ceed_data->spanstats.qf_stats_collect, "v", num_comp_stats, CEED_EVAL_NONE);

  CeedOperatorCreate(ceed, ceed_data->spanstats.qf_stats_collect, NULL, NULL, &user->spanstats.op_stats_collect);
  CeedOperatorSetField(user->spanstats.op_stats_collect, "q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(user->spanstats.op_stats_collect, "q_data", ceed_data->elem_restr_qd_i, CEED_BASIS_COLLOCATED, ceed_data->q_data);
  CeedOperatorSetField(user->spanstats.op_stats_collect, "x", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord);
  CeedOperatorSetField(user->spanstats.op_stats_collect, "v", ceed_data->spanstats.elem_restr_child_colloc, CEED_BASIS_COLLOCATED,
                       CEED_VECTOR_ACTIVE);

  // Create Operator for L^2 projection of statistics
  // Simply take collocated parent data (with quadrature weight already applied) and multiply by weight function.
  // Therefore, an Identity QF is sufficient
  CeedQFunctionCreateIdentity(ceed, num_comp_stats, CEED_EVAL_NONE, CEED_EVAL_INTERP, &ceed_data->spanstats.qf_stats_proj);

  CeedOperatorCreate(ceed, ceed_data->spanstats.qf_stats_proj, NULL, NULL, &user->spanstats.op_stats_proj);
  CeedOperatorSetField(user->spanstats.op_stats_proj, "input", ceed_data->spanstats.elem_restr_parent_colloc, CEED_BASIS_COLLOCATED,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(user->spanstats.op_stats_proj, "output", ceed_data->spanstats.elem_restr_parent_stats, ceed_data->spanstats.basis_stats,
                       CEED_VECTOR_ACTIVE);

  // Get q_data for lumped mass matrix formation
  CeedOperatorCreate(ceed, ceed_data->qf_setup_sur, NULL, NULL, &op_setup_sur);
  CeedOperatorSetField(op_setup_sur, "dx", ceed_data->spanstats.elem_restr_parent_x, ceed_data->spanstats.basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_sur, "weight", CEED_ELEMRESTRICTION_NONE, ceed_data->spanstats.basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_sur, "surface qdata", ceed_data->spanstats.elem_restr_parent_qd, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedOperatorApply(op_setup_sur, ceed_data->spanstats.x_coord, ceed_data->spanstats.q_data, CEED_REQUEST_IMMEDIATE);

  CeedOperatorDestroy(&op_setup_sur);
  PetscFunctionReturn(0);
}

PetscErrorCode SetupErrorTesting(Ceed ceed, User user, CeedData ceed_data) {
  CeedInt       num_comp_stats = user->spanstats.num_comp_stats, num_comp_x;
  CeedQFunction qf_error;
  CeedOperator  op_error;
  CeedInt       q_data_size;
  CeedVector    x_ceed, y_ceed;
  PetscFunctionBeginUser;

  CeedElemRestrictionGetNumComponents(ceed_data->spanstats.elem_restr_parent_qd, &q_data_size);
  CeedBasisGetNumComponents(ceed_data->spanstats.basis_x, &num_comp_x);

  CeedQFunctionCreateInterior(ceed, 1, ChildStatsCollectionTest_Error, ChildStatsCollectionTest_Error_loc, &qf_error);
  CeedQFunctionAddInput(qf_error, "q", num_comp_stats, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_error, "qdata", q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_error, "x", num_comp_x, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_error, "v", num_comp_stats, CEED_EVAL_INTERP);

  CeedOperatorCreate(ceed, qf_error, NULL, NULL, &op_error);
  CeedOperatorSetField(op_error, "q", ceed_data->spanstats.elem_restr_parent_stats, ceed_data->spanstats.basis_stats, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_error, "qdata", ceed_data->spanstats.elem_restr_parent_qd, CEED_BASIS_COLLOCATED, ceed_data->spanstats.q_data);
  CeedOperatorSetField(op_error, "x", ceed_data->spanstats.elem_restr_parent_x, ceed_data->spanstats.basis_x, ceed_data->spanstats.x_coord);
  CeedOperatorSetField(op_error, "v", ceed_data->spanstats.elem_restr_parent_stats, ceed_data->spanstats.basis_stats, CEED_VECTOR_ACTIVE);

  CeedElemRestrictionCreateVector(ceed_data->spanstats.elem_restr_parent_stats, &x_ceed, NULL);
  CeedElemRestrictionCreateVector(ceed_data->spanstats.elem_restr_parent_stats, &y_ceed, NULL);

  PetscCall(PetscCalloc1(1, &user->spanstats.test_error_ctx));
  PetscCall(SetupMatopApplyCtx(PETSC_COMM_WORLD, user->spanstats.dm, user->ceed, op_error, x_ceed, y_ceed, NULL, user->spanstats.test_error_ctx));

  PetscFunctionReturn(0);
}

// Setup for statistics collection
PetscErrorCode SetupStatsCollection(Ceed ceed, User user, CeedData ceed_data, ProblemData *problem) {
  DM                 dm   = user->spanstats.dm;
  MPI_Comm           comm = PetscObjectComm((PetscObject)dm);
  CeedInt            dim, P, Q, num_comp_x;
  Vec                X_loc;
  PetscMemType       X_loc_memtype;
  const PetscScalar *X_loc_array;
  PetscFunctionBeginUser;

  PetscCall(DMGetDimension(dm, &dim));
  CeedBasisGetNumQuadraturePoints1D(ceed_data->basis_q, &Q);
  CeedBasisGetNumNodes1D(ceed_data->basis_q, &P);

  PetscCall(GetRestrictionForDomain(ceed, dm, 0, 0, 0, Q, problem->q_data_size_sur, &ceed_data->spanstats.elem_restr_parent_stats,
                                    &ceed_data->spanstats.elem_restr_parent_x, &ceed_data->spanstats.elem_restr_parent_qd));
  CeedElemRestrictionGetNumComponents(ceed_data->spanstats.elem_restr_parent_x, &num_comp_x);
  CeedElemRestrictionCreateVector(ceed_data->spanstats.elem_restr_parent_x, &ceed_data->spanstats.x_coord, NULL);
  CeedElemRestrictionCreateVector(ceed_data->spanstats.elem_restr_parent_stats, &user->spanstats.rhs_ceed, NULL);
  CeedElemRestrictionCreateVector(ceed_data->spanstats.elem_restr_parent_qd, &ceed_data->spanstats.q_data, NULL);

  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_x, 2, Q, CEED_GAUSS, &ceed_data->spanstats.basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, user->spanstats.num_comp_stats, P, Q, CEED_GAUSS, &ceed_data->spanstats.basis_stats);

  PetscCall(CreateElemRestrColloc(ceed, user->spanstats.num_comp_stats, ceed_data->spanstats.basis_stats,
                                  ceed_data->spanstats.elem_restr_parent_stats, &ceed_data->spanstats.elem_restr_parent_colloc,
                                  &user->spanstats.parent_stats, NULL));
  PetscCall(CreateElemRestrColloc(ceed, user->spanstats.num_comp_stats, ceed_data->basis_q, ceed_data->elem_restr_q,
                                  &ceed_data->spanstats.elem_restr_child_colloc, &user->spanstats.child_stats, NULL));
  CeedElemRestrictionCreateVector(ceed_data->spanstats.elem_restr_child_colloc, &user->spanstats.child_inst_stats, NULL);
  CeedVectorSetValue(user->spanstats.child_stats, 0);

  // -- Copy DM coordinates into CeedVector
  {
    DM cdm;
    PetscCall(DMGetCellCoordinateDM(dm, &cdm));
    if (cdm) {
      PetscCall(DMGetCellCoordinatesLocal(dm, &X_loc));
    } else {
      PetscCall(DMGetCoordinatesLocal(dm, &X_loc));
    }
  }
  PetscCall(VecScale(X_loc, problem->dm_scale));
  PetscCall(VecGetArrayReadAndMemType(X_loc, &X_loc_array, &X_loc_memtype));
  CeedVectorSetArray(ceed_data->spanstats.x_coord, MemTypeP2C(X_loc_memtype), CEED_COPY_VALUES, (PetscScalar *)X_loc_array);
  PetscCall(VecRestoreArrayRead(X_loc, &X_loc_array));

  // Create SF for communicating child data back their respective parents
  PetscCall(PetscSFCreate(comm, &user->spanstats.sf));
  PetscCall(CreateStatsSF(ceed, ceed_data, user->dm, user->spanstats.dm, user->spanstats.sf));

  // Create CeedOperators for statistics collection
  PetscCall(CreateStatisticsOperators(ceed, user, ceed_data, problem));

  // Setup KSP and Mat for L^2 projection of statistics
  PetscCall(SetupL2ProjectionStats(ceed, user, ceed_data));

  if (user->app_ctx->stats_test) {
    PetscCall(SetupErrorTesting(ceed, user, ceed_data));
  }

  PetscFunctionReturn(0);
}

// Collect statistics based on the solution Q
PetscErrorCode CollectStatistics(User user, PetscScalar solution_time, Vec Q) {
  PetscMemType       q_mem_type;
  const PetscScalar *q_arr;
  Vec                Q_loc;
  PetscFunctionBeginUser;

  PetscCall(DMGetLocalVector(user->dm, &Q_loc));
  PetscCall(VecZeroEntries(Q_loc));
  PetscCall(DMGlobalToLocal(user->dm, Q, INSERT_VALUES, Q_loc));

  PetscCall(VecGetArrayReadAndMemType(Q_loc, &q_arr, &q_mem_type));
  CeedVectorSetArray(user->q_ceed, MemTypeP2C(q_mem_type), CEED_USE_POINTER, (PetscScalar *)q_arr);

  CeedOperatorApply(user->spanstats.op_stats_collect, user->q_ceed, user->spanstats.child_inst_stats, CEED_REQUEST_IMMEDIATE);

  CeedVectorTakeArray(user->q_ceed, MemTypeP2C(q_mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(Q_loc, &q_arr));
  PetscCall(DMRestoreLocalVector(user->dm, &Q_loc));

  // Record averaging using left rectangle rule
  PetscScalar delta_t           = solution_time - user->spanstats.prev_time;
  PetscScalar prev_timeinterval = user->spanstats.prev_time - user->app_ctx->cont_time;
  CeedVectorScale(user->spanstats.child_stats, prev_timeinterval / (prev_timeinterval + delta_t));
  CeedVectorAXPY(user->spanstats.child_stats, delta_t / (prev_timeinterval + delta_t), user->spanstats.child_inst_stats);
  user->spanstats.prev_time = solution_time;

  PetscFunctionReturn(0);
}

// Process the child statistics into parent statistics and project them onto stats
PetscErrorCode ProcessStatistics(User user, Vec *stats) {
  Span_Stats         user_stats = user->spanstats;
  const PetscScalar *child_stats;
  PetscScalar       *parent_stats;
  MPI_Datatype       unit;
  Vec                rhs_loc, rhs;
  PetscMemType       rhs_mem_type;
  CeedScalar        *rhs_arr;
  CeedMemType        ceed_mem_type;
  PetscFunctionBeginUser;

  CeedGetPreferredMemType(user->ceed, &ceed_mem_type);
  CeedVectorSetValue(user_stats.parent_stats, 0);

  CeedVectorGetArrayRead(user_stats.child_stats, ceed_mem_type, &child_stats);
  CeedVectorGetArray(user_stats.parent_stats, ceed_mem_type, &parent_stats);

  if (user_stats.num_comp_stats == 1) unit = MPIU_REAL;
  else {
    PetscCallMPI(MPI_Type_contiguous(user_stats.num_comp_stats, MPIU_REAL, &unit));
    PetscCallMPI(MPI_Type_commit(&unit));
  }

  PetscCall(PetscSFReduceBegin(user_stats.sf, unit, child_stats, parent_stats, MPI_SUM));
  PetscCall(PetscSFReduceEnd(user_stats.sf, unit, child_stats, parent_stats, MPI_SUM));

  CeedVectorRestoreArrayRead(user_stats.child_stats, &child_stats);
  CeedVectorRestoreArray(user_stats.parent_stats, &parent_stats);
  PetscCallMPI(MPI_Type_free(&unit));

  CeedVectorScale(user_stats.parent_stats, 1 / user_stats.span_width);

  // L^2 projection with the parent_data
  PetscCall(DMGetGlobalVector(user_stats.dm, &rhs));
  PetscCall(DMGetLocalVector(user_stats.dm, &rhs_loc));
  PetscCall(VecZeroEntries(rhs));
  PetscCall(VecZeroEntries(rhs_loc));
  PetscCall(VecGetArrayWriteAndMemType(rhs_loc, &rhs_arr, &rhs_mem_type));
  CeedVectorSetArray(user_stats.rhs_ceed, MemTypeP2C(rhs_mem_type), CEED_USE_POINTER, (PetscScalar *)rhs_arr);

  CeedOperatorApply(user_stats.op_stats_proj, user_stats.parent_stats, user_stats.rhs_ceed, CEED_REQUEST_IMMEDIATE);

  CeedVectorTakeArray(user_stats.rhs_ceed, MemTypeP2C(rhs_mem_type), &rhs_arr);
  PetscCall(VecRestoreArrayAndMemType(rhs_loc, &rhs_arr));
  PetscCall(DMLocalToGlobal(user_stats.dm, rhs_loc, ADD_VALUES, rhs));

  PetscCall(VecDuplicate(rhs, stats));
  PetscCall(VecPointwiseMult(*stats, rhs, user_stats.M_inv));

  PetscCall(KSPSolve(user_stats.ksp, rhs, *stats));

  PetscFunctionReturn(0);
}

// TSMonitor for the statistics collection and processing
PetscErrorCode TSMonitor_Statistics(TS ts, PetscInt steps, PetscReal solution_time, Vec Q, void *ctx) {
  User user = (User)ctx;
  Vec  stats;
  PetscFunctionBeginUser;

  // Do not collect or process on the first step of the run (ie. on the initial condition)
  if (steps == user->app_ctx->cont_steps) PetscFunctionReturn(0);

  if (steps % user->app_ctx->stats_collect_interval == 0) {
    PetscCall(CollectStatistics(user, solution_time, Q));
  }

  if (steps % user->app_ctx->stats_write_interval == 0 && user->app_ctx->stats_write_interval != -1) {
    PetscCall(DMGetGlobalVector(user->spanstats.dm, &stats));
    PetscCall(ProcessStatistics(user, &stats));
    PetscCall(VecViewFromOptions(stats, NULL, "-stats_write_view"));
    PetscCall(DMRestoreGlobalVector(user->spanstats.dm, &stats));
  }
  PetscFunctionReturn(0);
}

// Function to be called at the end of a simulation
PetscErrorCode StatsCollectFinalCall(User user, PetscReal solution_time, Vec Q) {
  Vec stats;
  PetscFunctionBeginUser;

  PetscCall(CollectStatistics(user, solution_time, Q));

  PetscCall(DMGetGlobalVector(user->spanstats.dm, &stats));
  PetscCall(ProcessStatistics(user, &stats));
  PetscCall(VecViewFromOptions(stats, NULL, "-stats_write_view"));

  if (user->app_ctx->stats_test) {
    Vec error;
    PetscCall(VecDuplicate(stats, &error));
    PetscCall(ApplyLocal_Ceed(stats, error, user->spanstats.test_error_ctx));
    PetscScalar error_sq = 0;
    PetscCall(VecSum(error, &error_sq));
    PetscScalar l2_error = sqrt(error_sq);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "l2 error: %.5e\n", l2_error));
  }
  PetscCall(DMRestoreGlobalVector(user->spanstats.dm, &stats));

  PetscFunctionReturn(0);
}
