// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
/// @file
/// Functions for setting up and performing statistics collection

#include "../qfunctions/turb_spanstats.h"

#include <ceed.h>
#include <petscdmplex.h>
#include <petscsf.h>

#include "../include/petsc_ops.h"
#include "../navierstokes.h"

typedef struct {
  CeedElemRestriction elem_restr_parent_x, elem_restr_parent_stats, elem_restr_parent_qd, elem_restr_parent_colloc, elem_restr_child_colloc;
  CeedBasis           basis_x, basis_stats;
  CeedVector          x_coord, q_data;
} *SpanStatsSetupData;

PetscErrorCode CreateStatsDM(User user, ProblemData *problem, PetscInt degree) {
  user->spanstats.num_comp_stats = TURB_NUM_COMPONENTS;
  PetscReal     domain_min[3], domain_max[3];
  PetscSection  section;
  PetscLogStage stage_stats_setup;
  MPI_Comm      comm = PetscObjectComm((PetscObject)user->dm);

  PetscFunctionBeginUser;
  PetscCall(PetscLogStageGetId("Stats Setup", &stage_stats_setup));
  if (stage_stats_setup == -1) PetscCall(PetscLogStageRegister("Stats Setup", &stage_stats_setup));
  PetscCall(PetscLogStagePush(stage_stats_setup));

  // Get spanwise length
  PetscCall(DMGetBoundingBox(user->dm, domain_min, domain_max));
  user->spanstats.span_width = domain_max[2] - domain_min[1];

  {  // Get DM from surface
    DM          parent_distributed_dm;
    PetscSF     isoperiodicface;
    DMLabel     label;
    PetscMPIInt size;

    PetscCall(DMPlexGetIsoperiodicFaceSF(user->dm, &isoperiodicface));

    if (isoperiodicface) {
      PetscSF         inv_isoperiodicface;
      PetscInt        nleaves;
      const PetscInt *ilocal;

      PetscCall(PetscSFCreateInverseSF(isoperiodicface, &inv_isoperiodicface));
      PetscCall(PetscSFGetGraph(inv_isoperiodicface, NULL, &nleaves, &ilocal, NULL));
      PetscCall(DMCreateLabel(user->dm, "Periodic Face"));
      PetscCall(DMGetLabel(user->dm, "Periodic Face", &label));
      for (PetscInt i = 0; i < nleaves; i++) {
        PetscCall(DMLabelSetValue(label, ilocal[i], 1));
      }
    } else {
      PetscCall(DMGetLabel(user->dm, "Face Sets", &label));
    }

    PetscCall(DMPlexLabelComplete(user->dm, label));
    PetscCall(DMPlexFilter(user->dm, label, 1, &user->spanstats.dm));
    PetscCall(DMProjectCoordinates(user->spanstats.dm, NULL));  // Ensure that a coordinate FE exists

    PetscCall(DMPlexDistribute(user->spanstats.dm, 0, NULL, &parent_distributed_dm));
    PetscCallMPI(MPI_Comm_size(comm, &size));
    if (parent_distributed_dm) {
      PetscCall(DMDestroy(&user->spanstats.dm));
      user->spanstats.dm = parent_distributed_dm;
    } else if (size > 1) {
      PetscCall(PetscPrintf(comm, "WARNING: Turbulent spanwise statistics: parent DM could not be distributed accross %d ranks.\n", size));
    }
  }
  {
    PetscBool is_simplex = PETSC_FALSE;
    PetscCall(DMPlexIsSimplex(user->spanstats.dm, &is_simplex));
    PetscCheck(is_simplex != PETSC_TRUE, comm, PETSC_ERR_ARG_WRONGSTATE, "Spanwise Statistics is not implemented for non-tensor product grids");
  }

  PetscCall(PetscObjectSetName((PetscObject)user->spanstats.dm, "Spanwise_Stats"));
  PetscCall(DMSetOptionsPrefix(user->spanstats.dm, "turbulence_spanstats_"));
  PetscCall(DMSetFromOptions(user->spanstats.dm));
  PetscCall(DMViewFromOptions(user->spanstats.dm, NULL, "-dm_view"));

  // Create FE space for parent DM
  PetscCall(DMSetupByOrder_FEM(PETSC_TRUE, PETSC_TRUE, user->app_ctx->degree, 1, user->app_ctx->q_extra, 1, &user->spanstats.num_comp_stats,
                               user->spanstats.dm));

  // Create Section for data
  PetscCall(DMGetLocalSection(user->spanstats.dm, &section));
  PetscCall(PetscSectionSetFieldName(section, 0, ""));
  PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_DENSITY, "MeanDensity"));
  PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_PRESSURE, "MeanPressure"));
  PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_PRESSURE_SQUARED, "MeanPressureSquared"));
  PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_PRESSURE_VELOCITY_X, "MeanPressureVelocityX"));
  PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_PRESSURE_VELOCITY_Y, "MeanPressureVelocityY"));
  PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_PRESSURE_VELOCITY_Z, "MeanPressureVelocityZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_DENSITY_TEMPERATURE, "MeanDensityTemperature"));
  PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_DENSITY_TEMPERATURE_FLUX_X, "MeanDensityTemperatureFluxX"));
  PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_DENSITY_TEMPERATURE_FLUX_Y, "MeanDensityTemperatureFluxY"));
  PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_DENSITY_TEMPERATURE_FLUX_Z, "MeanDensityTemperatureFluxZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_MOMENTUM_X, "MeanMomentumX"));
  PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_MOMENTUM_Y, "MeanMomentumY"));
  PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_MOMENTUM_Z, "MeanMomentumZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_MOMENTUMFLUX_XX, "MeanMomentumFluxXX"));
  PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_MOMENTUMFLUX_YY, "MeanMomentumFluxYY"));
  PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_MOMENTUMFLUX_ZZ, "MeanMomentumFluxZZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_MOMENTUMFLUX_YZ, "MeanMomentumFluxYZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_MOMENTUMFLUX_XZ, "MeanMomentumFluxXZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_MOMENTUMFLUX_XY, "MeanMomentumFluxXY"));
  PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_VELOCITY_X, "MeanVelocityX"));
  PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_VELOCITY_Y, "MeanVelocityY"));
  PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_VELOCITY_Z, "MeanVelocityZ"));

  PetscCall(PetscLogStagePop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Create CeedElemRestriction for collocated data based on associated CeedBasis and CeedElemRestriction
// Number of quadrature points is used from the CeedBasis, and number of elements is used from the CeedElemRestriction
PetscErrorCode CreateElemRestrColloc(Ceed ceed, CeedInt num_comp, CeedBasis basis, CeedElemRestriction elem_restr_base,
                                     CeedElemRestriction *elem_restr_collocated) {
  CeedInt num_elem_qpts, loc_num_elem;

  PetscFunctionBeginUser;
  PetscCallCeed(ceed, CeedBasisGetNumQuadraturePoints(basis, &num_elem_qpts));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumElements(elem_restr_base, &loc_num_elem));

  const CeedInt strides[] = {num_comp, 1, num_elem_qpts * num_comp};
  PetscCallCeed(ceed, CeedElemRestrictionCreateStrided(ceed, loc_num_elem, num_elem_qpts, num_comp, num_comp * loc_num_elem * num_elem_qpts, strides,
                                                       elem_restr_collocated));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Get coordinates of quadrature points
PetscErrorCode GetQuadratureCoords(Ceed ceed, DM dm, CeedElemRestriction elem_restr_x, CeedBasis basis_x, CeedVector x_coords, Vec *Qx_coords,
                                   PetscInt *total_nqpnts) {
  CeedElemRestriction  elem_restr_qx;
  CeedQFunction        qf_quad_coords;
  CeedOperator         op_quad_coords;
  CeedInt              num_comp_x, loc_num_elem, num_elem_qpts;
  OperatorApplyContext op_quad_coords_ctx;

  PetscFunctionBeginUser;
  // Create Element Restriction and CeedVector for quadrature coordinates
  PetscCallCeed(ceed, CeedBasisGetNumQuadraturePoints(basis_x, &num_elem_qpts));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumElements(elem_restr_x, &loc_num_elem));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_x, &num_comp_x));
  *total_nqpnts = num_elem_qpts * loc_num_elem;
  PetscCall(CreateElemRestrColloc(ceed, num_comp_x, basis_x, elem_restr_x, &elem_restr_qx));

  // Create QFunction
  PetscCallCeed(ceed, CeedQFunctionCreateIdentity(ceed, num_comp_x, CEED_EVAL_INTERP, CEED_EVAL_NONE, &qf_quad_coords));

  // Create Operator
  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_quad_coords, NULL, NULL, &op_quad_coords));
  PetscCallCeed(ceed, CeedOperatorSetField(op_quad_coords, "input", elem_restr_x, basis_x, x_coords));
  PetscCallCeed(ceed, CeedOperatorSetField(op_quad_coords, "output", elem_restr_qx, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));

  PetscCall(CeedOperatorCreateLocalVecs(op_quad_coords, DMReturnVecType(dm), PetscObjectComm((PetscObject)dm), NULL, Qx_coords));
  PetscCall(OperatorApplyContextCreate(NULL, NULL, ceed, op_quad_coords, CEED_VECTOR_NONE, NULL, NULL, NULL, &op_quad_coords_ctx));

  PetscCall(ApplyCeedOperatorLocalToLocal(NULL, *Qx_coords, op_quad_coords_ctx));

  PetscCall(OperatorApplyContextDestroy(op_quad_coords_ctx));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qx));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_quad_coords));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_quad_coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SpanStatsSetupDataCreate(Ceed ceed, User user, CeedData ceed_data, ProblemData *problem, SpanStatsSetupData *stats_data) {
  DM       dm = user->spanstats.dm;
  PetscInt dim;
  CeedInt  num_qpts_child1d, num_comp_x, num_comp_stats = user->spanstats.num_comp_stats;
  Vec      X_loc;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(stats_data));

  PetscCall(DMGetDimension(dm, &dim));
  PetscCallCeed(ceed, CeedBasisGetNumQuadraturePoints1D(ceed_data->basis_q, &num_qpts_child1d));
  CeedInt num_qpts_parent = CeedIntPow(num_qpts_child1d, dim);
  PetscCall(GetRestrictionForDomain(ceed, dm, 0, 0, 0, 0, num_qpts_parent, problem->q_data_size_sur, &(*stats_data)->elem_restr_parent_stats,
                                    &(*stats_data)->elem_restr_parent_x, &(*stats_data)->elem_restr_parent_qd));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents((*stats_data)->elem_restr_parent_x, &num_comp_x));
  PetscCallCeed(ceed, CeedElemRestrictionCreateVector((*stats_data)->elem_restr_parent_x, &(*stats_data)->x_coord, NULL));
  PetscCallCeed(ceed, CeedElemRestrictionCreateVector((*stats_data)->elem_restr_parent_qd, &(*stats_data)->q_data, NULL));

  {
    DM dm_coord;
    PetscCall(DMGetCoordinateDM(dm, &dm_coord));
    PetscCall(CreateBasisFromPlex(ceed, dm_coord, 0, 0, 0, 0, &(*stats_data)->basis_x));
    PetscCall(CreateBasisFromPlex(ceed, dm, 0, 0, 0, 0, &(*stats_data)->basis_stats));
  }

  PetscCall(CreateElemRestrColloc(ceed, num_comp_stats, (*stats_data)->basis_stats, (*stats_data)->elem_restr_parent_stats,
                                  &(*stats_data)->elem_restr_parent_colloc));
  PetscCall(CreateElemRestrColloc(ceed, num_comp_stats, ceed_data->basis_q, ceed_data->elem_restr_q, &(*stats_data)->elem_restr_child_colloc));

  {  // -- Copy DM coordinates into CeedVector
    DM cdm;
    PetscCall(DMGetCellCoordinateDM(dm, &cdm));
    if (cdm) {
      PetscCall(DMGetCellCoordinatesLocal(dm, &X_loc));
    } else {
      PetscCall(DMGetCoordinatesLocal(dm, &X_loc));
    }
  }
  PetscCall(VecScale(X_loc, problem->dm_scale));
  PetscCall(VecCopyP2C(X_loc, (*stats_data)->x_coord));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SpanStatsSetupDataDestroy(SpanStatsSetupData data) {
  Ceed ceed;

  PetscFunctionBeginUser;
  PetscCall(CeedElemRestrictionGetCeed(data->elem_restr_parent_x, &ceed));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&data->elem_restr_parent_x));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&data->elem_restr_parent_stats));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&data->elem_restr_parent_qd));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&data->elem_restr_parent_colloc));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&data->elem_restr_child_colloc));

  PetscCallCeed(ceed, CeedBasisDestroy(&data->basis_x));
  PetscCallCeed(ceed, CeedBasisDestroy(&data->basis_stats));

  PetscCallCeed(ceed, CeedVectorDestroy(&data->x_coord));
  PetscCallCeed(ceed, CeedVectorDestroy(&data->q_data));

  PetscCall(PetscFree(data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Create PetscSF for child-to-parent communication
PetscErrorCode CreateStatsSF(Ceed ceed, CeedData ceed_data, SpanStatsSetupData stats_data, DM parentdm, DM childdm, PetscSF *statssf) {
  PetscInt child_num_qpnts, parent_num_qpnts;
  CeedInt  num_comp_x;
  Vec      Child_qx_coords, Parent_qx_coords;

  PetscFunctionBeginUser;
  PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)childdm), statssf));

  // Assume that child and parent have the same number of components
  PetscCallCeed(ceed, CeedBasisGetNumComponents(ceed_data->basis_x, &num_comp_x));
  const PetscInt num_comp_sf = num_comp_x - 1;  // Number of coord components used in the creation of the SF

  // Get quad_coords for child and parent DM
  PetscCall(GetQuadratureCoords(ceed, childdm, ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord, &Child_qx_coords, &child_num_qpnts));
  PetscCall(GetQuadratureCoords(ceed, parentdm, stats_data->elem_restr_parent_x, stats_data->basis_x, stats_data->x_coord, &Parent_qx_coords,
                                &parent_num_qpnts));

  {  // Remove z component of coordinates for matching
    const PetscReal *child_quad_coords, *parent_quad_coords;
    PetscReal       *child_coords, *parent_coords;

    PetscCall(VecGetArrayRead(Child_qx_coords, &child_quad_coords));
    PetscCall(VecGetArrayRead(Parent_qx_coords, &parent_quad_coords));

    PetscCall(PetscMalloc2(child_num_qpnts * 2, &child_coords, parent_num_qpnts * 2, &parent_coords));
    for (int i = 0; i < child_num_qpnts; i++) {
      child_coords[0 + i * num_comp_sf] = child_quad_coords[0 + i * num_comp_x];
      child_coords[1 + i * num_comp_sf] = child_quad_coords[1 + i * num_comp_x];
    }
    for (int i = 0; i < parent_num_qpnts; i++) {
      parent_coords[0 + i * num_comp_sf] = parent_quad_coords[0 + i * num_comp_x];
      parent_coords[1 + i * num_comp_sf] = parent_quad_coords[1 + i * num_comp_x];
    }
    PetscCall(VecRestoreArrayRead(Child_qx_coords, &child_quad_coords));
    PetscCall(VecRestoreArrayRead(Parent_qx_coords, &parent_quad_coords));

    PetscCall(PetscSFSetGraphFromCoordinates(*statssf, parent_num_qpnts, child_num_qpnts, num_comp_sf, 1e-12, parent_coords, child_coords));
    PetscCall(PetscFree2(child_coords, parent_coords));
  }

  PetscCall(PetscSFViewFromOptions(*statssf, NULL, "-spanstats_sf_view"));

  PetscCall(VecDestroy(&Child_qx_coords));
  PetscCall(VecDestroy(&Parent_qx_coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Setup RHS and LHS for L^2 projection of statistics
PetscErrorCode SetupL2ProjectionStats(Ceed ceed, User user, CeedData ceed_data, SpanStatsSetupData stats_data) {
  CeedOperator  op_mass, op_setup_sur, op_proj_rhs;
  CeedQFunction qf_mass, qf_stats_proj;
  CeedInt       q_data_size, num_comp_stats = user->spanstats.num_comp_stats;
  MPI_Comm      comm = PetscObjectComm((PetscObject)user->spanstats.dm);

  PetscFunctionBeginUser;
  // -- Create Operator for RHS of L^2 projection of statistics
  // Simply take collocated parent data (with quadrature weight already applied) and multiply by weight function.
  // Therefore, an Identity QF is sufficient
  PetscCallCeed(ceed, CeedQFunctionCreateIdentity(ceed, num_comp_stats, CEED_EVAL_NONE, CEED_EVAL_INTERP, &qf_stats_proj));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_stats_proj, NULL, NULL, &op_proj_rhs));
  PetscCallCeed(ceed, CeedOperatorSetField(op_proj_rhs, "input", stats_data->elem_restr_parent_colloc, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_proj_rhs, "output", stats_data->elem_restr_parent_stats, stats_data->basis_stats, CEED_VECTOR_ACTIVE));

  PetscCall(OperatorApplyContextCreate(NULL, user->spanstats.dm, ceed, op_proj_rhs, NULL, NULL, NULL, NULL, &user->spanstats.op_proj_rhs_ctx));
  PetscCall(CeedOperatorCreateLocalVecs(op_proj_rhs, DMReturnVecType(user->spanstats.dm), comm, &user->spanstats.Parent_Stats_loc, NULL));

  // -- Setup LHS of L^2 projection
  // Get q_data for mass matrix operator
  PetscCallCeed(ceed, CeedOperatorCreate(ceed, ceed_data->qf_setup_sur, NULL, NULL, &op_setup_sur));
  PetscCallCeed(ceed, CeedOperatorSetField(op_setup_sur, "dx", stats_data->elem_restr_parent_x, stats_data->basis_x, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_setup_sur, "weight", CEED_ELEMRESTRICTION_NONE, stats_data->basis_x, CEED_VECTOR_NONE));
  PetscCallCeed(ceed,
                CeedOperatorSetField(op_setup_sur, "surface qdata", stats_data->elem_restr_parent_qd, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorApply(op_setup_sur, stats_data->x_coord, stats_data->q_data, CEED_REQUEST_IMMEDIATE));

  // CEED Restriction
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(stats_data->elem_restr_parent_qd, &q_data_size));

  // Create Mass CeedOperator
  PetscCall(CreateMassQFunction(ceed, num_comp_stats, q_data_size, &qf_mass));
  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "u", stats_data->elem_restr_parent_stats, stats_data->basis_stats, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "qdata", stats_data->elem_restr_parent_qd, CEED_BASIS_COLLOCATED, stats_data->q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "v", stats_data->elem_restr_parent_stats, stats_data->basis_stats, CEED_VECTOR_ACTIVE));

  {  // Setup KSP for L^2 projection
    OperatorApplyContext M_ctx;
    Mat                  mat_mass;
    KSP                  ksp;

    PetscCall(OperatorApplyContextCreate(user->spanstats.dm, user->spanstats.dm, user->ceed, op_mass, NULL, NULL, NULL, NULL, &M_ctx));
    PetscCall(CreateMatShell_Ceed(M_ctx, &mat_mass));

    PetscCall(KSPCreate(comm, &ksp));
    PetscCall(KSPSetOptionsPrefix(ksp, "turbulence_spanstats_"));
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
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_mass));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_stats_proj));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_mass));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_setup_sur));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_proj_rhs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Create CeedOperator for statistics collection
PetscErrorCode CreateStatisticCollectionOperator(Ceed ceed, User user, CeedData ceed_data, SpanStatsSetupData stats_data, ProblemData *problem) {
  CeedInt                     num_comp_stats = user->spanstats.num_comp_stats, num_comp_x = problem->dim, num_comp_q;
  Turbulence_SpanStatsContext collect_ctx;
  NewtonianIdealGasContext    newtonian_ig_ctx;
  CeedQFunctionContext        collect_context;
  CeedQFunction               qf_stats_collect;
  CeedOperator                op_stats_collect;

  PetscFunctionBeginUser;
  PetscCallCeed(ceed, CeedBasisGetNumComponents(ceed_data->basis_q, &num_comp_q));

  // Create Operator for statistics collection
  switch (user->phys->state_var) {
    case STATEVAR_PRIMITIVE:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, ChildStatsCollection_Prim, ChildStatsCollection_Prim_loc, &qf_stats_collect));
      break;
    case STATEVAR_CONSERVATIVE:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, ChildStatsCollection_Conserv, ChildStatsCollection_Conserv_loc, &qf_stats_collect));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)user->dm), PETSC_ERR_SUP, "No statisics collection available for chosen state variable");
  }

  if (user->spanstats.do_mms_test) {
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_stats_collect));
    PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, ChildStatsCollectionMMSTest, ChildStatsCollectionMMSTest_loc, &qf_stats_collect));
  }

  {  // Setup Collection Context
    PetscCall(PetscNew(&collect_ctx));
    PetscCallCeed(ceed, CeedQFunctionContextGetData(problem->apply_vol_rhs.qfunction_context, CEED_MEM_HOST, &newtonian_ig_ctx));
    collect_ctx->gas = *newtonian_ig_ctx;

    PetscCallCeed(ceed, CeedQFunctionContextCreate(user->ceed, &collect_context));
    PetscCallCeed(ceed, CeedQFunctionContextSetData(collect_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*collect_ctx), collect_ctx));
    PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(collect_context, CEED_MEM_HOST, FreeContextPetsc));

    PetscCallCeed(ceed, CeedQFunctionContextRegisterDouble(collect_context, "solution time",
                                                           offsetof(struct Turbulence_SpanStatsContext_, solution_time), 1, "Current solution time"));
    PetscCallCeed(
        ceed, CeedQFunctionContextRegisterDouble(collect_context, "previous time", offsetof(struct Turbulence_SpanStatsContext_, previous_time), 1,
                                                 "Previous time statistics collection was done"));

    PetscCallCeed(ceed, CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfunction_context, &newtonian_ig_ctx));
  }

  PetscCallCeed(ceed, CeedQFunctionSetContext(qf_stats_collect, collect_context));
  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&collect_context));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_stats_collect, "q", num_comp_q, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_stats_collect, "q_data", problem->q_data_size_vol, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_stats_collect, "x", num_comp_x, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_stats_collect, "v", num_comp_stats, CEED_EVAL_NONE));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_stats_collect, NULL, NULL, &op_stats_collect));
  PetscCallCeed(ceed, CeedOperatorSetField(op_stats_collect, "q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_stats_collect, "q_data", ceed_data->elem_restr_qd_i, CEED_BASIS_COLLOCATED, ceed_data->q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_stats_collect, "x", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord));
  PetscCallCeed(ceed, CeedOperatorSetField(op_stats_collect, "v", stats_data->elem_restr_child_colloc, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));

  PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(op_stats_collect, "solution time", &user->spanstats.solution_time_label));
  PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(op_stats_collect, "previous time", &user->spanstats.previous_time_label));

  PetscCall(OperatorApplyContextCreate(user->dm, user->spanstats.dm, user->ceed, op_stats_collect, user->q_ceed, NULL, NULL, NULL,
                                       &user->spanstats.op_stats_collect_ctx));

  PetscCall(CeedOperatorCreateLocalVecs(op_stats_collect, DMReturnVecType(user->spanstats.dm), PetscObjectComm((PetscObject)user->spanstats.dm), NULL,
                                        &user->spanstats.Child_Stats_loc));
  PetscCall(VecZeroEntries(user->spanstats.Child_Stats_loc));

  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_stats_collect));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_stats_collect));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Creates operator for calculating error of method of manufactured solution (MMS) test
PetscErrorCode SetupMMSErrorChecking(Ceed ceed, User user, CeedData ceed_data, SpanStatsSetupData stats_data) {
  CeedInt       num_comp_stats = user->spanstats.num_comp_stats, num_comp_x, q_data_size;
  CeedQFunction qf_error;
  CeedOperator  op_error;
  CeedVector    x_ceed, y_ceed;

  PetscFunctionBeginUser;
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(stats_data->elem_restr_parent_qd, &q_data_size));
  PetscCallCeed(ceed, CeedBasisGetNumComponents(stats_data->basis_x, &num_comp_x));

  PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, ChildStatsCollectionMMSTest_Error, ChildStatsCollectionMMSTest_Error_loc, &qf_error));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_error, "q", num_comp_stats, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_error, "qdata", q_data_size, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_error, "x", num_comp_x, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_error, "v", num_comp_stats, CEED_EVAL_INTERP));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_error, NULL, NULL, &op_error));
  PetscCallCeed(ceed, CeedOperatorSetField(op_error, "q", stats_data->elem_restr_parent_stats, stats_data->basis_stats, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_error, "qdata", stats_data->elem_restr_parent_qd, CEED_BASIS_COLLOCATED, stats_data->q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_error, "x", stats_data->elem_restr_parent_x, stats_data->basis_x, stats_data->x_coord));
  PetscCallCeed(ceed, CeedOperatorSetField(op_error, "v", stats_data->elem_restr_parent_stats, stats_data->basis_stats, CEED_VECTOR_ACTIVE));

  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(stats_data->elem_restr_parent_stats, &x_ceed, NULL));
  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(stats_data->elem_restr_parent_stats, &y_ceed, NULL));
  PetscCall(OperatorApplyContextCreate(user->spanstats.dm, user->spanstats.dm, user->ceed, op_error, x_ceed, y_ceed, NULL, NULL,
                                       &user->spanstats.mms_error_ctx));

  PetscCallCeed(ceed, CeedOperatorDestroy(&op_error));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_error));
  PetscCallCeed(ceed, CeedVectorDestroy(&x_ceed));
  PetscCallCeed(ceed, CeedVectorDestroy(&y_ceed));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Setup for statistics collection
PetscErrorCode TurbulenceStatisticsSetup(Ceed ceed, User user, CeedData ceed_data, ProblemData *problem) {
  SpanStatsSetupData stats_data;
  PetscLogStage      stage_stats_setup;

  PetscFunctionBeginUser;
  PetscCall(PetscLogStageGetId("Stats Setup", &stage_stats_setup));
  if (stage_stats_setup == -1) PetscCall(PetscLogStageRegister("Stats Setup", &stage_stats_setup));
  PetscCall(PetscLogStagePush(stage_stats_setup));

  // Create parent DM
  PetscCall(CreateStatsDM(user, problem, user->app_ctx->degree));

  // Create necessary CeedObjects for setting up statistics
  PetscCall(SpanStatsSetupDataCreate(ceed, user, ceed_data, problem, &stats_data));

  // Create SF for communicating child data back their respective parents
  PetscCall(CreateStatsSF(ceed, ceed_data, stats_data, user->dm, user->spanstats.dm, &user->spanstats.sf));

  // Create CeedOperators for statistics collection
  PetscCall(CreateStatisticCollectionOperator(ceed, user, ceed_data, stats_data, problem));

  // Setup KSP and Mat for L^2 projection of statistics
  PetscCall(SetupL2ProjectionStats(ceed, user, ceed_data, stats_data));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-ts_monitor_turbulence_spanstats_mms", &user->spanstats.do_mms_test, NULL));
  if (user->spanstats.do_mms_test) {
    PetscCall(SetupMMSErrorChecking(ceed, user, ceed_data, stats_data));
  }

  {  // Setup stats viewer with prefix
    PetscViewerType viewer_type;
    PetscCall(PetscViewerGetType(user->app_ctx->turb_spanstats_viewer, &viewer_type));
    PetscCall(PetscOptionsSetValue(NULL, "-ts_monitor_turbulence_spanstats_viewer_type", viewer_type));

    PetscCall(PetscViewerSetOptionsPrefix(user->app_ctx->turb_spanstats_viewer, "ts_monitor_turbulence_spanstats_"));
    PetscCall(PetscViewerSetFromOptions(user->app_ctx->turb_spanstats_viewer));
  }

  PetscCall(SpanStatsSetupDataDestroy(stats_data));
  PetscCall(PetscLogStagePop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Collect statistics based on the solution Q
PetscErrorCode CollectStatistics(User user, PetscScalar solution_time, Vec Q) {
  SpanStatsData user_stats = user->spanstats;

  PetscFunctionBeginUser;
  PetscLogStage stage_stats_collect;
  PetscCall(PetscLogStageGetId("Stats Collect", &stage_stats_collect));
  if (stage_stats_collect == -1) PetscCall(PetscLogStageRegister("Stats Collect", &stage_stats_collect));
  PetscCall(PetscLogStagePush(stage_stats_collect));

  PetscCall(UpdateBoundaryValues(user, user->Q_loc, solution_time));
  PetscCallCeed(user->ceed, CeedOperatorSetContextDouble(user_stats.op_stats_collect_ctx->op, user_stats.solution_time_label, &solution_time));
  PetscCall(DMGlobalToLocal(user->dm, Q, INSERT_VALUES, user->Q_loc));
  PetscCall(ApplyAddCeedOperatorLocalToLocal(user->Q_loc, user_stats.Child_Stats_loc, user_stats.op_stats_collect_ctx));

  PetscCallCeed(user->ceed, CeedOperatorSetContextDouble(user_stats.op_stats_collect_ctx->op, user_stats.previous_time_label, &solution_time));

  PetscCall(PetscLogStagePop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Process the child statistics into parent statistics and project them onto stats
PetscErrorCode ProcessStatistics(User user, Vec stats) {
  SpanStatsData      user_stats = user->spanstats;
  const PetscScalar *child_stats;
  PetscScalar       *parent_stats;
  MPI_Datatype       unit;
  Vec                RHS;

  PetscFunctionBeginUser;
  PetscLogStage stage_stats_process;
  PetscCall(PetscLogStageGetId("Stats Process", &stage_stats_process));
  if (stage_stats_process == -1) PetscCall(PetscLogStageRegister("Stats Process", &stage_stats_process));
  PetscCall(PetscLogStagePush(stage_stats_process));

  PetscCall(VecZeroEntries(user_stats.Parent_Stats_loc));

  PetscCall(VecGetArrayRead(user_stats.Child_Stats_loc, &child_stats));
  PetscCall(VecGetArray(user_stats.Parent_Stats_loc, &parent_stats));

  if (user_stats.num_comp_stats == 1) unit = MPIU_REAL;
  else {
    PetscCallMPI(MPI_Type_contiguous(user_stats.num_comp_stats, MPIU_REAL, &unit));
    PetscCallMPI(MPI_Type_commit(&unit));
  }

  PetscCall(PetscSFReduceBegin(user_stats.sf, unit, child_stats, parent_stats, MPI_SUM));
  PetscCall(PetscSFReduceEnd(user_stats.sf, unit, child_stats, parent_stats, MPI_SUM));

  PetscCall(VecRestoreArrayRead(user_stats.Child_Stats_loc, &child_stats));
  PetscCall(VecRestoreArray(user_stats.Parent_Stats_loc, &parent_stats));
  PetscCallMPI(MPI_Type_free(&unit));

  PetscReal solution_time;
  PetscCall(DMGetOutputSequenceNumber(user_stats.dm, NULL, &solution_time));
  PetscReal summing_duration = solution_time - user->app_ctx->cont_time;
  PetscCall(VecScale(user_stats.Parent_Stats_loc, 1 / (summing_duration * user_stats.span_width)));

  // L^2 projection with the parent_data
  PetscCall(DMGetGlobalVector(user_stats.dm, &RHS));
  PetscCall(ApplyCeedOperatorLocalToGlobal(user_stats.Parent_Stats_loc, RHS, user_stats.op_proj_rhs_ctx));

  PetscCall(KSPSolve(user_stats.ksp, RHS, stats));

  PetscCall(DMRestoreGlobalVector(user_stats.dm, &RHS));
  PetscCall(PetscLogStagePop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// TSMonitor for the statistics collection and processing
PetscErrorCode TSMonitor_TurbulenceStatistics(TS ts, PetscInt steps, PetscReal solution_time, Vec Q, void *ctx) {
  User              user = (User)ctx;
  Vec               stats;
  TSConvergedReason reason;
  PetscInt collect_interval = user->app_ctx->turb_spanstats_collect_interval, viewer_interval = user->app_ctx->turb_spanstats_viewer_interval;

  PetscFunctionBeginUser;
  PetscCall(TSGetConvergedReason(ts, &reason));
  // Do not collect or process on the first step of the run (ie. on the initial condition)
  if (steps == user->app_ctx->cont_steps && reason == TS_CONVERGED_ITERATING) PetscFunctionReturn(PETSC_SUCCESS);

  PetscBool run_processing_and_viewer = (steps % viewer_interval == 0 && viewer_interval != -1) || reason != TS_CONVERGED_ITERATING;

  if (steps % collect_interval == 0 || run_processing_and_viewer) {
    PetscCall(CollectStatistics(user, solution_time, Q));

    if (run_processing_and_viewer) {
      PetscCall(DMSetOutputSequenceNumber(user->spanstats.dm, steps, solution_time));
      PetscCall(DMGetGlobalVector(user->spanstats.dm, &stats));
      PetscCall(ProcessStatistics(user, stats));
      if (user->app_ctx->test_type == TESTTYPE_NONE) {
        PetscCall(PetscViewerPushFormat(user->app_ctx->turb_spanstats_viewer, user->app_ctx->turb_spanstats_viewer_format));
        PetscCall(VecView(stats, user->app_ctx->turb_spanstats_viewer));
        PetscCall(PetscViewerPopFormat(user->app_ctx->turb_spanstats_viewer));
      }
      if (user->app_ctx->test_type == TESTTYPE_TURB_SPANSTATS && reason != TS_CONVERGED_ITERATING) {
        PetscCall(RegressionTests_NS(user->app_ctx, stats));
      }
      if (user->spanstats.do_mms_test && reason != TS_CONVERGED_ITERATING) {
        Vec error;
        PetscCall(VecDuplicate(stats, &error));
        PetscCall(ApplyCeedOperatorGlobalToGlobal(stats, error, user->spanstats.mms_error_ctx));
        PetscScalar error_sq = 0;
        PetscCall(VecSum(error, &error_sq));
        PetscScalar l2_error = sqrt(error_sq);
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "l2 error: %.5e\n", l2_error));
      }
      PetscCall(DMRestoreGlobalVector(user->spanstats.dm, &stats));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TurbulenceStatisticsDestroy(User user, CeedData ceed_data) {
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&user->spanstats.Child_Stats_loc));
  PetscCall(VecDestroy(&user->spanstats.Parent_Stats_loc));

  PetscCall(OperatorApplyContextDestroy(user->spanstats.op_stats_collect_ctx));
  PetscCall(OperatorApplyContextDestroy(user->spanstats.op_proj_rhs_ctx));
  PetscCall(OperatorApplyContextDestroy(user->spanstats.mms_error_ctx));

  PetscCall(KSPDestroy(&user->spanstats.ksp));
  PetscCall(PetscSFDestroy(&user->spanstats.sf));
  PetscCall(DMDestroy(&user->spanstats.dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}
