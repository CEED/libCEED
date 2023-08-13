// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Time-stepping functions for Navier-Stokes example using PETSc

#include <ceed.h>
#include <petscdmplex.h>
#include <petscts.h>

#include "../navierstokes.h"
#include "../qfunctions/newtonian_state.h"

// Compute mass matrix for explicit scheme
PetscErrorCode ComputeLumpedMassMatrix(Ceed ceed, DM dm, CeedData ceed_data, Vec M) {
  CeedQFunction        qf_mass;
  CeedOperator         op_mass;
  OperatorApplyContext op_mass_ctx;
  Vec                  Ones_loc;
  CeedInt              num_comp_q, q_data_size;
  PetscFunctionBeginUser;

  // CEED Restriction
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_q, &num_comp_q));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_qd_i, &q_data_size));

  // CEED QFunction
  PetscCall(CreateMassQFunction(ceed, num_comp_q, q_data_size, &qf_mass));

  // CEED Operator
  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "u", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_COLLOCATED, ceed_data->q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "v", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));

  PetscCall(OperatorApplyContextCreate(NULL, dm, ceed, op_mass, NULL, NULL, NULL, NULL, &op_mass_ctx));

  PetscCall(DMGetLocalVector(dm, &Ones_loc));
  PetscCall(VecSet(Ones_loc, 1));
  PetscCall(ApplyCeedOperatorLocalToGlobal(Ones_loc, M, op_mass_ctx));

  // Invert diagonally lumped mass vector for RHS function
  PetscCall(VecReciprocal(M));

  // Cleanup
  PetscCall(OperatorApplyContextDestroy(op_mass_ctx));
  PetscCall(DMRestoreLocalVector(dm, &Ones_loc));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_mass));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_mass));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Insert Boundary values if it's a new time
PetscErrorCode UpdateBoundaryValues(User user, Vec Q_loc, PetscReal t) {
  PetscFunctionBeginUser;
  if (user->time_bc_set != t) {
    PetscCall(DMPlexInsertBoundaryValues(user->dm, PETSC_TRUE, Q_loc, t, NULL, NULL, NULL));
    user->time_bc_set = t;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Update the context label value to new value if necessary.
// @note This only supports labels with scalar label values (ie. not arrays)
PetscErrorCode UpdateContextLabel(Ceed ceed, MPI_Comm comm, PetscScalar update_value, CeedOperator op, CeedContextFieldLabel label) {
  PetscScalar label_value;

  PetscFunctionBeginUser;
  PetscCheck(label, comm, PETSC_ERR_ARG_BADPTR, "Label should be non-NULL");

  {
    size_t             num_elements;
    const PetscScalar *label_values;
    PetscCallCeed(ceed, CeedOperatorGetContextDoubleRead(op, label, &num_elements, &label_values));
    PetscCheck(num_elements == 1, comm, PETSC_ERR_SUP, "%s does not support labels with more than 1 value. Label has %zu values", __func__,
               num_elements);
    label_value = *label_values;
    PetscCallCeed(ceed, CeedOperatorRestoreContextDoubleRead(op, label, &label_values));
  }

  if (label_value != update_value) {
    PetscCallCeed(ceed, CeedOperatorSetContextDouble(op, label, &update_value));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// RHS (Explicit time-stepper) function setup
//   This is the RHS of the ODE, given as u_t = G(t,u)
//   This function takes in a state vector Q and writes into G
PetscErrorCode RHS_NS(TS ts, PetscReal t, Vec Q, Vec G, void *user_data) {
  User        user = *(User *)user_data;
  MPI_Comm    comm = PetscObjectComm((PetscObject)ts);
  PetscScalar dt;
  Vec         Q_loc = user->Q_loc;
  PetscFunctionBeginUser;

  // Update time dependent data
  PetscCall(UpdateBoundaryValues(user, Q_loc, t));
  if (user->phys->solution_time_label) PetscCall(UpdateContextLabel(user->ceed, comm, t, user->op_rhs_ctx->op, user->phys->solution_time_label));
  PetscCall(TSGetTimeStep(ts, &dt));
  if (user->phys->timestep_size_label) PetscCall(UpdateContextLabel(user->ceed, comm, dt, user->op_rhs_ctx->op, user->phys->timestep_size_label));

  PetscCall(ApplyCeedOperatorGlobalToGlobal(Q, G, user->op_rhs_ctx));

  // Inverse of the lumped mass matrix (M is Minv)
  PetscCall(VecPointwiseMult(G, G, user->M_inv));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Surface forces function setup
static PetscErrorCode Surface_Forces_NS(DM dm, Vec G_loc, PetscInt num_walls, const PetscInt walls[], PetscScalar *reaction_force) {
  DMLabel            face_label;
  const PetscScalar *g;
  PetscInt           dof, dim = 3;
  MPI_Comm           comm;
  PetscSection       s;

  PetscFunctionBeginUser;
  PetscCall(PetscArrayzero(reaction_force, num_walls * dim));
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(DMGetLabel(dm, "Face Sets", &face_label));
  PetscCall(VecGetArrayRead(G_loc, &g));
  for (PetscInt w = 0; w < num_walls; w++) {
    const PetscInt wall = walls[w];
    IS             wall_is;
    PetscCall(DMGetLocalSection(dm, &s));
    PetscCall(DMLabelGetStratumIS(face_label, wall, &wall_is));
    if (wall_is) {  // There exist such points on this process
      PetscInt        num_points;
      PetscInt        num_comp = 0;
      const PetscInt *points;
      PetscCall(PetscSectionGetFieldComponents(s, 0, &num_comp));
      PetscCall(ISGetSize(wall_is, &num_points));
      PetscCall(ISGetIndices(wall_is, &points));
      for (PetscInt i = 0; i < num_points; i++) {
        const PetscInt           p = points[i];
        const StateConservative *r;
        PetscCall(DMPlexPointLocalRead(dm, p, g, &r));
        PetscCall(PetscSectionGetDof(s, p, &dof));
        for (PetscInt node = 0; node < dof / num_comp; node++) {
          for (PetscInt j = 0; j < 3; j++) {
            reaction_force[w * dim + j] -= r[node].momentum[j];
          }
        }
      }
      PetscCall(ISRestoreIndices(wall_is, &points));
    }
    PetscCall(ISDestroy(&wall_is));
  }
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, reaction_force, dim * num_walls, MPIU_SCALAR, MPI_SUM, comm));
  //  Restore Vectors
  PetscCall(VecRestoreArrayRead(G_loc, &g));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Implicit time-stepper function setup
PetscErrorCode IFunction_NS(TS ts, PetscReal t, Vec Q, Vec Q_dot, Vec G, void *user_data) {
  User         user = *(User *)user_data;
  MPI_Comm     comm = PetscObjectComm((PetscObject)ts);
  PetscScalar  dt;
  Vec          Q_loc = user->Q_loc, Q_dot_loc = user->Q_dot_loc, G_loc;
  PetscMemType q_mem_type, q_dot_mem_type, g_mem_type;
  PetscFunctionBeginUser;

  // Get local vectors
  PetscCall(DMGetNamedLocalVector(user->dm, "ResidualLocal", &G_loc));

  // Update time dependent data
  PetscCall(UpdateBoundaryValues(user, Q_loc, t));
  if (user->phys->solution_time_label) PetscCall(UpdateContextLabel(user->ceed, comm, t, user->op_ifunction, user->phys->solution_time_label));
  PetscCall(TSGetTimeStep(ts, &dt));
  if (user->phys->timestep_size_label) PetscCall(UpdateContextLabel(user->ceed, comm, dt, user->op_ifunction, user->phys->timestep_size_label));

  // Global-to-local
  PetscCall(DMGlobalToLocalBegin(user->dm, Q, INSERT_VALUES, Q_loc));
  PetscCall(DMGlobalToLocalBegin(user->dm, Q_dot, INSERT_VALUES, Q_dot_loc));
  PetscCall(DMGlobalToLocalEnd(user->dm, Q, INSERT_VALUES, Q_loc));
  PetscCall(DMGlobalToLocalEnd(user->dm, Q_dot, INSERT_VALUES, Q_dot_loc));

  // Place PETSc vectors in CEED vectors
  PetscCall(VecReadP2C(Q_loc, &q_mem_type, user->q_ceed));
  PetscCall(VecReadP2C(Q_dot_loc, &q_dot_mem_type, user->q_dot_ceed));
  PetscCall(VecP2C(G_loc, &g_mem_type, user->g_ceed));

  // Apply CEED operator
  PetscCall(PetscLogEventBegin(FLUIDS_CeedOperatorApply, Q, G, 0, 0));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCeed(user->ceed, CeedOperatorApply(user->op_ifunction, user->q_ceed, user->g_ceed, CEED_REQUEST_IMMEDIATE));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogEventEnd(FLUIDS_CeedOperatorApply, Q, G, 0, 0));

  // Restore vectors
  PetscCall(VecReadC2P(user->q_ceed, q_mem_type, Q_loc));
  PetscCall(VecReadC2P(user->q_dot_ceed, q_dot_mem_type, Q_dot_loc));
  PetscCall(VecC2P(user->g_ceed, g_mem_type, G_loc));

  if (user->app_ctx->sgs_model_type == SGS_MODEL_DATA_DRIVEN) {
    PetscCall(SGS_DD_ModelApplyIFunction(user, Q_loc, G_loc));
  }

  // Local-to-Global
  PetscCall(VecZeroEntries(G));
  PetscCall(DMLocalToGlobal(user->dm, G_loc, ADD_VALUES, G));

  // Restore vectors
  PetscCall(DMRestoreNamedLocalVector(user->dm, "ResidualLocal", &G_loc));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormPreallocation(User user, PetscBool pbdiagonal, Mat J, CeedVector *coo_values) {
  PetscCount ncoo;
  PetscInt  *rows_petsc, *cols_petsc;

  PetscFunctionBeginUser;
  if (pbdiagonal) {
    CeedSize l_size;
    PetscCallCeed(user->ceed, CeedOperatorGetActiveVectorLengths(user->op_ijacobian, &l_size, NULL));
    ncoo       = l_size * 5;
    rows_petsc = malloc(ncoo * sizeof(rows_petsc));
    cols_petsc = malloc(ncoo * sizeof(cols_petsc));
    for (PetscCount n = 0; n < l_size / 5; n++) {
      for (PetscInt i = 0; i < 5; i++) {
        for (PetscInt j = 0; j < 5; j++) {
          rows_petsc[(n * 5 + i) * 5 + j] = n * 5 + i;
          cols_petsc[(n * 5 + i) * 5 + j] = n * 5 + j;
        }
      }
    }
  } else {
    CeedInt *rows_ceed, *cols_ceed;
    PetscCallCeed(user->ceed, CeedOperatorLinearAssembleSymbolic(user->op_ijacobian, &ncoo, &rows_ceed, &cols_ceed));
    PetscCall(IntArrayC2P(ncoo, &rows_ceed, &rows_petsc));
    PetscCall(IntArrayC2P(ncoo, &cols_ceed, &cols_petsc));
  }
  PetscCall(MatSetPreallocationCOOLocal(J, ncoo, rows_petsc, cols_petsc));
  free(rows_petsc);
  free(cols_petsc);
  PetscCallCeed(user->ceed, CeedVectorCreate(user->ceed, ncoo, coo_values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormSetValues(User user, PetscBool pbdiagonal, Mat J, CeedVector coo_values) {
  CeedMemType        mem_type = CEED_MEM_HOST;
  const PetscScalar *values;
  MatType            mat_type;

  PetscFunctionBeginUser;
  PetscCall(MatGetType(J, &mat_type));
  if (strstr(mat_type, "kokkos") || strstr(mat_type, "cusparse")) mem_type = CEED_MEM_DEVICE;
  if (pbdiagonal) {
    PetscCall(PetscLogEventBegin(FLUIDS_CeedOperatorAssemblePointBlockDiagonal, J, 0, 0, 0));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCeed(user->ceed, CeedOperatorLinearAssemblePointBlockDiagonal(user->op_ijacobian, coo_values, CEED_REQUEST_IMMEDIATE));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogEventEnd(FLUIDS_CeedOperatorAssemblePointBlockDiagonal, J, 0, 0, 0));
  } else {
    PetscCall(PetscLogEventBegin(FLUIDS_CeedOperatorAssemble, J, 0, 0, 0));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCeed(user->ceed, CeedOperatorLinearAssemble(user->op_ijacobian, coo_values));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogEventEnd(FLUIDS_CeedOperatorAssemble, J, 0, 0, 0));
  }
  PetscCallCeed(user->ceed, CeedVectorGetArrayRead(coo_values, mem_type, &values));
  PetscCall(MatSetValuesCOO(J, values, INSERT_VALUES));
  PetscCallCeed(user->ceed, CeedVectorRestoreArrayRead(coo_values, &values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormIJacobian_NS(TS ts, PetscReal t, Vec Q, Vec Q_dot, PetscReal shift, Mat J, Mat J_pre, void *user_data) {
  User      user = *(User *)user_data;
  Ceed      ceed = user->ceed;
  PetscBool J_is_shell, J_is_mffd, J_pre_is_shell;
  PetscFunctionBeginUser;
  if (user->phys->ijacobian_time_shift_label)
    PetscCallCeed(ceed, CeedOperatorSetContextDouble(user->op_ijacobian, user->phys->ijacobian_time_shift_label, &shift));
  PetscCall(PetscObjectTypeCompare((PetscObject)J, MATMFFD, &J_is_mffd));
  PetscCall(PetscObjectTypeCompare((PetscObject)J, MATSHELL, &J_is_shell));
  PetscCall(PetscObjectTypeCompare((PetscObject)J_pre, MATSHELL, &J_pre_is_shell));
  if (!user->matrices_set_up) {
    if (J_is_shell) {
      OperatorApplyContext op_ijacobian_ctx;
      OperatorApplyContextCreate(user->dm, user->dm, user->ceed, user->op_ijacobian, user->q_ceed, user->g_ceed, user->Q_dot_loc, NULL,
                                 &op_ijacobian_ctx);
      PetscCall(MatShellSetContext(J, op_ijacobian_ctx));
      PetscCall(MatShellSetContextDestroy(J, (PetscErrorCode(*)(void *))OperatorApplyContextDestroy));
      PetscCall(MatShellSetOperation(J, MATOP_MULT, (void (*)(void))MatMult_Ceed));
      PetscCall(MatShellSetOperation(J, MATOP_GET_DIAGONAL, (void (*)(void))MatGetDiag_Ceed));
      PetscCall(MatSetUp(J));
    }
    if (!J_pre_is_shell) {
      PetscCall(FormPreallocation(user, user->app_ctx->pmat_pbdiagonal, J_pre, &user->coo_values_pmat));
    }
    if (J != J_pre && !J_is_shell && !J_is_mffd) {
      PetscCall(FormPreallocation(user, PETSC_FALSE, J, &user->coo_values_amat));
    }
    user->matrices_set_up = true;
  }
  if (!J_pre_is_shell) {
    PetscCall(FormSetValues(user, user->app_ctx->pmat_pbdiagonal, J_pre, user->coo_values_pmat));
  }
  if (user->coo_values_amat) {
    PetscCall(FormSetValues(user, PETSC_FALSE, J, user->coo_values_amat));
  } else if (J_is_mffd) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode WriteOutput(User user, Vec Q, PetscInt step_no, PetscScalar time) {
  Vec         Q_loc;
  char        file_path[PETSC_MAX_PATH_LEN];
  PetscViewer viewer;
  PetscFunctionBeginUser;

  if (user->app_ctx->checkpoint_vtk) {
    // Set up output
    PetscCall(DMGetLocalVector(user->dm, &Q_loc));
    PetscCall(PetscObjectSetName((PetscObject)Q_loc, "StateVec"));
    PetscCall(VecZeroEntries(Q_loc));
    PetscCall(DMGlobalToLocal(user->dm, Q, INSERT_VALUES, Q_loc));

    // Output
    PetscCall(PetscSNPrintf(file_path, sizeof file_path, "%s/ns-%03" PetscInt_FMT ".vtu", user->app_ctx->output_dir, step_no));

    PetscCall(PetscViewerVTKOpen(PetscObjectComm((PetscObject)Q), file_path, FILE_MODE_WRITE, &viewer));
    PetscCall(VecView(Q_loc, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    if (user->dm_viz) {
      Vec         Q_refined, Q_refined_loc;
      char        file_path_refined[PETSC_MAX_PATH_LEN];
      PetscViewer viewer_refined;

      PetscCall(DMGetGlobalVector(user->dm_viz, &Q_refined));
      PetscCall(DMGetLocalVector(user->dm_viz, &Q_refined_loc));
      PetscCall(PetscObjectSetName((PetscObject)Q_refined_loc, "Refined"));

      PetscCall(MatInterpolate(user->interp_viz, Q, Q_refined));
      PetscCall(VecZeroEntries(Q_refined_loc));
      PetscCall(DMGlobalToLocal(user->dm_viz, Q_refined, INSERT_VALUES, Q_refined_loc));

      PetscCall(
          PetscSNPrintf(file_path_refined, sizeof file_path_refined, "%s/nsrefined-%03" PetscInt_FMT ".vtu", user->app_ctx->output_dir, step_no));

      PetscCall(PetscViewerVTKOpen(PetscObjectComm((PetscObject)Q_refined), file_path_refined, FILE_MODE_WRITE, &viewer_refined));
      PetscCall(VecView(Q_refined_loc, viewer_refined));
      PetscCall(DMRestoreLocalVector(user->dm_viz, &Q_refined_loc));
      PetscCall(DMRestoreGlobalVector(user->dm_viz, &Q_refined));
      PetscCall(PetscViewerDestroy(&viewer_refined));
    }
    PetscCall(DMRestoreLocalVector(user->dm, &Q_loc));
  }

  // Save data in a binary file for continuation of simulations
  if (user->app_ctx->add_stepnum2bin) {
    PetscCall(PetscSNPrintf(file_path, sizeof file_path, "%s/ns-solution-%" PetscInt_FMT ".bin", user->app_ctx->output_dir, step_no));
  } else {
    PetscCall(PetscSNPrintf(file_path, sizeof file_path, "%s/ns-solution.bin", user->app_ctx->output_dir));
  }
  PetscCall(PetscViewerBinaryOpen(user->comm, file_path, FILE_MODE_WRITE, &viewer));

  PetscInt32 token = PetscDefined(USE_64BIT_INDICES) ? FLUIDS_FILE_TOKEN_64 : FLUIDS_FILE_TOKEN_32;
  PetscCall(PetscViewerBinaryWrite(viewer, &token, 1, PETSC_INT32));
  PetscCall(PetscViewerBinaryWrite(viewer, &step_no, 1, PETSC_INT));
  time /= user->units->second;  // Dimensionalize time back
  PetscCall(PetscViewerBinaryWrite(viewer, &time, 1, PETSC_REAL));
  PetscCall(VecView(Q, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// CSV Monitor
PetscErrorCode TSMonitor_WallForce(TS ts, PetscInt step_no, PetscReal time, Vec Q, void *ctx) {
  User              user = ctx;
  Vec               G_loc;
  PetscInt          num_wall = user->app_ctx->wall_forces.num_wall, dim = 3;
  const PetscInt   *walls  = user->app_ctx->wall_forces.walls;
  PetscViewer       viewer = user->app_ctx->wall_forces.viewer;
  PetscViewerFormat format = user->app_ctx->wall_forces.viewer_format;
  PetscScalar      *reaction_force;
  PetscBool         iascii;

  PetscFunctionBeginUser;
  if (!viewer) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMGetNamedLocalVector(user->dm, "ResidualLocal", &G_loc));
  PetscCall(PetscMalloc1(num_wall * dim, &reaction_force));
  PetscCall(Surface_Forces_NS(user->dm, G_loc, num_wall, walls, reaction_force));
  PetscCall(DMRestoreNamedLocalVector(user->dm, "ResidualLocal", &G_loc));

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));

  if (iascii) {
    if (format == PETSC_VIEWER_ASCII_CSV && !user->app_ctx->wall_forces.header_written) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Step,Time,Wall,ForceX,ForceY,ForceZ\n"));
      user->app_ctx->wall_forces.header_written = PETSC_TRUE;
    }
    for (PetscInt w = 0; w < num_wall; w++) {
      PetscInt wall = walls[w];
      if (format == PETSC_VIEWER_ASCII_CSV) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT ",%g,%" PetscInt_FMT ",%g,%g,%g\n", step_no, time, wall,
                                         reaction_force[w * dim + 0], reaction_force[w * dim + 1], reaction_force[w * dim + 2]));

      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer, "Wall %" PetscInt_FMT " Forces: Force_x = %12g, Force_y = %12g, Force_z = %12g\n", wall,
                                         reaction_force[w * dim + 0], reaction_force[w * dim + 1], reaction_force[w * dim + 2]));
      }
    }
  }
  PetscCall(PetscFree(reaction_force));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// User provided TS Monitor
PetscErrorCode TSMonitor_NS(TS ts, PetscInt step_no, PetscReal time, Vec Q, void *ctx) {
  User user = ctx;
  PetscFunctionBeginUser;

  // Print every 'checkpoint_interval' steps
  if (user->app_ctx->checkpoint_interval <= 0 || step_no % user->app_ctx->checkpoint_interval != 0 ||
      (user->app_ctx->cont_steps == step_no && step_no != 0)) {
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(WriteOutput(user, Q, step_no, time));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// TS: Create, setup, and solve
PetscErrorCode TSSolve_NS(DM dm, User user, AppCtx app_ctx, Physics phys, Vec *Q, PetscScalar *f_time, TS *ts) {
  MPI_Comm    comm = user->comm;
  TSAdapt     adapt;
  PetscScalar final_time;
  PetscFunctionBeginUser;

  PetscCall(TSCreate(comm, ts));
  PetscCall(TSSetDM(*ts, dm));
  if (phys->implicit) {
    PetscCall(TSSetType(*ts, TSBDF));
    if (user->op_ifunction) {
      PetscCall(TSSetIFunction(*ts, NULL, IFunction_NS, &user));
    } else {  // Implicit integrators can fall back to using an RHSFunction
      PetscCall(TSSetRHSFunction(*ts, NULL, RHS_NS, &user));
    }
    if (user->op_ijacobian) {
      PetscCall(DMTSSetIJacobian(dm, FormIJacobian_NS, &user));
      if (app_ctx->amat_type) {
        Mat Pmat, Amat;
        PetscCall(DMCreateMatrix(dm, &Pmat));
        PetscCall(DMSetMatType(dm, app_ctx->amat_type));
        PetscCall(DMCreateMatrix(dm, &Amat));
        PetscCall(TSSetIJacobian(*ts, Amat, Pmat, NULL, NULL));
        PetscCall(MatDestroy(&Amat));
        PetscCall(MatDestroy(&Pmat));
      }
    }
  } else {
    PetscCheck(user->op_rhs_ctx, comm, PETSC_ERR_ARG_NULL, "Problem does not provide RHSFunction");
    PetscCall(TSSetType(*ts, TSRK));
    PetscCall(TSRKSetType(*ts, TSRK5F));
    PetscCall(TSSetRHSFunction(*ts, NULL, RHS_NS, &user));
  }
  PetscCall(TSSetMaxTime(*ts, 500. * user->units->second));
  PetscCall(TSSetExactFinalTime(*ts, TS_EXACTFINALTIME_STEPOVER));
  if (app_ctx->test_type == TESTTYPE_NONE) PetscCall(TSSetErrorIfStepFails(*ts, PETSC_FALSE));
  PetscCall(TSSetTimeStep(*ts, 1.e-2 * user->units->second));
  if (app_ctx->test_type != TESTTYPE_NONE) {
    PetscCall(TSSetMaxSteps(*ts, 10));
  }
  PetscCall(TSGetAdapt(*ts, &adapt));
  PetscCall(TSAdaptSetStepLimits(adapt, 1.e-12 * user->units->second, 1.e2 * user->units->second));
  PetscCall(TSSetFromOptions(*ts));
  user->time_bc_set = -1.0;   // require all BCs be updated
  if (app_ctx->cont_steps) {  // continue from previous timestep data
    PetscInt    count;
    PetscViewer viewer;

    if (app_ctx->cont_time <= 0) {  // Legacy files did not include step number and time
      PetscCall(PetscViewerBinaryOpen(comm, app_ctx->cont_time_file, FILE_MODE_READ, &viewer));
      PetscCall(PetscViewerBinaryRead(viewer, &app_ctx->cont_time, 1, &count, PETSC_REAL));
      PetscCall(PetscViewerDestroy(&viewer));
      PetscCheck(app_ctx->cont_steps != -1, comm, PETSC_ERR_ARG_INCOMP,
                 "-continue step number not specified, but checkpoint file does not contain a step number (likely written by older code version)");
    }
    PetscCall(TSSetTime(*ts, app_ctx->cont_time * user->units->second));
    PetscCall(TSSetStepNumber(*ts, app_ctx->cont_steps));
  }
  if (app_ctx->test_type == TESTTYPE_NONE) {
    PetscCall(TSMonitorSet(*ts, TSMonitor_NS, user, NULL));
  }
  if (app_ctx->wall_forces.viewer) {
    PetscCall(TSMonitorSet(*ts, TSMonitor_WallForce, user, NULL));
  }
  if (app_ctx->turb_spanstats_enable) {
    PetscCall(TSMonitorSet(*ts, TSMonitor_TurbulenceStatistics, user, NULL));
    CeedScalar previous_time = app_ctx->cont_time * user->units->second;
    PetscCallCeed(user->ceed,
                  CeedOperatorSetContextDouble(user->spanstats.op_stats_collect_ctx->op, user->spanstats.previous_time_label, &previous_time));
  }
  if (app_ctx->diff_filter_monitor) PetscCall(TSMonitorSet(*ts, TSMonitor_DifferentialFilter, user, NULL));

  // Solve
  PetscReal start_time;
  PetscInt  start_step;
  PetscCall(TSGetTime(*ts, &start_time));
  PetscCall(TSGetStepNumber(*ts, &start_step));

  PetscPreLoadBegin(PETSC_FALSE, "Fluids Solve");
  PetscCall(TSSetTime(*ts, start_time));
  PetscCall(TSSetStepNumber(*ts, start_step));
  if (PetscPreLoadingOn) {
    // LCOV_EXCL_START
    SNES      snes;
    Vec       Q_preload;
    PetscReal rtol;
    PetscCall(VecDuplicate(*Q, &Q_preload));
    PetscCall(VecCopy(*Q, Q_preload));
    PetscCall(TSGetSNES(*ts, &snes));
    PetscCall(SNESGetTolerances(snes, NULL, &rtol, NULL, NULL, NULL));
    PetscCall(SNESSetTolerances(snes, PETSC_DEFAULT, .99, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(TSSetSolution(*ts, *Q));
    PetscCall(TSStep(*ts));
    PetscCall(SNESSetTolerances(snes, PETSC_DEFAULT, rtol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(VecDestroy(&Q_preload));
    // LCOV_EXCL_STOP
  } else {
    PetscCall(PetscBarrier((PetscObject)*ts));
    PetscCall(TSSolve(*ts, *Q));
  }
  PetscPreLoadEnd();

  PetscCall(TSGetSolveTime(*ts, &final_time));
  *f_time = final_time;

  if (app_ctx->test_type == TESTTYPE_NONE) {
    PetscInt step_no;
    PetscCall(TSGetStepNumber(*ts, &step_no));
    if (user->app_ctx->checkpoint_interval > 0 || user->app_ctx->checkpoint_interval == -1) {
      PetscCall(WriteOutput(user, *Q, step_no, final_time));
    }

    PetscLogStage stage_id;
    PetscStageLog stage_log;

    PetscCall(PetscLogStageGetId("Fluids Solve", &stage_id));
    PetscCall(PetscLogGetStageLog(&stage_log));
    // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Time taken for solution (sec): %g\n", stage_log->stageInfo[stage_id].perfInfo.time));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
