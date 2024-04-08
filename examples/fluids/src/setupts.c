// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
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

// Insert Boundary values if it's a new time
PetscErrorCode UpdateBoundaryValues(User user, Vec Q_loc, PetscReal t) {
  PetscFunctionBeginUser;
  if (user->time_bc_set != t) {
    PetscCall(DMPlexInsertBoundaryValues(user->dm, PETSC_TRUE, Q_loc, t, NULL, NULL, NULL));
    user->time_bc_set = t;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// RHS (Explicit time-stepper) function setup
//   This is the RHS of the ODE, given as u_t = G(t,u)
//   This function takes in a state vector Q and writes into G
PetscErrorCode RHS_NS(TS ts, PetscReal t, Vec Q, Vec G, void *user_data) {
  User         user = *(User *)user_data;
  Ceed         ceed = user->ceed;
  PetscScalar  dt;
  Vec          Q_loc = user->Q_loc;
  PetscMemType q_mem_type;

  PetscFunctionBeginUser;
  // Update time dependent data
  PetscCall(UpdateBoundaryValues(user, Q_loc, t));
  if (user->phys->solution_time_label) PetscCallCeed(ceed, CeedOperatorSetContextDouble(user->op_rhs_ctx->op, user->phys->solution_time_label, &t));
  PetscCall(TSGetTimeStep(ts, &dt));
  if (user->phys->timestep_size_label) PetscCallCeed(ceed, CeedOperatorSetContextDouble(user->op_rhs_ctx->op, user->phys->timestep_size_label, &dt));

  PetscCall(ApplyCeedOperatorGlobalToGlobal(Q, G, user->op_rhs_ctx));

  PetscCall(VecReadPetscToCeed(Q_loc, &q_mem_type, user->q_ceed));

  // Inverse of the mass matrix
  PetscCall(KSPSolve(user->mass_ksp, G, G));

  PetscCall(VecReadCeedToPetsc(user->q_ceed, q_mem_type, Q_loc));
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
  Ceed         ceed = user->ceed;
  PetscScalar  dt;
  Vec          Q_loc = user->Q_loc, Q_dot_loc = user->Q_dot_loc, G_loc;
  PetscMemType q_mem_type, q_dot_mem_type, g_mem_type;

  PetscFunctionBeginUser;
  // Get local vectors
  PetscCall(DMGetNamedLocalVector(user->dm, "ResidualLocal", &G_loc));

  // Update time dependent data
  PetscCall(UpdateBoundaryValues(user, Q_loc, t));
  if (user->phys->solution_time_label) PetscCallCeed(ceed, CeedOperatorSetContextDouble(user->op_ifunction, user->phys->solution_time_label, &t));
  PetscCall(TSGetTimeStep(ts, &dt));
  if (user->phys->timestep_size_label) PetscCallCeed(ceed, CeedOperatorSetContextDouble(user->op_ifunction, user->phys->timestep_size_label, &dt));

  // Global-to-local
  PetscCall(DMGlobalToLocalBegin(user->dm, Q, INSERT_VALUES, Q_loc));
  PetscCall(DMGlobalToLocalBegin(user->dm, Q_dot, INSERT_VALUES, Q_dot_loc));
  PetscCall(DMGlobalToLocalEnd(user->dm, Q, INSERT_VALUES, Q_loc));
  PetscCall(DMGlobalToLocalEnd(user->dm, Q_dot, INSERT_VALUES, Q_dot_loc));

  // Place PETSc vectors in CEED vectors
  PetscCall(VecReadPetscToCeed(Q_loc, &q_mem_type, user->q_ceed));
  PetscCall(VecReadPetscToCeed(Q_dot_loc, &q_dot_mem_type, user->q_dot_ceed));
  PetscCall(VecPetscToCeed(G_loc, &g_mem_type, user->g_ceed));

  // Apply CEED operator
  PetscCall(PetscLogEventBegin(FLUIDS_CeedOperatorApply, Q, G, 0, 0));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCeed(user->ceed, CeedOperatorApply(user->op_ifunction, user->q_ceed, user->g_ceed, CEED_REQUEST_IMMEDIATE));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogEventEnd(FLUIDS_CeedOperatorApply, Q, G, 0, 0));

  // Restore vectors
  PetscCall(VecReadCeedToPetsc(user->q_ceed, q_mem_type, Q_loc));
  PetscCall(VecReadCeedToPetsc(user->q_dot_ceed, q_dot_mem_type, Q_dot_loc));
  PetscCall(VecCeedToPetsc(user->g_ceed, g_mem_type, G_loc));

  if (user->app_ctx->sgs_model_type == SGS_MODEL_DATA_DRIVEN) {
    PetscCall(SgsDDApplyIFunction(user, Q_loc, G_loc));
  }

  // Local-to-Global
  PetscCall(VecZeroEntries(G));
  PetscCall(DMLocalToGlobal(user->dm, G_loc, ADD_VALUES, G));

  // Restore vectors
  PetscCall(DMRestoreNamedLocalVector(user->dm, "ResidualLocal", &G_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormIJacobian_NS(TS ts, PetscReal t, Vec Q, Vec Q_dot, PetscReal shift, Mat J, Mat J_pre, void *user_data) {
  User      user = *(User *)user_data;
  Ceed      ceed = user->ceed;
  PetscBool J_is_matceed, J_is_mffd, J_pre_is_matceed, J_pre_is_mffd;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectTypeCompare((PetscObject)J, MATMFFD, &J_is_mffd));
  PetscCall(PetscObjectTypeCompare((PetscObject)J, MATCEED, &J_is_matceed));
  PetscCall(PetscObjectTypeCompare((PetscObject)J_pre, MATMFFD, &J_pre_is_mffd));
  PetscCall(PetscObjectTypeCompare((PetscObject)J_pre, MATCEED, &J_pre_is_matceed));
  if (user->phys->ijacobian_time_shift_label) {
    CeedOperator op_ijacobian;

    PetscCall(MatCeedGetCeedOperators(user->mat_ijacobian, &op_ijacobian, NULL));
    PetscCallCeed(ceed, CeedOperatorSetContextDouble(op_ijacobian, user->phys->ijacobian_time_shift_label, &shift));
  }

  if (J_is_matceed || J_is_mffd) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  } else PetscCall(MatCeedAssembleCOO(user->mat_ijacobian, J));

  if (J_pre_is_matceed && J != J_pre) {
    PetscCall(MatAssemblyBegin(J_pre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J_pre, MAT_FINAL_ASSEMBLY));
  } else if (!J_pre_is_matceed && !J_pre_is_mffd && J != J_pre) {
    PetscCall(MatCeedAssembleCOO(user->mat_ijacobian, J_pre));
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
  PetscCall(TSSetApplicationContext(*ts, user));
  if (phys->implicit) {
    PetscCall(TSSetType(*ts, TSBDF));
    if (user->op_ifunction) {
      PetscCall(TSSetIFunction(*ts, NULL, IFunction_NS, &user));
    } else {  // Implicit integrators can fall back to using an RHSFunction
      PetscCall(TSSetRHSFunction(*ts, NULL, RHS_NS, &user));
    }
    if (user->mat_ijacobian) {
      PetscCall(DMTSSetIJacobian(dm, FormIJacobian_NS, &user));
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
  PetscCall(TSGetAdapt(*ts, &adapt));
  PetscCall(TSAdaptSetStepLimits(adapt, 1.e-12 * user->units->second, 1.e2 * user->units->second));
  PetscCall(TSSetFromOptions(*ts));
  if (user->mat_ijacobian) {
    if (app_ctx->amat_type && !strcmp(app_ctx->amat_type, MATSHELL)) {
      SNES snes;
      KSP  ksp;
      Mat  Pmat, Amat;

      PetscCall(TSGetSNES(*ts, &snes));
      PetscCall(SNESGetKSP(snes, &ksp));
      PetscCall(CreateSolveOperatorsFromMatCeed(ksp, user->mat_ijacobian, PETSC_FALSE, &Amat, &Pmat));
      PetscCall(TSSetIJacobian(*ts, user->mat_ijacobian, Pmat, NULL, NULL));
      PetscCall(MatDestroy(&Amat));
      PetscCall(MatDestroy(&Pmat));
    }
  }
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

  if (app_ctx->sgs_train_enable) {
    PetscCall(TSMonitorSet(*ts, TSMonitor_SGS_DD_Training, user, NULL));
    PetscCall(TSSetPostStep(*ts, TSPostStep_SGS_DD_Training));
  }
  // Solve
  PetscReal start_time;
  PetscInt  start_step;
  PetscCall(TSGetTime(*ts, &start_time));
  PetscCall(TSGetStepNumber(*ts, &start_step));

  PetscCall(PetscLogDefaultBegin());  // So we can use PetscLogStageGetPerfInfo without -log_view
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
    PetscCall(TSSetSolution(*ts, Q_preload));
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

    PetscLogStage      stage_id;
    PetscEventPerfInfo stage_perf;

    PetscCall(PetscLogStageGetId("Fluids Solve", &stage_id));
    PetscCall(PetscLogStageGetPerfInfo(stage_id, &stage_perf));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Time taken for solution (sec): %g\n", stage_perf.time));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
