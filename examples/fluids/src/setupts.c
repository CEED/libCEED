// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

/// @file
/// Time-stepping functions for Navier-Stokes example using PETSc

#include "../navierstokes.h"
#include "../qfunctions/mass.h"

// Compute mass matrix for explicit scheme
PetscErrorCode ComputeLumpedMassMatrix(Ceed ceed, DM dm, CeedData ceed_data,
                                       Vec M) {
  Vec            M_loc;
  CeedQFunction  qf_mass;
  CeedOperator   op_mass;
  CeedVector     m_ceed, ones_vec;
  CeedInt        num_comp_q, q_data_size;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // CEED Restriction
  CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_q, &num_comp_q);
  CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_qd_i, &q_data_size);
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &m_ceed, NULL);
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &ones_vec, NULL);
  CeedVectorSetValue(ones_vec, 1.0);

  // CEED QFunction
  CeedQFunctionCreateInterior(ceed, 1, Mass, Mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "q", num_comp_q, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_mass, "q_data", q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_mass, "v", num_comp_q, CEED_EVAL_INTERP);

  // CEED Operator
  CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass);
  CeedOperatorSetField(op_mass, "q", ceed_data->elem_restr_q, ceed_data->basis_q,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "q_data", ceed_data->elem_restr_qd_i,
                       CEED_BASIS_COLLOCATED, ceed_data->q_data);
  CeedOperatorSetField(op_mass, "v", ceed_data->elem_restr_q, ceed_data->basis_q,
                       CEED_VECTOR_ACTIVE);

  // Place PETSc vector in CEED vector
  CeedScalar *m;
  PetscMemType m_mem_type;
  ierr = DMGetLocalVector(dm, &M_loc); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(M_loc, (PetscScalar **)&m, &m_mem_type);
  CHKERRQ(ierr);
  CeedVectorSetArray(m_ceed, MemTypeP2C(m_mem_type), CEED_USE_POINTER, m);

  // Apply CEED Operator
  CeedOperatorApply(op_mass, ones_vec, m_ceed, CEED_REQUEST_IMMEDIATE);

  // Restore vectors
  CeedVectorTakeArray(m_ceed, MemTypeP2C(m_mem_type), NULL);
  ierr = VecRestoreArrayReadAndMemType(M_loc, (const PetscScalar **)&m);
  CHKERRQ(ierr);

  // Local-to-Global
  ierr = VecZeroEntries(M); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dm, M_loc, ADD_VALUES, M); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &M_loc); CHKERRQ(ierr);

  // Invert diagonally lumped mass vector for RHS function
  ierr = VecReciprocal(M); CHKERRQ(ierr);

  // Cleanup
  CeedVectorDestroy(&ones_vec);
  CeedVectorDestroy(&m_ceed);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_mass);

  PetscFunctionReturn(0);
}

// RHS (Explicit time-stepper) function setup
//   This is the RHS of the ODE, given as u_t = G(t,u)
//   This function takes in a state vector Q and writes into G
PetscErrorCode RHS_NS(TS ts, PetscReal t, Vec Q, Vec G, void *user_data) {

  User           user = *(User *)user_data;
  PetscScalar    *q, *g;
  Vec            Q_loc, G_loc;
  PetscMemType   q_mem_type, g_mem_type;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // Update EulerContext
  if (user->phys->has_curr_time) user->phys->euler_ctx->curr_time = t;

  // Get local vectors
  ierr = DMGetLocalVector(user->dm, &Q_loc); CHKERRQ(ierr);
  ierr = DMGetLocalVector(user->dm, &G_loc); CHKERRQ(ierr);

  // Global-to-local
  ierr = VecZeroEntries(Q_loc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm, Q, INSERT_VALUES, Q_loc); CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(user->dm, PETSC_TRUE, Q_loc, 0.0,
                                    NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = VecZeroEntries(G_loc); CHKERRQ(ierr);

  // Place PETSc vectors in CEED vectors
  ierr = VecGetArrayReadAndMemType(Q_loc, (const PetscScalar **)&q, &q_mem_type);
  CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(G_loc, &g, &g_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(user->q_ceed, MemTypeP2C(q_mem_type), CEED_USE_POINTER, q);
  CeedVectorSetArray(user->g_ceed, MemTypeP2C(g_mem_type), CEED_USE_POINTER, g);

  // Apply CEED operator
  CeedOperatorApply(user->op_rhs, user->q_ceed, user->g_ceed,
                    CEED_REQUEST_IMMEDIATE);

  // Restore vectors
  CeedVectorTakeArray(user->q_ceed, MemTypeP2C(q_mem_type), NULL);
  CeedVectorTakeArray(user->g_ceed, MemTypeP2C(g_mem_type), NULL);
  ierr = VecRestoreArrayReadAndMemType(Q_loc, (const PetscScalar **)&q);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(G_loc, &g); CHKERRQ(ierr);

  // Local-to-Global
  ierr = VecZeroEntries(G); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, G_loc, ADD_VALUES, G); CHKERRQ(ierr);

  // Inverse of the lumped mass matrix (M is Minv)
  ierr = VecPointwiseMult(G, G, user->M); CHKERRQ(ierr);

  // Restore vectors
  ierr = DMRestoreLocalVector(user->dm, &Q_loc); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->dm, &G_loc); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Implicit time-stepper function setup
PetscErrorCode IFunction_NS(TS ts, PetscReal t, Vec Q, Vec Q_dot, Vec G,
                            void *user_data) {
  User              user = *(User *)user_data;
  const PetscScalar *q, *q_dot;
  PetscScalar       *g;
  Vec               Q_loc, Q_dot_loc, G_loc;
  PetscMemType      q_mem_type, q_dot_mem_type, g_mem_type;
  PetscErrorCode    ierr;
  PetscFunctionBeginUser;

  // Update EulerContext
  if (user->phys->has_curr_time) user->phys->euler_ctx->curr_time = t;

  // Get local vectors
  ierr = DMGetLocalVector(user->dm, &Q_loc); CHKERRQ(ierr);
  ierr = DMGetLocalVector(user->dm, &Q_dot_loc); CHKERRQ(ierr);
  ierr = DMGetLocalVector(user->dm, &G_loc); CHKERRQ(ierr);

  // Global-to-local
  ierr = VecZeroEntries(Q_loc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm, Q, INSERT_VALUES, Q_loc); CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(user->dm, PETSC_TRUE, Q_loc, 0.0,
                                    NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = VecZeroEntries(Q_dot_loc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm, Q_dot, INSERT_VALUES, Q_dot_loc);
  CHKERRQ(ierr);
  ierr = VecZeroEntries(G_loc); CHKERRQ(ierr);

  // Place PETSc vectors in CEED vectors
  ierr = VecGetArrayReadAndMemType(Q_loc, &q, &q_mem_type); CHKERRQ(ierr);
  ierr = VecGetArrayReadAndMemType(Q_dot_loc, &q_dot, &q_dot_mem_type);
  CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(G_loc, &g, &g_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(user->q_ceed, MemTypeP2C(q_mem_type), CEED_USE_POINTER,
                     (PetscScalar *)q);
  CeedVectorSetArray(user->q_dot_ceed, MemTypeP2C(q_dot_mem_type),
                     CEED_USE_POINTER,
                     (PetscScalar *)q_dot);
  CeedVectorSetArray(user->g_ceed, MemTypeP2C(g_mem_type), CEED_USE_POINTER, g);

  // Apply CEED operator
  CeedOperatorApply(user->op_ifunction, user->q_ceed, user->g_ceed,
                    CEED_REQUEST_IMMEDIATE);

  // Restore vectors
  CeedVectorTakeArray(user->q_ceed, MemTypeP2C(q_mem_type), NULL);
  CeedVectorTakeArray(user->q_dot_ceed, MemTypeP2C(q_dot_mem_type), NULL);
  CeedVectorTakeArray(user->g_ceed, MemTypeP2C(g_mem_type), NULL);
  ierr = VecRestoreArrayReadAndMemType(Q_loc, &q); CHKERRQ(ierr);
  ierr = VecRestoreArrayReadAndMemType(Q_dot_loc, &q_dot); CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(G_loc, &g); CHKERRQ(ierr);

  // Local-to-Global
  ierr = VecZeroEntries(G); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, G_loc, ADD_VALUES, G); CHKERRQ(ierr);

  // Restore vectors
  ierr = DMRestoreLocalVector(user->dm, &Q_loc); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->dm, &Q_dot_loc); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->dm, &G_loc); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// User provided TS Monitor
PetscErrorCode TSMonitor_NS(TS ts, PetscInt step_no, PetscReal time,
                            Vec Q, void *ctx) {
  User           user = ctx;
  Vec            Q_loc;
  char           file_path[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // Print every 'output_freq' steps
  if (step_no % user->app_ctx->output_freq != 0)
    PetscFunctionReturn(0);

  // Set up output
  ierr = DMGetLocalVector(user->dm, &Q_loc); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)Q_loc, "StateVec"); CHKERRQ(ierr);
  ierr = VecZeroEntries(Q_loc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm, Q, INSERT_VALUES, Q_loc); CHKERRQ(ierr);

  // Output
  ierr = PetscSNPrintf(file_path, sizeof file_path, "%s/ns-%03D.vtu",
                       user->app_ctx->output_dir, step_no + user->app_ctx->cont_steps);
  CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)Q), file_path,
                            FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
  ierr = VecView(Q_loc, viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  if (user->dm_viz) {
    Vec         Q_refined, Q_refined_loc;
    char        file_path_refined[PETSC_MAX_PATH_LEN];
    PetscViewer viewer_refined;

    ierr = DMGetGlobalVector(user->dm_viz, &Q_refined); CHKERRQ(ierr);
    ierr = DMGetLocalVector(user->dm_viz, &Q_refined_loc); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)Q_refined_loc, "Refined");
    CHKERRQ(ierr);
    ierr = MatInterpolate(user->interp_viz, Q, Q_refined); CHKERRQ(ierr);
    ierr = VecZeroEntries(Q_refined_loc); CHKERRQ(ierr);
    ierr = DMGlobalToLocal(user->dm_viz, Q_refined, INSERT_VALUES, Q_refined_loc);
    CHKERRQ(ierr);
    ierr = PetscSNPrintf(file_path_refined, sizeof file_path_refined,
                         "%s/nsrefined-%03D.vtu",
                         user->app_ctx->output_dir, step_no + user->app_ctx->cont_steps);
    CHKERRQ(ierr);
    ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)Q_refined),
                              file_path_refined,
                              FILE_MODE_WRITE, &viewer_refined); CHKERRQ(ierr);
    ierr = VecView(Q_refined_loc, viewer_refined); CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(user->dm_viz, &Q_refined_loc); CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(user->dm_viz, &Q_refined); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer_refined); CHKERRQ(ierr);
  }
  ierr = DMRestoreLocalVector(user->dm, &Q_loc); CHKERRQ(ierr);

  // Save data in a binary file for continuation of simulations
  ierr = PetscSNPrintf(file_path, sizeof file_path, "%s/ns-solution.bin",
                       user->app_ctx->output_dir); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(user->comm, file_path, FILE_MODE_WRITE, &viewer);
  CHKERRQ(ierr);
  ierr = VecView(Q, viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  // Save time stamp
  // Dimensionalize time back
  time /= user->units->second;
  ierr = PetscSNPrintf(file_path, sizeof file_path, "%s/ns-time.bin",
                       user->app_ctx->output_dir); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(user->comm, file_path, FILE_MODE_WRITE, &viewer);
  CHKERRQ(ierr);
  #if PETSC_VERSION_GE(3,13,0)
  ierr = PetscViewerBinaryWrite(viewer, &time, 1, PETSC_REAL);
  #else
  ierr = PetscViewerBinaryWrite(viewer, &time, 1, PETSC_REAL, true);
  #endif
  CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// TS: Create, setup, and solve
PetscErrorCode TSSolve_NS(DM dm, User user, AppCtx app_ctx, Physics phys,
                          Vec *Q, PetscScalar *f_time, TS *ts) {

  MPI_Comm       comm = user->comm;
  TSAdapt        adapt;
  PetscScalar    final_time;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  ierr = TSCreate(comm, ts); CHKERRQ(ierr);
  ierr = TSSetDM(*ts, dm); CHKERRQ(ierr);
  if (phys->implicit) {
    ierr = TSSetType(*ts, TSBDF); CHKERRQ(ierr);
    if (user->op_ifunction) {
      ierr = TSSetIFunction(*ts, NULL, IFunction_NS, &user); CHKERRQ(ierr);
    } else { // Implicit integrators can fall back to using an RHSFunction
      ierr = TSSetRHSFunction(*ts, NULL, RHS_NS, &user); CHKERRQ(ierr);
    }
  } else {
    if (!user->op_rhs) SETERRQ(comm, PETSC_ERR_ARG_NULL,
                                 "Problem does not provide RHSFunction");
    ierr = TSSetType(*ts, TSRK); CHKERRQ(ierr);
    ierr = TSRKSetType(*ts, TSRK5F); CHKERRQ(ierr);
    ierr = TSSetRHSFunction(*ts, NULL, RHS_NS, &user); CHKERRQ(ierr);
  }
  ierr = TSSetMaxTime(*ts, 500. * user->units->second); CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(*ts, TS_EXACTFINALTIME_STEPOVER); CHKERRQ(ierr);
  ierr = TSSetTimeStep(*ts, 1.e-2 * user->units->second); CHKERRQ(ierr);
  if (app_ctx->test_mode) {ierr = TSSetMaxSteps(*ts, 10); CHKERRQ(ierr);}
  ierr = TSGetAdapt(*ts, &adapt); CHKERRQ(ierr);
  ierr = TSAdaptSetStepLimits(adapt,
                              1.e-12 * user->units->second,
                              1.e2 * user->units->second); CHKERRQ(ierr);
  ierr = TSSetFromOptions(*ts); CHKERRQ(ierr);
  if (!app_ctx->cont_steps) { // print initial condition
    if (!app_ctx->test_mode) {
      ierr = TSMonitor_NS(*ts, 0, 0., *Q, user); CHKERRQ(ierr);
    }
  } else { // continue from time of last output
    PetscReal   time;
    PetscInt    count;
    PetscViewer viewer;
    char        file_path[PETSC_MAX_PATH_LEN];
    ierr = PetscSNPrintf(file_path, sizeof file_path, "%s/ns-time.bin",
                         app_ctx->output_dir); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm, file_path, FILE_MODE_READ, &viewer);
    CHKERRQ(ierr);
    ierr = PetscViewerBinaryRead(viewer, &time, 1, &count, PETSC_REAL);
    CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    ierr = TSSetTime(*ts, time * user->units->second); CHKERRQ(ierr);
  }
  if (!app_ctx->test_mode) {
    ierr = TSMonitorSet(*ts, TSMonitor_NS, user, NULL); CHKERRQ(ierr);
  }

  // Solve
  double start, cpu_time_used;
  start = MPI_Wtime();
  ierr = PetscBarrier((PetscObject) *ts); CHKERRQ(ierr);
  ierr = TSSolve(*ts, *Q); CHKERRQ(ierr);
  cpu_time_used = MPI_Wtime() - start;
  ierr = TSGetSolveTime(*ts, &final_time); CHKERRQ(ierr);
  *f_time = final_time;
  ierr = MPI_Allreduce(MPI_IN_PLACE, &cpu_time_used, 1, MPI_DOUBLE, MPI_MIN,
                       comm); CHKERRQ(ierr);
  if (!app_ctx->test_mode) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "Time taken for solution (sec): %g\n",
                       (double)cpu_time_used); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
