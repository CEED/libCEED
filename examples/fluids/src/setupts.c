// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

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
  CeedQFunctionAddInput(qf_mass, "qdata", q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_mass, "v", num_comp_q, CEED_EVAL_INTERP);

  // CEED Operator
  CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass);
  CeedOperatorSetField(op_mass, "q", ceed_data->elem_restr_q, ceed_data->basis_q,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "qdata", ceed_data->elem_restr_qd_i,
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
  Vec            Q_loc = user->Q_loc, G_loc;
  PetscMemType   q_mem_type, g_mem_type;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // Get local vector
  ierr = DMGetLocalVector(user->dm, &G_loc); CHKERRQ(ierr);

  // Update time dependent data
  if (user->time != t) {
    ierr = DMPlexInsertBoundaryValues(user->dm, PETSC_TRUE, Q_loc, t,
                                      NULL, NULL, NULL); CHKERRQ(ierr);
    if (user->phys->solution_time_label) {
      CeedOperatorContextSetDouble(user->op_rhs, user->phys->solution_time_label, &t);
    }
    user->time = t;
  }
  if (user->phys->timestep_size_label) {
    PetscScalar dt;
    ierr = TSGetTimeStep(ts, &dt); CHKERRQ(ierr);
    if (user->dt != dt) {
      CeedOperatorContextSetDouble(user->op_rhs, user->phys->timestep_size_label,
                                   &dt);
      user->dt = dt;
    }
  }

  // Global-to-local
  ierr = DMGlobalToLocal(user->dm, Q, INSERT_VALUES, Q_loc); CHKERRQ(ierr);

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
  ierr = DMRestoreLocalVector(user->dm, &G_loc); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Implicit time-stepper function setup
PetscErrorCode IFunction_NS(TS ts, PetscReal t, Vec Q, Vec Q_dot, Vec G,
                            void *user_data) {
  User              user = *(User *)user_data;
  const PetscScalar *q, *q_dot;
  PetscScalar       *g;
  Vec               Q_loc = user->Q_loc, Q_dot_loc = user->Q_dot_loc, G_loc;
  PetscMemType      q_mem_type, q_dot_mem_type, g_mem_type;
  PetscErrorCode    ierr;
  PetscFunctionBeginUser;

  // Get local vectors
  ierr = DMGetLocalVector(user->dm, &G_loc); CHKERRQ(ierr);

  // Update time dependent data
  if (user->time != t) {
    ierr = DMPlexInsertBoundaryValues(user->dm, PETSC_TRUE, Q_loc, t,
                                      NULL, NULL, NULL); CHKERRQ(ierr);
    if (user->phys->solution_time_label) {
      CeedOperatorContextSetDouble(user->op_ifunction,
                                   user->phys->solution_time_label, &t);
    }
    user->time = t;
  }
  if (user->phys->timestep_size_label) {
    PetscScalar dt;
    ierr = TSGetTimeStep(ts, &dt); CHKERRQ(ierr);
    if (user->dt != dt) {
      CeedOperatorContextSetDouble(user->op_ifunction,
                                   user->phys->timestep_size_label, &dt);
      user->dt = dt;
    }
  }

  // Global-to-local
  ierr = DMGlobalToLocal(user->dm, Q, INSERT_VALUES, Q_loc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm, Q_dot, INSERT_VALUES, Q_dot_loc);
  CHKERRQ(ierr);

  // Place PETSc vectors in CEED vectors
  ierr = VecGetArrayReadAndMemType(Q_loc, &q, &q_mem_type); CHKERRQ(ierr);
  ierr = VecGetArrayReadAndMemType(Q_dot_loc, &q_dot, &q_dot_mem_type);
  CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(G_loc, &g, &g_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(user->q_ceed, MemTypeP2C(q_mem_type), CEED_USE_POINTER,
                     (PetscScalar *)q);
  CeedVectorSetArray(user->q_dot_ceed, MemTypeP2C(q_dot_mem_type),
                     CEED_USE_POINTER, (PetscScalar *)q_dot);
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
  ierr = DMRestoreLocalVector(user->dm, &G_loc); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_NS_IJacobian(Mat J, Vec Q, Vec G) {
  User user;
  const PetscScalar *q;
  PetscScalar       *g;
  PetscMemType      q_mem_type, g_mem_type;
  PetscErrorCode    ierr;
  PetscFunctionBeginUser;
  ierr = MatShellGetContext(J, &user); CHKERRQ(ierr);
  Vec               Q_loc = user->Q_dot_loc, // Note - Q_dot_loc has zero BCs
                    G_loc;

  // Get local vectors
  ierr = DMGetLocalVector(user->dm, &G_loc); CHKERRQ(ierr);

  // Global-to-local
  ierr = DMGlobalToLocal(user->dm, Q, INSERT_VALUES, Q_loc); CHKERRQ(ierr);

  // Place PETSc vectors in CEED vectors
  ierr = VecGetArrayReadAndMemType(Q_loc, &q, &q_mem_type); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(G_loc, &g, &g_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(user->q_ceed, MemTypeP2C(q_mem_type), CEED_USE_POINTER,
                     (PetscScalar *)q);
  CeedVectorSetArray(user->g_ceed, MemTypeP2C(g_mem_type), CEED_USE_POINTER, g);

  // Apply CEED operator
  CeedOperatorApply(user->op_ijacobian, user->q_ceed, user->g_ceed,
                    CEED_REQUEST_IMMEDIATE);

  // Restore vectors
  CeedVectorTakeArray(user->q_ceed, MemTypeP2C(q_mem_type), NULL);
  CeedVectorTakeArray(user->g_ceed, MemTypeP2C(g_mem_type), NULL);
  ierr = VecRestoreArrayReadAndMemType(Q_loc, &q); CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(G_loc, &g); CHKERRQ(ierr);

  // Local-to-Global
  ierr = VecZeroEntries(G); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, G_loc, ADD_VALUES, G); CHKERRQ(ierr);

  // Restore vectors
  ierr = DMRestoreLocalVector(user->dm, &G_loc); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetDiagonal_NS_IJacobian(Mat A, Vec D) {
  User user;
  Vec D_loc;
  PetscScalar *d;
  PetscMemType mem_type;

  PetscFunctionBeginUser;
  MatShellGetContext(A, &user);
  PetscCall(DMGetLocalVector(user->dm, &D_loc));
  PetscCall(VecGetArrayAndMemType(D_loc, &d, &mem_type));
  CeedVectorSetArray(user->g_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, d);
  CeedOperatorLinearAssembleDiagonal(user->op_ijacobian, user->g_ceed,
                                     CEED_REQUEST_IMMEDIATE);
  CeedVectorTakeArray(user->g_ceed, MemTypeP2C(mem_type), NULL);
  PetscCall(VecRestoreArrayAndMemType(D_loc, &d));
  PetscCall(VecZeroEntries(D));
  PetscCall(DMLocalToGlobal(user->dm, D_loc, ADD_VALUES, D));
  PetscCall(DMRestoreLocalVector(user->dm, &D_loc));
  VecViewFromOptions(D, NULL, "-diag_vec_view");
  PetscFunctionReturn(0);
}

static PetscErrorCode FormPreallocation(User user, PetscBool pbdiagonal, Mat J,
                                        CeedVector *coo_values) {
  PetscCount ncoo;
  PetscInt *rows, *cols;

  PetscFunctionBeginUser;
  if (pbdiagonal) {
    CeedSize l_size;
    CeedOperatorGetActiveVectorLengths(user->op_ijacobian, &l_size, NULL);
    ncoo = l_size * 5;
    rows = malloc(ncoo*sizeof(rows[0]));
    cols = malloc(ncoo*sizeof(cols[0]));
    for (PetscCount n=0; n<l_size/5; n++) {
      for (PetscInt i=0; i<5; i++) {
        for (PetscInt j=0; j<5; j++) {
          rows[(n*5+i)*5+j] = n * 5 + i;
          cols[(n*5+i)*5+j] = n * 5 + j;
        }
      }
    }
  } else {
    PetscCall(CeedOperatorLinearAssembleSymbolic(user->op_ijacobian, &ncoo, &rows,
              &cols));
  }
  PetscCall(MatSetPreallocationCOOLocal(J, ncoo, rows, cols));
  free(rows);
  free(cols);
  CeedVectorCreate(user->ceed, ncoo, coo_values);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormSetValues(User user, PetscBool pbdiagonal, Mat J,
                                    CeedVector coo_values) {
  CeedMemType mem_type = CEED_MEM_HOST;
  const PetscScalar *values;
  MatType mat_type;

  PetscFunctionBeginUser;
  PetscCall(MatGetType(J, &mat_type));
  if (strstr(mat_type, "kokkos") || strstr(mat_type, "cusparse"))
    mem_type = CEED_MEM_DEVICE;
  if (user->app_ctx->pmat_pbdiagonal) {
    CeedOperatorLinearAssemblePointBlockDiagonal(user->op_ijacobian,
        coo_values, CEED_REQUEST_IMMEDIATE);
  } else {
    CeedOperatorLinearAssemble(user->op_ijacobian, coo_values);
  }
  CeedVectorGetArrayRead(coo_values, mem_type, &values);
  PetscCall(MatSetValuesCOO(J, values, INSERT_VALUES));
  CeedVectorRestoreArrayRead(coo_values, &values);
  PetscFunctionReturn(0);
}

PetscErrorCode FormIJacobian_NS(TS ts, PetscReal t, Vec Q, Vec Q_dot,
                                PetscReal shift, Mat J, Mat J_pre,
                                void *user_data) {
  User user = *(User *)user_data;
  PetscBool J_is_shell, J_is_mffd, J_pre_is_shell;
  PetscFunctionBeginUser;
  if (user->phys->ijacobian_time_shift_label)
    CeedOperatorContextSetDouble(user->op_ijacobian,
                                 user->phys->ijacobian_time_shift_label, &shift);
  PetscCall(PetscObjectTypeCompare((PetscObject)J, MATMFFD, &J_is_mffd));
  PetscCall(PetscObjectTypeCompare((PetscObject)J, MATSHELL, &J_is_shell));
  PetscCall(PetscObjectTypeCompare((PetscObject)J_pre, MATSHELL,
                                   &J_pre_is_shell));
  if (!user->matrices_set_up) {
    if (J_is_shell) {
      PetscCall(MatShellSetContext(J, user));
      PetscCall(MatShellSetOperation(J, MATOP_MULT,
                                     (void (*)(void))MatMult_NS_IJacobian));
      PetscCall(MatShellSetOperation(J, MATOP_GET_DIAGONAL,
                                     (void (*)(void))MatGetDiagonal_NS_IJacobian));
      PetscCall(MatSetUp(J));
    }
    if (!J_pre_is_shell) {
      PetscCall(FormPreallocation(user,user->app_ctx->pmat_pbdiagonal,J_pre,
                                  &user->coo_values_pmat));
    }
    if (J != J_pre && !J_is_shell && !J_is_mffd) {
      PetscCall(FormPreallocation(user,PETSC_FALSE,J, &user->coo_values_amat));
    }
    user->matrices_set_up = true;
  }
  if (!J_pre_is_shell) {
    PetscCall(FormSetValues(user, user->app_ctx->pmat_pbdiagonal, J_pre,
                            user->coo_values_pmat));
  }
  if (user->coo_values_amat) {
    PetscCall(FormSetValues(user, PETSC_FALSE, J, user->coo_values_amat));
  } else if (J_is_mffd) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode WriteOutput(User user, Vec Q, PetscInt step_no,
                           PetscScalar time)  {
  Vec            Q_loc;
  char           file_path[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  PetscFunctionBeginUser;

  // Set up output
  PetscCall(DMGetLocalVector(user->dm, &Q_loc));
  PetscCall(PetscObjectSetName((PetscObject)Q_loc, "StateVec"));
  PetscCall(VecZeroEntries(Q_loc));
  PetscCall(DMGlobalToLocal(user->dm, Q, INSERT_VALUES, Q_loc));

  // Output
  PetscCall(PetscSNPrintf(file_path, sizeof file_path,
                          "%s/ns-%03" PetscInt_FMT ".vtu",
                          user->app_ctx->output_dir, step_no + user->app_ctx->cont_steps));

  PetscCall(PetscViewerVTKOpen(PetscObjectComm((PetscObject)Q), file_path,
                               FILE_MODE_WRITE, &viewer));
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
    PetscCall(DMGlobalToLocal(user->dm_viz, Q_refined, INSERT_VALUES,
                              Q_refined_loc));

    PetscCall(PetscSNPrintf(file_path_refined, sizeof file_path_refined,
                            "%s/nsrefined-%03" PetscInt_FMT ".vtu", user->app_ctx->output_dir,
                            step_no + user->app_ctx->cont_steps));

    PetscCall(PetscViewerVTKOpen(PetscObjectComm((PetscObject)Q_refined),
                                 file_path_refined, FILE_MODE_WRITE, &viewer_refined));
    PetscCall(VecView(Q_refined_loc, viewer_refined));
    PetscCall(DMRestoreLocalVector(user->dm_viz, &Q_refined_loc));
    PetscCall(DMRestoreGlobalVector(user->dm_viz, &Q_refined));
    PetscCall(PetscViewerDestroy(&viewer_refined));
  }
  PetscCall(DMRestoreLocalVector(user->dm, &Q_loc));

  // Save data in a binary file for continuation of simulations
  PetscCall(PetscSNPrintf(file_path, sizeof file_path, "%s/ns-solution.bin",
                          user->app_ctx->output_dir));
  PetscCall(PetscViewerBinaryOpen(user->comm, file_path, FILE_MODE_WRITE,
                                  &viewer));

  PetscCall(VecView(Q, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  // Save time stamp
  // Dimensionalize time back
  time /= user->units->second;
  PetscCall(PetscSNPrintf(file_path, sizeof file_path, "%s/ns-time.bin",
                          user->app_ctx->output_dir));
  PetscCall(PetscViewerBinaryOpen(user->comm, file_path, FILE_MODE_WRITE,
                                  &viewer));

  #if PETSC_VERSION_GE(3,13,0)
  PetscCall(PetscViewerBinaryWrite(viewer, &time, 1, PETSC_REAL));
  #else
  PetscCall(PetscViewerBinaryWrite(viewer, &time, 1, PETSC_REAL, true));
  #endif
  PetscCall(PetscViewerDestroy(&viewer));

  PetscFunctionReturn(0);
}

// User provided TS Monitor
PetscErrorCode TSMonitor_NS(TS ts, PetscInt step_no, PetscReal time,
                            Vec Q, void *ctx) {
  User           user = ctx;
  PetscFunctionBeginUser;

  // Print every 'output_freq' steps
  if (user->app_ctx->output_freq <= 0
      || step_no % user->app_ctx->output_freq != 0)
    PetscFunctionReturn(0);

  PetscCall(WriteOutput(user, Q, step_no, time));

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
    if (user->op_ijacobian) {
      ierr = DMTSSetIJacobian(dm, FormIJacobian_NS, &user); CHKERRQ(ierr);
      if (app_ctx->amat_type) {
        Mat Pmat,Amat;
        ierr = DMCreateMatrix(dm, &Pmat); CHKERRQ(ierr);
        ierr = DMSetMatType(dm, app_ctx->amat_type); CHKERRQ(ierr);
        ierr = DMCreateMatrix(dm, &Amat); CHKERRQ(ierr);
        ierr = TSSetIJacobian(*ts, Amat, Pmat, NULL, NULL); CHKERRQ(ierr);
        ierr = MatDestroy(&Amat); CHKERRQ(ierr);
        ierr = MatDestroy(&Pmat); CHKERRQ(ierr);
      }
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
  user->time = -1.0; // require all BCs and ctx to be updated
  user->dt   = -1.0;
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
  PetscScalar start_time;
  ierr = TSGetTime(*ts, &start_time); CHKERRQ(ierr);

  PetscPreLoadBegin(PETSC_FALSE, "Fluids Solve");
  PetscCall(TSSetTime(*ts, start_time));
  PetscCall(TSSetStepNumber(*ts, 0));
  if (PetscPreLoadingOn) {
    // LCOV_EXCL_START
    SNES      snes;
    Vec       Q_preload;
    PetscReal rtol;
    PetscCall(VecDuplicate(*Q, &Q_preload));
    PetscCall(VecCopy(*Q, Q_preload));
    PetscCall(TSGetSNES(*ts, &snes));
    PetscCall(SNESGetTolerances(snes, NULL, &rtol, NULL, NULL, NULL));
    PetscCall(SNESSetTolerances(snes, PETSC_DEFAULT, .99, PETSC_DEFAULT,
                                PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(TSSetSolution(*ts, *Q));
    PetscCall(TSStep(*ts));
    PetscCall(SNESSetTolerances(snes, PETSC_DEFAULT, rtol, PETSC_DEFAULT,
                                PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(VecDestroy(&Q_preload));
    // LCOV_EXCL_STOP
  } else {
    ierr = PetscBarrier((PetscObject) *ts); CHKERRQ(ierr);
    ierr = TSSolve(*ts, *Q); CHKERRQ(ierr);
  }
  PetscPreLoadEnd();

  PetscCall(TSGetSolveTime(*ts, &final_time));
  *f_time = final_time;


  if (!app_ctx->test_mode) {
    if (user->app_ctx->output_freq > 0 || user->app_ctx->output_freq == -1) {
      PetscInt step_no;
      PetscCall(TSGetStepNumber(*ts, &step_no));
      PetscCall(WriteOutput(user, *Q, step_no, final_time));
    }

    PetscLogEvent stage_id;
    PetscStageLog stage_log;

    PetscCall(PetscLogStageGetId("Fluids Solve", &stage_id));
    PetscCall(PetscLogGetStageLog(&stage_log));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "Time taken for solution (sec): %g\n",
                          stage_log->stageInfo[stage_id].perfInfo.time));
  }
  PetscFunctionReturn(0);
}
