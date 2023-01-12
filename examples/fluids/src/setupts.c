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
#include "../qfunctions/newtonian_state.h"

// Compute mass matrix for explicit scheme
PetscErrorCode ComputeLumpedMassMatrix(Ceed ceed, DM dm, CeedData ceed_data, Vec M) {
  Vec           M_loc;
  CeedQFunction qf_mass;
  CeedOperator  op_mass;
  CeedVector    m_ceed, ones_vec;
  CeedInt       num_comp_q, q_data_size;
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
  CeedOperatorSetField(op_mass, "q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_COLLOCATED, ceed_data->q_data);
  CeedOperatorSetField(op_mass, "v", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE);

  // Place PETSc vector in CEED vector
  CeedScalar  *m;
  PetscMemType m_mem_type;
  PetscCall(DMGetLocalVector(dm, &M_loc));
  PetscCall(VecGetArrayAndMemType(M_loc, (PetscScalar **)&m, &m_mem_type));
  CeedVectorSetArray(m_ceed, MemTypeP2C(m_mem_type), CEED_USE_POINTER, m);

  // Apply CEED Operator
  CeedOperatorApply(op_mass, ones_vec, m_ceed, CEED_REQUEST_IMMEDIATE);

  // Restore vectors
  CeedVectorTakeArray(m_ceed, MemTypeP2C(m_mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(M_loc, (const PetscScalar **)&m));

  // Local-to-Global
  PetscCall(VecZeroEntries(M));
  PetscCall(DMLocalToGlobal(dm, M_loc, ADD_VALUES, M));
  PetscCall(DMRestoreLocalVector(dm, &M_loc));

  // Invert diagonally lumped mass vector for RHS function
  PetscCall(VecReciprocal(M));

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
  User         user = *(User *)user_data;
  PetscScalar *q, *g;
  Vec          Q_loc = user->Q_loc, G_loc;
  PetscMemType q_mem_type, g_mem_type;
  PetscFunctionBeginUser;

  // Get local vector
  PetscCall(DMGetLocalVector(user->dm, &G_loc));

  // Update time dependent data
  if (user->time != t) {
    PetscCall(DMPlexInsertBoundaryValues(user->dm, PETSC_TRUE, Q_loc, t, NULL, NULL, NULL));
    if (user->phys->solution_time_label) {
      CeedOperatorContextSetDouble(user->op_rhs, user->phys->solution_time_label, &t);
    }
    user->time = t;
  }
  if (user->phys->timestep_size_label) {
    PetscScalar dt;
    PetscCall(TSGetTimeStep(ts, &dt));
    if (user->dt != dt) {
      CeedOperatorContextSetDouble(user->op_rhs, user->phys->timestep_size_label, &dt);
      user->dt = dt;
    }
  }

  // Global-to-local
  PetscCall(DMGlobalToLocal(user->dm, Q, INSERT_VALUES, Q_loc));

  // Place PETSc vectors in CEED vectors
  PetscCall(VecGetArrayReadAndMemType(Q_loc, (const PetscScalar **)&q, &q_mem_type));
  PetscCall(VecGetArrayAndMemType(G_loc, &g, &g_mem_type));
  CeedVectorSetArray(user->q_ceed, MemTypeP2C(q_mem_type), CEED_USE_POINTER, q);
  CeedVectorSetArray(user->g_ceed, MemTypeP2C(g_mem_type), CEED_USE_POINTER, g);

  // Apply CEED operator
  CeedOperatorApply(user->op_rhs, user->q_ceed, user->g_ceed, CEED_REQUEST_IMMEDIATE);

  // Restore vectors
  CeedVectorTakeArray(user->q_ceed, MemTypeP2C(q_mem_type), NULL);
  CeedVectorTakeArray(user->g_ceed, MemTypeP2C(g_mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(Q_loc, (const PetscScalar **)&q));
  PetscCall(VecRestoreArrayAndMemType(G_loc, &g));

  // Local-to-Global
  PetscCall(VecZeroEntries(G));
  PetscCall(DMLocalToGlobal(user->dm, G_loc, ADD_VALUES, G));

  // Inverse of the lumped mass matrix (M is Minv)
  PetscCall(VecPointwiseMult(G, G, user->M));

  // Restore vectors
  PetscCall(DMRestoreLocalVector(user->dm, &G_loc));

  PetscFunctionReturn(0);
}

// Surface forces function setup
static PetscErrorCode Surface_Forces_NS(DM dm, Vec G_loc) {
  const PetscInt     num_walls = 1, walls[] = {5};  // "cylinderwalls" in meshes/cylinder-*.msh
  DMLabel            face_label;
  const PetscScalar *g;
  MPI_Comm           comm;

  PetscFunctionBeginUser;
  PetscObjectGetComm((PetscObject)dm, &comm);
  PetscCall(DMGetLabel(dm, "Face Sets", &face_label));
  PetscCall(VecGetArrayRead(G_loc, &g));
  for (PetscInt w = 0; w < num_walls; w++) {
    PetscScalar    reaction_force[3] = {0};
    const PetscInt wall              = walls[w];
    IS             wall_is;
    PetscCall(DMLabelGetStratumIS(face_label, wall, &wall_is));
    if (wall_is) {  // There exist such points on this process
      PetscInt        num_points;
      const PetscInt *points;
      PetscCall(ISGetSize(wall_is, &num_points));
      PetscCall(ISGetIndices(wall_is, &points));
      for (PetscInt i = 0; i < num_points; i++) {
        const PetscInt           p = points[i];
        const StateConservative *r;
        PetscCall(DMPlexPointLocalRead(dm, p, g, &r));
        if (!r) continue;
        for (PetscInt j = 0; j < 3; j++) {
          reaction_force[j] -= r->momentum[j];
        }
      }
      PetscCall(ISRestoreIndices(wall_is, &points));
    }
    PetscCall(ISDestroy(&wall_is));
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, reaction_force, 3, MPIU_SCALAR, MPI_SUM, comm));
    PetscCall(PetscPrintf(comm, "Wall %" PetscInt_FMT " force: %12g %12g %12g\n", wall, reaction_force[0], reaction_force[1], reaction_force[2]));
  }
  //  Restore Vectors
  PetscCall(VecRestoreArrayRead(G_loc, &g));

  PetscFunctionReturn(0);
}

// Implicit time-stepper function setup
PetscErrorCode IFunction_NS(TS ts, PetscReal t, Vec Q, Vec Q_dot, Vec G, void *user_data) {
  User               user = *(User *)user_data;
  const PetscScalar *q, *q_dot;
  PetscScalar       *g;
  Vec                Q_loc = user->Q_loc, Q_dot_loc = user->Q_dot_loc, G_loc;
  PetscMemType       q_mem_type, q_dot_mem_type, g_mem_type;
  PetscFunctionBeginUser;

  // Get local vectors
  PetscCall(DMGetLocalVector(user->dm, &G_loc));

  // Update time dependent data
  if (user->time != t) {
    PetscCall(DMPlexInsertBoundaryValues(user->dm, PETSC_TRUE, Q_loc, t, NULL, NULL, NULL));
    if (user->phys->solution_time_label) {
      CeedOperatorContextSetDouble(user->op_ifunction, user->phys->solution_time_label, &t);
    }
    user->time = t;
  }
  if (user->phys->timestep_size_label) {
    PetscScalar dt;
    PetscCall(TSGetTimeStep(ts, &dt));
    if (user->dt != dt) {
      CeedOperatorContextSetDouble(user->op_ifunction, user->phys->timestep_size_label, &dt);
      user->dt = dt;
    }
  }

  // Global-to-local
  PetscCall(DMGlobalToLocal(user->dm, Q, INSERT_VALUES, Q_loc));
  PetscCall(DMGlobalToLocal(user->dm, Q_dot, INSERT_VALUES, Q_dot_loc));

  // Place PETSc vectors in CEED vectors
  PetscCall(VecGetArrayReadAndMemType(Q_loc, &q, &q_mem_type));
  PetscCall(VecGetArrayReadAndMemType(Q_dot_loc, &q_dot, &q_dot_mem_type));
  PetscCall(VecGetArrayAndMemType(G_loc, &g, &g_mem_type));
  CeedVectorSetArray(user->q_ceed, MemTypeP2C(q_mem_type), CEED_USE_POINTER, (PetscScalar *)q);
  CeedVectorSetArray(user->q_dot_ceed, MemTypeP2C(q_dot_mem_type), CEED_USE_POINTER, (PetscScalar *)q_dot);
  CeedVectorSetArray(user->g_ceed, MemTypeP2C(g_mem_type), CEED_USE_POINTER, g);

  // Apply CEED operator
  CeedOperatorApply(user->op_ifunction, user->q_ceed, user->g_ceed, CEED_REQUEST_IMMEDIATE);

  // Restore vectors
  CeedVectorTakeArray(user->q_ceed, MemTypeP2C(q_mem_type), NULL);
  CeedVectorTakeArray(user->q_dot_ceed, MemTypeP2C(q_dot_mem_type), NULL);
  CeedVectorTakeArray(user->g_ceed, MemTypeP2C(g_mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(Q_loc, &q));
  PetscCall(VecRestoreArrayReadAndMemType(Q_dot_loc, &q_dot));
  PetscCall(VecRestoreArrayAndMemType(G_loc, &g));

  PetscCall(Surface_Forces_NS(user->dm, G_loc));

  // Local-to-Global
  PetscCall(VecZeroEntries(G));
  PetscCall(DMLocalToGlobal(user->dm, G_loc, ADD_VALUES, G));

  // Restore vectors
  PetscCall(DMRestoreLocalVector(user->dm, &G_loc));

  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_NS_IJacobian(Mat J, Vec Q, Vec G) {
  User               user;
  const PetscScalar *q;
  PetscScalar       *g;
  PetscMemType       q_mem_type, g_mem_type;
  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(J, &user));
  Vec Q_loc = user->Q_dot_loc,  // Note - Q_dot_loc has zero BCs
      G_loc;

  // Get local vectors
  PetscCall(DMGetLocalVector(user->dm, &G_loc));

  // Global-to-local
  PetscCall(DMGlobalToLocal(user->dm, Q, INSERT_VALUES, Q_loc));

  // Place PETSc vectors in CEED vectors
  PetscCall(VecGetArrayReadAndMemType(Q_loc, &q, &q_mem_type));
  PetscCall(VecGetArrayAndMemType(G_loc, &g, &g_mem_type));
  CeedVectorSetArray(user->q_ceed, MemTypeP2C(q_mem_type), CEED_USE_POINTER, (PetscScalar *)q);
  CeedVectorSetArray(user->g_ceed, MemTypeP2C(g_mem_type), CEED_USE_POINTER, g);

  // Apply CEED operator
  CeedOperatorApply(user->op_ijacobian, user->q_ceed, user->g_ceed, CEED_REQUEST_IMMEDIATE);

  // Restore vectors
  CeedVectorTakeArray(user->q_ceed, MemTypeP2C(q_mem_type), NULL);
  CeedVectorTakeArray(user->g_ceed, MemTypeP2C(g_mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(Q_loc, &q));
  PetscCall(VecRestoreArrayAndMemType(G_loc, &g));

  // Local-to-Global
  PetscCall(VecZeroEntries(G));
  PetscCall(DMLocalToGlobal(user->dm, G_loc, ADD_VALUES, G));

  // Restore vectors
  PetscCall(DMRestoreLocalVector(user->dm, &G_loc));
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetDiagonal_NS_IJacobian(Mat A, Vec D) {
  User         user;
  Vec          D_loc;
  PetscScalar *d;
  PetscMemType mem_type;

  PetscFunctionBeginUser;
  MatShellGetContext(A, &user);
  PetscCall(DMGetLocalVector(user->dm, &D_loc));
  PetscCall(VecGetArrayAndMemType(D_loc, &d, &mem_type));
  CeedVectorSetArray(user->g_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, d);
  CeedOperatorLinearAssembleDiagonal(user->op_ijacobian, user->g_ceed, CEED_REQUEST_IMMEDIATE);
  CeedVectorTakeArray(user->g_ceed, MemTypeP2C(mem_type), NULL);
  PetscCall(VecRestoreArrayAndMemType(D_loc, &d));
  PetscCall(VecZeroEntries(D));
  PetscCall(DMLocalToGlobal(user->dm, D_loc, ADD_VALUES, D));
  PetscCall(DMRestoreLocalVector(user->dm, &D_loc));
  VecViewFromOptions(D, NULL, "-diag_vec_view");
  PetscFunctionReturn(0);
}

static PetscErrorCode FormPreallocation(User user, PetscBool pbdiagonal, Mat J, CeedVector *coo_values) {
  PetscCount ncoo;
  PetscInt  *rows, *cols;

  PetscFunctionBeginUser;
  if (pbdiagonal) {
    CeedSize l_size;
    CeedOperatorGetActiveVectorLengths(user->op_ijacobian, &l_size, NULL);
    ncoo = l_size * 5;
    rows = malloc(ncoo * sizeof(rows[0]));
    cols = malloc(ncoo * sizeof(cols[0]));
    for (PetscCount n = 0; n < l_size / 5; n++) {
      for (PetscInt i = 0; i < 5; i++) {
        for (PetscInt j = 0; j < 5; j++) {
          rows[(n * 5 + i) * 5 + j] = n * 5 + i;
          cols[(n * 5 + i) * 5 + j] = n * 5 + j;
        }
      }
    }
  } else {
    PetscCall(CeedOperatorLinearAssembleSymbolic(user->op_ijacobian, &ncoo, &rows, &cols));
  }
  PetscCall(MatSetPreallocationCOOLocal(J, ncoo, rows, cols));
  free(rows);
  free(cols);
  CeedVectorCreate(user->ceed, ncoo, coo_values);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormSetValues(User user, PetscBool pbdiagonal, Mat J, CeedVector coo_values) {
  CeedMemType        mem_type = CEED_MEM_HOST;
  const PetscScalar *values;
  MatType            mat_type;

  PetscFunctionBeginUser;
  PetscCall(MatGetType(J, &mat_type));
  if (strstr(mat_type, "kokkos") || strstr(mat_type, "cusparse")) mem_type = CEED_MEM_DEVICE;
  if (user->app_ctx->pmat_pbdiagonal) {
    CeedOperatorLinearAssemblePointBlockDiagonal(user->op_ijacobian, coo_values, CEED_REQUEST_IMMEDIATE);
  } else {
    CeedOperatorLinearAssemble(user->op_ijacobian, coo_values);
  }
  CeedVectorGetArrayRead(coo_values, mem_type, &values);
  PetscCall(MatSetValuesCOO(J, values, INSERT_VALUES));
  CeedVectorRestoreArrayRead(coo_values, &values);
  PetscFunctionReturn(0);
}

PetscErrorCode FormIJacobian_NS(TS ts, PetscReal t, Vec Q, Vec Q_dot, PetscReal shift, Mat J, Mat J_pre, void *user_data) {
  User      user = *(User *)user_data;
  PetscBool J_is_shell, J_is_mffd, J_pre_is_shell;
  PetscFunctionBeginUser;
  if (user->phys->ijacobian_time_shift_label) CeedOperatorContextSetDouble(user->op_ijacobian, user->phys->ijacobian_time_shift_label, &shift);
  PetscCall(PetscObjectTypeCompare((PetscObject)J, MATMFFD, &J_is_mffd));
  PetscCall(PetscObjectTypeCompare((PetscObject)J, MATSHELL, &J_is_shell));
  PetscCall(PetscObjectTypeCompare((PetscObject)J_pre, MATSHELL, &J_pre_is_shell));
  if (!user->matrices_set_up) {
    if (J_is_shell) {
      PetscCall(MatShellSetContext(J, user));
      PetscCall(MatShellSetOperation(J, MATOP_MULT, (void (*)(void))MatMult_NS_IJacobian));
      PetscCall(MatShellSetOperation(J, MATOP_GET_DIAGONAL, (void (*)(void))MatGetDiagonal_NS_IJacobian));
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
  PetscFunctionReturn(0);
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

  PetscInt token = FLUIDS_FILE_TOKEN;
  PetscCall(PetscViewerBinaryWrite(viewer, &token, 1, PETSC_INT));
  PetscCall(PetscViewerBinaryWrite(viewer, &step_no, 1, PETSC_INT));
  time /= user->units->second;  // Dimensionalize time back
  PetscCall(PetscViewerBinaryWrite(viewer, &time, 1, PETSC_REAL));
  PetscCall(VecView(Q, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(0);
}

// User provided TS Monitor
PetscErrorCode TSMonitor_NS(TS ts, PetscInt step_no, PetscReal time, Vec Q, void *ctx) {
  User user = ctx;
  PetscFunctionBeginUser;

  // Print every 'checkpoint_interval' steps
  if (user->app_ctx->checkpoint_interval <= 0 || step_no % user->app_ctx->checkpoint_interval != 0 ||
      (user->app_ctx->cont_steps == step_no && step_no != 0))
    PetscFunctionReturn(0);

  PetscCall(WriteOutput(user, Q, step_no, time));

  PetscFunctionReturn(0);
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
    if (!user->op_rhs) SETERRQ(comm, PETSC_ERR_ARG_NULL, "Problem does not provide RHSFunction");
    PetscCall(TSSetType(*ts, TSRK));
    PetscCall(TSRKSetType(*ts, TSRK5F));
    PetscCall(TSSetRHSFunction(*ts, NULL, RHS_NS, &user));
  }
  PetscCall(TSSetMaxTime(*ts, 500. * user->units->second));
  PetscCall(TSSetExactFinalTime(*ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetErrorIfStepFails(*ts, PETSC_FALSE));
  PetscCall(TSSetTimeStep(*ts, 1.e-2 * user->units->second));
  if (app_ctx->test_mode) {
    PetscCall(TSSetMaxSteps(*ts, 10));
  }
  PetscCall(TSGetAdapt(*ts, &adapt));
  PetscCall(TSAdaptSetStepLimits(adapt, 1.e-12 * user->units->second, 1.e2 * user->units->second));
  PetscCall(TSSetFromOptions(*ts));
  user->time = -1.0;  // require all BCs and ctx to be updated
  user->dt   = -1.0;
  if (!app_ctx->cont_steps) {  // print initial condition
    if (!app_ctx->test_mode) {
      PetscCall(TSMonitor_NS(*ts, 0, 0., *Q, user));
    }
  } else {  // continue from time of last output
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
  if (!app_ctx->test_mode) {
    PetscCall(TSMonitorSet(*ts, TSMonitor_NS, user, NULL));
  }

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

  if (!app_ctx->test_mode) {
    if (user->app_ctx->checkpoint_interval > 0 || user->app_ctx->checkpoint_interval == -1) {
      PetscInt step_no;
      PetscCall(TSGetStepNumber(*ts, &step_no));
      PetscCall(WriteOutput(user, *Q, step_no, final_time));
    }

    PetscLogEvent stage_id;
    PetscStageLog stage_log;

    PetscCall(PetscLogStageGetId("Fluids Solve", &stage_id));
    PetscCall(PetscLogGetStageLog(&stage_log));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Time taken for solution (sec): %g\n", stage_log->stageInfo[stage_id].perfInfo.time));
  }
  PetscFunctionReturn(0);
}
