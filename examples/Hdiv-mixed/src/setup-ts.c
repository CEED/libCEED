#include "../include/setup-ts.h"
#include "../include/setup-matops.h"
#include "../include/setup-libceed.h"
#include "../include/setup-solvers.h"
#include "ceed/ceed.h"
#include "petscerror.h"
#include "petscsystypes.h"
#include <stdio.h>


// -----------------------------------------------------------------------------
// Setup operator context data for initial condition, u field
// -----------------------------------------------------------------------------
PetscErrorCode SetupResidualOperatorCtx_U0(MPI_Comm comm, DM dm_u0, Ceed ceed,
    CeedData ceed_data,
    OperatorApplyContext ctx_initial_u0) {
  PetscFunctionBeginUser;

  ctx_initial_u0->comm = comm;
  ctx_initial_u0->dm = dm_u0;
  PetscCall( DMCreateLocalVector(dm_u0, &ctx_initial_u0->X_loc) );
  PetscCall( VecDuplicate(ctx_initial_u0->X_loc, &ctx_initial_u0->Y_loc) );
  ctx_initial_u0->x_ceed = ceed_data->u0_ceed;
  ctx_initial_u0->y_ceed = ceed_data->v0_ceed;
  ctx_initial_u0->ceed = ceed;
  ctx_initial_u0->op_apply = ceed_data->op_ics_u;

  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// Setup operator context data for initial condition, p field
// -----------------------------------------------------------------------------
PetscErrorCode SetupResidualOperatorCtx_P0(MPI_Comm comm, DM dm_p0, Ceed ceed,
    CeedData ceed_data,
    OperatorApplyContext ctx_initial_p0) {
  PetscFunctionBeginUser;

  ctx_initial_p0->comm = comm;
  ctx_initial_p0->dm = dm_p0;
  PetscCall( DMCreateLocalVector(dm_p0, &ctx_initial_p0->X_loc) );
  PetscCall( VecDuplicate(ctx_initial_p0->X_loc, &ctx_initial_p0->Y_loc) );
  ctx_initial_p0->x_ceed = ceed_data->p0_ceed;
  ctx_initial_p0->y_ceed = ceed_data->q0_ceed;
  ctx_initial_p0->ceed = ceed;
  ctx_initial_p0->op_apply = ceed_data->op_ics_p;

  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// Setup operator context data for Residual of Richard problem
// -----------------------------------------------------------------------------
PetscErrorCode SetupResidualOperatorCtx_Ut(MPI_Comm comm, DM dm, Ceed ceed,
    CeedData ceed_data, OperatorApplyContext ctx_residual_ut) {
  PetscFunctionBeginUser;

  ctx_residual_ut->comm = comm;
  ctx_residual_ut->dm = dm;
  PetscCall( DMCreateLocalVector(dm, &ctx_residual_ut->X_loc) );
  PetscCall( VecDuplicate(ctx_residual_ut->X_loc, &ctx_residual_ut->Y_loc) );
  PetscCall( VecDuplicate(ctx_residual_ut->X_loc, &ctx_residual_ut->X_t_loc) );
  ctx_residual_ut->x_ceed = ceed_data->x_ceed;
  ctx_residual_ut->x_t_ceed = ceed_data->x_t_ceed;
  ctx_residual_ut->y_ceed = ceed_data->y_ceed;
  ctx_residual_ut->ceed = ceed;
  ctx_residual_ut->op_apply = ceed_data->op_residual;

  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// Create global initial conditions vector
// -----------------------------------------------------------------------------
PetscErrorCode CreateInitialConditions(CeedData ceed_data,
                                       Vec U, VecType vec_type,
                                       OperatorApplyContext ctx_initial_u0,
                                       OperatorApplyContext ctx_initial_p0,
                                       OperatorApplyContext ctx_residual_ut) {

  PetscFunctionBeginUser;
  // ----------------------------------------------
  // Create local rhs for u field
  // ----------------------------------------------
  Vec rhs_u_loc;
  PetscScalar *ru;
  PetscMemType ru_mem_type;
  PetscCall( DMCreateLocalVector(ctx_initial_u0->dm, &rhs_u_loc) );
  PetscCall( VecZeroEntries(rhs_u_loc) );
  PetscCall( VecGetArrayAndMemType(rhs_u_loc, &ru, &ru_mem_type) );
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_u0,
                                  &ceed_data->rhs_u0_ceed,
                                  NULL);
  CeedVectorSetArray(ceed_data->rhs_u0_ceed, MemTypeP2C(ru_mem_type),
                     CEED_USE_POINTER, ru);

  // Apply operator to create RHS for u field
  CeedOperatorApply(ceed_data->op_rhs_u0, ceed_data->x_coord,
                    ceed_data->rhs_u0_ceed, CEED_REQUEST_IMMEDIATE);

  // ----------------------------------------------
  // Create global rhs for u field
  // ----------------------------------------------
  Vec rhs_u0;
  CeedVectorTakeArray(ceed_data->rhs_u0_ceed, MemTypeP2C(ru_mem_type), NULL);
  PetscCall( VecRestoreArrayAndMemType(rhs_u_loc, &ru) );
  PetscCall( DMCreateGlobalVector(ctx_initial_u0->dm, &rhs_u0) );
  PetscCall( VecZeroEntries(rhs_u0) );
  PetscCall( DMLocalToGlobal(ctx_initial_u0->dm, rhs_u_loc, ADD_VALUES, rhs_u0) );

  // ----------------------------------------------
  // Solve for U0, M*U0 = rhs_u0
  // ----------------------------------------------
  Vec U0;
  PetscCall( DMCreateGlobalVector(ctx_initial_u0->dm, &U0) );
  PetscCall( VecZeroEntries(U0) );
  PetscInt U0_g_size, U0_l_size;
  PetscCall( VecGetSize(U0, &U0_g_size) );
  // Local size for matShell
  PetscCall( VecGetLocalSize(U0, &U0_l_size) );

  // Operator
  Mat mat_ksp_u0;
  // -- Form Action of residual on u
  PetscCall( MatCreateShell(ctx_initial_u0->comm, U0_l_size, U0_l_size, U0_g_size,
                            U0_g_size, ceed_data->ctx_initial_u0, &mat_ksp_u0) );
  PetscCall( MatShellSetOperation(mat_ksp_u0, MATOP_MULT,
                                  (void (*)(void))ApplyMatOp) );
  PetscCall( MatShellSetVecType(mat_ksp_u0, vec_type) );

  KSP ksp_u0;
  PetscCall( KSPCreate(ctx_initial_u0->comm, &ksp_u0) );
  PetscCall( KSPSetOperators(ksp_u0, mat_ksp_u0, mat_ksp_u0) );
  PetscCall( KSPSetFromOptions(ksp_u0) );
  PetscCall( KSPSetUp(ksp_u0) );
  PetscCall( KSPSolve(ksp_u0, rhs_u0, U0) );

  // ----------------------------------------------
  // Create local rhs for p field
  // ----------------------------------------------
  Vec rhs_p_loc;
  PetscScalar *rp;
  PetscMemType rp_mem_type;
  PetscCall( DMCreateLocalVector(ctx_initial_p0->dm, &rhs_p_loc) );
  PetscCall( VecZeroEntries(rhs_p_loc) );
  PetscCall( VecGetArrayAndMemType(rhs_p_loc, &rp, &rp_mem_type) );
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_p0,
                                  &ceed_data->rhs_p0_ceed,
                                  NULL);
  CeedVectorSetArray(ceed_data->rhs_p0_ceed, MemTypeP2C(rp_mem_type),
                     CEED_USE_POINTER, rp);

  // Apply operator to create RHS for p field
  CeedOperatorApply(ceed_data->op_rhs_p0, ceed_data->x_coord,
                    ceed_data->rhs_p0_ceed, CEED_REQUEST_IMMEDIATE);

  // ----------------------------------------------
  // Create global rhs for p field
  // ----------------------------------------------
  Vec rhs_p0;
  CeedVectorTakeArray(ceed_data->rhs_p0_ceed, MemTypeP2C(rp_mem_type), NULL);
  PetscCall( VecRestoreArrayAndMemType(rhs_p_loc, &rp) );
  PetscCall( DMCreateGlobalVector(ctx_initial_p0->dm, &rhs_p0) );
  PetscCall( VecZeroEntries(rhs_p0) );
  PetscCall( DMLocalToGlobal(ctx_initial_p0->dm, rhs_p_loc, ADD_VALUES, rhs_p0) );

  // ----------------------------------------------
  // Solve for P0, M*P0 = rhs_p0
  // ----------------------------------------------
  Vec P0;
  PetscCall( DMCreateGlobalVector(ctx_initial_p0->dm, &P0) );
  PetscCall( VecZeroEntries(P0) );
  PetscInt P0_g_size, P0_l_size;
  PetscCall( VecGetSize(P0, &P0_g_size) );
  // Local size for matShell
  PetscCall( VecGetLocalSize(P0, &P0_l_size) );

  // Operator
  Mat mat_ksp_p0;
  // -- Form Action of residual on u
  PetscCall( MatCreateShell(ctx_initial_p0->comm, P0_l_size, P0_l_size, P0_g_size,
                            P0_g_size, ceed_data->ctx_initial_p0, &mat_ksp_p0) );
  PetscCall( MatShellSetOperation(mat_ksp_p0, MATOP_MULT,
                                  (void (*)(void))ApplyMatOp) );
  PetscCall( MatShellSetVecType(mat_ksp_p0, vec_type) );

  KSP ksp_p0;
  PetscCall( KSPCreate(ctx_initial_p0->comm, &ksp_p0) );
  PetscCall( KSPSetOperators(ksp_p0, mat_ksp_p0, mat_ksp_p0) );
  PetscCall( KSPSetFromOptions(ksp_p0) );
  PetscCall( KSPSetUp(ksp_p0) );
  PetscCall( KSPSolve(ksp_p0, rhs_p0, P0) );

  // ----------------------------------------------
  // Create final initial conditions U
  // ----------------------------------------------
  // Global-to-local for U0, P0
  PetscCall( DMGlobalToLocal(ctx_initial_u0->dm, U0, INSERT_VALUES,
                             ctx_initial_u0->X_loc) );
  PetscCall( DMGlobalToLocal(ctx_initial_p0->dm, P0, INSERT_VALUES,
                             ctx_initial_p0->X_loc) );
  // Get array u0,p0
  const PetscScalar *u0, *p0;
  PetscCall( VecGetArrayRead(ctx_initial_u0->X_loc, &u0) );
  PetscCall( VecGetArrayRead(ctx_initial_p0->X_loc, &p0) );

  // Get array of local vector U = [p,u]
  PetscScalar *u;
  PetscInt U_l_size;
  PetscCall( VecGetLocalSize(U, &U_l_size) );
  PetscCall( VecZeroEntries(ctx_residual_ut->X_loc) );
  PetscCall( VecGetArray(ctx_residual_ut->X_loc, &u) );
  for (PetscInt i = 0; i<ceed_data->num_elem; i++) {
    u[i] = p0[i];
  }
  for (PetscInt i = ceed_data->num_elem; i<U_l_size; i++) {
    u[i] = u0[i-ceed_data->num_elem];
  }
  PetscCall( VecRestoreArray(ctx_residual_ut->X_loc, &u) );
  PetscCall( VecRestoreArrayRead(ctx_initial_p0->X_loc, &p0) );
  PetscCall( VecRestoreArrayRead(ctx_initial_u0->X_loc, &u0) );
  PetscCall( DMLocalToGlobal(ctx_residual_ut->dm, ctx_residual_ut->X_loc,
                             ADD_VALUES, U) );

  // Clean up
  PetscCall( VecDestroy(&rhs_u_loc) );
  PetscCall( VecDestroy(&rhs_u0) );
  PetscCall( VecDestroy(&U0) );
  PetscCall( VecDestroy(&ctx_initial_u0->X_loc) );
  PetscCall( VecDestroy(&ctx_initial_u0->Y_loc) );
  PetscCall( VecDestroy(&rhs_p_loc) );
  PetscCall( VecDestroy(&rhs_p0) );
  PetscCall( VecDestroy(&P0) );
  PetscCall( VecDestroy(&ctx_initial_p0->X_loc) );
  PetscCall( VecDestroy(&ctx_initial_p0->Y_loc) );
  PetscCall( MatDestroy(&mat_ksp_p0) );
  PetscCall( MatDestroy(&mat_ksp_u0) );
  PetscCall( KSPDestroy(&ksp_p0) );
  PetscCall( KSPDestroy(&ksp_u0) );
  PetscFunctionReturn(0);

}

PetscErrorCode TSFormIResidual(TS ts, PetscReal time, Vec X, Vec X_t, Vec Y,
                               void *ctx_residual_ut) {
  OperatorApplyContext ctx   = (OperatorApplyContext)ctx_residual_ut;
  const PetscScalar *x, *x_t;
  PetscScalar *y;
  PetscMemType      x_mem_type, x_t_mem_type, y_mem_type;
  PetscFunctionBeginUser;

  // Update time dependent data
  if(ctx->t != time) {
    CeedOperatorContextSetDouble(ctx->op_apply,
                                 ctx->solution_time_label, &time);
    ctx->t = time;
  }
  //PetscScalar dt;
  //PetscCall( TSGetTimeStep(ts, &dt) );
  //if (ctx->dt != dt) {
  //  CeedOperatorContextSetDouble(ctx->op_apply,
  //                               ctx->timestep_label, &dt);
  //  ctx->dt = dt;
  //}
  // Global-to-local
  PetscCall( DMGlobalToLocal(ctx->dm, X, INSERT_VALUES, ctx->X_loc) );
  PetscCall( DMGlobalToLocal(ctx->dm, X_t, INSERT_VALUES, ctx->X_t_loc) );

  // Place PETSc vectors in CEED vectors
  PetscCall( VecGetArrayReadAndMemType(ctx->X_loc, &x, &x_mem_type) );
  PetscCall( VecGetArrayReadAndMemType(ctx->X_t_loc, &x_t, &x_t_mem_type) );
  PetscCall( VecGetArrayAndMemType(ctx->Y_loc, &y, &y_mem_type) );
  CeedVectorSetArray(ctx->x_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER,
                     (PetscScalar *)x);
  CeedVectorSetArray(ctx->x_t_ceed, MemTypeP2C(x_t_mem_type),
                     CEED_USE_POINTER, (PetscScalar *)x_t);
  CeedVectorSetArray(ctx->y_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER, y);

  // Apply CEED operator
  CeedOperatorApply(ctx->op_apply, ctx->x_ceed, ctx->y_ceed,
                    CEED_REQUEST_IMMEDIATE);

  // Restore vectors
  CeedVectorTakeArray(ctx->x_ceed, MemTypeP2C(x_mem_type), NULL);
  CeedVectorTakeArray(ctx->x_t_ceed, MemTypeP2C(x_t_mem_type), NULL);
  CeedVectorTakeArray(ctx->y_ceed, MemTypeP2C(y_mem_type), NULL);
  PetscCall( VecRestoreArrayReadAndMemType(ctx->X_loc, &x) );
  PetscCall( VecRestoreArrayReadAndMemType(ctx->X_t_loc, &x_t) );
  PetscCall( VecRestoreArrayAndMemType(ctx->Y_loc, &y) );

  // Local-to-Global
  PetscCall( VecZeroEntries(Y) );
  PetscCall( DMLocalToGlobal(ctx->dm, ctx->Y_loc, ADD_VALUES, Y) );

  PetscFunctionReturn(0);
}

// TS: Create, setup, and solve
PetscErrorCode TSSolveRichard(DM dm, CeedData ceed_data, AppCtx app_ctx,
                              Vec *U, TS *ts) {
  MPI_Comm       comm = app_ctx->comm;
  TSAdapt        adapt;
  PetscFunctionBeginUser;

  PetscCall( TSCreate(comm, ts) );
  PetscCall( TSSetDM(*ts, dm) );
  PetscCall( TSSetType(*ts, TSBDF) );
  PetscCall( TSSetIFunction(*ts, NULL, TSFormIResidual,
                            ceed_data->ctx_residual_ut) );

  PetscCall( TSSetMaxTime(*ts, app_ctx->t_final) );
  PetscCall( TSSetExactFinalTime(*ts, TS_EXACTFINALTIME_STEPOVER) );
  PetscCall( TSSetTimeStep(*ts, 1.e-2) );
  PetscCall( TSGetAdapt(*ts, &adapt) );
  PetscCall( TSAdaptSetStepLimits(adapt, 1.e-12, 1.e2) );
  PetscCall( TSSetFromOptions(*ts) );
  ceed_data->ctx_residual_ut->t = -1.0;
  //ceed_data->ctx_residual_ut->dt = -1.0;
  // Solve
  PetscScalar start_time;
  PetscCall( TSGetTime(*ts, &start_time) );

  PetscCall(TSSetTime(*ts, start_time));
  PetscCall(TSSetStepNumber(*ts, 0));

  PetscCall( PetscBarrier((PetscObject) *ts) );
  PetscCall( TSSolve(*ts, *U) );

  PetscScalar    final_time;
  PetscCall( TSGetSolveTime(*ts, &final_time) );
  app_ctx->t_final = final_time;

  PetscFunctionReturn(0);
}


// -----------------------------------------------------------------------------
