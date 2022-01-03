#include "../include/setup-ts.h"
#include "../include/setup-matops.h"
#include "../include/setup-libceed.h"
#include "petscerror.h"


// -----------------------------------------------------------------------------
// Create global initial conditions vector
// -----------------------------------------------------------------------------
PetscErrorCode CreateInitialConditions(DM dm, CeedData ceed_data, Vec *U0) {

  PetscScalar *u0;
  PetscMemType u0_mem_type;
  Vec U0_loc;

  PetscFunctionBeginUser;

  PetscCall( DMCreateLocalVector(dm, &U0_loc) );
  PetscCall( VecZeroEntries(U0_loc) );

  PetscCall( VecGetArrayAndMemType(U0_loc, &u0, &u0_mem_type) );
  CeedVectorSetArray(ceed_data->U0_ceed, MemTypeP2C(u0_mem_type),
                     CEED_USE_POINTER, u0);
  // Apply libCEED operator
  CeedOperatorApply(ceed_data->op_ics, ceed_data->x_coord, ceed_data->U0_ceed,
                    CEED_REQUEST_IMMEDIATE);
  // Restore PETSc vectors
  CeedVectorTakeArray(ceed_data->U0_ceed, MemTypeP2C(u0_mem_type), NULL);
  PetscCall( VecRestoreArrayAndMemType(U0_loc, &u0) );

  // Create global initial conditions
  PetscCall( DMCreateGlobalVector(dm, U0) );
  PetscCall( VecZeroEntries(*U0) );
  // Local-to-global
  PetscCall( DMLocalToGlobal(dm, U0_loc, ADD_VALUES, *U0) );

  // -- Cleanup
  CeedVectorDestroy(&ceed_data->U0_ceed);
  CeedQFunctionDestroy(&ceed_data->qf_ics);
  CeedOperatorDestroy(&ceed_data->op_ics);
  // Free PETSc objects
  PetscCall( VecDestroy(&U0_loc) );
  PetscFunctionReturn(0);

}
// -----------------------------------------------------------------------------
// Setup operator context data
// -----------------------------------------------------------------------------
PetscErrorCode SetupResidualOperatorCtx_Ut(DM dm, Ceed ceed, CeedData ceed_data,
    OperatorApplyContext ctx_residual_ut) {
  PetscFunctionBeginUser;

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

PetscErrorCode TSFormIResidual(TS ts, PetscReal time, Vec X, Vec X_t, Vec Y,
                               void *ctx_residual_ut) {
  OperatorApplyContext ctx   = (OperatorApplyContext)ctx_residual_ut;
  const PetscScalar *x, *x_t;
  PetscScalar *y;
  Vec               X_loc = ctx->X_loc, X_t_loc = ctx->X_t_loc,
                    Y_loc = ctx->Y_loc;
  PetscMemType      x_mem_type, x_t_mem_type, y_mem_type;
  PetscFunctionBeginUser;

  // Update time dependent data


  // Global-to-local
  PetscCall( DMGlobalToLocal(ctx->dm, X, INSERT_VALUES, X_loc) );
  PetscCall( DMGlobalToLocal(ctx->dm, X_t, INSERT_VALUES, X_t_loc) );

  // Place PETSc vectors in CEED vectors
  PetscCall( VecGetArrayReadAndMemType(X_loc, &x, &x_mem_type) );
  PetscCall( VecGetArrayReadAndMemType(X_t_loc, &x_t, &x_t_mem_type) );
  PetscCall( VecGetArrayAndMemType(Y_loc, &y, &y_mem_type) );
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
  PetscCall( VecRestoreArrayReadAndMemType(X_loc, &x) );
  PetscCall( VecRestoreArrayReadAndMemType(X_t_loc, &x_t) );
  PetscCall( VecRestoreArrayAndMemType(Y_loc, &y) );

  // Local-to-Global
  PetscCall( VecZeroEntries(Y) );
  PetscCall( DMLocalToGlobal(ctx->dm, Y_loc, ADD_VALUES, Y) );

  // Restore vectors
  PetscCall( DMRestoreLocalVector(ctx->dm, &Y_loc) );

  PetscFunctionReturn(0);
}

// TS: Create, setup, and solve
PetscErrorCode TSSolveRichard(DM dm, CeedData ceed_data, AppCtx app_ctx,
                              Vec *U, PetscScalar *f_time, TS *ts) {
  MPI_Comm       comm = app_ctx->comm;
  TSAdapt        adapt;
  PetscFunctionBeginUser;

  PetscCall( TSCreate(comm, ts) );
  PetscCall( TSSetDM(*ts, dm) );
  PetscCall( TSSetIFunction(*ts, NULL, TSFormIResidual,
                            ceed_data->ctx_residual_ut) );

  PetscCall( TSSetMaxTime(*ts, 10) );
  PetscCall( TSSetExactFinalTime(*ts, TS_EXACTFINALTIME_STEPOVER) );
  PetscCall( TSSetTimeStep(*ts, 1.e-2) );
  PetscCall( TSGetAdapt(*ts, &adapt) );
  PetscCall( TSAdaptSetStepLimits(adapt, 1.e-12, 1.e2) );
  PetscCall( TSSetFromOptions(*ts) );

  // Solve
  PetscScalar start_time;
  PetscCall( TSGetTime(*ts, &start_time) );

  PetscCall(TSSetTime(*ts, start_time));
  PetscCall(TSSetStepNumber(*ts, 0));

  PetscCall( PetscBarrier((PetscObject) *ts) );
  PetscCall( TSSolve(*ts, *U) );

  PetscScalar    final_time;
  PetscCall( TSGetSolveTime(*ts, &final_time) );
  *f_time = final_time;

  PetscFunctionReturn(0);
}


// -----------------------------------------------------------------------------
