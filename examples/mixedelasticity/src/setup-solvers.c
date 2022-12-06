#include "../include/setup-solvers.h"

#include "../include/setup-libceed.h"
#include "../include/setup-matops.h"
#include "petscvec.h"

// -----------------------------------------------------------------------------
// Setup operator context data
// -----------------------------------------------------------------------------
PetscErrorCode SetupJacobianOperatorCtx(DM dm, Ceed ceed, CeedData ceed_data, OperatorApplyContext ctx_jacobian) {
  PetscFunctionBeginUser;

  ctx_jacobian->dm = dm;
  PetscCall(DMCreateLocalVector(dm, &ctx_jacobian->X_loc));
  PetscCall(VecDuplicate(ctx_jacobian->X_loc, &ctx_jacobian->Y_loc));
  ctx_jacobian->x_ceed   = ceed_data->x_ceed;
  ctx_jacobian->y_ceed   = ceed_data->y_ceed;
  ctx_jacobian->ceed     = ceed;
  ctx_jacobian->op_apply = ceed_data->op_jacobian;

  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// Setup operator context data
// -----------------------------------------------------------------------------
PetscErrorCode SetupResidualOperatorCtx(DM dm, Ceed ceed, CeedData ceed_data, OperatorApplyContext ctx_residual) {
  PetscFunctionBeginUser;

  ctx_residual->dm = dm;
  PetscCall(DMCreateLocalVector(dm, &ctx_residual->X_loc));
  PetscCall(VecDuplicate(ctx_residual->X_loc, &ctx_residual->Y_loc));
  ctx_residual->x_ceed   = ceed_data->x_ceed;
  ctx_residual->y_ceed   = ceed_data->y_ceed;
  ctx_residual->ceed     = ceed;
  ctx_residual->op_apply = ceed_data->op_residual;

  PetscFunctionReturn(0);
}

PetscErrorCode SetupErrorOperatorCtx_u(DM dm, Ceed ceed, CeedData ceed_data, OperatorApplyContext ctx_error_u) {
  PetscFunctionBeginUser;

  ctx_error_u->dm = dm;
  PetscCall(DMCreateLocalVector(dm, &ctx_error_u->X_loc));
  PetscCall(VecDuplicate(ctx_error_u->X_loc, &ctx_error_u->Y_loc));
  ctx_error_u->x_ceed   = ceed_data->x_ceed;
  ctx_error_u->y_ceed   = ceed_data->y_ceed;
  ctx_error_u->ceed     = ceed;
  ctx_error_u->op_apply = ceed_data->op_error_u;

  PetscFunctionReturn(0);
}

PetscErrorCode SetupErrorOperatorCtx_p(DM dm, Ceed ceed, CeedData ceed_data, OperatorApplyContext ctx_error_p) {
  PetscFunctionBeginUser;

  ctx_error_p->dm = dm;
  PetscCall(DMCreateLocalVector(dm, &ctx_error_p->X_loc));
  PetscCall(VecDuplicate(ctx_error_p->X_loc, &ctx_error_p->Y_loc));
  ctx_error_p->x_ceed   = ceed_data->x_ceed;
  ctx_error_p->y_ceed   = ceed_data->y_ceed;
  ctx_error_p->ceed     = ceed;
  ctx_error_p->op_apply = ceed_data->op_error_p;

  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// This function wraps the libCEED operator for a MatShell
// -----------------------------------------------------------------------------
PetscErrorCode ApplyMatOp(Mat A, Vec X, Vec Y) {
  OperatorApplyContext op_apply_ctx;

  PetscFunctionBeginUser;

  PetscCall(MatShellGetContext(A, &op_apply_ctx));

  // libCEED for local action of residual evaluator
  PetscCall(ApplyLocalCeedOp(X, Y, op_apply_ctx));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the non-linear residual
// -----------------------------------------------------------------------------
PetscErrorCode SNESFormResidual(SNES snes, Vec X, Vec Y, void *ctx_residual) {
  OperatorApplyContext ctx = (OperatorApplyContext)ctx_residual;

  PetscFunctionBeginUser;

  // Use computed BCs
  // PetscCall( DMPlexInsertBoundaryValues(ctx->dm, PETSC_TRUE,
  //                                      ctx->X_loc,
  //                                      1.0, NULL, NULL, NULL) );

  // libCEED for local action of residual evaluator
  PetscCall(ApplyLocalCeedOp(X, Y, ctx));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Jacobian setup
// -----------------------------------------------------------------------------
PetscErrorCode SNESFormJacobian(SNES snes, Vec U, Mat J, Mat J_pre, void *ctx_jacobian) {
  // OperatorApplyContext ctx = (OperatorApplyContext)ctx_jacobian;
  PetscFunctionBeginUser;

  // J_pre might be AIJ (e.g., when using coloring), so we need to assemble it
  PetscCall(MatAssemblyBegin(J_pre, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J_pre, MAT_FINAL_ASSEMBLY));
  if (J != J_pre) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
};

// ---------------------------------------------------------------------------
// Setup Solver
// ---------------------------------------------------------------------------
PetscErrorCode PDESolver(CeedData ceed_data, AppCtx app_ctx, SNES snes, KSP ksp, Vec rhs, Vec *X) {
  PetscInt X_l_size, X_g_size;

  PetscFunctionBeginUser;

  // Create global unknown solution U
  PetscCall(VecGetSize(*X, &X_g_size));
  // Local size for matShell
  PetscCall(VecGetLocalSize(*X, &X_l_size));
  Vec R;
  PetscCall(VecDuplicate(*X, &R));
  // ---------------------------------------------------------------------------
  // Setup SNES
  // ---------------------------------------------------------------------------
  // Operator
  Mat     mat_op;
  VecType vec_type;
  PetscCall(SNESSetDM(snes, app_ctx->ctx_jacobian->dm));
  PetscCall(DMGetVecType(app_ctx->ctx_residual->dm, &vec_type));
  // -- Form Action of Jacobian on delta_u
  PetscCall(MatCreateShell(app_ctx->comm, X_l_size, X_l_size, X_g_size, X_g_size, app_ctx->ctx_jacobian, &mat_op));
  PetscCall(MatShellSetOperation(mat_op, MATOP_MULT, (void (*)(void))ApplyMatOp));
  PetscCall(MatShellSetVecType(mat_op, vec_type));

  // Set SNES residual evaluation function
  PetscCall(SNESSetFunction(snes, R, SNESFormResidual, app_ctx->ctx_residual));
  // -- SNES Jacobian
  PetscCall(SNESSetJacobian(snes, mat_op, mat_op, SNESFormJacobian, app_ctx->ctx_jacobian));
  // Setup KSP
  PetscCall(KSPSetFromOptions(ksp));
  // Default to critical-point (CP) line search (related to Wolfe's curvature condition)
  SNESLineSearch line_search;
  PetscCall(SNESGetLineSearch(snes, &line_search));
  PetscCall(SNESLineSearchSetType(line_search, SNESLINESEARCHCP));
  PetscCall(SNESSetFromOptions(snes));
  // Solve
  PetscCall(VecSet(*X, 0.0));
  PetscCall(SNESSolve(snes, rhs, *X));
  // Free PETSc objects
  PetscCall(MatDestroy(&mat_op));
  PetscCall(VecDestroy(&R));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function calculates the error in the final solution
// -----------------------------------------------------------------------------
PetscErrorCode ComputeL2Error(Vec X, PetscScalar *l2_error, OperatorApplyContext op_error_ctx) {
  Vec E;
  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(X, &E));
  PetscCall(ApplyLocalCeedOp(X, E, op_error_ctx));
  PetscScalar error_sq = 1.0;
  PetscCall(VecSum(E, &error_sq));
  *l2_error = sqrt(error_sq);
  PetscCall(VecDestroy(&E));

  PetscFunctionReturn(0);
};

PetscErrorCode CtxVecDestroy(ProblemData problem_data, AppCtx app_ctx) {
  PetscFunctionBegin;

  PetscCall(VecDestroy(&app_ctx->ctx_residual->Y_loc));
  PetscCall(VecDestroy(&app_ctx->ctx_residual->X_loc));
  PetscCall(VecDestroy(&app_ctx->ctx_jacobian->Y_loc));
  PetscCall(VecDestroy(&app_ctx->ctx_jacobian->X_loc));
  PetscCall(VecDestroy(&app_ctx->ctx_error_u->Y_loc));
  PetscCall(VecDestroy(&app_ctx->ctx_error_u->X_loc));
  PetscCall(VecDestroy(&app_ctx->ctx_error_p->Y_loc));
  PetscCall(VecDestroy(&app_ctx->ctx_error_p->X_loc));

  PetscFunctionReturn(0);
}