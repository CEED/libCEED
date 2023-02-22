#include "../include/setup-solvers.h"

#include "../include/setup-libceed.h"
#include "../include/setup-matops.h"
#include "petscvec.h"

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

PetscErrorCode SetupErrorUOperatorCtx(DM dm, Ceed ceed, CeedData ceed_data, OperatorApplyContext ctx_error_u) {
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

PetscErrorCode SetupErrorPOperatorCtx(DM dm, Ceed ceed, CeedData ceed_data, OperatorApplyContext ctx_error_p) {
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

// ---------------------------------------------------------------------------
// Setup Solver
// ---------------------------------------------------------------------------
PetscErrorCode PDESolver(CeedData ceed_data, AppCtx app_ctx, KSP ksp, Vec rhs, Vec *X) {
  PetscInt X_l_size, X_g_size;

  PetscFunctionBeginUser;

  // Create global unknown solution U
  PetscCall(VecGetSize(*X, &X_g_size));
  // Local size for matShell
  PetscCall(VecGetLocalSize(*X, &X_l_size));

  // ---------------------------------------------------------------------------
  // Setup Mat
  // ---------------------------------------------------------------------------
  // Operator
  Mat     mat_op;
  VecType vec_type;
  PetscCall(DMGetVecType(app_ctx->ctx_residual->dm, &vec_type));
  // -- Form Action of Jacobian on delta_u
  PetscCall(MatCreateShell(app_ctx->comm, X_l_size, X_l_size, X_g_size, X_g_size, app_ctx->ctx_residual, &mat_op));
  PetscCall(MatShellSetOperation(mat_op, MATOP_MULT, (void (*)(void))ApplyMatOp));
  PetscCall(MatShellSetOperation(mat_op, MATOP_GET_DIAGONAL, (void (*)(void))GetDiagonal));
  PetscCall(MatShellSetVecType(mat_op, vec_type));

  // Setup KSP
  PetscCall(KSPSetType(ksp, KSPGMRES));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_PRECONDITIONED));
  // PetscCall(KSPSetType(ksp, KSPCG));
  // PetscCall(KSPSetNormType(ksp, KSP_NORM_NATURAL));
  // Setup PC
  PC pc;
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCNONE));

  PetscCall(KSPSetOperators(ksp, mat_op, mat_op));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPViewFromOptions(ksp, NULL, "-ksp_view"));
  PetscCall(VecZeroEntries(*X));
  PetscCall(KSPSolve(ksp, rhs, *X));

  PetscCall(MatDestroy(&mat_op));

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
  PetscCall(VecDestroy(&app_ctx->ctx_error_u->Y_loc));
  PetscCall(VecDestroy(&app_ctx->ctx_error_u->X_loc));
  if (problem_data->mixed) {
    PetscCall(VecDestroy(&app_ctx->ctx_error_p->Y_loc));
    PetscCall(VecDestroy(&app_ctx->ctx_error_p->X_loc));
  }
  PetscFunctionReturn(0);
}