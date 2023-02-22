#include "../include/setup-matops.h"

// -----------------------------------------------------------------------------
// Apply the local action of a libCEED operator and store result in PETSc vector
// i.e. compute A X = Y
// -----------------------------------------------------------------------------
PetscErrorCode ApplyLocalCeedOp(Vec X, Vec Y, OperatorApplyContext op_apply_ctx) {
  PetscFunctionBeginUser;

  // Zero target vector
  PetscCall(VecZeroEntries(Y));

  // Sum into target vector
  PetscCall(ApplyAddLocalCeedOp(X, Y, op_apply_ctx));

  PetscFunctionReturn(0);
}

PetscErrorCode ApplyAddLocalCeedOp(Vec X, Vec Y, OperatorApplyContext op_apply_ctx) {
  PetscScalar *x, *y;
  PetscMemType x_mem_type, y_mem_type;

  PetscFunctionBeginUser;

  // Global-to-local
  PetscCall(DMGlobalToLocal(op_apply_ctx->dm, X, INSERT_VALUES, op_apply_ctx->X_loc));

  // Setup libCEED vectors
  PetscCall(VecGetArrayReadAndMemType(op_apply_ctx->X_loc, (const PetscScalar **)&x, &x_mem_type));
  PetscCall(VecGetArrayAndMemType(op_apply_ctx->Y_loc, &y, &y_mem_type));
  CeedVectorSetArray(op_apply_ctx->x_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER, x);
  CeedVectorSetArray(op_apply_ctx->y_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER, y);

  // Apply libCEED operator
  CeedOperatorApply(op_apply_ctx->op_apply, op_apply_ctx->x_ceed, op_apply_ctx->y_ceed, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(op_apply_ctx->x_ceed, MemTypeP2C(x_mem_type), NULL);
  CeedVectorTakeArray(op_apply_ctx->y_ceed, MemTypeP2C(y_mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(op_apply_ctx->X_loc, (const PetscScalar **)&x));
  PetscCall(VecRestoreArrayAndMemType(op_apply_ctx->Y_loc, &y));

  // Local-to-global
  PetscCall(DMLocalToGlobal(op_apply_ctx->dm, op_apply_ctx->Y_loc, ADD_VALUES, Y));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function returns the computed diagonal of the operator
// -----------------------------------------------------------------------------
PetscErrorCode GetDiagonal(Mat A, Vec D) {
  OperatorApplyContext op_apply_ctx;
  PetscScalar         *x;
  PetscMemType         x_mem_type;

  PetscFunctionBeginUser;

  PetscCall(MatShellGetContext(A, &op_apply_ctx));

  // -- Place PETSc vector in libCEED vector
  PetscCall(VecGetArrayAndMemType(op_apply_ctx->X_loc, &x, &x_mem_type));
  CeedVectorSetArray(op_apply_ctx->x_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER, x);

  // -- Compute Diagonal
  CeedOperatorLinearAssembleDiagonal(op_apply_ctx->op_apply, op_apply_ctx->x_ceed, CEED_REQUEST_IMMEDIATE);

  // -- Local-to-Global
  CeedVectorTakeArray(op_apply_ctx->x_ceed, MemTypeP2C(x_mem_type), NULL);
  PetscCall(VecRestoreArrayAndMemType(op_apply_ctx->X_loc, &x));
  PetscCall(VecZeroEntries(D));
  PetscCall(DMLocalToGlobal(op_apply_ctx->dm, op_apply_ctx->X_loc, ADD_VALUES, D));

  // Cleanup
  PetscCall(VecZeroEntries(op_apply_ctx->X_loc));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
