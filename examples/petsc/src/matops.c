#include "../include/matops.h"

#include "../include/petscutils.h"

// -----------------------------------------------------------------------------
// Setup apply operator context data
// -----------------------------------------------------------------------------
PetscErrorCode SetupApplyOperatorCtx(MPI_Comm comm, DM dm, Ceed ceed, CeedData ceed_data, Vec X_loc, OperatorApplyContext op_apply_ctx) {
  PetscFunctionBeginUser;

  op_apply_ctx->comm  = comm;
  op_apply_ctx->dm    = dm;
  op_apply_ctx->X_loc = X_loc;
  PetscCall(VecDuplicate(X_loc, &op_apply_ctx->Y_loc));
  op_apply_ctx->x_ceed = ceed_data->x_ceed;
  op_apply_ctx->y_ceed = ceed_data->y_ceed;
  op_apply_ctx->op     = ceed_data->op_apply;
  op_apply_ctx->ceed   = ceed;

  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// Setup error operator context data
// -----------------------------------------------------------------------------
PetscErrorCode SetupErrorOperatorCtx(MPI_Comm comm, DM dm, Ceed ceed, CeedData ceed_data, Vec X_loc, CeedOperator op_error,
                                     OperatorApplyContext op_error_ctx) {
  PetscFunctionBeginUser;

  op_error_ctx->comm  = comm;
  op_error_ctx->dm    = dm;
  op_error_ctx->X_loc = X_loc;
  PetscCall(VecDuplicate(X_loc, &op_error_ctx->Y_loc));
  op_error_ctx->x_ceed = ceed_data->x_ceed;
  op_error_ctx->y_ceed = ceed_data->y_ceed;
  op_error_ctx->op     = op_error;
  op_error_ctx->ceed   = ceed;

  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// This function returns the computed diagonal of the operator
// -----------------------------------------------------------------------------
PetscErrorCode MatGetDiag(Mat A, Vec D) {
  OperatorApplyContext op_apply_ctx;

  PetscFunctionBeginUser;

  PetscCall(MatShellGetContext(A, &op_apply_ctx));

  // Compute Diagonal via libCEED
  PetscScalar *y;
  PetscMemType mem_type;

  // -- Place PETSc vector in libCEED vector
  PetscCall(VecGetArrayAndMemType(op_apply_ctx->Y_loc, &y, &mem_type));
  CeedVectorSetArray(op_apply_ctx->y_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, y);

  // -- Compute Diagonal
  CeedOperatorLinearAssembleDiagonal(op_apply_ctx->op, op_apply_ctx->y_ceed, CEED_REQUEST_IMMEDIATE);

  // -- Local-to-Global
  CeedVectorTakeArray(op_apply_ctx->y_ceed, MemTypeP2C(mem_type), NULL);
  PetscCall(VecRestoreArrayAndMemType(op_apply_ctx->Y_loc, &y));
  PetscCall(VecZeroEntries(D));
  PetscCall(DMLocalToGlobal(op_apply_ctx->dm, op_apply_ctx->Y_loc, ADD_VALUES, D));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the Laplacian with
// Dirichlet boundary conditions
// -----------------------------------------------------------------------------
PetscErrorCode ApplyLocal_Ceed(Vec X, Vec Y, OperatorApplyContext op_apply_ctx) {
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
  CeedOperatorApply(op_apply_ctx->op, op_apply_ctx->x_ceed, op_apply_ctx->y_ceed, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(op_apply_ctx->x_ceed, MemTypeP2C(x_mem_type), NULL);
  CeedVectorTakeArray(op_apply_ctx->y_ceed, MemTypeP2C(y_mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(op_apply_ctx->X_loc, (const PetscScalar **)&x));
  PetscCall(VecRestoreArrayAndMemType(op_apply_ctx->Y_loc, &y));

  // Local-to-global
  PetscCall(VecZeroEntries(Y));
  PetscCall(DMLocalToGlobal(op_apply_ctx->dm, op_apply_ctx->Y_loc, ADD_VALUES, Y));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function wraps the libCEED operator for a MatShell
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_Ceed(Mat A, Vec X, Vec Y) {
  OperatorApplyContext op_apply_ctx;

  PetscFunctionBeginUser;

  PetscCall(MatShellGetContext(A, &op_apply_ctx));

  // libCEED for local action of residual evaluator
  PetscCall(ApplyLocal_Ceed(X, Y, op_apply_ctx));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the prolongation operator
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_Prolong(Mat A, Vec X, Vec Y) {
  ProlongRestrContext pr_restr_ctx;
  PetscScalar        *c, *f;
  PetscMemType        c_mem_type, f_mem_type;

  PetscFunctionBeginUser;

  PetscCall(MatShellGetContext(A, &pr_restr_ctx));

  // Global-to-local
  PetscCall(VecZeroEntries(pr_restr_ctx->loc_vec_c));
  PetscCall(DMGlobalToLocal(pr_restr_ctx->dmc, X, INSERT_VALUES, pr_restr_ctx->loc_vec_c));

  // Setup libCEED vectors
  PetscCall(VecGetArrayReadAndMemType(pr_restr_ctx->loc_vec_c, (const PetscScalar **)&c, &c_mem_type));
  PetscCall(VecGetArrayAndMemType(pr_restr_ctx->loc_vec_f, &f, &f_mem_type));
  CeedVectorSetArray(pr_restr_ctx->ceed_vec_c, MemTypeP2C(c_mem_type), CEED_USE_POINTER, c);
  CeedVectorSetArray(pr_restr_ctx->ceed_vec_f, MemTypeP2C(f_mem_type), CEED_USE_POINTER, f);

  // Apply libCEED operator
  CeedOperatorApply(pr_restr_ctx->op_prolong, pr_restr_ctx->ceed_vec_c, pr_restr_ctx->ceed_vec_f, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(pr_restr_ctx->ceed_vec_c, MemTypeP2C(c_mem_type), NULL);
  CeedVectorTakeArray(pr_restr_ctx->ceed_vec_f, MemTypeP2C(f_mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(pr_restr_ctx->loc_vec_c, (const PetscScalar **)&c));
  PetscCall(VecRestoreArrayAndMemType(pr_restr_ctx->loc_vec_f, &f));

  // Multiplicity
  PetscCall(VecPointwiseMult(pr_restr_ctx->loc_vec_f, pr_restr_ctx->loc_vec_f, pr_restr_ctx->mult_vec));

  // Local-to-global
  PetscCall(VecZeroEntries(Y));
  PetscCall(DMLocalToGlobal(pr_restr_ctx->dmf, pr_restr_ctx->loc_vec_f, ADD_VALUES, Y));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the restriction operator
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_Restrict(Mat A, Vec X, Vec Y) {
  ProlongRestrContext pr_restr_ctx;
  PetscScalar        *c, *f;
  PetscMemType        c_mem_type, f_mem_type;

  PetscFunctionBeginUser;

  PetscCall(MatShellGetContext(A, &pr_restr_ctx));

  // Global-to-local
  PetscCall(VecZeroEntries(pr_restr_ctx->loc_vec_f));
  PetscCall(DMGlobalToLocal(pr_restr_ctx->dmf, X, INSERT_VALUES, pr_restr_ctx->loc_vec_f));

  // Multiplicity
  PetscCall(VecPointwiseMult(pr_restr_ctx->loc_vec_f, pr_restr_ctx->loc_vec_f, pr_restr_ctx->mult_vec));

  // Setup libCEED vectors
  PetscCall(VecGetArrayReadAndMemType(pr_restr_ctx->loc_vec_f, (const PetscScalar **)&f, &f_mem_type));
  PetscCall(VecGetArrayAndMemType(pr_restr_ctx->loc_vec_c, &c, &c_mem_type));
  CeedVectorSetArray(pr_restr_ctx->ceed_vec_f, MemTypeP2C(f_mem_type), CEED_USE_POINTER, f);
  CeedVectorSetArray(pr_restr_ctx->ceed_vec_c, MemTypeP2C(c_mem_type), CEED_USE_POINTER, c);

  // Apply CEED operator
  CeedOperatorApply(pr_restr_ctx->op_restrict, pr_restr_ctx->ceed_vec_f, pr_restr_ctx->ceed_vec_c, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(pr_restr_ctx->ceed_vec_c, MemTypeP2C(c_mem_type), NULL);
  CeedVectorTakeArray(pr_restr_ctx->ceed_vec_f, MemTypeP2C(f_mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(pr_restr_ctx->loc_vec_f, (const PetscScalar **)&f));
  PetscCall(VecRestoreArrayAndMemType(pr_restr_ctx->loc_vec_c, &c));

  // Local-to-global
  PetscCall(VecZeroEntries(Y));
  PetscCall(DMLocalToGlobal(pr_restr_ctx->dmc, pr_restr_ctx->loc_vec_c, ADD_VALUES, Y));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function calculates the error in the final solution
// -----------------------------------------------------------------------------
PetscErrorCode ComputeL2Error(Vec X, PetscScalar *l2_error, OperatorApplyContext op_error_ctx) {
  Vec E;
  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(X, &E));
  PetscCall(ApplyLocal_Ceed(X, E, op_error_ctx));
  PetscScalar error_sq = 1.0;
  PetscCall(VecSum(E, &error_sq));
  *l2_error = sqrt(error_sq);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
