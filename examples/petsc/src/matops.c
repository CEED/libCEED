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
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

// -----------------------------------------------------------------------------
// This function returns the computed diagonal of the operator
// -----------------------------------------------------------------------------
PetscErrorCode MatGetDiag(Mat A, Vec D) {
  OperatorApplyContext op_apply_ctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(A, &op_apply_ctx));

  // Compute Diagonal via libCEED
  PetscMemType mem_type;

  // Place PETSc vector in libCEED vector
  PetscCall(VecP2C(op_apply_ctx->Y_loc, &mem_type, op_apply_ctx->y_ceed));

  // Compute Diagonal
  CeedOperatorLinearAssembleDiagonal(op_apply_ctx->op, op_apply_ctx->y_ceed, CEED_REQUEST_IMMEDIATE);

  // Local-to-Global
  PetscCall(VecC2P(op_apply_ctx->y_ceed, mem_type, op_apply_ctx->Y_loc));
  PetscCall(VecZeroEntries(D));
  PetscCall(DMLocalToGlobal(op_apply_ctx->dm, op_apply_ctx->Y_loc, ADD_VALUES, D));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the Laplacian with Dirichlet boundary conditions
// -----------------------------------------------------------------------------
PetscErrorCode ApplyLocal_Ceed(Vec X, Vec Y, OperatorApplyContext op_apply_ctx) {
  PetscMemType x_mem_type, y_mem_type;

  PetscFunctionBeginUser;
  // Global-to-local
  PetscCall(DMGlobalToLocal(op_apply_ctx->dm, X, INSERT_VALUES, op_apply_ctx->X_loc));

  // Setup libCEED vectors
  PetscCall(VecReadP2C(op_apply_ctx->X_loc, &x_mem_type, op_apply_ctx->x_ceed));
  PetscCall(VecP2C(op_apply_ctx->Y_loc, &y_mem_type, op_apply_ctx->y_ceed));

  // Apply libCEED operator
  CeedOperatorApply(op_apply_ctx->op, op_apply_ctx->x_ceed, op_apply_ctx->y_ceed, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  PetscCall(VecReadC2P(op_apply_ctx->x_ceed, x_mem_type, op_apply_ctx->X_loc));
  PetscCall(VecC2P(op_apply_ctx->y_ceed, y_mem_type, op_apply_ctx->Y_loc));

  // Local-to-global
  PetscCall(VecZeroEntries(Y));
  PetscCall(DMLocalToGlobal(op_apply_ctx->dm, op_apply_ctx->Y_loc, ADD_VALUES, Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// -----------------------------------------------------------------------------
// This function wraps the libCEED operator for a MatShell
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_Ceed(Mat A, Vec X, Vec Y) {
  OperatorApplyContext op_apply_ctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(A, &op_apply_ctx));

  // libCEED for local action of residual evaluator
  PetscCall(ApplyLocal_Ceed(X, Y, op_apply_ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the prolongation operator
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_Prolong(Mat A, Vec X, Vec Y) {
  ProlongRestrContext pr_restr_ctx;
  PetscMemType        c_mem_type, f_mem_type;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(A, &pr_restr_ctx));

  // Global-to-local
  PetscCall(VecZeroEntries(pr_restr_ctx->loc_vec_c));
  PetscCall(DMGlobalToLocal(pr_restr_ctx->dmc, X, INSERT_VALUES, pr_restr_ctx->loc_vec_c));

  // Setup libCEED vectors
  PetscCall(VecReadP2C(pr_restr_ctx->loc_vec_c, &c_mem_type, pr_restr_ctx->ceed_vec_c));
  PetscCall(VecP2C(pr_restr_ctx->loc_vec_f, &f_mem_type, pr_restr_ctx->ceed_vec_f));

  // Apply libCEED operator
  CeedOperatorApply(pr_restr_ctx->op_prolong, pr_restr_ctx->ceed_vec_c, pr_restr_ctx->ceed_vec_f, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  PetscCall(VecReadC2P(pr_restr_ctx->ceed_vec_c, c_mem_type, pr_restr_ctx->loc_vec_c));
  PetscCall(VecC2P(pr_restr_ctx->ceed_vec_f, f_mem_type, pr_restr_ctx->loc_vec_f));

  // Multiplicity
  PetscCall(VecPointwiseMult(pr_restr_ctx->loc_vec_f, pr_restr_ctx->loc_vec_f, pr_restr_ctx->mult_vec));

  // Local-to-global
  PetscCall(VecZeroEntries(Y));
  PetscCall(DMLocalToGlobal(pr_restr_ctx->dmf, pr_restr_ctx->loc_vec_f, ADD_VALUES, Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the restriction operator
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_Restrict(Mat A, Vec X, Vec Y) {
  ProlongRestrContext pr_restr_ctx;
  PetscMemType        c_mem_type, f_mem_type;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(A, &pr_restr_ctx));

  // Global-to-local
  PetscCall(VecZeroEntries(pr_restr_ctx->loc_vec_f));
  PetscCall(DMGlobalToLocal(pr_restr_ctx->dmf, X, INSERT_VALUES, pr_restr_ctx->loc_vec_f));

  // Multiplicity
  PetscCall(VecPointwiseMult(pr_restr_ctx->loc_vec_f, pr_restr_ctx->loc_vec_f, pr_restr_ctx->mult_vec));

  // Setup libCEED vectors
  PetscCall(VecReadP2C(pr_restr_ctx->loc_vec_f, &f_mem_type, pr_restr_ctx->ceed_vec_f));
  PetscCall(VecP2C(pr_restr_ctx->loc_vec_c, &c_mem_type, pr_restr_ctx->ceed_vec_c));

  // Apply CEED operator
  CeedOperatorApply(pr_restr_ctx->op_restrict, pr_restr_ctx->ceed_vec_f, pr_restr_ctx->ceed_vec_c, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  PetscCall(VecReadC2P(pr_restr_ctx->ceed_vec_f, f_mem_type, pr_restr_ctx->loc_vec_f));
  PetscCall(VecC2P(pr_restr_ctx->ceed_vec_c, c_mem_type, pr_restr_ctx->loc_vec_c));

  // Local-to-global
  PetscCall(VecZeroEntries(Y));
  PetscCall(DMLocalToGlobal(pr_restr_ctx->dmc, pr_restr_ctx->loc_vec_c, ADD_VALUES, Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

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
  PetscFunctionReturn(PETSC_SUCCESS);
}

// -----------------------------------------------------------------------------
