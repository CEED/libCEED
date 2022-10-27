#include "../include/matops.h"
#include "../include/petscutils.h"

// -----------------------------------------------------------------------------
// Setup apply operator context data
// -----------------------------------------------------------------------------
PetscErrorCode SetupApplyOperatorCtx(MPI_Comm comm, DM dm, Ceed ceed,
                                     CeedData ceed_data, Vec X_loc,
                                     OperatorApplyContext op_apply_ctx) {
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  op_apply_ctx->comm = comm;
  op_apply_ctx->dm = dm;
  op_apply_ctx->X_loc = X_loc;
  ierr = VecDuplicate(X_loc, &op_apply_ctx->Y_loc); CHKERRQ(ierr);
  op_apply_ctx->x_ceed = ceed_data->x_ceed;
  op_apply_ctx->y_ceed = ceed_data->y_ceed;
  op_apply_ctx->op = ceed_data->op_apply;
  op_apply_ctx->ceed = ceed;
  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// Setup error operator context data
// -----------------------------------------------------------------------------
PetscErrorCode SetupErrorOperatorCtx(MPI_Comm comm, DM dm, Ceed ceed,
                                     CeedData ceed_data, Vec X_loc, CeedOperator op_error,
                                     OperatorApplyContext op_error_ctx) {
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  op_error_ctx->comm = comm;
  op_error_ctx->dm = dm;
  op_error_ctx->X_loc = X_loc;
  ierr = VecDuplicate(X_loc, &op_error_ctx->Y_loc); CHKERRQ(ierr);
  op_error_ctx->x_ceed = ceed_data->x_ceed;
  op_error_ctx->y_ceed = ceed_data->y_ceed;
  op_error_ctx->op = op_error;
  op_error_ctx->ceed = ceed;
  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// This function returns the computed diagonal of the operator
// -----------------------------------------------------------------------------
PetscErrorCode MatGetDiag(Mat A, Vec D) {
  PetscErrorCode ierr;
  OperatorApplyContext op_apply_ctx;

  PetscFunctionBeginUser;

  ierr = MatShellGetContext(A, &op_apply_ctx); CHKERRQ(ierr);

  // Compute Diagonal via libCEED
  PetscScalar *y;
  PetscMemType mem_type;

  // -- Place PETSc vector in libCEED vector
  ierr = VecGetArrayAndMemType(op_apply_ctx->Y_loc, &y, &mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(op_apply_ctx->y_ceed, MemTypeP2C(mem_type),
                     CEED_USE_POINTER, y);

  // -- Compute Diagonal
  CeedOperatorLinearAssembleDiagonal(op_apply_ctx->op, op_apply_ctx->y_ceed,
                                     CEED_REQUEST_IMMEDIATE);

  // -- Local-to-Global
  CeedVectorTakeArray(op_apply_ctx->y_ceed, MemTypeP2C(mem_type), NULL);
  ierr = VecRestoreArrayAndMemType(op_apply_ctx->Y_loc, &y); CHKERRQ(ierr);
  ierr = VecZeroEntries(D); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(op_apply_ctx->dm, op_apply_ctx->Y_loc, ADD_VALUES, D);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the Laplacian with
// Dirichlet boundary conditions
// -----------------------------------------------------------------------------
PetscErrorCode ApplyLocal_Ceed(Vec X, Vec Y,
                               OperatorApplyContext op_apply_ctx) {
  PetscErrorCode ierr;
  PetscScalar *x, *y;
  PetscMemType x_mem_type, y_mem_type;

  PetscFunctionBeginUser;

  // Global-to-local
  ierr = DMGlobalToLocal(op_apply_ctx->dm, X, INSERT_VALUES, op_apply_ctx->X_loc);
  CHKERRQ(ierr);

  // Setup libCEED vectors
  ierr = VecGetArrayReadAndMemType(op_apply_ctx->X_loc, (const PetscScalar **)&x,
                                   &x_mem_type); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(op_apply_ctx->Y_loc, &y, &y_mem_type);
  CHKERRQ(ierr);
  CeedVectorSetArray(op_apply_ctx->x_ceed, MemTypeP2C(x_mem_type),
                     CEED_USE_POINTER, x);
  CeedVectorSetArray(op_apply_ctx->y_ceed, MemTypeP2C(y_mem_type),
                     CEED_USE_POINTER, y);

  // Apply libCEED operator
  CeedOperatorApply(op_apply_ctx->op, op_apply_ctx->x_ceed, op_apply_ctx->y_ceed,
                    CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(op_apply_ctx->x_ceed, MemTypeP2C(x_mem_type), NULL);
  CeedVectorTakeArray(op_apply_ctx->y_ceed, MemTypeP2C(y_mem_type), NULL);
  ierr = VecRestoreArrayReadAndMemType(op_apply_ctx->X_loc,
                                       (const PetscScalar **)&x);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(op_apply_ctx->Y_loc, &y); CHKERRQ(ierr);

  // Local-to-global
  ierr = VecZeroEntries(Y); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(op_apply_ctx->dm, op_apply_ctx->Y_loc, ADD_VALUES, Y);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function wraps the libCEED operator for a MatShell
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_Ceed(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  OperatorApplyContext op_apply_ctx;

  PetscFunctionBeginUser;

  ierr = MatShellGetContext(A, &op_apply_ctx); CHKERRQ(ierr);

  // libCEED for local action of residual evaluator
  ierr = ApplyLocal_Ceed(X, Y, op_apply_ctx); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function wraps the libCEED operator for a SNES residual evaluation
// -----------------------------------------------------------------------------
PetscErrorCode FormResidual_Ceed(SNES snes, Vec X, Vec Y, void *ctx) {
  PetscErrorCode ierr;
  OperatorApplyContext op_apply_ctx = (OperatorApplyContext)ctx;

  PetscFunctionBeginUser;

  // libCEED for local action of residual evaluator
  ierr = ApplyLocal_Ceed(X, Y, op_apply_ctx); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the prolongation operator
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_Prolong(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  ProlongRestrContext pr_restr_ctx;
  PetscScalar *c, *f;
  PetscMemType c_mem_type, f_mem_type;

  PetscFunctionBeginUser;

  ierr = MatShellGetContext(A, &pr_restr_ctx); CHKERRQ(ierr);

  // Global-to-local
  ierr = VecZeroEntries(pr_restr_ctx->loc_vec_c); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(pr_restr_ctx->dmc, X, INSERT_VALUES,
                         pr_restr_ctx->loc_vec_c);
  CHKERRQ(ierr);

  // Setup libCEED vectors
  ierr = VecGetArrayReadAndMemType(pr_restr_ctx->loc_vec_c,
                                   (const PetscScalar **)&c,
                                   &c_mem_type); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(pr_restr_ctx->loc_vec_f, &f, &f_mem_type);
  CHKERRQ(ierr);
  CeedVectorSetArray(pr_restr_ctx->ceed_vec_c, MemTypeP2C(c_mem_type),
                     CEED_USE_POINTER, c);
  CeedVectorSetArray(pr_restr_ctx->ceed_vec_f, MemTypeP2C(f_mem_type),
                     CEED_USE_POINTER, f);

  // Apply libCEED operator
  CeedOperatorApply(pr_restr_ctx->op_prolong, pr_restr_ctx->ceed_vec_c,
                    pr_restr_ctx->ceed_vec_f, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(pr_restr_ctx->ceed_vec_c, MemTypeP2C(c_mem_type), NULL);
  CeedVectorTakeArray(pr_restr_ctx->ceed_vec_f, MemTypeP2C(f_mem_type), NULL);
  ierr = VecRestoreArrayReadAndMemType(pr_restr_ctx->loc_vec_c,
                                       (const PetscScalar **)&c);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(pr_restr_ctx->loc_vec_f, &f); CHKERRQ(ierr);

  // Multiplicity
  ierr = VecPointwiseMult(pr_restr_ctx->loc_vec_f, pr_restr_ctx->loc_vec_f,
                          pr_restr_ctx->mult_vec);

  // Local-to-global
  ierr = VecZeroEntries(Y); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(pr_restr_ctx->dmf, pr_restr_ctx->loc_vec_f, ADD_VALUES,
                         Y); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the restriction operator
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_Restrict(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  ProlongRestrContext pr_restr_ctx;
  PetscScalar *c, *f;
  PetscMemType c_mem_type, f_mem_type;

  PetscFunctionBeginUser;

  ierr = MatShellGetContext(A, &pr_restr_ctx); CHKERRQ(ierr);

  // Global-to-local
  ierr = VecZeroEntries(pr_restr_ctx->loc_vec_f); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(pr_restr_ctx->dmf, X, INSERT_VALUES,
                         pr_restr_ctx->loc_vec_f);
  CHKERRQ(ierr);

  // Multiplicity
  ierr = VecPointwiseMult(pr_restr_ctx->loc_vec_f, pr_restr_ctx->loc_vec_f,
                          pr_restr_ctx->mult_vec);
  CHKERRQ(ierr);

  // Setup libCEED vectors
  ierr = VecGetArrayReadAndMemType(pr_restr_ctx->loc_vec_f,
                                   (const PetscScalar **)&f,
                                   &f_mem_type); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(pr_restr_ctx->loc_vec_c, &c, &c_mem_type);
  CHKERRQ(ierr);
  CeedVectorSetArray(pr_restr_ctx->ceed_vec_f, MemTypeP2C(f_mem_type),
                     CEED_USE_POINTER, f);
  CeedVectorSetArray(pr_restr_ctx->ceed_vec_c, MemTypeP2C(c_mem_type),
                     CEED_USE_POINTER, c);

  // Apply CEED operator
  CeedOperatorApply(pr_restr_ctx->op_restrict, pr_restr_ctx->ceed_vec_f,
                    pr_restr_ctx->ceed_vec_c, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(pr_restr_ctx->ceed_vec_c, MemTypeP2C(c_mem_type), NULL);
  CeedVectorTakeArray(pr_restr_ctx->ceed_vec_f, MemTypeP2C(f_mem_type), NULL);
  ierr = VecRestoreArrayReadAndMemType(pr_restr_ctx->loc_vec_f,
                                       (const PetscScalar **)&f); CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(pr_restr_ctx->loc_vec_c, &c); CHKERRQ(ierr);

  // Local-to-global
  ierr = VecZeroEntries(Y); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(pr_restr_ctx->dmc, pr_restr_ctx->loc_vec_c, ADD_VALUES,
                         Y);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function calculates the error in the final solution
// -----------------------------------------------------------------------------
PetscErrorCode ComputeErrorMax(OperatorApplyContext op_error_ctx,
                               Vec X, CeedVector target,
                               PetscScalar *max_error) {
  PetscErrorCode ierr;
  PetscScalar *x;
  PetscMemType mem_type;
  CeedVector collocated_error;
  CeedSize length;

  PetscFunctionBeginUser;
  CeedVectorGetLength(target, &length);
  CeedVectorCreate(op_error_ctx->ceed, length, &collocated_error);

  // Global-to-local
  ierr = DMGlobalToLocal(op_error_ctx->dm, X, INSERT_VALUES, op_error_ctx->X_loc);
  CHKERRQ(ierr);

  // Setup CEED vector
  ierr = VecGetArrayAndMemType(op_error_ctx->X_loc, &x, &mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(op_error_ctx->x_ceed, MemTypeP2C(mem_type),
                     CEED_USE_POINTER, x);

  // Apply CEED operator
  CeedOperatorApply(op_error_ctx->op, op_error_ctx->x_ceed, collocated_error,
                    CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vector
  CeedVectorTakeArray(op_error_ctx->x_ceed, MemTypeP2C(mem_type), NULL);
  ierr = VecRestoreArrayReadAndMemType(op_error_ctx->X_loc,
                                       (const PetscScalar **)&x);
  CHKERRQ(ierr);

  // Reduce max error
  *max_error = 0;
  const CeedScalar *e;
  CeedVectorGetArrayRead(collocated_error, CEED_MEM_HOST, &e);
  for (CeedInt i=0; i<length; i++) {
    *max_error = PetscMax(*max_error, PetscAbsScalar(e[i]));
  }
  CeedVectorRestoreArrayRead(collocated_error, &e);
  ierr = MPI_Allreduce(MPI_IN_PLACE, max_error, 1, MPIU_REAL, MPIU_MAX,
                       op_error_ctx->comm);
  CHKERRQ(ierr);

  // Cleanup
  CeedVectorDestroy(&collocated_error);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
