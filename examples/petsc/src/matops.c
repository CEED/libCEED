#include "../include/matops.h"

#include "../include/petscutils.h"

// -----------------------------------------------------------------------------
// This function returns the computed diagonal of the operator
// -----------------------------------------------------------------------------
PetscErrorCode MatGetDiag(Mat A, Vec D) {
  UserO user;

  PetscFunctionBeginUser;

  PetscCall(MatShellGetContext(A, &user));

  // Compute Diagonal via libCEED
  PetscScalar *y;
  PetscMemType mem_type;

  // -- Place PETSc vector in libCEED vector
  PetscCall(VecGetArrayAndMemType(user->Y_loc, &y, &mem_type));
  CeedVectorSetArray(user->y_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, y);

  // -- Compute Diagonal
  CeedOperatorLinearAssembleDiagonal(user->op, user->y_ceed, CEED_REQUEST_IMMEDIATE);

  // -- Local-to-Global
  CeedVectorTakeArray(user->y_ceed, MemTypeP2C(mem_type), NULL);
  PetscCall(VecRestoreArrayAndMemType(user->Y_loc, &y));
  PetscCall(VecZeroEntries(D));
  PetscCall(DMLocalToGlobal(user->dm, user->Y_loc, ADD_VALUES, D));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the Laplacian with
// Dirichlet boundary conditions
// -----------------------------------------------------------------------------
PetscErrorCode ApplyLocal_Ceed(Vec X, Vec Y, UserO user) {
  PetscScalar *x, *y;
  PetscMemType x_mem_type, y_mem_type;

  PetscFunctionBeginUser;

  // Global-to-local
  PetscCall(DMGlobalToLocal(user->dm, X, INSERT_VALUES, user->X_loc));

  // Setup libCEED vectors
  PetscCall(VecGetArrayReadAndMemType(user->X_loc, (const PetscScalar **)&x, &x_mem_type));
  PetscCall(VecGetArrayAndMemType(user->Y_loc, &y, &y_mem_type));
  CeedVectorSetArray(user->x_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER, x);
  CeedVectorSetArray(user->y_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER, y);

  // Apply libCEED operator
  CeedOperatorApply(user->op, user->x_ceed, user->y_ceed, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(user->x_ceed, MemTypeP2C(x_mem_type), NULL);
  CeedVectorTakeArray(user->y_ceed, MemTypeP2C(y_mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(user->X_loc, (const PetscScalar **)&x));
  PetscCall(VecRestoreArrayAndMemType(user->Y_loc, &y));

  // Local-to-global
  PetscCall(VecZeroEntries(Y));
  PetscCall(DMLocalToGlobal(user->dm, user->Y_loc, ADD_VALUES, Y));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function wraps the libCEED operator for a MatShell
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_Ceed(Mat A, Vec X, Vec Y) {
  UserO user;

  PetscFunctionBeginUser;

  PetscCall(MatShellGetContext(A, &user));

  // libCEED for local action of residual evaluator
  PetscCall(ApplyLocal_Ceed(X, Y, user));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function wraps the libCEED operator for a SNES residual evaluation
// -----------------------------------------------------------------------------
PetscErrorCode FormResidual_Ceed(SNES snes, Vec X, Vec Y, void *ctx) {
  UserO user = (UserO)ctx;

  PetscFunctionBeginUser;

  // libCEED for local action of residual evaluator
  PetscCall(ApplyLocal_Ceed(X, Y, user));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the prolongation operator
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_Prolong(Mat A, Vec X, Vec Y) {
  UserProlongRestr user;
  PetscScalar     *c, *f;
  PetscMemType     c_mem_type, f_mem_type;

  PetscFunctionBeginUser;

  PetscCall(MatShellGetContext(A, &user));

  // Global-to-local
  PetscCall(VecZeroEntries(user->loc_vec_c));
  PetscCall(DMGlobalToLocal(user->dmc, X, INSERT_VALUES, user->loc_vec_c));

  // Setup libCEED vectors
  PetscCall(VecGetArrayReadAndMemType(user->loc_vec_c, (const PetscScalar **)&c, &c_mem_type));
  PetscCall(VecGetArrayAndMemType(user->loc_vec_f, &f, &f_mem_type));
  CeedVectorSetArray(user->ceed_vec_c, MemTypeP2C(c_mem_type), CEED_USE_POINTER, c);
  CeedVectorSetArray(user->ceed_vec_f, MemTypeP2C(f_mem_type), CEED_USE_POINTER, f);

  // Apply libCEED operator
  CeedOperatorApply(user->op_prolong, user->ceed_vec_c, user->ceed_vec_f, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(user->ceed_vec_c, MemTypeP2C(c_mem_type), NULL);
  CeedVectorTakeArray(user->ceed_vec_f, MemTypeP2C(f_mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(user->loc_vec_c, (const PetscScalar **)&c));
  PetscCall(VecRestoreArrayAndMemType(user->loc_vec_f, &f));

  // Multiplicity
  PetscCall(VecPointwiseMult(user->loc_vec_f, user->loc_vec_f, user->mult_vec));

  // Local-to-global
  PetscCall(VecZeroEntries(Y));
  PetscCall(DMLocalToGlobal(user->dmf, user->loc_vec_f, ADD_VALUES, Y));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the restriction operator
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_Restrict(Mat A, Vec X, Vec Y) {
  UserProlongRestr user;
  PetscScalar     *c, *f;
  PetscMemType     c_mem_type, f_mem_type;

  PetscFunctionBeginUser;

  PetscCall(MatShellGetContext(A, &user));

  // Global-to-local
  PetscCall(VecZeroEntries(user->loc_vec_f));
  PetscCall(DMGlobalToLocal(user->dmf, X, INSERT_VALUES, user->loc_vec_f));

  // Multiplicity
  PetscCall(VecPointwiseMult(user->loc_vec_f, user->loc_vec_f, user->mult_vec));

  // Setup libCEED vectors
  PetscCall(VecGetArrayReadAndMemType(user->loc_vec_f, (const PetscScalar **)&f, &f_mem_type));
  PetscCall(VecGetArrayAndMemType(user->loc_vec_c, &c, &c_mem_type));
  CeedVectorSetArray(user->ceed_vec_f, MemTypeP2C(f_mem_type), CEED_USE_POINTER, f);
  CeedVectorSetArray(user->ceed_vec_c, MemTypeP2C(c_mem_type), CEED_USE_POINTER, c);

  // Apply CEED operator
  CeedOperatorApply(user->op_restrict, user->ceed_vec_f, user->ceed_vec_c, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(user->ceed_vec_c, MemTypeP2C(c_mem_type), NULL);
  CeedVectorTakeArray(user->ceed_vec_f, MemTypeP2C(f_mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(user->loc_vec_f, (const PetscScalar **)&f));
  PetscCall(VecRestoreArrayAndMemType(user->loc_vec_c, &c));

  // Local-to-global
  PetscCall(VecZeroEntries(Y));
  PetscCall(DMLocalToGlobal(user->dmc, user->loc_vec_c, ADD_VALUES, Y));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function calculates the error in the final solution
// -----------------------------------------------------------------------------
PetscErrorCode ComputeErrorMax(UserO user, CeedOperator op_error, Vec X, CeedVector target, PetscScalar *max_error) {
  PetscScalar *x;
  PetscMemType mem_type;
  CeedVector   collocated_error;
  CeedSize     length;

  PetscFunctionBeginUser;
  CeedVectorGetLength(target, &length);
  CeedVectorCreate(user->ceed, length, &collocated_error);

  // Global-to-local
  PetscCall(DMGlobalToLocal(user->dm, X, INSERT_VALUES, user->X_loc));

  // Setup CEED vector
  PetscCall(VecGetArrayAndMemType(user->X_loc, &x, &mem_type));
  CeedVectorSetArray(user->x_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, x);

  // Apply CEED operator
  CeedOperatorApply(op_error, user->x_ceed, collocated_error, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vector
  CeedVectorTakeArray(user->x_ceed, MemTypeP2C(mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(user->X_loc, (const PetscScalar **)&x));

  // Reduce max error
  *max_error = 0;
  const CeedScalar *e;
  CeedVectorGetArrayRead(collocated_error, CEED_MEM_HOST, &e);
  for (CeedInt i = 0; i < length; i++) {
    *max_error = PetscMax(*max_error, PetscAbsScalar(e[i]));
  }
  CeedVectorRestoreArrayRead(collocated_error, &e);
  PetscCall(MPI_Allreduce(MPI_IN_PLACE, max_error, 1, MPIU_REAL, MPIU_MAX, user->comm));

  // Cleanup
  CeedVectorDestroy(&collocated_error);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
