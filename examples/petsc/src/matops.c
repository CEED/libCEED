#include "../include/matops.h"
#include "../include/petscutils.h"

// -----------------------------------------------------------------------------
// This function returns the computed diagonal of the operator
// -----------------------------------------------------------------------------
PetscErrorCode MatGetDiag(Mat A, Vec D) {
  PetscErrorCode ierr;
  UserO user;

  PetscFunctionBeginUser;

  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);

  // Compute Diagonal via libCEED
  PetscScalar *x;
  PetscMemType memtype;

  // -- Place PETSc vector in libCEED vector
  ierr = VecGetArrayAndMemType(user->Xloc, &x, &memtype); CHKERRQ(ierr);
  CeedVectorSetArray(user->Xceed, MemTypeP2C(memtype), CEED_USE_POINTER, x);

  // -- Compute Diagonal
  CeedOperatorLinearAssembleDiagonal(user->op, user->Xceed,
                                     CEED_REQUEST_IMMEDIATE);

  // -- Local-to-Global
  CeedVectorTakeArray(user->Xceed, MemTypeP2C(memtype), NULL);
  ierr = VecRestoreArrayAndMemType(user->Xloc, &x); CHKERRQ(ierr);
  ierr = VecZeroEntries(D); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, user->Xloc, ADD_VALUES, D); CHKERRQ(ierr);

  // Cleanup
  ierr = VecZeroEntries(user->Xloc); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the Laplacian with
// Dirichlet boundary conditions
// -----------------------------------------------------------------------------
PetscErrorCode ApplyLocal_Ceed(Vec X, Vec Y, UserO user) {
  PetscErrorCode ierr;
  PetscScalar *x, *y;
  PetscMemType xmemtype, ymemtype;

  PetscFunctionBeginUser;

  // Global-to-local
  ierr = DMGlobalToLocal(user->dm, X, INSERT_VALUES, user->Xloc); CHKERRQ(ierr);

  // Setup libCEED vectors
  ierr = VecGetArrayReadAndMemType(user->Xloc, (const PetscScalar **)&x,
                                   &xmemtype); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(user->Yloc, &y, &ymemtype); CHKERRQ(ierr);
  CeedVectorSetArray(user->Xceed, MemTypeP2C(xmemtype), CEED_USE_POINTER, x);
  CeedVectorSetArray(user->Yceed, MemTypeP2C(ymemtype), CEED_USE_POINTER, y);

  // Apply libCEED operator
  CeedOperatorApply(user->op, user->Xceed, user->Yceed, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(user->Xceed, MemTypeP2C(xmemtype), NULL);
  CeedVectorTakeArray(user->Yceed, MemTypeP2C(ymemtype), NULL);
  ierr = VecRestoreArrayReadAndMemType(user->Xloc, (const PetscScalar **)&x);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(user->Yloc, &y); CHKERRQ(ierr);

  // Local-to-global
  ierr = VecZeroEntries(Y); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, user->Yloc, ADD_VALUES, Y); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function wraps the libCEED operator for a MatShell
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_Ceed(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  UserO user;

  PetscFunctionBeginUser;

  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);

  // libCEED for local action of residual evaluator
  ierr = ApplyLocal_Ceed(X, Y, user); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function wraps the libCEED operator for a SNES residual evaluation
// -----------------------------------------------------------------------------
PetscErrorCode FormResidual_Ceed(SNES snes, Vec X, Vec Y, void *ctx) {
  PetscErrorCode ierr;
  UserO user = (UserO)ctx;

  PetscFunctionBeginUser;

  // libCEED for local action of residual evaluator
  ierr = ApplyLocal_Ceed(X, Y, user); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the prolongation operator
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_Prolong(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  UserProlongRestr user;
  PetscScalar *c, *f;
  PetscMemType cmemtype, fmemtype;

  PetscFunctionBeginUser;

  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);

  // Global-to-local
  ierr = VecZeroEntries(user->locvecc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dmc, X, INSERT_VALUES, user->locvecc);
  CHKERRQ(ierr);

  // Setup libCEED vectors
  ierr = VecGetArrayReadAndMemType(user->locvecc, (const PetscScalar **)&c,
                                   &cmemtype); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(user->locvecf, &f, &fmemtype); CHKERRQ(ierr);
  CeedVectorSetArray(user->ceedvecc, MemTypeP2C(cmemtype), CEED_USE_POINTER, c);
  CeedVectorSetArray(user->ceedvecf, MemTypeP2C(fmemtype), CEED_USE_POINTER, f);

  // Apply libCEED operator
  CeedOperatorApply(user->opProlong, user->ceedvecc, user->ceedvecf,
                    CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(user->ceedvecc, MemTypeP2C(cmemtype), NULL);
  CeedVectorTakeArray(user->ceedvecf, MemTypeP2C(fmemtype), NULL);
  ierr = VecRestoreArrayReadAndMemType(user->locvecc, (const PetscScalar **)&c);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(user->locvecf, &f); CHKERRQ(ierr);

  // Multiplicity
  ierr = VecPointwiseMult(user->locvecf, user->locvecf, user->multvec);

  // Local-to-global
  ierr = VecZeroEntries(Y); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dmf, user->locvecf, ADD_VALUES, Y);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the restriction operator
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_Restrict(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  UserProlongRestr user;
  PetscScalar *c, *f;
  PetscMemType cmemtype, fmemtype;

  PetscFunctionBeginUser;

  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);

  // Global-to-local
  ierr = VecZeroEntries(user->locvecf); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dmf, X, INSERT_VALUES, user->locvecf);
  CHKERRQ(ierr);

  // Multiplicity
  ierr = VecPointwiseMult(user->locvecf, user->locvecf, user->multvec);
  CHKERRQ(ierr);

  // Setup libCEED vectors
  ierr = VecGetArrayReadAndMemType(user->locvecf, (const PetscScalar **)&f,
                                   &fmemtype); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(user->locvecc, &c, &cmemtype); CHKERRQ(ierr);
  CeedVectorSetArray(user->ceedvecf, MemTypeP2C(fmemtype), CEED_USE_POINTER, f);
  CeedVectorSetArray(user->ceedvecc, MemTypeP2C(cmemtype), CEED_USE_POINTER, c);

  // Apply CEED operator
  CeedOperatorApply(user->opRestrict, user->ceedvecf, user->ceedvecc,
                    CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(user->ceedvecc, MemTypeP2C(cmemtype), NULL);
  CeedVectorTakeArray(user->ceedvecf, MemTypeP2C(fmemtype), NULL);
  ierr = VecRestoreArrayReadAndMemType(user->locvecf, (const PetscScalar **)&f);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(user->locvecc, &c); CHKERRQ(ierr);

  // Local-to-global
  ierr = VecZeroEntries(Y); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dmc, user->locvecc, ADD_VALUES, Y);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function calculates the error in the final solution
// -----------------------------------------------------------------------------
PetscErrorCode ComputeErrorMax(UserO user, CeedOperator opError,
                               Vec X, CeedVector target,
                               PetscScalar *maxerror) {
  PetscErrorCode ierr;
  PetscScalar *x;
  PetscMemType memtype;
  CeedVector collocated_error;
  CeedInt length;

  PetscFunctionBeginUser;
  CeedVectorGetLength(target, &length);
  CeedVectorCreate(user->ceed, length, &collocated_error);

  // Global-to-local
  ierr = DMGlobalToLocal(user->dm, X, INSERT_VALUES, user->Xloc); CHKERRQ(ierr);

  // Setup CEED vector
  ierr = VecGetArrayAndMemType(user->Xloc, &x, &memtype); CHKERRQ(ierr);
  CeedVectorSetArray(user->Xceed, MemTypeP2C(memtype), CEED_USE_POINTER, x);

  // Apply CEED operator
  CeedOperatorApply(opError, user->Xceed, collocated_error,
                    CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vector
  CeedVectorTakeArray(user->Xceed, MemTypeP2C(memtype), NULL);
  ierr = VecRestoreArrayReadAndMemType(user->Xloc, (const PetscScalar **)&x);
  CHKERRQ(ierr);

  // Reduce max error
  *maxerror = 0;
  const CeedScalar *e;
  CeedVectorGetArrayRead(collocated_error, CEED_MEM_HOST, &e);
  for (CeedInt i=0; i<length; i++) {
    *maxerror = PetscMax(*maxerror, PetscAbsScalar(e[i]));
  }
  CeedVectorRestoreArrayRead(collocated_error, &e);
  ierr = MPI_Allreduce(MPI_IN_PLACE, maxerror, 1, MPIU_REAL, MPIU_MAX,
                       user->comm);
  CHKERRQ(ierr);

  // Cleanup
  CeedVectorDestroy(&collocated_error);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
