#include "../include/matops.h"
#include "../include/setup-libceed.h"

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the Laplacian with
// Dirichlet boundary conditions
// -----------------------------------------------------------------------------
PetscErrorCode ApplyLocal_Ceed(Vec X, Vec Y, User user) {
  PetscErrorCode ierr;
  PetscScalar *x, *y;
  PetscMemType x_mem_type, y_mem_type;

  PetscFunctionBeginUser;

  // Global-to-local
  ierr = DMGlobalToLocal(user->dm, X, INSERT_VALUES, user->X_loc); CHKERRQ(ierr);

  // Setup libCEED vectors
  ierr = VecGetArrayReadAndMemType(user->X_loc, (const PetscScalar **)&x,
                                   &x_mem_type); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(user->Y_loc, &y, &y_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(user->x_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER, x);
  CeedVectorSetArray(user->y_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER, y);

  // Apply libCEED operator
  CeedOperatorApply(user->op, user->x_ceed, user->y_ceed, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(user->x_ceed, MemTypeP2C(x_mem_type), NULL);
  CeedVectorTakeArray(user->y_ceed, MemTypeP2C(y_mem_type), NULL);
  ierr = VecRestoreArrayReadAndMemType(user->X_loc, (const PetscScalar **)&x);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(user->Y_loc, &y); CHKERRQ(ierr);

  // Local-to-global
  ierr = VecZeroEntries(Y); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, user->Y_loc, ADD_VALUES, Y); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function wraps the libCEED operator for a MatShell
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_Ceed(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  User user;

  PetscFunctionBeginUser;

  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);

  // libCEED for local action of residual evaluator
  ierr = ApplyLocal_Ceed(X, Y, user); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
