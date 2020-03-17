// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "../elasticity.h"

// -----------------------------------------------------------------------------
// libCEED Operators for MatShell
// -----------------------------------------------------------------------------
// This function uses libCEED to compute the local action of an operator
PetscErrorCode ApplyLocalCeedOp(Vec X, Vec Y, UserMult user) {
  PetscErrorCode ierr;
  PetscScalar *x, *y;

  PetscFunctionBeginUser;

  // Global-to-local
  ierr = DMGlobalToLocal(user->dm, X, INSERT_VALUES, user->Xloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(user->Yloc); CHKERRQ(ierr);

  // Setup CEED vectors
  ierr = VecGetArrayRead(user->Xloc, (const PetscScalar **)&x); CHKERRQ(ierr);
  ierr = VecGetArray(user->Yloc, &y); CHKERRQ(ierr);
  CeedVectorSetArray(user->Xceed, CEED_MEM_HOST, CEED_USE_POINTER, x);
  CeedVectorSetArray(user->Yceed, CEED_MEM_HOST, CEED_USE_POINTER, y);

  // Apply CEED operator
  CeedOperatorApply(user->op, user->Xceed, user->Yceed, CEED_REQUEST_IMMEDIATE);
  CeedVectorSyncArray(user->Yceed, CEED_MEM_HOST);

  // Restore PETSc vectors
  ierr = VecRestoreArrayRead(user->Xloc, (const PetscScalar **)&x);
  CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Yloc, &y); CHKERRQ(ierr);

  // Local-to-global
  ierr = VecZeroEntries(Y); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, user->Yloc, ADD_VALUES, Y); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// This function uses libCEED to compute the non-linear residual
PetscErrorCode FormResidual_Ceed(SNES snes, Vec X, Vec Y, void *ctx) {
  PetscErrorCode ierr;
  UserMult user = (UserMult)ctx;

  PetscFunctionBeginUser;

  // Use computed BCs
  ierr = VecZeroEntries(user->Xloc); CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(user->dm, PETSC_TRUE, user->Xloc,
                                    user->loadIncrement, NULL, NULL, NULL);
  CHKERRQ(ierr);

  // libCEED for local action of residual evaluator
  ierr = ApplyLocalCeedOp(X, Y, user); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// This function uses libCEED to apply the Jacobian for assembly via a SNES
PetscErrorCode ApplyJacobianCoarse_Ceed(SNES snes, Vec X, Vec Y, void *ctx) {
  PetscErrorCode ierr;
  UserMult user = (UserMult)ctx;

  PetscFunctionBeginUser;

  // Use computed BCs
  ierr = VecZeroEntries(user->Xloc); CHKERRQ(ierr);

  // libCEED for local action of residual evaluator
  ierr = ApplyLocalCeedOp(X, Y, user); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// This function uses libCEED to compute the action of the Jacobian
PetscErrorCode ApplyJacobian_Ceed(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  UserMult user;

  PetscFunctionBeginUser;

  // Zero boundary values
  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);
  ierr = VecZeroEntries(user->Xloc); CHKERRQ(ierr);

  // libCEED for local action of Jacobian
  ierr = ApplyLocalCeedOp(X, Y, user); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// This function uses libCEED to compute the action of the prolongation operator
PetscErrorCode Prolong_Ceed(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  UserMultProlongRestr user;
  PetscScalar *c, *f;

  PetscFunctionBeginUser;

  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);

  // Global-to-local
  ierr = VecZeroEntries(user->locVecC); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dmC, X, INSERT_VALUES, user->locVecC);
  CHKERRQ(ierr);
  ierr = VecZeroEntries(user->locVecF); CHKERRQ(ierr);

  // Setup CEED vectors
  ierr = VecGetArrayRead(user->locVecC, (const PetscScalar **)&c);
  CHKERRQ(ierr);
  ierr = VecGetArray(user->locVecF, &f); CHKERRQ(ierr);
  CeedVectorSetArray(user->ceedVecC, CEED_MEM_HOST, CEED_USE_POINTER, c);
  CeedVectorSetArray(user->ceedVecF, CEED_MEM_HOST, CEED_USE_POINTER, f);

  // Apply CEED operator
  CeedOperatorApply(user->opProlong, user->ceedVecC, user->ceedVecF,
                    CEED_REQUEST_IMMEDIATE);
  CeedVectorSyncArray(user->ceedVecF, CEED_MEM_HOST);

  // Restore PETSc vectors
  ierr = VecRestoreArrayRead(user->locVecC, (const PetscScalar **)c);
  CHKERRQ(ierr);
  ierr = VecRestoreArray(user->locVecF, &f); CHKERRQ(ierr);

  // Multiplicity
  ierr = VecPointwiseMult(user->locVecF, user->locVecF, user->multVec);

  // Local-to-global
  ierr = VecZeroEntries(Y); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dmF, user->locVecF, ADD_VALUES, Y);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// This function uses libCEED to compute the action of the restriction operator
PetscErrorCode Restrict_Ceed(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  UserMultProlongRestr user;
  PetscScalar *c, *f;

  PetscFunctionBeginUser;

  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);

  // Global-to-local
  ierr = VecZeroEntries(user->locVecF); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dmF, X, INSERT_VALUES, user->locVecF);
  CHKERRQ(ierr);
  ierr = VecZeroEntries(user->locVecC); CHKERRQ(ierr);

  // Multiplicity
  ierr = VecPointwiseMult(user->locVecF, user->locVecF, user->multVec);
  CHKERRQ(ierr);

  // Setup CEED vectors
  ierr = VecGetArrayRead(user->locVecF, (const PetscScalar **)&f); CHKERRQ(ierr);
  ierr = VecGetArray(user->locVecC, &c); CHKERRQ(ierr);
  CeedVectorSetArray(user->ceedVecF, CEED_MEM_HOST, CEED_USE_POINTER, f);
  CeedVectorSetArray(user->ceedVecC, CEED_MEM_HOST, CEED_USE_POINTER, c);

  // Apply CEED operator
  CeedOperatorApply(user->opRestrict, user->ceedVecF, user->ceedVecC,
                    CEED_REQUEST_IMMEDIATE);
  CeedVectorSyncArray(user->ceedVecC, CEED_MEM_HOST);

  // Restore PETSc vectors
  ierr = VecRestoreArrayRead(user->locVecF, (const PetscScalar **)&f);
  CHKERRQ(ierr);
  ierr = VecRestoreArray(user->locVecC, &c); CHKERRQ(ierr);

  // Local-to-global
  ierr = VecZeroEntries(Y); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dmC, user->locVecC, ADD_VALUES, Y);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
};
// This function returns the computed diagonal of the operator
PetscErrorCode GetDiag_Ceed(Mat A, Vec D) {
  PetscErrorCode ierr;
  UserMult user;

  PetscFunctionBeginUser;

  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);

  // Compute Diagonal via libCEED
  CeedVector ceedDiagVec;
  const CeedScalar *diagArray;

  // -- Compute Diagonal
  CeedOperatorAssembleLinearDiagonal(user->op, &ceedDiagVec,
                                     CEED_REQUEST_IMMEDIATE);

  // -- Place in PETSc vector
  CeedVectorGetArrayRead(ceedDiagVec, CEED_MEM_HOST, &diagArray);
  ierr = VecPlaceArray(user->Xloc, diagArray); CHKERRQ(ierr);

  // -- Local-to-Global
  ierr = VecZeroEntries(D); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, user->Xloc, ADD_VALUES, D); CHKERRQ(ierr);

  // -- Cleanup
  ierr = VecResetArray(user->Xloc); CHKERRQ(ierr);
  CeedVectorRestoreArrayRead(ceedDiagVec, &diagArray);
  CeedVectorDestroy(&ceedDiagVec);

  PetscFunctionReturn(0);
};
