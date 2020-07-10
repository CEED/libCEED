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

/// @file
/// Matrix shell operations for solid mechanics example using PETSc

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
  ierr = user->VecGetArrayRead(user->Xloc, (const PetscScalar **)&x);
  CHKERRQ(ierr);
  ierr = user->VecGetArray(user->Yloc, &y); CHKERRQ(ierr);
  CeedVectorSetArray(user->Xceed, user->memType, CEED_USE_POINTER, x);
  CeedVectorSetArray(user->Yceed, user->memType, CEED_USE_POINTER, y);

  // Apply CEED operator
  // Note: We could use VecGetArrayInPlace. Instead, we use SetArray/TakeArray
  //         so we can request host memory for easier debugging.
  CeedOperatorApply(user->op, user->Xceed, user->Yceed, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(user->Xceed, user->memType, NULL);
  CeedVectorTakeArray(user->Yceed, user->memType, NULL);
  ierr = user->VecRestoreArrayRead(user->Xloc, (const PetscScalar **)&x);
  CHKERRQ(ierr);
  ierr = user->VecRestoreArray(user->Yloc, &y); CHKERRQ(ierr);

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

  // Zero boundary values
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
  ierr = user->VecGetArrayRead(user->locVecC, (const PetscScalar **)&c);
  CHKERRQ(ierr);
  ierr = user->VecGetArray(user->locVecF, &f); CHKERRQ(ierr);
  CeedVectorSetArray(user->ceedVecC, user->memType, CEED_USE_POINTER, c);
  CeedVectorSetArray(user->ceedVecF, user->memType, CEED_USE_POINTER, f);

  // Apply CEED operator
  CeedOperatorApply(user->opProlong, user->ceedVecC, user->ceedVecF,
                    CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(user->ceedVecC, user->memType, NULL);
  CeedVectorTakeArray(user->ceedVecF, user->memType, NULL);
  ierr = user->VecRestoreArrayRead(user->locVecC, (const PetscScalar **)&c);
  CHKERRQ(ierr);
  ierr = user->VecRestoreArray(user->locVecF, &f); CHKERRQ(ierr);

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

  // Setup CEED vectors
  ierr = user->VecGetArrayRead(user->locVecF, (const PetscScalar **)&f);
  CHKERRQ(ierr);
  ierr = user->VecGetArray(user->locVecC, &c); CHKERRQ(ierr);
  CeedVectorSetArray(user->ceedVecF, user->memType, CEED_USE_POINTER, f);
  CeedVectorSetArray(user->ceedVecC, user->memType, CEED_USE_POINTER, c);

  // Apply CEED operator
  CeedOperatorApply(user->opRestrict, user->ceedVecF, user->ceedVecC,
                    CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(user->ceedVecF, user->memType, NULL);
  CeedVectorTakeArray(user->ceedVecC, user->memType, NULL);
  ierr = user->VecRestoreArrayRead(user->locVecF, (const PetscScalar **)&f);
  CHKERRQ(ierr);
  ierr = user->VecRestoreArray(user->locVecC, &c); CHKERRQ(ierr);

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

  // -- Set physics context
  if (user->physSmoother)
    CeedQFunctionSetContext(user->qf, user->physSmoother,
                            sizeof(*user->physSmoother));

  // Compute Diagonal via libCEED
  PetscScalar *x;

  // -- Place PETSc vector in libCEED vector
  ierr = user->VecGetArray(user->Xloc, &x); CHKERRQ(ierr);
  CeedVectorSetArray(user->Xceed, user->memType, CEED_USE_POINTER, x);

  // -- Compute Diagonal
  CeedOperatorLinearAssembleDiagonal(user->op, user->Xceed,
                                     CEED_REQUEST_IMMEDIATE);

  // -- Reset physics context
  if (user->physSmoother)
    CeedQFunctionSetContext(user->qf, user->phys, sizeof(*user->phys));

  // -- Local-to-Global
  CeedVectorTakeArray(user->Xceed, user->memType, NULL);
  ierr = user->VecRestoreArray(user->Xloc, &x); CHKERRQ(ierr);
  ierr = VecZeroEntries(D); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, user->Xloc, ADD_VALUES, D); CHKERRQ(ierr);

  // Cleanup
  ierr = VecZeroEntries(user->Xloc); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// This function calculates the strain energy in the final solution
PetscErrorCode ComputeStrainEnergy(DM dmEnergy, UserMult user,
                                   CeedOperator opEnergy, Vec X,
                                   PetscReal *energy) {
  PetscErrorCode ierr;
  PetscScalar *x;
  CeedInt length;

  PetscFunctionBeginUser;

  // Global-to-local
  ierr = VecZeroEntries(user->Xloc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm, X, INSERT_VALUES, user->Xloc); CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(user->dm, PETSC_TRUE, user->Xloc,
                                    user->loadIncrement, NULL, NULL, NULL);
  CHKERRQ(ierr);

  // Setup libCEED input vector
  ierr = user->VecGetArrayRead(user->Xloc, (const PetscScalar **)&x);
  CHKERRQ(ierr);
  CeedVectorSetArray(user->Xceed, user->memType, CEED_USE_POINTER, x);

  // Setup libCEED output vector
  Vec Eloc;
  CeedVector eloc;
  ierr = DMCreateLocalVector(dmEnergy, &Eloc); CHKERRQ(ierr);
  ierr = VecGetSize(Eloc, &length); CHKERRQ(ierr);
  ierr = VecDestroy(&Eloc); CHKERRQ(ierr);
  CeedVectorCreate(user->ceed, length, &eloc);

  // Apply libCEED operator
  CeedOperatorApply(opEnergy, user->Xceed, eloc, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vector
  ierr = user->VecRestoreArrayRead(user->Xloc, (const PetscScalar **)&x);
  CHKERRQ(ierr);

  // Reduce max error
  const CeedScalar *e;
  CeedVectorGetArrayRead(eloc, CEED_MEM_HOST, &e);
  (*energy) = 0;
  for (CeedInt i=0; i<length; i++)
    (*energy) += e[i];
  CeedVectorRestoreArrayRead(eloc, &e);
  CeedVectorDestroy(&eloc);

  ierr = MPI_Allreduce(MPI_IN_PLACE, energy, 1, MPIU_REAL, MPIU_SUM,
                       user->comm); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};
