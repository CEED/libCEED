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
  PetscMemType xmemtype, ymemtype;

  PetscFunctionBeginUser;

  // Global-to-local
  ierr = DMGlobalToLocal(user->dm, X, INSERT_VALUES, user->Xloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(user->Yloc); CHKERRQ(ierr);

  // Setup CEED vectors
  ierr = VecGetArrayReadAndMemType(user->Xloc, (const PetscScalar **)&x,
                                   &xmemtype); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(user->Yloc, &y, &ymemtype); CHKERRQ(ierr);
  CeedVectorSetArray(user->Xceed, MemTypeP2C(xmemtype), CEED_USE_POINTER, x);
  CeedVectorSetArray(user->Yceed, MemTypeP2C(ymemtype), CEED_USE_POINTER, y);

  // Apply CEED operator
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

  // Neumann BCs
  if (user->NBCs) {
    ierr = VecAXPY(Y, -user->loadIncrement, user->NBCs); CHKERRQ(ierr);
  }

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
  PetscMemType cmemtype, fmemtype;

  PetscFunctionBeginUser;

  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);

  // Global-to-local
  ierr = VecZeroEntries(user->locVecC); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dmC, X, INSERT_VALUES, user->locVecC);
  CHKERRQ(ierr);
  ierr = VecZeroEntries(user->locVecF); CHKERRQ(ierr);

  // Setup CEED vectors
  ierr = VecGetArrayReadAndMemType(user->locVecC, (const PetscScalar **)&c,
                                   &cmemtype); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(user->locVecF, &f, &fmemtype); CHKERRQ(ierr);
  CeedVectorSetArray(user->ceedVecC, MemTypeP2C(cmemtype), CEED_USE_POINTER, c);
  CeedVectorSetArray(user->ceedVecF, MemTypeP2C(fmemtype), CEED_USE_POINTER, f);

  // Apply CEED operator
  CeedOperatorApply(user->opProlong, user->ceedVecC, user->ceedVecF,
                    CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(user->ceedVecC, MemTypeP2C(cmemtype), NULL);
  CeedVectorTakeArray(user->ceedVecF, MemTypeP2C(fmemtype), NULL);
  ierr = VecRestoreArrayReadAndMemType(user->locVecC, (const PetscScalar **)&c);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(user->locVecF, &f); CHKERRQ(ierr);

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
  PetscMemType cmemtype, fmemtype;

  PetscFunctionBeginUser;

  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);

  // Global-to-local
  ierr = VecZeroEntries(user->locVecF); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dmF, X, INSERT_VALUES, user->locVecF);
  CHKERRQ(ierr);
  ierr = VecZeroEntries(user->locVecC); CHKERRQ(ierr);

  // Setup CEED vectors
  ierr = VecGetArrayReadAndMemType(user->locVecF, (const PetscScalar **)&f,
                                   &fmemtype); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(user->locVecC, &c, &cmemtype); CHKERRQ(ierr);
  CeedVectorSetArray(user->ceedVecF, MemTypeP2C(fmemtype), CEED_USE_POINTER, f);
  CeedVectorSetArray(user->ceedVecC, MemTypeP2C(cmemtype), CEED_USE_POINTER, c);

  // Apply CEED operator
  CeedOperatorApply(user->opRestrict, user->ceedVecF, user->ceedVecC,
                    CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(user->ceedVecF, MemTypeP2C(fmemtype), NULL);
  CeedVectorTakeArray(user->ceedVecC, MemTypeP2C(cmemtype), NULL);
  ierr = VecRestoreArrayReadAndMemType(user->locVecF, (const PetscScalar **)&f);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(user->locVecC, &c); CHKERRQ(ierr);

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
  if (user->ctxPhysSmoother)
    CeedQFunctionSetContext(user->qf, user->ctxPhysSmoother);

  // Compute Diagonal via libCEED
  PetscScalar *x;
  PetscMemType xmemtype;

  // -- Place PETSc vector in libCEED vector
  ierr = VecGetArrayAndMemType(user->Xloc, &x, &xmemtype); CHKERRQ(ierr);
  CeedVectorSetArray(user->Xceed, MemTypeP2C(xmemtype), CEED_USE_POINTER, x);

  // -- Compute Diagonal
  CeedOperatorLinearAssembleDiagonal(user->op, user->Xceed,
                                     CEED_REQUEST_IMMEDIATE);

  // -- Reset physics context
  if (user->ctxPhysSmoother)
    CeedQFunctionSetContext(user->qf, user->ctxPhys);

  // -- Local-to-Global
  CeedVectorTakeArray(user->Xceed, MemTypeP2C(xmemtype), NULL);
  ierr = VecRestoreArrayAndMemType(user->Xloc, &x); CHKERRQ(ierr);
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
  PetscMemType xmemtype;
  CeedInt length;

  PetscFunctionBeginUser;

  // Global-to-local
  ierr = VecZeroEntries(user->Xloc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm, X, INSERT_VALUES, user->Xloc); CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(user->dm, PETSC_TRUE, user->Xloc,
                                    user->loadIncrement, NULL, NULL, NULL);
  CHKERRQ(ierr);

  // Setup libCEED input vector
  ierr = VecGetArrayReadAndMemType(user->Xloc, (const PetscScalar **)&x,
                                   &xmemtype); CHKERRQ(ierr);
  CeedVectorSetArray(user->Xceed, MemTypeP2C(xmemtype), CEED_USE_POINTER, x);

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
  CeedVectorTakeArray(user->Xceed, MemTypeP2C(xmemtype), NULL);
  ierr = VecRestoreArrayRead(user->Xloc, (const PetscScalar **)&x);
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
