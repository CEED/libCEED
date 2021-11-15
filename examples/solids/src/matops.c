// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Matrix shell operations for solid mechanics example using PETSc

#include "../include/matops.h"
#include "../include/structs.h"
#include "../include/utils.h"

// -----------------------------------------------------------------------------
// libCEED Operators for MatShell
// -----------------------------------------------------------------------------
// This function uses libCEED to compute the local action of an operator
PetscErrorCode ApplyLocalCeedOp(Vec X, Vec Y, UserMult user) {
  PetscErrorCode ierr;
  PetscScalar *x, *y;
  PetscMemType x_mem_type, y_mem_type;

  PetscFunctionBeginUser;

  // Global-to-local
  ierr = DMGlobalToLocal(user->dm, X, INSERT_VALUES, user->X_loc); CHKERRQ(ierr);
  ierr = VecZeroEntries(user->Y_loc); CHKERRQ(ierr);

  // Setup CEED vectors
  ierr = VecGetArrayReadAndMemType(user->X_loc, (const PetscScalar **)&x,
                                   &x_mem_type); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(user->Y_loc, &y, &y_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(user->x_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER, x);
  CeedVectorSetArray(user->y_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER, y);

  // Apply CEED operator
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

// This function uses libCEED to compute the non-linear residual
PetscErrorCode FormResidual_Ceed(SNES snes, Vec X, Vec Y, void *ctx) {
  PetscErrorCode ierr;
  UserMult user = (UserMult)ctx;

  PetscFunctionBeginUser;

  // Use computed BCs
  ierr = VecZeroEntries(user->X_loc); CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(user->dm, PETSC_TRUE, user->X_loc,
                                    user->load_increment, NULL, NULL, NULL);
  CHKERRQ(ierr);

  // libCEED for local action of residual evaluator
  ierr = ApplyLocalCeedOp(X, Y, user); CHKERRQ(ierr);

  // Neumann BCs
  if (user->neumann_bcs) {
    ierr = VecAXPY(Y, -user->load_increment, user->neumann_bcs); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
};

// This function uses libCEED to apply the Jacobian for assembly via a SNES
PetscErrorCode ApplyJacobianCoarse_Ceed(SNES snes, Vec X, Vec Y, void *ctx) {
  PetscErrorCode ierr;
  UserMult user = (UserMult)ctx;

  PetscFunctionBeginUser;

  // Zero boundary values
  ierr = VecZeroEntries(user->X_loc); CHKERRQ(ierr);

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
  ierr = VecZeroEntries(user->X_loc); CHKERRQ(ierr);

  // libCEED for local action of Jacobian
  ierr = ApplyLocalCeedOp(X, Y, user); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// This function uses libCEED to compute the action of the prolongation operator
PetscErrorCode Prolong_Ceed(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  UserMultProlongRestr user;
  PetscScalar *c, *f;
  PetscMemType c_mem_type, f_mem_type;

  PetscFunctionBeginUser;

  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);

  // Global-to-local
  ierr = VecZeroEntries(user->loc_vec_c); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm_c, X, INSERT_VALUES, user->loc_vec_c);
  CHKERRQ(ierr);
  ierr = VecZeroEntries(user->loc_vec_f); CHKERRQ(ierr);

  // Setup CEED vectors
  ierr = VecGetArrayReadAndMemType(user->loc_vec_c, (const PetscScalar **)&c,
                                   &c_mem_type); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(user->loc_vec_f, &f, &f_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(user->ceed_vec_c, MemTypeP2C(c_mem_type), CEED_USE_POINTER,
                     c);
  CeedVectorSetArray(user->ceed_vec_f, MemTypeP2C(f_mem_type), CEED_USE_POINTER,
                     f);

  // Apply CEED operator
  CeedOperatorApply(user->op_prolong, user->ceed_vec_c, user->ceed_vec_f,
                    CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(user->ceed_vec_c, MemTypeP2C(c_mem_type), NULL);
  CeedVectorTakeArray(user->ceed_vec_f, MemTypeP2C(f_mem_type), NULL);
  ierr = VecRestoreArrayReadAndMemType(user->loc_vec_c, (const PetscScalar **)&c);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(user->loc_vec_f, &f); CHKERRQ(ierr);

  // Local-to-global
  ierr = VecZeroEntries(Y); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm_f, user->loc_vec_f, ADD_VALUES, Y);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// This function uses libCEED to compute the action of the restriction operator
PetscErrorCode Restrict_Ceed(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  UserMultProlongRestr user;
  PetscScalar *c, *f;
  PetscMemType c_mem_type, f_mem_type;

  PetscFunctionBeginUser;

  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);

  // Global-to-local
  ierr = VecZeroEntries(user->loc_vec_f); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm_f, X, INSERT_VALUES, user->loc_vec_f);
  CHKERRQ(ierr);
  ierr = VecZeroEntries(user->loc_vec_c); CHKERRQ(ierr);

  // Setup CEED vectors
  ierr = VecGetArrayReadAndMemType(user->loc_vec_f, (const PetscScalar **)&f,
                                   &f_mem_type); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(user->loc_vec_c, &c, &c_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(user->ceed_vec_f, MemTypeP2C(f_mem_type), CEED_USE_POINTER,
                     f);
  CeedVectorSetArray(user->ceed_vec_c, MemTypeP2C(c_mem_type), CEED_USE_POINTER,
                     c);

  // Apply CEED operator
  CeedOperatorApply(user->op_restrict, user->ceed_vec_f, user->ceed_vec_c,
                    CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(user->ceed_vec_f, MemTypeP2C(f_mem_type), NULL);
  CeedVectorTakeArray(user->ceed_vec_c, MemTypeP2C(c_mem_type), NULL);
  ierr = VecRestoreArrayReadAndMemType(user->loc_vec_f, (const PetscScalar **)&f);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(user->loc_vec_c, &c); CHKERRQ(ierr);

  // Local-to-global
  ierr = VecZeroEntries(Y); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm_c, user->loc_vec_c, ADD_VALUES, Y);
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
  if (user->ctx_phys_smoother)
    CeedQFunctionSetContext(user->qf, user->ctx_phys_smoother);

  // Compute Diagonal via libCEED
  PetscScalar *x;
  PetscMemType x_mem_type;

  // -- Place PETSc vector in libCEED vector
  ierr = VecGetArrayAndMemType(user->X_loc, &x, &x_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(user->x_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER, x);

  // -- Compute Diagonal
  CeedOperatorLinearAssembleDiagonal(user->op, user->x_ceed,
                                     CEED_REQUEST_IMMEDIATE);

  // -- Reset physics context
  if (user->ctx_phys_smoother)
    CeedQFunctionSetContext(user->qf, user->ctx_phys);

  // -- Local-to-Global
  CeedVectorTakeArray(user->x_ceed, MemTypeP2C(x_mem_type), NULL);
  ierr = VecRestoreArrayAndMemType(user->X_loc, &x); CHKERRQ(ierr);
  ierr = VecZeroEntries(D); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, user->X_loc, ADD_VALUES, D); CHKERRQ(ierr);

  // Cleanup
  ierr = VecZeroEntries(user->X_loc); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// This function calculates the strain energy in the final solution
PetscErrorCode ComputeStrainEnergy(DM dm_energy, UserMult user,
                                   CeedOperator op_energy, Vec X,
                                   PetscReal *energy) {
  PetscErrorCode ierr;
  PetscScalar *x;
  PetscMemType x_mem_type;
  CeedInt length;

  PetscFunctionBeginUser;

  // Global-to-local
  ierr = VecZeroEntries(user->X_loc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm, X, INSERT_VALUES, user->X_loc); CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(user->dm, PETSC_TRUE, user->X_loc,
                                    user->load_increment, NULL, NULL, NULL);
  CHKERRQ(ierr);

  // Setup libCEED input vector
  ierr = VecGetArrayReadAndMemType(user->X_loc, (const PetscScalar **)&x,
                                   &x_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(user->x_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER, x);

  // Setup libCEED output vector
  Vec E_loc;
  CeedVector e_loc;
  ierr = DMCreateLocalVector(dm_energy, &E_loc); CHKERRQ(ierr);
  ierr = VecGetSize(E_loc, &length); CHKERRQ(ierr);
  ierr = VecDestroy(&E_loc); CHKERRQ(ierr);
  CeedVectorCreate(user->ceed, length, &e_loc);

  // Apply libCEED operator
  CeedOperatorApply(op_energy, user->x_ceed, e_loc, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vector
  CeedVectorTakeArray(user->x_ceed, MemTypeP2C(x_mem_type), NULL);
  ierr = VecRestoreArrayRead(user->X_loc, (const PetscScalar **)&x);
  CHKERRQ(ierr);

  // Reduce max error
  const CeedScalar *e;
  CeedVectorGetArrayRead(e_loc, CEED_MEM_HOST, &e);
  (*energy) = 0;
  for (CeedInt i=0; i<length; i++)
    (*energy) += e[i];
  CeedVectorRestoreArrayRead(e_loc, &e);
  CeedVectorDestroy(&e_loc);

  ierr = MPI_Allreduce(MPI_IN_PLACE, energy, 1, MPIU_REAL, MPIU_SUM,
                       user->comm); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// This function frees memory allocated for Enzyme-AD
PetscErrorCode FreeTapeMemory(DM dm_tape, UserMult user, CeedOperator op_tape,
                              Vec X) {
  PetscErrorCode ierr;
  PetscScalar *x;
  PetscMemType x_mem_type;
  CeedInt length;

  PetscFunctionBeginUser;

  // Global-to-local
  ierr = VecZeroEntries(user->X_loc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm, X, INSERT_VALUES, user->X_loc); CHKERRQ(ierr);

  // Setup libCEED input vector
  ierr = VecGetArrayReadAndMemType(user->X_loc, (const PetscScalar **)&x,
                                   &x_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(user->x_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER, x);

  // Setup libCEED output vector
  Vec E_loc;
  CeedVector e_loc;
  ierr = DMCreateLocalVector(dm_tape, &E_loc); CHKERRQ(ierr);
  ierr = VecGetSize(E_loc, &length); CHKERRQ(ierr);
  ierr = VecDestroy(&E_loc); CHKERRQ(ierr);
  CeedVectorCreate(user->ceed, length, &e_loc);

  // Apply libCEED operator
  CeedOperatorApply(op_tape, user->x_ceed, e_loc, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vector
  CeedVectorTakeArray(user->x_ceed, MemTypeP2C(x_mem_type), NULL);
  ierr = VecRestoreArrayRead(user->X_loc, (const PetscScalar **)&x);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
};
