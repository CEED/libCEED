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
  PetscScalar *x, *y;
  PetscMemType x_mem_type, y_mem_type;

  PetscFunctionBeginUser;

  // Global-to-local
  PetscCall(DMGlobalToLocal(user->dm, X, INSERT_VALUES, user->X_loc));
  PetscCall(VecZeroEntries(user->Y_loc));

  // Setup CEED vectors
  PetscCall(VecGetArrayReadAndMemType(user->X_loc, (const PetscScalar **)&x, &x_mem_type));
  PetscCall(VecGetArrayAndMemType(user->Y_loc, &y, &y_mem_type));
  CeedVectorSetArray(user->x_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER, x);
  CeedVectorSetArray(user->y_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER, y);

  // Apply CEED operator
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

// This function uses libCEED to compute the non-linear residual
PetscErrorCode FormResidual_Ceed(SNES snes, Vec X, Vec Y, void *ctx) {
  UserMult user = (UserMult)ctx;

  PetscFunctionBeginUser;

  // Use computed BCs
  PetscCall(VecZeroEntries(user->X_loc));
  PetscCall(DMPlexInsertBoundaryValues(user->dm, PETSC_TRUE, user->X_loc, user->load_increment, NULL, NULL, NULL));

  // libCEED for local action of residual evaluator
  PetscCall(ApplyLocalCeedOp(X, Y, user));

  // Neumann BCs
  if (user->neumann_bcs) {
    PetscCall(VecAXPY(Y, -user->load_increment, user->neumann_bcs));
  }

  PetscFunctionReturn(0);
};

// This function uses libCEED to apply the Jacobian for assembly via a SNES
PetscErrorCode ApplyJacobianCoarse_Ceed(SNES snes, Vec X, Vec Y, void *ctx) {
  UserMult user = (UserMult)ctx;

  PetscFunctionBeginUser;

  // Zero boundary values
  PetscCall(VecZeroEntries(user->X_loc));

  // libCEED for local action of residual evaluator
  PetscCall(ApplyLocalCeedOp(X, Y, user));

  PetscFunctionReturn(0);
};

// This function uses libCEED to compute the action of the Jacobian
PetscErrorCode ApplyJacobian_Ceed(Mat A, Vec X, Vec Y) {
  UserMult user;

  PetscFunctionBeginUser;

  // Zero boundary values
  PetscCall(MatShellGetContext(A, &user));
  PetscCall(VecZeroEntries(user->X_loc));

  // libCEED for local action of Jacobian
  PetscCall(ApplyLocalCeedOp(X, Y, user));

  PetscFunctionReturn(0);
};

// This function uses libCEED to compute the action of the prolongation operator
PetscErrorCode Prolong_Ceed(Mat A, Vec X, Vec Y) {
  UserMultProlongRestr user;
  PetscScalar         *c, *f;
  PetscMemType         c_mem_type, f_mem_type;

  PetscFunctionBeginUser;

  PetscCall(MatShellGetContext(A, &user));

  // Global-to-local
  PetscCall(VecZeroEntries(user->loc_vec_c));
  PetscCall(DMGlobalToLocal(user->dm_c, X, INSERT_VALUES, user->loc_vec_c));
  PetscCall(VecZeroEntries(user->loc_vec_f));

  // Setup CEED vectors
  PetscCall(VecGetArrayReadAndMemType(user->loc_vec_c, (const PetscScalar **)&c, &c_mem_type));
  PetscCall(VecGetArrayAndMemType(user->loc_vec_f, &f, &f_mem_type));
  CeedVectorSetArray(user->ceed_vec_c, MemTypeP2C(c_mem_type), CEED_USE_POINTER, c);
  CeedVectorSetArray(user->ceed_vec_f, MemTypeP2C(f_mem_type), CEED_USE_POINTER, f);

  // Apply CEED operator
  CeedOperatorApply(user->op_prolong, user->ceed_vec_c, user->ceed_vec_f, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(user->ceed_vec_c, MemTypeP2C(c_mem_type), NULL);
  CeedVectorTakeArray(user->ceed_vec_f, MemTypeP2C(f_mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(user->loc_vec_c, (const PetscScalar **)&c));
  PetscCall(VecRestoreArrayAndMemType(user->loc_vec_f, &f));

  // Local-to-global
  PetscCall(VecZeroEntries(Y));
  PetscCall(DMLocalToGlobal(user->dm_f, user->loc_vec_f, ADD_VALUES, Y));

  PetscFunctionReturn(0);
}

// This function uses libCEED to compute the action of the restriction operator
PetscErrorCode Restrict_Ceed(Mat A, Vec X, Vec Y) {
  UserMultProlongRestr user;
  PetscScalar         *c, *f;
  PetscMemType         c_mem_type, f_mem_type;

  PetscFunctionBeginUser;

  PetscCall(MatShellGetContext(A, &user));

  // Global-to-local
  PetscCall(VecZeroEntries(user->loc_vec_f));
  PetscCall(DMGlobalToLocal(user->dm_f, X, INSERT_VALUES, user->loc_vec_f));
  PetscCall(VecZeroEntries(user->loc_vec_c));

  // Setup CEED vectors
  PetscCall(VecGetArrayReadAndMemType(user->loc_vec_f, (const PetscScalar **)&f, &f_mem_type));
  PetscCall(VecGetArrayAndMemType(user->loc_vec_c, &c, &c_mem_type));
  CeedVectorSetArray(user->ceed_vec_f, MemTypeP2C(f_mem_type), CEED_USE_POINTER, f);
  CeedVectorSetArray(user->ceed_vec_c, MemTypeP2C(c_mem_type), CEED_USE_POINTER, c);

  // Apply CEED operator
  CeedOperatorApply(user->op_restrict, user->ceed_vec_f, user->ceed_vec_c, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(user->ceed_vec_f, MemTypeP2C(f_mem_type), NULL);
  CeedVectorTakeArray(user->ceed_vec_c, MemTypeP2C(c_mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(user->loc_vec_f, (const PetscScalar **)&f));
  PetscCall(VecRestoreArrayAndMemType(user->loc_vec_c, &c));

  // Local-to-global
  PetscCall(VecZeroEntries(Y));
  PetscCall(DMLocalToGlobal(user->dm_c, user->loc_vec_c, ADD_VALUES, Y));

  PetscFunctionReturn(0);
};

// This function returns the computed diagonal of the operator
PetscErrorCode GetDiag_Ceed(Mat A, Vec D) {
  UserMult user;

  PetscFunctionBeginUser;

  PetscCall(MatShellGetContext(A, &user));

  // -- Set physics context
  if (user->ctx_phys_smoother) CeedQFunctionSetContext(user->qf, user->ctx_phys_smoother);

  // Compute Diagonal via libCEED
  PetscScalar *x;
  PetscMemType x_mem_type;

  // -- Place PETSc vector in libCEED vector
  PetscCall(VecGetArrayAndMemType(user->X_loc, &x, &x_mem_type));
  CeedVectorSetArray(user->x_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER, x);

  // -- Compute Diagonal
  CeedOperatorLinearAssembleDiagonal(user->op, user->x_ceed, CEED_REQUEST_IMMEDIATE);

  // -- Reset physics context
  if (user->ctx_phys_smoother) CeedQFunctionSetContext(user->qf, user->ctx_phys);

  // -- Local-to-Global
  CeedVectorTakeArray(user->x_ceed, MemTypeP2C(x_mem_type), NULL);
  PetscCall(VecRestoreArrayAndMemType(user->X_loc, &x));
  PetscCall(VecZeroEntries(D));
  PetscCall(DMLocalToGlobal(user->dm, user->X_loc, ADD_VALUES, D));

  // Cleanup
  PetscCall(VecZeroEntries(user->X_loc));

  PetscFunctionReturn(0);
};

// This function calculates the strain energy in the final solution
PetscErrorCode ComputeStrainEnergy(DM dmEnergy, UserMult user, CeedOperator op_energy, Vec X, PetscReal *energy) {
  PetscScalar *x;
  PetscMemType x_mem_type;
  CeedInt      length;

  PetscFunctionBeginUser;

  // Global-to-local
  PetscCall(VecZeroEntries(user->X_loc));
  PetscCall(DMGlobalToLocal(user->dm, X, INSERT_VALUES, user->X_loc));
  PetscCall(DMPlexInsertBoundaryValues(user->dm, PETSC_TRUE, user->X_loc, user->load_increment, NULL, NULL, NULL));

  // Setup libCEED input vector
  PetscCall(VecGetArrayReadAndMemType(user->X_loc, (const PetscScalar **)&x, &x_mem_type));
  CeedVectorSetArray(user->x_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER, x);

  // Setup libCEED output vector
  Vec        E_loc;
  CeedVector e_loc;
  PetscCall(DMCreateLocalVector(dmEnergy, &E_loc));
  PetscCall(VecGetSize(E_loc, &length));
  PetscCall(VecDestroy(&E_loc));
  CeedVectorCreate(user->ceed, length, &e_loc);

  // Apply libCEED operator
  CeedOperatorApply(op_energy, user->x_ceed, e_loc, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vector
  CeedVectorTakeArray(user->x_ceed, MemTypeP2C(x_mem_type), NULL);
  PetscCall(VecRestoreArrayRead(user->X_loc, (const PetscScalar **)&x));

  // Reduce max error
  const CeedScalar *e;
  CeedVectorGetArrayRead(e_loc, CEED_MEM_HOST, &e);
  (*energy) = 0;
  for (CeedInt i = 0; i < length; i++) (*energy) += e[i];
  CeedVectorRestoreArrayRead(e_loc, &e);
  CeedVectorDestroy(&e_loc);

  PetscCall(MPI_Allreduce(MPI_IN_PLACE, energy, 1, MPIU_REAL, MPIU_SUM, user->comm));

  PetscFunctionReturn(0);
};
