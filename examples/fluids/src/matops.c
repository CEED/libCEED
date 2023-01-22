// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../include/matops.h"

#include "../navierstokes.h"

// -----------------------------------------------------------------------------
// Setup apply operator context data
// -----------------------------------------------------------------------------
PetscErrorCode MatopApplyContextCreate(DM dm, Ceed ceed, CeedOperator op_apply, CeedVector x_ceed, CeedVector y_ceed, Vec X_loc,
                                       MatopApplyContext *op_apply_ctx) {
  PetscFunctionBeginUser;

  PetscCall(PetscNew(op_apply_ctx));

  // Copy PETSc Objects
  PetscCall(PetscObjectReference((PetscObject)dm));
  (*op_apply_ctx)->dm = dm;
  if (X_loc) {
    PetscCall(PetscObjectReference((PetscObject)X_loc));
    (*op_apply_ctx)->X_loc = X_loc;
    PetscCall(VecDuplicate(X_loc, &(*op_apply_ctx)->Y_loc));
  } else {
    (*op_apply_ctx)->X_loc = (*op_apply_ctx)->Y_loc = NULL;
  }

  // Copy libCEED objects
  CeedVectorReferenceCopy(x_ceed, &(*op_apply_ctx)->x_ceed);
  CeedVectorReferenceCopy(y_ceed, &(*op_apply_ctx)->y_ceed);
  CeedOperatorReferenceCopy(op_apply, &(*op_apply_ctx)->op);
  CeedReferenceCopy(ceed, &(*op_apply_ctx)->ceed);

  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// Destroy apply operator context data
// -----------------------------------------------------------------------------
PetscErrorCode MatopApplyContextDestroy(MatopApplyContext op_apply_ctx) {
  PetscFunctionBeginUser;

  if (!op_apply_ctx) PetscFunctionReturn(0);

  // Destroy PETSc Objects
  PetscCall(DMDestroy(&op_apply_ctx->dm));
  PetscCall(VecDestroy(&op_apply_ctx->X_loc));
  PetscCall(VecDestroy(&op_apply_ctx->Y_loc));

  // Destroy libCEED Objects
  CeedVectorDestroy(&op_apply_ctx->x_ceed);
  CeedVectorDestroy(&op_apply_ctx->y_ceed);
  CeedOperatorDestroy(&op_apply_ctx->op);
  CeedDestroy(&op_apply_ctx->ceed);

  PetscCall(PetscFree(op_apply_ctx));

  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// This function returns the computed diagonal of the operator
// -----------------------------------------------------------------------------
PetscErrorCode MatGetDiag_Ceed(Mat A, Vec D) {
  MatopApplyContext op_apply_ctx;
  Vec               Y_loc;
  PetscFunctionBeginUser;

  PetscCall(MatShellGetContext(A, &op_apply_ctx));
  if (op_apply_ctx->Y_loc) Y_loc = op_apply_ctx->Y_loc;
  else PetscCall(DMGetLocalVector(op_apply_ctx->dm, &Y_loc));

  // Compute Diagonal via libCEED
  PetscScalar *y;
  PetscMemType mem_type;

  // -- Place PETSc vector in libCEED vector
  PetscCall(VecGetArrayAndMemType(Y_loc, &y, &mem_type));
  CeedVectorSetArray(op_apply_ctx->y_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, y);

  // -- Compute Diagonal
  CeedOperatorLinearAssembleDiagonal(op_apply_ctx->op, op_apply_ctx->y_ceed, CEED_REQUEST_IMMEDIATE);

  // -- Local-to-Global
  CeedVectorTakeArray(op_apply_ctx->y_ceed, MemTypeP2C(mem_type), NULL);
  PetscCall(VecRestoreArrayAndMemType(Y_loc, &y));
  PetscCall(VecZeroEntries(D));
  PetscCall(DMLocalToGlobal(op_apply_ctx->dm, Y_loc, ADD_VALUES, D));

  if (!op_apply_ctx->Y_loc) PetscCall(DMRestoreLocalVector(op_apply_ctx->dm, &Y_loc));
  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the Laplacian with Dirichlet boundary conditions
// -----------------------------------------------------------------------------
PetscErrorCode ApplyLocal_Ceed(Vec X, Vec Y, MatopApplyContext op_apply_ctx) {
  PetscScalar *x, *y;
  PetscMemType x_mem_type, y_mem_type;
  Vec          Y_loc, X_loc;
  PetscFunctionBeginUser;

  if (op_apply_ctx->Y_loc) Y_loc = op_apply_ctx->Y_loc;
  else PetscCall(DMGetLocalVector(op_apply_ctx->dm, &Y_loc));
  if (op_apply_ctx->X_loc) X_loc = op_apply_ctx->X_loc;
  else PetscCall(DMGetLocalVector(op_apply_ctx->dm, &X_loc));

  // Global-to-local
  PetscCall(DMGlobalToLocal(op_apply_ctx->dm, X, INSERT_VALUES, X_loc));

  // Setup libCEED vectors
  PetscCall(VecGetArrayReadAndMemType(X_loc, (const PetscScalar **)&x, &x_mem_type));
  PetscCall(VecGetArrayAndMemType(Y_loc, &y, &y_mem_type));
  CeedVectorSetArray(op_apply_ctx->x_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER, x);
  CeedVectorSetArray(op_apply_ctx->y_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER, y);

  // Apply libCEED operator
  CeedOperatorApply(op_apply_ctx->op, op_apply_ctx->x_ceed, op_apply_ctx->y_ceed, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(op_apply_ctx->x_ceed, MemTypeP2C(x_mem_type), NULL);
  CeedVectorTakeArray(op_apply_ctx->y_ceed, MemTypeP2C(y_mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(X_loc, (const PetscScalar **)&x));
  PetscCall(VecRestoreArrayAndMemType(Y_loc, &y));

  // Local-to-global
  PetscCall(VecZeroEntries(Y));
  PetscCall(DMLocalToGlobal(op_apply_ctx->dm, Y_loc, ADD_VALUES, Y));

  if (!op_apply_ctx->Y_loc) PetscCall(DMRestoreLocalVector(op_apply_ctx->dm, &Y_loc));
  if (!op_apply_ctx->X_loc) PetscCall(DMRestoreLocalVector(op_apply_ctx->dm, &X_loc));
  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function wraps the libCEED operator for a MatShell
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_Ceed(Mat A, Vec X, Vec Y) {
  MatopApplyContext op_apply_ctx;
  PetscFunctionBeginUser;

  PetscCall(MatShellGetContext(A, &op_apply_ctx));

  // libCEED for local action of residual evaluator
  PetscCall(ApplyLocal_Ceed(X, Y, op_apply_ctx));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
