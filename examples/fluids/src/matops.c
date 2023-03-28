// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../include/matops.h"

#include <ceed.h>
#include <petscdm.h>

#include "../navierstokes.h"

// -----------------------------------------------------------------------------
// Setup apply operator context data
// -----------------------------------------------------------------------------
PetscErrorCode MatopApplyContextCreate(DM dm_x, DM dm_y, Ceed ceed, CeedOperator op_apply, CeedVector x_ceed, CeedVector y_ceed, Vec X_loc, Vec Y_loc,
                                       MatopApplyContext *op_apply_ctx) {
  PetscFunctionBeginUser;

  PetscCall(PetscNew(op_apply_ctx));

  // Copy PETSc Objects
  PetscCall(PetscObjectReference((PetscObject)dm_x));
  (*op_apply_ctx)->dm_x = dm_x;
  PetscCall(PetscObjectReference((PetscObject)dm_y));
  (*op_apply_ctx)->dm_y = dm_y;

  if (X_loc) PetscCall(PetscObjectReference((PetscObject)X_loc));
  (*op_apply_ctx)->X_loc = X_loc;
  if (Y_loc) PetscCall(PetscObjectReference((PetscObject)Y_loc));
  (*op_apply_ctx)->Y_loc = Y_loc;

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
  PetscCall(DMDestroy(&op_apply_ctx->dm_x));
  PetscCall(DMDestroy(&op_apply_ctx->dm_y));
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

/**
  @brief Transfer array from PETSc Vec to CeedVector

  @param[in]   X_petsc   PETSc Vec
  @param[out]  mem_type  PETSc MemType
  @param[out]  x_ceed    CeedVector

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode VecP2C(Vec X_petsc, PetscMemType *mem_type, CeedVector x_ceed) {
  PetscScalar *x;

  PetscFunctionBeginUser;
  Ceed        ceed;
  CeedMemType cmem_type;
  CeedVectorGetCeed(x_ceed, &ceed);
  CeedGetPreferredMemType(ceed, &cmem_type);

  switch (cmem_type) {
    case CEED_MEM_HOST:
      PetscCall(VecGetArray(X_petsc, &x));
      *mem_type = PETSC_MEMTYPE_HOST;
      break;
    case CEED_MEM_DEVICE:
      PetscCall(VecGetArrayAndMemType(X_petsc, &x, mem_type));
      break;
  }
  CeedVectorSetArray(x_ceed, MemTypeP2C(*mem_type), CEED_USE_POINTER, x);

  PetscFunctionReturn(0);
}

/**
  @brief Transfer array from CeedVector to PETSc Vec

  @param[in]   x_ceed    CeedVector
  @param[in]   mem_type  PETSc MemType
  @param[out]  X_petsc   PETSc Vec

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode VecC2P(CeedVector x_ceed, PetscMemType mem_type, Vec X_petsc) {
  PetscScalar *x;

  PetscFunctionBeginUser;
  Ceed        ceed;
  CeedMemType cmem_type;
  CeedVectorGetCeed(x_ceed, &ceed);
  CeedGetPreferredMemType(ceed, &cmem_type);

  CeedVectorTakeArray(x_ceed, MemTypeP2C(mem_type), &x);
  switch (cmem_type) {
    case CEED_MEM_HOST:
      PetscCall(VecRestoreArray(X_petsc, &x));
      break;
    case CEED_MEM_DEVICE:
      PetscCall(VecRestoreArrayAndMemType(X_petsc, &x));
      break;
  }
  PetscFunctionReturn(0);
}

/**
  @brief Transfer read-only array from PETSc Vec to CeedVector

  @param[in]   X_petsc   PETSc Vec
  @param[out]  mem_type  PETSc MemType
  @param[out]  x_ceed    CeedVector

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode VecReadP2C(Vec X_petsc, PetscMemType *mem_type, CeedVector x_ceed) {
  PetscScalar *x;

  PetscFunctionBeginUser;
  Ceed        ceed;
  CeedMemType cmem_type;
  CeedVectorGetCeed(x_ceed, &ceed);
  CeedGetPreferredMemType(ceed, &cmem_type);

  switch (cmem_type) {
    case CEED_MEM_HOST:
      PetscCall(VecGetArrayRead(X_petsc, (const PetscScalar **)&x));
      *mem_type = PETSC_MEMTYPE_HOST;
      break;
    case CEED_MEM_DEVICE:
      PetscCall(VecGetArrayReadAndMemType(X_petsc, (const PetscScalar **)&x, mem_type));
      break;
  }
  CeedVectorSetArray(x_ceed, MemTypeP2C(*mem_type), CEED_USE_POINTER, x);

  PetscFunctionReturn(0);
}

/**
  @brief Transfer read-only array from CeedVector to PETSc Vec

  @param[in]   x_ceed    CeedVector
  @param[in]   mem_type  PETSc MemType
  @param[out]  X_petsc   PETSc Vec

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode VecReadC2P(CeedVector x_ceed, PetscMemType mem_type, Vec X_petsc) {
  PetscScalar *x;

  PetscFunctionBeginUser;
  Ceed        ceed;
  CeedMemType cmem_type;
  CeedVectorGetCeed(x_ceed, &ceed);
  CeedGetPreferredMemType(ceed, &cmem_type);

  CeedVectorTakeArray(x_ceed, MemTypeP2C(mem_type), &x);
  switch (cmem_type) {
    case CEED_MEM_HOST:
      PetscCall(VecRestoreArrayRead(X_petsc, (const PetscScalar **)&x));
      break;
    case CEED_MEM_DEVICE:
      PetscCall(VecRestoreArrayReadAndMemType(X_petsc, (const PetscScalar **)&x));
      break;
  }

  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// Returns the computed diagonal of the operator
// -----------------------------------------------------------------------------
PetscErrorCode MatGetDiag_Ceed(Mat A, Vec D) {
  MatopApplyContext op_apply_ctx;
  Vec               Y_loc;
  PetscMemType      mem_type;
  PetscFunctionBeginUser;

  PetscCall(MatShellGetContext(A, &op_apply_ctx));
  if (op_apply_ctx->Y_loc) Y_loc = op_apply_ctx->Y_loc;
  else PetscCall(DMGetLocalVector(op_apply_ctx->dm_y, &Y_loc));

  // -- Place PETSc vector in libCEED vector
  PetscCall(VecP2C(Y_loc, &mem_type, op_apply_ctx->y_ceed));

  // -- Compute Diagonal
  CeedOperatorLinearAssembleDiagonal(op_apply_ctx->op, op_apply_ctx->y_ceed, CEED_REQUEST_IMMEDIATE);

  // -- Local-to-Global
  PetscCall(VecC2P(op_apply_ctx->y_ceed, mem_type, Y_loc));
  PetscCall(VecZeroEntries(D));
  PetscCall(DMLocalToGlobal(op_apply_ctx->dm_y, Y_loc, ADD_VALUES, D));

  if (!op_apply_ctx->Y_loc) PetscCall(DMRestoreLocalVector(op_apply_ctx->dm_y, &Y_loc));
  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the Laplacian with Dirichlet boundary conditions
// -----------------------------------------------------------------------------
PetscErrorCode ApplyLocal_Ceed(Vec X, Vec Y, MatopApplyContext op_apply_ctx) {
  PetscMemType x_mem_type, y_mem_type;
  Vec          Y_loc, X_loc;
  MPI_Comm     comm = PetscObjectComm((PetscObject)op_apply_ctx->dm_x);
  PetscFunctionBeginUser;

  PetscCheck((X || op_apply_ctx->X_loc) && (Y || op_apply_ctx->Y_loc), comm, PETSC_ERR_SUP,
             "A global or local PETSc Vec must be provided for both input and output");

  if (op_apply_ctx->Y_loc) Y_loc = op_apply_ctx->Y_loc;
  else PetscCall(DMGetLocalVector(op_apply_ctx->dm_y, &Y_loc));
  if (op_apply_ctx->X_loc) X_loc = op_apply_ctx->X_loc;
  else PetscCall(DMGetLocalVector(op_apply_ctx->dm_x, &X_loc));

  // Global-to-local
  if (X) PetscCall(DMGlobalToLocal(op_apply_ctx->dm_x, X, INSERT_VALUES, X_loc));

  // Setup libCEED vectors
  PetscCall(VecReadP2C(X_loc, &x_mem_type, op_apply_ctx->x_ceed));
  PetscCall(VecP2C(Y_loc, &y_mem_type, op_apply_ctx->y_ceed));

  // Apply libCEED operator
  CeedOperatorApply(op_apply_ctx->op, op_apply_ctx->x_ceed, op_apply_ctx->y_ceed, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  PetscCall(VecReadC2P(op_apply_ctx->x_ceed, x_mem_type, X_loc));
  PetscCall(VecC2P(op_apply_ctx->y_ceed, y_mem_type, Y_loc));

  // Local-to-global
  if (Y) {
    PetscCall(VecZeroEntries(Y));
    PetscCall(DMLocalToGlobal(op_apply_ctx->dm_y, Y_loc, ADD_VALUES, Y));
  }

  if (!op_apply_ctx->Y_loc) PetscCall(DMRestoreLocalVector(op_apply_ctx->dm_y, &Y_loc));
  if (!op_apply_ctx->X_loc) PetscCall(DMRestoreLocalVector(op_apply_ctx->dm_x, &X_loc));
  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Wraps the libCEED operator for a MatShell
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
