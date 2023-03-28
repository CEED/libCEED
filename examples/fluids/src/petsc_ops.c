// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../include/petsc_ops.h"

#include <ceed.h>
#include <petscdm.h>

#include "../navierstokes.h"

// -----------------------------------------------------------------------------
// Setup apply operator context data
// -----------------------------------------------------------------------------
PetscErrorCode OperatorApplyContextCreate(DM dm_x, DM dm_y, Ceed ceed, CeedOperator op_apply, CeedVector x_ceed, CeedVector y_ceed, Vec X_loc,
                                          Vec Y_loc, OperatorApplyContext *op_apply_ctx) {
  PetscFunctionBeginUser;

  {  // Verify sizes
    CeedSize x_size, y_size;
    PetscInt X_size, Y_size;
    CeedOperatorGetActiveVectorLengths(op_apply, &x_size, &y_size);
    if (X_loc) {
      PetscCall(VecGetLocalSize(X_loc, &X_size));
      PetscCheck(X_size == x_size, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ,
                 "X_loc (%" PetscInt_FMT ") not correct size for CeedOperator active input size (%" CeedSize_FMT ")", X_size, x_size);
    }
    if (Y_loc) {
      PetscCall(VecGetLocalSize(Y_loc, &Y_size));
      PetscCheck(Y_size == y_size, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ,
                 "Y_loc (%" PetscInt_FMT ") not correct size for CeedOperator active output size (%" CeedSize_FMT ")", Y_size, y_size);
    }
  }

  PetscCall(PetscNew(op_apply_ctx));

  // Copy PETSc Objects
  if (dm_x) PetscCall(PetscObjectReference((PetscObject)dm_x));
  (*op_apply_ctx)->dm_x = dm_x;
  if (dm_y) PetscCall(PetscObjectReference((PetscObject)dm_y));
  (*op_apply_ctx)->dm_y = dm_y;

  if (X_loc) PetscCall(PetscObjectReference((PetscObject)X_loc));
  (*op_apply_ctx)->X_loc = X_loc;
  if (Y_loc) PetscCall(PetscObjectReference((PetscObject)Y_loc));
  (*op_apply_ctx)->Y_loc = Y_loc;

  // Copy libCEED objects
  if (x_ceed) CeedVectorReferenceCopy(x_ceed, &(*op_apply_ctx)->x_ceed);
  if (y_ceed) CeedVectorReferenceCopy(y_ceed, &(*op_apply_ctx)->y_ceed);
  CeedOperatorReferenceCopy(op_apply, &(*op_apply_ctx)->op);
  CeedReferenceCopy(ceed, &(*op_apply_ctx)->ceed);

  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// Destroy apply operator context data
// -----------------------------------------------------------------------------
PetscErrorCode OperatorApplyContextDestroy(OperatorApplyContext op_apply_ctx) {
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
  PetscInt     X_size;
  CeedSize     x_size;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(X_petsc, &X_size));
  CeedVectorGetLength(x_ceed, &x_size);
  PetscCheck(X_size == x_size, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "X_petsc (%" PetscInt_FMT ") and x_ceed (%" CeedSize_FMT ") must be same size",
             X_size, x_size);

  PetscCall(VecGetArrayAndMemType(X_petsc, &x, mem_type));
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
  PetscInt     X_size;
  CeedSize     x_size;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(X_petsc, &X_size));
  CeedVectorGetLength(x_ceed, &x_size);
  PetscCheck(X_size == x_size, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "X_petsc (%" PetscInt_FMT ") and x_ceed (%" CeedSize_FMT ") must be same size",
             X_size, x_size);

  CeedVectorTakeArray(x_ceed, MemTypeP2C(mem_type), &x);
  PetscCall(VecRestoreArrayAndMemType(X_petsc, &x));

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
  PetscInt     X_size;
  CeedSize     x_size;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(X_petsc, &X_size));
  CeedVectorGetLength(x_ceed, &x_size);
  PetscCheck(X_size == x_size, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "X_petsc (%" PetscInt_FMT ") and x_ceed (%" CeedSize_FMT ") must be same size",
             X_size, x_size);

  PetscCall(VecGetArrayReadAndMemType(X_petsc, (const PetscScalar **)&x, mem_type));
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
  PetscInt     X_size;
  CeedSize     x_size;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(X_petsc, &X_size));
  CeedVectorGetLength(x_ceed, &x_size);
  PetscCheck(X_size == x_size, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "X_petsc (%" PetscInt_FMT ") and x_ceed (%" CeedSize_FMT ") must be same size",
             X_size, x_size);

  CeedVectorTakeArray(x_ceed, MemTypeP2C(mem_type), &x);
  PetscCall(VecRestoreArrayReadAndMemType(X_petsc, (const PetscScalar **)&x));

  PetscFunctionReturn(0);
}

/**
  @brief Copy PETSc Vec data into CeedVector

  @param[in]   X_petsc PETSc Vec
  @param[out]  x_ceed  CeedVector

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode VecCopyP2C(Vec X_petsc, CeedVector x_ceed) {
  PetscScalar *x;
  PetscMemType mem_type;
  PetscInt     X_size;
  CeedSize     x_size;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(X_petsc, &X_size));
  CeedVectorGetLength(x_ceed, &x_size);
  PetscCheck(X_size == x_size, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "X_petsc (%" PetscInt_FMT ") and x_ceed (%" CeedSize_FMT ") must be same size",
             X_size, x_size);

  PetscCall(VecGetArrayReadAndMemType(X_petsc, (const PetscScalar **)&x, &mem_type));
  CeedVectorSetArray(x_ceed, MemTypeP2C(mem_type), CEED_COPY_VALUES, x);
  PetscCall(VecRestoreArrayReadAndMemType(X_petsc, (const PetscScalar **)&x));

  PetscFunctionReturn(0);
}

//@brief Return VecType for the given DM
VecType DMReturnVecType(DM dm) {
  VecType vec_type;
  DMGetVecType(dm, &vec_type);
  return vec_type;
}

/**
 * @brief Create local PETSc Vecs for CeedOperator's active input/outputs
 *
 * This is primarily used for when the active input/ouput vector does not correspond to a `DM` object, and thus `DMCreateLocalVector` or
 * `DMGetLocalVector` are not applicable.
 * For example, if statitics are being store at quadrature points, a `DM`-created `Vec` will not have the
 * correct size.
 *
 * @param[in]  dm     DM overwhich the Vecs would be used
 * @param[in]  op     Operator to make the Vecs for
 * @param[out] input  Vec for CeedOperator active input
 * @param[out] output Vec for CeedOperator active output
 */
PetscErrorCode CeedOperatorCreateLocalVecs(CeedOperator op, VecType vec_type, MPI_Comm comm, Vec *input, Vec *output) {
  CeedSize input_size, output_size;

  PetscFunctionBeginUser;
  CeedOperatorGetActiveVectorLengths(op, &input_size, &output_size);
  if (input) {
    PetscCall(VecCreate(comm, input));
    PetscCall(VecSetType(*input, vec_type));
    PetscCall(VecSetSizes(*input, input_size, input_size));
  }
  if (output) {
    PetscCall(VecCreate(comm, output));
    PetscCall(VecSetType(*output, vec_type));
    PetscCall(VecSetSizes(*output, output_size, output_size));
  }

  PetscFunctionReturn(0);
}

/**
 * @brief Apply FEM Operator defined by `OperatorApplyContext` to various input and output vectors
 *
 * @param X             Input global `Vec`, maybe `NULL`
 * @param X_loc         Input local `Vec`, maybe `NULL`
 * @param x_ceed        Input `CeedVector`, maybe `CEED_VECTOR_NONE`
 * @param y_ceed        Output `CeedVector`, maybe `CEED_VECTOR_NONE`
 * @param Y_loc         Output local `Vec`, maybe `NULL`
 * @param Y             Output global `Vec`, maybe `NULL`
 * @param ctx           Context for the operator apply
 * @param use_apply_add Whether to use `CeedOperatorApply` or `CeedOperatorApplyAdd`
 */
PetscErrorCode ApplyCeedOperator_Core(Vec X, Vec X_loc, CeedVector x_ceed, CeedVector y_ceed, Vec Y_loc, Vec Y, OperatorApplyContext ctx,
                                      bool use_apply_add) {
  PetscMemType x_mem_type, y_mem_type;

  PetscFunctionBeginUser;
  if (X) PetscCall(DMGlobalToLocal(ctx->dm_x, X, INSERT_VALUES, X_loc));
  if (X_loc) PetscCall(VecReadP2C(X_loc, &x_mem_type, x_ceed));

  if (Y_loc) PetscCall(VecP2C(Y_loc, &y_mem_type, y_ceed));

  if (use_apply_add) CeedOperatorApplyAdd(ctx->op, x_ceed, y_ceed, CEED_REQUEST_IMMEDIATE);
  else CeedOperatorApply(ctx->op, x_ceed, y_ceed, CEED_REQUEST_IMMEDIATE);

  if (X_loc) PetscCall(VecReadC2P(ctx->x_ceed, x_mem_type, X_loc));

  if (Y_loc) PetscCall(VecC2P(ctx->y_ceed, y_mem_type, Y_loc));
  if (Y) PetscCall(DMLocalToGlobal(ctx->dm_y, Y_loc, ADD_VALUES, Y));

  PetscFunctionReturn(0);
};

PetscErrorCode ApplyCeedOperatorGlobalToGlobal(Vec X, Vec Y, OperatorApplyContext ctx) {
  Vec X_loc = ctx->X_loc, Y_loc = ctx->Y_loc;

  PetscFunctionBeginUser;
  PetscCall(VecZeroEntries(Y));

  // Get local vectors (if needed)
  if (!ctx->X_loc) PetscCall(DMGetLocalVector(ctx->dm_x, &X_loc));
  if (!ctx->Y_loc) PetscCall(DMGetLocalVector(ctx->dm_y, &Y_loc));

  PetscCall(ApplyCeedOperator_Core(X, X_loc, ctx->x_ceed, ctx->y_ceed, Y_loc, Y, ctx, false));

  // Restore local vector (if needed)
  if (!ctx->X_loc) PetscCall(DMRestoreLocalVector(ctx->dm_x, &X_loc));
  if (!ctx->Y_loc) PetscCall(DMRestoreLocalVector(ctx->dm_y, &Y_loc));

  PetscFunctionReturn(0);
}

PetscErrorCode ApplyCeedOperatorLocalToGlobal(Vec X_loc, Vec Y, OperatorApplyContext ctx) {
  Vec Y_loc = ctx->Y_loc;

  PetscFunctionBeginUser;
  PetscCall(VecZeroEntries(Y));

  // Get local vectors (if needed)
  if (!ctx->Y_loc) PetscCall(DMGetLocalVector(ctx->dm_y, &Y_loc));

  PetscCall(ApplyCeedOperator_Core(NULL, X_loc, ctx->x_ceed, ctx->y_ceed, Y_loc, Y, ctx, false));

  // Restore local vectors (if needed)
  if (!ctx->Y_loc) PetscCall(DMRestoreLocalVector(ctx->dm_y, &Y_loc));

  PetscFunctionReturn(0);
}

PetscErrorCode ApplyCeedOperatorGlobalToLocal(Vec X, Vec Y_loc, OperatorApplyContext ctx) {
  Vec X_loc = ctx->X_loc;

  PetscFunctionBeginUser;
  // Get local vectors (if needed)
  if (!ctx->X_loc) PetscCall(DMGetLocalVector(ctx->dm_x, &X_loc));

  PetscCall(ApplyCeedOperator_Core(X, X_loc, ctx->x_ceed, ctx->y_ceed, Y_loc, NULL, ctx, false));

  // Restore local vector (if needed)
  if (!ctx->X_loc) PetscCall(DMRestoreLocalVector(ctx->dm_x, &X_loc));

  PetscFunctionReturn(0);
}

PetscErrorCode ApplyAddCeedOperatorLocalToLocal(Vec X_loc, Vec Y_loc, OperatorApplyContext ctx) {
  PetscFunctionBeginUser;
  PetscCall(ApplyCeedOperator_Core(NULL, X_loc, ctx->x_ceed, ctx->y_ceed, Y_loc, NULL, ctx, true));
  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// Wraps the libCEED operator for a MatShell
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_Ceed(Mat A, Vec X, Vec Y) {
  OperatorApplyContext ctx;
  PetscFunctionBeginUser;

  PetscCall(MatShellGetContext(A, &ctx));
  PetscCall(ApplyCeedOperatorGlobalToGlobal(X, Y, ctx));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Returns the computed diagonal of the operator
// -----------------------------------------------------------------------------
PetscErrorCode MatGetDiag_Ceed(Mat A, Vec D) {
  OperatorApplyContext ctx;
  Vec                  Y_loc;
  PetscMemType         mem_type;
  PetscFunctionBeginUser;

  PetscCall(MatShellGetContext(A, &ctx));
  if (ctx->Y_loc) Y_loc = ctx->Y_loc;
  else PetscCall(DMGetLocalVector(ctx->dm_y, &Y_loc));

  // -- Place PETSc vector in libCEED vector
  PetscCall(VecP2C(Y_loc, &mem_type, ctx->y_ceed));

  // -- Compute Diagonal
  CeedOperatorLinearAssembleDiagonal(ctx->op, ctx->y_ceed, CEED_REQUEST_IMMEDIATE);

  // -- Local-to-Global
  PetscCall(VecC2P(ctx->y_ceed, mem_type, Y_loc));
  PetscCall(VecZeroEntries(D));
  PetscCall(DMLocalToGlobal(ctx->dm_y, Y_loc, ADD_VALUES, D));

  if (!ctx->Y_loc) PetscCall(DMRestoreLocalVector(ctx->dm_y, &Y_loc));
  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
