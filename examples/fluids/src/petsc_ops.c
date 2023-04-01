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

// @brief Get information about a DM's local vector
PetscErrorCode DMGetLocalVectorInfo(DM dm, PetscInt *local_size, PetscInt *global_size, VecType *vec_type) {
  Vec V_loc;

  PetscFunctionBeginUser;
  PetscCall(DMGetLocalVector(dm, &V_loc));
  if (local_size) PetscCall(VecGetLocalSize(V_loc, local_size));
  if (global_size) PetscCall(VecGetSize(V_loc, global_size));
  if (vec_type) PetscCall(VecGetType(V_loc, vec_type));
  PetscCall(DMRestoreLocalVector(dm, &V_loc));
  PetscFunctionReturn(0);
}

// @brief Get information about a DM's global vector
PetscErrorCode DMGetGlobalVectorInfo(DM dm, PetscInt *local_size, PetscInt *global_size, VecType *vec_type) {
  Vec V;

  PetscFunctionBeginUser;
  PetscCall(DMGetGlobalVector(dm, &V));
  if (local_size) PetscCall(VecGetLocalSize(V, local_size));
  if (global_size) PetscCall(VecGetSize(V, global_size));
  if (vec_type) PetscCall(VecGetType(V, vec_type));
  PetscCall(DMRestoreGlobalVector(dm, &V));
  PetscFunctionReturn(0);
}

/**
 * @brief Create OperatorApplyContext struct for applying FEM operator in a PETSc context
 *
 * All passed in objects are reference copied and may be destroyed if desired (with the exception of `CEED_VECTOR_NONE`).
 * Resulting context should be destroyed with `OperatorApplyContextDestroy()`.
 *
 * @param[in]  dm_x     `DM` associated with the operator active input. May be `NULL`
 * @param[in]  dm_y     `DM` associated with the operator active output. May be `NULL`
 * @param[in]  ceed     `Ceed` object
 * @param[in]  op_apply `CeedOperator` representing the local action of the FEM operator
 * @param[in]  x_ceed   `CeedVector` for operator active input. May be `CEED_VECTOR_NONE` or `NULL`. If `NULL`, `CeedVector` will be automatically
 *                      generated.
 * @param[in]  y_ceed   `CeedVector` for operator active output. May be `CEED_VECTOR_NONE` or `NULL`. If `NULL`, `CeedVector` will be automatically
 *                      generated.
 * @param[in]  X_loc    Local `Vec` for operator active input. If `NULL`, vector will be obtained if needed at ApplyCeedOperator time.
 * @param[in]  Y_loc    Local `Vec` for operator active output. If `NULL`, vector will be obtained if needed at ApplyCeedOperator time.
 * @param[out] ctx      Struct containing all data necessary for applying the operator
 */
PetscErrorCode OperatorApplyContextCreate(DM dm_x, DM dm_y, Ceed ceed, CeedOperator op_apply, CeedVector x_ceed, CeedVector y_ceed, Vec X_loc,
                                          Vec Y_loc, OperatorApplyContext *ctx) {
  CeedSize x_size, y_size;

  PetscFunctionBeginUser;
  CeedOperatorGetActiveVectorLengths(op_apply, &x_size, &y_size);
  {  // Verify sizes
    PetscInt X_size, Y_size;
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

  PetscCall(PetscNew(ctx));

  // Copy PETSc Objects
  if (dm_x) PetscCall(PetscObjectReference((PetscObject)dm_x));
  (*ctx)->dm_x = dm_x;
  if (dm_y) PetscCall(PetscObjectReference((PetscObject)dm_y));
  (*ctx)->dm_y = dm_y;

  if (X_loc) PetscCall(PetscObjectReference((PetscObject)X_loc));
  (*ctx)->X_loc = X_loc;
  if (Y_loc) PetscCall(PetscObjectReference((PetscObject)Y_loc));
  (*ctx)->Y_loc = Y_loc;

  // Copy libCEED objects
  if (x_ceed) CeedVectorReferenceCopy(x_ceed, &(*ctx)->x_ceed);
  else CeedVectorCreate(ceed, x_size, &(*ctx)->x_ceed);

  if (y_ceed) CeedVectorReferenceCopy(y_ceed, &(*ctx)->y_ceed);
  else CeedVectorCreate(ceed, y_size, &(*ctx)->y_ceed);

  CeedOperatorReferenceCopy(op_apply, &(*ctx)->op);
  CeedReferenceCopy(ceed, &(*ctx)->ceed);

  PetscFunctionReturn(0);
}

/**
 * @brief Destroy OperatorApplyContext struct
 *
 * @param[in,out] ctx Context to destroy
 */
PetscErrorCode OperatorApplyContextDestroy(OperatorApplyContext ctx) {
  PetscFunctionBeginUser;

  if (!ctx) PetscFunctionReturn(0);

  // Destroy PETSc Objects
  PetscCall(DMDestroy(&ctx->dm_x));
  PetscCall(DMDestroy(&ctx->dm_y));
  PetscCall(VecDestroy(&ctx->X_loc));
  PetscCall(VecDestroy(&ctx->Y_loc));

  // Destroy libCEED Objects
  CeedVectorDestroy(&ctx->x_ceed);
  CeedVectorDestroy(&ctx->y_ceed);
  CeedOperatorDestroy(&ctx->op);
  CeedDestroy(&ctx->ceed);

  PetscCall(PetscFree(ctx));

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

PetscErrorCode ApplyCeedOperatorLocalToLocal(Vec X_loc, Vec Y_loc, OperatorApplyContext ctx) {
  PetscFunctionBeginUser;
  PetscCall(ApplyCeedOperator_Core(NULL, X_loc, ctx->x_ceed, ctx->y_ceed, Y_loc, NULL, ctx, false));
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

/**
 * @brief Create PETSc MatShell object for the corresponding OperatorApplyContext
 *
 * @param[in]  ctx Context that does the action of the operator
 * @param[out] mat MatShell for the operator
 */
PetscErrorCode CreateMatShell_Ceed(OperatorApplyContext ctx, Mat *mat) {
  MPI_Comm comm_x = PetscObjectComm((PetscObject)(ctx->dm_x));
  MPI_Comm comm_y = PetscObjectComm((PetscObject)(ctx->dm_y));
  PetscInt X_loc_size, X_size, Y_size, Y_loc_size;
  VecType  X_vec_type, Y_vec_type;

  PetscFunctionBeginUser;

  PetscCheck(comm_x == comm_y, PETSC_COMM_WORLD, PETSC_ERR_ARG_NOTSAMECOMM, "Input and output comm must be the same");

  PetscCall(DMGetGlobalVectorInfo(ctx->dm_x, &X_loc_size, &X_size, &X_vec_type));
  PetscCall(DMGetGlobalVectorInfo(ctx->dm_y, &Y_loc_size, &Y_size, &Y_vec_type));

  PetscCall(MatCreateShell(comm_x, Y_loc_size, X_loc_size, Y_size, X_size, ctx, mat));
  PetscCall(MatShellSetContextDestroy(*mat, (PetscErrorCode(*)(void *))OperatorApplyContextDestroy));
  PetscCall(MatShellSetOperation(*mat, MATOP_MULT, (void (*)(void))MatMult_Ceed));
  PetscCall(MatShellSetOperation(*mat, MATOP_GET_DIAGONAL, (void (*)(void))MatGetDiag_Ceed));

  PetscCheck(X_vec_type == Y_vec_type, PETSC_COMM_WORLD, PETSC_ERR_ARG_NOTSAMETYPE, "Vec_type of ctx->dm_x (%s) and ctx->dm_y (%s) must be the same",
             X_vec_type, Y_vec_type);
  PetscCall(MatShellSetVecType(*mat, X_vec_type));

  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
