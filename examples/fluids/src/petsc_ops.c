// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
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
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCallCeed(ceed, CeedOperatorGetActiveVectorLengths(op_apply, &x_size, &y_size));
  {  // Verify sizes
    PetscInt X_size, Y_size, dm_X_size, dm_Y_size;
    CeedSize x_ceed_size, y_ceed_size;
    if (dm_x) PetscCall(DMGetLocalVectorInfo(dm_x, &dm_X_size, NULL, NULL));
    if (dm_y) PetscCall(DMGetLocalVectorInfo(dm_y, &dm_Y_size, NULL, NULL));
    if (X_loc) {
      PetscCall(VecGetLocalSize(X_loc, &X_size));
      PetscCheck(X_size == x_size, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ,
                 "X_loc (%" PetscInt_FMT ") not correct size for CeedOperator active input size (%" CeedSize_FMT ")", X_size, x_size);
      if (dm_x)
        PetscCheck(X_size == dm_X_size, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ,
                   "X_loc size (%" PetscInt_FMT ") does not match dm_x local vector size (%" PetscInt_FMT ")", X_size, dm_X_size);
    }
    if (Y_loc) {
      PetscCall(VecGetLocalSize(Y_loc, &Y_size));
      PetscCheck(Y_size == y_size, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ,
                 "Y_loc (%" PetscInt_FMT ") not correct size for CeedOperator active output size (%" CeedSize_FMT ")", Y_size, y_size);
      if (dm_y)
        PetscCheck(Y_size == dm_Y_size, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ,
                   "Y_loc size (%" PetscInt_FMT ") does not match dm_y local vector size (%" PetscInt_FMT ")", Y_size, dm_Y_size);
    }
    if (x_ceed && x_ceed != CEED_VECTOR_NONE) {
      PetscCallCeed(ceed, CeedVectorGetLength(x_ceed, &x_ceed_size));
      PetscCheck(x_size >= 0 ? x_ceed_size == x_size : true, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ,
                 "x_ceed (%" CeedSize_FMT ") not correct size for CeedOperator active input size (%" CeedSize_FMT ")", x_ceed_size, x_size);
      if (dm_x)
        PetscCheck(x_ceed_size == dm_X_size, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ,
                   "x_ceed size (%" CeedSize_FMT ") does not match dm_x local vector size (%" PetscInt_FMT ")", x_ceed_size, dm_X_size);
    }
    if (y_ceed && y_ceed != CEED_VECTOR_NONE) {
      PetscCallCeed(ceed, CeedVectorGetLength(y_ceed, &y_ceed_size));
      PetscCheck(y_ceed_size == y_size, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ,
                 "y_ceed (%" CeedSize_FMT ") not correct size for CeedOperator active input size (%" CeedSize_FMT ")", y_ceed_size, y_size);
      if (dm_y)
        PetscCheck(y_ceed_size == dm_Y_size, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ,
                   "y_ceed size (%" CeedSize_FMT ") does not match dm_y local vector size (%" PetscInt_FMT ")", y_ceed_size, dm_Y_size);
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
  if (x_ceed) PetscCallCeed(ceed, CeedVectorReferenceCopy(x_ceed, &(*ctx)->x_ceed));
  else PetscCallCeed(ceed, CeedVectorCreate(ceed, x_size, &(*ctx)->x_ceed));

  if (y_ceed) PetscCallCeed(ceed, CeedVectorReferenceCopy(y_ceed, &(*ctx)->y_ceed));
  else PetscCallCeed(ceed, CeedVectorCreate(ceed, y_size, &(*ctx)->y_ceed));

  PetscCallCeed(ceed, CeedOperatorReferenceCopy(op_apply, &(*ctx)->op));
  PetscCallCeed(ceed, CeedReferenceCopy(ceed, &(*ctx)->ceed));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Destroy OperatorApplyContext struct
 *
 * @param[in,out] ctx Context to destroy
 */
PetscErrorCode OperatorApplyContextDestroy(OperatorApplyContext ctx) {
  PetscFunctionBeginUser;
  if (!ctx) PetscFunctionReturn(PETSC_SUCCESS);
  Ceed ceed = ctx->ceed;

  // Destroy PETSc Objects
  PetscCall(DMDestroy(&ctx->dm_x));
  PetscCall(DMDestroy(&ctx->dm_y));
  PetscCall(VecDestroy(&ctx->X_loc));
  PetscCall(VecDestroy(&ctx->Y_loc));

  // Destroy libCEED Objects
  PetscCallCeed(ceed, CeedVectorDestroy(&ctx->x_ceed));
  PetscCallCeed(ceed, CeedVectorDestroy(&ctx->y_ceed));
  PetscCallCeed(ceed, CeedOperatorDestroy(&ctx->op));
  PetscCallCeed(ceed, CeedDestroy(&ctx->ceed));

  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
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
 * @param[in]  op       Operator to make the Vecs for
 * @param[in]  vec_type `VecType` for the new Vecs
 * @param[out] input    Vec for CeedOperator active input
 * @param[out] output   Vec for CeedOperator active output
 */
PetscErrorCode CeedOperatorCreateLocalVecs(CeedOperator op, VecType vec_type, Vec *input, Vec *output) {
  CeedSize input_size, output_size;
  Ceed     ceed;

  PetscFunctionBeginUser;
  PetscCall(CeedOperatorGetCeed(op, &ceed));
  PetscCallCeed(ceed, CeedOperatorGetActiveVectorLengths(op, &input_size, &output_size));
  if (input) {
    PetscCall(VecCreate(PETSC_COMM_SELF, input));
    PetscCall(VecSetType(*input, vec_type));
    PetscCall(VecSetSizes(*input, input_size, input_size));
  }
  if (output) {
    PetscCall(VecCreate(PETSC_COMM_SELF, output));
    PetscCall(VecSetType(*output, vec_type));
    PetscCall(VecSetSizes(*output, output_size, output_size));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Apply FEM Operator defined by `OperatorApplyContext` to various input and output vectors
 *
 * @param[in]     X             Input global `Vec`, maybe `NULL`
 * @param[in]     X_loc         Input local `Vec`, maybe `NULL`
 * @param[in]     x_ceed        Input `CeedVector`, maybe `CEED_VECTOR_NONE`
 * @param[in,out] y_ceed        Output `CeedVector`, maybe `CEED_VECTOR_NONE`
 * @param[in,out] Y_loc         Output local `Vec`, maybe `NULL`
 * @param[in,out] Y             Output global `Vec`, maybe `NULL`
 * @param[in]     ctx           Context for the operator apply
 * @param[in]     use_apply_add Whether to use `CeedOperatorApply` or `CeedOperatorApplyAdd`
 */
PetscErrorCode ApplyCeedOperator_Core(Vec X, Vec X_loc, CeedVector x_ceed, CeedVector y_ceed, Vec Y_loc, Vec Y, OperatorApplyContext ctx,
                                      bool use_apply_add) {
  PetscMemType x_mem_type, y_mem_type;
  Ceed         ceed = ctx->ceed;

  PetscFunctionBeginUser;
  if (X) PetscCall(DMGlobalToLocal(ctx->dm_x, X, INSERT_VALUES, X_loc));
  if (X_loc) PetscCall(VecReadPetscToCeed(X_loc, &x_mem_type, x_ceed));

  if (Y_loc) PetscCall(VecPetscToCeed(Y_loc, &y_mem_type, y_ceed));

  PetscCall(PetscLogEventBegin(FLUIDS_CeedOperatorApply, X, Y, 0, 0));
  PetscCall(PetscLogGpuTimeBegin());
  if (use_apply_add) PetscCallCeed(ceed, CeedOperatorApplyAdd(ctx->op, x_ceed, y_ceed, CEED_REQUEST_IMMEDIATE));
  else PetscCallCeed(ceed, CeedOperatorApply(ctx->op, x_ceed, y_ceed, CEED_REQUEST_IMMEDIATE));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogEventEnd(FLUIDS_CeedOperatorApply, X, Y, 0, 0));

  if (X_loc) PetscCall(VecReadCeedToPetsc(ctx->x_ceed, x_mem_type, X_loc));

  if (Y_loc) PetscCall(VecCeedToPetsc(ctx->y_ceed, y_mem_type, Y_loc));
  if (Y) PetscCall(DMLocalToGlobal(ctx->dm_y, Y_loc, ADD_VALUES, Y));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ApplyCeedOperatorGlobalToLocal(Vec X, Vec Y_loc, OperatorApplyContext ctx) {
  Vec X_loc = ctx->X_loc;

  PetscFunctionBeginUser;
  // Get local vectors (if needed)
  if (!ctx->X_loc) PetscCall(DMGetLocalVector(ctx->dm_x, &X_loc));

  PetscCall(ApplyCeedOperator_Core(X, X_loc, ctx->x_ceed, ctx->y_ceed, Y_loc, NULL, ctx, false));

  // Restore local vector (if needed)
  if (!ctx->X_loc) PetscCall(DMRestoreLocalVector(ctx->dm_x, &X_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ApplyCeedOperatorLocalToLocal(Vec X_loc, Vec Y_loc, OperatorApplyContext ctx) {
  PetscFunctionBeginUser;
  PetscCall(ApplyCeedOperator_Core(NULL, X_loc, ctx->x_ceed, ctx->y_ceed, Y_loc, NULL, ctx, false));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ApplyAddCeedOperatorLocalToLocal(Vec X_loc, Vec Y_loc, OperatorApplyContext ctx) {
  PetscFunctionBeginUser;
  PetscCall(ApplyCeedOperator_Core(NULL, X_loc, ctx->x_ceed, ctx->y_ceed, Y_loc, NULL, ctx, true));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Return Mats for KSP solve
 *
 * Uses command-line flag with `ksp`'s prefix to determine if mat_ceed should be used directly or whether it should be assembled.
 * If `Amat` is to be assembled, then `Pmat` is equal to `Amat`.
 *
 * If `Amat` uses `mat_ceed`, then `Pmat` is either assembled or uses `mat_ceed` based on the preconditioner choice in `ksp`.
 *
 * @param[in]  ksp      `KSP` object for used for solving
 * @param[in]  mat_ceed `MATCEED` for the linear operator
 * @param[in]  assemble Whether to assemble `Amat` and `Pmat` if they are not `mat_ceed`
 * @param[out] Amat     `Mat` to be used for the solver `Amat`
 * @param[out] Pmat     `Mat` to be used for the solver `Pmat`
 */
PetscErrorCode CreateSolveOperatorsFromMatCeed(KSP ksp, Mat mat_ceed, PetscBool assemble, Mat *Amat, Mat *Pmat) {
  PetscBool use_matceed_pmat, assemble_amat = PETSC_FALSE;
  MatType   mat_ceed_inner_type;

  PetscFunctionBeginUser;
  PetscCall(MatCeedGetInnerMatType(mat_ceed, &mat_ceed_inner_type));
  {  // Determine if Amat should be MATCEED or assembled
    const char *ksp_prefix = NULL;

    PetscCall(KSPGetOptionsPrefix(ksp, &ksp_prefix));
    PetscOptionsBegin(PetscObjectComm((PetscObject)mat_ceed), ksp_prefix, "", NULL);
    PetscCall(PetscOptionsBool("-matceed_assemble_amat", "Assemble the A matrix for KSP solve", NULL, assemble_amat, &assemble_amat, NULL));
    PetscOptionsEnd();
  }

  if (assemble_amat) {
    PetscCall(MatConvert(mat_ceed, mat_ceed_inner_type, MAT_INITIAL_MATRIX, Amat));
    if (assemble) PetscCall(MatCeedAssembleCOO(mat_ceed, *Amat));

    PetscCall(PetscObjectReference((PetscObject)*Amat));
    *Pmat = *Amat;
    PetscFunctionReturn(PETSC_SUCCESS);
  } else {
    PetscCall(PetscObjectReference((PetscObject)mat_ceed));
    *Amat = mat_ceed;
  }

  {  // Determine if Pmat should be MATCEED or assembled
    PC     pc;
    PCType pc_type;

    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCGetType(pc, &pc_type));
    PetscCall(PetscStrcmpAny(pc_type, &use_matceed_pmat, PCJACOBI, PCVPBJACOBI, PCPBJACOBI, ""));
  }

  if (use_matceed_pmat) {
    PetscCall(PetscObjectReference((PetscObject)mat_ceed));
    *Pmat = mat_ceed;
  } else {
    PetscCall(MatConvert(mat_ceed, mat_ceed_inner_type, MAT_INITIAL_MATRIX, Pmat));
    if (assemble) PetscCall(MatCeedAssembleCOO(mat_ceed, *Pmat));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Runs KSPSetFromOptions and sets Operators based on mat_ceed
 *
 * See CreateSolveOperatorsFromMatCeed for details on how the KSPSolve operators are set.
 *
 * @param[in] ksp      `KSP` of the solve
 * @param[in] mat_ceed `MatCeed` linear operator to solve for
 */
PetscErrorCode KSPSetFromOptions_WithMatCeed(KSP ksp, Mat mat_ceed) {
  Mat Amat, Pmat;

  PetscFunctionBeginUser;
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(CreateSolveOperatorsFromMatCeed(ksp, mat_ceed, PETSC_TRUE, &Amat, &Pmat));
  PetscCall(KSPSetOperators(ksp, Amat, Pmat));
  PetscCall(MatDestroy(&Amat));
  PetscCall(MatDestroy(&Pmat));
  PetscFunctionReturn(PETSC_SUCCESS);
}
// -----------------------------------------------------------------------------
