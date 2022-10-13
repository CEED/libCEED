#ifndef libceed_petsc_examples_matops_h
#define libceed_petsc_examples_matops_h

#include <ceed.h>
#include <petsc.h>
#include <petscdmplex.h>

#include "structs.h"

PetscErrorCode MatGetDiag(Mat A, Vec D);
PetscErrorCode ApplyLocal_Ceed(Vec X, Vec Y, OperatorApplyContext op_apply_ctx);
PetscErrorCode MatMult_Ceed(Mat A, Vec X, Vec Y);
PetscErrorCode FormResidual_Ceed(SNES snes, Vec X, Vec Y, void *ctx);
PetscErrorCode MatMult_Prolong(Mat A, Vec X, Vec Y);
PetscErrorCode MatMult_Restrict(Mat A, Vec X, Vec Y);
PetscErrorCode ComputeErrorMax(OperatorApplyContext pr_restr_ctx,
                               CeedOperator op_error,
                               Vec X, CeedVector target, PetscReal *max_error);

#endif // libceed_petsc_examples_matops_h
