#ifndef setup_matops_h
#define setup_matops_h

#include <ceed.h>
#include <petsc.h>

#include "setup-fe.h"
#include "structs.h"

PetscErrorCode ApplyLocalCeedOp(Vec X, Vec Y, OperatorApplyContext op_apply_ctx);
PetscErrorCode ApplyAddLocalCeedOp(Vec X, Vec Y, OperatorApplyContext op_apply_ctx);
PetscErrorCode GetDiagonal(Mat A, Vec D);
#endif  // setup_matops_h
