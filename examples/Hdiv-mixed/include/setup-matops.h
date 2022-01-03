#ifndef setup_matops_h
#define setup_matops_h

#include <ceed.h>
#include <petsc.h>

#include "structs.h"

PetscErrorCode ApplyLocalCeedOp(Vec X, Vec Y,
                                OperatorApplyContext op_apply_ctx);
PetscErrorCode ApplyAddLocalCeedOp(Vec X, Vec Y,
                                   OperatorApplyContext op_apply_ctx);

#endif // setup_matops_h
