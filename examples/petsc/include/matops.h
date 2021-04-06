#ifndef matops_h
#define matops_h

#include <ceed.h>
#include <petsc.h>
#include <petscdmplex.h>

#include "structs.h"

PetscErrorCode MatGetDiag(Mat A, Vec D);
PetscErrorCode ApplyLocal_Ceed(Vec X, Vec Y, UserO user);
PetscErrorCode MatMult_Ceed(Mat A, Vec X, Vec Y);
PetscErrorCode FormResidual_Ceed(SNES snes, Vec X, Vec Y, void *ctx);
PetscErrorCode MatMult_Prolong(Mat A, Vec X, Vec Y);
PetscErrorCode MatMult_Restrict(Mat A, Vec X, Vec Y);
PetscErrorCode ComputeErrorMax(UserO user, CeedOperator opError,
                               Vec X, CeedVector target, PetscReal *maxerror);

#endif // matops_h
