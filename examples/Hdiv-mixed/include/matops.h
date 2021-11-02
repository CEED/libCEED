#ifndef matops_h
#define matops_h

#include <ceed.h>
#include <petsc.h>

#include "structs.h"

PetscErrorCode MatGetDiag(Mat A, Vec D);
PetscErrorCode ApplyLocal_Ceed(Vec X, Vec Y, User user);
PetscErrorCode MatMult_Ceed(Mat A, Vec X, Vec Y);

#endif // matops_h
