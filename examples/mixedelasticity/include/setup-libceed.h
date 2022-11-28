#ifndef libceed_petsc_examples_setup_h
#define libceed_petsc_examples_setup_h

#include <ceed.h>
#include <petsc.h>

#include "structs.h"

PetscErrorCode CeedDataDestroy(CeedData ceed_data, ProblemData problem_data);
PetscErrorCode SetupLibceed(DM dm, Ceed ceed, AppCtx app_ctx, ProblemData problem_data, CeedData ceed_data, CeedVector rhs_ceed);

#endif  // libceed_petsc_examples_setup_h
