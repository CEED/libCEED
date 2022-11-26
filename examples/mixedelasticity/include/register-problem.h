#ifndef register_problems_h
#define register_problems_h

#include "structs.h"

// Register problems to be available on the command line
PetscErrorCode RegisterProblems_MixedElasticity(AppCtx app_ctx);
// -----------------------------------------------------------------------------
// Set up problems function prototype
// -----------------------------------------------------------------------------
// 1) BP4
PetscErrorCode BP4(Ceed ceed, ProblemData problem_data, DM dm, void *ctx);

// 1) Linear
PetscErrorCode Linear(Ceed ceed, ProblemData problem_data, DM dm, void *ctx);

#endif  // register_problems_h
