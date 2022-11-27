#ifndef register_problems_h
#define register_problems_h

#include "structs.h"

// Register problems to be available on the command line
PetscErrorCode RegisterProblems_MixedElasticity(AppCtx app_ctx);
// -----------------------------------------------------------------------------
// Set up problems function prototype
// -----------------------------------------------------------------------------
// 1) BP4
PetscErrorCode BP4_2D(Ceed ceed, ProblemData problem_data, void *ctx);

PetscErrorCode BP4_3D(Ceed ceed, ProblemData problem_data, void *ctx);

// 1) Linear Elasticity
PetscErrorCode Linear_2D(Ceed ceed, ProblemData problem_data, void *ctx);

PetscErrorCode Linear_3D(Ceed ceed, ProblemData problem_data, void *ctx);

#endif  // register_problems_h
