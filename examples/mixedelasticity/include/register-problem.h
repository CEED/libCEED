#ifndef register_problems_h
#define register_problems_h

#include "structs.h"

// Register problems to be available on the command line
PetscErrorCode RegisterProblems_MixedElasticity(AppCtx app_ctx);
// -----------------------------------------------------------------------------
// Set up problems function prototype
// -----------------------------------------------------------------------------
// 3) Mixed Linear Elasticity
PetscErrorCode Mixed_Linear_2D(Ceed ceed, ProblemData problem_data, void *ctx);
PetscErrorCode Mixed_Linear_3D(Ceed ceed, ProblemData problem_data, void *ctx);

#endif  // register_problems_h
