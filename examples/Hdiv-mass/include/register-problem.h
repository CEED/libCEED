#ifndef register_problem_h
#define register_problem_h

#include "structs.h"

// Register problems to be available on the command line
PetscErrorCode RegisterProblems_Hdiv(AppCtx app_ctx);

// -----------------------------------------------------------------------------
// Set up problems function prototype
// -----------------------------------------------------------------------------
// 1) poisson-quad2d
PetscErrorCode Hdiv_POISSON_MASS2D(ProblemData problem_data, void *ctx);

// 2) poisson-hex3d
PetscErrorCode Hdiv_POISSON_MASS3D(ProblemData problem_data, void *ctx);

#endif  // register_problem_h
