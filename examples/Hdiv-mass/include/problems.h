#ifndef problems_h
#define problems_h

#include "../include/structs.h"

// -----------------------------------------------------------------------------
// Set up problems function prototype
// -----------------------------------------------------------------------------
// 1) poisson-quad2d
PetscErrorCode Hdiv_POISSON_MASS2D(ProblemData *problem_data, void *ctx);

// 2) poisson-hex3d
PetscErrorCode Hdiv_POISSON_MASS3D(ProblemData *problem_data, void *ctx);

// 3) poisson-prism3d

// 4) richard

#endif // problems_h
