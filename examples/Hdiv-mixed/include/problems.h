#ifndef problems_h
#define problems_h

#include "../include/structs.h"

// -----------------------------------------------------------------------------
// Set up problems function prototype
// -----------------------------------------------------------------------------
// 1) poisson-quad2d
PetscErrorCode Hdiv_POISSON_QUAD2D(ProblemData *problem_data, void *ctx);

PetscErrorCode SetupContext_POISSON_QUAD2D(Ceed ceed, CeedData ceed_data,
    Physics phys);

// 2) poisson-hex3d

// 3) poisson-prism3d

// 4) richard

#endif // problems_h
