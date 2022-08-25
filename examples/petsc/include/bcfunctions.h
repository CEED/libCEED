#ifndef libceed_petsc_examples_bc_functions_h
#define libceed_petsc_examples_bc_functions_h

#include <petsc.h>

PetscErrorCode BCsDiff(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt num_comp_u, PetscScalar *u, void *ctx);
PetscErrorCode BCsMass(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt num_comp_u, PetscScalar *u, void *ctx);

#endif  // libceed_petsc_examples_bc_functions_h
