#include <petsc.h>

PetscErrorCode BCsDiff(PetscInt dim, PetscReal time, const PetscReal x[],
                       PetscInt ncompu, PetscScalar *u, void *ctx);
PetscErrorCode BCsMass(PetscInt dim, PetscReal time, const PetscReal x[],
                       PetscInt ncompu, PetscScalar *u, void *ctx);
