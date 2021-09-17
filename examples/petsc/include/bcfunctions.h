#ifndef bcfunctions_h
#define bcfunctions_h

#include <petsc.h>

PetscErrorCode BCsDiff(PetscInt dim, PetscReal time, const PetscReal x[],
                       PetscInt num_comp_u, PetscScalar *u, void *ctx);
PetscErrorCode BCsMass(PetscInt dim, PetscReal time, const PetscReal x[],
                       PetscInt num_comp_u, PetscScalar *u, void *ctx);

#endif // bcfunctions_h
