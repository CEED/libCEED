#ifndef setup_boundary_h
#define setup_boundary_h

#include <ceed.h>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscsys.h>

#include "structs.h"

PetscErrorCode CreateBCLabel(DM dm, const char name[]);
PetscErrorCode DMAddBoundariesDirichlet(DM dm);
PetscErrorCode BoundaryDirichletMMS(PetscInt dim, PetscReal t, const PetscReal coords[], PetscInt num_comp_u, PetscScalar *u, void *ctx);
#endif  // setup_boundary_h
