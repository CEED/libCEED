#ifndef setup_boundary_h
#define setup_boundary_h

#include <petsc.h>
#include <petscdmplex.h>
#include <petscsys.h>
#include <ceed.h>
#include "structs.h"

// ---------------------------------------------------------------------------
// Create boundary label
// ---------------------------------------------------------------------------
PetscErrorCode CreateBCLabel(DM dm, const char name[]);

// ---------------------------------------------------------------------------
// Add Dirichlet boundaries to DM
// ---------------------------------------------------------------------------
PetscErrorCode DMAddBoundariesDirichlet(DM dm);
PetscErrorCode BoundaryDirichletMMS(PetscInt dim, PetscReal t,
                                    const PetscReal coords[],
                                    PetscInt num_comp_u, PetscScalar *u, void *ctx);
PetscErrorCode DMAddBoundariesPressure(Ceed ceed, CeedData ceed_data,
                                       AppCtx app_ctx, ProblemData problem_data, DM dm,
                                       CeedVector bc_pressure);
#endif // setup_boundary_h
