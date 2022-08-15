#ifndef setupfe_h
#define setupfe_h

#include <petsc.h>
#include <petscdmplex.h>
#include <petscsys.h>
#include <ceed.h>
#include "structs.h"

// ---------------------------------------------------------------------------
// Setup FE
// ---------------------------------------------------------------------------
PetscErrorCode SetupFEHdiv(MPI_Comm comm, DM dm, DM dm_u0, DM dm_p0);
PetscErrorCode SetupFEH1(ProblemData problem_data,
                         AppCtx app_ctx, DM dm_H1);
#endif // setupfe_h
