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
PetscErrorCode SetupFE(MPI_Comm comm, DM dm);

#endif // setupfe_h
