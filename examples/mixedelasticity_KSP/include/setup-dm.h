#ifndef setupdm_h
#define setupdm_h

#include <ceed.h>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscsys.h>

#include "structs.h"

// ---------------------------------------------------------------------------
// Create DM
// ---------------------------------------------------------------------------
PetscErrorCode CreateDM(MPI_Comm comm, Ceed ceed, DM *dm);
PetscErrorCode PerturbVerticesSmooth(DM dm);
PetscErrorCode PerturbVerticesRandom(DM dm);
#endif  // setupdm_h
