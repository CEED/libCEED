#ifndef setupdm_h
#define setupdm_h

#include <petsc.h>
#include <petscdmplex.h>
#include <petscsys.h>
#include <ceed.h>
#include "structs.h"

// ---------------------------------------------------------------------------
// Setup DM
// ---------------------------------------------------------------------------
PetscErrorCode CreateDM(MPI_Comm comm, MatType mat_type,
                        VecType vec_type, DM *dm);
PetscErrorCode PerturbVerticesSmooth(DM dm);
PetscErrorCode PerturbVerticesRandom(DM dm);

#endif // setupdm_h
