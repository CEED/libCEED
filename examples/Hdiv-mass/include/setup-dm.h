#ifndef setupdm_h
#define setupdm_h

#include <petsc.h>
#include <petscdmplex.h>
#include <petscsys.h>
#include <ceed.h>
#include "../include/structs.h"

// ---------------------------------------------------------------------------
// Set-up DM
// ---------------------------------------------------------------------------
PetscErrorCode CreateDistributedDM(MPI_Comm comm, ProblemData *problem_data,
                                   DM *dm);

#endif // setupdm_h
