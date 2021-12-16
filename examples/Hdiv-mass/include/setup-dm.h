#ifndef setupdm_h
#define setupdm_h

#include <petsc.h>
#include <petscdmplex.h>
#include <petscsys.h>
#include <ceed.h>

// ---------------------------------------------------------------------------
// Set-up DM
// ---------------------------------------------------------------------------
PetscErrorCode CreateDistributedDM(MPI_Comm comm, DM *dm);

#endif // setupdm_h
