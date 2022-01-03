#include "../include/setup-dm.h"
#include "petscerror.h"

// ---------------------------------------------------------------------------
// Setup DM
// ---------------------------------------------------------------------------
PetscErrorCode CreateDM(MPI_Comm comm, VecType vec_type, DM *dm) {

  PetscFunctionBeginUser;

  // Create DMPLEX
  PetscCall( DMCreate(comm, dm) );
  PetscCall( DMSetType(*dm, DMPLEX) );
  PetscCall( DMSetVecType(*dm, vec_type) );
  // Set Tensor elements
  PetscCall( PetscOptionsSetValue(NULL, "-dm_plex_simplex", "0") );
  // Set CL options
  PetscCall( DMSetFromOptions(*dm) );
  PetscCall( DMViewFromOptions(*dm, NULL, "-dm_view") );

  PetscFunctionReturn(0);
};

PetscErrorCode PerturbVerticesSmooth(DM dm) {

  Vec          coordinates;
  PetscSection coordSection;
  PetscScalar *coords;
  PetscInt     v,vStart,vEnd,offset,dim;
  PetscReal    x,y,z;

  PetscFunctionBeginUser;

  PetscCall( DMGetDimension(dm, &dim) );
  PetscCall( DMGetCoordinateSection(dm, &coordSection) );
  PetscCall( DMGetCoordinatesLocal(dm, &coordinates) );
  PetscCall( DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd) );
  PetscCall( VecGetArray(coordinates,&coords) );
  for(v=vStart; v<vEnd; v++) {
    PetscCall( PetscSectionGetOffset(coordSection,v,&offset) );
    if(dim==2) {
      x = coords[offset]; y = coords[offset+1];
      coords[offset]   = x + 0.06*PetscSinReal(2.0*PETSC_PI*x)*PetscSinReal(
                           2.0*PETSC_PI*y);
      coords[offset+1] = y - 0.05*PetscSinReal(2.0*PETSC_PI*x)*PetscSinReal(
                           2.0*PETSC_PI*y);
    } else {
      x = coords[offset]; y = coords[offset+1]; z = coords[offset+2];
      coords[offset]   = x + 0.03*PetscSinReal(3*PETSC_PI*x)*PetscCosReal(
                           3*PETSC_PI*y)*PetscCosReal(3*PETSC_PI*z);
      coords[offset+1] = y - 0.04*PetscCosReal(3*PETSC_PI*x)*PetscSinReal(
                           3*PETSC_PI*y)*PetscCosReal(3*PETSC_PI*z);
      coords[offset+2] = z + 0.05*PetscCosReal(3*PETSC_PI*x)*PetscCosReal(
                           3*PETSC_PI*y)*PetscSinReal(3*PETSC_PI*z);
    }
  }
  PetscCall( VecRestoreArray(coordinates,&coords) );
  PetscFunctionReturn(0);
}
