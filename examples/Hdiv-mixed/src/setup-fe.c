#include "../include/setup-fe.h"

// ---------------------------------------------------------------------------
// Setup FE
// ---------------------------------------------------------------------------
PetscErrorCode SetupFE(MPI_Comm comm, DM dm, DM dm_u0, DM dm_p0) {
  PetscSection   sec, sec_u0, sec_p0;
  PetscInt       dofs_per_face;
  PetscInt       p_start, p_end;
  PetscInt       c_start, c_end; // cells
  PetscInt       f_start, f_end; // faces
  PetscInt       v_start, v_end; // vertices

  PetscFunctionBeginUser;

  // Get plex limits
  PetscCall( DMPlexGetChart(dm, &p_start, &p_end) );
  PetscCall( DMPlexGetHeightStratum(dm, 0, &c_start, &c_end) );
  PetscCall( DMPlexGetHeightStratum(dm, 1, &f_start, &f_end) );
  PetscCall( DMPlexGetDepthStratum(dm, 0, &v_start, &v_end) );
  // Create section for coupled problem
  PetscCall( PetscSectionCreate(comm, &sec) );
  PetscCall( PetscSectionSetNumFields(sec, 2) );
  PetscCall( PetscSectionSetFieldName(sec, 0, "Velocity") );
  PetscCall( PetscSectionSetFieldComponents(sec, 0, 1) );
  PetscCall( PetscSectionSetFieldName(sec, 1, "Pressure") );
  PetscCall( PetscSectionSetFieldComponents(sec, 1, 1) );
  PetscCall( PetscSectionSetChart(sec, p_start, p_end) );
  // Create section for initial conditions u0
  PetscCall( PetscSectionCreate(comm, &sec_u0) );
  PetscCall( PetscSectionSetNumFields(sec_u0, 1) );
  PetscCall( PetscSectionSetFieldName(sec_u0, 0, "Velocity") );
  PetscCall( PetscSectionSetFieldComponents(sec_u0, 0, 1) );
  PetscCall( PetscSectionSetChart(sec_u0, p_start, p_end) );
  // Create section for initial conditions p0
  PetscCall( PetscSectionCreate(comm, &sec_p0) );
  PetscCall( PetscSectionSetNumFields(sec_p0, 1) );
  PetscCall( PetscSectionSetFieldName(sec_p0, 0, "Pressure") );
  PetscCall( PetscSectionSetFieldComponents(sec_p0, 0, 1) );
  PetscCall( PetscSectionSetChart(sec_p0, p_start, p_end) );
  // Setup dofs per face for velocity field
  for (PetscInt f = f_start; f < f_end; f++) {
    PetscCall( DMPlexGetConeSize(dm, f, &dofs_per_face) );
    PetscCall( PetscSectionSetFieldDof(sec, f, 0, dofs_per_face) );
    PetscCall( PetscSectionSetDof     (sec, f, dofs_per_face) );

    PetscCall( DMPlexGetConeSize(dm_u0, f, &dofs_per_face) );
    PetscCall( PetscSectionSetFieldDof(sec_u0, f, 0, dofs_per_face) );
    PetscCall( PetscSectionSetDof     (sec_u0, f, dofs_per_face) );
  }
  // Setup 1 dof per cell for pressure field
  for(PetscInt c = c_start; c < c_end; c++) {
    PetscCall( PetscSectionSetFieldDof(sec, c, 1, 1) );
    PetscCall( PetscSectionSetDof     (sec, c, 1) );

    PetscCall( PetscSectionSetFieldDof(sec_p0, c, 0, 1) );
    PetscCall( PetscSectionSetDof     (sec_p0, c, 1) );
  }
  PetscCall( PetscSectionSetUp(sec) );
  PetscCall( DMSetSection(dm,sec) );
  PetscCall( DMCreateDS(dm) );
  PetscCall( PetscSectionDestroy(&sec) );
  PetscCall( PetscSectionSetUp(sec_u0) );
  PetscCall( DMSetSection(dm_u0, sec_u0) );
  PetscCall( DMCreateDS(dm_u0) );
  PetscCall( PetscSectionDestroy(&sec_u0) );
  PetscCall( PetscSectionSetUp(sec_p0) );
  PetscCall( DMSetSection(dm_p0, sec_p0) );
  PetscCall( DMCreateDS(dm_p0) );
  PetscCall( PetscSectionDestroy(&sec_p0) );

  PetscFunctionReturn(0);
};