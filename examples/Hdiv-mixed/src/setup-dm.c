#include "../include/setup-dm.h"

// ---------------------------------------------------------------------------
// Set-up DM
// ---------------------------------------------------------------------------
PetscErrorCode CreateDistributedDM(MPI_Comm comm, DM *dm) {
  PetscErrorCode  ierr;
  PetscSection   sec;
  PetscBool      interpolate = PETSC_TRUE;
  PetscInt       nx = 1, ny = 2;
  PetscInt       faces[2] = {nx, ny};
  PetscInt       dim = 2, num_comp_u = dim;
  PetscInt       p_start, p_end;
  PetscInt       c_start, c_end; // cells
  PetscInt       e_start, e_end, e; // edges
  PetscInt       v_start, v_end; // vertices

  PetscFunctionBeginUser;

  ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, faces, NULL,
                             NULL, NULL, interpolate, dm); CHKERRQ(ierr);
  // Get plex limits
  ierr = DMPlexGetChart(*dm, &p_start, &p_end); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(*dm, 0, &c_start, &c_end); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(*dm, 1, &e_start, &e_end); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(*dm, 0, &v_start, &v_end); CHKERRQ(ierr);
  // Create section
  ierr = PetscSectionCreate(comm, &sec); CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(sec,1); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(sec,0,"Velocity"); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(sec,0,1); CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sec,p_start,p_end); CHKERRQ(ierr);
  // Setup dofs
  for (e = e_start; e < e_end; e++) {
    ierr = PetscSectionSetFieldDof(sec, e, 0, num_comp_u); CHKERRQ(ierr);
    ierr = PetscSectionSetDof     (sec, e, num_comp_u); CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(sec); CHKERRQ(ierr);
  ierr = DMSetSection(*dm,sec); CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view"); CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sec); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};