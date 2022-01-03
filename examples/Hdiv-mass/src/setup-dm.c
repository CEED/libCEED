#include "../include/setup-dm.h"

// ---------------------------------------------------------------------------
// Set-up DM
// ---------------------------------------------------------------------------
PetscErrorCode CreateDistributedDM(MPI_Comm comm, ProblemData *problem_data,
                                   DM *dm) {
  PetscErrorCode  ierr;
  PetscSection   sec;
  PetscInt       dofs_per_face;
  PetscInt       p_start, p_end;
  PetscInt       c_start, c_end; // cells
  PetscInt       f_start, f_end; // faces
  PetscInt       v_start, v_end; // vertices

  PetscFunctionBeginUser;

  // Create DMPLEX
  ierr = DMCreate(comm, dm); CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX); CHKERRQ(ierr);
  // Set Tensor elements
  ierr = PetscOptionsSetValue(NULL, "-dm_plex_simplex", "0"); CHKERRQ(ierr);
  // Set CL options
  ierr = DMSetFromOptions(*dm); CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view"); CHKERRQ(ierr);

  // Get plex limits
  ierr = DMPlexGetChart(*dm, &p_start, &p_end); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(*dm, 0, &c_start, &c_end); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(*dm, 1, &f_start, &f_end); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(*dm, 0, &v_start, &v_end); CHKERRQ(ierr);
  // Create section
  ierr = PetscSectionCreate(comm, &sec); CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(sec,1); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(sec,0,"Velocity"); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(sec,0,1); CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sec,p_start,p_end); CHKERRQ(ierr);
  // Setup dofs per face
  for (PetscInt f = f_start; f < f_end; f++) {
    ierr = DMPlexGetConeSize(*dm, f, &dofs_per_face); CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(sec, f, 0, dofs_per_face); CHKERRQ(ierr);
    ierr = PetscSectionSetDof     (sec, f, dofs_per_face); CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(sec); CHKERRQ(ierr);
  ierr = DMSetSection(*dm,sec); CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sec); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};