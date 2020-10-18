#include "navierstokes.h"

PetscErrorCode NS_ADVECTION2D(problemData *problem) {

  PetscFunctionBeginUser;
  problem->dim                       = 2;
  problem->qdatasizeVol              = 5;
  problem->qdatasizeSur              = 3;
  problem->setupVol                  = Setup2d;
  problem->setupVol_loc              = Setup2d_loc;
  problem->setupSur                  = SetupBoundary2d;
  problem->setupSur_loc              = SetupBoundary2d_loc;
  problem->ics                       = ICsAdvection2d;
  problem->ics_loc                   = ICsAdvection2d_loc;
  problem->applyVol_rhs              = Advection2d;
  problem->applyVol_rhs_loc          = Advection2d_loc;
  problem->applyVol_ifunction        = IFunction_Advection2d;
  problem->applyVol_ifunction_loc    = IFunction_Advection2d_loc;
  problem->applySur                  = Advection2d_Sur;
  problem->applySur_loc              = Advection2d_Sur_loc;
  problem->bc_func                   = BC_ADVECTION2D;
  problem->non_zero_time             = PETSC_TRUE;
  PetscFunctionReturn(0);
}

// Wall boundary conditions are zero energy density and zero flux.
PetscErrorCode BC_ADVECTION2D(DM dm, MPI_Comm comm, SimpleBC bc,
                              void *ctxSetupData) {
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  PetscInt comps[1] = {4};
  ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", "Face Sets", 0,
                       1, comps, (void(*)(void))Exact_Advection2d, NULL,
                       bc->nwall, bc->walls, ctxSetupData); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}