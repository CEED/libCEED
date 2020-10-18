#include "navierstokes.h"

PetscErrorCode NS_ADVECTION(problemData *problem) {

  PetscFunctionBeginUser;
  problem->dim                       = 3;
  problem->qdatasizeVol              = 10;
  problem->qdatasizeSur              = 4;
  problem->setupVol                  = Setup;
  problem->setupVol_loc              = Setup_loc;
  problem->setupSur                  = SetupBoundary;
  problem->setupSur_loc              = SetupBoundary_loc;
  problem->ics                       = ICsAdvection;
  problem->ics_loc                   = ICsAdvection_loc;
  problem->applyVol_rhs              = Advection;
  problem->applyVol_rhs_loc          = Advection_loc;
  problem->applyVol_ifunction        = IFunction_Advection;
  problem->applyVol_ifunction_loc    = IFunction_Advection_loc;
  problem->applySur                  = Advection_Sur;
  problem->applySur_loc              = Advection_Sur_loc;
  problem->bc_func                   = BC_ADVECTION;
  problem->non_zero_time             = PETSC_FALSE;
  PetscFunctionReturn(0);
}

// Wall boundary conditions are zero energy density and zero flux.
PetscErrorCode BC_ADVECTION(DM dm, MPI_Comm comm, SimpleBC bc,
                            void *ctxSetupData) {
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  PetscInt comps[1] = {4};
  ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", "Face Sets", 0,
                       1, comps, (void(*)(void))Exact_Advection, NULL,
                       bc->nwall, bc->walls, ctxSetupData); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}