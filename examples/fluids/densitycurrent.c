#include "navierstokes.h"

PetscErrorCode NS_DENSITY_CURRENT(problemData *problem) {

  PetscFunctionBeginUser;
  problem->dim                       = 3;
  problem->qdatasizeVol              = 10;
  problem->qdatasizeSur              = 4;
  problem->setupVol                  = Setup;
  problem->setupVol_loc              = Setup_loc;
  problem->setupSur                  = SetupBoundary;
  problem->setupSur_loc              = SetupBoundary_loc;
  problem->ics                       = ICsDC;
  problem->ics_loc                   = ICsDC_loc;
  problem->applyVol_rhs              = DC;
  problem->applyVol_rhs_loc          = DC_loc;
  problem->applyVol_ifunction        = IFunction_DC;
  problem->applyVol_ifunction_loc    = IFunction_DC_loc;
  problem->bc_func                   = BC_DENSITY_CURRENT;
  problem->non_zero_time             = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode BC_DENSITY_CURRENT(DM dm, MPI_Comm comm,
                                  SimpleBC bc, void *ctxSetupData) {

  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  PetscInt comps[3] = {1, 2, 3};
  ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", "Face Sets", 0,
                       3, comps, (void(*)(void))Exact_DC, NULL,
                       bc->nwall, bc->walls, ctxSetupData); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}