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
  problem->bc                        = Exact_DC;
  problem->non_zero_time             = PETSC_FALSE;
  PetscFunctionReturn(0);
}
