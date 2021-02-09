#include "navierstokes.h"

PetscErrorCode NS_DENSITY_CURRENT(problemData *problem, void **ctxSetupData) {

  MPI_Comm comm = PETSC_COMM_WORLD;
  PetscInt ierr;
  SetupContext ctxSetup = *(SetupContext *)ctxSetupData;
  SetupContext ctxSetup_;
  ierr = PetscMalloc1(1, &ctxSetup_); CHKERRQ(ierr);

  PetscFunctionBeginUser;
  // ------------------------------------------------------
  //                  SET UP PROBLEM
  // ------------------------------------------------------
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
  problem->bc                        = BC_DENSITY_CURRENT;
  problem->non_zero_time             = PETSC_FALSE;

  // ------------------------------------------------------
  //             Create the libCEED context
  // ------------------------------------------------------
  ctxSetup_->theta0      = 300.;     // K
  ctxSetup_->thetaC      = -15.;     // K
  ctxSetup_->P0          = 1.e5;     // Pa
  ctxSetup_->N           = 0.01;     // 1/s
  ctxSetup_->cv          = 717.;     // J/(kg K)
  ctxSetup_->cp          = 1004.;    // J/(kg K)
  ctxSetup_->g           = 9.81;     // m/s^2
  ctxSetup_->lx          = 8000.;    // m
  ctxSetup_->ly          = 8000.;    // m
  ctxSetup_->lz          = 4000.;    // m
  ctxSetup_->rc          = 1000.;    // m (Radius of bubble)
  ctxSetup_->dc_axis[0] = 0;
  ctxSetup_->dc_axis[1] = 0;
  ctxSetup_->dc_axis[2] = 0;

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  ierr = PetscOptionsBegin(comm, NULL, "Options for DC",
                           NULL); CHKERRQ(ierr);

  ierr = PetscOptionsScalar("-theta0", "Reference potential temperature",
                            NULL, ctxSetup_->theta0, &ctxSetup_->theta0, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-thetaC", "Perturbation of potential temperature",
                            NULL, ctxSetup_->thetaC, &ctxSetup_->thetaC, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-P0", "Atmospheric pressure",
                            NULL, ctxSetup_->P0, &ctxSetup_->P0, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-N", "Brunt-Vaisala frequency",
                            NULL, ctxSetup_->N, &ctxSetup_->N, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-cv", "Heat capacity at constant volume",
                            NULL, ctxSetup_->cv, &ctxSetup_->cv, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-cp", "Heat capacity at constant pressure",
                            NULL, ctxSetup_->cp, &ctxSetup_->cp, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-g", "Gravitational acceleration",
                            NULL, ctxSetup_->g, &ctxSetup_->g, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-lx", "Length scale in x direction",
                            NULL, ctxSetup_->lx, &ctxSetup_->lx, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-ly", "Length scale in y direction",
                            NULL, ctxSetup_->ly, &ctxSetup_->ly, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-lz", "Length scale in z direction",
                            NULL, ctxSetup_->lz, &ctxSetup_->lz, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-rc", "Characteristic radius of thermal bubble",
                            NULL, ctxSetup_->rc, &ctxSetup_->rc, NULL); CHKERRQ(ierr);
  PetscInt n = problem->dim;
  ctxSetup_->center[0] = 0.5 * ctxSetup_->lx;
  ctxSetup_->center[1] = 0.5 * ctxSetup_->ly;
  ctxSetup_->center[2] = 0.5 * ctxSetup_->lz;
  ierr = PetscOptionsRealArray("-center", "Location of bubble center",
                               NULL, ctxSetup_->center, &n, NULL); CHKERRQ(ierr);
  n = problem->dim;
  ierr = PetscOptionsRealArray("-dc_axis",
                               "Axis of density current cylindrical anomaly, or {0,0,0} for spherically symmetric",
                               NULL, ctxSetup_->dc_axis, &n, NULL); CHKERRQ(ierr);
  {
    PetscReal norm = PetscSqrtReal(PetscSqr(ctxSetup_->dc_axis[0]) +
                                   PetscSqr(ctxSetup_->dc_axis[1]) +
                                   PetscSqr(ctxSetup_->dc_axis[2]));
    if (norm > 0) {
      for (int i=0; i<3; i++) ctxSetup_->dc_axis[i] /= norm;
    }
  }
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // ------------------------------------------------------
  //           Set up the libCEED context
  // ------------------------------------------------------
  ctxSetup->theta0 = ctxSetup_->theta0;
  ctxSetup->thetaC = ctxSetup_->thetaC;
  ctxSetup->P0 = ctxSetup_->P0;
  ctxSetup->N = ctxSetup_->N;
  ctxSetup->cv = ctxSetup_->cv;
  ctxSetup->cp = ctxSetup_->cp;
  ctxSetup->g = ctxSetup_->g;
  ctxSetup->rc = ctxSetup_->rc;
  ctxSetup->lx = ctxSetup_->lx;
  ctxSetup->ly = ctxSetup_->ly;
  ctxSetup->lz = ctxSetup_->lz;
  ctxSetup->center[0] = ctxSetup_->center[0];
  ctxSetup->center[1] = ctxSetup_->center[1];
  ctxSetup->center[2] = ctxSetup_->center[2];
  ctxSetup->dc_axis[0] = ctxSetup_->dc_axis[0];
  ctxSetup->dc_axis[1] = ctxSetup_->dc_axis[1];
  ctxSetup->dc_axis[2] = ctxSetup_->dc_axis[2];

  PetscFunctionReturn(0);
}

PetscErrorCode BC_DENSITY_CURRENT(DM dm, SimpleBC bc, void *ctxSetupData) {

  PetscErrorCode ierr;
  PetscInt len;
  PetscBool flg;
  MPI_Comm comm = PETSC_COMM_WORLD;

  // Default boundary conditions
  //   slip bc on all faces and no wall bc
  bc->nslip[0] = bc->nslip[1] = bc->nslip[2] = 2;
  bc->slips[0][0] = 5;
  bc->slips[0][1] = 6;
  bc->slips[1][0] = 3;
  bc->slips[1][1] = 4;
  bc->slips[2][0] = 1;
  bc->slips[2][1] = 2;

  PetscFunctionBeginUser;
  // Parse command line options
  ierr = PetscOptionsBegin(comm, NULL, "Options for density_current",
                           NULL); CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-bc_wall",
                              "Use wall boundary conditions on this list of faces",
                              NULL, bc->walls,
                              (len = sizeof(bc->walls) / sizeof(bc->walls[0]),
                               &len), &flg); CHKERRQ(ierr);
  if (flg) {
    bc->nwall = len;
    // Using a no-slip wall disables automatic slip walls (they must be set explicitly)
    bc->nslip[0] = bc->nslip[1] = bc->nslip[2] = 0;
  }
  for (PetscInt j=0; j<3; j++) {
    const char *flags[3] = {"-bc_slip_x", "-bc_slip_y", "-bc_slip_z"};
    ierr = PetscOptionsIntArray(flags[j],
                                "Use slip boundary conditions on this list of faces",
                                NULL, bc->slips[j],
                                (len = sizeof(bc->slips[j]) / sizeof(bc->slips[j][0]),
                                 &len), &flg); CHKERRQ(ierr);
    if (flg) {
      bc->nslip[j] = len;
      bc->userbc = PETSC_TRUE;
    }
  }
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  {
    // Slip boundary conditions
    PetscInt comps[1] = {1};
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipx", "Face Sets", 0,
                         1, comps, (void(*)(void))NULL, NULL, bc->nslip[0],
                         bc->slips[0], ctxSetupData); CHKERRQ(ierr);
    comps[0] = 2;
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipy", "Face Sets", 0,
                         1, comps, (void(*)(void))NULL, NULL, bc->nslip[1],
                         bc->slips[1], ctxSetupData); CHKERRQ(ierr);
    comps[0] = 3;
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipz", "Face Sets", 0,
                         1, comps, (void(*)(void))NULL, NULL, bc->nslip[2],
                         bc->slips[2], ctxSetupData); CHKERRQ(ierr);
  }
  if (bc->userbc == PETSC_TRUE) {
    for (PetscInt c = 0; c < 3; c++) {
      for (PetscInt s = 0; s < bc->nslip[c]; s++) {
        for (PetscInt w = 0; w < bc->nwall; w++) {
          if (bc->slips[c][s] == bc->walls[w])
            SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
                     "Boundary condition already set on face %D!\n",
                     bc->walls[w]);
        }
      }
    }
  }
  {
    // Wall boundary conditions
    //   zero velocity and zero flux for mass density and energy density
    PetscInt comps[3] = {1, 2, 3};
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", "Face Sets", 0,
                         3, comps, (void(*)(void))Exact_DC, NULL,
                         bc->nwall, bc->walls, ctxSetupData); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}