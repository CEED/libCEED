#include "../navierstokes.h"

PetscErrorCode NS_EULER_VORTEX(problemData *problem, void *ctxSetupData,
                               void *ctx, void *ctxPhys) {
  PetscInt ierr;
  MPI_Comm comm = PETSC_COMM_WORLD;
  EulerTestType eulertest;
  PetscBool implicit;
  PetscBool hasCurrentTime = PETSC_TRUE;
  PetscBool hasNeumann = PETSC_TRUE;
  SetupContext ctxSetup = *(SetupContext *)ctxSetupData;
  Units units = *(Units *)ctx;
  Physics ctxPhysData = *(Physics *)ctxPhys;
  ierr = PetscMalloc1(1, &ctxPhysData->ctxEulerData); CHKERRQ(ierr);


  PetscFunctionBeginUser;
  // ------------------------------------------------------
  //               SET UP DENSITY_CURRENT
  // ------------------------------------------------------
  problem->dim                       = 3;
  problem->qdatasizeVol              = 10;
  problem->qdatasizeSur              = 4;
  problem->setupVol                  = Setup;
  problem->setupVol_loc              = Setup_loc;
  problem->setupSur                  = SetupBoundary;
  problem->setupSur_loc              = SetupBoundary_loc;
  problem->ics                       = ICsEuler;
  problem->ics_loc                   = ICsEuler_loc;
  problem->applyVol_rhs              = Euler;
  problem->applyVol_rhs_loc          = Euler_loc;
  problem->applyVol_ifunction        = IFunction_Euler;
  problem->applyVol_ifunction_loc    = IFunction_Euler_loc;
  problem->applySur                  = Euler_Sur;
  problem->applySur_loc              = Euler_Sur_loc;
  problem->bc                        = Exact_Euler;
  problem->bc_fnc                    = BC_EULER_VORTEX;
  problem->non_zero_time             = PETSC_TRUE;

  // ------------------------------------------------------
  //             Create the libCEED context
  // ------------------------------------------------------
  CeedScalar time; // todo: check if needed
  CeedScalar currentTime     = 0.;
  CeedScalar vortex_strength = 5.;      // -
  PetscScalar lx             = 8000.;       // m
  PetscScalar ly             = 8000.;       // m
  PetscScalar lz             = 4000.;       // m
  PetscReal center[3], etv_mean_velocity[3] = {1., 1., 0}; // to-do: etv -> euler

  // ------------------------------------------------------
  //             Create the PETSc context
  // ------------------------------------------------------
  PetscScalar meter    = 1e-2;  // 1 meter in scaled length units
  PetscScalar second   = 1e-2;  // 1 second in scaled time units

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  ierr = PetscOptionsBegin(comm, NULL, "Options for EULER_VORTEX problem",
                           NULL); CHKERRQ(ierr);
  // -- Physics
  PetscBool userVortex;
  ierr = PetscOptionsScalar("-vortex_strength", "Strength of Vortex",
                            NULL, vortex_strength, &vortex_strength, &userVortex);
  CHKERRQ(ierr);
  PetscInt n = problem->dim;
  ierr = PetscOptionsRealArray("-problem_euler_mean_velocity",
                               "Mean velocity vector",
                               NULL, etv_mean_velocity, &n, NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-lx", "Length scale in x direction",
                            NULL, lx, &lx, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-ly", "Length scale in y direction",
                            NULL, ly, &ly, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-lz", "Length scale in z direction",
                            NULL, lz, &lz, NULL); CHKERRQ(ierr);
  n = problem->dim;
  center[0] = 0.5 * lx;
  center[1] = 0.5 * ly;
  center[2] = 0.5 * lz;

  ierr = PetscOptionsBool("-implicit", "Use implicit (IFunction) formulation",
                          NULL, implicit=PETSC_FALSE, &implicit, NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-euler_test", "Euler test option", NULL,
                          EulerTestTypes, (PetscEnum)(eulertest = EULER_TEST_NONE),
                          (PetscEnum *)&eulertest, NULL); CHKERRQ(ierr);

  // -- Units
  ierr = PetscOptionsScalar("-units_meter", "1 meter in scaled length units",
                            NULL, meter, &meter, NULL); CHKERRQ(ierr);
  meter = fabs(meter);
  ierr = PetscOptionsScalar("-units_second","1 second in scaled time units",
                            NULL, second, &second, NULL); CHKERRQ(ierr);
  second = fabs(second);

  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // ------------------------------------------------------
  //           Set up the PETSc context
  // ------------------------------------------------------
  units->meter = meter;
  units->second = second;

  // ------------------------------------------------------
  //           Set up the libCEED context
  // ------------------------------------------------------
  // -- Scale variables to desired units
  lx = fabs(lx) * meter;
  ly = fabs(ly) * meter;
  lz = fabs(lz) * meter;
  for (int i=0; i<3; i++) center[i] *= meter;
  // todo: scale etv_mean_velocity

  // -- Setup Context
  ctxSetup->lx         = lx;
  ctxSetup->ly         = ly;
  ctxSetup->lz         = lz;
  ctxSetup->center[0]  = center[0];
  ctxSetup->center[1]  = center[1];
  ctxSetup->center[2]  = center[2];
  ctxSetup->time = 0;

  // -- QFunction Context
  ctxPhysData->eulertest = eulertest;
  ctxPhysData->implicit = implicit;
  ctxPhysData->hasCurrentTime = hasCurrentTime;
  ctxPhysData->hasNeumann = hasNeumann;
  ctxPhysData->ctxEulerData->time = 0.; // todo: check if really needed
  ctxPhysData->ctxEulerData->currentTime = 0.;
  ctxPhysData->ctxEulerData->center[0]  = center[0];
  ctxPhysData->ctxEulerData->center[1]  = center[1];
  ctxPhysData->ctxEulerData->center[2]  = center[2];
  ctxPhysData->ctxEulerData->etv_mean_velocity[0] = etv_mean_velocity[0];
  ctxPhysData->ctxEulerData->etv_mean_velocity[1] = etv_mean_velocity[1];
  ctxPhysData->ctxEulerData->etv_mean_velocity[2] = etv_mean_velocity[2];
  ctxPhysData->ctxEulerData->vortex_strength = vortex_strength;
  ctxPhysData->ctxEulerData->implicit = implicit;
  ctxPhysData->ctxEulerData->euler_test = eulertest;

  PetscFunctionReturn(0);
}

PetscErrorCode BC_EULER_VORTEX(DM dm, SimpleBC bc, Physics phys,
                               void *ctxSetupData) {

  PetscErrorCode ierr;
  PetscInt len;
  PetscBool flg;
  MPI_Comm comm = PETSC_COMM_WORLD;

  // Default boundary conditions
  bc->nwall = bc->nslip[0] = bc->nslip[1] = 0;
  bc->nslip[2] = 2;
  bc->slips[2][0] = 1;
  bc->slips[2][1] = 2;

  PetscFunctionBeginUser;
  // Parse command line options
  ierr = PetscOptionsBegin(comm, NULL, "Options for DENSITY_CURRENT BCs ",
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
                         3, comps, (void(*)(void))Exact_Euler, NULL,
                         bc->nwall, bc->walls, ctxSetupData); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
