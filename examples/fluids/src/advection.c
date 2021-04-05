#include "../navierstokes.h"

PetscErrorCode NS_ADVECTION(problemData *problem, void *ctxSetupData,
                            void *ctx, void *ctxPhys) {
  PetscInt ierr;
  MPI_Comm comm = PETSC_COMM_WORLD;
  WindType wind_type;
  StabilizationType stab;
  PetscBool implicit;
  PetscBool hasCurrentTime = PETSC_FALSE;
  SetupContext ctxSetup = *(SetupContext *)ctxSetupData;
  Units units = *(Units *)ctx;
  Physics ctxPhysData = *(Physics *)ctxPhys;
  ierr = PetscMalloc1(1, &ctxPhysData->ctxAdvectionData); CHKERRQ(ierr);

  PetscFunctionBeginUser;
  // ------------------------------------------------------
  //               SET UP ADVECTION
  // ------------------------------------------------------
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
  problem->bc                        = Exact_Advection;
  problem->bc_fnc                    = BC_ADVECTION;
  problem->non_zero_time             = PETSC_FALSE;

  // ------------------------------------------------------
  //             Create the libCEED context
  // ------------------------------------------------------
  PetscScalar lx         = 8000.;       // m
  PetscScalar ly         = 8000.;       // m
  PetscScalar lz         = 4000.;       // m
  CeedScalar rc          = 1000.;       // m (Radius of bubble)
  CeedScalar CtauS       = 0.;          // dimensionless
  CeedScalar strong_form = 0.;          // [0,1]
  CeedScalar E_wind      = 1.e6;        // J
  PetscReal wind[3]      = {1., 0, 0};  // m/s

  // ------------------------------------------------------
  //             Create the PETSc context
  // ------------------------------------------------------
  PetscScalar meter    = 1e-2;  // 1 meter in scaled length units
  PetscScalar kilogram = 1e-6;  // 1 kilogram in scaled mass units
  PetscScalar second   = 1e-2;  // 1 second in scaled time units
  PetscScalar Joule;

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  ierr = PetscOptionsBegin(comm, NULL, "Options for ADVECTION problem",
                           NULL); CHKERRQ(ierr);
  // -- Physics
  ierr = PetscOptionsScalar("-lx", "Length scale in x direction",
                            NULL, lx, &lx, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-ly", "Length scale in y direction",
                            NULL, ly, &ly, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-lz", "Length scale in z direction",
                            NULL, lz, &lz, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-rc", "Characteristic radius of thermal bubble",
                            NULL, rc, &rc, NULL); CHKERRQ(ierr);
  PetscInt n = problem->dim;
  PetscBool userWind;
  ierr = PetscOptionsRealArray("-problem_advection_wind_translation",
                               "Constant wind vector",
                               NULL, wind, &n, &userWind); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-CtauS",
                            "Scale coefficient for tau (nondimensional)",
                            NULL, CtauS, &CtauS, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-strong_form",
                            "Strong (1) or weak/integrated by parts (0) advection residual",
                            NULL, strong_form, &strong_form, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-E_wind", "Total energy of inflow wind",
                            NULL, E_wind, &E_wind, NULL); CHKERRQ(ierr);
  PetscBool translation;
  ierr = PetscOptionsEnum("-problem_advection_wind", "Wind type in Advection",
                          NULL, WindTypes,
                          (PetscEnum)(wind_type = ADVECTION_WIND_ROTATION),
                          (PetscEnum *)&wind_type, &translation); CHKERRQ(ierr);
  if (translation) ctxPhysData->hasNeumann = translation;
  ierr = PetscOptionsEnum("-stab", "Stabilization method", NULL,
                          StabilizationTypes, (PetscEnum)(stab = STAB_NONE),
                          (PetscEnum *)&stab, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-implicit", "Use implicit (IFunction) formulation",
                          NULL, implicit=PETSC_FALSE, &implicit, NULL);
  CHKERRQ(ierr);

  // -- Units
  ierr = PetscOptionsScalar("-units_meter", "1 meter in scaled length units",
                            NULL, meter, &meter, NULL); CHKERRQ(ierr);
  meter = fabs(meter);
  ierr = PetscOptionsScalar("-units_kilogram","1 kilogram in scaled mass units",
                            NULL, kilogram, &kilogram, NULL); CHKERRQ(ierr);
  kilogram = fabs(kilogram);
  ierr = PetscOptionsScalar("-units_second","1 second in scaled time units",
                            NULL, second, &second, NULL); CHKERRQ(ierr);
  second = fabs(second);

  // -- Warnings
  if (wind_type == ADVECTION_WIND_ROTATION && userWind) {
    ierr = PetscPrintf(comm,
                       "Warning! Use -problem_advection_wind_translation only with -problem_advection_wind translation\n");
    CHKERRQ(ierr);
  }
  if (stab == STAB_NONE && CtauS != 0) {
    ierr = PetscPrintf(comm,
                       "Warning! Use -CtauS only with -stab su or -stab supg\n");
    CHKERRQ(ierr);
  }
  if (stab == STAB_SUPG && !implicit) {
    ierr = PetscPrintf(comm,
                       "Warning! Use -stab supg only with -implicit\n");
    CHKERRQ(ierr);
  }

  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // ------------------------------------------------------
  //           Set up the PETSc context
  // ------------------------------------------------------
  // -- Define derived units
  Joule = kilogram * PetscSqr(meter) / PetscSqr(second);

  units->meter = meter;
  units->kilogram = kilogram;
  units->second = second;
  units->Joule = Joule;

  // ------------------------------------------------------
  //           Set up the libCEED context
  // ------------------------------------------------------
  // -- Scale variables to desired units
  E_wind *= Joule;
  lx = fabs(lx) * meter;
  ly = fabs(ly) * meter;
  lz = fabs(lz) * meter;
  rc = fabs(rc) * meter;

  // -- Setup Context
  ctxSetup->rc         = rc;
  ctxSetup->lx         = lx;
  ctxSetup->ly         = ly;
  ctxSetup->lz         = lz;
  ctxSetup->wind[0]  = wind[0];
  ctxSetup->wind[1]  = wind[1];
  ctxSetup->wind[2]  = wind[2];
  ctxSetup->wind_type = wind_type;
  ctxSetup->time = 0;

  // -- QFunction Context
  ctxPhysData->stab = stab;
  ctxPhysData->wind_type = wind_type; // to-do: check if passed correctly
  ctxPhysData->implicit = implicit;
  ctxPhysData->hasCurrentTime = hasCurrentTime;
  ctxPhysData->ctxAdvectionData->CtauS = CtauS;
  ctxPhysData->ctxAdvectionData->strong_form = strong_form;
  ctxPhysData->ctxAdvectionData->E_wind = E_wind;
  ctxPhysData->ctxAdvectionData->implicit = implicit;
  ctxPhysData->ctxAdvectionData->stabilization = stab;

  PetscFunctionReturn(0);
}

PetscErrorCode BC_ADVECTION(DM dm, SimpleBC bc, Physics phys,
                            void *ctxSetupData) {

  PetscErrorCode ierr;
  PetscInt len;
  PetscBool flg;
  MPI_Comm comm = PETSC_COMM_WORLD;

  // Default boundary conditions
  bc->nslip[0] = bc->nslip[1] = bc->nslip[2] = 2;
  bc->slips[0][0] = 5;
  bc->slips[0][1] = 6;
  bc->slips[1][0] = 3;
  bc->slips[1][1] = 4;
  bc->slips[2][0] = 1;
  bc->slips[2][1] = 2;

  PetscFunctionBeginUser;
  // Parse command line options
  ierr = PetscOptionsBegin(comm, NULL, "Options for ADVECTION BCs",
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
    //   zero energy density and zero flux
    //   ToDo: need to set "wall" as the default BC after checking with the regression tests
    PetscInt comps[1] = {4};
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", "Face Sets", 0,
                         1, comps, (void(*)(void))Exact_Advection, NULL,
                         bc->nwall, bc->walls, ctxSetupData); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
