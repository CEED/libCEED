#include "../navierstokes.h"

PetscErrorCode NS_ADVECTION2D(ProblemData *problem, void *setup_ctx,
                              void *ctx, void *phys) {
  PetscInt ierr;
  MPI_Comm comm = PETSC_COMM_WORLD;
  WindType wind_type;
  StabilizationType stab;
  PetscBool implicit;
  PetscBool has_current_time = PETSC_FALSE;
  SetupContext setup_context = *(SetupContext *)setup_ctx;
  Units units = *(Units *)ctx;
  Physics phys_ctx = *(Physics *)phys;
  ierr = PetscMalloc1(1, &phys_ctx->advection_ctx); CHKERRQ(ierr);

  PetscFunctionBeginUser;
  // ------------------------------------------------------
  //               SET UP ADVECTION2D
  // ------------------------------------------------------
  problem->dim                       = 2;
  problem->q_data_size_vol              = 5;
  problem->q_data_size_sur              = 3;
  problem->setup_vol                  = Setup2d;
  problem->setup_vol_loc              = Setup2d_loc;
  problem->setup_sur                  = SetupBoundary2d;
  problem->setup_sur_loc              = SetupBoundary2d_loc;
  problem->ics                       = ICsAdvection2d;
  problem->ics_loc                   = ICsAdvection2d_loc;
  problem->apply_vol_rhs              = Advection2d;
  problem->apply_vol_rhs_loc          = Advection2d_loc;
  problem->apply_vol_ifunction        = IFunction_Advection2d;
  problem->apply_vol_ifunction_loc    = IFunction_Advection2d_loc;
  problem->apply_sur                  = Advection2d_Sur;
  problem->apply_sur_loc              = Advection2d_Sur_loc;
  problem->bc                        = Exact_Advection2d;
  problem->bc_fnc                    = BC_ADVECTION2D;
  problem->non_zero_time             = PETSC_TRUE;

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
  ierr = PetscOptionsBegin(comm, NULL, "Options for ADVECTION2D problem",
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
  if (translation) phys_ctx->has_neumann = translation;
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
  setup_context->rc         = rc;
  setup_context->lx         = lx;
  setup_context->ly         = ly;
  setup_context->lz         = lz;
  setup_context->wind[0]  = wind[0];
  setup_context->wind[1]  = wind[1];
  setup_context->wind_type = wind_type;
  setup_context->time = 0;

  // -- QFunction Context
  phys_ctx->stab = stab;
  phys_ctx->wind_type = wind_type;
  phys_ctx->implicit = implicit;  // todo: check
  phys_ctx->has_current_time = has_current_time;
  phys_ctx->advection_ctx->CtauS = CtauS;
  phys_ctx->advection_ctx->strong_form = strong_form;
  phys_ctx->advection_ctx->E_wind = E_wind;
  phys_ctx->advection_ctx->implicit = implicit;
  phys_ctx->advection_ctx->stabilization = stab;

  PetscFunctionReturn(0);
}

PetscErrorCode BC_ADVECTION2D(DM dm, SimpleBC bc, Physics phys,
                              void *setup_ctx) {

  PetscErrorCode ierr;
  PetscInt len;
  PetscBool flg;
  MPI_Comm comm = PETSC_COMM_WORLD;

  // Default boundary conditions
  // ToDo: fix the dimension
  // todo: check if we can define bcs for translation in
  //       function instead
  bc->num_slip[0] = bc->num_slip[1] = bc->num_slip[2] = 2;
  bc->slips[0][0] = 5;
  bc->slips[0][1] = 6;
  bc->slips[1][0] = 3;
  bc->slips[1][1] = 4;
  bc->slips[2][0] = 1;
  bc->slips[2][1] = 2;

  PetscFunctionBeginUser;
  // Parse command line options
  ierr = PetscOptionsBegin(comm, NULL, "Options for ADVECTION2D BCs",
                           NULL); CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-bc_wall",
                              "Use wall boundary conditions on this list of faces",
                              NULL, bc->walls,
                              (len = sizeof(bc->walls) / sizeof(bc->walls[0]),
                               &len), &flg); CHKERRQ(ierr);
  if (flg) {
    bc->num_wall = len;
    // Using a no-slip wall disables automatic slip walls (they must be set explicitly)
    bc->num_slip[0] = bc->num_slip[1] = bc->num_slip[2] = 0;
  }
  for (PetscInt j=0; j<3; j++) {
    const char *flags[3] = {"-bc_slip_x", "-bc_slip_y", "-bc_slip_z"};
    ierr = PetscOptionsIntArray(flags[j],
                                "Use slip boundary conditions on this list of faces",
                                NULL, bc->slips[j],
                                (len = sizeof(bc->slips[j]) / sizeof(bc->slips[j][0]),
                                 &len), &flg); CHKERRQ(ierr);
    if (flg) {
      bc->num_slip[j] = len;
      bc->user_bc = PETSC_TRUE;
    }
  }
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  {
    // Slip boundary conditions
    PetscInt comps[1] = {1};
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipx", "Face Sets", 0,
                         1, comps, (void(*)(void))NULL, NULL, bc->num_slip[0],
                         bc->slips[0], setup_ctx); CHKERRQ(ierr);
    comps[0] = 2;
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipy", "Face Sets", 0,
                         1, comps, (void(*)(void))NULL, NULL, bc->num_slip[1],
                         bc->slips[1], setup_ctx); CHKERRQ(ierr);
    comps[0] = 3;
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipz", "Face Sets", 0,
                         1, comps, (void(*)(void))NULL, NULL, bc->num_slip[2],
                         bc->slips[2], setup_ctx); CHKERRQ(ierr);
  }
  if (bc->user_bc == PETSC_TRUE) {
    for (PetscInt c = 0; c < 3; c++) {
      for (PetscInt s = 0; s < bc->num_slip[c]; s++) {
        for (PetscInt w = 0; w < bc->num_wall; w++) {
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
                         1, comps, (void(*)(void))Exact_Advection2d, NULL,
                         bc->num_wall, bc->walls, setup_ctx); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
