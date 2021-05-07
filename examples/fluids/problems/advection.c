#include "../navierstokes.h"

PetscErrorCode NS_ADVECTION(ProblemData *problem, void *setup_ctx,
                            void *ctx, void *phys) {
  WindType             wind_type;
  BubbleDimType        bubble_dim_type;
  BubbleContinuityType bubble_continuity_type;
  StabilizationType    stab;
  SetupContext         setup_context = *(SetupContext *)setup_ctx;
  Units                units = *(Units *)ctx;
  Physics              phys_ctx = *(Physics *)phys;
  MPI_Comm             comm = PETSC_COMM_WORLD;
  PetscBool            implicit;
  PetscBool            has_current_time = PETSC_FALSE;
  PetscInt             ierr;
  PetscFunctionBeginUser;

  ierr = PetscCalloc1(1, &phys_ctx->advection_ctx); CHKERRQ(ierr);

  // ------------------------------------------------------
  //               SET UP ADVECTION
  // ------------------------------------------------------
  problem->dim                     = 3;
  problem->q_data_size_vol         = 10;
  problem->q_data_size_sur         = 4;
  problem->setup_vol               = Setup;
  problem->setup_vol_loc           = Setup_loc;
  problem->setup_sur               = SetupBoundary;
  problem->setup_sur_loc           = SetupBoundary_loc;
  problem->ics                     = ICsAdvection;
  problem->ics_loc                 = ICsAdvection_loc;
  problem->apply_vol_rhs           = Advection;
  problem->apply_vol_rhs_loc       = Advection_loc;
  problem->apply_vol_ifunction     = IFunction_Advection;
  problem->apply_vol_ifunction_loc = IFunction_Advection_loc;
  problem->apply_sur               = Advection_Sur;
  problem->apply_sur_loc           = Advection_Sur_loc;
  problem->bc                      = Exact_Advection;
  problem->bc_func                 = BC_ADVECTION;
  problem->non_zero_time           = PETSC_FALSE;

  // ------------------------------------------------------
  //             Create the libCEED context
  // ------------------------------------------------------
  PetscScalar lx         = 8000.;      // m
  PetscScalar ly         = 8000.;      // m
  PetscScalar lz         = 4000.;      // m
  CeedScalar rc          = 1000.;      // m (Radius of bubble)
  CeedScalar CtauS       = 0.;         // dimensionless
  CeedScalar strong_form = 0.;         // [0,1]
  CeedScalar E_wind      = 1.e6;       // J
  PetscReal wind[3]      = {1., 0, 0}; // m/s

  // ------------------------------------------------------
  //             Create the PETSc context
  // ------------------------------------------------------
  PetscScalar meter    = 1e-2; // 1 meter in scaled length units
  PetscScalar kilogram = 1e-6; // 1 kilogram in scaled mass units
  PetscScalar second   = 1e-2; // 1 second in scaled time units
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
  PetscBool user_wind;
  ierr = PetscOptionsRealArray("-problem_advection_wind_translation",
                               "Constant wind vector",
                               NULL, wind, &n, &user_wind); CHKERRQ(ierr);
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
  ierr = PetscOptionsEnum("-bubble_dim", "Bubble dimension",
                          NULL, BubbleDimTypes,
                          (PetscEnum)(bubble_dim_type = ADVECTION_BUBBLE_DIM_SPHERE),
                          (PetscEnum *)&bubble_dim_type, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-bubble_continuity", "Bubble continuity",
                          NULL, BubbleContinuityTypes,
                          (PetscEnum)(bubble_continuity_type = ADVECTION_BUBBLE_CONTINUITY_SMOOTH),
                          (PetscEnum *)&bubble_continuity_type, NULL); CHKERRQ(ierr);
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
  if (wind_type == ADVECTION_WIND_ROTATION && user_wind) {
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

  units->meter    = meter;
  units->kilogram = kilogram;
  units->second   = second;
  units->Joule    = Joule;

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
  setup_context->rc                     = rc;
  setup_context->lx                     = lx;
  setup_context->ly                     = ly;
  setup_context->lz                     = lz;
  setup_context->wind[0]                = wind[0];
  setup_context->wind[1]                = wind[1];
  setup_context->wind[2]                = wind[2];
  setup_context->wind_type              = wind_type;
  setup_context->bubble_dim_type        = bubble_dim_type;
  setup_context->bubble_continuity_type = bubble_continuity_type;
  setup_context->time = 0;

  // -- QFunction Context
  phys_ctx->stab                         = stab;
  phys_ctx->wind_type                    = wind_type;
  phys_ctx->bubble_dim_type              = bubble_dim_type;
  phys_ctx->bubble_continuity_type       = bubble_continuity_type;
  //  if passed correctly
  phys_ctx->implicit                     = implicit;
  phys_ctx->has_current_time             = has_current_time;
  phys_ctx->advection_ctx->CtauS         = CtauS;
  phys_ctx->advection_ctx->E_wind        = E_wind;
  phys_ctx->advection_ctx->implicit      = implicit;
  phys_ctx->advection_ctx->strong_form   = strong_form;
  phys_ctx->advection_ctx->stabilization = stab;

  PetscFunctionReturn(0);
}

PetscErrorCode BC_ADVECTION(DM dm, SimpleBC bc, Physics phys,
                            void *setup_ctx) {
  PetscInt       len;
  PetscBool      flg;
  MPI_Comm       comm = PETSC_COMM_WORLD;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // Define boundary conditions
  if (phys->wind_type == ADVECTION_WIND_TRANSLATION) {
    bc->num_slip[0] = bc->num_slip[1] = bc->num_slip[2] = 2;
    bc->slips[0][0] = 5;
    bc->slips[0][1] = 6;
    bc->slips[1][0] = 3;
    bc->slips[1][1] = 4;
    bc->slips[2][0] = 1;
    bc->slips[2][1] = 2;
  } else if (phys->wind_type == ADVECTION_WIND_ROTATION &&
             phys->bubble_dim_type == ADVECTION_BUBBLE_DIM_CYLINDER) {
    bc->num_slip[0] = bc->num_slip[1] = 0;
    bc->num_slip[2] = 2; bc->slips[2][0] = 1; bc->slips[2][1] = 2;
    bc->num_wall = 6;
    bc->walls[0] = 3; bc->walls[1] = 4; bc->walls[2] = 5; bc->walls[3] = 6;
  } else {
    bc->num_slip[0] = bc->num_slip[1] = bc->num_slip[2] = 0;
    bc->num_wall = 6;
    bc->walls[0] = 1; bc->walls[1] = 2; bc->walls[2] = 3;
    bc->walls[3] = 4; bc->walls[4] = 5; bc->walls[5] = 6;
  }

  {
    // Slip boundary conditions
    DMLabel label;
    ierr = DMGetLabel(dm, "Face Sets", &label); CHKERRQ(ierr);
    PetscInt comps[1] = {1};
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipx", label, "Face Sets",
                         bc->num_slip[0], bc->slips[0], 0, 1, comps,
                         (void(*)(void))NULL, NULL, setup_ctx, NULL);
    CHKERRQ(ierr);
    comps[0] = 2;
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipy", label, "Face Sets",
                         bc->num_slip[1], bc->slips[1], 0, 1, comps,
                         (void(*)(void))NULL, NULL, setup_ctx, NULL);
    CHKERRQ(ierr);
    comps[0] = 3;
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipz", label, "Face Sets",
                         bc->num_slip[2], bc->slips[2], 0, 1, comps,
                         (void(*)(void))NULL, NULL, setup_ctx, NULL);
    CHKERRQ(ierr);
  }

  // Wall boundary conditions
  //   zero energy density and zero flux
  {
    DMLabel  label;
    PetscInt comps[1] = {4};
    ierr = DMGetLabel(dm, "Face Sets", &label); CHKERRQ(ierr);
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, "Face Sets",
                         bc->num_wall, bc->walls, 0,
                         1, comps, (void(*)(void))Exact_Advection, NULL,
                         setup_ctx, NULL); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
