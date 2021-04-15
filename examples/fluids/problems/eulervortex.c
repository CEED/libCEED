#include "../navierstokes.h"

PetscErrorCode NS_EULER_VORTEX(ProblemData *problem, void *setup_ctx,
                               void *ctx, void *phys) {
  PetscInt ierr;
  MPI_Comm comm = PETSC_COMM_WORLD;
  EulerTestType euler_test;
  PetscBool implicit;
  PetscBool has_current_time = PETSC_TRUE;
  PetscBool has_neumann = PETSC_TRUE;
  SetupContext setup_context = *(SetupContext *)setup_ctx;
  Units units = *(Units *)ctx;
  Physics phys_ctx = *(Physics *)phys;
  ierr = PetscMalloc1(1, &phys_ctx->euler_ctx); CHKERRQ(ierr);


  PetscFunctionBeginUser;
  // ------------------------------------------------------
  //               SET UP DENSITY_CURRENT
  // ------------------------------------------------------
  problem->dim                       = 3;
  problem->q_data_size_vol              = 10;
  problem->q_data_size_sur              = 4;
  problem->setup_vol                  = Setup;
  problem->setup_vol_loc              = Setup_loc;
  problem->setup_sur                  = SetupBoundary;
  problem->setup_sur_loc              = SetupBoundary_loc;
  problem->ics                       = ICsEuler;
  problem->ics_loc                   = ICsEuler_loc;
  problem->apply_vol_rhs              = Euler;
  problem->apply_vol_rhs_loc          = Euler_loc;
  problem->apply_vol_ifunction        = IFunction_Euler;
  problem->apply_vol_ifunction_loc    = IFunction_Euler_loc;
  problem->apply_sur                  = Euler_Sur;
  problem->apply_sur_loc              = Euler_Sur_loc;
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
                          EulerTestTypes, (PetscEnum)(euler_test = EULER_TEST_NONE),
                          (PetscEnum *)&euler_test, NULL); CHKERRQ(ierr);

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
  setup_context->lx         = lx;
  setup_context->ly         = ly;
  setup_context->lz         = lz;
  setup_context->center[0]  = center[0];
  setup_context->center[1]  = center[1];
  setup_context->center[2]  = center[2];
  setup_context->time = 0;

  // -- QFunction Context
  phys_ctx->euler_test = euler_test;
  phys_ctx->implicit = implicit;
  phys_ctx->has_current_time = has_current_time;
  phys_ctx->has_neumann = has_neumann;
  phys_ctx->euler_ctx->time = 0.; // todo: check if really needed
  phys_ctx->euler_ctx->currentTime = 0.;
  phys_ctx->euler_ctx->center[0]  = center[0];
  phys_ctx->euler_ctx->center[1]  = center[1];
  phys_ctx->euler_ctx->center[2]  = center[2];
  phys_ctx->euler_ctx->etv_mean_velocity[0] = etv_mean_velocity[0];
  phys_ctx->euler_ctx->etv_mean_velocity[1] = etv_mean_velocity[1];
  phys_ctx->euler_ctx->etv_mean_velocity[2] = etv_mean_velocity[2];
  phys_ctx->euler_ctx->vortex_strength = vortex_strength;
  phys_ctx->euler_ctx->implicit = implicit;
  phys_ctx->euler_ctx->euler_test = euler_test;

  PetscFunctionReturn(0);
}

PetscErrorCode BC_EULER_VORTEX(DM dm, SimpleBC bc, Physics phys,
                               void *setup_ctx) {

  PetscErrorCode ierr;
  PetscInt len;
  PetscBool flg;
  MPI_Comm comm = PETSC_COMM_WORLD;

  // Default boundary conditions
  bc->num_wall = bc->num_slip[0] = bc->num_slip[1] = 0;
  bc->num_slip[2] = 2;
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
    //   zero velocity and zero flux for mass density and energy density
    PetscInt comps[3] = {1, 2, 3};
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", "Face Sets", 0,
                         3, comps, (void(*)(void))Exact_Euler, NULL,
                         bc->num_wall, bc->walls, setup_ctx); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
