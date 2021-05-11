#include "../navierstokes.h"

PetscErrorCode NS_EULER_VORTEX(ProblemData *problem, void *setup_ctx,
                               void *ctx) {
  EulerTestType euler_test;
  SetupContext  setup_context = *(SetupContext *)setup_ctx;
  User          user = *(User *)ctx;
  MPI_Comm      comm = PETSC_COMM_WORLD;
  PetscBool     implicit;
  PetscBool     has_curr_time = PETSC_TRUE;
  PetscBool     has_neumann = PETSC_TRUE;
  PetscInt      ierr;
  PetscFunctionBeginUser;

  ierr = PetscCalloc1(1, &user->phys->euler_ctx); CHKERRQ(ierr);

  // ------------------------------------------------------
  //               SET UP DENSITY_CURRENT
  // ------------------------------------------------------
  problem->dim                     = 3;
  problem->q_data_size_vol         = 10;
  problem->q_data_size_sur         = 4;
  problem->setup_vol               = Setup;
  problem->setup_vol_loc           = Setup_loc;
  problem->setup_sur               = SetupBoundary;
  problem->setup_sur_loc           = SetupBoundary_loc;
  problem->ics                     = ICsEuler;
  problem->ics_loc                 = ICsEuler_loc;
  problem->apply_vol_rhs           = Euler;
  problem->apply_vol_rhs_loc       = Euler_loc;
  problem->apply_vol_ifunction     = IFunction_Euler;
  problem->apply_vol_ifunction_loc = IFunction_Euler_loc;
  problem->apply_sur               = Euler_Sur;
  problem->apply_sur_loc           = Euler_Sur_loc;
  problem->bc                      = Exact_Euler;
  problem->bc_func                 = BC_EULER_VORTEX;
  problem->non_zero_time           = PETSC_TRUE;
  problem->print_info              = PRINT_EULER_VORTEX;

  // ------------------------------------------------------
  //             Create the libCEED context
  // ------------------------------------------------------
  CeedScalar curr_time       = 0.;    // s
  CeedScalar vortex_strength = 5.;    // -
  PetscScalar lx             = 1000.; // m
  PetscScalar ly             = 1000.; // m
  PetscScalar lz             = 1.;    // m
  PetscReal center[3], mean_velocity[3] = {1., 1., 0};

  // ------------------------------------------------------
  //             Create the PETSc context
  // ------------------------------------------------------
  PetscScalar meter    = 1e-2; // 1 meter in scaled length units
  PetscScalar second   = 1e-2; // 1 second in scaled time units

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  ierr = PetscOptionsBegin(comm, NULL, "Options for EULER_VORTEX problem",
                           NULL); CHKERRQ(ierr);
  // -- Physics
  PetscBool user_vortex;
  ierr = PetscOptionsScalar("-vortex_strength", "Strength of Vortex",
                            NULL, vortex_strength, &vortex_strength, &user_vortex);
  CHKERRQ(ierr);
  PetscInt n = problem->dim;
  ierr = PetscOptionsRealArray("-mean_velocity", "Background velocity vector",
                               NULL, mean_velocity, &n, NULL);
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
  ierr = PetscOptionsRealArray("-center", "Location of vortex center",
                               NULL, center, &n, NULL); CHKERRQ(ierr);
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
  user->units->meter  = meter;
  user->units->second = second;

  // ------------------------------------------------------
  //           Set up the libCEED context
  // ------------------------------------------------------
  // -- Scale variables to desired units
  lx = fabs(lx) * meter;
  ly = fabs(ly) * meter;
  lz = fabs(lz) * meter;
  for (int i=0; i<3; i++) center[i] *= meter;
  // todo: scale mean_velocity

  // -- Setup Context
  setup_context->lx        = lx;
  setup_context->ly        = ly;
  setup_context->lz        = lz;
  setup_context->center[0] = center[0];
  setup_context->center[1] = center[1];
  setup_context->center[2] = center[2];
  setup_context->time      = 0;

  // -- QFunction Context
  user->phys->euler_test                  = euler_test;
  user->phys->implicit                    = implicit;
  user->phys->has_curr_time               = has_curr_time;
  user->phys->has_neumann                 = has_neumann;
  user->phys->euler_ctx->curr_time        = 0.;
  user->phys->euler_ctx->implicit         = implicit;
  user->phys->euler_ctx->euler_test       = euler_test;
  user->phys->euler_ctx->center[0]        = center[0];
  user->phys->euler_ctx->center[1]        = center[1];
  user->phys->euler_ctx->center[2]        = center[2];
  user->phys->euler_ctx->vortex_strength  = vortex_strength;
  user->phys->euler_ctx->mean_velocity[0] = mean_velocity[0];
  user->phys->euler_ctx->mean_velocity[1] = mean_velocity[1];
  user->phys->euler_ctx->mean_velocity[2] = mean_velocity[2];

  PetscFunctionReturn(0);
}

PetscErrorCode BC_EULER_VORTEX(DM dm, SimpleBC bc, Physics phys,
                               void *setup_ctx) {
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // Define boundary conditions
  bc->num_slip[2] = 2; bc->slips[2][0] = 1; bc->slips[2][1] = 2;

  // Set boundary conditions
  DMLabel label;
  ierr = DMGetLabel(dm, "Face Sets", &label); CHKERRQ(ierr);
  PetscInt comps[1] = {3};
  ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipz", label, "Face Sets",
                       bc->num_slip[2], bc->slips[2], 0, 1, comps,
                       (void(*)(void))NULL, NULL, setup_ctx, NULL);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode PRINT_EULER_VORTEX(Physics phys, SetupContext setup_ctx,
                                  AppCtx app_ctx) {
  MPI_Comm       comm = PETSC_COMM_WORLD;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  ierr = PetscPrintf(comm,
                     "  Problem:\n"
                     "    Problem Name                       : %s\n"
                     "    Test Case                          : %s\n"
                     "    Background Velocity                : %f,%f\n",
                     app_ctx->problem_name, EulerTestTypes[phys->euler_test],
                     phys->euler_ctx->mean_velocity[0],
                     phys->euler_ctx->mean_velocity[1]); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
