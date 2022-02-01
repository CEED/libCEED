// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

/// @file
/// Utility functions for setting up EULER_VORTEX

#include "../navierstokes.h"
#include "../qfunctions/setupgeo.h"
#include "../qfunctions/eulervortex.h"

PetscErrorCode NS_EULER_VORTEX(ProblemData *problem, void *setup_ctx,
                               void *ctx) {
  EulerTestType     euler_test;
  SetupContext      setup_context = *(SetupContext *)setup_ctx;
  User              user = *(User *)ctx;
  StabilizationType stab;
  MPI_Comm          comm = PETSC_COMM_WORLD;
  PetscBool         implicit;
  PetscBool         has_curr_time = PETSC_TRUE;
  PetscBool         has_neumann = PETSC_TRUE;
  PetscInt          ierr;
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
  problem->setup_ctx               = SetupContext_EULER_VORTEX;
  problem->bc_func                 = BC_EULER_VORTEX;
  problem->non_zero_time           = PETSC_TRUE;
  problem->print_info              = PRINT_EULER_VORTEX;

  // ------------------------------------------------------
  //             Create the libCEED context
  // ------------------------------------------------------
  CeedScalar vortex_strength = 5.;          // -
  CeedScalar c_tau           = 0.5;         // -
  // c_tau = 0.5 is reported as "optimal" in Hughes et al 2010
  PetscScalar lx             = 1000.;       // m
  PetscScalar ly             = 1000.;       // m
  PetscScalar lz             = 1.;          // m
  PetscReal center[3],                      // m
            mean_velocity[3] = {1., 1., 0}; // m/s

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
  ierr = PetscOptionsScalar("-vortex_strength", "Strength of Vortex",
                            NULL, vortex_strength, &vortex_strength, NULL);
  CHKERRQ(ierr);
  PetscInt n = problem->dim;
  PetscBool user_velocity;
  ierr = PetscOptionsRealArray("-mean_velocity", "Background velocity vector",
                               NULL, mean_velocity, &n, &user_velocity);
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
                          EulerTestTypes, (PetscEnum)(euler_test = EULER_TEST_ISENTROPIC_VORTEX),
                          (PetscEnum *)&euler_test, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-stab", "Stabilization method", NULL,
                          StabilizationTypes, (PetscEnum)(stab = STAB_NONE),
                          (PetscEnum *)&stab, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-c_tau", "Stabilization constant",
                            NULL, c_tau, &c_tau, NULL); CHKERRQ(ierr);
  // -- Units
  ierr = PetscOptionsScalar("-units_meter", "1 meter in scaled length units",
                            NULL, meter, &meter, NULL); CHKERRQ(ierr);
  meter = fabs(meter);
  ierr = PetscOptionsScalar("-units_second","1 second in scaled time units",
                            NULL, second, &second, NULL); CHKERRQ(ierr);
  second = fabs(second);

  // -- Warnings
  if (stab == STAB_SUPG && !implicit) {
    ierr = PetscPrintf(comm,
                       "Warning! Use -stab supg only with -implicit\n");
    CHKERRQ(ierr);
  }
  if (user_velocity && (euler_test == EULER_TEST_1
                        || euler_test == EULER_TEST_3)) {
    ierr = PetscPrintf(comm,
                       "Warning! Background velocity vector for -euler_test t1 and -euler_test t3 is (0,0,0)\n");
    CHKERRQ(ierr);
  }

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
  for (int i=0; i<3; i++) {
    center[i] *= meter;
    mean_velocity[i] = mean_velocity[i] * (meter/second);
  }

  // -- Setup Context
  setup_context->lx        = lx;
  setup_context->ly        = ly;
  setup_context->lz        = lz;
  setup_context->center[0] = center[0];
  setup_context->center[1] = center[1];
  setup_context->center[2] = center[2];
  setup_context->time      = 0;

  // -- QFunction Context
  user->phys->stab                        = stab;
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
  user->phys->euler_ctx->c_tau            = c_tau;
  user->phys->euler_ctx->mean_velocity[0] = mean_velocity[0];
  user->phys->euler_ctx->mean_velocity[1] = mean_velocity[1];
  user->phys->euler_ctx->mean_velocity[2] = mean_velocity[2];
  user->phys->euler_ctx->stabilization    = stab;

  PetscFunctionReturn(0);
}

PetscErrorCode SetupContext_EULER_VORTEX(Ceed ceed, CeedData ceed_data,
    AppCtx app_ctx, SetupContext setup_ctx, Physics phys) {
  PetscFunctionBeginUser;

  CeedQFunctionContextCreate(ceed, &ceed_data->setup_context);
  CeedQFunctionContextSetData(ceed_data->setup_context, CEED_MEM_HOST,
                              CEED_USE_POINTER,
                              sizeof(*setup_ctx), setup_ctx);
  CeedQFunctionContextCreate(ceed, &ceed_data->euler_context);
  CeedQFunctionContextSetData(ceed_data->euler_context, CEED_MEM_HOST,
                              CEED_USE_POINTER,
                              sizeof(*phys->euler_ctx), phys->euler_ctx);
  if (ceed_data->qf_ics)
    CeedQFunctionSetContext(ceed_data->qf_ics, ceed_data->euler_context);
  if (ceed_data->qf_apply_sur)
    CeedQFunctionSetContext(ceed_data->qf_apply_sur, ceed_data->euler_context);
  if (ceed_data->qf_rhs_vol)
    CeedQFunctionSetContext(ceed_data->qf_rhs_vol, ceed_data->euler_context);
  if (ceed_data->qf_ifunction_vol)
    CeedQFunctionSetContext(ceed_data->qf_ifunction_vol, ceed_data->euler_context);

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
  ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipz", label,
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
                     "    Background Velocity                : %f,%f,%f\n"
                     "    Stabilization                      : %s\n",
                     app_ctx->problem_name, EulerTestTypes[phys->euler_test],
                     phys->euler_ctx->mean_velocity[0],
                     phys->euler_ctx->mean_velocity[1],
                     phys->euler_ctx->mean_velocity[2],
                     StabilizationTypes[phys->stab]); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
