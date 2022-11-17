// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for setting up EULER_VORTEX

#include "../qfunctions/eulervortex.h"

#include "../navierstokes.h"
#include "../qfunctions/setupgeo.h"

PetscErrorCode NS_EULER_VORTEX(ProblemData *problem, DM dm, void *ctx) {
  EulerTestType        euler_test;
  User                 user = *(User *)ctx;
  StabilizationType    stab;
  MPI_Comm             comm = PETSC_COMM_WORLD;
  PetscBool            implicit;
  PetscBool            has_curr_time = PETSC_TRUE;
  PetscBool            has_neumann   = PETSC_TRUE;
  EulerContext         euler_ctx;
  CeedQFunctionContext euler_context;

  PetscFunctionBeginUser;
  PetscCall(PetscCalloc1(1, &euler_ctx));

  // ------------------------------------------------------
  //               SET UP DENSITY_CURRENT
  // ------------------------------------------------------
  problem->dim                               = 3;
  problem->q_data_size_vol                   = 10;
  problem->q_data_size_sur                   = 10;
  problem->setup_vol.qfunction               = Setup;
  problem->setup_vol.qfunction_loc           = Setup_loc;
  problem->setup_sur.qfunction               = SetupBoundary;
  problem->setup_sur.qfunction_loc           = SetupBoundary_loc;
  problem->ics.qfunction                     = ICsEuler;
  problem->ics.qfunction_loc                 = ICsEuler_loc;
  problem->apply_vol_rhs.qfunction           = Euler;
  problem->apply_vol_rhs.qfunction_loc       = Euler_loc;
  problem->apply_vol_ifunction.qfunction     = IFunction_Euler;
  problem->apply_vol_ifunction.qfunction_loc = IFunction_Euler_loc;
  problem->apply_inflow.qfunction            = TravelingVortex_Inflow;
  problem->apply_inflow.qfunction_loc        = TravelingVortex_Inflow_loc;
  problem->apply_outflow.qfunction           = Euler_Outflow;
  problem->apply_outflow.qfunction_loc       = Euler_Outflow_loc;
  problem->bc                                = Exact_Euler;
  problem->bc_ctx                            = euler_ctx;
  problem->non_zero_time                     = PETSC_TRUE;
  problem->print_info                        = PRINT_EULER_VORTEX;

  // ------------------------------------------------------
  //             Create the libCEED context
  // ------------------------------------------------------
  CeedScalar vortex_strength = 5.;   // -
  CeedScalar c_tau           = 0.5;  // -
  // c_tau = 0.5 is reported as "optimal" in Hughes et al 2010
  PetscReal center[3],                 // m
      mean_velocity[3] = {1., 1., 0};  // m/s
  PetscReal domain_min[3], domain_max[3], domain_size[3];
  PetscCall(DMGetBoundingBox(dm, domain_min, domain_max));
  for (PetscInt i = 0; i < 3; i++) domain_size[i] = domain_max[i] - domain_min[i];

  // ------------------------------------------------------
  //             Create the PETSc context
  // ------------------------------------------------------
  PetscScalar meter  = 1e-2;  // 1 meter in scaled length units
  PetscScalar second = 1e-2;  // 1 second in scaled time units

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  PetscOptionsBegin(comm, NULL, "Options for EULER_VORTEX problem", NULL);
  // -- Physics
  PetscCall(PetscOptionsScalar("-vortex_strength", "Strength of Vortex", NULL, vortex_strength, &vortex_strength, NULL));
  PetscInt  n = problem->dim;
  PetscBool user_velocity;
  PetscCall(PetscOptionsRealArray("-mean_velocity", "Background velocity vector", NULL, mean_velocity, &n, &user_velocity));
  for (PetscInt i = 0; i < 3; i++) center[i] = .5 * domain_size[i];
  n = problem->dim;
  PetscCall(PetscOptionsRealArray("-center", "Location of vortex center", NULL, center, &n, NULL));
  PetscCall(PetscOptionsBool("-implicit", "Use implicit (IFunction) formulation", NULL, implicit = PETSC_FALSE, &implicit, NULL));
  PetscCall(PetscOptionsEnum("-euler_test", "Euler test option", NULL, EulerTestTypes, (PetscEnum)(euler_test = EULER_TEST_ISENTROPIC_VORTEX),
                             (PetscEnum *)&euler_test, NULL));
  PetscCall(PetscOptionsEnum("-stab", "Stabilization method", NULL, StabilizationTypes, (PetscEnum)(stab = STAB_NONE), (PetscEnum *)&stab, NULL));
  PetscCall(PetscOptionsScalar("-c_tau", "Stabilization constant", NULL, c_tau, &c_tau, NULL));
  // -- Units
  PetscCall(PetscOptionsScalar("-units_meter", "1 meter in scaled length units", NULL, meter, &meter, NULL));
  meter = fabs(meter);
  PetscCall(PetscOptionsScalar("-units_second", "1 second in scaled time units", NULL, second, &second, NULL));
  second = fabs(second);

  // -- Warnings
  if (stab == STAB_SUPG && !implicit) {
    PetscCall(PetscPrintf(comm, "Warning! Use -stab supg only with -implicit\n"));
  }
  if (user_velocity && (euler_test == EULER_TEST_1 || euler_test == EULER_TEST_3)) {
    PetscCall(PetscPrintf(comm, "Warning! Background velocity vector for -euler_test t1 and -euler_test t3 is (0,0,0)\n"));
  }

  PetscOptionsEnd();

  // ------------------------------------------------------
  //           Set up the PETSc context
  // ------------------------------------------------------
  user->units->meter  = meter;
  user->units->second = second;

  // ------------------------------------------------------
  //           Set up the libCEED context
  // ------------------------------------------------------
  // -- Scale variables to desired units
  for (PetscInt i = 0; i < 3; i++) {
    center[i] *= meter;
    domain_size[i] *= meter;
    mean_velocity[i] *= (meter / second);
  }
  problem->dm_scale = meter;

  // -- QFunction Context
  user->phys->stab            = stab;
  user->phys->euler_test      = euler_test;
  user->phys->implicit        = implicit;
  user->phys->has_curr_time   = has_curr_time;
  user->phys->has_neumann     = has_neumann;
  euler_ctx->curr_time        = 0.;
  euler_ctx->implicit         = implicit;
  euler_ctx->euler_test       = euler_test;
  euler_ctx->center[0]        = center[0];
  euler_ctx->center[1]        = center[1];
  euler_ctx->center[2]        = center[2];
  euler_ctx->vortex_strength  = vortex_strength;
  euler_ctx->c_tau            = c_tau;
  euler_ctx->mean_velocity[0] = mean_velocity[0];
  euler_ctx->mean_velocity[1] = mean_velocity[1];
  euler_ctx->mean_velocity[2] = mean_velocity[2];
  euler_ctx->stabilization    = stab;

  CeedQFunctionContextCreate(user->ceed, &euler_context);
  CeedQFunctionContextSetData(euler_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*euler_ctx), euler_ctx);
  CeedQFunctionContextSetDataDestroy(euler_context, CEED_MEM_HOST, FreeContextPetsc);
  CeedQFunctionContextRegisterDouble(euler_context, "solution time", offsetof(struct EulerContext_, curr_time), 1, "Physical time of the solution");
  CeedQFunctionContextReferenceCopy(euler_context, &problem->ics.qfunction_context);
  CeedQFunctionContextReferenceCopy(euler_context, &problem->apply_vol_rhs.qfunction_context);
  CeedQFunctionContextReferenceCopy(euler_context, &problem->apply_vol_ifunction.qfunction_context);
  CeedQFunctionContextReferenceCopy(euler_context, &problem->apply_inflow.qfunction_context);
  CeedQFunctionContextReferenceCopy(euler_context, &problem->apply_outflow.qfunction_context);
  PetscFunctionReturn(0);
}

PetscErrorCode PRINT_EULER_VORTEX(ProblemData *problem, AppCtx app_ctx) {
  MPI_Comm     comm = PETSC_COMM_WORLD;
  EulerContext euler_ctx;

  PetscFunctionBeginUser;
  CeedQFunctionContextGetData(problem->ics.qfunction_context, CEED_MEM_HOST, &euler_ctx);
  PetscCall(PetscPrintf(comm,
                        "  Problem:\n"
                        "    Problem Name                       : %s\n"
                        "    Test Case                          : %s\n"
                        "    Background Velocity                : %f,%f,%f\n"
                        "    Stabilization                      : %s\n",
                        app_ctx->problem_name, EulerTestTypes[euler_ctx->euler_test], euler_ctx->mean_velocity[0], euler_ctx->mean_velocity[1],
                        euler_ctx->mean_velocity[2], StabilizationTypes[euler_ctx->stabilization]));

  CeedQFunctionContextRestoreData(problem->ics.qfunction_context, &euler_ctx);
  PetscFunctionReturn(0);
}
