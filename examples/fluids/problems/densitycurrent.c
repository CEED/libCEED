// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for setting up DENSITY_CURRENT

#include "../navierstokes.h"
#include "../qfunctions/densitycurrent.h"

PetscErrorCode NS_DENSITY_CURRENT(ProblemData *problem, DM dm, void *setup_ctx,
                                  void *ctx) {

  PetscInt ierr;
  ierr = NS_NEWTONIAN_IG(problem, dm, setup_ctx, ctx); CHKERRQ(ierr);
  SetupContext setup_context = *(SetupContext *)setup_ctx;
  User         user          = *(User *)ctx;
  MPI_Comm     comm          = PETSC_COMM_WORLD;
  PetscFunctionBeginUser;

  // ------------------------------------------------------
  //               SET UP DENSITY_CURRENT
  // ------------------------------------------------------
  problem->ics     = ICsDC;
  problem->ics_loc = ICsDC_loc;
  problem->bc      = Exact_DC;

  // ------------------------------------------------------
  //             Create the libCEED context
  // ------------------------------------------------------
  CeedScalar rc     = 1000.;   // m (Radius of bubble)
  PetscReal center[3], dc_axis[3] = {0, 0, 0};
  PetscReal domain_min[3], domain_max[3], domain_size[3];
  ierr = DMGetBoundingBox(dm, domain_min, domain_max); CHKERRQ(ierr);
  for (int i=0; i<3; i++) domain_size[i] = domain_max[i] - domain_min[i];

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  PetscOptionsBegin(comm, NULL, "Options for DENSITY_CURRENT problem", NULL);
  ierr = PetscOptionsScalar("-rc", "Characteristic radius of thermal bubble",
                            NULL, rc, &rc, NULL); CHKERRQ(ierr);
  for (int i=0; i<3; i++) center[i] = .5*domain_size[i];
  PetscInt n = problem->dim;
  ierr = PetscOptionsRealArray("-center", "Location of bubble center",
                               NULL, center, &n, NULL); CHKERRQ(ierr);
  n = problem->dim;
  ierr = PetscOptionsRealArray("-dc_axis",
                               "Axis of density current cylindrical anomaly, or {0,0,0} for spherically symmetric",
                               NULL, dc_axis, &n, NULL); CHKERRQ(ierr);
  {
    PetscReal norm = PetscSqrtReal(PetscSqr(dc_axis[0]) + PetscSqr(dc_axis[1]) +
                                   PetscSqr(dc_axis[2]));
    if (norm > 0) {
      for (int i=0; i<3; i++)  dc_axis[i] /= norm;
    }
  }

  PetscOptionsEnd();

  PetscScalar meter = user->units->meter;
  rc = fabs(rc) * meter;
  for (int i=0; i<3; i++) center[i] *= meter;

  setup_context->rc         = rc;
  setup_context->center[0]  = center[0];
  setup_context->center[1]  = center[1];
  setup_context->center[2]  = center[2];
  setup_context->dc_axis[0] = dc_axis[0];
  setup_context->dc_axis[1] = dc_axis[1];
  setup_context->dc_axis[2] = dc_axis[2];

  PetscFunctionReturn(0);
}

PetscErrorCode SetupContext_DENSITY_CURRENT(Ceed ceed, CeedData ceed_data,
    AppCtx app_ctx, SetupContext setup_ctx, Physics phys) {
  PetscFunctionBeginUser;
  PetscInt ierr = SetupContext_NEWTONIAN_IG(ceed, ceed_data, app_ctx, setup_ctx,
                  phys);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PRINT_DENSITY_CURRENT(Physics phys, SetupContext setup_ctx,
                                     AppCtx app_ctx) {
  MPI_Comm       comm = PETSC_COMM_WORLD;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  ierr = PetscPrintf(comm,
                     "  Problem:\n"
                     "    Problem Name                       : %s\n"
                     "    Stabilization                      : %s\n",
                     app_ctx->problem_name, StabilizationTypes[phys->stab]);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
