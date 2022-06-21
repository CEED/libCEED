// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other
// CEED contributors. All Rights Reserved. See the top-level LICENSE and NOTICE
// files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for setting up DENSITY_CURRENT

#include "../qfunctions/densitycurrent.h"
#include "../navierstokes.h"

PetscErrorCode NS_DENSITY_CURRENT(ProblemData *problem, DM dm, void *ctx) {

  PetscInt ierr;
  SetupContext setup_context;
  User user = *(User *)ctx;
  PetscBool prim_var;
  MPI_Comm comm = PETSC_COMM_WORLD;

  PetscFunctionBeginUser;
  ierr = NS_NEWTONIAN_IG(problem, dm, ctx); CHKERRQ(ierr);
  // ------------------------------------------------------
  //               SET UP DENSITY_CURRENT
  // ------------------------------------------------------
  // -- Command line for Conservative vs Primitive variables
  PetscOptionsBegin(comm, NULL, "Options for DENSITY_CURRENT problem", NULL);
  ierr = PetscOptionsBool("-primitive", "Use primitive variables",
                          NULL, prim_var=PETSC_FALSE, &prim_var, NULL); CHKERRQ(ierr);
  PetscOptionsEnd();

  if(!prim_var) {
    problem->ics.qfunction = ICsDC;
    problem->ics.qfunction_loc = ICsDC_loc;
    problem->bc = Exact_DC;
  } else {
    problem->ics.qfunction = ICsDC_Prim;
    problem->ics.qfunction_loc = ICsDC_Prim_loc;
    problem->bc = Exact_DC_Prim;
  }

  CeedQFunctionContextGetData(problem->ics.qfunction_context, CEED_MEM_HOST,
                              &setup_context);

  // ------------------------------------------------------
  //             Create the libCEED context
  // ------------------------------------------------------
  CeedScalar theta0 = 300.; // K
  CeedScalar thetaC = -15.; // K
  CeedScalar P0 = 1.e5;     // Pa
  CeedScalar N = 0.01;      // 1/s
  CeedScalar rc = 1000.;    // m (Radius of bubble)
  PetscReal center[3], dc_axis[3] = {0, 0, 0};
  PetscReal domain_min[3], domain_max[3], domain_size[3];
  ierr = DMGetBoundingBox(dm, domain_min, domain_max);
  CHKERRQ(ierr);
  for (PetscInt i = 0; i < 3; i++)
    domain_size[i] = domain_max[i] - domain_min[i];

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  PetscOptionsBegin(comm, NULL, "Options for DENSITY_CURRENT problem", NULL);
  ierr = PetscOptionsScalar("-theta0", "Reference potential temperature", NULL,
                            theta0, &theta0, NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-thetaC", "Perturbation of potential temperature",
                            NULL, thetaC, &thetaC, NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-P0", "Atmospheric pressure", NULL, P0, &P0, NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-N", "Brunt-Vaisala frequency", NULL, N, &N, NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-rc", "Characteristic radius of thermal bubble",
                            NULL, rc, &rc, NULL);
  CHKERRQ(ierr);
  for (PetscInt i = 0; i < 3; i++)
    center[i] = .5 * domain_size[i];
  PetscInt n = problem->dim;
  ierr = PetscOptionsRealArray("-center", "Location of bubble center", NULL,
                               center, &n, NULL);
  CHKERRQ(ierr);
  n = problem->dim;
  ierr = PetscOptionsRealArray("-dc_axis",
                               "Axis of density current cylindrical anomaly, "
                               "or {0,0,0} for spherically symmetric",
                               NULL, dc_axis, &n, NULL);
  CHKERRQ(ierr);
  {
    PetscReal norm = PetscSqrtReal(PetscSqr(dc_axis[0]) + PetscSqr(dc_axis[1]) +
                                   PetscSqr(dc_axis[2]));
    if (norm > 0) {
      for (PetscInt i = 0; i < 3; i++)
        dc_axis[i] /= norm;
    }
  }

  PetscOptionsEnd();

  PetscScalar meter  = user->units->meter;
  PetscScalar second = user->units->second;
  PetscScalar Kelvin = user->units->Kelvin;
  PetscScalar Pascal = user->units->Pascal;
  rc = fabs(rc) * meter;
  theta0 *= Kelvin;
  thetaC *= Kelvin;
  P0 *= Pascal;
  N *= (1. / second);
  for (PetscInt i = 0; i < 3; i++)
    center[i] *= meter;

  setup_context->theta0 = theta0;
  setup_context->thetaC = thetaC;
  setup_context->P0 = P0;
  setup_context->N = N;
  setup_context->rc = rc;
  setup_context->center[0] = center[0];
  setup_context->center[1] = center[1];
  setup_context->center[2] = center[2];
  setup_context->dc_axis[0] = dc_axis[0];
  setup_context->dc_axis[1] = dc_axis[1];
  setup_context->dc_axis[2] = dc_axis[2];

  problem->bc_ctx =
    setup_context; // This is bad, context data should only be accessed via Get/Restore
  CeedQFunctionContextRestoreData(problem->ics.qfunction_context, &setup_context);

  PetscFunctionReturn(0);
}
