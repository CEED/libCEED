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
  MPI_Comm                 comm = PETSC_COMM_WORLD;
  User                     user = *(User *)ctx;
  DensityCurrentContext    dc_ctx;
  CeedQFunctionContext     density_current_context;
  NewtonianIdealGasContext newtonian_ig_ctx;

  PetscFunctionBeginUser;
  PetscCall(NS_NEWTONIAN_IG(problem, dm, ctx));
  PetscCall(PetscCalloc1(1, &dc_ctx));
  // ------------------------------------------------------
  //               SET UP DENSITY_CURRENT
  // ------------------------------------------------------
  CeedQFunctionContextDestroy(&problem->ics.qfunction_context);
  problem->ics.qfunction     = ICsDC;
  problem->ics.qfunction_loc = ICsDC_loc;

  // ------------------------------------------------------
  //             Create the libCEED context
  // ------------------------------------------------------
  CeedScalar theta0 = 300.;   // K
  CeedScalar thetaC = -15.;   // K
  CeedScalar P0     = 1.e5;   // Pa
  CeedScalar N      = 0.01;   // 1/s
  CeedScalar rc     = 1000.;  // m (Radius of bubble)
  PetscReal  center[3], dc_axis[3] = {0, 0, 0};
  PetscReal  domain_min[3], domain_max[3], domain_size[3];
  PetscCall(DMGetBoundingBox(dm, domain_min, domain_max));
  for (PetscInt i = 0; i < 3; i++) domain_size[i] = domain_max[i] - domain_min[i];

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  PetscOptionsBegin(comm, NULL, "Options for DENSITY_CURRENT problem", NULL);
  PetscCall(PetscOptionsScalar("-theta0", "Reference potential temperature", NULL, theta0, &theta0, NULL));
  PetscCall(PetscOptionsScalar("-thetaC", "Perturbation of potential temperature", NULL, thetaC, &thetaC, NULL));
  PetscCall(PetscOptionsScalar("-P0", "Atmospheric pressure", NULL, P0, &P0, NULL));
  PetscCall(PetscOptionsScalar("-N", "Brunt-Vaisala frequency", NULL, N, &N, NULL));
  PetscCall(PetscOptionsScalar("-rc", "Characteristic radius of thermal bubble", NULL, rc, &rc, NULL));
  for (PetscInt i = 0; i < 3; i++) center[i] = .5 * domain_size[i];
  PetscInt n = problem->dim;
  PetscCall(PetscOptionsRealArray("-center", "Location of bubble center", NULL, center, &n, NULL));
  n = problem->dim;
  PetscCall(PetscOptionsRealArray("-dc_axis",
                                  "Axis of density current cylindrical anomaly, "
                                  "or {0,0,0} for spherically symmetric",
                                  NULL, dc_axis, &n, NULL));
  {
    PetscReal norm = PetscSqrtReal(PetscSqr(dc_axis[0]) + PetscSqr(dc_axis[1]) + PetscSqr(dc_axis[2]));
    if (norm > 0) {
      for (PetscInt i = 0; i < 3; i++) dc_axis[i] /= norm;
    }
  }

  PetscOptionsEnd();

  PetscScalar meter  = user->units->meter;
  PetscScalar second = user->units->second;
  PetscScalar Kelvin = user->units->Kelvin;
  PetscScalar Pascal = user->units->Pascal;
  rc                 = fabs(rc) * meter;
  theta0 *= Kelvin;
  thetaC *= Kelvin;
  P0 *= Pascal;
  N *= (1. / second);
  for (PetscInt i = 0; i < 3; i++) center[i] *= meter;

  dc_ctx->theta0     = theta0;
  dc_ctx->thetaC     = thetaC;
  dc_ctx->P0         = P0;
  dc_ctx->N          = N;
  dc_ctx->rc         = rc;
  dc_ctx->center[0]  = center[0];
  dc_ctx->center[1]  = center[1];
  dc_ctx->center[2]  = center[2];
  dc_ctx->dc_axis[0] = dc_axis[0];
  dc_ctx->dc_axis[1] = dc_axis[1];
  dc_ctx->dc_axis[2] = dc_axis[2];

  CeedQFunctionContextGetData(problem->apply_vol_rhs.qfunction_context, CEED_MEM_HOST, &newtonian_ig_ctx);
  dc_ctx->newtonian_ctx = *newtonian_ig_ctx;
  CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfunction_context, &newtonian_ig_ctx);
  CeedQFunctionContextCreate(user->ceed, &density_current_context);
  CeedQFunctionContextSetData(density_current_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*dc_ctx), dc_ctx);
  CeedQFunctionContextSetDataDestroy(density_current_context, CEED_MEM_HOST, FreeContextPetsc);
  problem->ics.qfunction_context = density_current_context;

  PetscFunctionReturn(0);
}
