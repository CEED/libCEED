// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for setting up Channel flow

#include "../qfunctions/channel.h"

#include "../navierstokes.h"

PetscErrorCode NS_CHANNEL(ProblemData *problem, DM dm, void *ctx) {
  User                     user = *(User *)ctx;
  MPI_Comm                 comm = PETSC_COMM_WORLD;
  ChannelContext           channel_ctx;
  NewtonianIdealGasContext newtonian_ig_ctx;
  CeedQFunctionContext     channel_context;

  PetscFunctionBeginUser;
  PetscCall(NS_NEWTONIAN_IG(problem, dm, ctx));
  PetscCall(PetscCalloc1(1, &channel_ctx));

  // ------------------------------------------------------
  //               SET UP Channel
  // ------------------------------------------------------
  CeedQFunctionContextDestroy(&problem->ics.qfunction_context);
  problem->ics.qfunction     = ICsChannel;
  problem->ics.qfunction_loc = ICsChannel_loc;
  if (user->phys->state_var == STATEVAR_CONSERVATIVE) {
    problem->apply_inflow.qfunction      = Channel_Inflow;
    problem->apply_inflow.qfunction_loc  = Channel_Inflow_loc;
    problem->apply_outflow.qfunction     = Channel_Outflow;
    problem->apply_outflow.qfunction_loc = Channel_Outflow_loc;
  }

  // -- Command Line Options
  CeedScalar umax             = 10.;   // m/s
  CeedScalar theta0           = 300.;  // K
  CeedScalar P0               = 1.e5;  // Pa
  PetscReal  body_force_scale = 1.;
  PetscOptionsBegin(comm, NULL, "Options for CHANNEL problem", NULL);
  PetscCall(PetscOptionsScalar("-umax", "Centerline velocity of the Channel", NULL, umax, &umax, NULL));
  PetscCall(PetscOptionsScalar("-theta0", "Wall temperature", NULL, theta0, &theta0, NULL));
  PetscCall(PetscOptionsScalar("-P0", "Pressure at outflow", NULL, P0, &P0, NULL));
  PetscCall(PetscOptionsReal("-body_force_scale", "Multiplier for body force", NULL, body_force_scale = 1, &body_force_scale, NULL));
  PetscOptionsEnd();

  PetscScalar meter  = user->units->meter;
  PetscScalar second = user->units->second;
  PetscScalar Kelvin = user->units->Kelvin;
  PetscScalar Pascal = user->units->Pascal;

  theta0 *= Kelvin;
  P0 *= Pascal;
  umax *= meter / second;

  //-- Setup Problem information
  CeedScalar H, center;
  {
    PetscReal domain_min[3], domain_max[3], domain_size[3];
    PetscCall(DMGetBoundingBox(dm, domain_min, domain_max));
    for (PetscInt i = 0; i < 3; i++) domain_size[i] = domain_max[i] - domain_min[i];

    H      = 0.5 * domain_size[1] * meter;
    center = H + domain_min[1] * meter;
  }

  // Some properties depend on parameters from NewtonianIdealGas
  CeedQFunctionContextGetData(problem->apply_vol_rhs.qfunction_context, CEED_MEM_HOST, &newtonian_ig_ctx);

  channel_ctx->center   = center;
  channel_ctx->H        = H;
  channel_ctx->theta0   = theta0;
  channel_ctx->P0       = P0;
  channel_ctx->umax     = umax;
  channel_ctx->implicit = user->phys->implicit;
  channel_ctx->B        = body_force_scale * 2 * umax * newtonian_ig_ctx->mu / (H * H);

  {
    // Calculate Body force
    CeedScalar cv = newtonian_ig_ctx->cv, cp = newtonian_ig_ctx->cp;
    CeedScalar Rd  = cp - cv;
    CeedScalar rho = P0 / (Rd * theta0);
    CeedScalar g[] = {channel_ctx->B / rho, 0., 0.};
    PetscCall(PetscArraycpy(newtonian_ig_ctx->g, g, 3));
  }
  channel_ctx->newtonian_ctx = *newtonian_ig_ctx;
  CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfunction_context, &newtonian_ig_ctx);

  CeedQFunctionContextCreate(user->ceed, &channel_context);
  CeedQFunctionContextSetData(channel_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*channel_ctx), channel_ctx);
  CeedQFunctionContextSetDataDestroy(channel_context, CEED_MEM_HOST, FreeContextPetsc);

  problem->ics.qfunction_context = channel_context;
  CeedQFunctionContextReferenceCopy(channel_context, &problem->apply_inflow.qfunction_context);
  CeedQFunctionContextReferenceCopy(channel_context, &problem->apply_outflow.qfunction_context);
  PetscFunctionReturn(0);
}
