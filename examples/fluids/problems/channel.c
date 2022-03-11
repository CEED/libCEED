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
/// Utility functions for setting up Channel flow

#include "../navierstokes.h"
#include "../qfunctions/newtonian.h"
#include "../qfunctions/channel.h"

#ifndef channel_context_struct
#define channel_context_struct
typedef struct ChannelContext_ *ChannelContext;
struct ChannelContext_ {
  bool       implicit; // !< Using implicit timesteping or not
  CeedScalar theta0;   // !< Reference temperature
  CeedScalar P0;       // !< Reference Pressure
  CeedScalar umax;     // !< Centerline velocity
  CeedScalar center;   // !< Y Coordinate for center of channel
  CeedScalar H;        // !< Channel half-height
  struct NewtonianIdealGasContext_ newtonian_ctx;
};
#endif

PetscErrorCode NS_CHANNEL(ProblemData *problem, DM dm, void *setup_ctx,
                          void *ctx) {

  PetscInt ierr;
  ierr = NS_NEWTONIAN_IG(problem, dm, setup_ctx, ctx); CHKERRQ(ierr);
  SetupContext      setup_context = *(SetupContext *)setup_ctx;
  User              user = *(User *)ctx;
  MPI_Comm          comm = PETSC_COMM_WORLD;
  PetscFunctionBeginUser;
  ierr = PetscCalloc1(1, &user->phys->channel_ctx); CHKERRQ(ierr);

  // ------------------------------------------------------
  //               SET UP Channel
  // ------------------------------------------------------
  problem->ics               = ICsChannel;
  problem->ics_loc           = ICsChannel_loc;
  problem->apply_inflow      = Channel_Inflow;
  problem->apply_inflow_loc  = Channel_Inflow_loc;
  problem->apply_outflow     = Channel_Outflow;
  problem->apply_outflow_loc = Channel_Outflow_loc;
  problem->setup_ctx         = SetupContext_CHANNEL;

  // -- Command Line Options
  CeedScalar umax   = 10.;  // m/s
  CeedScalar mu     = .01;  // Pa s, dynamic viscosity
    //TODO ^^ make optional/respect explicit user set
  CeedScalar theta0 = 300.; // K
  CeedScalar P0     = 1.e5; // Pa
  ierr = PetscOptionsBegin(comm, NULL, "Options for CHANNEL problem",
                           NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-umax", "Centerline velocity of the Channel",
                            NULL, umax, &umax, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-theta0", "Wall temperature",
                            NULL, theta0, &theta0, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-P0", "Pressure at outflow",
                            NULL, P0, &P0, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  PetscScalar meter  = user->units->meter;
  PetscScalar second = user->units->second;
  PetscScalar Kelvin = user->units->Kelvin;
  PetscScalar Pascal = user->units->Pascal;

  mu     *= Pascal * second;
  theta0 *= Kelvin;
  P0     *= Pascal;
  umax   *= meter / second;

  //-- Setup Problem information
  CeedScalar H, center;
  {
    PetscReal domain_min[3], domain_max[3], domain_size[3];
    ierr = DMGetBoundingBox(dm, domain_min, domain_max); CHKERRQ(ierr);
    for (int i=0; i<3; i++) domain_size[i] = domain_max[i] - domain_min[i];

    H      = 0.5*(domain_max[1] - domain_min[1])*meter;
    center = H + domain_min[1]*meter;
  }

  user->phys->channel_ctx->center   = center;
  user->phys->channel_ctx->H        = H;
  user->phys->channel_ctx->theta0   = theta0;
  user->phys->channel_ctx->P0       = P0;
  user->phys->channel_ctx->umax     = umax;
  user->phys->channel_ctx->implicit = user->phys->implicit;

  user->phys->newtonian_ig_ctx->mu = mu;
  user->phys->channel_ctx->newtonian_ctx = *user->phys->newtonian_ig_ctx;

  PetscFunctionReturn(0);
}

PetscErrorCode SetupContext_CHANNEL(Ceed ceed, CeedData ceed_data,
                                    AppCtx app_ctx, SetupContext setup_ctx, Physics phys) {
  PetscFunctionBeginUser;
  PetscInt ierr;
  ierr = SetupContext_NEWTONIAN_IG(ceed, ceed_data, app_ctx, setup_ctx, phys);
    CHKERRQ(ierr);
  CeedQFunctionContextCreate(ceed, &ceed_data->channel_context);
  CeedQFunctionContextSetData(ceed_data->channel_context, CEED_MEM_HOST,
                              CEED_USE_POINTER,
                              sizeof(*phys->channel_ctx), phys->channel_ctx);
  phys->has_neumann = PETSC_TRUE;
  if (ceed_data->qf_ics)
    CeedQFunctionSetContext(ceed_data->qf_ics, ceed_data->channel_context);
  if (ceed_data->qf_apply_inflow)
    CeedQFunctionSetContext(ceed_data->qf_apply_inflow, ceed_data->channel_context);
  if (ceed_data->qf_apply_outflow)
    CeedQFunctionSetContext(ceed_data->qf_apply_outflow,
                            ceed_data->channel_context);
  PetscFunctionReturn(0);
}
