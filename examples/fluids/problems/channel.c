// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for setting up Channel flow

#include "../navierstokes.h"
#include "../qfunctions/channel.h"


static PetscErrorCode ModifyMesh(MPI_Comm comm, DM dm, PetscInt dim) {
  PetscInt ierr, narr, ncoords;
  PetscReal domain_min[3], domain_max[3], domain_size[3];
  PetscScalar *arr_coords;
  Vec vec_coords;
  PetscFunctionBeginUser;

  // Get domain boundary information
  ierr = DMGetBoundingBox(dm, domain_min, domain_max); CHKERRQ(ierr);
  for (PetscInt i=0; i<3; i++) domain_size[i] = domain_max[i] - domain_min[i];

  // Get coords array from DM
  ierr = DMGetCoordinatesLocal(dm, &vec_coords); CHKERRQ(ierr);
  ierr = VecGetLocalSize(vec_coords, &narr); CHKERRQ(ierr);
  ierr = VecGetArray(vec_coords, &arr_coords); CHKERRQ(ierr);

  PetscScalar (*coords)[dim] = (PetscScalar(*)[dim]) arr_coords;
  ncoords = narr/dim;

  // Get mesh information
  PetscInt nmax = 3, faces[3];
  ierr = PetscOptionsGetIntArray(NULL, NULL, "-dm_plex_box_faces", faces, &nmax,
                                 NULL); CHKERRQ(ierr);
  // Get element size of the box mesh, for indexing each node
  const PetscReal dxbox = domain_size[0]/faces[0];

  for (PetscInt i=0; i<ncoords; i++) {
    PetscInt x_box_index = round(coords[i][0]/dxbox);
    PetscInt y_box_index = round(coords[i][1]/dxbox);
      // coords[i][0] = 0.;
      // coords[i][1] = 0.;
      coords[i][2] *= 10.;
    if (x_box_index % 2) {
      coords[i][0] = (x_box_index-1)*dxbox + 0.5*dxbox;
    }
    if (y_box_index % 2) {
      coords[i][1] = (y_box_index-1)*dxbox + 0.5*dxbox;
    }
  }

  ierr = VecRestoreArray(vec_coords, &arr_coords); CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm, vec_coords); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode NS_CHANNEL(ProblemData *problem, DM dm, void *ctx) {

  PetscInt ierr;
  User              user = *(User *)ctx;
  MPI_Comm          comm = PETSC_COMM_WORLD;
  ChannelContext    channel_ctx;
  NewtonianIdealGasContext newtonian_ig_ctx;
  CeedQFunctionContext channel_context;

  PetscFunctionBeginUser;
  ierr = NS_NEWTONIAN_IG(problem, dm, ctx); CHKERRQ(ierr);
  ierr = PetscCalloc1(1, &channel_ctx); CHKERRQ(ierr);

  ierr = ModifyMesh(comm, dm, problem->dim); CHKERRQ(ierr);

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
  CeedScalar umax   = 10.;  // m/s
  CeedScalar theta0 = 300.; // K
  CeedScalar P0     = 1.e5; // Pa
  PetscReal body_force_scale = 1.;
  PetscOptionsBegin(comm, NULL, "Options for CHANNEL problem", NULL);
  ierr = PetscOptionsScalar("-umax", "Centerline velocity of the Channel",
                            NULL, umax, &umax, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-theta0", "Wall temperature",
                            NULL, theta0, &theta0, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-P0", "Pressure at outflow",
                            NULL, P0, &P0, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-body_force_scale", "Multiplier for body force",
                          NULL, body_force_scale=1, &body_force_scale, NULL); CHKERRQ(ierr);
  PetscOptionsEnd();

  PetscScalar meter  = user->units->meter;
  PetscScalar second = user->units->second;
  PetscScalar Kelvin = user->units->Kelvin;
  PetscScalar Pascal = user->units->Pascal;

  theta0 *= Kelvin;
  P0     *= Pascal;
  umax   *= meter / second;

  //-- Setup Problem information
  CeedScalar H, center;
  {
    PetscReal domain_min[3], domain_max[3], domain_size[3];
    ierr = DMGetBoundingBox(dm, domain_min, domain_max); CHKERRQ(ierr);
    for (PetscInt i=0; i<3; i++) domain_size[i] = domain_max[i] - domain_min[i];

    H      = 0.5*domain_size[1]*meter;
    center = H + domain_min[1]*meter;
  }

  // Some properties depend on parameters from NewtonianIdealGas
  CeedQFunctionContextGetData(problem->apply_vol_rhs.qfunction_context,
                              CEED_MEM_HOST, &newtonian_ig_ctx);

  channel_ctx->center   = center;
  channel_ctx->H        = H;
  channel_ctx->theta0   = theta0;
  channel_ctx->P0       = P0;
  channel_ctx->umax     = umax;
  channel_ctx->implicit = user->phys->implicit;
  channel_ctx->B = body_force_scale * 2 * umax*newtonian_ig_ctx->mu / (H*H);

  {
    // Calculate Body force
    CeedScalar cv  = newtonian_ig_ctx->cv,
               cp  = newtonian_ig_ctx->cp;
    CeedScalar Rd  = cp - cv;
    CeedScalar rho = P0 / (Rd*theta0);
    CeedScalar g[] = {channel_ctx->B / rho, 0., 0.};
    ierr = PetscArraycpy(newtonian_ig_ctx->g, g, 3); CHKERRQ(ierr);
  }
  channel_ctx->newtonian_ctx = *newtonian_ig_ctx;
  CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfunction_context,
                                  &newtonian_ig_ctx);

  CeedQFunctionContextCreate(user->ceed, &channel_context);
  CeedQFunctionContextSetData(channel_context, CEED_MEM_HOST,
                              CEED_USE_POINTER,
                              sizeof(*channel_ctx), channel_ctx);
  CeedQFunctionContextSetDataDestroy(channel_context, CEED_MEM_HOST,
                                     FreeContextPetsc);

  problem->ics.qfunction_context = channel_context;
  CeedQFunctionContextReferenceCopy(channel_context,
                                    &problem->apply_inflow.qfunction_context);
  CeedQFunctionContextReferenceCopy(channel_context,
                                    &problem->apply_outflow.qfunction_context);
  PetscFunctionReturn(0);
}
