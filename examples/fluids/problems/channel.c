// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for setting up Channel flow

#include "../qfunctions/channel.h"

#include <ceed.h>
#include <petscdm.h>

#include "../navierstokes.h"

// /* \brief Modify the domain and mesh for blasius
//  *
//  * Modifies mesh such that `N` elements are within `refine_height` with a geometric growth ratio of `growth`. Excess elements are then distributed
//  * linearly in logspace to the top surface.
//  *
//  * The top surface is also angled downwards, so that it may be used as an outflow.
//  * It's angle is controlled by `top_angle` (in units of degrees).
//  *
//  * If `node_locs` is not NULL, then the nodes will be placed at `node_locs` locations.
//  * If it is NULL, then the modified coordinate values will be set in the array, along with `num_node_locs`.
//  */
// static PetscErrorCode ModifyMesh(DM dm, PetscInt dim, PetscReal growth, PetscInt N, PetscReal refine_height, PetscReal top_angle,
//                                  PetscReal *node_locs[], PetscInt *num_node_locs) {
//   PetscInt     narr, ncoords;
//   PetscReal    domain_min[3], domain_max[3], domain_size[3];
//   PetscScalar *arr_coords;
//   Vec          vec_coords;
//   MPI_Comm comm = PetscObjectComm((PetscObject)dm);
//
//   PetscFunctionBeginUser;
//   PetscReal angle_coeff = tan(top_angle * (M_PI / 180));
//   // Get domain boundary information
//   PetscCall(DMGetBoundingBox(dm, domain_min, domain_max));
//   for (PetscInt i = 0; i < 3; i++) domain_size[i] = domain_max[i] - domain_min[i];
//
//   // Get coords array from DM
//   PetscCall(DMGetCoordinatesLocal(dm, &vec_coords));
//   PetscCall(VecGetLocalSize(vec_coords, &narr));
//   PetscCall(VecGetArray(vec_coords, &arr_coords));
//
//   PetscScalar(*coords)[dim] = (PetscScalar(*)[dim])arr_coords;
//   ncoords                   = narr / dim;
//
//   // Get mesh information
//   PetscInt nmax = 3, faces[3];
//   PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-dm_plex_box_faces", faces, &nmax, NULL));
//   // Get element size of the box mesh, for indexing each node
//   const PetscReal dybox = domain_size[1] / faces[1];
//
//   if (!*node_locs) {
//     // Calculate the first element height
//     PetscReal dy1 = refine_height * (growth - 1) / (pow(growth, N) - 1);
//
//     // Calculate log of sizing outside BL
//     PetscReal logdy = (log(domain_max[1]) - log(refine_height)) / (faces[1] - N);
//
//     *num_node_locs = faces[1] + 1;
//     PetscReal *temp_node_locs;
//     PetscCall(PetscMalloc1(*num_node_locs, &temp_node_locs));
//
//     for (PetscInt i = 0; i < ncoords; i++) {
//       PetscInt y_box_index = round(coords[i][1] / dybox);
//       if (y_box_index <= N) {
//         coords[i][1] =
//             (1 - (coords[i][0] - domain_min[0]) * angle_coeff / domain_max[1]) * dy1 * (pow(growth, coords[i][1] / dybox) - 1) / (growth - 1);
//       } else {
//         PetscInt j   = y_box_index - N;
//         coords[i][1] = (1 - (coords[i][0] - domain_min[0]) * angle_coeff / domain_max[1]) * exp(log(refine_height) + logdy * j);
//       }
//       if (coords[i][0] == domain_min[0] && coords[i][2] == domain_min[2]) temp_node_locs[y_box_index] = coords[i][1];
//     }
//
//     *node_locs = temp_node_locs;
//   } else {
//     PetscCheck(*num_node_locs >= faces[1] + 1, comm, PETSC_ERR_FILE_UNEXPECTED,
//                "The y_node_locs_path has too few locations; There are %" PetscInt_FMT " + 1 nodes, but only %" PetscInt_FMT " locations given",
//                faces[1] + 1, *num_node_locs);
//     if (*num_node_locs > faces[1] + 1) {
//       PetscCall(PetscPrintf(comm,
//                             "WARNING: y_node_locs_path has more locations (%" PetscInt_FMT ") "
//                             "than the mesh has nodes (%" PetscInt_FMT "). This maybe unintended.\n",
//                             *num_node_locs, faces[1] + 1));
//     }
//     PetscScalar max_y = (*node_locs)[faces[1]];
//
//     for (PetscInt i = 0; i < ncoords; i++) {
//       // Determine which y-node we're at
//       PetscInt y_box_index = round(coords[i][1] / dybox);
//       coords[i][1]         = (1 - (coords[i][0] - domain_min[0]) * angle_coeff / max_y) * (*node_locs)[y_box_index];
//     }
//   }
//
//   PetscCall(VecRestoreArray(vec_coords, &arr_coords));
//   PetscCall(DMSetCoordinatesLocal(dm, vec_coords));
//   PetscFunctionReturn(PETSC_SUCCESS);
// }

static PetscErrorCode ModifyMesh(MPI_Comm comm, DM dm, PetscInt dim) {
  PetscInt     narr, ncoords;
  PetscReal    domain_min[3], domain_max[3], domain_size[3];
  PetscScalar *arr_coords, *arr_cellcoords;
  Vec          vec_coords, vec_cellcoords;
  PetscFunctionBeginUser;

  // Get domain boundary information
  PetscCall(DMGetBoundingBox(dm, domain_min, domain_max));
  for (PetscInt i = 0; i < 3; i++) domain_size[i] = domain_max[i] - domain_min[i];

  // Get coords array from DM
  PetscCall(DMGetCoordinatesLocal(dm, &vec_coords));
  // PetscCall(DMGetCellCoordinatesLocal(dm, &vec_cellcoords));
  PetscCall(VecGetLocalSize(vec_coords, &narr));
  PetscCall(VecGetArray(vec_coords, &arr_coords));

  PetscScalar(*coords)[dim] = (PetscScalar(*)[dim])arr_coords;
  ncoords                   = narr / dim;
  printf("narr_coords: %" PetscInt_FMT " \nncoords: %" PetscInt_FMT " \n", narr, ncoords);

  // PetscCall(VecGetLocalSize(vec_cellcoords, &narr));
  // PetscCall(VecGetArray(vec_cellcoords, &arr_cellcoords));
  //
  // PetscScalar(*cellcoords)[dim] = (PetscScalar(*)[dim])arr_cellcoords;
  // PetscInt ncellcoords          = narr / dim;
  //
  // printf("narr_cellcoords: %" PetscInt_FMT " \nncellcoords: %" PetscInt_FMT " \n", narr, ncellcoords);

  // Get mesh information
  PetscInt nmax = 3, faces[3];
  PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-dm_plex_box_faces", faces, &nmax, NULL));
  // Get element size of the box mesh, for indexing each node
  const PetscReal dxbox = domain_size[0] / (faces[0]);

  for (PetscInt i = 0; i < ncoords; i++) {
    PetscInt x_box_index = round(coords[i][0] / dxbox);
    // PetscInt y_box_index = round(coords[i][1]/dxbox);
    if (x_box_index % 2) {
      coords[i][0] = (x_box_index - 1) * dxbox + 0.5 * dxbox;
      // PetscReal test = (x_box_index-1)*dxbox + dxbox;
      // coords[i][0] = (x_box_index-1)*dxbox + dxbox;
    }
    // if (y_box_index % 2) {
    //   coords[i][1] = (y_box_index-1)*dxbox + 0.5*dxbox;
    // }
  }

  // for (PetscInt i = 0; i < ncellcoords; i++) {
  //   PetscInt x_box_index = round(cellcoords[i][0] / dxbox);
  //   // PetscInt y_box_index = round(coords[i][1]/dxbox);
  //   if (x_box_index % 2) {
  //     cellcoords[i][0] = (x_box_index - 1) * dxbox + 0.5 * dxbox;
  //     // PetscReal test = (x_box_index-1)*dxbox + dxbox;
  //     // cellcoords[i][0] = (x_box_index-1)*dxbox + dxbox;
  //   }
  //   // if (y_box_index % 2) {
  //   //   coords[i][1] = (y_box_index-1)*dxbox + 0.5*dxbox;
  //   // }
  // }

  PetscCall(VecRestoreArray(vec_coords, &arr_coords));
  PetscCall(DMSetCoordinatesLocal(dm, vec_coords));

  // PetscCall(VecRestoreArray(vec_cellcoords, &arr_cellcoords));
  // PetscCall(DMSetCellCoordinatesLocal(dm, vec_cellcoords));

  PetscFunctionReturn(0);
}

PetscErrorCode NS_CHANNEL(ProblemData problem, DM dm, void *ctx, SimpleBC bc) {
  User                     user = *(User *)ctx;
  MPI_Comm                 comm = user->comm;
  Ceed                     ceed = user->ceed;
  ChannelContext           channel_ctx;
  NewtonianIdealGasContext newtonian_ig_ctx;
  CeedQFunctionContext     channel_context;

  PetscFunctionBeginUser;
  PetscCall(NS_NEWTONIAN_IG(problem, dm, ctx, bc));
  PetscCall(PetscCalloc1(1, &channel_ctx));

  PetscCall(ModifyMesh(comm, dm, problem->dim));

  // ------------------------------------------------------
  //               SET UP Channel
  // ------------------------------------------------------
  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&problem->ics.qfunction_context));
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
  PetscCallCeed(ceed, CeedQFunctionContextGetData(problem->apply_vol_rhs.qfunction_context, CEED_MEM_HOST, &newtonian_ig_ctx));

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
  PetscCallCeed(ceed, CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfunction_context, &newtonian_ig_ctx));

  PetscCallCeed(ceed, CeedQFunctionContextCreate(user->ceed, &channel_context));
  PetscCallCeed(ceed, CeedQFunctionContextSetData(channel_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*channel_ctx), channel_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(channel_context, CEED_MEM_HOST, FreeContextPetsc));

  problem->ics.qfunction_context = channel_context;
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(channel_context, &problem->apply_inflow.qfunction_context));
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(channel_context, &problem->apply_outflow.qfunction_context));
  PetscFunctionReturn(PETSC_SUCCESS);
}
