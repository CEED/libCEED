// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for setting up Blasius Boundary Layer

#include "../navierstokes.h"
#include "../qfunctions/blasius.h"
#include "stg_shur14.h"

static PetscErrorCode GetYNodeLocs(const MPI_Comm comm,
                                   const char path[PETSC_MAX_PATH_LEN], PetscReal **pynodes,
                                   PetscInt *nynodes) {
  PetscErrorCode ierr;
  PetscInt ndims, dims[2];
  FILE *fp;
  const PetscInt char_array_len = 512;
  char line[char_array_len];
  char **array;
  PetscReal *node_locs;
  PetscFunctionBeginUser;

  ierr = PetscFOpen(comm, path, "r", &fp); CHKERRQ(ierr);
  ierr = PetscSynchronizedFGets(comm, fp, char_array_len, line); CHKERRQ(ierr);
  ierr = PetscStrToArray(line, ' ', &ndims, &array); CHKERRQ(ierr);

  for (PetscInt i=0; i<ndims; i++)  dims[i] = atoi(array[i]);
  if (ndims<2) dims[1] = 1; // Assume 1 column of data is not otherwise specified
  *nynodes = dims[0];
  ierr = PetscMalloc1(*nynodes, &node_locs); CHKERRQ(ierr);

  for (PetscInt i=0; i<dims[0]; i++) {
    ierr = PetscSynchronizedFGets(comm, fp, char_array_len, line); CHKERRQ(ierr);
    ierr = PetscStrToArray(line, ' ', &ndims, &array); CHKERRQ(ierr);
    if (ndims < dims[1]) SETERRQ(comm, -1,
                                   "Line %d of %s does not contain enough columns (%d instead of %d)", i,
                                   path, ndims, dims[1]);

    node_locs[i] = (PetscReal) atof(array[0]);
  }
  ierr = PetscFClose(comm, fp); CHKERRQ(ierr);
  *pynodes = node_locs;
  PetscFunctionReturn(0);
}

/* \brief Modify the domain and mesh for blasius
 *
 * Modifies mesh such that `N` elements are within `refine_height` with a
 * geometric growth ratio of `growth`. Excess elements are then distributed
 * linearly in logspace to the top surface.
 *
 * The top surface is also angled downwards, so that it may be used as an
 * outflow. It's angle is controlled by `top_angle` (in units of degrees).
 *
 * If `node_locs` is not NULL, then the nodes will be placed at `node_locs`
 * locations. If it is NULL, then the modified coordinate values will be set in
 * the array, along with `num_node_locs`.
 */
static PetscErrorCode ModifyMesh(MPI_Comm comm, DM dm, PetscInt dim,
                                 PetscReal growth, PetscInt N,
                                 PetscReal refine_height, PetscReal top_angle,
                                 PetscReal *node_locs[], PetscInt *num_node_locs) {
  PetscInt ierr, narr, ncoords;
  PetscReal domain_min[3], domain_max[3], domain_size[3];
  PetscScalar *arr_coords;
  Vec vec_coords;
  PetscFunctionBeginUser;

  PetscReal angle_coeff = tan(top_angle*(M_PI/180));

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
  const PetscReal dybox = domain_size[1]/faces[1];

  if (!*node_locs) {
    // Calculate the first element height
    PetscReal dy1   = refine_height*(growth-1)/(pow(growth, N)-1);

    // Calculate log of sizing outside BL
    PetscReal logdy = (log(domain_max[1]) - log(refine_height)) / (faces[1] - N);

    *num_node_locs = faces[1] + 1;
    PetscReal *temp_node_locs;
    ierr = PetscMalloc1(*num_node_locs, &temp_node_locs); CHKERRQ(ierr);

    for (PetscInt i=0; i<ncoords; i++) {
      PetscInt y_box_index = round(coords[i][1]/dybox);
      if (y_box_index <= N) {
        coords[i][1] = (1 - (coords[i][0] - domain_min[0])*angle_coeff/domain_max[1])
                       * dy1 * (pow(growth, coords[i][1]/dybox)-1)/(growth-1);
      } else {
        PetscInt j = y_box_index - N;
        coords[i][1] = (1 - (coords[i][0] - domain_min[0])*angle_coeff/domain_max[1])
                       * exp(log(refine_height) + logdy*j);
      }
      if (coords[i][0] == domain_min[0] && coords[i][2] == domain_min[2])
        temp_node_locs[y_box_index] = coords[i][1];
    }

    *node_locs = temp_node_locs;
  } else {
    // Error checking
    if (*num_node_locs < faces[1] +1)
      SETERRQ(comm, -1, "The y_node_locs_path has too few locations; "
              "There are %d + 1 nodes, but only %d locations given",
              faces[1]+1, *num_node_locs);
    if (*num_node_locs > faces[1] +1) {
      ierr = PetscPrintf(comm, "WARNING: y_node_locs_path has more locations (%d) "
                         "than the mesh has nodes (%d). This maybe unintended.",
                         *num_node_locs, faces[1]+1); CHKERRQ(ierr);
    }
    PetscScalar max_y = (*node_locs)[faces[1]];

    for (PetscInt i=0; i<ncoords; i++) {
      // Determine which y-node we're at
      PetscInt y_box_index = round(coords[i][1]/dybox);
      coords[i][1] = (1 - (coords[i][0] - domain_min[0])*angle_coeff/max_y)
                     * (*node_locs)[y_box_index];
    }
  }

  ierr = VecRestoreArray(vec_coords, &arr_coords); CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm, vec_coords); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode NS_BLASIUS(ProblemData *problem, DM dm, void *ctx) {

  PetscInt ierr;
  User      user    = *(User *)ctx;
  MPI_Comm  comm    = PETSC_COMM_WORLD;
  PetscBool use_stg = PETSC_FALSE;
  BlasiusContext blasius_ctx;
  NewtonianIdealGasContext newtonian_ig_ctx;
  CeedQFunctionContext blasius_context;

  PetscFunctionBeginUser;
  ierr = NS_NEWTONIAN_IG(problem, dm, ctx); CHKERRQ(ierr);
  ierr = PetscCalloc1(1, &blasius_ctx); CHKERRQ(ierr);

  // ------------------------------------------------------
  //               SET UP Blasius
  // ------------------------------------------------------
  problem->ics.qfunction                       = ICsBlasius;
  problem->ics.qfunction_loc                   = ICsBlasius_loc;

  CeedScalar Uinf   = 40;          // m/s
  CeedScalar delta0 = 4.2e-4;      // m
  CeedScalar theta0 = 288.;        // K
  CeedScalar P0     = 1.01e5;      // Pa
  PetscBool  weakT  = PETSC_FALSE; // weak density or temperature
  PetscReal  mesh_refine_height = 5.9e-4; // m
  PetscReal  mesh_growth        = 1.08;   // [-]
  PetscInt   mesh_Ndelta        = 45;     // [-]
  PetscReal  mesh_top_angle     = 5;      // degrees
  char mesh_ynodes_path[PETSC_MAX_PATH_LEN] = "";

  PetscOptionsBegin(comm, NULL, "Options for CHANNEL problem", NULL);
  ierr = PetscOptionsBool("-weakT", "Change from rho weak to T weak at inflow",
                          NULL, weakT, &weakT, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-Uinf", "Velocity at boundary layer edge",
                            NULL, Uinf, &Uinf, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-delta0", "Boundary layer height at inflow",
                            NULL, delta0, &delta0, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-theta0", "Wall temperature",
                            NULL, theta0, &theta0, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-P0", "Pressure at outflow",
                            NULL, P0, &P0, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-platemesh_Ndelta",
                                "Velocity at boundary layer edge",
                                NULL, mesh_Ndelta, &mesh_Ndelta, NULL, 1); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-platemesh_refine_height",
                            "Height of boundary layer mesh refinement",
                            NULL, mesh_refine_height, &mesh_refine_height, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-platemesh_growth",
                            "Geometric growth rate of boundary layer mesh",
                            NULL, mesh_growth, &mesh_growth, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-platemesh_top_angle",
                            "Geometric top_angle rate of boundary layer mesh",
                            NULL, mesh_top_angle, &mesh_top_angle, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-platemesh_y_node_locs_path",
                            "Path to file with y node locations. "
                            "If empty, will use the algorithmic mesh warping.", NULL,
                            mesh_ynodes_path, mesh_ynodes_path,
                            sizeof(mesh_ynodes_path), NULL); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-stg_use", "Use STG inflow boundary condition",
                          NULL, use_stg, &use_stg, NULL); CHKERRQ(ierr);
  PetscOptionsEnd();

  PetscScalar meter  = user->units->meter;
  PetscScalar second = user->units->second;
  PetscScalar Kelvin = user->units->Kelvin;
  PetscScalar Pascal = user->units->Pascal;

  theta0 *= Kelvin;
  P0     *= Pascal;
  Uinf   *= meter / second;
  delta0 *= meter;

  PetscReal *mesh_ynodes = NULL;
  PetscInt  mesh_nynodes = 0;
  if (strcmp(mesh_ynodes_path, "")) {
    ierr = GetYNodeLocs(comm, mesh_ynodes_path, &mesh_ynodes, &mesh_nynodes);
    CHKERRQ(ierr);
  }
  ierr = ModifyMesh(comm, dm, problem->dim, mesh_growth, mesh_Ndelta,
                    mesh_refine_height, mesh_top_angle, &mesh_ynodes,
                    &mesh_nynodes); CHKERRQ(ierr);

  // Some properties depend on parameters from NewtonianIdealGas
  CeedQFunctionContextGetData(problem->apply_vol_rhs.qfunction_context,
                              CEED_MEM_HOST, &newtonian_ig_ctx);

  blasius_ctx->weakT         = weakT;
  blasius_ctx->Uinf          = Uinf;
  blasius_ctx->delta0        = delta0;
  blasius_ctx->theta0        = theta0;
  blasius_ctx->P0            = P0;
  newtonian_ig_ctx->P0       = P0;
  blasius_ctx->implicit      = user->phys->implicit;
  blasius_ctx->newtonian_ctx = *newtonian_ig_ctx;

  {
    PetscReal domain_min[3];
    ierr = DMGetBoundingBox(dm, domain_min, NULL); CHKERRQ(ierr);
    blasius_ctx->x_inflow = domain_min[0];
  }

  CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfunction_context,
                                  &newtonian_ig_ctx);

  CeedQFunctionContextCreate(user->ceed, &blasius_context);
  CeedQFunctionContextSetData(blasius_context, CEED_MEM_HOST,
                              CEED_USE_POINTER,
                              sizeof(*blasius_ctx), blasius_ctx);
  CeedQFunctionContextSetDataDestroy(blasius_context, CEED_MEM_HOST,
                                     FreeContextPetsc);

  CeedQFunctionContextDestroy(&problem->ics.qfunction_context);
  problem->ics.qfunction_context = blasius_context;
  if (use_stg) {
    ierr = SetupSTG(comm, dm, problem, user, weakT, theta0, P0, mesh_ynodes,
                    mesh_nynodes); CHKERRQ(ierr);
  } else {
    problem->apply_inflow.qfunction              = Blasius_Inflow;
    problem->apply_inflow.qfunction_loc          = Blasius_Inflow_loc;
    problem->apply_inflow_jacobian.qfunction     = Blasius_Inflow_Jacobian;
    problem->apply_inflow_jacobian.qfunction_loc = Blasius_Inflow_Jacobian_loc;
    CeedQFunctionContextReferenceCopy(blasius_context,
                                      &problem->apply_inflow.qfunction_context);
    CeedQFunctionContextReferenceCopy(blasius_context,
                                      &problem->apply_inflow_jacobian.qfunction_context);
  }
  ierr = PetscFree(mesh_ynodes); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
