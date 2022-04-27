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

#ifndef blasius_context_struct
#define blasius_context_struct
typedef struct BlasiusContext_ *BlasiusContext;
struct BlasiusContext_ {
  bool       implicit;  // !< Using implicit timesteping or not
  CeedScalar delta0;    // !< Boundary layer height at inflow
  CeedScalar Uinf;      // !< Velocity at boundary layer edge
  CeedScalar P0;        // !< Pressure at outflow
  CeedScalar theta0;    // !< Temperature at inflow
  CeedInt weakT;    // !< flag to weakly set Temperature at inflow if not set weak rho instead
  struct NewtonianIdealGasContext_ newtonian_ctx;
};
#endif

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

/* \brief Modify the domain and mesh for blasius
 *
 * Modifies mesh such that `N` elements are within 1.2*`delta0` with a geometric
 * growth ratio of `growth`. Excess elements are then geometrically distributed
 * to the top surface.
 *
 * The top surface is also angled downwards, so that it may be used as an
 * outflow. It's angle is controlled by top_angle (in units of degrees).
 */
PetscErrorCode modifyMesh(DM dm, PetscInt dim, PetscReal growth, PetscInt N,
                          PetscReal refine_height, PetscReal top_angle) {

  PetscInt ierr, narr, ncoords;
  PetscReal domain_min[3], domain_max[3], domain_size[3];
  PetscScalar *arr_coords;
  Vec vec_coords;
  PetscFunctionBeginUser;

  PetscReal angle_coeff = tan(top_angle*(M_PI/180));

  // Get domain boundary information
  ierr = DMGetBoundingBox(dm, domain_min, domain_max); CHKERRQ(ierr);
  for (int i=0; i<3; i++) domain_size[i] = domain_max[i] - domain_min[i];

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

  // Calculate the first element height
  PetscReal dybox = domain_size[1]/faces[1];
  PetscReal dy1   = refine_height*(growth-1)/(pow(growth, N)-1);

  // Calculate log of sizing outside BL
  PetscReal logdy = (log(domain_max[1]) - log(refine_height)) / (faces[1] - N);

  for(int i=0; i<ncoords; i++) {
    PetscInt y_box_index = round(coords[i][1]/dybox);
    if(y_box_index <= N) {
      coords[i][1] = (1 - (coords[i][0]/domain_max[0])*angle_coeff) *
                     dy1*(pow(growth, coords[i][1]/dybox)-1)/(growth-1);
    } else {
      PetscInt j = y_box_index - N;
      coords[i][1] = (1 - (coords[i][0]/domain_max[0])*angle_coeff) *
                     exp(log(refine_height) + logdy*j);
    }
  }

  ierr = VecRestoreArray(vec_coords, &arr_coords); CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm, vec_coords); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode NS_BLASIUS(ProblemData *problem, DM dm, void *setup_ctx,
                          void *ctx) {

  PetscInt ierr;
  ierr = NS_NEWTONIAN_IG(problem, dm, setup_ctx, ctx); CHKERRQ(ierr);
  User              user = *(User *)ctx;
  MPI_Comm          comm = PETSC_COMM_WORLD;
  PetscFunctionBeginUser;
  ierr = PetscCalloc1(1, &user->phys->blasius_ctx); CHKERRQ(ierr);

  // ------------------------------------------------------
  //               SET UP Blasius
  // ------------------------------------------------------
  problem->ics                     = ICsBlasius;
  problem->ics_loc                 = ICsBlasius_loc;
  problem->apply_inflow            = Blasius_Inflow;
  problem->apply_inflow_loc        = Blasius_Inflow_loc;
  problem->apply_outflow           = Blasius_Outflow;
  problem->apply_outflow_loc       = Blasius_Outflow_loc;
  problem->setup_ctx               = SetupContext_BLASIUS;

  // CeedScalar mu = .04; // Pa s, dynamic viscosity
  CeedScalar mu            = 1.8e-5;   // Pa s, dynamic viscosity
  CeedScalar Uinf          = 40;   // m/s
  CeedScalar delta0        = 4.2e-4;    // m
  PetscReal  refine_height = 5.9e-4;    // m
  PetscReal  growth        = 1.08; // [-]
  PetscInt   Ndelta        = 45;   // [-]
  PetscReal  top_angle     = 5;    // degrees
  CeedScalar theta0        = 288.; // K
  CeedScalar P0            = 1.01e5; // Pa
  CeedInt    weakT         = 0; // if not changed to 1 weak density will be chosen

  PetscOptionsBegin(comm, NULL, "Options for CHANNEL problem", NULL);
  ierr = PetscOptionsInt("-weakT", "Change from rho weak to T weak at inflow",
                            NULL, weakT, &weakT, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-Uinf", "Velocity at boundary layer edge",
                            NULL, Uinf, &Uinf, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-delta0", "Boundary layer height at inflow",
                            NULL, delta0, &delta0, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-theta0", "Wall temperature",
                            NULL, theta0, &theta0, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-P0", "Pressure at outflow",
                            NULL, P0, &P0, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-Ndelta", "Velocity at boundary layer edge",
                                NULL, Ndelta, &Ndelta, NULL, 1); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-refine_height",
                            "Height of boundary layer mesh refinement",
                            NULL, refine_height, &refine_height, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-growth",
                            "Geometric growth rate of boundary layer mesh",
                            NULL, growth, &growth, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-top_angle",
                            "Geometric top_angle rate of boundary layer mesh",
                            NULL, top_angle, &top_angle, NULL); CHKERRQ(ierr);
  PetscOptionsEnd();

  PetscScalar meter           = user->units->meter;
  PetscScalar second          = user->units->second;
  PetscScalar Kelvin          = user->units->Kelvin;
  PetscScalar Pascal          = user->units->Pascal;

  mu     *= Pascal * second;
  theta0 *= Kelvin;
  P0     *= Pascal;
  Uinf   *= meter / second;
  delta0 *= meter;

  ierr = modifyMesh(dm, problem->dim, growth, Ndelta, refine_height, top_angle);
  CHKERRQ(ierr);

  user->phys->blasius_ctx->weakT     = weakT;
  user->phys->blasius_ctx->Uinf      = Uinf;
  user->phys->blasius_ctx->delta0    = delta0;
  user->phys->blasius_ctx->theta0    = theta0;
  user->phys->blasius_ctx->P0        = P0;
  user->phys->blasius_ctx->implicit  = user->phys->implicit;

  user->phys->newtonian_ig_ctx->mu = mu;
  user->phys->blasius_ctx->newtonian_ctx = *user->phys->newtonian_ig_ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode SetupContext_BLASIUS(Ceed ceed, CeedData ceed_data,
                                    AppCtx app_ctx, SetupContext setup_ctx, Physics phys) {
  PetscFunctionBeginUser;
  PetscInt ierr;
  CeedQFunctionContextCreate(ceed, &ceed_data->setup_context);
  CeedQFunctionContextSetData(ceed_data->setup_context, CEED_MEM_HOST,
                              CEED_USE_POINTER,
                              sizeof(*setup_ctx), setup_ctx);
  ierr = SetupContext_NEWTONIAN_IG(ceed, ceed_data, app_ctx, setup_ctx, phys);
  CHKERRQ(ierr);

  CeedQFunctionContextCreate(ceed, &ceed_data->blasius_context);
  CeedQFunctionContextSetData(ceed_data->blasius_context, CEED_MEM_HOST,
                              CEED_USE_POINTER, sizeof(*phys->blasius_ctx), phys->blasius_ctx);
  phys->has_neumann = PETSC_TRUE;
  if (ceed_data->qf_ics)
    CeedQFunctionSetContext(ceed_data->qf_ics, ceed_data->blasius_context);
  if (ceed_data->qf_apply_inflow)
    CeedQFunctionSetContext(ceed_data->qf_apply_inflow, ceed_data->blasius_context);
  if (ceed_data->qf_apply_outflow)
    CeedQFunctionSetContext(ceed_data->qf_apply_outflow,
                            ceed_data->blasius_context);
  PetscFunctionReturn(0);
}

