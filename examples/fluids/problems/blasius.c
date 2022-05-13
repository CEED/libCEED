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
#include "../qfunctions/stg_shur14.h"
#include "stg_shur14.h"

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

PetscErrorCode NS_BLASIUS(ProblemData *problem, DM dm, void *ctx) {

  PetscInt ierr;
  User           user     = *(User *)ctx;
  MPI_Comm       comm     = PETSC_COMM_WORLD;
  PetscBool      stg_bool = PETSC_FALSE;
  BlasiusContext blasius_ctx;
  NewtonianIdealGasContext newtonian_ig_ctx;
  STGShur14Context stg_shur14_ctx;
  CeedQFunctionContext blasius_context;
  CeedQFunctionContext stg_shur14_context;

  PetscFunctionBeginUser;
  ierr = NS_NEWTONIAN_IG(problem, dm, ctx); CHKERRQ(ierr);
  ierr = PetscCalloc1(1, &blasius_ctx); CHKERRQ(ierr);

  // ------------------------------------------------------
  //               SET UP Blasius
  // ------------------------------------------------------
  CeedQFunctionContextDestroy(&problem->ics.qfunction_context);
  problem->ics.qfunction               = ICsBlasius;
  problem->ics.qfunction_loc           = ICsBlasius_loc;
  problem->apply_outflow.qfunction     = Blasius_Outflow;
  problem->apply_outflow.qfunction_loc = Blasius_Outflow_loc;

  // CeedScalar mu = .04; // Pa s, dynamic viscosity
  CeedScalar Uinf          = 40;   // m/s
  CeedScalar delta0        = 4.2e-4;    // m
  PetscReal  refine_height = 5.9e-4;    // m
  PetscReal  growth        = 1.08; // [-]
  PetscInt   Ndelta        = 45;   // [-]
  PetscReal  top_angle     = 5;    // degrees
  CeedScalar theta0        = 288.; // K
  CeedScalar P0            = 1.01e5; // Pa
  PetscBool  weakT         = PETSC_FALSE; // weak density or temperature

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
  ierr = PetscOptionsBool("-stg", "Use STG inflow boundary condition",
                          NULL, stg_bool, &stg_bool, NULL); CHKERRQ(ierr);
  PetscOptionsEnd();

  if (!stg_bool) {
    problem->apply_inflow.qfunction     = Blasius_Inflow;
    problem->apply_inflow.qfunction_loc = Blasius_Inflow_loc;
  } else {
    problem->apply_inflow.qfunction     = STGShur14_Inflow;
    problem->apply_inflow.qfunction_loc = STGShur14_Inflow_loc;
  }

  PetscScalar meter           = user->units->meter;
  PetscScalar second          = user->units->second;
  PetscScalar Kelvin          = user->units->Kelvin;
  PetscScalar Pascal          = user->units->Pascal;

  theta0 *= Kelvin;
  P0     *= Pascal;
  Uinf   *= meter / second;
  delta0 *= meter;

  ierr = modifyMesh(dm, problem->dim, growth, Ndelta, refine_height, top_angle);
  CHKERRQ(ierr);

  // Some properties depend on parameters from NewtonianIdealGas
  CeedQFunctionContextGetData(problem->apply_vol_rhs.qfunction_context,
                              CEED_MEM_HOST, &newtonian_ig_ctx);

  blasius_ctx->weakT     = weakT;
  blasius_ctx->Uinf      = Uinf;
  blasius_ctx->delta0    = delta0;
  blasius_ctx->theta0    = theta0;
  blasius_ctx->P0        = P0;
  blasius_ctx->implicit  = user->phys->implicit;
  blasius_ctx->newtonian_ctx = *newtonian_ig_ctx;
  if (stg_bool) {
    ierr = CreateSTGContext(comm, dm, &stg_shur14_ctx,
                            newtonian_ig_ctx,
                            user->phys->implicit, theta0);
    CHKERRQ(ierr);
  }
  CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfunction_context,
                                  &newtonian_ig_ctx);

  CeedQFunctionContextCreate(user->ceed, &blasius_context);
  CeedQFunctionContextSetData(blasius_context, CEED_MEM_HOST,
                              CEED_USE_POINTER,
                              sizeof(*blasius_ctx), blasius_ctx);
  CeedQFunctionContextSetDataDestroy(blasius_context, CEED_MEM_HOST,
                                     FreeContextPetsc);

  problem->ics.qfunction_context = blasius_context;
  if (!stg_bool) {
    CeedQFunctionContextReferenceCopy(blasius_context,
                                      &problem->apply_inflow.qfunction_context);
  } else {
    CeedQFunctionContextCreate(user->ceed, &stg_shur14_context);
    CeedQFunctionContextSetData(stg_shur14_context, CEED_MEM_HOST,
                                CEED_USE_POINTER,
                                sizeof(*stg_shur14_ctx), stg_shur14_ctx);
    CeedQFunctionContextSetDataDestroy(stg_shur14_context, CEED_MEM_HOST,
                                       FreeContextPetsc);
    CeedQFunctionContextRegisterDouble(stg_shur14_context, "solution time",
                                       offsetof(struct STGShur14Context_, time), 1,
                                       "Phyiscal time of the solution");
    problem->apply_inflow.qfunction_context = stg_shur14_context;
  }
  CeedQFunctionContextReferenceCopy(blasius_context,
                                    &problem->apply_outflow.qfunction_context);
  PetscFunctionReturn(0);
}
