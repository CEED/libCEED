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
/// Utility functions for setting up ADVECTION

#include "../navierstokes.h"
#include "../qfunctions/setupgeo.h"
#include "../qfunctions/advection.h"

PetscErrorCode NS_ADVECTION(ProblemData *problem, void *setup_ctx, void *ctx) {
  WindType             wind_type;
  BubbleType           bubble_type;
  BubbleContinuityType bubble_continuity_type;
  StabilizationType    stab;
  SetupContext         setup_context = *(SetupContext *)setup_ctx;
  User                 user = *(User *)ctx;
  MPI_Comm             comm = PETSC_COMM_WORLD;
  PetscBool            implicit;
  PetscBool            has_curr_time = PETSC_FALSE;
  PetscInt             ierr;
  PetscFunctionBeginUser;

  ierr = PetscCalloc1(1, &user->phys->advection_ctx); CHKERRQ(ierr);

  // ------------------------------------------------------
  //               SET UP ADVECTION
  // ------------------------------------------------------
  problem->dim                     = 3;
  problem->q_data_size_vol         = 10;
  problem->q_data_size_sur         = 4;
  problem->setup_vol               = Setup;
  problem->setup_vol_loc           = Setup_loc;
  problem->setup_sur               = SetupBoundary;
  problem->setup_sur_loc           = SetupBoundary_loc;
  problem->ics                     = ICsAdvection;
  problem->ics_loc                 = ICsAdvection_loc;
  problem->apply_vol_rhs           = Advection;
  problem->apply_vol_rhs_loc       = Advection_loc;
  problem->apply_vol_ifunction     = IFunction_Advection;
  problem->apply_vol_ifunction_loc = IFunction_Advection_loc;
  problem->apply_sur               = Advection_Sur;
  problem->apply_sur_loc           = Advection_Sur_loc;
  problem->bc                      = Exact_Advection;
  problem->bc_func                 = BC_ADVECTION;
  problem->non_zero_time           = PETSC_FALSE;
  problem->print_info              = PRINT_ADVECTION;

  // ------------------------------------------------------
  //             Create the libCEED context
  // ------------------------------------------------------
  PetscScalar lx         = 8000.;      // m
  PetscScalar ly         = 8000.;      // m
  PetscScalar lz         = 4000.;      // m
  CeedScalar rc          = 1000.;      // m (Radius of bubble)
  CeedScalar CtauS       = 0.;         // dimensionless
  CeedScalar strong_form = 0.;         // [0,1]
  CeedScalar E_wind      = 1.e6;       // J
  PetscReal wind[3]      = {1., 0, 0}; // m/s

  // ------------------------------------------------------
  //             Create the PETSc context
  // ------------------------------------------------------
  PetscScalar meter    = 1e-2; // 1 meter in scaled length units
  PetscScalar kilogram = 1e-6; // 1 kilogram in scaled mass units
  PetscScalar second   = 1e-2; // 1 second in scaled time units
  PetscScalar Joule;

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  ierr = PetscOptionsBegin(comm, NULL, "Options for ADVECTION problem",
                           NULL); CHKERRQ(ierr);
  // -- Physics
  ierr = PetscOptionsScalar("-lx", "Length scale in x direction",
                            NULL, lx, &lx, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-ly", "Length scale in y direction",
                            NULL, ly, &ly, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-lz", "Length scale in z direction",
                            NULL, lz, &lz, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-rc", "Characteristic radius of thermal bubble",
                            NULL, rc, &rc, NULL); CHKERRQ(ierr);
  PetscBool translation;
  ierr = PetscOptionsEnum("-wind_type", "Wind type in Advection",
                          NULL, WindTypes,
                          (PetscEnum)(wind_type = WIND_ROTATION),
                          (PetscEnum *)&wind_type, &translation); CHKERRQ(ierr);
  if (translation) user->phys->has_neumann = PETSC_TRUE;
  PetscInt n = problem->dim;
  PetscBool user_wind;
  ierr = PetscOptionsRealArray("-wind_translation", "Constant wind vector",
                               NULL, wind, &n, &user_wind); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-CtauS",
                            "Scale coefficient for tau (nondimensional)",
                            NULL, CtauS, &CtauS, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-strong_form",
                            "Strong (1) or weak/integrated by parts (0) advection residual",
                            NULL, strong_form, &strong_form, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-E_wind", "Total energy of inflow wind",
                            NULL, E_wind, &E_wind, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-bubble_type", "Sphere (3D) or cylinder (2D)",
                          NULL, BubbleTypes,
                          (PetscEnum)(bubble_type = BUBBLE_SPHERE),
                          (PetscEnum *)&bubble_type, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-bubble_continuity", "Smooth, back_sharp, or thick",
                          NULL, BubbleContinuityTypes,
                          (PetscEnum)(bubble_continuity_type = BUBBLE_CONTINUITY_SMOOTH),
                          (PetscEnum *)&bubble_continuity_type, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-stab", "Stabilization method", NULL,
                          StabilizationTypes, (PetscEnum)(stab = STAB_NONE),
                          (PetscEnum *)&stab, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-implicit", "Use implicit (IFunction) formulation",
                          NULL, implicit=PETSC_FALSE, &implicit, NULL);
  CHKERRQ(ierr);

  // -- Units
  ierr = PetscOptionsScalar("-units_meter", "1 meter in scaled length units",
                            NULL, meter, &meter, NULL); CHKERRQ(ierr);
  meter = fabs(meter);
  ierr = PetscOptionsScalar("-units_kilogram","1 kilogram in scaled mass units",
                            NULL, kilogram, &kilogram, NULL); CHKERRQ(ierr);
  kilogram = fabs(kilogram);
  ierr = PetscOptionsScalar("-units_second","1 second in scaled time units",
                            NULL, second, &second, NULL); CHKERRQ(ierr);
  second = fabs(second);

  // -- Warnings
  if (wind_type == WIND_ROTATION && user_wind) {
    ierr = PetscPrintf(comm,
                       "Warning! Use -wind_translation only with -wind_type translation\n");
    CHKERRQ(ierr);
  }
  if (wind_type == WIND_TRANSLATION
      && bubble_type == BUBBLE_CYLINDER && wind[2] != 0.) {
    wind[2] = 0;
    ierr = PetscPrintf(comm,
                       "Warning! Background wind in the z direction should be zero (-wind_translation x,x,0) with -bubble_type cylinder\n");
    CHKERRQ(ierr);
  }
  if (stab == STAB_NONE && CtauS != 0) {
    ierr = PetscPrintf(comm,
                       "Warning! Use -CtauS only with -stab su or -stab supg\n");
    CHKERRQ(ierr);
  }
  if (stab == STAB_SUPG && !implicit) {
    ierr = PetscPrintf(comm,
                       "Warning! Use -stab supg only with -implicit\n");
    CHKERRQ(ierr);
  }

  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // ------------------------------------------------------
  //           Set up the PETSc context
  // ------------------------------------------------------
  // -- Define derived units
  Joule = kilogram * PetscSqr(meter) / PetscSqr(second);

  user->units->meter    = meter;
  user->units->kilogram = kilogram;
  user->units->second   = second;
  user->units->Joule    = Joule;

  // ------------------------------------------------------
  //           Set up the libCEED context
  // ------------------------------------------------------
  // -- Scale variables to desired units
  E_wind *= Joule;
  lx = fabs(lx) * meter;
  ly = fabs(ly) * meter;
  lz = fabs(lz) * meter;
  rc = fabs(rc) * meter;

  // -- Setup Context
  setup_context->rc                     = rc;
  setup_context->lx                     = lx;
  setup_context->ly                     = ly;
  setup_context->lz                     = lz;
  setup_context->wind[0]                = wind[0];
  setup_context->wind[1]                = wind[1];
  setup_context->wind[2]                = wind[2];
  setup_context->wind_type              = wind_type;
  setup_context->bubble_type            = bubble_type;
  setup_context->bubble_continuity_type = bubble_continuity_type;
  setup_context->time = 0;

  // -- QFunction Context
  user->phys->stab                         = stab;
  user->phys->wind_type                    = wind_type;
  user->phys->bubble_type                  = bubble_type;
  user->phys->bubble_continuity_type       = bubble_continuity_type;
  //  if passed correctly
  user->phys->implicit                     = implicit;
  user->phys->has_curr_time                = has_curr_time;
  user->phys->advection_ctx->CtauS         = CtauS;
  user->phys->advection_ctx->E_wind        = E_wind;
  user->phys->advection_ctx->implicit      = implicit;
  user->phys->advection_ctx->strong_form   = strong_form;
  user->phys->advection_ctx->stabilization = stab;

  PetscFunctionReturn(0);
}

PetscErrorCode BC_ADVECTION(DM dm, SimpleBC bc, Physics phys,
                            void *setup_ctx) {
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // Define boundary conditions
  if (phys->wind_type == WIND_TRANSLATION) {
    bc->num_wall = bc->num_slip[2] = 0;
  } else if (phys->wind_type == WIND_ROTATION &&
             phys->bubble_type == BUBBLE_CYLINDER) {
    bc->num_slip[2] = 2; bc->slips[2][0] = 1; bc->slips[2][1] = 2;
    bc->num_wall = 4;
    bc->walls[0] = 3; bc->walls[1] = 4; bc->walls[2] = 5; bc->walls[3] = 6;
  } else {
    bc->num_slip[2] = 0;
    bc->num_wall = 6;
    bc->walls[0] = 1; bc->walls[1] = 2; bc->walls[2] = 3;
    bc->walls[3] = 4; bc->walls[4] = 5; bc->walls[5] = 6;
  }

  {
    // Set slip boundary conditions
    DMLabel label;
    ierr = DMGetLabel(dm, "Face Sets", &label); CHKERRQ(ierr);
    PetscInt comps[1] = {3};
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipz", label, "Face Sets",
                         bc->num_slip[2], bc->slips[2], 0, 1, comps,
                         (void(*)(void))NULL, NULL, setup_ctx, NULL);
    CHKERRQ(ierr);
  }

  // Set wall boundary conditions
  //   zero energy density and zero flux
  {
    DMLabel  label;
    PetscInt comps[1] = {4};
    ierr = DMGetLabel(dm, "Face Sets", &label); CHKERRQ(ierr);
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, "Face Sets",
                         bc->num_wall, bc->walls, 0,
                         1, comps, (void(*)(void))Exact_Advection, NULL,
                         setup_ctx, NULL); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode PRINT_ADVECTION(Physics phys, SetupContext setup_ctx,
                               AppCtx app_ctx) {
  MPI_Comm       comm = PETSC_COMM_WORLD;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  ierr = PetscPrintf(comm,
                     "  Problem:\n"
                     "    Problem Name                       : %s\n"
                     "    Stabilization                      : %s\n"
                     "    Bubble Type                        : %s (%dD)\n"
                     "    Bubble Continuity                  : %s\n"
                     "    Wind Type                          : %s\n",
                     app_ctx->problem_name, StabilizationTypes[phys->stab],
                     BubbleTypes[phys->bubble_type],
                     phys->bubble_type == BUBBLE_SPHERE ? 3 : 2,
                     BubbleContinuityTypes[phys->bubble_continuity_type],
                     WindTypes[phys->wind_type]); CHKERRQ(ierr);

  if (phys->wind_type == WIND_TRANSLATION) {
    ierr = PetscPrintf(comm,
                       "    Background Wind                    : %f,%f,%f\n",
                       setup_ctx->wind[0], setup_ctx->wind[1], setup_ctx->wind[2]); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
