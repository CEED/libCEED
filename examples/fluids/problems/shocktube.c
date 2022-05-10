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
/// Utility functions for setting up SHOCKTUBE

#include "../navierstokes.h"
#include "../qfunctions/setupgeo.h"
#include "../qfunctions/shocktube.h"

PetscErrorCode NS_SHOCKTUBE(ProblemData *problem, DM dm, void *setup_ctx,
                            void *ctx) {
  SetupContext      setup_context = *(SetupContext *)setup_ctx;
  User              user = *(User *)ctx;
  MPI_Comm          comm = PETSC_COMM_WORLD;
  PetscBool         implicit;
  PetscBool         yzb;
  PetscInt          stab;
  PetscBool         has_curr_time = PETSC_FALSE;
  PetscInt          ierr;
  ShockTubeContext  shocktube_ctx;
  CeedQFunctionContext shocktube_context;


  PetscFunctionBeginUser;
  ierr = PetscCalloc1(1, &shocktube_ctx); CHKERRQ(ierr);

  // ------------------------------------------------------
  //               SET UP SHOCKTUBE
  // ------------------------------------------------------
  problem->dim                               = 3;
  problem->q_data_size_vol                   = 10;
  problem->q_data_size_sur                   = 4;
  problem->setup_vol.qfunction               = Setup;
  problem->setup_vol.qfunction_loc           = Setup_loc;
  problem->setup_sur.qfunction               = SetupBoundary;
  problem->setup_sur.qfunction_loc           = SetupBoundary_loc;
  problem->ics.qfunction                     = ICsShockTube;
  problem->ics.qfunction_loc                 = ICsShockTube_loc;
  problem->apply_vol_rhs.qfunction           = EulerShockTube;
  problem->apply_vol_rhs.qfunction_loc       = EulerShockTube_loc;
  problem->apply_vol_ifunction.qfunction     = NULL;
  problem->apply_vol_ifunction.qfunction_loc = NULL;
  problem->bc                                = Exact_ShockTube;
  problem->bc_ctx                            = setup_ctx;
  problem->non_zero_time                     = PETSC_FALSE;
  problem->print_info                        = PRINT_SHOCKTUBE;

  // ------------------------------------------------------
  //             Create the libCEED context
  // ------------------------------------------------------
  // Driver section initial conditions
  CeedScalar P_high          = 1.0;     // Pa
  CeedScalar rho_high        = 1.0;     // kg/m^3
  // Driven section initial conditions
  CeedScalar P_low           = 0.1;     // Pa
  CeedScalar rho_low         = 0.125;   // kg/m^3
  // Stabilization parameter
  CeedScalar c_tau           = 0.5;     // -, based on Hughes et al (2010)
  // Tuning parameters for the YZB shock capturing
  CeedScalar Cyzb            = 0.1;     // -, used in approximation of (Na),x
  CeedScalar Byzb            = 2.0;     // -, 1 for smooth shocks
  //                                          2 for sharp shocks
  PetscReal domain_min[3], domain_max[3], domain_size[3];
  ierr = DMGetBoundingBox(dm, domain_min, domain_max); CHKERRQ(ierr);
  for (int i=0; i<3; i++) domain_size[i] = domain_max[i] - domain_min[i];

  // ------------------------------------------------------
  //             Create the PETSc context
  // ------------------------------------------------------
  PetscScalar meter    = 1e-2; // 1 meter in scaled length units
  PetscScalar second   = 1e-2; // 1 second in scaled time units

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  PetscOptionsBegin(comm, NULL, "Options for SHOCKTUBE problem", NULL);

  // -- Numerical formulation options
  ierr = PetscOptionsBool("-implicit", "Use implicit (IFunction) formulation",
                          NULL, implicit=PETSC_FALSE, &implicit, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-stab", "Stabilization method", NULL,
                          StabilizationTypes, (PetscEnum)(stab = STAB_NONE),
                          (PetscEnum *)&stab, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-c_tau", "Stabilization constant",
                            NULL, c_tau, &c_tau, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-yzb", "Use YZB discontinuity capturing",
                          NULL, yzb=PETSC_FALSE, &yzb, NULL); CHKERRQ(ierr);

  // -- Units
  ierr = PetscOptionsScalar("-units_meter", "1 meter in scaled length units",
                            NULL, meter, &meter, NULL); CHKERRQ(ierr);
  meter = fabs(meter);
  ierr = PetscOptionsScalar("-units_second","1 second in scaled time units",
                            NULL, second, &second, NULL); CHKERRQ(ierr);
  second = fabs(second);

  // -- Warnings
  if (stab == STAB_SUPG) {
    ierr = PetscPrintf(comm,
                       "Warning! -stab supg not implemented for the shocktube problem. \n");
    CHKERRQ(ierr);
  }
  if (yzb && implicit) {
    ierr = PetscPrintf(comm,
                       "Warning! -yzb only implemented for explicit timestepping. \n");
    CHKERRQ(ierr);
  }


  PetscOptionsEnd();

  // ------------------------------------------------------
  //           Set up the PETSc context
  // ------------------------------------------------------
  user->units->meter  = meter;
  user->units->second = second;

  // ------------------------------------------------------
  //           Set up the libCEED context
  // ------------------------------------------------------
  // -- Scale variables to desired units
  for (int i=0; i<3; i++) {
    domain_size[i] *= meter;
    domain_min[i] *= meter;
  }
  problem->dm_scale = meter;
  CeedScalar mid_point = 0.5*(domain_size[0]+domain_min[0]);

  // -- Setup Context
  setup_context->lx        = domain_size[0];
  setup_context->ly        = domain_size[1];
  setup_context->lz        = domain_size[2];
  setup_context->mid_point = mid_point;
  setup_context->time      = 0.0;
  setup_context->P_high    = P_high;
  setup_context->rho_high  = rho_high;
  setup_context->P_low     = P_low;
  setup_context->rho_low   = rho_low;

  // -- QFunction Context
  user->phys->implicit                      = implicit;
  user->phys->has_curr_time                 = has_curr_time;
  shocktube_ctx->implicit       = implicit;
  shocktube_ctx->stabilization  = stab;
  shocktube_ctx->yzb            = yzb;
  shocktube_ctx->Cyzb           = Cyzb;
  shocktube_ctx->Byzb           = Byzb;
  shocktube_ctx->c_tau          = c_tau;

  CeedQFunctionContextCreate(user->ceed, &problem->ics.qfunction_context);
  CeedQFunctionContextSetData(problem->ics.qfunction_context, CEED_MEM_HOST,
                              CEED_USE_POINTER, sizeof(*setup_context), setup_context);

  CeedQFunctionContextCreate(user->ceed, &shocktube_context);
  CeedQFunctionContextSetData(shocktube_context, CEED_MEM_HOST,
                              CEED_USE_POINTER,
                              sizeof(*shocktube_ctx), shocktube_ctx);
  CeedQFunctionContextSetDataDestroy(shocktube_context, CEED_MEM_HOST,
                                     FreeContextPetsc);
  problem->apply_vol_rhs.qfunction_context = shocktube_context;
  PetscFunctionReturn(0);
}

PetscErrorCode PRINT_SHOCKTUBE(ProblemData *problem, AppCtx app_ctx) {
  MPI_Comm       comm = PETSC_COMM_WORLD;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  ierr = PetscPrintf(comm,
                     "  Problem:\n"
                     "    Problem Name                       : %s\n",
                     app_ctx->problem_name); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
