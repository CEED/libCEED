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
/// Utility functions for setting up DENSITY_CURRENT

#include "../navierstokes.h"
#include "../qfunctions/setupgeo.h"
#include "../qfunctions/densitycurrent.h"

PetscErrorCode NS_DENSITY_CURRENT(ProblemData *problem, DM dm, void *setup_ctx,
                                  void *ctx) {
  SetupContext      setup_context = *(SetupContext *)setup_ctx;
  User              user = *(User *)ctx;
  StabilizationType stab;
  MPI_Comm          comm = PETSC_COMM_WORLD;
  PetscBool         implicit;
  PetscBool         has_curr_time = PETSC_FALSE;
  PetscInt          ierr;
  PetscFunctionBeginUser;

  ierr = PetscCalloc1(1, &user->phys->dc_ctx); CHKERRQ(ierr);

  // ------------------------------------------------------
  //               SET UP DENSITY_CURRENT
  // ------------------------------------------------------
  problem->dim                     = 3;
  problem->q_data_size_vol         = 10;
  problem->q_data_size_sur         = 4;
  problem->setup_vol               = Setup;
  problem->setup_vol_loc           = Setup_loc;
  problem->setup_sur               = SetupBoundary;
  problem->setup_sur_loc           = SetupBoundary_loc;
  problem->ics                     = ICsDC;
  problem->ics_loc                 = ICsDC_loc;
  problem->apply_vol_rhs           = DC;
  problem->apply_vol_rhs_loc       = DC_loc;
  problem->apply_vol_ifunction     = IFunction_DC;
  problem->apply_vol_ifunction_loc = IFunction_DC_loc;
  problem->bc                      = Exact_DC;
  problem->setup_ctx               = SetupContext_DENSITY_CURRENT;
  problem->bc_func                 = BC_DENSITY_CURRENT;
  problem->non_zero_time           = PETSC_FALSE;
  problem->print_info              = PRINT_DENSITY_CURRENT;

  // ------------------------------------------------------
  //             Create the libCEED context
  // ------------------------------------------------------
  CeedScalar theta0 = 300.;    // K
  CeedScalar thetaC = -15.;    // K
  CeedScalar P0     = 1.e5;    // Pa
  CeedScalar N      = 0.01;    // 1/s
  CeedScalar cv     = 717.;    // J/(kg K)
  CeedScalar cp     = 1004.;   // J/(kg K)
  CeedScalar g      = 9.81;    // m/s^2
  CeedScalar lambda = -2./3.;  // -
  CeedScalar mu     = 75.;     // Pa s, dynamic viscosity
  // mu = 75 is not physical for air, but is good for numerical stability
  CeedScalar k      = 0.02638; // W/(m K)
  CeedScalar c_tau  = 0.5;     // -
  // c_tau = 0.5 is reported as "optimal" in Hughes et al 2010
  CeedScalar rc     = 1000.;   // m (Radius of bubble)
  PetscReal center[3], dc_axis[3] = {0, 0, 0};
  PetscReal domain_min[3], domain_max[3], domain_size[3];
  ierr = DMGetBoundingBox(dm, domain_min, domain_max); CHKERRQ(ierr);
  for (int i=0; i<3; i++) domain_size[i] = domain_max[i] - domain_min[i];

  // ------------------------------------------------------
  //             Create the PETSc context
  // ------------------------------------------------------
  PetscScalar meter    = 1e-2;  // 1 meter in scaled length units
  PetscScalar kilogram = 1e-6;  // 1 kilogram in scaled mass units
  PetscScalar second   = 1e-2;  // 1 second in scaled time units
  PetscScalar Kelvin   = 1;     // 1 Kelvin in scaled temperature units
  PetscScalar W_per_m_K, Pascal, J_per_kg_K, m_per_squared_s;

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  ierr = PetscOptionsBegin(comm, NULL, "Options for DENSITY_CURRENT problem",
                           NULL); CHKERRQ(ierr);
  // -- Physics
  ierr = PetscOptionsScalar("-theta0", "Reference potential temperature",
                            NULL, theta0, &theta0, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-thetaC", "Perturbation of potential temperature",
                            NULL, thetaC, &thetaC, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-P0", "Atmospheric pressure",
                            NULL, P0, &P0, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-N", "Brunt-Vaisala frequency",
                            NULL, N, &N, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-cv", "Heat capacity at constant volume",
                            NULL, cv, &cv, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-cp", "Heat capacity at constant pressure",
                            NULL, cp, &cp, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-g", "Gravitational acceleration",
                            NULL, g, &g, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-lambda",
                            "Stokes hypothesis second viscosity coefficient",
                            NULL, lambda, &lambda, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-mu", "Shear dynamic viscosity coefficient",
                            NULL, mu, &mu, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-k", "Thermal conductivity",
                            NULL, k, &k, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-rc", "Characteristic radius of thermal bubble",
                            NULL, rc, &rc, NULL); CHKERRQ(ierr);
  for (int i=0; i<3; i++) center[i] = .5*domain_size[i];
  PetscInt n = problem->dim;
  ierr = PetscOptionsRealArray("-center", "Location of bubble center",
                               NULL, center, &n, NULL); CHKERRQ(ierr);
  n = problem->dim;
  ierr = PetscOptionsRealArray("-dc_axis",
                               "Axis of density current cylindrical anomaly, or {0,0,0} for spherically symmetric",
                               NULL, dc_axis, &n, NULL); CHKERRQ(ierr);
  {
    PetscReal norm = PetscSqrtReal(PetscSqr(dc_axis[0]) + PetscSqr(dc_axis[1]) +
                                   PetscSqr(dc_axis[2]));
    if (norm > 0) {
      for (int i=0; i<3; i++)  dc_axis[i] /= norm;
    }
  }
  ierr = PetscOptionsEnum("-stab", "Stabilization method", NULL,
                          StabilizationTypes, (PetscEnum)(stab = STAB_NONE),
                          (PetscEnum *)&stab, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-c_tau", "Stabilization constant",
                            NULL, c_tau, &c_tau, NULL); CHKERRQ(ierr);
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
  ierr = PetscOptionsScalar("-units_Kelvin",
                            "1 Kelvin in scaled temperature units",
                            NULL, Kelvin, &Kelvin, NULL); CHKERRQ(ierr);
  Kelvin = fabs(Kelvin);

  // -- Warnings
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
  Pascal          = kilogram / (meter * PetscSqr(second));
  J_per_kg_K      =  PetscSqr(meter) / (PetscSqr(second) * Kelvin);
  m_per_squared_s = meter / PetscSqr(second);
  W_per_m_K       = kilogram * meter / (pow(second,3) * Kelvin);

  user->units->meter           = meter;
  user->units->kilogram        = kilogram;
  user->units->second          = second;
  user->units->Kelvin          = Kelvin;
  user->units->Pascal          = Pascal;
  user->units->J_per_kg_K      = J_per_kg_K;
  user->units->m_per_squared_s = m_per_squared_s;
  user->units->W_per_m_K       = W_per_m_K;

  // ------------------------------------------------------
  //           Set up the libCEED context
  // ------------------------------------------------------
  // -- Scale variables to desired units
  theta0 *= Kelvin;
  thetaC *= Kelvin;
  P0     *= Pascal;
  N      *= (1./second);
  cv     *= J_per_kg_K;
  cp     *= J_per_kg_K;
  g      *= m_per_squared_s;
  mu     *= Pascal * second;
  k      *= W_per_m_K;
  rc     = fabs(rc) * meter;
  for (int i=0; i<3; i++) domain_size[i] *= meter;
  for (int i=0; i<3; i++) center[i] *= meter;
  problem->dm_scale = meter;

  // -- Setup Context
  setup_context->theta0     = theta0;
  setup_context->thetaC     = thetaC;
  setup_context->P0         = P0;
  setup_context->N          = N;
  setup_context->cv         = cv;
  setup_context->cp         = cp;
  setup_context->g          = g;
  setup_context->rc         = rc;
  setup_context->lx         = domain_size[0];
  setup_context->ly         = domain_size[1];
  setup_context->lz         = domain_size[2];
  setup_context->center[0]  = center[0];
  setup_context->center[1]  = center[1];
  setup_context->center[2]  = center[2];
  setup_context->dc_axis[0] = dc_axis[0];
  setup_context->dc_axis[1] = dc_axis[1];
  setup_context->dc_axis[2] = dc_axis[2];
  setup_context->time       = 0;

  // -- QFunction Context
  user->phys->stab             = stab;
  user->phys->implicit         = implicit;
  user->phys->has_curr_time    = has_curr_time;
  user->phys->dc_ctx->lambda   = lambda;
  user->phys->dc_ctx->mu       = mu;
  user->phys->dc_ctx->k        = k;
  user->phys->dc_ctx->cv       = cv;
  user->phys->dc_ctx->cp       = cp;
  user->phys->dc_ctx->g        = g;
  user->phys->dc_ctx->c_tau    = c_tau;
  user->phys->dc_ctx->stabilization = stab;

  PetscFunctionReturn(0);
}

PetscErrorCode SetupContext_DENSITY_CURRENT(Ceed ceed, CeedData ceed_data,
    AppCtx app_ctx, SetupContext setup_ctx,
    Physics phys) {
  PetscFunctionBeginUser;

  CeedQFunctionContextCreate(ceed, &ceed_data->setup_context);
  CeedQFunctionContextSetData(ceed_data->setup_context, CEED_MEM_HOST,
                              CEED_USE_POINTER, sizeof(*setup_ctx), setup_ctx);
  CeedQFunctionSetContext(ceed_data->qf_ics, ceed_data->setup_context);
  CeedQFunctionContextCreate(ceed, &ceed_data->dc_context);
  CeedQFunctionContextSetData(ceed_data->dc_context, CEED_MEM_HOST,
                              CEED_USE_POINTER,
                              sizeof(*phys->dc_ctx), phys->dc_ctx);
  if (ceed_data->qf_rhs_vol)
    CeedQFunctionSetContext(ceed_data->qf_rhs_vol, ceed_data->dc_context);
  if (ceed_data->qf_ifunction_vol)
    CeedQFunctionSetContext(ceed_data->qf_ifunction_vol, ceed_data->dc_context);

  PetscFunctionReturn(0);
}

PetscErrorCode BC_DENSITY_CURRENT(DM dm, SimpleBC bc, Physics phys,
                                  void *setup_ctx) {

  PetscInt       len;
  PetscBool      flg;
  MPI_Comm       comm = PETSC_COMM_WORLD;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // Default boundary conditions
  //   slip bc on all faces and no wall bc
  bc->num_slip[0] = bc->num_slip[1] = bc->num_slip[2] = 2;
  bc->slips[0][0] = 5;
  bc->slips[0][1] = 6;
  bc->slips[1][0] = 3;
  bc->slips[1][1] = 4;
  bc->slips[2][0] = 1;
  bc->slips[2][1] = 2;

  // Parse command line options
  ierr = PetscOptionsBegin(comm, NULL, "Options for DENSITY_CURRENT BCs ",
                           NULL); CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-bc_wall",
                              "Use wall boundary conditions on this list of faces",
                              NULL, bc->walls,
                              (len = sizeof(bc->walls) / sizeof(bc->walls[0]),
                               &len), &flg); CHKERRQ(ierr);
  if (flg) {
    bc->num_wall = len;
    // Using a no-slip wall disables automatic slip walls (they must be set explicitly)
    bc->num_slip[0] = bc->num_slip[1] = bc->num_slip[2] = 0;
  }
  for (PetscInt j=0; j<3; j++) {
    const char *flags[3] = {"-bc_slip_x", "-bc_slip_y", "-bc_slip_z"};
    ierr = PetscOptionsIntArray(flags[j],
                                "Use slip boundary conditions on this list of faces",
                                NULL, bc->slips[j],
                                (len = sizeof(bc->slips[j]) / sizeof(bc->slips[j][0]),
                                 &len), &flg); CHKERRQ(ierr);
    if (flg) {
      bc->num_slip[j] = len;
      bc->user_bc = PETSC_TRUE;
    }
  }
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  {
    // Set slip boundary conditions
    DMLabel label;
    ierr = DMGetLabel(dm, "Face Sets", &label); CHKERRQ(ierr);
    PetscInt comps[1] = {1};
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipx", label,
                         bc->num_slip[0], bc->slips[0], 0, 1, comps,
                         (void(*)(void))NULL, NULL, setup_ctx, NULL);
    CHKERRQ(ierr);
    comps[0] = 2;
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipy", label,
                         bc->num_slip[1], bc->slips[1], 0, 1, comps,
                         (void(*)(void))NULL, NULL, setup_ctx, NULL);
    CHKERRQ(ierr);
    comps[0] = 3;
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipz", label,
                         bc->num_slip[2], bc->slips[2], 0, 1, comps,
                         (void(*)(void))NULL, NULL, setup_ctx, NULL);
    CHKERRQ(ierr);
  }

  if (bc->user_bc) {
    for (PetscInt c = 0; c < 3; c++) {
      for (PetscInt s = 0; s < bc->num_slip[c]; s++) {
        for (PetscInt w = 0; w < bc->num_wall; w++) {
          if (bc->slips[c][s] == bc->walls[w])
            SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
                     "Boundary condition already set on face %D!\n",
                     bc->walls[w]);
        }
      }
    }
  }

  // Set wall boundary conditions
  //   zero velocity and zero flux for mass density and energy density
  {
    DMLabel  label;
    PetscInt comps[3] = {1, 2, 3};
    ierr = DMGetLabel(dm, "Face Sets", &label); CHKERRQ(ierr);
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label,
                         bc->num_wall, bc->walls, 0, 3, comps,
                         (void(*)(void))Exact_DC, NULL,
                         setup_ctx, NULL); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode PRINT_DENSITY_CURRENT(Physics phys, SetupContext setup_ctx,
                                     AppCtx app_ctx) {
  MPI_Comm       comm = PETSC_COMM_WORLD;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  ierr = PetscPrintf(comm,
                     "  Problem:\n"
                     "    Problem Name                       : %s\n"
                     "    Stabilization                      : %s\n",
                     app_ctx->problem_name, StabilizationTypes[phys->stab]);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
