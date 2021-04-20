#include "../navierstokes.h"

PetscErrorCode NS_DENSITY_CURRENT(ProblemData *problem, void *setup_ctx,
                                  void *ctx, void *phys) {
  SetupContext      setup_context = *(SetupContext *)setup_ctx;
  Units             units = *(Units *)ctx;
  Physics           phys_ctx = *(Physics *)phys;
  StabilizationType stab;
  MPI_Comm          comm = PETSC_COMM_WORLD;
  PetscBool         implicit;
  PetscBool         has_current_time = PETSC_FALSE;
  PetscInt          ierr;
  PetscFunctionBeginUser;

  ierr = PetscCalloc1(1, &phys_ctx->dc_ctx); CHKERRQ(ierr);

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
  problem->bc_func                 = BC_DENSITY_CURRENT;
  problem->non_zero_time           = PETSC_FALSE;

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
  PetscScalar lx    = 8000.;   // m
  PetscScalar ly    = 8000.;   // m
  PetscScalar lz    = 4000.;   // m
  CeedScalar rc     = 1000.;   // m (Radius of bubble)
  PetscReal center[3], dc_axis[3] = {0, 0, 0};
  CeedScalar Rd;

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
  ierr = PetscOptionsScalar("-lx", "Length scale in x direction",
                            NULL, lx, &lx, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-ly", "Length scale in y direction",
                            NULL, ly, &ly, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-lz", "Length scale in z direction",
                            NULL, lz, &lz, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-rc", "Characteristic radius of thermal bubble",
                            NULL, rc, &rc, NULL); CHKERRQ(ierr);
  PetscInt n = problem->dim;
  center[0] = 0.5 * lx;
  center[1] = 0.5 * ly;
  center[2] = 0.5 * lz;
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

  units->meter           = meter;
  units->kilogram        = kilogram;
  units->second          = second;
  units->Kelvin          = Kelvin;
  units->Pascal          = Pascal;
  units->J_per_kg_K      = J_per_kg_K;
  units->m_per_squared_s = m_per_squared_s;
  units->W_per_m_K       = W_per_m_K;

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
  Rd     = cp - cv;
  g      *= m_per_squared_s;
  mu     *= Pascal * second;
  k      *= W_per_m_K;
  lx     = fabs(lx) * meter;
  ly     = fabs(ly) * meter;
  lz     = fabs(lz) * meter;
  rc     = fabs(rc) * meter;
  for (int i=0; i<3; i++) center[i] *= meter;

  // -- Setup Context
  setup_context->theta0     = theta0;
  setup_context->thetaC     = thetaC;
  setup_context->P0         = P0;
  setup_context->N          = N;
  setup_context->cv         = cv;
  setup_context->cp         = cp;
  setup_context->Rd         = Rd;
  setup_context->g          = g;
  setup_context->rc         = rc;
  setup_context->lx         = lx;
  setup_context->ly         = ly;
  setup_context->lz         = lz;
  setup_context->center[0]  = center[0];
  setup_context->center[1]  = center[1];
  setup_context->center[2]  = center[2];
  setup_context->dc_axis[0] = dc_axis[0];
  setup_context->dc_axis[1] = dc_axis[1];
  setup_context->dc_axis[2] = dc_axis[2];
  setup_context->time       = 0;

  // -- QFunction Context
  phys_ctx->stab             = stab;
  phys_ctx->implicit         = implicit;
  phys_ctx->has_current_time = has_current_time;
  phys_ctx->dc_ctx->lambda   = lambda;
  phys_ctx->dc_ctx->mu       = mu;
  phys_ctx->dc_ctx->k        = k;
  phys_ctx->dc_ctx->cv       = cv;
  phys_ctx->dc_ctx->cp       = cp;
  phys_ctx->dc_ctx->g        = g;
  phys_ctx->dc_ctx->Rd       = Rd;
  phys_ctx->dc_ctx->stabilization = stab;

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
    // Slip boundary conditions
    DMLabel label;
    ierr = DMGetLabel(dm, "Face Sets", &label); CHKERRQ(ierr);
    PetscInt comps[1] = {1};
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipx", label, "Face Sets",
                         bc->num_slip[0], bc->slips[0], 0, 1, comps,
                         (void(*)(void))NULL, NULL, setup_ctx, NULL);
    CHKERRQ(ierr);
    comps[0] = 2;
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipy", label, "Face Sets",
                         bc->num_slip[1], bc->slips[1], 0, 1, comps,
                         (void(*)(void))NULL, NULL, setup_ctx, NULL);
    CHKERRQ(ierr);
    comps[0] = 3;
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipz", label, "Face Sets",
                         bc->num_slip[2], bc->slips[2], 0, 1, comps,
                         (void(*)(void))NULL, NULL, setup_ctx, NULL);
    CHKERRQ(ierr);
  }

  if (bc->user_bc == PETSC_TRUE) {
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

  // Wall boundary conditions
  //   zero velocity and zero flux for mass density and energy density
  {
    DMLabel  label;
    PetscInt comps[3] = {1, 2, 3};
    ierr = DMGetLabel(dm, "Face Sets", &label); CHKERRQ(ierr);
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, "Face Sets",
                         bc->num_wall, bc->walls, 0,
                         3, comps, (void(*)(void))Exact_DC, NULL,
                         setup_ctx, NULL); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
