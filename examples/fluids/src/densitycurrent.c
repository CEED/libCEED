#include "../navierstokes.h"

PetscErrorCode NS_DENSITY_CURRENT(problemData *problem, void **ctxSetupData,
                                  void **ctx, void **ctxPhys) {
  PetscInt ierr;
  MPI_Comm comm = PETSC_COMM_WORLD;
  StabilizationType stab;
  SetupContext ctxSetup = *(SetupContext *)ctxSetupData;
  Units units = *(Units *)ctx;
  Physics ctxPhysData = *(Physics *)ctxPhys;
  ierr = PetscMalloc1(1, &ctxPhysData->ctxNSData); CHKERRQ(ierr);

  PetscFunctionBeginUser;
  // ------------------------------------------------------
  //               SET UP DENSITY_CURRENT
  // ------------------------------------------------------
  problem->dim                       = 3;
  problem->qdatasizeVol              = 10;
  problem->qdatasizeSur              = 4;
  problem->setupVol                  = Setup;
  problem->setupVol_loc              = Setup_loc;
  problem->setupSur                  = SetupBoundary;
  problem->setupSur_loc              = SetupBoundary_loc;
  problem->ics                       = ICsDC;
  problem->ics_loc                   = ICsDC_loc;
  problem->applyVol_rhs              = DC;
  problem->applyVol_rhs_loc          = DC_loc;
  problem->applyVol_ifunction        = IFunction_DC;
  problem->applyVol_ifunction_loc    = IFunction_DC_loc;
  problem->bc                        = BC_DENSITY_CURRENT;
  problem->non_zero_time             = PETSC_FALSE;

  // ------------------------------------------------------
  //             Create the libCEED context
  // ------------------------------------------------------
  CeedScalar theta0      = 300.;   // K
  CeedScalar thetaC      = -15.;   // K
  CeedScalar P0          = 1.e5;   // Pa
  CeedScalar N           = 0.01;   // 1/s
  CeedScalar cv          = 717.;   // J/(kg K)
  CeedScalar cp          = 1004.;  // J/(kg K)
  CeedScalar g           = 9.81;   // m/s^2
  CeedScalar lambda      = -2./3.;   // -
  CeedScalar mu          = 75.;      // Pa s, dynamic viscosity
  // mu = 75 is not physical for air, but is good for numerical stability
  CeedScalar k           = 0.02638;  // W/(m K)
  PetscScalar lx         = 8000.;  // m
  PetscScalar ly         = 8000.;  // m
  PetscScalar lz         = 4000.;  // m
  CeedScalar rc          = 1000.;  // m (Radius of bubble)
  PetscReal center[3], dc_axis[3] = {0, 0, 0};
  CeedScalar Rd;

  // ------------------------------------------------------
  //             Create the PETSc context
  // ------------------------------------------------------
  PetscScalar meter    = 1e-2;  // 1 meter in scaled length units
  PetscScalar kilogram = 1e-6;  // 1 kilogram in scaled mass units
  PetscScalar second   = 1e-2;  // 1 second in scaled time units
  PetscScalar Kelvin   = 1;     // 1 Kelvin in scaled temperature units
  PetscScalar WpermK, Pascal, JperkgK, mpersquareds;

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  ierr = PetscOptionsBegin(comm, NULL, "Options for DC",
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
  ierr = PetscOptionsEnum("-stab", "Stabilization method", NULL,
                          StabilizationTypes, (PetscEnum)(stab = STAB_NONE),
                          (PetscEnum *)&stab, NULL); CHKERRQ(ierr);

  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // ------------------------------------------------------
  //           Set up the PETSc context
  // ------------------------------------------------------
  // -- Define derived units
  Pascal = kilogram / (meter * PetscSqr(second));
  JperkgK =  PetscSqr(meter) / (PetscSqr(second) * Kelvin);
  mpersquareds = meter / PetscSqr(second);
  WpermK = kilogram * meter / (pow(second,3) * Kelvin);

  units->meter = meter;
  units->kilogram = kilogram;
  units->second = second;
  units->Kelvin = Kelvin;
  units->Pascal = Pascal;
  units->JperkgK = JperkgK;
  units->mpersquareds = mpersquareds;
  units->WpermK = WpermK;

  // ------------------------------------------------------
  //           Set up the libCEED context
  // ------------------------------------------------------
  // -- Scale variables to desired units
  theta0 *= Kelvin;
  thetaC *= Kelvin;
  P0 *= Pascal;
  N *= (1./second);
  cv *= JperkgK;
  cp *= JperkgK;
  Rd = cp - cv;
  g *= mpersquareds;
  mu *= Pascal * second;
  k *= WpermK;
  lx = fabs(lx) * meter;
  ly = fabs(ly) * meter;
  lz = fabs(lz) * meter;
  rc = fabs(rc) * meter;
  for (int i=0; i<3; i++) center[i] *= meter;

  // -- Setup Context
  ctxSetup->theta0     = theta0;
  ctxSetup->thetaC     = thetaC;
  ctxSetup->P0         = P0;
  ctxSetup->N          = N;
  ctxSetup->cv         = cv;
  ctxSetup->cp         = cp;
  ctxSetup->Rd         = Rd;
  ctxSetup->g          = g;
  ctxSetup->rc         = rc;
  ctxSetup->lx         = lx;
  ctxSetup->ly         = ly;
  ctxSetup->lz         = lz;
  ctxSetup->center[0]  = center[0];
  ctxSetup->center[1]  = center[1];
  ctxSetup->center[2]  = center[2];
  ctxSetup->dc_axis[0] = dc_axis[0];
  ctxSetup->dc_axis[1] = dc_axis[1];
  ctxSetup->dc_axis[2] = dc_axis[2];

  // -- QFunction Context
  ctxPhysData->ctxNSData->lambda = lambda;
  ctxPhysData->ctxNSData->mu = mu;
  ctxPhysData->ctxNSData->k = k;
  ctxPhysData->ctxNSData->cv = cv;
  ctxPhysData->ctxNSData->cp = cp;
  ctxPhysData->ctxNSData->g = g;
  ctxPhysData->ctxNSData->Rd = Rd;
  ctxPhysData->stab = stab;
  ctxPhysData->ctxNSData->stabilization = stab;

  PetscFunctionReturn(0);
}

PetscErrorCode BC_DENSITY_CURRENT(DM dm, SimpleBC bc, WindType wind_type,
                                  void *ctxSetupData) {

  PetscErrorCode ierr;
  PetscInt len;
  PetscBool flg;
  MPI_Comm comm = PETSC_COMM_WORLD;

  // Default boundary conditions
  //   slip bc on all faces and no wall bc
  bc->nslip[0] = bc->nslip[1] = bc->nslip[2] = 2;
  bc->slips[0][0] = 5;
  bc->slips[0][1] = 6;
  bc->slips[1][0] = 3;
  bc->slips[1][1] = 4;
  bc->slips[2][0] = 1;
  bc->slips[2][1] = 2;

  PetscFunctionBeginUser;
  // Parse command line options
  ierr = PetscOptionsBegin(comm, NULL, "Options for density_current",
                           NULL); CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-bc_wall",
                              "Use wall boundary conditions on this list of faces",
                              NULL, bc->walls,
                              (len = sizeof(bc->walls) / sizeof(bc->walls[0]),
                               &len), &flg); CHKERRQ(ierr);
  if (flg) {
    bc->nwall = len;
    // Using a no-slip wall disables automatic slip walls (they must be set explicitly)
    bc->nslip[0] = bc->nslip[1] = bc->nslip[2] = 0;
  }
  for (PetscInt j=0; j<3; j++) {
    const char *flags[3] = {"-bc_slip_x", "-bc_slip_y", "-bc_slip_z"};
    ierr = PetscOptionsIntArray(flags[j],
                                "Use slip boundary conditions on this list of faces",
                                NULL, bc->slips[j],
                                (len = sizeof(bc->slips[j]) / sizeof(bc->slips[j][0]),
                                 &len), &flg); CHKERRQ(ierr);
    if (flg) {
      bc->nslip[j] = len;
      bc->userbc = PETSC_TRUE;
    }
  }
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  {
    // Slip boundary conditions
    PetscInt comps[1] = {1};
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipx", "Face Sets", 0,
                         1, comps, (void(*)(void))NULL, NULL, bc->nslip[0],
                         bc->slips[0], ctxSetupData); CHKERRQ(ierr);
    comps[0] = 2;
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipy", "Face Sets", 0,
                         1, comps, (void(*)(void))NULL, NULL, bc->nslip[1],
                         bc->slips[1], ctxSetupData); CHKERRQ(ierr);
    comps[0] = 3;
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipz", "Face Sets", 0,
                         1, comps, (void(*)(void))NULL, NULL, bc->nslip[2],
                         bc->slips[2], ctxSetupData); CHKERRQ(ierr);
  }
  if (bc->userbc == PETSC_TRUE) {
    for (PetscInt c = 0; c < 3; c++) {
      for (PetscInt s = 0; s < bc->nslip[c]; s++) {
        for (PetscInt w = 0; w < bc->nwall; w++) {
          if (bc->slips[c][s] == bc->walls[w])
            SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
                     "Boundary condition already set on face %D!\n",
                     bc->walls[w]);
        }
      }
    }
  }
  {
    // Wall boundary conditions
    //   zero velocity and zero flux for mass density and energy density
    PetscInt comps[3] = {1, 2, 3};
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", "Face Sets", 0,
                         3, comps, (void(*)(void))Exact_DC, NULL,
                         bc->nwall, bc->walls, ctxSetupData); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}