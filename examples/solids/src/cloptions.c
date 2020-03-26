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
/// Command line option processing for solid mechanics example using PETSc

#include "../elasticity.h"

// -----------------------------------------------------------------------------
// Process command line options
// -----------------------------------------------------------------------------
// Process general command line options
PetscErrorCode ProcessCommandLineOptions(MPI_Comm comm, AppCtx appCtx) {
  PetscErrorCode ierr;
  PetscBool meshFileFlag = PETSC_FALSE;
  PetscBool degreeFalg   = PETSC_FALSE;
  PetscBool ceedFlag     = PETSC_FALSE;
  appCtx->problemChoice  = ELAS_LIN;       // Default - Linear Elasticity
  appCtx->degree         = 3;
  appCtx->forcingChoice  = FORCE_NONE;     // Default - no forcing term

  PetscFunctionBeginUser;

  ierr = PetscOptionsBegin(comm, NULL,
                           "Elasticity / Hyperelasticity in PETSc with libCEED",
                           NULL); CHKERRQ(ierr);

  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, appCtx->ceedResource, appCtx->ceedResource,
                            sizeof(appCtx->ceedResource), &ceedFlag);
  CHKERRQ(ierr);

  ierr = PetscOptionsString("-ceed_fine",
                            "CEED resource specifier for high order elements",
                            NULL, appCtx->ceedResourceFine,
                            appCtx->ceedResourceFine,
                            sizeof(appCtx->ceedResourceFine), NULL);
  CHKERRQ(ierr);

  ierr = PetscOptionsInt("-degree", "Polynomial degree of tensor product basis",
                         NULL, appCtx->degree, &appCtx->degree,
                         &degreeFalg); CHKERRQ(ierr);

  ierr = PetscOptionsString("-mesh", "Read mesh from file", NULL,
                            appCtx->meshFile, appCtx->meshFile,
                            sizeof(appCtx->meshFile), &meshFileFlag);
  CHKERRQ(ierr);

  ierr = PetscOptionsEnum("-problem",
                          "Solves Elasticity & Hyperelasticity Problems",
                          NULL, problemTypes, (PetscEnum)appCtx->problemChoice,
                          (PetscEnum *)&appCtx->problemChoice, NULL);
  CHKERRQ(ierr);

  appCtx->numIncrements = appCtx->problemChoice == ELAS_LIN ? 1 : 10;
  ierr = PetscOptionsInt("-num_steps", "Number of pseudo-time steps",
                         NULL, appCtx->numIncrements, &appCtx->numIncrements,
                         NULL); CHKERRQ(ierr);

  ierr = PetscOptionsEnum("-forcing", "Set forcing function option", NULL,
                          forcingTypes, (PetscEnum)appCtx->forcingChoice,
                          (PetscEnum *)&appCtx->forcingChoice, NULL);
  CHKERRQ(ierr);

  appCtx->bcZeroCount = 16;
  ierr = PetscOptionsIntArray("-bc_zero", "Face IDs to apply zero Dirichlet BC",
                              NULL, appCtx->bcZeroFaces, &appCtx->bcZeroCount,
                              NULL); CHKERRQ(ierr);
  appCtx->bcClampCount = 16;
  ierr = PetscOptionsIntArray("-bc_clamp",
                              "Face IDs to apply incremental Dirichlet BC",
                              NULL, appCtx->bcClampFaces, &appCtx->bcClampCount,
                              NULL); CHKERRQ(ierr);

  appCtx->bcClampMax = -1;
  ierr = PetscOptionsScalar("-bc_clamp_max",
                            "Maximum value to displace clamped boundary",
                            NULL, appCtx->bcClampMax, &appCtx->bcClampMax,
                            NULL); CHKERRQ(ierr);

  appCtx->multigridChoice = MULTIGRID_LOGARITHMIC;
  ierr = PetscOptionsEnum("-multigrid", "Set multigrid type option", NULL,
                          multigridTypes, (PetscEnum)appCtx->multigridChoice,
                          (PetscEnum *)&appCtx->multigridChoice, NULL);
  CHKERRQ(ierr);

  appCtx->testMode = PETSC_FALSE;
  ierr = PetscOptionsBool("-test",
                          "Testing mode (do not print unless error is large)",
                          NULL, appCtx->testMode, &(appCtx->testMode), NULL);
  CHKERRQ(ierr);

  appCtx->viewSoln = PETSC_FALSE;
  ierr = PetscOptionsBool("-view_soln",
                          "Write out solution vector for viewing",
                          NULL, appCtx->viewSoln, &(appCtx->viewSoln), NULL);
  CHKERRQ(ierr);

  ierr = PetscOptionsEnd(); CHKERRQ(ierr); // End of setting AppCtx

  // Check for all required values set and Exodus-II support
  if (!appCtx->testMode) {
    #if !defined(PETSC_HAVE_EXODUSII)
    SETERRQ(comm, PETSC_ERR_ARG_WRONG,
            "ExodusII support needed. Reconfigure your Arch with --download-exodusii");
    #endif
    if (!degreeFalg) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "-degree option needed");
    }
    if (!meshFileFlag) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "-mesh option needed (file)");
    }
    if (!(appCtx->bcZeroCount + appCtx->bcClampCount) &&
        appCtx->forcingChoice != FORCE_MMS) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "-boundary options needed");
    }
  } else {
    appCtx->forcingChoice = FORCE_MMS;
  }

  // Provide default ceed resource if not specified
  if (!ceedFlag) {
    const char *ceedResource = "/cpu/self";
    strncpy(appCtx->ceedResource, ceedResource, 10);
  }

  // Determine number of levels
  switch (appCtx->multigridChoice) {
  case MULTIGRID_LOGARITHMIC:
    appCtx->numLevels = ceil(log(appCtx->degree)/log(2)) + 1;
    break;
  case MULTIGRID_UNIFORM:
    appCtx->numLevels = appCtx->degree;
    break;
  case MULTIGRID_NONE:
    appCtx->numLevels = 1;
    break;
  }

  // Populate array of degrees for each level for multigrid
  ierr = PetscMalloc1(appCtx->numLevels, &(appCtx->levelDegrees));
  CHKERRQ(ierr);

  switch (appCtx->multigridChoice) {
  case MULTIGRID_LOGARITHMIC:
    for (int i = 0; i < appCtx->numLevels-1; i++)
      appCtx->levelDegrees[i] = pow(2,i);
    appCtx->levelDegrees[appCtx->numLevels-1] = appCtx->degree;
    break;
  case MULTIGRID_UNIFORM:
    for (int i = 0; i < appCtx->numLevels; i++)
      appCtx->levelDegrees[i] = i + 1;
    break;
  case MULTIGRID_NONE:
    appCtx->levelDegrees[0] = appCtx->degree;
    break;
  }

  PetscFunctionReturn(0);
};

// Process physics options
PetscErrorCode ProcessPhysics(MPI_Comm comm, Physics phys, Units units) {
  PetscErrorCode ierr;
  PetscBool nuFlag = PETSC_FALSE;
  PetscBool YoungFlag = PETSC_FALSE;
  phys->nu = 0;
  phys->E = 0;
  units->meter     = 1;        // 1 meter in scaled length units
  units->second    = 1;        // 1 second in scaled time units
  units->kilogram  = 1;        // 1 kilogram in scaled mass units

  PetscFunctionBeginUser;

  ierr = PetscOptionsBegin(comm, NULL,
                           "Elasticity / Hyperelasticity in PETSc with libCEED",
                           NULL); CHKERRQ(ierr);

  ierr = PetscOptionsScalar("-nu", "Poisson's ratio", NULL, phys->nu, &phys->nu,
                            &nuFlag); CHKERRQ(ierr);

  ierr = PetscOptionsScalar("-E", "Young's Modulus", NULL, phys->E, &phys->E,
                            &YoungFlag); CHKERRQ(ierr);

  ierr = PetscOptionsScalar("-units_meter", "1 meter in scaled length units",
                            NULL, units->meter, &units->meter, NULL);
  CHKERRQ(ierr);
  units->meter = fabs(units->meter);

  ierr = PetscOptionsScalar("-units_second", "1 second in scaled time units",
                            NULL, units->second, &units->second, NULL);
  CHKERRQ(ierr);
  units->second = fabs(units->second);

  ierr = PetscOptionsScalar("-units_kilogram", "1 kilogram in scaled mass units",
                            NULL, units->kilogram, &units->kilogram, NULL);
  CHKERRQ(ierr);
  units->kilogram = fabs(units->kilogram);

  ierr = PetscOptionsEnd(); CHKERRQ(ierr); // End of setting Physics

  // Check for all required options to be set
  if (!nuFlag) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "-nu option needed");
  }
  if (!YoungFlag) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "-E option needed");
  }

  // Define derived units
  units->Pascal = units->kilogram / (units->meter * PetscSqr(units->second));

  // Scale E to GPa
  phys->E *= units->Pascal;

  PetscFunctionReturn(0);
};
