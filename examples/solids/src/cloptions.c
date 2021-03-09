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
  PetscBool ceedFlag     = PETSC_FALSE;

  PetscFunctionBeginUser;

  ierr = PetscOptionsBegin(comm, NULL,
                           "Elasticity / Hyperelasticity in PETSc with libCEED",
                           NULL); CHKERRQ(ierr);

  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, appCtx->ceedResource, appCtx->ceedResource,
                            sizeof(appCtx->ceedResource), &ceedFlag);
  CHKERRQ(ierr);

  ierr = PetscStrncpy(appCtx->outputdir, ".", 2);
  CHKERRQ(ierr); // Default - current directory
  ierr = PetscOptionsString("-output_dir", "Output directory",
                            NULL, appCtx->outputdir, appCtx->outputdir,
                            sizeof(appCtx->outputdir), NULL); CHKERRQ(ierr);

  appCtx->degree         = 3;
  ierr = PetscOptionsInt("-degree", "Polynomial degree of tensor product basis",
                         NULL, appCtx->degree, &appCtx->degree, NULL);
  CHKERRQ(ierr);

  appCtx->qextra         = 0;
  ierr = PetscOptionsInt("-qextra", "Number of extra quadrature points",
                         NULL, appCtx->qextra, &appCtx->qextra, NULL);
  CHKERRQ(ierr);

  ierr = PetscOptionsString("-mesh", "Read mesh from file", NULL,
                            appCtx->meshFile, appCtx->meshFile,
                            sizeof(appCtx->meshFile), NULL); CHKERRQ(ierr);

  appCtx->problemChoice  = ELAS_LINEAR;       // Default - Linear Elasticity
  ierr = PetscOptionsEnum("-problem",
                          "Solves Elasticity & Hyperelasticity Problems",
                          NULL, problemTypes, (PetscEnum)appCtx->problemChoice,
                          (PetscEnum *)&appCtx->problemChoice, NULL);
  CHKERRQ(ierr);

  appCtx->numIncrements = appCtx->problemChoice == ELAS_LINEAR ? 1 : 10;
  ierr = PetscOptionsInt("-num_steps", "Number of pseudo-time steps",
                         NULL, appCtx->numIncrements, &appCtx->numIncrements,
                         NULL); CHKERRQ(ierr);

  appCtx->forcingChoice  = FORCE_NONE;     // Default - no forcing term
  ierr = PetscOptionsEnum("-forcing", "Set forcing function option", NULL,
                          forcingTypes, (PetscEnum)appCtx->forcingChoice,
                          (PetscEnum *)&appCtx->forcingChoice, NULL);
  CHKERRQ(ierr);

  PetscInt maxn = 3;
  appCtx->forcingVector[0] = 0;
  appCtx->forcingVector[1] = -1;
  appCtx->forcingVector[2] = 0;
  ierr = PetscOptionsScalarArray("-forcing_vec",
                                 "Direction to apply constant force", NULL,
                                 appCtx->forcingVector, &maxn, NULL);
  CHKERRQ(ierr);

  if ((appCtx->problemChoice == ELAS_FSInitial_NH1
       || appCtx->problemChoice == ELAS_FSInitial_NH2
       || appCtx->problemChoice == ELAS_FSCurrent_NH1
       || appCtx->problemChoice == ELAS_FSCurrent_NH2) &&
      appCtx->forcingChoice == FORCE_CONST)
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP,
            "Cannot use constant forcing and finite strain formulation. "
            "Constant forcing in reference frame currently unavaliable.");

  // Dirichlet boundary conditions
  appCtx->bcClampCount = 16;
  ierr = PetscOptionsIntArray("-bc_clamp",
                              "Face IDs to apply incremental Dirichlet BC",
                              NULL, appCtx->bcClampFaces, &appCtx->bcClampCount,
                              NULL); CHKERRQ(ierr);
  // Set vector for each clamped BC
  for (PetscInt i = 0; i < appCtx->bcClampCount; i++) {
    // Translation vector
    char optionName[25];
    const size_t nclamp_params = sizeof(appCtx->bcClampMax[0])/sizeof(
                                   appCtx->bcClampMax[0][0]);
    for (PetscInt j = 0; j < nclamp_params; j++)
      appCtx->bcClampMax[i][j] = 0.;

    snprintf(optionName, sizeof optionName, "-bc_clamp_%d_translate",
             appCtx->bcClampFaces[i]);
    maxn = 3;
    ierr = PetscOptionsScalarArray(optionName,
                                   "Vector to translate clamped end by", NULL,
                                   appCtx->bcClampMax[i], &maxn, NULL);
    CHKERRQ(ierr);

    // Rotation vector
    maxn = 5;
    snprintf(optionName, sizeof optionName, "-bc_clamp_%d_rotate",
             appCtx->bcClampFaces[i]);
    ierr = PetscOptionsScalarArray(optionName,
                                   "Vector with axis of rotation and rotation, in radians",
                                   NULL, &appCtx->bcClampMax[i][3], &maxn, NULL);
    CHKERRQ(ierr);

    // Normalize
    PetscScalar norm = sqrt(appCtx->bcClampMax[i][3]*appCtx->bcClampMax[i][3] +
                            appCtx->bcClampMax[i][4]*appCtx->bcClampMax[i][4] +
                            appCtx->bcClampMax[i][5]*appCtx->bcClampMax[i][5]);
    if (fabs(norm) < 1e-16)
      norm = 1;
    for (PetscInt j = 0; j < 3; j++)
      appCtx->bcClampMax[i][3 + j] /= norm;
  }

  // Neumann boundary conditions
  appCtx->bcTractionCount = 16;
  ierr = PetscOptionsIntArray("-bc_traction",
                              "Face IDs to apply traction (Neumann) BC",
                              NULL, appCtx->bcTractionFaces,
                              &appCtx->bcTractionCount, NULL); CHKERRQ(ierr);
  // Set vector for each traction BC
  for (PetscInt i = 0; i < appCtx->bcTractionCount; i++) {
    // Translation vector
    char optionName[25];
    for (PetscInt j = 0; j < 3; j++)
      appCtx->bcTractionVector[i][j] = 0.;

    snprintf(optionName, sizeof optionName, "-bc_traction_%d",
             appCtx->bcTractionFaces[i]);
    maxn = 3;
    PetscBool set = false;
    ierr = PetscOptionsScalarArray(optionName,
                                   "Traction vector for constrained face", NULL,
                                   appCtx->bcTractionVector[i], &maxn, &set);
    CHKERRQ(ierr);

    if (!set)
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP,
              "Traction vector must be set for all traction boundary conditions.");
  }

  appCtx->multigridChoice = MULTIGRID_LOGARITHMIC;
  ierr = PetscOptionsEnum("-multigrid", "Set multigrid type option", NULL,
                          multigridTypes, (PetscEnum)appCtx->multigridChoice,
                          (PetscEnum *)&appCtx->multigridChoice, NULL);
  CHKERRQ(ierr);

  appCtx->nuSmoother = 0.;
  ierr = PetscOptionsScalar("-nu_smoother", "Poisson's ratio for smoother",
                            NULL, appCtx->nuSmoother, &appCtx->nuSmoother, NULL);
  CHKERRQ(ierr);

  appCtx->testMode = PETSC_FALSE;
  ierr = PetscOptionsBool("-test",
                          "Testing mode (do not print unless error is large)",
                          NULL, appCtx->testMode, &(appCtx->testMode), NULL);
  CHKERRQ(ierr);

  appCtx->viewSoln = PETSC_FALSE;
  ierr = PetscOptionsBool("-view_soln", "Write out solution vector for viewing",
                          NULL, appCtx->viewSoln, &(appCtx->viewSoln), NULL);
  CHKERRQ(ierr);

  appCtx->viewFinalSoln = PETSC_FALSE;
  ierr = PetscOptionsBool("-view_final_soln",
                          "Write out final solution vector for viewing",
                          NULL, appCtx->viewFinalSoln, &(appCtx->viewFinalSoln),
                          NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr); // End of setting AppCtx

  // Check for all required values set
  if (!appCtx->testMode) {
    if (!appCtx->bcClampCount && (appCtx->forcingChoice != FORCE_MMS)) {
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

  // Scale E to Pa
  phys->E *= units->Pascal;

  PetscFunctionReturn(0);
};
