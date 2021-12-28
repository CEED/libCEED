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
/// Command line option processing for H(div) example using PETSc

#include "../include/cl-options.h"
#include "../include/problems.h"

// Register problems to be available on the command line
PetscErrorCode RegisterProblems_Hdiv(AppCtx app_ctx) {
  app_ctx->problems = NULL;
  PetscErrorCode   ierr;
  PetscFunctionBeginUser;
  // 1) poisson-quad2d (Hdiv_POISSON_MASS2D is created in poisson-mass2d.c)
  ierr = PetscFunctionListAdd(&app_ctx->problems, "poisson_mass2d",
                              Hdiv_POISSON_MASS2D); CHKERRQ(ierr);
  // 2) poisson-hex3d

  // 3) poisson-prism3d

  // 4) richard

  PetscFunctionReturn(0);
}

// Process general command line options
PetscErrorCode ProcessCommandLineOptions(MPI_Comm comm, AppCtx app_ctx) {

  PetscBool problem_flag = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  ierr = PetscOptionsBegin(comm, NULL,
                           "H(div) examples in PETSc with libCEED",
                           NULL); CHKERRQ(ierr);

  ierr = PetscOptionsFList("-problem", "Problem to solve", NULL,
                           app_ctx->problems,
                           app_ctx->problem_name, app_ctx->problem_name, sizeof(app_ctx->problem_name),
                           &problem_flag); CHKERRQ(ierr);

  app_ctx->degree = 1;
  ierr = PetscOptionsInt("-degree", "Polynomial degree of finite elements",
                         NULL, app_ctx->degree, &app_ctx->degree, NULL); CHKERRQ(ierr);

  app_ctx->q_extra = 2;
  ierr = PetscOptionsInt("-q_extra", "Number of extra quadrature points",
                         NULL, app_ctx->q_extra, &app_ctx->q_extra, NULL); CHKERRQ(ierr);

  // Provide default problem if not specified
  if (!problem_flag) {
    const char *problem_name = "poisson_mass2d";
    strncpy(app_ctx->problem_name, problem_name, 16);
  }

  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
