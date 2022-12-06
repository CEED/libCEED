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
/// Command line option processing for mixed-elasticity example using PETSc

#include "../include/cl-options.h"

// Process general command line options
PetscErrorCode ProcessCommandLineOptions(AppCtx app_ctx) {
  PetscBool problem_flag = PETSC_FALSE;
  PetscBool ceed_flag    = PETSC_FALSE;
  PetscFunctionBeginUser;

  PetscOptionsBegin(app_ctx->comm, NULL, "mixed-elasticity examples in PETSc with libCEED", NULL);

  PetscCall(PetscOptionsString("-ceed", "CEED resource specifier", NULL, app_ctx->ceed_resource, app_ctx->ceed_resource,
                               sizeof(app_ctx->ceed_resource), &ceed_flag));

  // Provide default ceed resource if not specified
  if (!ceed_flag) {
    const char *ceed_resource = "/cpu/self";
    strncpy(app_ctx->ceed_resource, ceed_resource, 10);
  }

  PetscCall(PetscOptionsFList("-problem", "Problem to solve", NULL, app_ctx->problems, app_ctx->problem_name, app_ctx->problem_name,
                              sizeof(app_ctx->problem_name), &problem_flag));
  // Provide default problem if not specified
  if (!problem_flag) {
    const char *problem_name = "bp4-3d";
    strncpy(app_ctx->problem_name, problem_name, 16);
  }
  app_ctx->u_order = 2;  // order of basis for u field
  PetscCall(PetscOptionsInt("-u_order", "Polynomial degree of basis u", NULL, app_ctx->u_order, &app_ctx->u_order, NULL));

  app_ctx->p_order = 1;  // order of basis for p field
  PetscCall(PetscOptionsInt("-p_order", "Polynomial degree of basis p", NULL, app_ctx->p_order, &app_ctx->p_order, NULL));

  app_ctx->q_extra = 0;
  PetscCall(PetscOptionsInt("-q_extra", "Number of extra quadrature points", NULL, app_ctx->q_extra, &app_ctx->q_extra, NULL));
  PetscOptionsEnd();

  PetscFunctionReturn(0);
}
