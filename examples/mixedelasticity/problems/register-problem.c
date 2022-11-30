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

#include "../include/register-problem.h"

// Register problems to be available on the command line
PetscErrorCode RegisterProblems_MixedElasticity(AppCtx app_ctx) {
  app_ctx->problems = NULL;
  PetscFunctionBeginUser;
  // 1) bp4 (BP4 is created in bp4-2d.c/bp4-3d.c)
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "bp4-3d", BP4_3D));
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "bp4-2d", BP4_2D));

  // 2) linear elasticity (Linear is created in linear-2d.c/linear-3d.c)
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "linear-3d", Linear_3D));
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "linear-2d", Linear_2D));

  // 3) mixed-linear elasticity (Linear is created in mixed-linear-2d.c/mixed-linear-3d.c)
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "mixed-linear-2d", Mixed_Linear_2D));
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "mixed-linear-3d", Mixed_Linear_3D));

  PetscFunctionReturn(0);
}
