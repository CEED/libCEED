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
/// Utility functions for setting up Linear mixed-elasticity problem in 3D
#include "../include/register-problem.h"
#include "../qfunctions/compute-error_p-2d.h"
#include "../qfunctions/compute-error_u-2d.h"
#include "../qfunctions/mixed-linear-elasticity-2d.h"
#include "../qfunctions/volumetric-geometry-2d.h"

PetscErrorCode Mixed_Linear_2D(Ceed ceed, ProblemData problem_data, void *ctx) {
  AppCtx               app_ctx = *(AppCtx *)ctx;
  LINEARContext        linear_ctx;
  CeedQFunctionContext linear_context;

  PetscFunctionBeginUser;

  PetscCall(PetscCalloc1(1, &linear_ctx));
  // ------------------------------------------------------
  //               SET UP POISSON_QUAD2D
  // ------------------------------------------------------
  problem_data->quadrature_mode = CEED_GAUSS;
  problem_data->q_data_size     = 5;
  problem_data->setup_geo       = VolumeGeometry2D;
  problem_data->setup_geo_loc   = VolumeGeometry2D_loc;
  problem_data->setup_rhs       = MixedLinearRhs2D;
  problem_data->setup_rhs_loc   = MixedLinearRhs2D_loc;
  problem_data->residual        = MixedLinearResidual2D;
  problem_data->residual_loc    = MixedLinearResidual2D_loc;
  problem_data->jacobian        = MixedLinearJacobian2D;
  problem_data->jacobian_loc    = MixedLinearJacobian2D_loc;
  problem_data->error_u         = Error2Du;
  problem_data->error_u_loc     = Error2Du_loc;
  problem_data->error_p         = Error2Dp;
  problem_data->error_p_loc     = Error2Dp_loc;

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  CeedScalar E = 1., nu = 0.3;
  PetscOptionsBegin(app_ctx->comm, NULL, "Options for Linear Elasticity problem", NULL);
  PetscCall(PetscOptionsScalar("-E", "Young Modulus", NULL, E, &E, NULL));
  PetscCall(PetscOptionsScalar("-nu", "Poisson ratio", NULL, nu, &nu, NULL));
  PetscOptionsEnd();

  linear_ctx->E  = E;
  linear_ctx->nu = nu;

  CeedQFunctionContextCreate(ceed, &linear_context);
  CeedQFunctionContextSetData(linear_context, CEED_MEM_HOST, CEED_COPY_VALUES, sizeof(*linear_ctx), linear_ctx);

  problem_data->rhs_qfunction_ctx = linear_context;
  CeedQFunctionContextReferenceCopy(linear_context, &problem_data->residual_qfunction_ctx);
  CeedQFunctionContextReferenceCopy(linear_context, &problem_data->jacobian_qfunction_ctx);

  PetscCall(PetscFree(linear_ctx));

  PetscFunctionReturn(0);
}