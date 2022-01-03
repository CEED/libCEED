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
/// Utility functions for setting up Richard problem in 2D

#include "../include/register-problem.h"
#include "../qfunctions/richard-system2d.h"
#include "../qfunctions/richard-mms2d.h"
#include "../qfunctions/pressure-boundary2d.h"
#include "petscsystypes.h"

PetscErrorCode Hdiv_RICHARD2D(Ceed ceed, ProblemData problem_data, void *ctx) {
  AppCtx               app_ctx = *(AppCtx *)ctx;
  RICHARDContext       richard_ctx;
  CeedQFunctionContext richard_context;

  PetscFunctionBeginUser;

  PetscCall( PetscCalloc1(1, &richard_ctx) );

  // ------------------------------------------------------
  //               SET UP POISSON_QUAD2D
  // ------------------------------------------------------
  problem_data->dim                     = 2;
  problem_data->elem_node               = 4;
  problem_data->q_data_size_face        = 3;
  problem_data->quadrature_mode         = CEED_GAUSS;
  problem_data->ics                     = RichardICs2D;
  problem_data->ics_loc                 = RichardICs2D_loc;
  //problem_data->force                   = RichardTrue2D;
  //problem_data->force_loc               = RichardTrue2D_loc;
  //problem_data->residual                = RichardSystem2D;
  //problem_data->residual_loc            = RichardSystem2D_loc;
  //problem_data->jacobian                = JacobianRichardSystem2D;
  //problem_data->jacobian_loc            = JacobianRichardSystem2D_loc;
  //problem_data->error                   = DarcyError2D;
  //problem_data->error_loc               = DarcyError2D_loc;
  problem_data->bc_pressure             = BCPressure2D;
  problem_data->bc_pressure_loc         = BCPressure2D_loc;
  problem_data->has_ts                  = PETSC_TRUE;

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  CeedScalar kappa = 10., alpha_a = 1., b_a = 10., rho_a0 = 998.2,
             beta = 0., g = 9.8, p0 = 101325;

  PetscOptionsBegin(app_ctx->comm, NULL, "Options for Hdiv-mixed problem", NULL);
  PetscCall( PetscOptionsScalar("-kappa", "Hydraulic Conductivity", NULL,
                                kappa, &kappa, NULL));
  PetscCall( PetscOptionsScalar("-alpha_a", "Parameter for relative permeability",
                                NULL,
                                alpha_a, &alpha_a, NULL));
  PetscCall( PetscOptionsScalar("-b_a", "Parameter for relative permeability",
                                NULL,
                                b_a, &b_a, NULL));
  PetscCall( PetscOptionsScalar("-rho_a0", "Density at p0", NULL,
                                rho_a0, &rho_a0, NULL));
  PetscCall( PetscOptionsScalar("-beta", "Water compressibility", NULL,
                                beta, &beta, NULL));
  PetscOptionsEnd();

  richard_ctx->kappa = kappa;
  richard_ctx->alpha_a = alpha_a;
  richard_ctx->b_a = b_a;
  richard_ctx->rho_a0 = rho_a0;
  richard_ctx->beta = beta;
  richard_ctx->g = g;
  richard_ctx->p0 = p0;
  richard_ctx->time = 0.;
  richard_ctx->gamma = 5.;
  CeedQFunctionContextCreate(ceed, &richard_context);
  CeedQFunctionContextSetData(richard_context, CEED_MEM_HOST, CEED_COPY_VALUES,
                              sizeof(*richard_ctx), richard_ctx);
  problem_data->qfunction_context = richard_context;
  CeedQFunctionContextSetDataDestroy(richard_context, CEED_MEM_HOST,
                                     FreeContextPetsc);

  PetscCall( PetscFree(richard_ctx) );
  PetscFunctionReturn(0);
}
