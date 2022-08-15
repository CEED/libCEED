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
/// Utility functions for setting up Darcy problem in 2D

#include "../include/register-problem.h"
#include "../qfunctions/darcy-true2d.h"
#include "../qfunctions/darcy-system2d.h"
#include "../qfunctions/darcy-error2d.h"
#include "../qfunctions/post-processing2d.h"
#include "../qfunctions/darcy-true-quartic2d.h"
#include "../qfunctions/darcy-system-quartic2d.h"
//#include "../qfunctions/pressure-boundary2d.h"

PetscErrorCode Hdiv_DARCY2D(Ceed ceed, ProblemData problem_data, DM dm,
                            void *ctx) {
  AppCtx               app_ctx = *(AppCtx *)ctx;
  DARCYContext         darcy_ctx;
  CeedQFunctionContext darcy_context;

  PetscFunctionBeginUser;

  PetscCall( PetscCalloc1(1, &darcy_ctx) );

  // ------------------------------------------------------
  //               SET UP POISSON_QUAD2D
  // ------------------------------------------------------
  problem_data->dim                     = 2;
  problem_data->elem_node               = 4;
  problem_data->q_data_size_face        = 3;
  problem_data->quadrature_mode         = CEED_GAUSS;
  problem_data->true_solution           = DarcyTrue2D;
  problem_data->true_solution_loc       = DarcyTrue2D_loc;
  problem_data->residual                = DarcySystem2D;
  problem_data->residual_loc            = DarcySystem2D_loc;
  problem_data->jacobian                = JacobianDarcySystem2D;
  problem_data->jacobian_loc            = JacobianDarcySystem2D_loc;
  problem_data->error                   = DarcyError2D;
  problem_data->error_loc               = DarcyError2D_loc;
  //problem_data->bc_pressure             = BCPressure2D;
  //problem_data->bc_pressure_loc         = BCPressure2D_loc;
  problem_data->post_rhs                = PostProcessingRhs2D;
  problem_data->post_rhs_loc            = PostProcessingRhs2D_loc;
  problem_data->post_mass               = PostProcessingMass2D;
  problem_data->post_mass_loc           = PostProcessingMass2D_loc;
  problem_data->has_ts                  = PETSC_FALSE;
  problem_data->view_solution           = app_ctx->view_solution;
  problem_data->quartic                 = app_ctx->quartic;

  if (app_ctx->quartic) {
    problem_data->true_solution           = DarcyTrueQuartic2D;
    problem_data->true_solution_loc       = DarcyTrueQuartic2D_loc;
    problem_data->residual                = DarcySystemQuartic2D;
    problem_data->residual_loc            = DarcySystemQuartic2D_loc;
    problem_data->jacobian                = JacobianDarcySystemQuartic2D;
    problem_data->jacobian_loc            = JacobianDarcySystemQuartic2D_loc;
  }

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  CeedScalar kappa = 10., rho_a0 = 998.2, g = 9.8, alpha_a = 1., b_a = 10.;
  PetscOptionsBegin(app_ctx->comm, NULL, "Options for Hdiv-mixed problem", NULL);
  PetscCall( PetscOptionsScalar("-kappa", "Hydraulic Conductivity", NULL,
                                kappa, &kappa, NULL));
  PetscCall( PetscOptionsScalar("-rho_a0", "Density at p0", NULL,
                                rho_a0, &rho_a0, NULL));
  PetscCall( PetscOptionsScalar("-alpha_a", "Parameter for relative permeability",
                                NULL,
                                alpha_a, &alpha_a, NULL));
  PetscCall( PetscOptionsScalar("-b_a", "Parameter for relative permeability",
                                NULL,
                                b_a, &b_a, NULL));
  PetscOptionsEnd();

  PetscReal domain_min[2], domain_max[2], domain_size[2];
  PetscCall( DMGetBoundingBox(dm, domain_min, domain_max) );
  for (PetscInt i=0; i<2; i++) domain_size[i] = domain_max[i] - domain_min[i];

  darcy_ctx->kappa = kappa;
  darcy_ctx->rho_a0 = rho_a0;
  darcy_ctx->g = g;
  darcy_ctx->alpha_a = alpha_a;
  darcy_ctx->b_a = b_a;
  darcy_ctx->lx = domain_size[0];
  darcy_ctx->ly = domain_size[1];

  CeedQFunctionContextCreate(ceed, &darcy_context);
  CeedQFunctionContextSetData(darcy_context, CEED_MEM_HOST, CEED_COPY_VALUES,
                              sizeof(*darcy_ctx), darcy_ctx);
  //CeedQFunctionContextSetDataDestroy(darcy_context, CEED_MEM_HOST,
  //                                   FreeContextPetsc);
  problem_data->true_qfunction_ctx = darcy_context;
  CeedQFunctionContextReferenceCopy(darcy_context,
                                    &problem_data->residual_qfunction_ctx);
  CeedQFunctionContextReferenceCopy(darcy_context,
                                    &problem_data->jacobian_qfunction_ctx);
  CeedQFunctionContextReferenceCopy(darcy_context,
                                    &problem_data->error_qfunction_ctx);

  PetscCall( PetscFree(darcy_ctx) );

  PetscFunctionReturn(0);
}
