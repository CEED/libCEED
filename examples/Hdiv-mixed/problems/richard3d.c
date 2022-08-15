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
/// Utility functions for setting up Richard problem in 3D

#include "../include/register-problem.h"
#include "../qfunctions/richard-system3d.h"
#include "../qfunctions/richard-true3d.h"
#include "../qfunctions/richard-ics3d.h"
#include "../qfunctions/darcy-error3d.h"
#include "../qfunctions/post-processing3d.h"
//#include "../qfunctions/pressure-boundary2d.h"
#include "petscsystypes.h"

PetscErrorCode Hdiv_RICHARD3D(Ceed ceed, ProblemData problem_data, DM dm,
                              void *ctx) {
  AppCtx               app_ctx = *(AppCtx *)ctx;
  RICHARDContext       richard_ctx;
  CeedQFunctionContext richard_context;

  PetscFunctionBeginUser;

  PetscCall( PetscCalloc1(1, &richard_ctx) );

  // ------------------------------------------------------
  //               SET UP POISSON_QUAD2D
  // ------------------------------------------------------
  problem_data->dim                     = 3;
  problem_data->elem_node               = 8;
  problem_data->q_data_size_face        = 4;
  problem_data->quadrature_mode         = CEED_GAUSS;
  problem_data->true_solution           = RichardTrue3D;
  problem_data->true_solution_loc       = RichardTrue3D_loc;
  problem_data->rhs_u0                  = RichardRhsU03D;
  problem_data->rhs_u0_loc              = RichardRhsU03D_loc;
  problem_data->ics_u                   = RichardICsU3D;
  problem_data->ics_u_loc               = RichardICsU3D_loc;
  problem_data->rhs_p0                  = RichardRhsP03D;
  problem_data->rhs_p0_loc              = RichardRhsP03D_loc;
  problem_data->ics_p                   = RichardICsP3D;
  problem_data->ics_p_loc               = RichardICsP3D_loc;
  problem_data->residual                = RichardSystem3D;
  problem_data->residual_loc            = RichardSystem3D_loc;
  //problem_data->jacobian                = JacobianRichardSystem2D;
  //problem_data->jacobian_loc            = JacobianRichardSystem2D_loc;
  problem_data->error                   = DarcyError3D;
  problem_data->error_loc               = DarcyError3D_loc;
  //problem_data->bc_pressure             = BCPressure2D;
  //problem_data->bc_pressure_loc         = BCPressure2D_loc;
  problem_data->post_rhs                = PostProcessingRhs3D;
  problem_data->post_rhs_loc            = PostProcessingRhs3D_loc;
  problem_data->post_mass               = PostProcessingMass3D;
  problem_data->post_mass_loc           = PostProcessingMass3D_loc;
  problem_data->has_ts                  = PETSC_TRUE;
  problem_data->view_solution           = app_ctx->view_solution;

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
  app_ctx->t_final = 0.5;
  PetscCall( PetscOptionsScalar("-t_final", "End time", NULL,
                                app_ctx->t_final, &app_ctx->t_final, NULL));
  PetscOptionsEnd();

  PetscReal domain_min[3], domain_max[3], domain_size[3];
  PetscCall( DMGetBoundingBox(dm, domain_min, domain_max) );
  for (PetscInt i=0; i<3; i++) domain_size[i] = domain_max[i] - domain_min[i];

  richard_ctx->kappa = kappa;
  richard_ctx->alpha_a = alpha_a;
  richard_ctx->b_a = b_a;
  richard_ctx->rho_a0 = rho_a0;
  richard_ctx->beta = beta;
  richard_ctx->g = g;
  richard_ctx->p0 = p0;
  richard_ctx->gamma = 5.;
  richard_ctx->t = 0.;
  richard_ctx->t_final = app_ctx->t_final;
  richard_ctx->lx = domain_size[0];
  richard_ctx->ly = domain_size[1];
  richard_ctx->lz = domain_size[2];

  CeedQFunctionContextCreate(ceed, &richard_context);
  CeedQFunctionContextSetData(richard_context, CEED_MEM_HOST, CEED_COPY_VALUES,
                              sizeof(*richard_ctx), richard_ctx);
  //CeedQFunctionContextSetDataDestroy(richard_context, CEED_MEM_HOST,
  //                                   FreeContextPetsc);
  CeedQFunctionContextRegisterDouble(richard_context, "time",
                                     offsetof(struct RICHARDContext_, t), 1, "current solver time");
  CeedQFunctionContextRegisterDouble(richard_context, "final_time",
                                     offsetof(struct RICHARDContext_, t_final), 1, "final time");
  CeedQFunctionContextRegisterDouble(richard_context, "time_step",
                                     offsetof(struct RICHARDContext_, dt), 1, "time step");
  problem_data->true_qfunction_ctx = richard_context;
  CeedQFunctionContextReferenceCopy(richard_context,
                                    &problem_data->rhs_u0_qfunction_ctx);
  CeedQFunctionContextReferenceCopy(richard_context,
                                    &problem_data->residual_qfunction_ctx);
  CeedQFunctionContextReferenceCopy(richard_context,
                                    &problem_data->error_qfunction_ctx);
  PetscCall( PetscFree(richard_ctx) );

  PetscFunctionReturn(0);
}
