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
/// Utility functions for setting up POISSON_QUAD2D

#include "../include/setup-libceed.h"
#include "../include/problems.h"
#include "../qfunctions/setup-geo2d.h"
#include "../qfunctions/poisson-quad2d.h"
// Hdiv_POISSON_QUAD2D is registered in cl-option.c
PetscErrorCode Hdiv_POISSON_QUAD2D(ProblemData *problem_data, void *ctx) {
  User              user = *(User *)ctx;
  MPI_Comm          comm = PETSC_COMM_WORLD;
  PetscInt          ierr;
  PetscFunctionBeginUser;

  ierr = PetscCalloc1(1, &user->phys->pq2d_ctx); CHKERRQ(ierr);

  // ------------------------------------------------------
  //               SET UP POISSON_QUAD2D
  // ------------------------------------------------------
  problem_data->elem_node               = 4;
  problem_data->geo_data_size           = 5;
  problem_data->quadrature_mode         = CEED_GAUSS;
  problem_data->setup_geo               = SetupGeo2D;
  problem_data->setup_geo_loc           = SetupGeo2D_loc;
  problem_data->residual                = PoissonQuadF;
  problem_data->residual_loc            = PoissonQuadF_loc;
  problem_data->setup_rhs               = SetupRhs;
  problem_data->setup_rhs_loc           = SetupRhs_loc;
  problem_data->setup_ctx               = SetupContext_POISSON_QUAD2D;

  // ------------------------------------------------------
  //             Create the libCEED context
  // ------------------------------------------------------
  CeedScalar kappa = 1.;

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  ierr = PetscOptionsBegin(comm, NULL, "Options for DENSITY_CURRENT problem",
                           NULL); CHKERRQ(ierr);
  // -- Physics
  ierr = PetscOptionsScalar("-kappa", "hydraulic conductivity",
                            NULL, kappa, &kappa, NULL); CHKERRQ(ierr);

  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // -- QFunction Context
  user->phys->pq2d_ctx->kappa = kappa;

  PetscFunctionReturn(0);
}

PetscErrorCode SetupContext_POISSON_QUAD2D(Ceed ceed, CeedData ceed_data,
    Physics phys) {
  PetscFunctionBeginUser;

  CeedQFunctionContextCreate(ceed, &ceed_data->pq2d_context);
  CeedQFunctionContextSetData(ceed_data->pq2d_context, CEED_MEM_HOST,
                              CEED_USE_POINTER, sizeof(*phys->pq2d_ctx), phys->pq2d_ctx);
  CeedQFunctionSetContext(ceed_data->qf_residual, ceed_data->pq2d_context);

  PetscFunctionReturn(0);
}