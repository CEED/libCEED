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
#include "../qfunctions/bp4-3d.h"

#include "../include/register-problem.h"
#include "../qfunctions/compute-error-3d.h"
#include "../qfunctions/volumetric-geometry-3d.h"

PetscErrorCode BP4_3D(Ceed ceed, ProblemData problem_data, void *ctx) {
  // AppCtx app_ctx = *(AppCtx *)ctx;

  PetscFunctionBeginUser;

  // ------------------------------------------------------
  //               SET UP POISSON_QUAD2D
  // ------------------------------------------------------
  problem_data->quadrature_mode = CEED_GAUSS;
  problem_data->q_data_size     = 10;
  problem_data->setup_geo       = SetupVolumeGeometry3D;
  problem_data->setup_geo_loc   = SetupVolumeGeometry3D_loc;
  problem_data->setup_rhs       = SetupDiffRhs3D;
  problem_data->setup_rhs_loc   = SetupDiffRhs3D_loc;
  problem_data->residual        = SetupDiff3D;
  problem_data->residual_loc    = SetupDiff3D_loc;
  problem_data->error           = SetupError3D;
  problem_data->error_loc       = SetupError3D_loc;
  problem_data->bp4             = PETSC_TRUE;
  problem_data->linear          = PETSC_FALSE;
  PetscFunctionReturn(0);
}
