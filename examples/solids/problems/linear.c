// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include "../include/structs.h"
#include "../include/setup-libceed.h"
#include "../problems/problems.h"
#include "../problems/neo-hookean.h"
#include "../qfunctions/common.h"
#include "../qfunctions/linear.h"
#include "../qfunctions/manufactured-true.h"

ProblemData linear_elasticity = {
  .setup_geo = SetupGeo,
  .setup_geo_loc = SetupGeo_loc,
  .q_data_size = 10,
  .quadrature_mode = CEED_GAUSS,
  .residual = ElasLinearF,
  .residual_loc = ElasLinearF_loc,
  .number_fields_stored = 0,
  .jacobian = ElasLineardF,
  .jacobian_loc = ElasLineardF_loc,
  .energy = ElasLinearEnergy,
  .energy_loc = ElasLinearEnergy_loc,
  .diagnostic = ElasLinearDiagnostic,
  .diagnostic_loc = ElasLinearDiagnostic_loc,
  .true_soln = MMSTrueSoln,
  .true_soln_loc = MMSTrueSoln_loc,
};

PetscErrorCode SetupLibceedFineLevel_ElasLinear(DM dm, DM dm_energy,
    DM dm_diagnostic, Ceed ceed, AppCtx app_ctx, CeedQFunctionContext phys_ctx,
    PetscInt fine_level, PetscInt num_comp_u, PetscInt U_g_size,
    PetscInt U_loc_size, CeedVector force_ceed, CeedVector neumann_ceed,
    CeedData *data) {
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = SetupLibceedFineLevel(dm, dm_energy, dm_diagnostic, ceed, app_ctx,
                               phys_ctx, linear_elasticity,
                               fine_level, num_comp_u, U_g_size, U_loc_size,
                               force_ceed, neumann_ceed, data); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

PetscErrorCode SetupLibceedLevel_ElasLinear(DM dm, Ceed ceed, AppCtx app_ctx,
    PetscInt level, PetscInt num_comp_u, PetscInt U_g_size, PetscInt U_loc_size,
    CeedVector fine_mult, CeedData *data) {
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = SetupLibceedLevel(dm, ceed, app_ctx, linear_elasticity,
                           level, num_comp_u, U_g_size, U_loc_size, fine_mult, data);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

PetscErrorCode ProblemRegister_ElasLinear(ProblemFunctions problem_functions) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&problem_functions->setupPhysics, "Linear",
                              PhysicsContext_NH); CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&problem_functions->setupSmootherPhysics, "Linear",
                              PhysicsSmootherContext_NH); CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&problem_functions->setupLibceedFineLevel, "Linear",
                              SetupLibceedFineLevel_ElasLinear); CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&problem_functions->setupLibceedLevel, "Linear",
                              SetupLibceedLevel_ElasLinear); CHKERRQ(ierr);
  PetscFunctionReturn(0);
};
