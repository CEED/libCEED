// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../qfunctions/linear.h"

#include <ceed.h>
#include <petscsys.h>

#include "../include/setup-libceed.h"
#include "../include/structs.h"
#include "../problems/neo-hookean.h"
#include "../problems/problems.h"
#include "../qfunctions/common.h"
#include "../qfunctions/manufactured-true.h"

ProblemData linear_elasticity = {
    .setup_geo            = SetupGeo,
    .setup_geo_loc        = SetupGeo_loc,
    .q_data_size          = 10,
    .quadrature_mode      = CEED_GAUSS,
    .residual             = ElasResidual_Linear,
    .residual_loc         = ElasResidual_Linear_loc,
    .number_fields_stored = 0,
    .jacobian             = ElasJacobian_Linear,
    .jacobian_loc         = ElasJacobian_Linear_loc,
    .energy               = ElasEnergy_Linear,
    .energy_loc           = ElasEnergy_Linear_loc,
    .diagnostic           = ElasDiagnostic_Linear,
    .diagnostic_loc       = ElasDiagnostic_Linear_loc,
    .true_soln            = MMSTrueSoln,
    .true_soln_loc        = MMSTrueSoln_loc,
};

PetscErrorCode SetupLibceedFineLevel_ElasLinear(DM dm, DM dm_energy, DM dm_diagnostic, Ceed ceed, AppCtx app_ctx, CeedQFunctionContext phys_ctx,
                                                PetscInt fine_level, PetscInt num_comp_u, PetscInt U_g_size, PetscInt U_loc_size,
                                                CeedVector force_ceed, CeedVector neumann_ceed, CeedData *data) {
  PetscFunctionBegin;

  PetscCall(SetupLibceedFineLevel(dm, dm_energy, dm_diagnostic, ceed, app_ctx, phys_ctx, linear_elasticity, fine_level, num_comp_u, U_g_size,
                                  U_loc_size, force_ceed, neumann_ceed, data));

  PetscFunctionReturn(PETSC_SUCCESS);
};

PetscErrorCode SetupLibceedLevel_ElasLinear(DM dm, Ceed ceed, AppCtx app_ctx, PetscInt level, PetscInt num_comp_u, PetscInt U_g_size,
                                            PetscInt U_loc_size, CeedVector fine_mult, CeedData *data) {
  PetscFunctionBegin;

  PetscCall(SetupLibceedLevel(dm, ceed, app_ctx, linear_elasticity, level, num_comp_u, U_g_size, U_loc_size, fine_mult, data));

  PetscFunctionReturn(PETSC_SUCCESS);
};
