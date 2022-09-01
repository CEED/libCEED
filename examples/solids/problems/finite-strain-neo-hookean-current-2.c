// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../qfunctions/finite-strain-neo-hookean-current-2.h"

#include <ceed.h>

#include "../include/setup-libceed.h"
#include "../include/structs.h"
#include "../problems/neo-hookean.h"
#include "../problems/problems.h"
#include "../qfunctions/common.h"

static const char *const field_names[] = {"dXdx", "tau", "lambda_log_J"};
static CeedInt           field_sizes[] = {9, 6, 1};

ProblemData finite_strain_neo_Hookean_current_2 = {
    .setup_geo            = SetupGeo,
    .setup_geo_loc        = SetupGeo_loc,
    .q_data_size          = 10,
    .quadrature_mode      = CEED_GAUSS,
    .residual             = ElasFSCurrentNH2F,
    .residual_loc         = ElasFSCurrentNH2F_loc,
    .number_fields_stored = 3,
    .field_names          = field_names,
    .field_sizes          = field_sizes,
    .jacobian             = ElasFSCurrentNH2dF,
    .jacobian_loc         = ElasFSCurrentNH2dF_loc,
    .energy               = ElasFSCurrentNH2Energy,
    .energy_loc           = ElasFSCurrentNH2Energy_loc,
    .diagnostic           = ElasFSCurrentNH2Diagnostic,
    .diagnostic_loc       = ElasFSCurrentNH2Diagnostic_loc,
};

PetscErrorCode SetupLibceedFineLevel_ElasFSCurrentNH2(DM dm, DM dm_energy, DM dm_diagnostic, Ceed ceed, AppCtx app_ctx, CeedQFunctionContext phys_ctx,
                                                      PetscInt fine_level, PetscInt num_comp_u, PetscInt U_g_size, PetscInt U_loc_size,
                                                      CeedVector force_ceed, CeedVector neumann_ceed, CeedData *data) {
  PetscFunctionBegin;

  PetscCall(SetupLibceedFineLevel(dm, dm_energy, dm_diagnostic, ceed, app_ctx, phys_ctx, finite_strain_neo_Hookean_current_2, fine_level, num_comp_u,
                                  U_g_size, U_loc_size, force_ceed, neumann_ceed, data));

  PetscFunctionReturn(0);
};

PetscErrorCode SetupLibceedLevel_ElasFSCurrentNH2(DM dm, Ceed ceed, AppCtx app_ctx, PetscInt level, PetscInt num_comp_u, PetscInt U_g_size,
                                                  PetscInt U_loc_size, CeedVector fine_mult, CeedData *data) {
  PetscFunctionBegin;

  PetscCall(SetupLibceedLevel(dm, ceed, app_ctx, finite_strain_neo_Hookean_current_2, level, num_comp_u, U_g_size, U_loc_size, fine_mult, data));

  PetscFunctionReturn(0);
};
