// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../qfunctions/finite-strain-neo-hookean-initial-2.h"

#include <ceed.h>

#include "../include/setup-libceed.h"
#include "../include/structs.h"
#include "../problems/neo-hookean.h"
#include "../problems/problems.h"
#include "../qfunctions/common.h"

static const char *const field_names[] = {"gradu", "C_inv", "lambda_log_J"};
static CeedInt           field_sizes[] = {9, 6, 1};

ProblemData finite_strain_neo_Hookean_initial_2 = {
    .setup_geo            = SetupGeo,
    .setup_geo_loc        = SetupGeo_loc,
    .q_data_size          = 10,
    .quadrature_mode      = CEED_GAUSS,
    .residual             = ElasFSInitialNH2F,
    .residual_loc         = ElasFSInitialNH2F_loc,
    .number_fields_stored = 3,
    .field_names          = field_names,
    .field_sizes          = field_sizes,
    .jacobian             = ElasFSInitialNH2dF,
    .jacobian_loc         = ElasFSInitialNH2dF_loc,
    .energy               = ElasFSInitialNH2Energy,
    .energy_loc           = ElasFSInitialNH2Energy_loc,
    .diagnostic           = ElasFSInitialNH2Diagnostic,
    .diagnostic_loc       = ElasFSInitialNH2Diagnostic_loc,
};

PetscErrorCode SetupLibceedFineLevel_ElasFSInitialNH2(DM dm, DM dm_energy, DM dm_diagnostic, Ceed ceed, AppCtx app_ctx, CeedQFunctionContext phys_ctx,
                                                      PetscInt fine_level, PetscInt num_comp_u, PetscInt U_g_size, PetscInt U_loc_size,
                                                      CeedVector force_ceed, CeedVector neumann_ceed, CeedData *data) {
  PetscFunctionBegin;

  PetscCall(SetupLibceedFineLevel(dm, dm_energy, dm_diagnostic, ceed, app_ctx, phys_ctx, finite_strain_neo_Hookean_initial_2, fine_level, num_comp_u,
                                  U_g_size, U_loc_size, force_ceed, neumann_ceed, data));

  PetscFunctionReturn(0);
};

PetscErrorCode SetupLibceedLevel_ElasFSInitialNH2(DM dm, Ceed ceed, AppCtx app_ctx, PetscInt level, PetscInt num_comp_u, PetscInt U_g_size,
                                                  PetscInt U_loc_size, CeedVector fine_mult, CeedData *data) {
  PetscFunctionBegin;

  PetscCall(SetupLibceedLevel(dm, ceed, app_ctx, finite_strain_neo_Hookean_initial_2, level, num_comp_u, U_g_size, U_loc_size, fine_mult, data));

  PetscFunctionReturn(0);
};
