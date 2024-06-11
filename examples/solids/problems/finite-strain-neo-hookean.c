// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../qfunctions/finite-strain-neo-hookean.h"

#include <ceed.h>
#include <petscsys.h>

#include "../include/setup-libceed.h"
#include "../include/structs.h"
#include "../problems/neo-hookean.h"
#include "../problems/problems.h"
#include "../qfunctions/common.h"

static const char *const field_names[] = {"gradu"};
static CeedInt           field_sizes[] = {9};

ProblemData finite_strain_neo_Hookean = {
    .setup_geo            = SetupGeo,
    .setup_geo_loc        = SetupGeo_loc,
    .q_data_size          = 10,
    .quadrature_mode      = CEED_GAUSS,
    .residual             = ElasFSResidual_NH,
    .residual_loc         = ElasFSResidual_NH_loc,
    .number_fields_stored = 1,
    .field_names          = field_names,
    .field_sizes          = field_sizes,
    .jacobian             = ElasFSJacobian_NH,
    .jacobian_loc         = ElasFSJacobian_NH_loc,
    .energy               = ElasFSEnergy_NH,
    .energy_loc           = ElasFSEnergy_NH_loc,
    .diagnostic           = ElasFSDiagnostic_NH,
    .diagnostic_loc       = ElasFSDiagnostic_NH_loc,
};

PetscErrorCode SetupLibceedFineLevel_ElasFSNH(DM dm, DM dm_energy, DM dm_diagnostic, Ceed ceed, AppCtx app_ctx, CeedQFunctionContext phys_ctx,
                                              PetscInt fine_level, PetscInt num_comp_u, PetscInt U_g_size, PetscInt U_loc_size, CeedVector force_ceed,
                                              CeedVector neumann_ceed, CeedData *data) {
  PetscFunctionBegin;

  PetscCall(SetupLibceedFineLevel(dm, dm_energy, dm_diagnostic, ceed, app_ctx, phys_ctx, finite_strain_neo_Hookean, fine_level, num_comp_u, U_g_size,
                                  U_loc_size, force_ceed, neumann_ceed, data));

  PetscFunctionReturn(PETSC_SUCCESS);
};

PetscErrorCode SetupLibceedLevel_ElasFSNH(DM dm, Ceed ceed, AppCtx app_ctx, PetscInt level, PetscInt num_comp_u, PetscInt U_g_size,
                                          PetscInt U_loc_size, CeedVector fine_mult, CeedData *data) {
  PetscFunctionBegin;

  PetscCall(SetupLibceedLevel(dm, ceed, app_ctx, finite_strain_neo_Hookean, level, num_comp_u, U_g_size, U_loc_size, fine_mult, data));

  PetscFunctionReturn(PETSC_SUCCESS);
};
