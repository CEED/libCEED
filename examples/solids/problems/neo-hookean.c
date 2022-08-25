// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../problems/neo-hookean.h"

#include <ceed.h>
#include <petsc.h>

// Build libCEED context object
PetscErrorCode PhysicsContext_NH(MPI_Comm comm, Ceed ceed, Units *units, CeedQFunctionContext *ctx) {
  Physics_NH phys;

  PetscFunctionBegin;

  PetscCall(PetscMalloc1(1, units));
  PetscCall(PetscMalloc1(1, &phys));
  PetscCall(ProcessPhysics_NH(comm, phys, *units));
  CeedQFunctionContextCreate(ceed, ctx);
  CeedQFunctionContextSetData(*ctx, CEED_MEM_HOST, CEED_COPY_VALUES, sizeof(*phys), phys);
  PetscCall(PetscFree(phys));

  PetscFunctionReturn(0);
}

// Build libCEED smoother context object
PetscErrorCode PhysicsSmootherContext_NH(MPI_Comm comm, Ceed ceed, CeedQFunctionContext ctx, CeedQFunctionContext *ctx_smoother) {
  PetscScalar nu_smoother = 0;
  PetscBool   nu_flag     = PETSC_FALSE;
  Physics_NH  phys, phys_smoother;

  PetscFunctionBegin;

  PetscOptionsBegin(comm, NULL, "Neo-Hookean physical parameters for smoother", NULL);

  PetscCall(PetscOptionsScalar("-nu_smoother", "Poisson's ratio for smoother", NULL, nu_smoother, &nu_smoother, &nu_flag));

  PetscOptionsEnd();  // End of setting Physics

  if (nu_flag) {
    // Copy context
    CeedQFunctionContextGetData(ctx, CEED_MEM_HOST, &phys);
    PetscCall(PetscMalloc1(1, &phys_smoother));
    PetscCall(PetscMemcpy(phys_smoother, phys, sizeof(*phys)));
    CeedQFunctionContextRestoreData(ctx, &phys);
    // Create smoother context
    CeedQFunctionContextCreate(ceed, ctx_smoother);
    phys_smoother->nu = nu_smoother;
    CeedQFunctionContextSetData(*ctx_smoother, CEED_MEM_HOST, CEED_COPY_VALUES, sizeof(*phys_smoother), phys_smoother);
    PetscCall(PetscFree(phys_smoother));
  } else {
    *ctx_smoother = NULL;
  }

  PetscFunctionReturn(0);
}

// Process physics options - Neo-Hookean
PetscErrorCode ProcessPhysics_NH(MPI_Comm comm, Physics_NH phys, Units units) {
  PetscBool nu_flag    = PETSC_FALSE;
  PetscBool Young_flag = PETSC_FALSE;
  phys->nu             = 0;
  phys->E              = 0;
  units->meter         = 1;  // 1 meter in scaled length units
  units->second        = 1;  // 1 second in scaled time units
  units->kilogram      = 1;  // 1 kilogram in scaled mass units

  PetscFunctionBeginUser;

  PetscOptionsBegin(comm, NULL, "Neo-Hookean physical parameters", NULL);

  PetscCall(PetscOptionsScalar("-nu", "Poisson's ratio", NULL, phys->nu, &phys->nu, &nu_flag));

  PetscCall(PetscOptionsScalar("-E", "Young's Modulus", NULL, phys->E, &phys->E, &Young_flag));

  PetscCall(PetscOptionsScalar("-units_meter", "1 meter in scaled length units", NULL, units->meter, &units->meter, NULL));
  units->meter = fabs(units->meter);

  PetscCall(PetscOptionsScalar("-units_second", "1 second in scaled time units", NULL, units->second, &units->second, NULL));
  units->second = fabs(units->second);

  PetscCall(PetscOptionsScalar("-units_kilogram", "1 kilogram in scaled mass units", NULL, units->kilogram, &units->kilogram, NULL));
  units->kilogram = fabs(units->kilogram);

  PetscOptionsEnd();  // End of setting Physics

  // Check for all required options to be set
  if (!nu_flag) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "-nu option needed");
  }
  if (!Young_flag) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "-E option needed");
  }

  // Define derived units
  units->Pascal = units->kilogram / (units->meter * PetscSqr(units->second));

  // Scale E to Pa
  phys->E *= units->Pascal;

  PetscFunctionReturn(0);
};