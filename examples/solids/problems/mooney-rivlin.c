// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../problems/mooney-rivlin.h"

#include <ceed.h>
#include <petsc.h>

// Build libCEED context object
PetscErrorCode PhysicsContext_MR(MPI_Comm comm, Ceed ceed, Units *units, CeedQFunctionContext *ctx) {
  Physics_MR phys;

  PetscFunctionBegin;

  PetscCall(PetscMalloc1(1, units));
  PetscCall(PetscMalloc1(1, &phys));
  PetscCall(ProcessPhysics_MR(comm, phys, *units));
  CeedQFunctionContextCreate(ceed, ctx);
  CeedQFunctionContextSetData(*ctx, CEED_MEM_HOST, CEED_COPY_VALUES, sizeof(*phys), phys);
  PetscCall(PetscFree(phys));

  PetscFunctionReturn(0);
}

// Build libCEED smoother context object
PetscErrorCode PhysicsSmootherContext_MR(MPI_Comm comm, Ceed ceed, CeedQFunctionContext ctx, CeedQFunctionContext *ctx_smoother) {
  PetscScalar nu_smoother = 0;
  PetscBool   nu_flag     = PETSC_FALSE;
  Physics_MR  phys, phys_smoother;

  PetscFunctionBegin;

  PetscOptionsBegin(comm, NULL, "Mooney Rivlin physical parameters for smoother", NULL);

  PetscCall(PetscOptionsScalar("-nu_smoother", "Poisson's ratio for smoother", NULL, nu_smoother, &nu_smoother, &nu_flag));
  if (nu_smoother < 0 || nu_smoother >= 0.5)
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Mooney-Rivlin model requires Poisson ratio -nu option in [0, .5)");

  PetscOptionsEnd();  // End of setting Physics

  if (nu_flag) {
    // Copy context
    CeedQFunctionContextGetData(ctx, CEED_MEM_HOST, &phys);
    PetscCall(PetscMalloc1(1, &phys_smoother));
    PetscCall(PetscMemcpy(phys_smoother, phys, sizeof(*phys)));
    CeedQFunctionContextRestoreData(ctx, &phys);
    // Create smoother context
    CeedQFunctionContextCreate(ceed, ctx_smoother);
    phys_smoother->lambda = 2 * (phys_smoother->mu_1 + phys_smoother->mu_2) * nu_smoother / (1 - 2 * nu_smoother);
    CeedQFunctionContextSetData(*ctx_smoother, CEED_MEM_HOST, CEED_COPY_VALUES, sizeof(*phys_smoother), phys_smoother);
    PetscCall(PetscFree(phys_smoother));
  } else {
    *ctx_smoother = NULL;
  }

  PetscFunctionReturn(0);
}

// Process physics options - Mooney-Rivlin
PetscErrorCode ProcessPhysics_MR(MPI_Comm comm, Physics_MR phys, Units units) {
  PetscReal nu    = -1;
  phys->mu_1      = -1;
  phys->mu_2      = -1;
  phys->lambda    = -1;
  units->meter    = 1;  // 1 meter in scaled length units
  units->second   = 1;  // 1 second in scaled time units
  units->kilogram = 1;  // 1 kilogram in scaled mass units

  PetscFunctionBeginUser;

  PetscOptionsBegin(comm, NULL, "Mooney Rivlin physical parameters", NULL);

  PetscCall(PetscOptionsScalar("-mu_1", "Material Property mu_1", NULL, phys->mu_1, &phys->mu_1, NULL));
  if (phys->mu_1 < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Mooney-Rivlin model requires non-negative -mu_1 option (Pa)");

  PetscCall(PetscOptionsScalar("-mu_2", "Material Property mu_2", NULL, phys->mu_2, &phys->mu_2, NULL));
  if (phys->mu_2 < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Mooney-Rivlin model requires non-negative -mu_2 option (Pa)");

  PetscCall(PetscOptionsScalar("-nu", "Poisson ratio", NULL, nu, &nu, NULL));
  if (nu < 0 || nu >= 0.5) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Mooney-Rivlin model requires Poisson ratio -nu option in [0, .5)");
  phys->lambda = 2 * (phys->mu_1 + phys->mu_2) * nu / (1 - 2 * nu);

  PetscCall(PetscOptionsScalar("-units_meter", "1 meter in scaled length units", NULL, units->meter, &units->meter, NULL));
  units->meter = fabs(units->meter);

  PetscCall(PetscOptionsScalar("-units_second", "1 second in scaled time units", NULL, units->second, &units->second, NULL));
  units->second = fabs(units->second);

  PetscCall(PetscOptionsScalar("-units_kilogram", "1 kilogram in scaled mass units", NULL, units->kilogram, &units->kilogram, NULL));
  units->kilogram = fabs(units->kilogram);

  PetscOptionsEnd();  // End of setting Physics

  // Define derived units
  units->Pascal = units->kilogram / (units->meter * PetscSqr(units->second));

  // Scale material parameters based on units of Pa
  phys->mu_1 *= units->Pascal;
  phys->mu_2 *= units->Pascal;
  phys->lambda *= units->Pascal;

  PetscFunctionReturn(0);
};