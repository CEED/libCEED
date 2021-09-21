// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <petsc.h>
#include "../problems/neo-hookean.h"

// Build libCEED context object
PetscErrorCode PhysicsContext_NH(MPI_Comm comm, Ceed ceed, Units *units,
                                 CeedQFunctionContext *ctx) {
  PetscErrorCode ierr;
  Physics_NH phys;

  PetscFunctionBegin;

  ierr = PetscMalloc1(1, units); CHKERRQ(ierr);
  ierr = PetscMalloc1(1, &phys); CHKERRQ(ierr);
  ierr = ProcessPhysics_NH(comm, phys, *units); CHKERRQ(ierr);
  CeedQFunctionContextCreate(ceed, ctx);
  CeedQFunctionContextSetData(*ctx, CEED_MEM_HOST, CEED_COPY_VALUES,
                              sizeof(*phys), phys);
  ierr = PetscFree(phys); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Build libCEED smoother context object
PetscErrorCode PhysicsSmootherContext_NH(MPI_Comm comm, Ceed ceed,
    CeedQFunctionContext ctx, CeedQFunctionContext *ctx_smoother) {
  PetscErrorCode ierr;
  PetscScalar nu_smoother = 0;
  PetscBool nu_flag = PETSC_FALSE;
  Physics_NH phys, phys_smoother;

  PetscFunctionBegin;

  PetscOptionsBegin(comm, NULL, "Neo-Hookean physical parameters for smoother",
                    NULL);

  ierr = PetscOptionsScalar("-nu_smoother", "Poisson's ratio for smoother",
                            NULL, nu_smoother, &nu_smoother, &nu_flag);
  CHKERRQ(ierr);

  PetscOptionsEnd(); // End of setting Physics

  if (nu_flag) {
    // Copy context
    CeedQFunctionContextGetData(ctx, CEED_MEM_HOST, &phys);
    ierr = PetscMalloc1(1, &phys_smoother); CHKERRQ(ierr);
    ierr = PetscMemcpy(phys_smoother, phys, sizeof(*phys)); CHKERRQ(ierr);
    CeedQFunctionContextRestoreData(ctx, &phys);
    // Create smoother context
    CeedQFunctionContextCreate(ceed, ctx_smoother);
    phys_smoother->nu = nu_smoother;
    CeedQFunctionContextSetData(*ctx_smoother, CEED_MEM_HOST, CEED_COPY_VALUES,
                                sizeof(*phys_smoother), phys_smoother);
    ierr = PetscFree(phys_smoother); CHKERRQ(ierr);
  } else {
    *ctx_smoother = NULL;
  }

  PetscFunctionReturn(0);
}

// Process physics options - Neo-Hookean
PetscErrorCode ProcessPhysics_NH(MPI_Comm comm, Physics_NH phys, Units units) {
  PetscErrorCode ierr;
  PetscBool nu_flag = PETSC_FALSE;
  PetscBool Young_flag = PETSC_FALSE;
  phys->nu = 0;
  phys->E = 0;
  units->meter     = 1;        // 1 meter in scaled length units
  units->second    = 1;        // 1 second in scaled time units
  units->kilogram  = 1;        // 1 kilogram in scaled mass units

  PetscFunctionBeginUser;

  PetscOptionsBegin(comm, NULL, "Neo-Hookean physical parameters", NULL);

  ierr = PetscOptionsScalar("-nu", "Poisson's ratio", NULL, phys->nu, &phys->nu,
                            &nu_flag); CHKERRQ(ierr);

  ierr = PetscOptionsScalar("-E", "Young's Modulus", NULL, phys->E, &phys->E,
                            &Young_flag); CHKERRQ(ierr);

  ierr = PetscOptionsScalar("-units_meter", "1 meter in scaled length units",
                            NULL, units->meter, &units->meter, NULL);
  CHKERRQ(ierr);
  units->meter = fabs(units->meter);

  ierr = PetscOptionsScalar("-units_second", "1 second in scaled time units",
                            NULL, units->second, &units->second, NULL);
  CHKERRQ(ierr);
  units->second = fabs(units->second);

  ierr = PetscOptionsScalar("-units_kilogram", "1 kilogram in scaled mass units",
                            NULL, units->kilogram, &units->kilogram, NULL);
  CHKERRQ(ierr);
  units->kilogram = fabs(units->kilogram);

  PetscOptionsEnd(); // End of setting Physics

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
