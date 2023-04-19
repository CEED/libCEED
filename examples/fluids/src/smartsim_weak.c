// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
//
// @file This creates weak functions for smartsim dependent functions. If the smartsim-dependent functions are actually built, these functions are not
// linked to the final executable.

#include "../navierstokes.h"

PetscErrorCode SGS_DD_TrainingSetup(Ceed ceed, User user, CeedData ceed_data, ProblemData *problem) __attribute__((weak));
PetscErrorCode SGS_DD_TrainingSetup(Ceed ceed, User user, CeedData ceed_data, ProblemData *problem) {
  PetscFunctionBeginUser;
  SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Must build with SMARTREDIS_DIR set to run %s", __func__);
};

PetscErrorCode TSMonitor_SGS_DD_Training(TS ts, PetscInt step_num, PetscReal solution_time, Vec Q, void *ctx) __attribute__((weak));
PetscErrorCode TSMonitor_SGS_DD_Training(TS ts, PetscInt step_num, PetscReal solution_time, Vec Q, void *ctx) {
  PetscFunctionBeginUser;
  SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Must build with SMARTREDIS_DIR set to run %s", __func__);
};
