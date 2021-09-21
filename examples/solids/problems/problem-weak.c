// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <stdbool.h>
#include <stdlib.h>
#include <ceed/backend.h>
#include "problems.h"

// This function provides a debug target for weak symbols
// LCOV_EXCL_START
static PetscErrorCode RegisterProblem_Weak(const char *name) {
  if (getenv("CEED_DEBUG")) fprintf(stderr, "Weak %s\n", name);
  PetscFunctionReturn(0);
}
// LCOV_EXCL_STOP

#define MACRO(name)                                                                                  \
  PetscErrorCode name(ProblemFunctions problem_functions) __attribute__((weak));                     \
  PetscErrorCode name(ProblemFunctions problem_functions) { return RegisterProblem_Weak(__func__); }
#include "problem-list.h"
#undef MACRO
