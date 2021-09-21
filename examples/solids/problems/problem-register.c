// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <petsc.h>
#include "../problems/problems.h"

static bool register_all_called;

#define MACRO(name) PetscErrorCode name(ProblemFunctions problem_functions);
#include "problem-list.h"
#undef MACRO

PetscErrorCode RegisterProblems(ProblemFunctions problem_functions) {
  PetscFunctionBeginUser;
  if (register_all_called) return 0;
  register_all_called = true;

  PetscFunctionBegin;

#define MACRO(name) CHKERRQ(name(problem_functions));
#include "problem-list.h"
#undef MACRO
  PetscFunctionReturn(0);
};
