// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../problems/problems.h"

#include <ceed.h>
#include <petscsys.h>

PetscErrorCode RegisterProblems(ProblemFunctions problem_functions) {
  PetscFunctionBegin;

  SOLIDS_PROBLEM_REGISTER(problem_functions, "Linear", ElasLinear, NH);
  SOLIDS_PROBLEM_REGISTER(problem_functions, "FS-NH", ElasFSNH, NH);
  SOLIDS_PROBLEM_REGISTER(problem_functions, "FS-MR", ElasFSMR, MR);

  PetscFunctionReturn(PETSC_SUCCESS);
};
