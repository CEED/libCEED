// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <string.h>
#include "ceed-mass3dbuild.h"

/**
  @brief Set fields for Ceed QFunction building the geometric data for the 3D
           mass matrix
**/
static int CeedQFunctionInit_Mass3DBuild(Ceed ceed, const char *requested,
    CeedQFunction qf) {
  int ierr;

  // Check QFunction name
  const char *name = "Mass3DBuild";
  if (strcmp(name, requested))
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                     "QFunction '%s' does not match requested name: %s",
                     name, requested);
  // LCOV_EXCL_STOP

  // Add QFunction fields
  const CeedInt dim = 3;
  ierr = CeedQFunctionAddInput(qf, "dx", dim*dim, CEED_EVAL_GRAD);
  CeedChk(ierr);
  ierr = CeedQFunctionAddInput(qf, "weights", 1, CEED_EVAL_WEIGHT);
  CeedChk(ierr);
  ierr = CeedQFunctionAddOutput(qf, "qdata", 1, CEED_EVAL_NONE); CeedChk(ierr);

  ierr = CeedQFunctionSetUserFlopsEstimate(qf, 15); CeedChk(ierr);

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Register Ceed QFunction for building the geometric data for the 3D mass
           matrix
**/
CEED_INTERN int CeedQFunctionRegister_Mass3DBuild(void) {
  return CeedQFunctionRegister("Mass3DBuild", Mass3DBuild_loc, 1, Mass3DBuild,
                               CeedQFunctionInit_Mass3DBuild);
}
