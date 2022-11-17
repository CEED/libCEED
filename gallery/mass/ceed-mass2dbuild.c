// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-source/gallery/ceed-mass2dbuild.h>
#include <string.h>

/**
  @brief Set fields for Ceed QFunction building the geometric data for the 2D
           mass matrix
**/
static int CeedQFunctionInit_Mass2DBuild(Ceed ceed, const char *requested, CeedQFunction qf) {
  // Check QFunction name
  const char *name = "Mass2DBuild";
  if (strcmp(name, requested)) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "QFunction '%s' does not match requested name: %s", name, requested);
    // LCOV_EXCL_STOP
  }

  // Add QFunction fields
  const CeedInt dim = 2;
  CeedCall(CeedQFunctionAddInput(qf, "dx", dim * dim, CEED_EVAL_GRAD));
  CeedCall(CeedQFunctionAddInput(qf, "weights", 1, CEED_EVAL_WEIGHT));
  CeedCall(CeedQFunctionAddOutput(qf, "qdata", 1, CEED_EVAL_NONE));

  CeedCall(CeedQFunctionSetUserFlopsEstimate(qf, 4));

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Register Ceed QFunction for building the geometric data for the 2D mass
           matrix
**/
CEED_INTERN int CeedQFunctionRegister_Mass2DBuild(void) {
  return CeedQFunctionRegister("Mass2DBuild", Mass2DBuild_loc, 1, Mass2DBuild, CeedQFunctionInit_Mass2DBuild);
}
