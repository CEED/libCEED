// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-source/gallery/ceed-poisson2dbuild.h>
#include <string.h>

/**
  @brief Set fields for Ceed QFunction building the geometric data for the 2D
           Poisson operator
**/
static int CeedQFunctionInit_Poisson2DBuild(Ceed ceed, const char *requested, CeedQFunction qf) {
  // Check QFunction name
  const char *name = "Poisson2DBuild";
  if (strcmp(name, requested)) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "QFunction '%s' does not match requested name: %s", name, requested);
    // LCOV_EXCL_STOP
  }

  // Add QFunction fields
  const CeedInt dim = 2;
  CeedCall(CeedQFunctionAddInput(qf, "dx", dim * dim, CEED_EVAL_GRAD));
  CeedCall(CeedQFunctionAddInput(qf, "weights", 1, CEED_EVAL_WEIGHT));
  CeedCall(CeedQFunctionAddOutput(qf, "qdata", dim * (dim + 1) / 2, CEED_EVAL_NONE));

  CeedCall(CeedQFunctionSetUserFlopsEstimate(qf, 17));

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Register Ceed QFunction for building the geometric data for the 2D
           Poisson operator
**/
CEED_INTERN int CeedQFunctionRegister_Poisson2DBuild(void) {
  return CeedQFunctionRegister("Poisson2DBuild", Poisson2DBuild_loc, 1, Poisson2DBuild, CeedQFunctionInit_Poisson2DBuild);
}
