// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-source/gallery/ceed-poisson2dapply.h>
#include <string.h>

/**
  @brief Set fields for Ceed QFunction applying the 2D Poisson operator
**/
static int CeedQFunctionInit_Poisson2DApply(Ceed ceed, const char *requested, CeedQFunction qf) {
  // Check QFunction name
  const char *name = "Poisson2DApply";
  if (strcmp(name, requested)) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "QFunction '%s' does not match requested name: %s", name, requested);
    // LCOV_EXCL_STOP
  }

  // Add QFunction fields
  const CeedInt dim = 2;
  CeedCall(CeedQFunctionAddInput(qf, "du", dim, CEED_EVAL_GRAD));
  CeedCall(CeedQFunctionAddInput(qf, "qdata", dim * (dim + 1) / 2, CEED_EVAL_NONE));
  CeedCall(CeedQFunctionAddOutput(qf, "dv", dim, CEED_EVAL_GRAD));

  CeedCall(CeedQFunctionSetUserFlopsEstimate(qf, 6));

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Register Ceed QFunction for applying the 2D Poisson operator
**/
CEED_INTERN int CeedQFunctionRegister_Poisson2DApply(void) {
  return CeedQFunctionRegister("Poisson2DApply", Poisson2DApply_loc, 1, Poisson2DApply, CeedQFunctionInit_Poisson2DApply);
}
