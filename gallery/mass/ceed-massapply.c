// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-source/gallery/ceed-massapply.h>
#include <string.h>

/**
  @brief Set fields for Ceed QFunction for applying the mass matrix
**/
static int CeedQFunctionInit_MassApply(Ceed ceed, const char *requested, CeedQFunction qf) {
  // Check QFunction name
  const char *name = "MassApply";
  if (strcmp(name, requested)) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "QFunction '%s' does not match requested name: %s", name, requested);
    // LCOV_EXCL_STOP
  }

  // Add QFunction fields
  CeedCall(CeedQFunctionAddInput(qf, "u", 1, CEED_EVAL_INTERP));
  CeedCall(CeedQFunctionAddInput(qf, "qdata", 1, CEED_EVAL_NONE));
  CeedCall(CeedQFunctionAddOutput(qf, "v", 1, CEED_EVAL_INTERP));

  CeedCall(CeedQFunctionSetUserFlopsEstimate(qf, 1));

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Register Ceed QFunction for applying the mass matrix
**/
CEED_INTERN int CeedQFunctionRegister_MassApply(void) {
  return CeedQFunctionRegister("MassApply", MassApply_loc, 1, MassApply, CeedQFunctionInit_MassApply);
}
