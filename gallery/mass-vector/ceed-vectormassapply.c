// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-source/gallery/ceed-vectormassapply.h>
#include <string.h>

/**
  @brief Set fields for Ceed QFunction for applying the mass matrix
           on a vector system with three components
**/
static int CeedQFunctionInit_Vector3MassApply(Ceed ceed, const char *requested, CeedQFunction qf) {
  // Check QFunction name
  const char *name = "Vector3MassApply";
  if (strcmp(name, requested)) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "QFunction '%s' does not match requested name: %s", name, requested);
    // LCOV_EXCL_STOP
  }

  // Add QFunction fields
  const CeedInt num_comp = 3;
  CeedCall(CeedQFunctionAddInput(qf, "u", num_comp, CEED_EVAL_INTERP));
  CeedCall(CeedQFunctionAddInput(qf, "qdata", 1, CEED_EVAL_NONE));
  CeedCall(CeedQFunctionAddOutput(qf, "v", num_comp, CEED_EVAL_INTERP));

  CeedCall(CeedQFunctionSetUserFlopsEstimate(qf, num_comp));

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Register Ceed QFunction for applying the mass matrix on a vector system
           with three components
**/
CEED_INTERN int CeedQFunctionRegister_Vector3MassApply(void) {
  return CeedQFunctionRegister("Vector3MassApply", Vector3MassApply_loc, 1, Vector3MassApply, CeedQFunctionInit_Vector3MassApply);
}
