// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-source/gallery/ceed-massapply.h>
#include <string.h>

/**
  @brief Set fields for `CeedQFunction` for applying the mass matrix
**/
static int CeedQFunctionInit_MassApply(Ceed ceed, const char *requested, CeedQFunction qf) {
  // Check QFunction name
  const char *name = "MassApply";
  CeedCheck(!strcmp(name, requested), ceed, CEED_ERROR_UNSUPPORTED, "QFunction '%s' does not match requested name: %s", name, requested);

  // Add QFunction fields
  CeedCall(CeedQFunctionAddInput(qf, "u", 1, CEED_EVAL_INTERP));
  CeedCall(CeedQFunctionAddInput(qf, "qdata", 1, CEED_EVAL_NONE));
  CeedCall(CeedQFunctionAddOutput(qf, "v", 1, CEED_EVAL_INTERP));

  CeedCall(CeedQFunctionSetUserFlopsEstimate(qf, 1));

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Register `CeedQFunction` for applying the mass matrix
**/
CEED_INTERN int CeedQFunctionRegister_MassApply(void) {
  return CeedQFunctionRegister("MassApply", MassApply_loc, 1, MassApply, CeedQFunctionInit_MassApply);
}
