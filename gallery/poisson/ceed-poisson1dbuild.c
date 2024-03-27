// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-source/gallery/ceed-poisson1dbuild.h>
#include <string.h>

/**
  @brief Set fields for `CeedQFunction` building the geometric data for the 1D Poisson operator
**/
static int CeedQFunctionInit_Poisson1DBuild(Ceed ceed, const char *requested, CeedQFunction qf) {
  // Check QFunction name
  const char *name = "Poisson1DBuild";
  CeedCheck(!strcmp(name, requested), ceed, CEED_ERROR_UNSUPPORTED, "QFunction '%s' does not match requested name: %s", name, requested);

  // Add QFunction fields
  const CeedInt dim = 1;
  CeedCall(CeedQFunctionAddInput(qf, "dx", dim * dim, CEED_EVAL_GRAD));
  CeedCall(CeedQFunctionAddInput(qf, "weights", 1, CEED_EVAL_WEIGHT));
  CeedCall(CeedQFunctionAddOutput(qf, "qdata", dim * (dim + 1) / 2, CEED_EVAL_NONE));

  CeedCall(CeedQFunctionSetUserFlopsEstimate(qf, 1));

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Register `CeedQFunction` for building the geometric data for the 1D Poisson operator
**/
CEED_INTERN int CeedQFunctionRegister_Poisson1DBuild(void) {
  return CeedQFunctionRegister("Poisson1DBuild", Poisson1DBuild_loc, 1, Poisson1DBuild, CeedQFunctionInit_Poisson1DBuild);
}
