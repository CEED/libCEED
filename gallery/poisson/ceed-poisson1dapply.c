// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-source/gallery/ceed-poisson1dapply.h>
#include <string.h>

/**
  @brief Set fields for Ceed QFunction applying the 1D Poisson operator
**/
static int CeedQFunctionInit_Poisson1DApply(Ceed ceed, const char *requested, CeedQFunction qf) {
  // Check QFunction name
  const char *name = "Poisson1DApply";
  CeedCheck(!strcmp(name, requested), ceed, CEED_ERROR_UNSUPPORTED, "QFunction '%s' does not match requested name: %s", name, requested);

  // Add QFunction fields
  const CeedInt dim = 1;
  CeedCall(CeedQFunctionAddInput(qf, "du", dim, CEED_EVAL_GRAD));
  CeedCall(CeedQFunctionAddInput(qf, "qdata", dim * (dim + 1) / 2, CEED_EVAL_NONE));
  CeedCall(CeedQFunctionAddOutput(qf, "dv", dim, CEED_EVAL_GRAD));

  CeedCall(CeedQFunctionSetUserFlopsEstimate(qf, 1));

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Register Ceed QFunction for applying the 1D Poisson operator
**/
CEED_INTERN int CeedQFunctionRegister_Poisson1DApply(void) {
  return CeedQFunctionRegister("Poisson1DApply", Poisson1DApply_loc, 1, Poisson1DApply, CeedQFunctionInit_Poisson1DApply);
}
