// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-source/gallery/ceed-poisson3dbuild.h>
#include <string.h>

/**
  @brief Set fields for `CeedQFunction` building the geometric data for the 3D Poisson operator
**/
static int CeedQFunctionInit_Poisson3DBuild(Ceed ceed, const char *requested, CeedQFunction qf) {
  // Check QFunction name
  const char *name = "Poisson3DBuild";
  CeedCheck(!strcmp(name, requested), ceed, CEED_ERROR_UNSUPPORTED, "QFunction '%s' does not match requested name: %s", name, requested);

  // Add QFunction fields
  const CeedInt dim = 3;
  CeedCall(CeedQFunctionAddInput(qf, "dx", dim * dim, CEED_EVAL_GRAD));
  CeedCall(CeedQFunctionAddInput(qf, "weights", 1, CEED_EVAL_WEIGHT));
  CeedCall(CeedQFunctionAddOutput(qf, "qdata", dim * (dim + 1) / 2, CEED_EVAL_NONE));

  CeedCall(CeedQFunctionSetUserFlopsEstimate(qf, 69));

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Register `CeedQFunction` for building the geometric data for the 3D Poisson operator
**/
CEED_INTERN int CeedQFunctionRegister_Poisson3DBuild(void) {
  return CeedQFunctionRegister("Poisson3DBuild", Poisson3DBuild_loc, 1, Poisson3DBuild, CeedQFunctionInit_Poisson3DBuild);
}
