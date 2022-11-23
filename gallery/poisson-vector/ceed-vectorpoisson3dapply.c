// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-source/gallery/ceed-vectorpoisson3dapply.h>
#include <string.h>

/**
  @brief Set fields for Ceed QFunction applying the 3D Poisson operator
           on a vector system with three components
**/
static int CeedQFunctionInit_Vector3Poisson3DApply(Ceed ceed, const char *requested, CeedQFunction qf) {
  // Check QFunction name
  const char *name = "Vector3Poisson3DApply";
  if (strcmp(name, requested)) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "QFunction '%s' does not match requested name: %s", name, requested);
    // LCOV_EXCL_STOP
  }

  // Add QFunction fields
  const CeedInt dim = 3, num_comp = 3;
  CeedCall(CeedQFunctionAddInput(qf, "du", num_comp * dim, CEED_EVAL_GRAD));
  CeedCall(CeedQFunctionAddInput(qf, "qdata", dim * (dim + 1) / 2, CEED_EVAL_NONE));
  CeedCall(CeedQFunctionAddOutput(qf, "dv", num_comp * dim, CEED_EVAL_GRAD));

  CeedCall(CeedQFunctionSetUserFlopsEstimate(qf, num_comp * 15));

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Register Ceed QFunction for applying the 3D Poisson operator
           on a vector system with three components
**/
CEED_INTERN int CeedQFunctionRegister_Vector3Poisson3DApply(void) {
  return CeedQFunctionRegister("Vector3Poisson3DApply", Vector3Poisson3DApply_loc, 1, Vector3Poisson3DApply, CeedQFunctionInit_Vector3Poisson3DApply);
}
