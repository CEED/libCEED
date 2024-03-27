// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed-impl.h>
#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/cuda.h>
#include <cuda.h>

/**
  @brief Set CUDA function pointer to evaluate action at quadrature points

  @param[in,out] qf `CeedQFunction` to set device pointer
  @param[in]     f  Device function pointer to evaluate action at quadrature points

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionSetCUDAUserFunction(CeedQFunction qf, CUfunction f) {
  if (!qf->SetCUDAUserFunction) {
    Ceed ceed;

    CeedCall(CeedQFunctionGetCeed(qf, &ceed));
    CeedDebug(ceed, "Backend does not support CUfunction pointers for QFunctions.");
  } else {
    CeedCall(qf->SetCUDAUserFunction(qf, f));
  }
  return CEED_ERROR_SUCCESS;
}
