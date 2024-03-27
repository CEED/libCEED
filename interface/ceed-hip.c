// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed-impl.h>
#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/hip.h>
#include <hip/hip_runtime_api.h>

/**
  @brief Set HIP function pointer to evaluate action at quadrature points

  @param[in,out] qf `CeedQFunction` to set device pointer
  @param[in]     f  Device function pointer to evaluate action at quadrature points

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionSetHIPUserFunction(CeedQFunction qf, hipFunction_t f) {
  if (!qf->SetHIPUserFunction) {
    Ceed ceed;

    CeedCall(CeedQFunctionGetCeed(qf, &ceed));
    CeedDebug(ceed, "Backend does not support hipFunction_t pointers for QFunctions.");
  } else {
    CeedCall(qf->SetHIPUserFunction(qf, f));
  }
  return CEED_ERROR_SUCCESS;
}
