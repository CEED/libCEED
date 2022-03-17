// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <ceed/cuda.h>
#include <ceed-impl.h>

/**
  @brief Set CUDA function pointer to evaluate action at quadrature points

  @param qf  CeedQFunction to set device pointer
  @param f   Device function pointer to evaluate action at quadrature points

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionSetCUDAUserFunction(CeedQFunction qf, CUfunction f) {
  int ierr;
  if (!qf->SetCUDAUserFunction) {
    Ceed ceed;
    ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
    CeedDebug(ceed, "Backend does not support CUfunction pointers for QFunctions.");
  } else {
    ierr = qf->SetCUDAUserFunction(qf, f); CeedChk(ierr);
  }
  return CEED_ERROR_SUCCESS;
}
