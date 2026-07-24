// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
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
    CeedDebug(CeedQFunctionReturnCeed(qf), "Backend does not support CUfunction pointers for QFunctions.");
  } else {
    CeedCall(qf->SetCUDAUserFunction(qf, f));
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Enable or disable CUDA Graph capture/replay for a `CeedOperator`

  @param[in,out] op           `CeedOperator`
  @param[in]     enable_graph Boolean flag to enable CUDA Graph use

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorSetEnableCudaGraph(CeedOperator op, bool enable_graph) {
  if (!op->SetEnableCudaGraph) {
    CeedDebug(CeedOperatorReturnCeed(op), "Backend does not support CUDA Graphs for operators.");
  } else {
    CeedCall(op->SetEnableCudaGraph(op, enable_graph));
  }
  return CEED_ERROR_SUCCESS;
}
