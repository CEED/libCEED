// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other
// CEED contributors. All Rights Reserved. See the top-level LICENSE and NOTICE
// files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>

#include <string>
#include <sycl/sycl.hpp>
#include <vector>

#include "../sycl/ceed-sycl-common.hpp"
#include "../sycl/ceed-sycl-compile.hpp"
#include "ceed-sycl-ref-qfunction-load.hpp"
#include "ceed-sycl-ref.hpp"

#define WG_SIZE_QF 384

//------------------------------------------------------------------------------
// Apply QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionApply_Sycl(CeedQFunction qf, CeedInt Q, CeedVector *U, CeedVector *V) {
  Ceed                ceed;
  Ceed_Sycl          *ceed_Sycl;
  void               *context_data;
  CeedInt             num_input_fields, num_output_fields;
  CeedQFunction_Sycl *impl;

  CeedCallBackend(CeedQFunctionGetData(qf, &impl));

  // Build and compile kernel, if not done
  if (!impl->QFunction) CeedCallBackend(CeedQFunctionBuildKernel_Sycl(qf));

  CeedCallBackend(CeedQFunctionGetCeed(qf, &ceed));
  CeedCallBackend(CeedGetData(ceed, &ceed_Sycl));

  CeedCallBackend(CeedQFunctionGetNumArgs(qf, &num_input_fields, &num_output_fields));

  // Read vectors
  std::vector<const CeedScalar *> inputs(num_input_fields);
  const CeedVector               *U_i = U;
  for (auto &input_i : inputs) {
    CeedCallBackend(CeedVectorGetArrayRead(*U_i, CEED_MEM_DEVICE, &input_i));
    ++U_i;
  }

  std::vector<CeedScalar *> outputs(num_output_fields);
  CeedVector               *V_i = V;
  for (auto &output_i : outputs) {
    CeedCallBackend(CeedVectorGetArrayWrite(*V_i, CEED_MEM_DEVICE, &output_i));
    ++V_i;
  }

  // Get context data
  CeedCallBackend(CeedQFunctionGetInnerContextData(qf, CEED_MEM_DEVICE, &context_data));

  // Order queue
  sycl::event e = ceed_Sycl->sycl_queue.ext_oneapi_submit_barrier();

  // Launch as a basic parallel_for over Q quadrature points
  ceed_Sycl->sycl_queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on({e});

    int iarg{};
    cgh.set_arg(iarg, context_data);
    ++iarg;
    cgh.set_arg(iarg, Q);
    ++iarg;
    for (auto &input_i : inputs) {
      cgh.set_arg(iarg, input_i);
      ++iarg;
    }
    for (auto &output_i : outputs) {
      cgh.set_arg(iarg, output_i);
      ++iarg;
    }
    // Hard-coding the work-group size for now
    // We could use the Level Zero API to query and set an appropriate size in future
    // Equivalent of CUDA Occupancy Calculator
    int               wg_size   = WG_SIZE_QF;
    sycl::range<1>    rounded_Q = ((Q + (wg_size - 1)) / wg_size) * wg_size;
    sycl::nd_range<1> kernel_range(rounded_Q, wg_size);
    cgh.parallel_for(kernel_range, *(impl->QFunction));
  });

  // Restore vectors
  U_i = U;
  for (auto &input_i : inputs) {
    CeedCallBackend(CeedVectorRestoreArrayRead(*U_i, &input_i));
    ++U_i;
  }

  V_i = V;
  for (auto &output_i : outputs) {
    CeedCallBackend(CeedVectorRestoreArray(*V_i, &output_i));
    ++V_i;
  }

  // Restore context
  CeedCallBackend(CeedQFunctionRestoreInnerContextData(qf, &context_data));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionDestroy_Sycl(CeedQFunction qf) {
  Ceed                ceed;
  CeedQFunction_Sycl *impl;

  CeedCallBackend(CeedQFunctionGetData(qf, &impl));
  CeedCallBackend(CeedQFunctionGetCeed(qf, &ceed));
  delete impl->QFunction;
  delete impl->sycl_module;
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create QFunction
//------------------------------------------------------------------------------
int CeedQFunctionCreate_Sycl(CeedQFunction qf) {
  Ceed                ceed;
  CeedQFunction_Sycl *impl;

  CeedCallBackend(CeedQFunctionGetCeed(qf, &ceed));
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedQFunctionSetData(qf, impl));
  // Register backend functions
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "QFunction", qf, "Apply", CeedQFunctionApply_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "QFunction", qf, "Destroy", CeedQFunctionDestroy_Sycl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
