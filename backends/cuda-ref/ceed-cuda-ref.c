// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-cuda-ref.h"

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>

//------------------------------------------------------------------------------
// CUDA preferred MemType
//------------------------------------------------------------------------------
static int CeedGetPreferredMemType_Cuda(CeedMemType *mem_type) {
  *mem_type = CEED_MEM_DEVICE;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get CUBLAS handle
//------------------------------------------------------------------------------
int CeedCudaGetCublasHandle(Ceed ceed, cublasHandle_t *handle) {
  Ceed_Cuda *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  if (!data->cublas_handle) CeedCallCublas(ceed, cublasCreate(&data->cublas_handle));
  *handle = data->cublas_handle;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Init
//------------------------------------------------------------------------------
static int CeedInit_Cuda(const char *resource, Ceed ceed) {
  char *resource_root;
  CeedCallBackend(CeedCudaGetResourceRoot(ceed, resource, &resource_root));
  if (strcmp(resource_root, "/gpu/cuda/ref")) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Cuda backend cannot use resource: %s", resource);
    // LCOV_EXCL_STOP
  }
  CeedCallBackend(CeedFree(&resource_root));
  CeedCallBackend(CeedSetDeterministic(ceed, true));

  Ceed_Cuda *data;
  CeedCallBackend(CeedCalloc(1, &data));
  CeedCallBackend(CeedSetData(ceed, data));
  CeedCallBackend(CeedCudaInit(ceed, resource));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "GetPreferredMemType", CeedGetPreferredMemType_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "VectorCreate", CeedVectorCreate_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1", CeedBasisCreateTensorH1_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateH1", CeedBasisCreateH1_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "ElemRestrictionCreate", CeedElemRestrictionCreate_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "ElemRestrictionCreateBlocked", CeedElemRestrictionCreateBlocked_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionCreate", CeedQFunctionCreate_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionContextCreate", CeedQFunctionContextCreate_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate", CeedOperatorCreate_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy", CeedDestroy_Cuda));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Register
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Cuda(void) { return CeedRegister("/gpu/cuda/ref", CeedInit_Cuda, 40); }
//------------------------------------------------------------------------------
