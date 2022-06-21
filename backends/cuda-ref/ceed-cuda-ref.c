// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include "ceed-cuda-ref.h"

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
  int ierr;
  Ceed_Cuda *data;
  ierr = CeedGetData(ceed, &data); CeedChkBackend(ierr);

  if (!data->cublas_handle) {
    ierr = cublasCreate(&data->cublas_handle); CeedChk_Cublas(ceed, ierr);
  }
  *handle = data->cublas_handle;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Init
//------------------------------------------------------------------------------
static int CeedInit_Cuda(const char *resource, Ceed ceed) {
  int ierr;

  char *resource_root;
  ierr = CeedCudaGetResourceRoot(ceed, resource, &resource_root);
  CeedChkBackend(ierr);
  if (strcmp(resource_root, "/gpu/cuda/ref"))
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Cuda backend cannot use resource: %s", resource);
  // LCOV_EXCL_STOP
  ierr = CeedFree(&resource_root); CeedChkBackend(ierr);
  ierr = CeedSetDeterministic(ceed, true); CeedChkBackend(ierr);

  Ceed_Cuda *data;
  ierr = CeedCalloc(1, &data); CeedChkBackend(ierr);
  ierr = CeedSetData(ceed, data); CeedChkBackend(ierr);
  ierr = CeedCudaInit(ceed, resource); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "GetPreferredMemType",
                                CeedGetPreferredMemType_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "VectorCreate",
                                CeedVectorCreate_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1",
                                CeedBasisCreateTensorH1_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateH1",
                                CeedBasisCreateH1_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "ElemRestrictionCreate",
                                CeedElemRestrictionCreate_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed,
                                "ElemRestrictionCreateBlocked",
                                CeedElemRestrictionCreateBlocked_Cuda);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionCreate",
                                CeedQFunctionCreate_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionContextCreate",
                                CeedQFunctionContextCreate_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate",
                                CeedOperatorCreate_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy",
                                CeedDestroy_Cuda); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Register
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Cuda(void) {
  return CeedRegister("/gpu/cuda/ref", CeedInit_Cuda, 40);
}
//------------------------------------------------------------------------------
