// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <string.h>

#include "../ceed-backend-init.h"
#include "../hip/ceed-hip-common.h"
#include "ceed-hip-ref.h"

//------------------------------------------------------------------------------
// HIP preferred MemType
//------------------------------------------------------------------------------
static int CeedGetPreferredMemType_Hip(CeedMemType *mem_type) {
  *mem_type = CEED_MEM_DEVICE;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get hipBLAS handle
//------------------------------------------------------------------------------
int CeedGetHipblasHandle_Hip(Ceed ceed, hipblasHandle_t *handle) {
  Ceed_Hip *data;

  CeedCallBackend(CeedGetData(ceed, &data));
  if (!data->hipblas_handle) {
    CeedCallHipblas(ceed, hipblasCreate(&data->hipblas_handle));
    CeedCallHipblas(ceed, hipblasSetPointerMode(data->hipblas_handle, HIPBLAS_POINTER_MODE_HOST));
  }
  *handle = data->hipblas_handle;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Init
//------------------------------------------------------------------------------
CEED_INTERN int CeedInit_Hip_Ref(const char *resource, Ceed ceed) {
  Ceed_Hip *data;
  char     *resource_root;

  CeedCallBackend(CeedGetResourceRoot(ceed, resource, ":", &resource_root));
  CeedCheck(!strcmp(resource_root, "/gpu/hip/ref"), ceed, CEED_ERROR_BACKEND, "Hip backend cannot use resource: %s", resource);
  CeedCallBackend(CeedFree(&resource_root));
  CeedCallBackend(CeedSetDeterministic(ceed, true));

  CeedCallBackend(CeedCalloc(1, &data));
  CeedCallBackend(CeedSetData(ceed, data));
  CeedCallBackend(CeedInit_Hip(ceed, resource));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "GetPreferredMemType", CeedGetPreferredMemType_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "VectorCreate", CeedVectorCreate_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1", CeedBasisCreateTensorH1_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateH1", CeedBasisCreateH1_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateHdiv", CeedBasisCreateHdiv_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateHcurl", CeedBasisCreateHcurl_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "ElemRestrictionCreate", CeedElemRestrictionCreate_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "ElemRestrictionCreateAtPoints", CeedElemRestrictionCreate_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionCreate", CeedQFunctionCreate_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionContextCreate", CeedQFunctionContextCreate_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate", CeedOperatorCreate_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreateAtPoints", CeedOperatorCreateAtPoints_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy", CeedDestroy_Hip));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
