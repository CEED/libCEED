// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other
// CEED contributors. All Rights Reserved. See the top-level LICENSE and NOTICE
// files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-sycl-ref.hpp"

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <string>

#include <sycl/sycl.hpp>

//------------------------------------------------------------------------------
// SYCL preferred MemType
//------------------------------------------------------------------------------
static int CeedGetPreferredMemType_Sycl(CeedMemType *mem_type) {
  *mem_type = CEED_MEM_DEVICE;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Init
//------------------------------------------------------------------------------
static int CeedInit_Sycl(const char *resource, Ceed ceed) {
  char *resource_root;
  CeedCallBackend(CeedSyclGetResourceRoot(ceed, resource, &resource_root));
  if (strcmp(resource_root, "/gpu/sycl/ref")) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Sycl backend cannot use resource: %s", resource);
    // LCOV_EXCL_STOP
  }
  CeedCallBackend(CeedFree(&resource_root));
  CeedCallBackend(CeedSetDeterministic(ceed, true));

  Ceed_Sycl *data;
  CeedCallBackend(CeedCalloc(1, &data));
  CeedCallBackend(CeedSetData(ceed, data));
  CeedCallBackend(CeedSyclInit(ceed, resource));

  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Ceed", "GetPreferredMemType", CeedGetPreferredMemType_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Ceed", "VectorCreate",&CeedVectorCreate_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Ceed", "BasisCreateTensorH1",&CeedBasisCreateTensorH1_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Ceed", "BasisCreateH1",&CeedBasisCreateH1_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Ceed", "ElemRestrictionCreate",&CeedElemRestrictionCreate_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Ceed", "ElemRestrictionCreateBlocked",&CeedElemRestrictionCreateBlocked_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Ceed", "QFunctionCreate",&CeedQFunctionCreate_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Ceed", "QFunctionContextCreate",&CeedQFunctionContextCreate_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Ceed", "OperatorCreate",&CeedOperatorCreate_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Ceed", "Destroy",&CeedDestroy_Sycl));
  
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Register
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Sycl(void) {
  return CeedRegister("/gpu/sycl/ref", CeedInit_Sycl, 40);
}
//------------------------------------------------------------------------------
