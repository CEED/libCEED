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
  CeedCallBackend(CeedGetResourceRoot(ceed, resource, ":", &resource_root));
  CeedCheck(!std::strcmp(resource_root, "/gpu/sycl/ref") || !std::strcmp(resource_root, "/cpu/sycl/ref"), ceed, CEED_ERROR_BACKEND,
            "Sycl backend cannot use resource: %s", resource);
  CeedCallBackend(CeedFree(&resource_root));
  CeedCallBackend(CeedSetDeterministic(ceed, true));

  Ceed_Sycl *data;
  CeedCallBackend(CeedCalloc(1, &data));
  CeedCallBackend(CeedSetData(ceed, data));
  CeedCallBackend(CeedSyclInit(ceed, resource));

  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Ceed", ceed, "SetStream", CeedSetStream_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Ceed", ceed, "GetPreferredMemType", CeedGetPreferredMemType_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Ceed", ceed, "VectorCreate", &CeedVectorCreate_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Ceed", ceed, "BasisCreateTensorH1", &CeedBasisCreateTensorH1_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Ceed", ceed, "BasisCreateH1", &CeedBasisCreateH1_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Ceed", ceed, "ElemRestrictionCreate", &CeedElemRestrictionCreate_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Ceed", ceed, "QFunctionCreate", &CeedQFunctionCreate_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Ceed", ceed, "QFunctionContextCreate", &CeedQFunctionContextCreate_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Ceed", ceed, "OperatorCreate", &CeedOperatorCreate_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Ceed", ceed, "Destroy", &CeedDestroy_Sycl));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Register
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Sycl(void) {
  CeedCallBackend(CeedRegister("/gpu/sycl/ref", CeedInit_Sycl, 40));
  CeedCallBackend(CeedRegister("/cpu/sycl/ref", CeedInit_Sycl, 50));
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
