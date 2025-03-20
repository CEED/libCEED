// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-sycl-shared.hpp"

#include <ceed/backend.h>
#include <ceed/ceed.h>

#include <sstream>
#include <string>
#include <string_view>

//------------------------------------------------------------------------------
// Backend init
//------------------------------------------------------------------------------
static int CeedInit_Sycl_shared(const char *resource, Ceed ceed) {
  Ceed       ceed_ref;
  Ceed_Sycl *data;
  char      *resource_root;

  CeedCallBackend(CeedGetResourceRoot(ceed, resource, ":", &resource_root));
  CeedCheck(!std::strcmp(resource_root, "/gpu/sycl/shared") || !std::strcmp(resource_root, "/cpu/sycl/shared"), ceed, CEED_ERROR_BACKEND,
            "Sycl backend cannot use resource: %s", resource);
  std::string_view root_view = resource_root;

  auto suffix_length = root_view.size() - root_view.rfind("shared");
  root_view.remove_suffix(suffix_length);

  std::ostringstream ref_resource;
  ref_resource << root_view << "ref";

  CeedCallBackend(CeedFree(&resource_root));
  CeedCallBackend(CeedSetDeterministic(ceed, true));

  CeedCallBackend(CeedCalloc(1, &data));
  CeedCallBackend(CeedSetData(ceed, data));
  CeedCallBackend(CeedInit_Sycl(ceed, resource));

  CeedCallBackend(CeedInit(ref_resource.str().c_str(), &ceed_ref));
  CeedCallBackend(CeedSetStream_Sycl(ceed_ref, &(data->sycl_queue)));
  CeedCallBackend(CeedSetDelegate(ceed, ceed_ref));

  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Ceed", ceed, "BasisCreateTensorH1", CeedBasisCreateTensorH1_Sycl_shared));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Ceed", ceed, "Destroy", CeedDestroy_Sycl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Register backend
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Sycl_Shared(void) {
  CeedCallBackend(CeedRegister("/gpu/sycl/shared", CeedInit_Sycl_shared, 25));
  CeedCallBackend(CeedRegister("/cpu/sycl/shared", CeedInit_Sycl_shared, 35));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
