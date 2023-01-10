// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-sycl-common.hpp"

#include <sycl/sycl.hpp>
#include <string>

//------------------------------------------------------------------------------
// Get root resource without device spec
//------------------------------------------------------------------------------
int CeedSyclGetResourceRoot(Ceed ceed, const char *resource, char **resource_root) {
  const char  *device_spec = strstr(resource, ":device_id=");
  size_t resource_root_len = device_spec ? (size_t)(device_spec - resource) + 1 : strlen(resource) + 1;
  CeedCallBackend(CeedCalloc(resource_root_len, resource_root));
  memcpy(*resource_root, resource, resource_root_len - 1);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Device information backend init
//------------------------------------------------------------------------------
int CeedSyclInit(Ceed ceed, const char *resource) {
  const char *device_spec = strstr(resource, ":device_id=");
  const int   device_id   = (device_spec) ? atoi(device_spec + 11) : 0;
  
  // For now assume we want GPU devices and ignore the possibility of multiple platforms
  auto gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  int device_count = gpu_devices.size();

  // Validate the requested device_id
  if (device_id < 0 || (device_count < device_id+1)) {
    return CeedError(ceed, CEED_ERROR_BACKEND, "Invalid SYCL device id requested");
  } 

  sycl::device sycl_device{gpu_devices[device_id]};
  sycl::context sycl_context{gpu_devices};
  sycl::queue sycl_queue{sycl_context,sycl_device};

  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));
  
  data->sycl_device = sycl_device;
  data->sycl_context= sycl_context;
  data->sycl_queue  = sycl_queue;
  
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend destroy
//------------------------------------------------------------------------------
int CeedDestroy_Sycl(Ceed ceed) {
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));
  CeedCallBackend(CeedFree(&data));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
