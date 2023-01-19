// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other
// CEED contributors. All Rights Reserved. See the top-level LICENSE and NOTICE
// files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-sycl-common.hpp"

#include <string>
#include <sycl/sycl.hpp>

//------------------------------------------------------------------------------
// Get root resource without device spec
//------------------------------------------------------------------------------
int CeedSyclGetResourceRoot(Ceed ceed, const char *resource, char **resource_root) {
  const char *device_spec       = std::strstr(resource, ":device_id=");
  size_t      resource_root_len = device_spec ? (size_t)(device_spec - resource) + 1 : strlen(resource) + 1;
  CeedCallBackend(CeedCalloc(resource_root_len, resource_root));
  memcpy(*resource_root, resource, resource_root_len - 1);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Device information backend init
//------------------------------------------------------------------------------
int CeedSyclInit(Ceed ceed, const char *resource) {
  const char *device_spec = std::strstr(resource, ":device_id=");
  const int   device_id   = (device_spec) ? atoi(device_spec + 11) : 0;

  sycl::info::device_type device_type;
  if (std::strstr(resource, "/gpu/sycl/ref")) {
    device_type = sycl::info::device_type::gpu;
  } else if (std::strstr(resource, "/cpu/sycl/ref")) {
    device_type = sycl::info::device_type::cpu;
  } else {
    return CeedError(ceed, CEED_ERROR_BACKEND, "Unsupported SYCL device type requested");
  }

  auto sycl_devices = sycl::device::get_devices(device_type);
  int  device_count = sycl_devices.size();

  if (0 == device_count) {
    return CeedError(ceed, CEED_ERROR_BACKEND, "No SYCL devices of the requested type are available");
  }

  // Validate the requested device_id
  if (device_count < device_id + 1) {
    return CeedError(ceed, CEED_ERROR_BACKEND, "Invalid SYCL device id requested");
  }

  sycl::device sycl_device{sycl_devices[device_id]};
  // Check that the device supports explicit device allocations
  if (!sycl_device.has(sycl::aspect::usm_device_allocations)) {
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "The requested SYCL device does not support explicit "
                     "device allocations.");
  }

  sycl::context sycl_context{sycl_device.get_platform().get_devices()};
  sycl::queue   sycl_queue{sycl_context, sycl_device};

  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  data->sycl_device  = sycl_device;
  data->sycl_context = sycl_context;
  data->sycl_queue   = sycl_queue;

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
