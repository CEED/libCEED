// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-cuda-common.h"

#include <string.h>

//------------------------------------------------------------------------------
// Get root resource without device spec
//------------------------------------------------------------------------------
int CeedCudaGetResourceRoot(Ceed ceed, const char *resource, char **resource_root) {
  char  *device_spec       = strstr(resource, ":device_id=");
  size_t resource_root_len = device_spec ? (size_t)(device_spec - resource) + 1 : strlen(resource) + 1;
  CeedCallBackend(CeedCalloc(resource_root_len, resource_root));
  memcpy(*resource_root, resource, resource_root_len - 1);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Device information backend init
//------------------------------------------------------------------------------
int CeedCudaInit(Ceed ceed, const char *resource) {
  const char *device_spec = strstr(resource, ":device_id=");
  const int   device_id   = (device_spec) ? atoi(device_spec + 11) : -1;

  int current_device_id;
  CeedCallCuda(ceed, cudaGetDevice(&current_device_id));
  if (device_id >= 0 && current_device_id != device_id) {
    CeedCallCuda(ceed, cudaSetDevice(device_id));
    current_device_id = device_id;
  }
  Ceed_Cuda *data;
  CeedCallBackend(CeedGetData(ceed, &data));
  data->device_id = current_device_id;
  CeedCallCuda(ceed, cudaGetDeviceProperties(&data->device_prop, current_device_id));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend destroy
//------------------------------------------------------------------------------
int CeedDestroy_Cuda(Ceed ceed) {
  Ceed_Cuda *data;
  CeedCallBackend(CeedGetData(ceed, &data));
  if (data->cublas_handle) CeedCallCublas(ceed, cublasDestroy(data->cublas_handle));
  CeedCallBackend(CeedFree(&data));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
