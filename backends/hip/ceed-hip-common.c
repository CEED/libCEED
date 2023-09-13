// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-hip-common.h"

#include <ceed.h>
#include <ceed/backend.h>
#include <stdlib.h>
#include <string.h>

//------------------------------------------------------------------------------
// Device information backend init
//------------------------------------------------------------------------------
int CeedInit_Hip(Ceed ceed, const char *resource) {
  Ceed_Hip   *data;
  const char *device_spec = strstr(resource, ":device_id=");
  const int   device_id   = (device_spec) ? atoi(device_spec + 11) : -1;
  int         current_device_id;

  CeedCallHip(ceed, hipGetDevice(&current_device_id));
  if (device_id >= 0 && current_device_id != device_id) {
    CeedCallHip(ceed, hipSetDevice(device_id));
    current_device_id = device_id;
  }

  CeedCallBackend(CeedGetData(ceed, &data));
  data->device_id = current_device_id;
  CeedCallHip(ceed, hipGetDeviceProperties(&data->device_prop, current_device_id));
  data->opt_block_size = 256;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Destroy
//------------------------------------------------------------------------------
int CeedDestroy_Hip(Ceed ceed) {
  Ceed_Hip *data;

  CeedCallBackend(CeedGetData(ceed, &data));
  if (data->hipblas_handle) CeedCallHipblas(ceed, hipblasDestroy(data->hipblas_handle));
  CeedCallBackend(CeedFree(&data));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
