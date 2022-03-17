// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <string.h>
#include <stdlib.h>
#include "ceed-hip-common.h"

//------------------------------------------------------------------------------
// Device information backend init
//------------------------------------------------------------------------------
int CeedHipInit(Ceed ceed, const char *resource) {
  int ierr;
  const char *device_spec = strstr(resource, ":device_id=");
  const int device_id = (device_spec) ? atoi(device_spec + 11) : -1;

  int current_device_id;
  ierr = hipGetDevice(&current_device_id); CeedChk_Hip(ceed, ierr);
  if (device_id >= 0 && current_device_id != device_id) {
    ierr = hipSetDevice(device_id); CeedChk_Hip(ceed, ierr);
    current_device_id = device_id;
  }

  struct hipDeviceProp_t device_prop;
  ierr = hipGetDeviceProperties(&device_prop, current_device_id);
  CeedChk_Hip(ceed, ierr);

  Ceed_Hip *data;
  ierr = CeedGetData(ceed, &data); CeedChkBackend(ierr);
  data->device_id = current_device_id;
  data->opt_block_size = 256;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Destroy
//------------------------------------------------------------------------------
int CeedDestroy_Hip(Ceed ceed) {
  int ierr;
  Ceed_Hip *data;
  ierr = CeedGetData(ceed, &data); CeedChkBackend(ierr);
  if (data->hipblas_handle) {
    ierr = hipblasDestroy(data->hipblas_handle); CeedChk_Hipblas(ceed, ierr);
  }
  ierr = CeedFree(&data); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
