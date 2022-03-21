// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <string.h>
#include "ceed-cuda-common.h"

//------------------------------------------------------------------------------
// Device information backend init
//------------------------------------------------------------------------------
int CeedCudaInit(Ceed ceed, const char *resource) {
  int ierr;
  const char *device_spec = strstr(resource, ":device_id=");
  const int device_id = (device_spec) ? atoi(device_spec + 11) : -1;

  int current_device_id;
  ierr = cudaGetDevice(&current_device_id); CeedChk_Cu(ceed, ierr);
  if (device_id >= 0 && current_device_id != device_id) {
    ierr = cudaSetDevice(device_id); CeedChk_Cu(ceed, ierr);
    current_device_id = device_id;
  }
  Ceed_Cuda *data;
  ierr = CeedGetData(ceed, &data); CeedChkBackend(ierr);
  data->device_id = current_device_id;
  ierr = cudaGetDeviceProperties(&data->device_prop, current_device_id);
  CeedChk_Cu(ceed, ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend destroy
//------------------------------------------------------------------------------
int CeedDestroy_Cuda(Ceed ceed) {
  int ierr;
  Ceed_Cuda *data;
  ierr = CeedGetData(ceed, &data); CeedChkBackend(ierr);
  if (data->cublas_handle) {
    ierr = cublasDestroy(data->cublas_handle); CeedChk_Cublas(ceed, ierr);
  }
  ierr = CeedFree(&data); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
