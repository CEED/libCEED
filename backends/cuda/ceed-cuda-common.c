// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

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
