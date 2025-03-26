// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
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
  int         current_device_id, xnack_value;
  const char *xnack;

  CeedCallHip(ceed, hipGetDevice(&current_device_id));
  if (device_id >= 0 && current_device_id != device_id) {
    CeedCallHip(ceed, hipSetDevice(device_id));
    current_device_id = device_id;
  }

  CeedCallBackend(CeedGetData(ceed, &data));
  data->device_id = current_device_id;
  CeedCallHip(ceed, hipGetDeviceProperties(&data->device_prop, current_device_id));
  xnack                        = getenv("HSA_XNACK");
  xnack_value                  = !!xnack ? atol(xnack) : 0;
  data->has_unified_addressing = xnack_value > 0 ? data->device_prop.unifiedAddressing : 0;
  if (data->has_unified_addressing) {
    CeedDebug(ceed, "Using unified memory addressing");
  }
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
// Memory transfer utilities
//------------------------------------------------------------------------------
static inline int CeedSetDeviceGenericArray_Hip(Ceed ceed, const void *source_array, CeedCopyMode copy_mode, size_t size_unit, CeedSize num_values,
                                                void *target_array_owned, void *target_array_borrowed, void *target_array) {
  switch (copy_mode) {
    case CEED_COPY_VALUES:
      if (!*(void **)target_array) {
        if (*(void **)target_array_borrowed) {
          *(void **)target_array = *(void **)target_array_borrowed;
        } else {
          if (!*(void **)target_array_owned) CeedCallHip(ceed, hipMalloc(target_array_owned, size_unit * num_values));
          *(void **)target_array = *(void **)target_array_owned;
        }
      }
      if (source_array) CeedCallHip(ceed, hipMemcpy(*(void **)target_array, source_array, size_unit * num_values, hipMemcpyDeviceToDevice));
      break;
    case CEED_OWN_POINTER:
      CeedCallHip(ceed, hipFree(*(void **)target_array_owned));
      *(void **)target_array_owned    = (void *)source_array;
      *(void **)target_array_borrowed = NULL;
      *(void **)target_array          = *(void **)target_array_owned;
      break;
    case CEED_USE_POINTER:
      CeedCallHip(ceed, hipFree(*(void **)target_array_owned));
      *(void **)target_array_owned    = NULL;
      *(void **)target_array_borrowed = (void *)source_array;
      *(void **)target_array          = *(void **)target_array_borrowed;
  }
  return CEED_ERROR_SUCCESS;
}

int CeedSetDeviceBoolArray_Hip(Ceed ceed, const bool *source_array, CeedCopyMode copy_mode, CeedSize num_values, const bool **target_array_owned,
                               const bool **target_array_borrowed, const bool **target_array) {
  CeedCallBackend(CeedSetDeviceGenericArray_Hip(ceed, source_array, copy_mode, sizeof(bool), num_values, target_array_owned, target_array_borrowed,
                                                target_array));
  return CEED_ERROR_SUCCESS;
}

int CeedSetDeviceCeedInt8Array_Hip(Ceed ceed, const CeedInt8 *source_array, CeedCopyMode copy_mode, CeedSize num_values,
                                   const CeedInt8 **target_array_owned, const CeedInt8 **target_array_borrowed, const CeedInt8 **target_array) {
  CeedCallBackend(CeedSetDeviceGenericArray_Hip(ceed, source_array, copy_mode, sizeof(CeedInt8), num_values, target_array_owned,
                                                target_array_borrowed, target_array));
  return CEED_ERROR_SUCCESS;
}

int CeedSetDeviceCeedIntArray_Hip(Ceed ceed, const CeedInt *source_array, CeedCopyMode copy_mode, CeedSize num_values,
                                  const CeedInt **target_array_owned, const CeedInt **target_array_borrowed, const CeedInt **target_array) {
  CeedCallBackend(CeedSetDeviceGenericArray_Hip(ceed, source_array, copy_mode, sizeof(CeedInt), num_values, target_array_owned, target_array_borrowed,
                                                target_array));
  return CEED_ERROR_SUCCESS;
}

int CeedSetDeviceCeedScalarArray_Hip(Ceed ceed, const CeedScalar *source_array, CeedCopyMode copy_mode, CeedSize num_values,
                                     const CeedScalar **target_array_owned, const CeedScalar **target_array_borrowed,
                                     const CeedScalar **target_array) {
  CeedCallBackend(CeedSetDeviceGenericArray_Hip(ceed, source_array, copy_mode, sizeof(CeedScalar), num_values, target_array_owned,
                                                target_array_borrowed, target_array));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
