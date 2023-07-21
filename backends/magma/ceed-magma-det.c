// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <stdlib.h>
#include <string.h>

#include "ceed-magma.h"
#ifdef CEED_MAGMA_USE_SYCL
#include "ceed-magma-sycl.h"
#endif

static int CeedInit_Magma_Det(const char *resource, Ceed ceed) {
  const int nrc = 18;  // number of characters in resource
  CeedCheck(!strncmp(resource, "/gpu/cuda/magma/det", nrc) || !strncmp(resource, "/gpu/hip/magma/det", nrc) ||
                !strncmp(resource, "/gpu/sycl/magma/det", nrc),
            ceed, CEED_ERROR_BACKEND, "Magma backend cannot use resource: %s", resource);
  CeedCallBackend(CeedSetDeterministic(ceed, true));

  Ceed_Magma *data;
  CeedCallBackend(CeedCalloc(sizeof(Ceed_Magma), &data));
  CeedCallBackend(CeedSetData(ceed, data));

  // get/set device ID
  const char *device_spec = strstr(resource, ":device_id=");
  int         deviceID    = (device_spec) ? atoi(device_spec + 11) : -1;

#ifndef CEED_MAGMA_USE_SYCL
  int currentDeviceID;
  magma_getdevice(&currentDeviceID);
  if (deviceID >= 0 && currentDeviceID != deviceID) {
    magma_setdevice(deviceID);
    currentDeviceID = deviceID;
  }
  data->device = currentDeviceID;
#endif

  // Create reference CEED that implementation will be dispatched
  //   through unless overridden
  Ceed ceedref;
#if defined(CEED_MAGMA_USE_HIP)
  CeedCallBackend(CeedInit("/gpu/hip/magma", &ceedref));
#elif defined(CEED_MAGMA_USE_CUDA)
  CeedCallBackend(CeedInit("/gpu/cuda/magma", &ceedref));
#else
  if (deviceID < 0) deviceID = 0;
  CeedInitMagma_Sycl(ceed, deviceID);
  CeedCallBackend(CeedInit("/gpu/sycl/magma", &ceedref));
  // Set the delegate SYCL queue to match the one we created
  void *sycl_queue = NULL;
  CeedCallBackend(CeedMagmaGetSyclHandle(ceed, &sycl_queue));
  CeedCallBackend(CeedSetStream(ceedref, sycl_queue));
  // Enable CeedSetStream for this backend
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "SetStream", CeedSetStream_Magma));
#endif
  CeedCallBackend(CeedSetDelegate(ceed, ceedref));

  // Create reference CEED for restriction
  Ceed restrictionceedref;
#if defined(CEED_MAGMA_USE_HIP)
  CeedInit("/gpu/hip/ref", &restrictionceedref);
#elif defined(CEED_MAGMA_USE_CUDA)
  CeedInit("/gpu/cuda/ref", &restrictionceedref);
#else
  CeedCallBackend(CeedInit("/gpu/sycl/ref", &restrictionceedref));
  // Set the delegate SYCL queue to match the one we created
  CeedCallBackend(CeedSetStream(restrictionceedref, sycl_queue));
#endif
  CeedCallBackend(CeedSetObjectDelegate(ceed, restrictionceedref, "ElemRestriction"));

  return CEED_ERROR_SUCCESS;
}

CEED_INTERN int CeedRegister_Magma_Det(void) {
#if defined(CEED_MAGMA_USE_HIP)
  return CeedRegister("/gpu/hip/magma/det", CeedInit_Magma_Det, 125);
#elif defined(CEED_MAGMA_USE_CUDA)
  return CeedRegister("/gpu/cuda/magma/det", CeedInit_Magma_Det, 125);
#else
  return CeedRegister("/gpu/sycl/magma/det", CeedInit_Magma_Det, 125);
#endif
}
