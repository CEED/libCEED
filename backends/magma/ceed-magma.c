// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-magma.h"
#ifdef CEED_MAGMA_USE_SYCL
#include "ceed-magma-sycl.h"
#endif

#include <ceed.h>
#include <ceed/backend.h>
#include <stdlib.h>
#include <string.h>

static int CeedDestroy_Magma(Ceed ceed) {
  Ceed_Magma *data;
  CeedCallBackend(CeedGetData(ceed, &data));
  magma_queue_destroy(data->queue);
  CeedCallBackend(CeedFree(&data));
  return CEED_ERROR_SUCCESS;
}

static int CeedInit_Magma(const char *resource, Ceed ceed) {
  int       ierr;
  const int nrc = 14;  // number of characters in resource
  CeedCheck(!strncmp(resource, "/gpu/cuda/magma", nrc) || !strncmp(resource, "/gpu/hip/magma", nrc) || !strncmp(resource, "/gpu/sycl/magma", nrc),
            ceed, CEED_ERROR_BACKEND, "Magma backend cannot use resource: %s", resource);

  ierr = magma_init();
  CeedCheck(ierr == 0, ceed, CEED_ERROR_BACKEND, "error in magma_init(): %d\n", ierr);

  Ceed_Magma *data;
  CeedCallBackend(CeedCalloc(sizeof(Ceed_Magma), &data));
  CeedCallBackend(CeedSetData(ceed, data));

  // kernel selection
  data->basis_kernel_mode = MAGMA_KERNEL_DIM_SPECIFIC;

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

  // create a queue that uses the null stream
  data->device = currentDeviceID;
#if defined(CEED_MAGMA_USE_HIP)
  magma_queue_create_from_hip(data->device, NULL, NULL, NULL, &(data->queue));
#elif defined(CEED_MAGMA_USE_CUDA)
  magma_queue_create_from_cuda(data->device, NULL, NULL, NULL, &(data->queue));
#endif
#else  // Using SYCL
  if (deviceID < 0) deviceID = 0;
  CeedInitMagma_Sycl(ceed, deviceID);
#endif

  // Create reference CEED that implementation will be dispatched
  //   through unless overridden
  Ceed ceedref;
#if defined(CEED_MAGMA_USE_HIP)
  CeedCallBackend(CeedInit("/gpu/hip/ref", &ceedref));
#elif defined(CEED_MAGMA_USE_CUDA)
  CeedCallBackend(CeedInit("/gpu/cuda/ref", &ceedref));
#else
  CeedCallBackend(CeedInit("/gpu/sycl/ref", &ceedref));
  void *sycl_queue = NULL;
  CeedCallBackend(CeedMagmaGetSyclHandle(ceed, &sycl_queue));
  // Set the delegate SYCL queue to match the one we created
  CeedCallBackend(CeedSetStream(ceedref, sycl_queue));
  // Enable CeedSetStream for this backend
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "SetStream", CeedSetStream_Magma));
#endif
  CeedCallBackend(CeedSetDelegate(ceed, ceedref));
#ifndef CEED_MAGMA_USE_SYCL
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "ElemRestrictionCreate", CeedElemRestrictionCreate_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1", CeedBasisCreateTensorH1_Magma));
#endif
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateH1", CeedBasisCreateH1_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy", CeedDestroy_Magma));
  return CEED_ERROR_SUCCESS;
}

CEED_INTERN int CeedRegister_Magma(void) {
#if defined(CEED_MAGMA_USE_HIP)
  return CeedRegister("/gpu/hip/magma", CeedInit_Magma, 120);
#elif defined(CEED_MAGMA_USE_CUDA)
  return CeedRegister("/gpu/cuda/magma", CeedInit_Magma, 120);
#else
  return CeedRegister("/gpu/sycl/magma", CeedInit_Magma, 120);
#endif
}
