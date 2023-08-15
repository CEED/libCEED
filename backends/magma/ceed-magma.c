// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-magma.h"

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
  const int nrc = 14;  // number of characters in resource
  CeedCheck(!strncmp(resource, "/gpu/cuda/magma", nrc) || !strncmp(resource, "/gpu/hip/magma", nrc), ceed, CEED_ERROR_BACKEND,
            "Magma backend cannot use resource: %s", resource);

  Ceed_Magma *data;
  CeedCallBackend(CeedCalloc(1, &data));
  CeedCallBackend(CeedSetData(ceed, data));

  // Get/set device ID
  const char *device_spec = strstr(resource, ":device_id=");
  const int   device_id   = (device_spec) ? atoi(device_spec + 11) : -1;
  int         current_device_id;
  CeedCallBackend(magma_init());
  magma_getdevice(&current_device_id);
  if (device_id >= 0 && current_device_id != device_id) {
    magma_setdevice(device_id);
    current_device_id = device_id;
  }
  data->device_id = current_device_id;

  // Create a queue that uses the null stream
#ifdef CEED_MAGMA_USE_HIP
  magma_queue_create_from_hip(data->device_id, NULL, NULL, NULL, &(data->queue));
#else
  magma_queue_create_from_cuda(data->device_id, NULL, NULL, NULL, &(data->queue));
#endif

  // Create reference Ceed that implementation will be dispatched through unless overridden
  Ceed ceed_ref;
#ifdef CEED_MAGMA_USE_HIP
  CeedCallBackend(CeedInit("/gpu/hip/ref", &ceed_ref));
#else
  CeedCallBackend(CeedInit("/gpu/cuda/ref", &ceed_ref));
#endif
  CeedCallBackend(CeedSetDelegate(ceed, ceed_ref));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1", CeedBasisCreateTensorH1_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateH1", CeedBasisCreateH1_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy", CeedDestroy_Magma));

  return CEED_ERROR_SUCCESS;
}

CEED_INTERN int CeedRegister_Magma(void) {
#ifdef CEED_MAGMA_USE_HIP
  return CeedRegister("/gpu/hip/magma", CeedInit_Magma, 120);
#else
  return CeedRegister("/gpu/cuda/magma", CeedInit_Magma, 120);
#endif
}
