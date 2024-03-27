// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-magma-common.h"

#include <ceed.h>
#include <ceed/backend.h>
#include <stdlib.h>
#include <string.h>

//------------------------------------------------------------------------------
// Device information backend init
//------------------------------------------------------------------------------
int CeedInit_Magma_common(Ceed ceed, const char *resource) {
  Ceed_Magma *data;
  const char *device_spec = strstr(resource, ":device_id=");
  const int   device_id   = (device_spec) ? atoi(device_spec + 11) : -1;
  int         current_device_id;

  CeedCallBackend(magma_init());

  magma_getdevice(&current_device_id);
  if (device_id >= 0 && current_device_id != device_id) {
    magma_setdevice(device_id);
    current_device_id = device_id;
  }

  CeedCallBackend(CeedGetData(ceed, &data));
  data->device_id = current_device_id;
#ifdef CEED_MAGMA_USE_HIP
  magma_queue_create_from_hip(data->device_id, NULL, NULL, NULL, &(data->queue));
#else
  magma_queue_create_from_cuda(data->device_id, NULL, NULL, NULL, &(data->queue));
#endif
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend destroy
//------------------------------------------------------------------------------
int CeedDestroy_Magma(Ceed ceed) {
  Ceed_Magma *data;

  CeedCallBackend(CeedGetData(ceed, &data));
  magma_queue_destroy(data->queue);
  CeedCallBackend(magma_finalize());
  CeedCallBackend(CeedFree(&data));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
