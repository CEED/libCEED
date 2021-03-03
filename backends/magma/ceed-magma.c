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

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <string.h>
#include <stdlib.h>
#include "ceed-magma.h"

static int CeedDestroy_Magma(Ceed ceed) {
  int ierr;
  Ceed_Magma *data;
  ierr = CeedGetData(ceed, &data); CeedChkBackend(ierr);
  magma_queue_destroy( data->queue );
  ierr = CeedFree(&data); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

static int CeedInit_Magma(const char *resource, Ceed ceed) {
  int ierr;
  const int nrc = 14; // number of characters in resource
  if (strncmp(resource, "/gpu/cuda/magma", nrc)
      && strncmp(resource, "/gpu/hip/magma", nrc))
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Magma backend cannot use resource: %s", resource);
  // LCOV_EXCL_STOP

  ierr = magma_init();
  if (ierr)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "error in magma_init(): %d\n", ierr);
  // LCOV_EXCL_STOP

  Ceed_Magma *data;
  ierr = CeedCalloc(sizeof(Ceed_Magma), &data); CeedChkBackend(ierr);
  ierr = CeedSetData(ceed, data); CeedChkBackend(ierr);

  // kernel selection
  data->basis_kernel_mode = MAGMA_KERNEL_DIM_SPECIFIC;

  // get/set device ID
  const char *device_spec = strstr(resource, ":device_id=");
  const int deviceID = (device_spec) ? atoi(device_spec+11) : -1;

  int currentDeviceID;
  magma_getdevice(&currentDeviceID);
  if (deviceID >= 0 && currentDeviceID != deviceID) {
    magma_setdevice(deviceID);
    currentDeviceID = deviceID;
  }
  // create a queue that uses the null stream
  data->device = currentDeviceID;
  #ifdef HAVE_HIP
  magma_queue_create_from_hip(data->device, NULL, NULL, NULL, &(data->queue));
  #else
  magma_queue_create_from_cuda(data->device, NULL, NULL, NULL, &(data->queue));
  #endif

  // Create reference CEED that implementation will be dispatched
  //   through unless overridden
  Ceed ceedref;
  #ifdef HAVE_HIP
  CeedInit("/gpu/hip/ref", &ceedref);
  #else
  CeedInit("/gpu/cuda/ref", &ceedref);
  #endif
  ierr = CeedSetDelegate(ceed, ceedref); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "ElemRestrictionCreate",
                                CeedElemRestrictionCreate_Magma); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed,
                                "ElemRestrictionCreateBlocked",
                                CeedElemRestrictionCreateBlocked_Magma); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1",
                                CeedBasisCreateTensorH1_Magma); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateH1",
                                CeedBasisCreateH1_Magma); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy",
                                CeedDestroy_Magma); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

CEED_INTERN int CeedRegister_Magma(void) {
  #ifdef HAVE_HIP
  return CeedRegister("/gpu/hip/magma", CeedInit_Magma, 120);
  #else
  return CeedRegister("/gpu/cuda/magma", CeedInit_Magma, 120);
  #endif
}
