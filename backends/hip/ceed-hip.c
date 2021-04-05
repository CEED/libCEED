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
#include "ceed-hip.h"

//------------------------------------------------------------------------------
// HIP preferred MemType
//------------------------------------------------------------------------------
static int CeedGetPreferredMemType_Hip(CeedMemType *type) {
  *type = CEED_MEM_DEVICE;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Device information backend init
//------------------------------------------------------------------------------
int CeedHipInit(Ceed ceed, const char *resource, int nrc) {
  int ierr;
  const char *device_spec = strstr(resource, ":device_id=");
  const int deviceID = (device_spec) ? atoi(device_spec+11) : -1;

  int currentDeviceID;
  ierr = hipGetDevice(&currentDeviceID); CeedChk_Hip(ceed,ierr);
  if (deviceID >= 0 && currentDeviceID != deviceID) {
    ierr = hipSetDevice(deviceID); CeedChk_Hip(ceed,ierr);
    currentDeviceID = deviceID;
  }

  struct hipDeviceProp_t deviceProp;
  ierr = hipGetDeviceProperties(&deviceProp, currentDeviceID);
  CeedChk_Hip(ceed,ierr);

  Ceed_Hip *data;
  ierr = CeedGetData(ceed, &data); CeedChkBackend(ierr);
  data->deviceId = currentDeviceID;
  data->optblocksize = 256;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get hipBLAS handle
//------------------------------------------------------------------------------
int CeedHipGetHipblasHandle(Ceed ceed, hipblasHandle_t *handle) {
  int ierr;
  Ceed_Hip *data;
  ierr = CeedGetData(ceed, &data); CeedChkBackend(ierr);

  if (!data->hipblasHandle) {
    ierr = hipblasCreate(&data->hipblasHandle); CeedChk_Hipblas(ceed, ierr);
  }
  *handle = data->hipblasHandle;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Destroy
//------------------------------------------------------------------------------
int CeedDestroy_Hip(Ceed ceed) {
  int ierr;
  Ceed_Hip *data;
  ierr = CeedGetData(ceed, &data); CeedChkBackend(ierr);
  if (data->hipblasHandle) {
    ierr = hipblasDestroy(data->hipblasHandle); CeedChk_Hipblas(ceed, ierr);
  }
  ierr = CeedFree(&data); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Init
//------------------------------------------------------------------------------
static int CeedInit_Hip(const char *resource, Ceed ceed) {
  int ierr;
  const int nrc = 8; // number of characters in resource
  if (strncmp(resource, "/gpu/hip/ref", nrc))
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Hip backend cannot use resource: %s", resource);
  // LCOV_EXCL_STOP
  ierr = CeedSetDeterministic(ceed, true); CeedChk(ierr);

  Ceed_Hip *data;
  ierr = CeedCalloc(1, &data); CeedChkBackend(ierr);
  ierr = CeedSetData(ceed, data); CeedChkBackend(ierr);
  ierr = CeedHipInit(ceed, resource, nrc); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "GetPreferredMemType",
                                CeedGetPreferredMemType_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "VectorCreate",
                                CeedVectorCreate_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1",
                                CeedBasisCreateTensorH1_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateH1",
                                CeedBasisCreateH1_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "ElemRestrictionCreate",
                                CeedElemRestrictionCreate_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed,
                                "ElemRestrictionCreateBlocked",
                                CeedElemRestrictionCreateBlocked_Hip);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionCreate",
                                CeedQFunctionCreate_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionContextCreate",
                                CeedQFunctionContextCreate_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate",
                                CeedOperatorCreate_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "CompositeOperatorCreate",
                                CeedCompositeOperatorCreate_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy",
                                CeedDestroy_Hip); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Register
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Hip(void) {
  return CeedRegister("/gpu/hip/ref", CeedInit_Hip, 40);
}
//------------------------------------------------------------------------------
