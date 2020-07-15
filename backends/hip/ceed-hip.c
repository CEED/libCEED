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
#include <stdarg.h>
#include "ceed-hip.h"

//------------------------------------------------------------------------------
// HIP preferred MemType
//------------------------------------------------------------------------------
static int CeedGetPreferredMemType_Hip(CeedMemType *type) {
  *type = CEED_MEM_DEVICE;
  return 0;
}

//------------------------------------------------------------------------------
// Device information backend init
//------------------------------------------------------------------------------
int CeedHipInit(Ceed ceed, const char *resource, int nrc) {
  int ierr;
  const int rlen = strlen(resource);
  const bool slash = (rlen>nrc) ? (resource[nrc] == '/') : false;
  const int deviceID = (slash && rlen > nrc + 1) ? atoi(&resource[nrc + 1]) : 0;

  int currentDeviceID;
  ierr = hipGetDevice(&currentDeviceID); CeedChk_Hip(ceed,ierr);
  if (currentDeviceID!=deviceID) {
    ierr = hipSetDevice(deviceID); CeedChk_Hip(ceed,ierr);
  }

  struct hipDeviceProp_t deviceProp;
  ierr = hipGetDeviceProperties(&deviceProp, deviceID); CeedChk_Hip(ceed,ierr);

  Ceed_Hip *data;
  ierr = CeedGetData(ceed, (void *)&data); CeedChk(ierr);
  data->deviceId = deviceID;
  data->optblocksize = deviceProp.maxThreadsPerBlock;
  return 0;
}

//------------------------------------------------------------------------------
// Backend Destroy 
//------------------------------------------------------------------------------
int CeedDestroy_Hip(Ceed ceed) {
  int ierr;
  Ceed_Hip *data;
  ierr = CeedGetData(ceed, (void *)&data); CeedChk(ierr);
  ierr = CeedFree(&data); CeedChk(ierr);
  return 0;
}


//------------------------------------------------------------------------------
// Backend Init
//------------------------------------------------------------------------------
static int CeedInit_Hip(const char *resource, Ceed ceed) {
  int ierr;
  const int nrc = 9; // number of characters in resource
  if (strncmp(resource, "/gpu/hip/ref", nrc))
    // LCOV_EXCL_START
    return CeedError(ceed, 1, "Hip backend cannot use resource: %s", resource);
  // LCOV_EXCL_STOP

  Ceed_Hip *data;
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  ierr = CeedSetData(ceed,(void *)&data); CeedChk(ierr);
  ierr = CeedHipInit(ceed, resource, nrc); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "GetPreferredMemType",
                                CeedGetPreferredMemType_Hip); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "VectorCreate",
                                CeedVectorCreate_Hip); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1",
                                CeedBasisCreateTensorH1_Hip); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateH1",
                                CeedBasisCreateH1_Hip); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "ElemRestrictionCreate",
                                CeedElemRestrictionCreate_Hip); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed,
                                "ElemRestrictionCreateBlocked",
                                CeedElemRestrictionCreateBlocked_Hip);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionCreate",
                                CeedQFunctionCreate_Hip); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate",
                                CeedOperatorCreate_Hip); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "CompositeOperatorCreate",
                                CeedCompositeOperatorCreate_Hip); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy",
                                CeedDestroy_Hip); CeedChk(ierr);
  return 0;
}

//------------------------------------------------------------------------------
// Backend Register
//------------------------------------------------------------------------------
__attribute__((constructor))
static void Register(void) {
  CeedRegister("/gpu/hip/ref", CeedInit_Hip, 20);
}
//------------------------------------------------------------------------------
