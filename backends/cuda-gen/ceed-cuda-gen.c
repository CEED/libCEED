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

#include <ceed-backend.h>
#include <string.h>
#include <stdarg.h>
#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "ceed-cuda-gen.h"

static int CeedGetPreferredMemType_Cuda_gen(CeedMemType *type) {
  *type = CEED_MEM_DEVICE;
  return 0;
}

static int CeedInit_Cuda_gen(const char *resource, Ceed ceed) {
  int ierr;
  const int nrc = 9; // number of characters in resource
  if (strncmp(resource, "/gpu/cuda/gen", nrc))
    return CeedError(ceed, 1, "Cuda backend cannot use resource: %s", resource);

  Ceed ceedshared;
  CeedInit("/gpu/cuda/shared", &ceedshared);
  ierr = CeedSetDelegate(ceed, ceedshared); CeedChk(ierr);

  const int rlen = strlen(resource);
  const bool slash = (rlen>nrc) ? (resource[nrc] == '/') : false;
  const int deviceID = (slash && rlen > nrc + 1) ? atoi(&resource[nrc + 1]) : 0;

  ierr = cudaSetDevice(deviceID); CeedChk(ierr);

  Ceed_Cuda_gen *data;
  ierr = CeedCalloc(1,&data); CeedChk(ierr);

  ierr = CeedSetData(ceed,(void *)&data); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "GetPreferredMemType",
                                CeedGetPreferredMemType_Cuda_gen); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionCreate",
                                CeedQFunctionCreate_Cuda_gen); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate",
                                CeedOperatorCreate_Cuda_gen); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "CompositeOperatorCreate",
                                CeedCompositeOperatorCreate_Cuda_gen);
  CeedChk(ierr);
  return 0;
}

__attribute__((constructor))
static void Register(void) {
  CeedRegister("/gpu/cuda/gen", CeedInit_Cuda_gen, 40);
}
