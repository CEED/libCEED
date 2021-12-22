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

#ifndef _ceed_common_hip_h
#define _ceed_common_hip_h

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <hip/hip_runtime.h>
#include <hipblas.h>

#define QUOTE(...) #__VA_ARGS__

#define CeedChk_Hip(ceed, x) \
do { \
  hipError_t hip_result = x; \
  if (hip_result != hipSuccess) { \
    const char *msg = hipGetErrorName(hip_result); \
    return CeedError((ceed), CEED_ERROR_BACKEND, msg); \
  } \
} while (0)

#define CeedChk_Hipblas(ceed, x) \
do { \
  hipblasStatus_t hipblas_result = x; \
  if (hipblas_result != HIPBLAS_STATUS_SUCCESS) { \
    const char *msg = hipblasGetErrorName(hipblas_result); \
    return CeedError((ceed), CEED_ERROR_BACKEND, msg); \
   } \
} while (0)

#define CASE(name) case name: return #name
// LCOV_EXCL_START
CEED_UNUSED static const char *hipblasGetErrorName(hipblasStatus_t error) {
  switch (error) {
    CASE(HIPBLAS_STATUS_SUCCESS);
    CASE(HIPBLAS_STATUS_NOT_INITIALIZED);
    CASE(HIPBLAS_STATUS_ALLOC_FAILED);
    CASE(HIPBLAS_STATUS_INVALID_VALUE);
    CASE(HIPBLAS_STATUS_ARCH_MISMATCH);
    CASE(HIPBLAS_STATUS_MAPPING_ERROR);
    CASE(HIPBLAS_STATUS_EXECUTION_FAILED);
    CASE(HIPBLAS_STATUS_INTERNAL_ERROR);
  default: return "HIPBLAS_STATUS_UNKNOWN_ERROR";
  }
}
// LCOV_EXCL_STOP

typedef struct {
  int optblocksize;
  int deviceId;
  hipblasHandle_t hipblasHandle;
} Ceed_Hip;

CEED_INTERN int CeedHipInit(Ceed ceed, const char *resource, int nrc);

CEED_INTERN int CeedDestroy_Hip(Ceed ceed);

#endif // _ceed_hip_common_h
