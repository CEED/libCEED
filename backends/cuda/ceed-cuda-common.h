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

#ifndef _ceed_cuda_common_h
#define _ceed_cuda_common_h

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define QUOTE(...) #__VA_ARGS__

#define CeedChk_Cu(ceed, x) \
do { \
  CUresult cuda_result = x; \
  if (cuda_result != CUDA_SUCCESS) { \
    const char *msg; \
    cuGetErrorName(cuda_result, &msg); \
    return CeedError((ceed), CEED_ERROR_BACKEND, msg); \
  } \
} while (0)

#define CeedChk_Cublas(ceed, x) \
do { \
  cublasStatus_t cublas_result = x; \
  if (cublas_result != CUBLAS_STATUS_SUCCESS) { \
    const char *msg = cublasGetErrorName(cublas_result); \
    return CeedError((ceed), CEED_ERROR_BACKEND, msg); \
   } \
} while (0)

#define CASE(name) case name: return #name
// LCOV_EXCL_START
static const char *cublasGetErrorName(cublasStatus_t error) {
  switch (error) {
    CASE(CUBLAS_STATUS_SUCCESS);
    CASE(CUBLAS_STATUS_NOT_INITIALIZED);
    CASE(CUBLAS_STATUS_ALLOC_FAILED);
    CASE(CUBLAS_STATUS_INVALID_VALUE);
    CASE(CUBLAS_STATUS_ARCH_MISMATCH);
    CASE(CUBLAS_STATUS_MAPPING_ERROR);
    CASE(CUBLAS_STATUS_EXECUTION_FAILED);
    CASE(CUBLAS_STATUS_INTERNAL_ERROR);
  default: return "CUBLAS_STATUS_UNKNOWN_ERROR";
  }
}
// LCOV_EXCL_STOP

typedef struct {
  int deviceId;
  cublasHandle_t cublasHandle;
  struct cudaDeviceProp deviceProp;
} Ceed_Cuda;

CEED_INTERN int CeedCudaInit(Ceed ceed, const char *resource, int nrc);

CEED_INTERN int CeedDestroy_Cuda(Ceed ceed);

#endif // _ceed_cuda_common_h