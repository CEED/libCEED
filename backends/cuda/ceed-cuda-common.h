// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_CUDA_COMMON_H
#define CEED_CUDA_COMMON_H

#include <ceed.h>
#include <ceed/backend.h>
#include <cublas_v2.h>
#include <cuda.h>

#define QUOTE(...) #__VA_ARGS__

#define CeedChk_Cu(ceed, x)                              \
  do {                                                   \
    CUresult cuda_result = (CUresult)x;                  \
    if (cuda_result != CUDA_SUCCESS) {                   \
      const char *msg;                                   \
      cuGetErrorName(cuda_result, &msg);                 \
      return CeedError((ceed), CEED_ERROR_BACKEND, msg); \
    }                                                    \
  } while (0)

#define CeedChk_Cublas(ceed, x)                            \
  do {                                                     \
    cublasStatus_t cublas_result = x;                      \
    if (cublas_result != CUBLAS_STATUS_SUCCESS) {          \
      const char *msg = cublasGetErrorName(cublas_result); \
      return CeedError((ceed), CEED_ERROR_BACKEND, msg);   \
    }                                                      \
  } while (0)

#define CeedCallCuda(ceed, ...) \
  do {                          \
    int ierr_q_ = __VA_ARGS__;  \
    CeedChk_Cu(ceed, ierr_q_);  \
  } while (0)

#define CeedCallCublas(ceed, ...)  \
  do {                             \
    int ierr_q_ = __VA_ARGS__;     \
    CeedChk_Cublas(ceed, ierr_q_); \
  } while (0)

#define CASE(name) \
  case name:       \
    return #name
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
    default:
      return "CUBLAS_STATUS_UNKNOWN_ERROR";
  }
}
// LCOV_EXCL_STOP

typedef struct {
  int                   device_id;
  cublasHandle_t        cublas_handle;
  struct cudaDeviceProp device_prop;
} Ceed_Cuda;

CEED_INTERN int CeedInit_Cuda(Ceed ceed, const char *resource);

CEED_INTERN int CeedDestroy_Cuda(Ceed ceed);

CEED_INTERN int CeedSetDeviceBoolArray_Cuda(Ceed ceed, const bool *source_array, CeedCopyMode copy_mode, CeedSize num_values,
                                            const bool **target_array_owned, const bool **target_array_borrowed, const bool **target_array);
CEED_INTERN int CeedSetDeviceCeedInt8Array_Cuda(Ceed ceed, const CeedInt8 *source_array, CeedCopyMode copy_mode, CeedSize num_values,
                                                const CeedInt8 **target_array_owned, const CeedInt8 **target_array_borrowed,
                                                const CeedInt8 **target_array);
CEED_INTERN int CeedSetDeviceCeedIntArray_Cuda(Ceed ceed, const CeedInt *source_array, CeedCopyMode copy_mode, CeedSize num_values,
                                               const CeedInt **target_array_owned, const CeedInt **target_array_borrowed,
                                               const CeedInt **target_array);
CEED_INTERN int CeedSetDeviceCeedScalarArray_Cuda(Ceed ceed, const CeedScalar *source_array, CeedCopyMode copy_mode, CeedSize num_values,
                                                  const CeedScalar **target_array_owned, const CeedScalar **target_array_borrowed,
                                                  const CeedScalar **target_array);

#endif  // CEED_CUDA_COMMON_H
