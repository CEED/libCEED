// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

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

// LCOV_EXCL_START
#define CASE(name, string)    \
  case name: {                \
    return #name ": " string; \
  }
// Helper to get an error string for cublasStatus_t
// Strings from https://docs.nvidia.com/cuda/cublas/index.html#cublasstatus-t
static const char *cublasGetErrorName(cublasStatus_t error) {
  switch (error) {
    CASE(CUBLAS_STATUS_SUCCESS, "The operation completed successfully.")
    CASE(CUBLAS_STATUS_NOT_INITIALIZED, "The cuBLAS library was not initialized.")
    CASE(CUBLAS_STATUS_ALLOC_FAILED, "Resource allocation failed inside the cuBLAS library.")
    CASE(CUBLAS_STATUS_INVALID_VALUE, "An unsupported value or parameter was passed to the function.")
    CASE(CUBLAS_STATUS_ARCH_MISMATCH, "The function requires a feature absent from the device architecture.")
    CASE(CUBLAS_STATUS_MAPPING_ERROR, "An access to GPU memory space failed, which is usually caused by a failure to bind a texture.")
    CASE(CUBLAS_STATUS_EXECUTION_FAILED,
         "The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple "
         "reasons.")
    CASE(CUBLAS_STATUS_INTERNAL_ERROR, "An internal cuBLAS operation failed. This error is usually caused by a cudaMemcpyAsync() failure.")
    CASE(CUBLAS_STATUS_NOT_SUPPORTED, "The functionality requested is not supported.")
    default:
      return "Unknown cublasStatus_t";
  }
}

// Helper to get an error string for cudaGraphExecUpdateResultInfo
// Strings from https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__TYPES.html#_CPPv425cudaGraphExecUpdateResult
static const char *CeedCudaGetGraphExecUpdateResultInfoString(cudaGraphExecUpdateResultInfo result) {
  switch (result.result) {
    CASE(cudaGraphExecUpdateSuccess, "The update succeeded.")
    CASE(cudaGraphExecUpdateError, "The update failed for an unexpected reason which is described in the return value of the function.")
    CASE(cudaGraphExecUpdateErrorTopologyChanged, "The update failed because the topology changed.")
    CASE(cudaGraphExecUpdateErrorNodeTypeChanged, "The update failed because a node type changed.")
    CASE(cudaGraphExecUpdateErrorFunctionChanged, "The update failed because the function of a kernel node changed.")
    CASE(cudaGraphExecUpdateErrorParametersChanged, "The update failed because the parameters changed in a way that is not supported.")
    CASE(cudaGraphExecUpdateErrorNotSupported, "The update failed because something about the node is not supported.")
    CASE(cudaGraphExecUpdateErrorUnsupportedFunctionChange, "The update failed because the function of a kernel node changed in an unsupported way.")
    CASE(cudaGraphExecUpdateErrorAttributesChanged, "The update failed because the node attributes changed in a way that is not supported.")
    default:
      return "Unknown cudaGraphExecUpdateResult";
  }
}

#undef CASE
// LCOV_EXCL_STOP

typedef struct {
  int                   device_id;
  char                 *llvm_cxx;
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
