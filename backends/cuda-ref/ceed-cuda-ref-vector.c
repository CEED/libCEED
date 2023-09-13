// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "../cuda/ceed-cuda-common.h"
#include "ceed-cuda-ref.h"

//------------------------------------------------------------------------------
// Check if host/device sync is needed
//------------------------------------------------------------------------------
static inline int CeedVectorNeedSync_Cuda(const CeedVector vec, CeedMemType mem_type, bool *need_sync) {
  bool             has_valid_array = false;
  CeedVector_Cuda *impl;

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedVectorHasValidArray(vec, &has_valid_array));
  switch (mem_type) {
    case CEED_MEM_HOST:
      *need_sync = has_valid_array && !impl->h_array;
      break;
    case CEED_MEM_DEVICE:
      *need_sync = has_valid_array && !impl->d_array;
      break;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync host to device
//------------------------------------------------------------------------------
static inline int CeedVectorSyncH2D_Cuda(const CeedVector vec) {
  Ceed             ceed;
  CeedSize         length;
  CeedVector_Cuda *impl;

  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  CeedCheck(impl->h_array, ceed, CEED_ERROR_BACKEND, "No valid host data to sync to device");

  CeedCallBackend(CeedVectorGetLength(vec, &length));
  size_t bytes = length * sizeof(CeedScalar);

  if (impl->d_array_borrowed) {
    impl->d_array = impl->d_array_borrowed;
  } else if (impl->d_array_owned) {
    impl->d_array = impl->d_array_owned;
  } else {
    CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_array_owned, bytes));
    impl->d_array = impl->d_array_owned;
  }
  CeedCallCuda(ceed, cudaMemcpy(impl->d_array, impl->h_array, bytes, cudaMemcpyHostToDevice));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync device to host
//------------------------------------------------------------------------------
static inline int CeedVectorSyncD2H_Cuda(const CeedVector vec) {
  Ceed             ceed;
  CeedSize         length;
  CeedVector_Cuda *impl;

  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  CeedCheck(impl->d_array, ceed, CEED_ERROR_BACKEND, "No valid device data to sync to host");

  if (impl->h_array_borrowed) {
    impl->h_array = impl->h_array_borrowed;
  } else if (impl->h_array_owned) {
    impl->h_array = impl->h_array_owned;
  } else {
    CeedSize length;

    CeedCallBackend(CeedVectorGetLength(vec, &length));
    CeedCallBackend(CeedCalloc(length, &impl->h_array_owned));
    impl->h_array = impl->h_array_owned;
  }

  CeedCallBackend(CeedVectorGetLength(vec, &length));
  size_t bytes = length * sizeof(CeedScalar);

  CeedCallCuda(ceed, cudaMemcpy(impl->h_array, impl->d_array, bytes, cudaMemcpyDeviceToHost));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync arrays
//------------------------------------------------------------------------------
static int CeedVectorSyncArray_Cuda(const CeedVector vec, CeedMemType mem_type) {
  bool need_sync = false;

  // Check whether device/host sync is needed
  CeedCallBackend(CeedVectorNeedSync_Cuda(vec, mem_type, &need_sync));
  if (!need_sync) return CEED_ERROR_SUCCESS;

  switch (mem_type) {
    case CEED_MEM_HOST:
      return CeedVectorSyncD2H_Cuda(vec);
    case CEED_MEM_DEVICE:
      return CeedVectorSyncH2D_Cuda(vec);
  }
  return CEED_ERROR_UNSUPPORTED;
}

//------------------------------------------------------------------------------
// Set all pointers as invalid
//------------------------------------------------------------------------------
static inline int CeedVectorSetAllInvalid_Cuda(const CeedVector vec) {
  CeedVector_Cuda *impl;

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  impl->h_array = NULL;
  impl->d_array = NULL;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if CeedVector has any valid pointer
//------------------------------------------------------------------------------
static inline int CeedVectorHasValidArray_Cuda(const CeedVector vec, bool *has_valid_array) {
  CeedVector_Cuda *impl;

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  *has_valid_array = impl->h_array || impl->d_array;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if has array of given type
//------------------------------------------------------------------------------
static inline int CeedVectorHasArrayOfType_Cuda(const CeedVector vec, CeedMemType mem_type, bool *has_array_of_type) {
  CeedVector_Cuda *impl;

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  switch (mem_type) {
    case CEED_MEM_HOST:
      *has_array_of_type = impl->h_array_borrowed || impl->h_array_owned;
      break;
    case CEED_MEM_DEVICE:
      *has_array_of_type = impl->d_array_borrowed || impl->d_array_owned;
      break;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if has borrowed array of given type
//------------------------------------------------------------------------------
static inline int CeedVectorHasBorrowedArrayOfType_Cuda(const CeedVector vec, CeedMemType mem_type, bool *has_borrowed_array_of_type) {
  CeedVector_Cuda *impl;

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  switch (mem_type) {
    case CEED_MEM_HOST:
      *has_borrowed_array_of_type = impl->h_array_borrowed;
      break;
    case CEED_MEM_DEVICE:
      *has_borrowed_array_of_type = impl->d_array_borrowed;
      break;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set array from host
//------------------------------------------------------------------------------
static int CeedVectorSetArrayHost_Cuda(const CeedVector vec, const CeedCopyMode copy_mode, CeedScalar *array) {
  CeedVector_Cuda *impl;

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  switch (copy_mode) {
    case CEED_COPY_VALUES: {
      if (!impl->h_array_owned) {
        CeedSize length;

        CeedCallBackend(CeedVectorGetLength(vec, &length));
        CeedCallBackend(CeedMalloc(length, &impl->h_array_owned));
      }
      impl->h_array_borrowed = NULL;
      impl->h_array          = impl->h_array_owned;
      if (array) {
        CeedSize length;

        CeedCallBackend(CeedVectorGetLength(vec, &length));
        size_t bytes = length * sizeof(CeedScalar);
        memcpy(impl->h_array, array, bytes);
      }
    } break;
    case CEED_OWN_POINTER:
      CeedCallBackend(CeedFree(&impl->h_array_owned));
      impl->h_array_owned    = array;
      impl->h_array_borrowed = NULL;
      impl->h_array          = array;
      break;
    case CEED_USE_POINTER:
      CeedCallBackend(CeedFree(&impl->h_array_owned));
      impl->h_array_borrowed = array;
      impl->h_array          = array;
      break;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set array from device
//------------------------------------------------------------------------------
static int CeedVectorSetArrayDevice_Cuda(const CeedVector vec, const CeedCopyMode copy_mode, CeedScalar *array) {
  Ceed             ceed;
  CeedVector_Cuda *impl;

  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  switch (copy_mode) {
    case CEED_COPY_VALUES: {
      CeedSize length;

      CeedCallBackend(CeedVectorGetLength(vec, &length));
      size_t bytes = length * sizeof(CeedScalar);

      if (!impl->d_array_owned) {
        CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_array_owned, bytes));
      }
      impl->d_array_borrowed = NULL;
      impl->d_array          = impl->d_array_owned;
      if (array) CeedCallCuda(ceed, cudaMemcpy(impl->d_array, array, bytes, cudaMemcpyDeviceToDevice));
    } break;
    case CEED_OWN_POINTER:
      CeedCallCuda(ceed, cudaFree(impl->d_array_owned));
      impl->d_array_owned    = array;
      impl->d_array_borrowed = NULL;
      impl->d_array          = array;
      break;
    case CEED_USE_POINTER:
      CeedCallCuda(ceed, cudaFree(impl->d_array_owned));
      impl->d_array_owned    = NULL;
      impl->d_array_borrowed = array;
      impl->d_array          = array;
      break;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set the array used by a vector,
//   freeing any previously allocated array if applicable
//------------------------------------------------------------------------------
static int CeedVectorSetArray_Cuda(const CeedVector vec, const CeedMemType mem_type, const CeedCopyMode copy_mode, CeedScalar *array) {
  Ceed             ceed;
  CeedVector_Cuda *impl;

  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedVectorSetAllInvalid_Cuda(vec));
  switch (mem_type) {
    case CEED_MEM_HOST:
      return CeedVectorSetArrayHost_Cuda(vec, copy_mode, array);
    case CEED_MEM_DEVICE:
      return CeedVectorSetArrayDevice_Cuda(vec, copy_mode, array);
  }
  return CEED_ERROR_UNSUPPORTED;
}

//------------------------------------------------------------------------------
// Set host array to value
//------------------------------------------------------------------------------
static int CeedHostSetValue_Cuda(CeedScalar *h_array, CeedSize length, CeedScalar val) {
  for (CeedSize i = 0; i < length; i++) h_array[i] = val;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set device array to value (impl in .cu file)
//------------------------------------------------------------------------------
int CeedDeviceSetValue_Cuda(CeedScalar *d_array, CeedSize length, CeedScalar val);

//------------------------------------------------------------------------------
// Set a vector to a value
//------------------------------------------------------------------------------
static int CeedVectorSetValue_Cuda(CeedVector vec, CeedScalar val) {
  Ceed             ceed;
  CeedSize         length;
  CeedVector_Cuda *impl;

  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedVectorGetLength(vec, &length));
  // Set value for synced device/host array
  if (!impl->d_array && !impl->h_array) {
    if (impl->d_array_borrowed) {
      impl->d_array = impl->d_array_borrowed;
    } else if (impl->h_array_borrowed) {
      impl->h_array = impl->h_array_borrowed;
    } else if (impl->d_array_owned) {
      impl->d_array = impl->d_array_owned;
    } else if (impl->h_array_owned) {
      impl->h_array = impl->h_array_owned;
    } else {
      CeedCallBackend(CeedVectorSetArray(vec, CEED_MEM_DEVICE, CEED_COPY_VALUES, NULL));
    }
  }
  if (impl->d_array) {
    CeedCallBackend(CeedDeviceSetValue_Cuda(impl->d_array, length, val));
    impl->h_array = NULL;
  }
  if (impl->h_array) {
    CeedCallBackend(CeedHostSetValue_Cuda(impl->h_array, length, val));
    impl->d_array = NULL;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Take Array
//------------------------------------------------------------------------------
static int CeedVectorTakeArray_Cuda(CeedVector vec, CeedMemType mem_type, CeedScalar **array) {
  Ceed             ceed;
  CeedVector_Cuda *impl;

  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  // Sync array to requested mem_type
  CeedCallBackend(CeedVectorSyncArray(vec, mem_type));
  // Update pointer
  switch (mem_type) {
    case CEED_MEM_HOST:
      (*array)               = impl->h_array_borrowed;
      impl->h_array_borrowed = NULL;
      impl->h_array          = NULL;
      break;
    case CEED_MEM_DEVICE:
      (*array)               = impl->d_array_borrowed;
      impl->d_array_borrowed = NULL;
      impl->d_array          = NULL;
      break;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Core logic for array syncronization for GetArray.
//   If a different memory type is most up to date, this will perform a copy
//------------------------------------------------------------------------------
static int CeedVectorGetArrayCore_Cuda(const CeedVector vec, const CeedMemType mem_type, CeedScalar **array) {
  Ceed             ceed;
  CeedVector_Cuda *impl;

  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  // Sync array to requested mem_type
  CeedCallBackend(CeedVectorSyncArray(vec, mem_type));
  // Update pointer
  switch (mem_type) {
    case CEED_MEM_HOST:
      *array = impl->h_array;
      break;
    case CEED_MEM_DEVICE:
      *array = impl->d_array;
      break;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get read-only access to a vector via the specified mem_type
//------------------------------------------------------------------------------
static int CeedVectorGetArrayRead_Cuda(const CeedVector vec, const CeedMemType mem_type, const CeedScalar **array) {
  return CeedVectorGetArrayCore_Cuda(vec, mem_type, (CeedScalar **)array);
}

//------------------------------------------------------------------------------
// Get read/write access to a vector via the specified mem_type
//------------------------------------------------------------------------------
static int CeedVectorGetArray_Cuda(const CeedVector vec, const CeedMemType mem_type, CeedScalar **array) {
  CeedVector_Cuda *impl;

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedVectorGetArrayCore_Cuda(vec, mem_type, array));
  CeedCallBackend(CeedVectorSetAllInvalid_Cuda(vec));
  switch (mem_type) {
    case CEED_MEM_HOST:
      impl->h_array = *array;
      break;
    case CEED_MEM_DEVICE:
      impl->d_array = *array;
      break;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get write access to a vector via the specified mem_type
//------------------------------------------------------------------------------
static int CeedVectorGetArrayWrite_Cuda(const CeedVector vec, const CeedMemType mem_type, CeedScalar **array) {
  bool             has_array_of_type = true;
  CeedVector_Cuda *impl;

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedVectorHasArrayOfType_Cuda(vec, mem_type, &has_array_of_type));
  if (!has_array_of_type) {
    // Allocate if array is not yet allocated
    CeedCallBackend(CeedVectorSetArray(vec, mem_type, CEED_COPY_VALUES, NULL));
  } else {
    // Select dirty array
    switch (mem_type) {
      case CEED_MEM_HOST:
        if (impl->h_array_borrowed) impl->h_array = impl->h_array_borrowed;
        else impl->h_array = impl->h_array_owned;
        break;
      case CEED_MEM_DEVICE:
        if (impl->d_array_borrowed) impl->d_array = impl->d_array_borrowed;
        else impl->d_array = impl->d_array_owned;
    }
  }
  return CeedVectorGetArray_Cuda(vec, mem_type, array);
}

//------------------------------------------------------------------------------
// Get the norm of a CeedVector
//------------------------------------------------------------------------------
static int CeedVectorNorm_Cuda(CeedVector vec, CeedNormType type, CeedScalar *norm) {
  Ceed              ceed;
  cublasHandle_t    handle;
  CeedSize          length;
  const CeedScalar *d_array;
  CeedVector_Cuda  *impl;

  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedVectorGetLength(vec, &length));
  CeedCallBackend(CeedGetCublasHandle_Cuda(ceed, &handle));

#if CUDA_VERSION < 12000
  // With CUDA 12, we can use the 64-bit integer interface. Prior to that,
  // we need to check if the vector is too long to handle with int32,
  // and if so, divide it into subsections for repeated cuBLAS calls.
  CeedSize num_calls = length / INT_MAX;

  if (length % INT_MAX > 0) num_calls += 1;
#endif

  // Compute norm
  CeedCallBackend(CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE, &d_array));
  switch (type) {
    case CEED_NORM_1: {
      *norm = 0.0;
      if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
#if CUDA_VERSION >= 12000  // We have CUDA 12, and can use 64-bit integers
        CeedCallCublas(ceed, cublasSasum_64(handle, (int64_t)length, (float *)d_array, 1, (float *)norm));
#else
        float  sub_norm = 0.0;
        float *d_array_start;

        for (CeedInt i = 0; i < num_calls; i++) {
          d_array_start             = (float *)d_array + (CeedSize)(i)*INT_MAX;
          CeedSize remaining_length = length - (CeedSize)(i)*INT_MAX;
          CeedInt  sub_length       = (i == num_calls - 1) ? (CeedInt)(remaining_length) : INT_MAX;

          CeedCallCublas(ceed, cublasSasum(handle, (CeedInt)sub_length, (float *)d_array_start, 1, &sub_norm));
          *norm += sub_norm;
        }
#endif
      } else {
#if CUDA_VERSION >= 12000
        CeedCallCublas(ceed, cublasDasum_64(handle, (int64_t)length, (double *)d_array, 1, (double *)norm));
#else
        double  sub_norm = 0.0;
        double *d_array_start;

        for (CeedInt i = 0; i < num_calls; i++) {
          d_array_start             = (double *)d_array + (CeedSize)(i)*INT_MAX;
          CeedSize remaining_length = length - (CeedSize)(i)*INT_MAX;
          CeedInt  sub_length       = (i == num_calls - 1) ? (CeedInt)(remaining_length) : INT_MAX;

          CeedCallCublas(ceed, cublasDasum(handle, (CeedInt)sub_length, (double *)d_array_start, 1, &sub_norm));
          *norm += sub_norm;
        }
#endif
      }
      break;
    }
    case CEED_NORM_2: {
      if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
#if CUDA_VERSION >= 12000
        CeedCallCublas(ceed, cublasSnrm2_64(handle, (int64_t)length, (float *)d_array, 1, (float *)norm));
#else
        float  sub_norm = 0.0, norm_sum = 0.0;
        float *d_array_start;

        for (CeedInt i = 0; i < num_calls; i++) {
          d_array_start             = (float *)d_array + (CeedSize)(i)*INT_MAX;
          CeedSize remaining_length = length - (CeedSize)(i)*INT_MAX;
          CeedInt  sub_length       = (i == num_calls - 1) ? (CeedInt)(remaining_length) : INT_MAX;

          CeedCallCublas(ceed, cublasSnrm2(handle, (CeedInt)sub_length, (float *)d_array_start, 1, &sub_norm));
          norm_sum += sub_norm * sub_norm;
        }
        *norm            = sqrt(norm_sum);
#endif
      } else {
#if CUDA_VERSION >= 12000
        CeedCallCublas(ceed, cublasDnrm2_64(handle, (int64_t)length, (double *)d_array, 1, (double *)norm));
#else
        double  sub_norm = 0.0, norm_sum = 0.0;
        double *d_array_start;

        for (CeedInt i = 0; i < num_calls; i++) {
          d_array_start             = (double *)d_array + (CeedSize)(i)*INT_MAX;
          CeedSize remaining_length = length - (CeedSize)(i)*INT_MAX;
          CeedInt  sub_length       = (i == num_calls - 1) ? (CeedInt)(remaining_length) : INT_MAX;

          CeedCallCublas(ceed, cublasDnrm2(handle, (CeedInt)sub_length, (double *)d_array_start, 1, &sub_norm));
          norm_sum += sub_norm * sub_norm;
        }
        *norm = sqrt(norm_sum);
#endif
      }
      break;
    }
    case CEED_NORM_MAX: {
      if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
#if CUDA_VERSION >= 12000
        int64_t    index;
        CeedScalar norm_no_abs;

        CeedCallCublas(ceed, cublasIsamax_64(handle, (int64_t)length, (float *)d_array, 1, &index));
        CeedCallCuda(ceed, cudaMemcpy(&norm_no_abs, impl->d_array + index - 1, sizeof(CeedScalar), cudaMemcpyDeviceToHost));
        *norm = fabs(norm_no_abs);
#else
        CeedInt index;
        float   sub_max = 0.0, current_max = 0.0;
        float  *d_array_start;

        for (CeedInt i = 0; i < num_calls; i++) {
          d_array_start             = (float *)d_array + (CeedSize)(i)*INT_MAX;
          CeedSize remaining_length = length - (CeedSize)(i)*INT_MAX;
          CeedInt  sub_length       = (i == num_calls - 1) ? (CeedInt)(remaining_length) : INT_MAX;

          CeedCallCublas(ceed, cublasIsamax(handle, (CeedInt)sub_length, (float *)d_array_start, 1, &index));
          CeedCallCuda(ceed, cudaMemcpy(&sub_max, d_array_start + index - 1, sizeof(CeedScalar), cudaMemcpyDeviceToHost));
          if (fabs(sub_max) > current_max) current_max = fabs(sub_max);
        }
        *norm = current_max;
#endif
      } else {
#if CUDA_VERSION >= 12000
        int64_t    index;
        CeedScalar norm_no_abs;

        CeedCallCublas(ceed, cublasIdamax_64(handle, (int64_t)length, (double *)d_array, 1, &index));
        CeedCallCuda(ceed, cudaMemcpy(&norm_no_abs, impl->d_array + index - 1, sizeof(CeedScalar), cudaMemcpyDeviceToHost));
        *norm = fabs(norm_no_abs);
#else
        CeedInt index;
        double  sub_max = 0.0, current_max = 0.0;
        double *d_array_start;

        for (CeedInt i = 0; i < num_calls; i++) {
          d_array_start             = (double *)d_array + (CeedSize)(i)*INT_MAX;
          CeedSize remaining_length = length - (CeedSize)(i)*INT_MAX;
          CeedInt  sub_length       = (i == num_calls - 1) ? (CeedInt)(remaining_length) : INT_MAX;

          CeedCallCublas(ceed, cublasIdamax(handle, (CeedInt)sub_length, (double *)d_array_start, 1, &index));
          CeedCallCuda(ceed, cudaMemcpy(&sub_max, d_array_start + index - 1, sizeof(CeedScalar), cudaMemcpyDeviceToHost));
          if (fabs(sub_max) > current_max) current_max = fabs(sub_max);
        }
        *norm = current_max;
#endif
      }
      break;
    }
  }
  CeedCallBackend(CeedVectorRestoreArrayRead(vec, &d_array));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Take reciprocal of a vector on host
//------------------------------------------------------------------------------
static int CeedHostReciprocal_Cuda(CeedScalar *h_array, CeedSize length) {
  for (CeedSize i = 0; i < length; i++) {
    if (fabs(h_array[i]) > CEED_EPSILON) h_array[i] = 1. / h_array[i];
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Take reciprocal of a vector on device (impl in .cu file)
//------------------------------------------------------------------------------
int CeedDeviceReciprocal_Cuda(CeedScalar *d_array, CeedSize length);

//------------------------------------------------------------------------------
// Take reciprocal of a vector
//------------------------------------------------------------------------------
static int CeedVectorReciprocal_Cuda(CeedVector vec) {
  Ceed             ceed;
  CeedSize         length;
  CeedVector_Cuda *impl;

  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedVectorGetLength(vec, &length));
  // Set value for synced device/host array
  if (impl->d_array) CeedCallBackend(CeedDeviceReciprocal_Cuda(impl->d_array, length));
  if (impl->h_array) CeedCallBackend(CeedHostReciprocal_Cuda(impl->h_array, length));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute x = alpha x on the host
//------------------------------------------------------------------------------
static int CeedHostScale_Cuda(CeedScalar *x_array, CeedScalar alpha, CeedSize length) {
  for (CeedSize i = 0; i < length; i++) x_array[i] *= alpha;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute x = alpha x on device (impl in .cu file)
//------------------------------------------------------------------------------
int CeedDeviceScale_Cuda(CeedScalar *x_array, CeedScalar alpha, CeedSize length);

//------------------------------------------------------------------------------
// Compute x = alpha x
//------------------------------------------------------------------------------
static int CeedVectorScale_Cuda(CeedVector x, CeedScalar alpha) {
  Ceed             ceed;
  CeedSize         length;
  CeedVector_Cuda *x_impl;

  CeedCallBackend(CeedVectorGetCeed(x, &ceed));
  CeedCallBackend(CeedVectorGetData(x, &x_impl));
  CeedCallBackend(CeedVectorGetLength(x, &length));
  // Set value for synced device/host array
  if (x_impl->d_array) CeedCallBackend(CeedDeviceScale_Cuda(x_impl->d_array, alpha, length));
  if (x_impl->h_array) CeedCallBackend(CeedHostScale_Cuda(x_impl->h_array, alpha, length));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute y = alpha x + y on the host
//------------------------------------------------------------------------------
static int CeedHostAXPY_Cuda(CeedScalar *y_array, CeedScalar alpha, CeedScalar *x_array, CeedSize length) {
  for (CeedSize i = 0; i < length; i++) y_array[i] += alpha * x_array[i];
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute y = alpha x + y on device (impl in .cu file)
//------------------------------------------------------------------------------
int CeedDeviceAXPY_Cuda(CeedScalar *y_array, CeedScalar alpha, CeedScalar *x_array, CeedSize length);

//------------------------------------------------------------------------------
// Compute y = alpha x + y
//------------------------------------------------------------------------------
static int CeedVectorAXPY_Cuda(CeedVector y, CeedScalar alpha, CeedVector x) {
  Ceed             ceed;
  CeedSize         length;
  CeedVector_Cuda *y_impl, *x_impl;

  CeedCallBackend(CeedVectorGetCeed(y, &ceed));
  CeedCallBackend(CeedVectorGetData(y, &y_impl));
  CeedCallBackend(CeedVectorGetData(x, &x_impl));
  CeedCallBackend(CeedVectorGetLength(y, &length));
  // Set value for synced device/host array
  if (y_impl->d_array) {
    CeedCallBackend(CeedVectorSyncArray(x, CEED_MEM_DEVICE));
    CeedCallBackend(CeedDeviceAXPY_Cuda(y_impl->d_array, alpha, x_impl->d_array, length));
  }
  if (y_impl->h_array) {
    CeedCallBackend(CeedVectorSyncArray(x, CEED_MEM_HOST));
    CeedCallBackend(CeedHostAXPY_Cuda(y_impl->h_array, alpha, x_impl->h_array, length));
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute y = alpha x + beta y on the host
//------------------------------------------------------------------------------
static int CeedHostAXPBY_Cuda(CeedScalar *y_array, CeedScalar alpha, CeedScalar beta, CeedScalar *x_array, CeedSize length) {
  for (CeedSize i = 0; i < length; i++) y_array[i] += alpha * x_array[i] + beta * y_array[i];
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute y = alpha x + beta y on device (impl in .cu file)
//------------------------------------------------------------------------------
int CeedDeviceAXPBY_Cuda(CeedScalar *y_array, CeedScalar alpha, CeedScalar beta, CeedScalar *x_array, CeedSize length);

//------------------------------------------------------------------------------
// Compute y = alpha x + beta y
//------------------------------------------------------------------------------
static int CeedVectorAXPBY_Cuda(CeedVector y, CeedScalar alpha, CeedScalar beta, CeedVector x) {
  Ceed             ceed;
  CeedSize         length;
  CeedVector_Cuda *y_impl, *x_impl;

  CeedCallBackend(CeedVectorGetCeed(y, &ceed));
  CeedCallBackend(CeedVectorGetData(y, &y_impl));
  CeedCallBackend(CeedVectorGetData(x, &x_impl));
  CeedCallBackend(CeedVectorGetLength(y, &length));
  // Set value for synced device/host array
  if (y_impl->d_array) {
    CeedCallBackend(CeedVectorSyncArray(x, CEED_MEM_DEVICE));
    CeedCallBackend(CeedDeviceAXPBY_Cuda(y_impl->d_array, alpha, beta, x_impl->d_array, length));
  }
  if (y_impl->h_array) {
    CeedCallBackend(CeedVectorSyncArray(x, CEED_MEM_HOST));
    CeedCallBackend(CeedHostAXPBY_Cuda(y_impl->h_array, alpha, beta, x_impl->h_array, length));
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute the pointwise multiplication w = x .* y on the host
//------------------------------------------------------------------------------
static int CeedHostPointwiseMult_Cuda(CeedScalar *w_array, CeedScalar *x_array, CeedScalar *y_array, CeedSize length) {
  for (CeedSize i = 0; i < length; i++) w_array[i] = x_array[i] * y_array[i];
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute the pointwise multiplication w = x .* y on device (impl in .cu file)
//------------------------------------------------------------------------------
int CeedDevicePointwiseMult_Cuda(CeedScalar *w_array, CeedScalar *x_array, CeedScalar *y_array, CeedSize length);

//------------------------------------------------------------------------------
// Compute the pointwise multiplication w = x .* y
//------------------------------------------------------------------------------
static int CeedVectorPointwiseMult_Cuda(CeedVector w, CeedVector x, CeedVector y) {
  Ceed             ceed;
  CeedSize         length;
  CeedVector_Cuda *w_impl, *x_impl, *y_impl;

  CeedCallBackend(CeedVectorGetCeed(w, &ceed));
  CeedCallBackend(CeedVectorGetData(w, &w_impl));
  CeedCallBackend(CeedVectorGetData(x, &x_impl));
  CeedCallBackend(CeedVectorGetData(y, &y_impl));
  CeedCallBackend(CeedVectorGetLength(w, &length));
  // Set value for synced device/host array
  if (!w_impl->d_array && !w_impl->h_array) {
    CeedCallBackend(CeedVectorSetValue(w, 0.0));
  }
  if (w_impl->d_array) {
    CeedCallBackend(CeedVectorSyncArray(x, CEED_MEM_DEVICE));
    CeedCallBackend(CeedVectorSyncArray(y, CEED_MEM_DEVICE));
    CeedCallBackend(CeedDevicePointwiseMult_Cuda(w_impl->d_array, x_impl->d_array, y_impl->d_array, length));
  }
  if (w_impl->h_array) {
    CeedCallBackend(CeedVectorSyncArray(x, CEED_MEM_HOST));
    CeedCallBackend(CeedVectorSyncArray(y, CEED_MEM_HOST));
    CeedCallBackend(CeedHostPointwiseMult_Cuda(w_impl->h_array, x_impl->h_array, y_impl->h_array, length));
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy the vector
//------------------------------------------------------------------------------
static int CeedVectorDestroy_Cuda(const CeedVector vec) {
  Ceed             ceed;
  CeedVector_Cuda *impl;

  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallCuda(ceed, cudaFree(impl->d_array_owned));
  CeedCallBackend(CeedFree(&impl->h_array_owned));
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create a vector of the specified length (does not allocate memory)
//------------------------------------------------------------------------------
int CeedVectorCreate_Cuda(CeedSize n, CeedVector vec) {
  CeedVector_Cuda *impl;
  Ceed             ceed;

  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "HasValidArray", CeedVectorHasValidArray_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "HasBorrowedArrayOfType", CeedVectorHasBorrowedArrayOfType_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "SetArray", CeedVectorSetArray_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "TakeArray", CeedVectorTakeArray_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "SetValue", (int (*)())CeedVectorSetValue_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "SyncArray", CeedVectorSyncArray_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "GetArray", CeedVectorGetArray_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayRead", CeedVectorGetArrayRead_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayWrite", CeedVectorGetArrayWrite_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "Norm", CeedVectorNorm_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "Reciprocal", CeedVectorReciprocal_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "Scale", (int (*)())CeedVectorScale_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "AXPY", (int (*)())CeedVectorAXPY_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "AXPBY", (int (*)())CeedVectorAXPBY_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "PointwiseMult", CeedVectorPointwiseMult_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "Destroy", CeedVectorDestroy_Cuda));
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedVectorSetData(vec, impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
