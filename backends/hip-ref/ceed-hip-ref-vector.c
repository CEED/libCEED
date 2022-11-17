// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <string.h>

#include "ceed-hip-ref.h"

//------------------------------------------------------------------------------
// Check if host/device sync is needed
//------------------------------------------------------------------------------
static inline int CeedVectorNeedSync_Hip(const CeedVector vec, CeedMemType mem_type, bool *need_sync) {
  CeedVector_Hip *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  bool has_valid_array = false;
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
static inline int CeedVectorSyncH2D_Hip(const CeedVector vec) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedVector_Hip *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  CeedSize length;
  CeedCallBackend(CeedVectorGetLength(vec, &length));
  size_t bytes = length * sizeof(CeedScalar);

  if (!impl->h_array) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "No valid host data to sync to device");
    // LCOV_EXCL_STOP
  }

  if (impl->d_array_borrowed) {
    impl->d_array = impl->d_array_borrowed;
  } else if (impl->d_array_owned) {
    impl->d_array = impl->d_array_owned;
  } else {
    CeedCallHip(ceed, hipMalloc((void **)&impl->d_array_owned, bytes));
    impl->d_array = impl->d_array_owned;
  }

  CeedCallHip(ceed, hipMemcpy(impl->d_array, impl->h_array, bytes, hipMemcpyHostToDevice));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync device to host
//------------------------------------------------------------------------------
static inline int CeedVectorSyncD2H_Hip(const CeedVector vec) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedVector_Hip *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  if (!impl->d_array) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "No valid device data to sync to host");
    // LCOV_EXCL_STOP
  }

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

  CeedSize length;
  CeedCallBackend(CeedVectorGetLength(vec, &length));
  size_t bytes = length * sizeof(CeedScalar);
  CeedCallHip(ceed, hipMemcpy(impl->h_array, impl->d_array, bytes, hipMemcpyDeviceToHost));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync arrays
//------------------------------------------------------------------------------
static int CeedVectorSyncArray_Hip(const CeedVector vec, CeedMemType mem_type) {
  // Check whether device/host sync is needed
  bool need_sync = false;
  CeedCallBackend(CeedVectorNeedSync_Hip(vec, mem_type, &need_sync));
  if (!need_sync) return CEED_ERROR_SUCCESS;

  switch (mem_type) {
    case CEED_MEM_HOST:
      return CeedVectorSyncD2H_Hip(vec);
    case CEED_MEM_DEVICE:
      return CeedVectorSyncH2D_Hip(vec);
  }
  return CEED_ERROR_UNSUPPORTED;
}

//------------------------------------------------------------------------------
// Set all pointers as invalid
//------------------------------------------------------------------------------
static inline int CeedVectorSetAllInvalid_Hip(const CeedVector vec) {
  CeedVector_Hip *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  impl->h_array = NULL;
  impl->d_array = NULL;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if CeedVector has any valid pointers
//------------------------------------------------------------------------------
static inline int CeedVectorHasValidArray_Hip(const CeedVector vec, bool *has_valid_array) {
  CeedVector_Hip *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  *has_valid_array = !!impl->h_array || !!impl->d_array;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if has any array of given type
//------------------------------------------------------------------------------
static inline int CeedVectorHasArrayOfType_Hip(const CeedVector vec, CeedMemType mem_type, bool *has_array_of_type) {
  CeedVector_Hip *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  switch (mem_type) {
    case CEED_MEM_HOST:
      *has_array_of_type = !!impl->h_array_borrowed || !!impl->h_array_owned;
      break;
    case CEED_MEM_DEVICE:
      *has_array_of_type = !!impl->d_array_borrowed || !!impl->d_array_owned;
      break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if has borrowed array of given type
//------------------------------------------------------------------------------
static inline int CeedVectorHasBorrowedArrayOfType_Hip(const CeedVector vec, CeedMemType mem_type, bool *has_borrowed_array_of_type) {
  CeedVector_Hip *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  switch (mem_type) {
    case CEED_MEM_HOST:
      *has_borrowed_array_of_type = !!impl->h_array_borrowed;
      break;
    case CEED_MEM_DEVICE:
      *has_borrowed_array_of_type = !!impl->d_array_borrowed;
      break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set array from host
//------------------------------------------------------------------------------
static int CeedVectorSetArrayHost_Hip(const CeedVector vec, const CeedCopyMode copy_mode, CeedScalar *array) {
  CeedVector_Hip *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  switch (copy_mode) {
    case CEED_COPY_VALUES: {
      CeedSize length;
      if (!impl->h_array_owned) {
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
static int CeedVectorSetArrayDevice_Hip(const CeedVector vec, const CeedCopyMode copy_mode, CeedScalar *array) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedVector_Hip *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  switch (copy_mode) {
    case CEED_COPY_VALUES: {
      CeedSize length;
      CeedCallBackend(CeedVectorGetLength(vec, &length));
      size_t bytes = length * sizeof(CeedScalar);
      if (!impl->d_array_owned) {
        CeedCallHip(ceed, hipMalloc((void **)&impl->d_array_owned, bytes));
      }
      impl->d_array_borrowed = NULL;
      impl->d_array          = impl->d_array_owned;
      if (array) {
        CeedCallHip(ceed, hipMemcpy(impl->d_array, array, bytes, hipMemcpyDeviceToDevice));
      }
    } break;
    case CEED_OWN_POINTER:
      CeedCallHip(ceed, hipFree(impl->d_array_owned));
      impl->d_array_owned    = array;
      impl->d_array_borrowed = NULL;
      impl->d_array          = array;
      break;
    case CEED_USE_POINTER:
      CeedCallHip(ceed, hipFree(impl->d_array_owned));
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
static int CeedVectorSetArray_Hip(const CeedVector vec, const CeedMemType mem_type, const CeedCopyMode copy_mode, CeedScalar *array) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedVector_Hip *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  CeedCallBackend(CeedVectorSetAllInvalid_Hip(vec));
  switch (mem_type) {
    case CEED_MEM_HOST:
      return CeedVectorSetArrayHost_Hip(vec, copy_mode, array);
    case CEED_MEM_DEVICE:
      return CeedVectorSetArrayDevice_Hip(vec, copy_mode, array);
  }

  return CEED_ERROR_UNSUPPORTED;
}

//------------------------------------------------------------------------------
// Set host array to value
//------------------------------------------------------------------------------
static int CeedHostSetValue_Hip(CeedScalar *h_array, CeedInt length, CeedScalar val) {
  for (int i = 0; i < length; i++) h_array[i] = val;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set device array to value (impl in .hip file)
//------------------------------------------------------------------------------
int CeedDeviceSetValue_Hip(CeedScalar *d_array, CeedInt length, CeedScalar val);

//------------------------------------------------------------------------------
// Set a vector to a value,
//------------------------------------------------------------------------------
static int CeedVectorSetValue_Hip(CeedVector vec, CeedScalar val) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedVector_Hip *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedSize length;
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
    CeedCallBackend(CeedDeviceSetValue_Hip(impl->d_array, length, val));
  }
  if (impl->h_array) {
    CeedCallBackend(CeedHostSetValue_Hip(impl->h_array, length, val));
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Take Array
//------------------------------------------------------------------------------
static int CeedVectorTakeArray_Hip(CeedVector vec, CeedMemType mem_type, CeedScalar **array) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedVector_Hip *impl;
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
static int CeedVectorGetArrayCore_Hip(const CeedVector vec, const CeedMemType mem_type, CeedScalar **array) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedVector_Hip *impl;
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
static int CeedVectorGetArrayRead_Hip(const CeedVector vec, const CeedMemType mem_type, const CeedScalar **array) {
  return CeedVectorGetArrayCore_Hip(vec, mem_type, (CeedScalar **)array);
}

//------------------------------------------------------------------------------
// Get read/write access to a vector via the specified mem_type
//------------------------------------------------------------------------------
static int CeedVectorGetArray_Hip(const CeedVector vec, const CeedMemType mem_type, CeedScalar **array) {
  CeedVector_Hip *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  CeedCallBackend(CeedVectorGetArrayCore_Hip(vec, mem_type, array));

  CeedCallBackend(CeedVectorSetAllInvalid_Hip(vec));
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
static int CeedVectorGetArrayWrite_Hip(const CeedVector vec, const CeedMemType mem_type, CeedScalar **array) {
  CeedVector_Hip *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  bool has_array_of_type = true;
  CeedCallBackend(CeedVectorHasArrayOfType_Hip(vec, mem_type, &has_array_of_type));
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

  return CeedVectorGetArray_Hip(vec, mem_type, array);
}

//------------------------------------------------------------------------------
// Get the norm of a CeedVector
//------------------------------------------------------------------------------
static int CeedVectorNorm_Hip(CeedVector vec, CeedNormType type, CeedScalar *norm) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedVector_Hip *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedSize length;
  CeedCallBackend(CeedVectorGetLength(vec, &length));
  hipblasHandle_t handle;
  CeedCallBackend(CeedHipGetHipblasHandle(ceed, &handle));

  // Compute norm
  const CeedScalar *d_array;
  CeedCallBackend(CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE, &d_array));
  switch (type) {
    case CEED_NORM_1: {
      if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
        CeedCallHipblas(ceed, hipblasSasum(handle, length, (float *)d_array, 1, (float *)norm));
      } else {
        CeedCallHipblas(ceed, hipblasDasum(handle, length, (double *)d_array, 1, (double *)norm));
      }
      break;
    }
    case CEED_NORM_2: {
      if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
        CeedCallHipblas(ceed, hipblasSnrm2(handle, length, (float *)d_array, 1, (float *)norm));
      } else {
        CeedCallHipblas(ceed, hipblasDnrm2(handle, length, (double *)d_array, 1, (double *)norm));
      }
      break;
    }
    case CEED_NORM_MAX: {
      CeedInt indx;
      if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
        CeedCallHipblas(ceed, hipblasIsamax(handle, length, (float *)d_array, 1, &indx));
      } else {
        CeedCallHipblas(ceed, hipblasIdamax(handle, length, (double *)d_array, 1, &indx));
      }
      CeedScalar normNoAbs;
      CeedCallHip(ceed, hipMemcpy(&normNoAbs, impl->d_array + indx - 1, sizeof(CeedScalar), hipMemcpyDeviceToHost));
      *norm = fabs(normNoAbs);
      break;
    }
  }
  CeedCallBackend(CeedVectorRestoreArrayRead(vec, &d_array));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Take reciprocal of a vector on host
//------------------------------------------------------------------------------
static int CeedHostReciprocal_Hip(CeedScalar *h_array, CeedInt length) {
  for (int i = 0; i < length; i++) {
    if (fabs(h_array[i]) > CEED_EPSILON) h_array[i] = 1. / h_array[i];
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Take reciprocal of a vector on device (impl in .cu file)
//------------------------------------------------------------------------------
int CeedDeviceReciprocal_Hip(CeedScalar *d_array, CeedInt length);

//------------------------------------------------------------------------------
// Take reciprocal of a vector
//------------------------------------------------------------------------------
static int CeedVectorReciprocal_Hip(CeedVector vec) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedVector_Hip *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedSize length;
  CeedCallBackend(CeedVectorGetLength(vec, &length));

  // Set value for synced device/host array
  if (impl->d_array) CeedCallBackend(CeedDeviceReciprocal_Hip(impl->d_array, length));
  if (impl->h_array) CeedCallBackend(CeedHostReciprocal_Hip(impl->h_array, length));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute x = alpha x on the host
//------------------------------------------------------------------------------
static int CeedHostScale_Hip(CeedScalar *x_array, CeedScalar alpha, CeedInt length) {
  for (int i = 0; i < length; i++) x_array[i] *= alpha;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute x = alpha x on device (impl in .cu file)
//------------------------------------------------------------------------------
int CeedDeviceScale_Hip(CeedScalar *x_array, CeedScalar alpha, CeedInt length);

//------------------------------------------------------------------------------
// Compute x = alpha x
//------------------------------------------------------------------------------
static int CeedVectorScale_Hip(CeedVector x, CeedScalar alpha) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(x, &ceed));
  CeedVector_Hip *x_impl;
  CeedCallBackend(CeedVectorGetData(x, &x_impl));
  CeedSize length;
  CeedCallBackend(CeedVectorGetLength(x, &length));

  // Set value for synced device/host array
  if (x_impl->d_array) CeedCallBackend(CeedDeviceScale_Hip(x_impl->d_array, alpha, length));
  if (x_impl->h_array) CeedCallBackend(CeedHostScale_Hip(x_impl->h_array, alpha, length));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute y = alpha x + y on the host
//------------------------------------------------------------------------------
static int CeedHostAXPY_Hip(CeedScalar *y_array, CeedScalar alpha, CeedScalar *x_array, CeedInt length) {
  for (int i = 0; i < length; i++) y_array[i] += alpha * x_array[i];
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute y = alpha x + y on device (impl in .cu file)
//------------------------------------------------------------------------------
int CeedDeviceAXPY_Hip(CeedScalar *y_array, CeedScalar alpha, CeedScalar *x_array, CeedInt length);

//------------------------------------------------------------------------------
// Compute y = alpha x + y
//------------------------------------------------------------------------------
static int CeedVectorAXPY_Hip(CeedVector y, CeedScalar alpha, CeedVector x) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(y, &ceed));
  CeedVector_Hip *y_impl, *x_impl;
  CeedCallBackend(CeedVectorGetData(y, &y_impl));
  CeedCallBackend(CeedVectorGetData(x, &x_impl));
  CeedSize length;
  CeedCallBackend(CeedVectorGetLength(y, &length));

  // Set value for synced device/host array
  if (y_impl->d_array) {
    CeedCallBackend(CeedVectorSyncArray(x, CEED_MEM_DEVICE));
    CeedCallBackend(CeedDeviceAXPY_Hip(y_impl->d_array, alpha, x_impl->d_array, length));
  }
  if (y_impl->h_array) {
    CeedCallBackend(CeedVectorSyncArray(x, CEED_MEM_HOST));
    CeedCallBackend(CeedHostAXPY_Hip(y_impl->h_array, alpha, x_impl->h_array, length));
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute the pointwise multiplication w = x .* y on the host
//------------------------------------------------------------------------------
static int CeedHostPointwiseMult_Hip(CeedScalar *w_array, CeedScalar *x_array, CeedScalar *y_array, CeedInt length) {
  for (int i = 0; i < length; i++) w_array[i] = x_array[i] * y_array[i];
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute the pointwise multiplication w = x .* y on device (impl in .cu file)
//------------------------------------------------------------------------------
int CeedDevicePointwiseMult_Hip(CeedScalar *w_array, CeedScalar *x_array, CeedScalar *y_array, CeedInt length);

//------------------------------------------------------------------------------
// Compute the pointwise multiplication w = x .* y
//------------------------------------------------------------------------------
static int CeedVectorPointwiseMult_Hip(CeedVector w, CeedVector x, CeedVector y) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(w, &ceed));
  CeedVector_Hip *w_impl, *x_impl, *y_impl;
  CeedCallBackend(CeedVectorGetData(w, &w_impl));
  CeedCallBackend(CeedVectorGetData(x, &x_impl));
  CeedCallBackend(CeedVectorGetData(y, &y_impl));
  CeedSize length;
  CeedCallBackend(CeedVectorGetLength(w, &length));

  // Set value for synced device/host array
  if (!w_impl->d_array && !w_impl->h_array) {
    CeedCallBackend(CeedVectorSetValue(w, 0.0));
  }
  if (w_impl->d_array) {
    CeedCallBackend(CeedVectorSyncArray(x, CEED_MEM_DEVICE));
    CeedCallBackend(CeedVectorSyncArray(y, CEED_MEM_DEVICE));
    CeedCallBackend(CeedDevicePointwiseMult_Hip(w_impl->d_array, x_impl->d_array, y_impl->d_array, length));
  }
  if (w_impl->h_array) {
    CeedCallBackend(CeedVectorSyncArray(x, CEED_MEM_HOST));
    CeedCallBackend(CeedVectorSyncArray(y, CEED_MEM_HOST));
    CeedCallBackend(CeedHostPointwiseMult_Hip(w_impl->h_array, x_impl->h_array, y_impl->h_array, length));
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy the vector
//------------------------------------------------------------------------------
static int CeedVectorDestroy_Hip(const CeedVector vec) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedVector_Hip *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  CeedCallHip(ceed, hipFree(impl->d_array_owned));
  CeedCallBackend(CeedFree(&impl->h_array_owned));
  CeedCallBackend(CeedFree(&impl));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create a vector of the specified length (does not allocate memory)
//------------------------------------------------------------------------------
int CeedVectorCreate_Hip(CeedSize n, CeedVector vec) {
  CeedVector_Hip *impl;
  Ceed            ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "HasValidArray", CeedVectorHasValidArray_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "HasBorrowedArrayOfType", CeedVectorHasBorrowedArrayOfType_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "SetArray", CeedVectorSetArray_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "TakeArray", CeedVectorTakeArray_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "SetValue", (int (*)())(CeedVectorSetValue_Hip)));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "SyncArray", CeedVectorSyncArray_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "GetArray", CeedVectorGetArray_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayRead", CeedVectorGetArrayRead_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayWrite", CeedVectorGetArrayWrite_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "Norm", CeedVectorNorm_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "Reciprocal", CeedVectorReciprocal_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "Scale", (int (*)())(CeedVectorScale_Hip)));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "AXPY", (int (*)())(CeedVectorAXPY_Hip)));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "PointwiseMult", CeedVectorPointwiseMult_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "Destroy", CeedVectorDestroy_Hip));

  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedVectorSetData(vec, impl));

  return CEED_ERROR_SUCCESS;
}
