// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <hip/hip_runtime.h>

#include "../hip/ceed-hip-common.h"
#include "ceed-hip-ref.h"

//------------------------------------------------------------------------------
// Check if host/device sync is needed
//------------------------------------------------------------------------------
static inline int CeedVectorNeedSync_Hip(const CeedVector vec, CeedMemType mem_type, bool *need_sync) {
  CeedVector_Hip *impl;
  bool            has_valid_array = false;

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
static inline int CeedVectorSyncH2D_Hip(const CeedVector vec) {
  CeedSize        length;
  size_t          bytes;
  CeedVector_Hip *impl;

  CeedCallBackend(CeedVectorGetData(vec, &impl));

  CeedCheck(impl->h_array, CeedVectorReturnCeed(vec), CEED_ERROR_BACKEND, "No valid host data to sync to device");

  CeedCallBackend(CeedVectorGetLength(vec, &length));
  bytes = length * sizeof(CeedScalar);
  if (impl->d_array_borrowed) {
    impl->d_array = impl->d_array_borrowed;
  } else if (impl->d_array_owned) {
    impl->d_array = impl->d_array_owned;
  } else {
    CeedCallHip(CeedVectorReturnCeed(vec), hipMalloc((void **)&impl->d_array_owned, bytes));
    impl->d_array = impl->d_array_owned;
  }
  CeedCallHip(CeedVectorReturnCeed(vec), hipMemcpy(impl->d_array, impl->h_array, bytes, hipMemcpyHostToDevice));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync device to host
//------------------------------------------------------------------------------
static inline int CeedVectorSyncD2H_Hip(const CeedVector vec) {
  CeedSize        length;
  size_t          bytes;
  CeedVector_Hip *impl;

  CeedCallBackend(CeedVectorGetData(vec, &impl));

  CeedCheck(impl->d_array, CeedVectorReturnCeed(vec), CEED_ERROR_BACKEND, "No valid device data to sync to host");

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
  bytes = length * sizeof(CeedScalar);
  CeedCallHip(CeedVectorReturnCeed(vec), hipMemcpy(impl->h_array, impl->d_array, bytes, hipMemcpyDeviceToHost));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync arrays
//------------------------------------------------------------------------------
static int CeedVectorSyncArray_Hip(const CeedVector vec, CeedMemType mem_type) {
  bool      need_sync = false;
  Ceed_Hip *hip_data;

  CeedCallBackend(CeedGetData(CeedVectorReturnCeed(vec), &hip_data));

  // Unified memory doesn't require syncing, always use the host pointer
  if (hip_data->has_unified_addressing) {
    CeedVector_Hip *impl;

    CeedCallBackend(CeedVectorGetData(vec, &impl));
    CeedCallHip(CeedVectorReturnCeed(vec), hipDeviceSynchronize());
    CeedCheck(impl->d_array && !impl->h_array, CeedVectorReturnCeed(vec), CEED_ERROR_BACKEND,
              "Unified shared memory should only use device pointers");
    return CEED_ERROR_SUCCESS;
  }

  // Check whether device/host sync is needed
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
// Check if CeedVector has any valid pointer
//------------------------------------------------------------------------------
static inline int CeedVectorHasValidArray_Hip(const CeedVector vec, bool *has_valid_array) {
  CeedVector_Hip *impl;

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  *has_valid_array = impl->h_array || impl->d_array;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if has array of given type
//------------------------------------------------------------------------------
static inline int CeedVectorHasArrayOfType_Hip(const CeedVector vec, CeedMemType mem_type, bool *has_array_of_type) {
  CeedVector_Hip *impl;
  Ceed_Hip       *hip_data;

  CeedCallBackend(CeedGetData(CeedVectorReturnCeed(vec), &hip_data));
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  // Use device memory for unified memory
  mem_type = hip_data->has_unified_addressing ? CEED_MEM_DEVICE : mem_type;

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
static inline int CeedVectorHasBorrowedArrayOfType_Hip(const CeedVector vec, CeedMemType mem_type, bool *has_borrowed_array_of_type) {
  CeedVector_Hip *impl;
  Ceed_Hip       *hip_data;

  CeedCallBackend(CeedGetData(CeedVectorReturnCeed(vec), &hip_data));
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  // Use device memory for unified memory
  mem_type = hip_data->has_unified_addressing ? CEED_MEM_DEVICE : mem_type;

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
static int CeedVectorSetArrayHost_Hip(const CeedVector vec, const CeedCopyMode copy_mode, CeedScalar *array) {
  CeedSize        length;
  CeedVector_Hip *impl;

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedVectorGetLength(vec, &length));

  CeedCallBackend(CeedSetHostCeedScalarArray(array, copy_mode, length, (const CeedScalar **)&impl->h_array_owned,
                                             (const CeedScalar **)&impl->h_array_borrowed, (const CeedScalar **)&impl->h_array));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set array from device
//------------------------------------------------------------------------------
static int CeedVectorSetArrayDevice_Hip(const CeedVector vec, const CeedCopyMode copy_mode, CeedScalar *array) {
  CeedSize        length;
  Ceed            ceed;
  CeedVector_Hip *impl;

  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedVectorGetLength(vec, &length));

  CeedCallBackend(CeedSetDeviceCeedScalarArray_Hip(ceed, array, copy_mode, length, (const CeedScalar **)&impl->d_array_owned,
                                                   (const CeedScalar **)&impl->d_array_borrowed, (const CeedScalar **)&impl->d_array));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set the array used by a vector,
//   freeing any previously allocated array if applicable
//------------------------------------------------------------------------------
static int CeedVectorSetArray_Hip(const CeedVector vec, const CeedMemType mem_type, const CeedCopyMode copy_mode, CeedScalar *array) {
  CeedVector_Hip *impl;
  Ceed_Hip       *hip_data;
  CeedMemType     local_mem_type;

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedGetData(CeedVectorReturnCeed(vec), &hip_data));
  CeedCallBackend(CeedVectorSetAllInvalid_Hip(vec));

  // Use device memory for unified memory
  local_mem_type = hip_data->has_unified_addressing ? CEED_MEM_DEVICE : mem_type;

  switch (local_mem_type) {
    case CEED_MEM_HOST:
      return CeedVectorSetArrayHost_Hip(vec, copy_mode, array);
    case CEED_MEM_DEVICE:
      return CeedVectorSetArrayDevice_Hip(vec, copy_mode, array);
  }
  return CEED_ERROR_UNSUPPORTED;
}

//------------------------------------------------------------------------------
// Copy host array to value strided
//------------------------------------------------------------------------------
static int CeedHostCopyStrided_Hip(CeedScalar *h_array, CeedSize start, CeedSize step, CeedSize length, CeedScalar *h_copy_array) {
  for (CeedSize i = start; i < length; i += step) h_copy_array[i] = h_array[i];
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Copy a vector to a value strided
//------------------------------------------------------------------------------
static int CeedVectorCopyStrided_Hip(CeedVector vec, CeedSize start, CeedSize step, CeedVector vec_copy) {
  CeedSize        length;
  CeedVector_Hip *impl;
  Ceed_Hip       *hip_data;
  hipblasHandle_t handle;

  CeedCallBackend(CeedGetHipblasHandle_Hip(CeedVectorReturnCeed(vec), &handle));
  CeedCallBackend(CeedGetData(CeedVectorReturnCeed(vec), &hip_data));
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  {
    CeedSize length_vec, length_copy;

    CeedCallBackend(CeedVectorGetLength(vec, &length_vec));
    CeedCallBackend(CeedVectorGetLength(vec_copy, &length_copy));
    length = length_vec < length_copy ? length_vec : length_copy;
  }

  // Set value for synced device/host array
  if (impl->d_array) {
    CeedScalar *copy_array;

    // Number of values to copy
    // start value (1) + floor((length - start) / step) addl. values
    length = (length - start) / step + 1;
    CeedCallBackend(CeedVectorGetArray(vec_copy, CEED_MEM_DEVICE, &copy_array));
#if defined(CEED_SCALAR_IS_FP32)
    CeedCallHipblas(CeedVectorReturnCeed(vec),
                    hipblasScopy_64(handle, (int64_t)length, impl->d_array + start, (int64_t)step, copy_array + start, (int64_t)step));
#else
    CeedCallHipblas(CeedVectorReturnCeed(vec),
                    hipblasDcopy_64(handle, (int64_t)length, impl->d_array + start, (int64_t)step, copy_array + start, (int64_t)step));
#endif
    CeedCallBackend(CeedVectorRestoreArray(vec_copy, &copy_array));
  } else if (impl->h_array) {
    CeedScalar *copy_array;

    CeedCallBackend(CeedVectorGetArray(vec_copy, CEED_MEM_HOST, &copy_array));
    CeedCallBackend(CeedHostCopyStrided_Hip(impl->h_array, start, step, length, copy_array));
    CeedCallBackend(CeedVectorRestoreArray(vec_copy, &copy_array));
  } else {
    return CeedError(CeedVectorReturnCeed(vec), CEED_ERROR_BACKEND, "CeedVector must have valid data set");
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set host array to value
//------------------------------------------------------------------------------
static int CeedHostSetValue_Hip(CeedScalar *h_array, CeedSize length, CeedScalar val) {
  for (CeedSize i = 0; i < length; i++) h_array[i] = val;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set device array to value (impl in .hip file)
//------------------------------------------------------------------------------
int CeedDeviceSetValue_Hip(CeedScalar *d_array, CeedSize length, CeedScalar val);

//------------------------------------------------------------------------------
// Set a vector to a value
//------------------------------------------------------------------------------
static int CeedVectorSetValue_Hip(CeedVector vec, CeedScalar val) {
  CeedSize        length;
  CeedVector_Hip *impl;
  Ceed_Hip       *hip_data;

  CeedCallBackend(CeedGetData(CeedVectorReturnCeed(vec), &hip_data));
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
    if (!hip_data->has_unified_addressing && val == 0) {
      CeedCallHip(CeedVectorReturnCeed(vec), hipMemset(impl->d_array, 0, length * sizeof(CeedScalar)));
    } else {
      CeedCallBackend(CeedDeviceSetValue_Hip(impl->d_array, length, val));
    }
    impl->h_array = NULL;
  } else if (impl->h_array) {
    CeedCallBackend(CeedHostSetValue_Hip(impl->h_array, length, val));
    impl->d_array = NULL;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set host array to value strided
//------------------------------------------------------------------------------
static int CeedHostSetValueStrided_Hip(CeedScalar *h_array, CeedSize start, CeedSize step, CeedSize length, CeedScalar val) {
  for (CeedSize i = start; i < length; i += step) h_array[i] = val;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set device array to value strided (impl in .hip.cpp file)
//------------------------------------------------------------------------------
int CeedDeviceSetValueStrided_Hip(CeedScalar *d_array, CeedSize start, CeedSize step, CeedSize length, CeedScalar val);

//------------------------------------------------------------------------------
// Set a vector to a value strided
//------------------------------------------------------------------------------
static int CeedVectorSetValueStrided_Hip(CeedVector vec, CeedSize start, CeedSize step, CeedScalar val) {
  CeedSize        length;
  CeedVector_Hip *impl;
  Ceed_Hip       *hip_data;

  CeedCallBackend(CeedGetData(CeedVectorReturnCeed(vec), &hip_data));
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedVectorGetLength(vec, &length));

  // Set value for synced device/host array
  if (impl->d_array) {
    CeedCallBackend(CeedDeviceSetValueStrided_Hip(impl->d_array, start, step, length, val));
    impl->h_array = NULL;
  } else if (impl->h_array) {
    CeedCallBackend(CeedHostSetValueStrided_Hip(impl->h_array, start, step, length, val));
    impl->d_array = NULL;
  } else {
    return CeedError(CeedVectorReturnCeed(vec), CEED_ERROR_BACKEND, "CeedVector must have valid data set");
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Take Array
//------------------------------------------------------------------------------
static int CeedVectorTakeArray_Hip(CeedVector vec, CeedMemType mem_type, CeedScalar **array) {
  CeedVector_Hip *impl;
  Ceed_Hip       *hip_data;

  CeedCallBackend(CeedGetData(CeedVectorReturnCeed(vec), &hip_data));
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  // Use device memory for unified memory
  mem_type = hip_data->has_unified_addressing ? CEED_MEM_DEVICE : mem_type;

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
static int CeedVectorGetArrayCore_Hip(const CeedVector vec, CeedMemType mem_type, CeedScalar **array) {
  CeedVector_Hip *impl;
  Ceed_Hip       *hip_data;

  CeedCallBackend(CeedGetData(CeedVectorReturnCeed(vec), &hip_data));
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  // Use device memory for unified memory
  mem_type = hip_data->has_unified_addressing ? CEED_MEM_DEVICE : mem_type;

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
  CeedMemType     local_mem_type;
  Ceed_Hip       *hip_data;

  CeedCallBackend(CeedGetData(CeedVectorReturnCeed(vec), &hip_data));
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  // Use device memory for unified memory
  local_mem_type = hip_data->has_unified_addressing ? CEED_MEM_DEVICE : mem_type;

  CeedCallBackend(CeedVectorGetArrayCore_Hip(vec, local_mem_type, array));
  CeedCallBackend(CeedVectorSetAllInvalid_Hip(vec));
  switch (local_mem_type) {
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
  bool            has_array_of_type = true;
  CeedVector_Hip *impl;
  CeedMemType     local_mem_type;
  Ceed_Hip       *hip_data;

  CeedCallBackend(CeedGetData(CeedVectorReturnCeed(vec), &hip_data));
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  // Use device memory for unified memory
  local_mem_type = hip_data->has_unified_addressing ? CEED_MEM_DEVICE : mem_type;

  CeedCallBackend(CeedVectorHasArrayOfType_Hip(vec, local_mem_type, &has_array_of_type));
  if (!has_array_of_type) {
    // Allocate if array is not yet allocated
    CeedCallBackend(CeedVectorSetArray(vec, local_mem_type, CEED_COPY_VALUES, NULL));
  } else {
    // Select dirty array
    switch (local_mem_type) {
      case CEED_MEM_HOST:
        if (impl->h_array_borrowed) impl->h_array = impl->h_array_borrowed;
        else impl->h_array = impl->h_array_owned;
        break;
      case CEED_MEM_DEVICE:
        if (impl->d_array_borrowed) impl->d_array = impl->d_array_borrowed;
        else impl->d_array = impl->d_array_owned;
    }
  }
  return CeedVectorGetArray_Hip(vec, local_mem_type, array);
}

//------------------------------------------------------------------------------
// Get the norm of a CeedVector
//------------------------------------------------------------------------------
static int CeedVectorNorm_Hip(CeedVector vec, CeedNormType type, CeedScalar *norm) {
  Ceed              ceed;
  CeedSize          length;
  const CeedScalar *d_array;
  CeedVector_Hip   *impl;
  hipblasHandle_t   handle;
  Ceed_Hip         *hip_data;

  CeedCallBackend(CeedGetData(CeedVectorReturnCeed(vec), &hip_data));
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedVectorGetLength(vec, &length));
  CeedCallBackend(CeedGetHipblasHandle_Hip(ceed, &handle));

  // Compute norm
  CeedCallBackend(CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE, &d_array));

  *norm = 0.0;
  switch (type) {
    case CEED_NORM_1: {
#if defined(CEED_SCALAR_IS_FP32)
      CeedCallHipblas(ceed, hipblasSasum_64(handle, (int64_t)length, (CeedScalar *)d_array, 1, norm));
#else
      CeedCallHipblas(ceed, hipblasDasum_64(handle, (int64_t)length, (CeedScalar *)d_array, 1, norm));
#endif
      CeedCallHip(ceed, hipDeviceSynchronize());
      break;
    }
    case CEED_NORM_2: {
#if defined(CEED_SCALAR_IS_FP32)
      CeedCallHipblas(ceed, hipblasSnrm2_64(handle, (int64_t)length, (CeedScalar *)d_array, 1, norm));
#else
      CeedCallHipblas(ceed, hipblasDnrm2_64(handle, (int64_t)length, (CeedScalar *)d_array, 1, norm));
#endif
      CeedCallHip(ceed, hipDeviceSynchronize());
      break;
    }
    case CEED_NORM_MAX: {
      int64_t index;

#if defined(CEED_SCALAR_IS_FP32)
      CeedCallHipblas(ceed, hipblasIsamax_64(handle, (int64_t)length, (CeedScalar *)d_array, 1, &index));
#else
      CeedCallHipblas(ceed, hipblasIdamax_64(handle, (int64_t)length, (CeedScalar *)d_array, 1, &index));
#endif
      CeedCallHip(ceed, hipDeviceSynchronize());
      if (hip_data->has_unified_addressing) {
        *norm = fabs(d_array[index - 1]);
      } else {
        CeedCallHip(ceed, hipMemcpy(norm, d_array + index - 1, sizeof(CeedScalar), hipMemcpyDeviceToHost));
        *norm = fabs(*norm);
      }
      break;
    }
  }
  CeedCallBackend(CeedVectorRestoreArrayRead(vec, &d_array));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Take reciprocal of a vector on host
//------------------------------------------------------------------------------
static int CeedHostReciprocal_Hip(CeedScalar *h_array, CeedSize length) {
  for (CeedSize i = 0; i < length; i++) {
    if (fabs(h_array[i]) > CEED_EPSILON) h_array[i] = 1. / h_array[i];
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Take reciprocal of a vector on device (impl in .hip.cpp file)
//------------------------------------------------------------------------------
int CeedDeviceReciprocal_Hip(CeedScalar *d_array, CeedSize length);

//------------------------------------------------------------------------------
// Take reciprocal of a vector
//------------------------------------------------------------------------------
static int CeedVectorReciprocal_Hip(CeedVector vec) {
  CeedSize        length;
  CeedVector_Hip *impl;
  Ceed_Hip       *hip_data;

  CeedCallBackend(CeedGetData(CeedVectorReturnCeed(vec), &hip_data));
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedVectorGetLength(vec, &length));

  // Set value for synced device/host array
  if (impl->d_array) CeedCallBackend(CeedDeviceReciprocal_Hip(impl->d_array, length));
  if (impl->h_array) CeedCallBackend(CeedHostReciprocal_Hip(impl->h_array, length));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute x = alpha x on the host
//------------------------------------------------------------------------------
static int CeedHostScale_Hip(CeedScalar *x_array, CeedScalar alpha, CeedSize length) {
  for (CeedSize i = 0; i < length; i++) x_array[i] *= alpha;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute x = alpha x
//------------------------------------------------------------------------------
static int CeedVectorScale_Hip(CeedVector x, CeedScalar alpha) {
  CeedSize        length;
  CeedVector_Hip *impl;
  Ceed_Hip       *hip_data;
  hipblasHandle_t handle;

  CeedCallBackend(CeedGetHipblasHandle_Hip(CeedVectorReturnCeed(x), &handle));
  CeedCallBackend(CeedGetData(CeedVectorReturnCeed(x), &hip_data));
  CeedCallBackend(CeedVectorGetData(x, &impl));
  CeedCallBackend(CeedVectorGetLength(x, &length));

  // Set value for synced device/host array
  if (impl->d_array) {
#if defined(CEED_SCALAR_IS_FP32)
    CeedCallHipblas(CeedVectorReturnCeed(x), hipblasSscal_64(handle, (int64_t)length, &alpha, impl->d_array, 1));
#else
    CeedCallHipblas(CeedVectorReturnCeed(x), hipblasDscal_64(handle, (int64_t)length, &alpha, impl->d_array, 1));
#endif
  }
  if (impl->h_array) CeedCallBackend(CeedHostScale_Hip(impl->h_array, alpha, length));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute y = alpha x + y on the host
//------------------------------------------------------------------------------
static int CeedHostAXPY_Hip(CeedScalar *y_array, CeedScalar alpha, CeedScalar *x_array, CeedSize length) {
  for (CeedSize i = 0; i < length; i++) y_array[i] += alpha * x_array[i];
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute y = alpha x + y
//------------------------------------------------------------------------------
static int CeedVectorAXPY_Hip(CeedVector y, CeedScalar alpha, CeedVector x) {
  CeedSize        length;
  CeedVector_Hip *y_impl, *x_impl;
  Ceed_Hip       *hip_data;
  hipblasHandle_t handle;

  CeedCallBackend(CeedGetData(CeedVectorReturnCeed(y), &hip_data));
  CeedCallBackend(CeedVectorGetData(y, &y_impl));
  CeedCallBackend(CeedVectorGetData(x, &x_impl));
  CeedCallBackend(CeedGetHipblasHandle_Hip(CeedVectorReturnCeed(y), &handle));

  CeedCallBackend(CeedVectorGetLength(y, &length));

  // Set value for synced device/host array
  if (y_impl->d_array) {
    CeedCallBackend(CeedVectorSyncArray(x, CEED_MEM_DEVICE));
#if defined(CEED_SCALAR_IS_FP32)
    CeedCallHipblas(CeedVectorReturnCeed(y), hipblasSaxpy_64(handle, (int64_t)length, &alpha, x_impl->d_array, 1, y_impl->d_array, 1));
#else
    CeedCallHipblas(CeedVectorReturnCeed(y), hipblasDaxpy_64(handle, (int64_t)length, &alpha, x_impl->d_array, 1, y_impl->d_array, 1));
#endif
  }
  if (y_impl->h_array) {
    CeedCallBackend(CeedVectorSyncArray(x, CEED_MEM_HOST));
    CeedCallBackend(CeedHostAXPY_Hip(y_impl->h_array, alpha, x_impl->h_array, length));
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute y = alpha x + beta y on the host
//------------------------------------------------------------------------------
static int CeedHostAXPBY_Hip(CeedScalar *y_array, CeedScalar alpha, CeedScalar beta, CeedScalar *x_array, CeedSize length) {
  for (CeedSize i = 0; i < length; i++) y_array[i] = alpha * x_array[i] + beta * y_array[i];
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute y = alpha x + beta y on device (impl in .hip.cpp file)
//------------------------------------------------------------------------------
int CeedDeviceAXPBY_Hip(CeedScalar *y_array, CeedScalar alpha, CeedScalar beta, CeedScalar *x_array, CeedSize length);

//------------------------------------------------------------------------------
// Compute y = alpha x + beta y
//------------------------------------------------------------------------------
static int CeedVectorAXPBY_Hip(CeedVector y, CeedScalar alpha, CeedScalar beta, CeedVector x) {
  CeedSize        length;
  CeedVector_Hip *y_impl, *x_impl;
  Ceed_Hip       *hip_data;
  hipblasHandle_t handle;

  CeedCallBackend(CeedGetData(CeedVectorReturnCeed(y), &hip_data));
  CeedCallBackend(CeedGetHipblasHandle_Hip(CeedVectorReturnCeed(y), &handle));
  CeedCallBackend(CeedVectorGetData(y, &y_impl));
  CeedCallBackend(CeedVectorGetData(x, &x_impl));

  CeedCallBackend(CeedVectorGetLength(y, &length));

  // Set value for synced device/host array
  if (y_impl->d_array) {
    CeedCallBackend(CeedVectorSyncArray(x, CEED_MEM_DEVICE));
    CeedCallBackend(CeedDeviceAXPBY_Hip(y_impl->d_array, alpha, beta, x_impl->d_array, length));
  }
  if (y_impl->h_array) {
    CeedCallBackend(CeedVectorSyncArray(x, CEED_MEM_HOST));
    CeedCallBackend(CeedHostAXPBY_Hip(y_impl->h_array, alpha, beta, x_impl->h_array, length));
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute the pointwise multiplication w = x .* y on the host
//------------------------------------------------------------------------------
static int CeedHostPointwiseMult_Hip(CeedScalar *w_array, CeedScalar *x_array, CeedScalar *y_array, CeedSize length) {
  for (CeedSize i = 0; i < length; i++) w_array[i] = x_array[i] * y_array[i];
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute the pointwise multiplication w = x .* y on device (impl in .hip.cpp file)
//------------------------------------------------------------------------------
int CeedDevicePointwiseMult_Hip(CeedScalar *w_array, CeedScalar *x_array, CeedScalar *y_array, CeedSize length);

//------------------------------------------------------------------------------
// Compute the pointwise multiplication w = x .* y
//------------------------------------------------------------------------------
static int CeedVectorPointwiseMult_Hip(CeedVector w, CeedVector x, CeedVector y) {
  CeedSize        length;
  CeedVector_Hip *w_impl, *x_impl, *y_impl;
  Ceed_Hip       *hip_data;

  CeedCallBackend(CeedGetData(CeedVectorReturnCeed(x), &hip_data));
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
  CeedVector_Hip *impl;
  Ceed_Hip       *hip_data;

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedGetData(CeedVectorReturnCeed(vec), &hip_data));

  CeedCallHip(CeedVectorReturnCeed(vec), hipFree(impl->d_array_owned));
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
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "CopyStrided", CeedVectorCopyStrided_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "SetValue", CeedVectorSetValue_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "SetValueStrided", CeedVectorSetValueStrided_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "SyncArray", CeedVectorSyncArray_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "GetArray", CeedVectorGetArray_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayRead", CeedVectorGetArrayRead_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayWrite", CeedVectorGetArrayWrite_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "Norm", CeedVectorNorm_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "Reciprocal", CeedVectorReciprocal_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "Scale", CeedVectorScale_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "AXPY", CeedVectorAXPY_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "AXPBY", CeedVectorAXPBY_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "PointwiseMult", CeedVectorPointwiseMult_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "Destroy", CeedVectorDestroy_Hip));
  CeedCallBackend(CeedDestroy(&ceed));
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedVectorSetData(vec, impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
