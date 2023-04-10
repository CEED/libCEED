// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>

#include <cmath>
#include <string>
#include <sycl/sycl.hpp>

#include "ceed-sycl-ref.hpp"

//------------------------------------------------------------------------------
// Check if host/device sync is needed
//------------------------------------------------------------------------------
static inline int CeedVectorNeedSync_Sycl(const CeedVector vec, CeedMemType mem_type, bool *need_sync) {
  CeedVector_Sycl *impl;
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
static inline int CeedVectorSyncH2D_Sycl(const CeedVector vec) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedVector_Sycl *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  if (!impl->h_array) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "No valid host data to sync to device");
    // LCOV_EXCL_STOP
  }

  CeedSize length;
  CeedCallBackend(CeedVectorGetLength(vec, &length));

  if (impl->d_array_borrowed) {
    impl->d_array = impl->d_array_borrowed;
  } else if (impl->d_array_owned) {
    impl->d_array = impl->d_array_owned;
  } else {
    CeedCallSycl(ceed, impl->d_array_owned = sycl::malloc_device<CeedScalar>(length, data->sycl_device, data->sycl_context));
    impl->d_array = impl->d_array_owned;
  }

  sycl::event e = data->sycl_queue.ext_oneapi_submit_barrier();
  // Copy from host to device
  sycl::event copy_event = data->sycl_queue.copy<CeedScalar>(impl->h_array, impl->d_array, length, {e});
  // Wait for copy to finish and handle exceptions.
  CeedCallSycl(ceed, copy_event.wait_and_throw());

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync device to host
//------------------------------------------------------------------------------
static inline int CeedVectorSyncD2H_Sycl(const CeedVector vec) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedVector_Sycl *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data))

      if (!impl->d_array) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "No valid device data to sync to host");
    // LCOV_EXCL_STOP
  }

  CeedSize length;
  CeedCallBackend(CeedVectorGetLength(vec, &length));

  if (impl->h_array_borrowed) {
    impl->h_array = impl->h_array_borrowed;
  } else if (impl->h_array_owned) {
    impl->h_array = impl->h_array_owned;
  } else {
    CeedCallBackend(CeedCalloc(length, &impl->h_array_owned));
    impl->h_array = impl->h_array_owned;
  }

  // Order queue
  sycl::event e = data->sycl_queue.ext_oneapi_submit_barrier();
  // Copy from device to host
  sycl::event copy_event = data->sycl_queue.copy<CeedScalar>(impl->d_array, impl->h_array, length, {e});

  // Wait for copy to finish and handle exceptions.
  CeedCallSycl(ceed, copy_event.wait_and_throw());

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync arrays
//------------------------------------------------------------------------------
static int CeedVectorSyncArray_Sycl(const CeedVector vec, CeedMemType mem_type) {
  // Check whether device/host sync is needed
  bool need_sync = false;
  CeedCallBackend(CeedVectorNeedSync_Sycl(vec, mem_type, &need_sync));
  if (!need_sync) return CEED_ERROR_SUCCESS;

  switch (mem_type) {
    case CEED_MEM_HOST:
      return CeedVectorSyncD2H_Sycl(vec);
    case CEED_MEM_DEVICE:
      return CeedVectorSyncH2D_Sycl(vec);
  }
  return CEED_ERROR_UNSUPPORTED;
}

//------------------------------------------------------------------------------
// Set all pointers as invalid
//------------------------------------------------------------------------------
static inline int CeedVectorSetAllInvalid_Sycl(const CeedVector vec) {
  CeedVector_Sycl *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  impl->h_array = NULL;
  impl->d_array = NULL;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if CeedVector has any valid pointer
//------------------------------------------------------------------------------
static inline int CeedVectorHasValidArray_Sycl(const CeedVector vec, bool *has_valid_array) {
  CeedVector_Sycl *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  *has_valid_array = !!impl->h_array || !!impl->d_array;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if has array of given type
//------------------------------------------------------------------------------
static inline int CeedVectorHasArrayOfType_Sycl(const CeedVector vec, CeedMemType mem_type, bool *has_array_of_type) {
  CeedVector_Sycl *impl;
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
static inline int CeedVectorHasBorrowedArrayOfType_Sycl(const CeedVector vec, CeedMemType mem_type, bool *has_borrowed_array_of_type) {
  CeedVector_Sycl *impl;
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
static int CeedVectorSetArrayHost_Sycl(const CeedVector vec, const CeedCopyMode copy_mode, CeedScalar *array) {
  CeedVector_Sycl *impl;
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
static int CeedVectorSetArrayDevice_Sycl(const CeedVector vec, const CeedCopyMode copy_mode, CeedScalar *array) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedVector_Sycl *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  // Order queue
  sycl::event e = data->sycl_queue.ext_oneapi_submit_barrier();

  switch (copy_mode) {
    case CEED_COPY_VALUES: {
      CeedSize length;
      CeedCallBackend(CeedVectorGetLength(vec, &length));
      if (!impl->d_array_owned) {
        CeedCallSycl(ceed, impl->d_array_owned = sycl::malloc_device<CeedScalar>(length, data->sycl_device, data->sycl_context));
        impl->d_array = impl->d_array_owned;
      }
      if (array) {
        sycl::event copy_event = data->sycl_queue.copy<CeedScalar>(array, impl->d_array, length, {e});
        // Wait for copy to finish and handle exceptions.
        CeedCallSycl(ceed, copy_event.wait_and_throw());
      }
    } break;
    case CEED_OWN_POINTER:
      if (impl->d_array_owned) {
        // Wait for all work to finish before freeing memory
        CeedCallSycl(ceed, data->sycl_queue.wait_and_throw());
        CeedCallSycl(ceed, sycl::free(impl->d_array_owned, data->sycl_context));
      }
      impl->d_array_owned    = array;
      impl->d_array_borrowed = NULL;
      impl->d_array          = array;
      break;
    case CEED_USE_POINTER:
      if (impl->d_array_owned) {
        // Wait for all work to finish before freeing memory
        CeedCallSycl(ceed, data->sycl_queue.wait_and_throw());
        CeedCallSycl(ceed, sycl::free(impl->d_array_owned, data->sycl_context));
      }
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
static int CeedVectorSetArray_Sycl(const CeedVector vec, const CeedMemType mem_type, const CeedCopyMode copy_mode, CeedScalar *array) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedVector_Sycl *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  CeedCallBackend(CeedVectorSetAllInvalid_Sycl(vec));
  switch (mem_type) {
    case CEED_MEM_HOST:
      return CeedVectorSetArrayHost_Sycl(vec, copy_mode, array);
    case CEED_MEM_DEVICE:
      return CeedVectorSetArrayDevice_Sycl(vec, copy_mode, array);
  }

  return CEED_ERROR_UNSUPPORTED;
}

//------------------------------------------------------------------------------
// Set host array to value
//------------------------------------------------------------------------------
static int CeedHostSetValue_Sycl(CeedScalar *h_array, CeedInt length, CeedScalar val) {
  for (int i = 0; i < length; i++) h_array[i] = val;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set device array to value
//------------------------------------------------------------------------------
static int CeedDeviceSetValue_Sycl(sycl::queue &sycl_queue, CeedScalar *d_array, CeedInt length, CeedScalar val) {
  // Order queue
  sycl::event e = sycl_queue.ext_oneapi_submit_barrier();
  sycl_queue.fill(d_array, val, length, {e});
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set a vector to a value,
//------------------------------------------------------------------------------
static int CeedVectorSetValue_Sycl(CeedVector vec, CeedScalar val) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedVector_Sycl *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedSize length;
  CeedCallBackend(CeedVectorGetLength(vec, &length));
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

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
    CeedCallBackend(CeedDeviceSetValue_Sycl(data->sycl_queue, impl->d_array, length, val));
    impl->h_array = NULL;
  }
  if (impl->h_array) {
    CeedCallBackend(CeedHostSetValue_Sycl(impl->h_array, length, val));
    impl->d_array = NULL;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Take Array
//------------------------------------------------------------------------------
static int CeedVectorTakeArray_Sycl(CeedVector vec, CeedMemType mem_type, CeedScalar **array) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedVector_Sycl *impl;
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
static int CeedVectorGetArrayCore_Sycl(const CeedVector vec, const CeedMemType mem_type, CeedScalar **array) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedVector_Sycl *impl;
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
static int CeedVectorGetArrayRead_Sycl(const CeedVector vec, const CeedMemType mem_type, const CeedScalar **array) {
  return CeedVectorGetArrayCore_Sycl(vec, mem_type, (CeedScalar **)array);
}

//------------------------------------------------------------------------------
// Get read/write access to a vector via the specified mem_type
//------------------------------------------------------------------------------
static int CeedVectorGetArray_Sycl(const CeedVector vec, const CeedMemType mem_type, CeedScalar **array) {
  CeedVector_Sycl *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  CeedCallBackend(CeedVectorGetArrayCore_Sycl(vec, mem_type, array));

  CeedCallBackend(CeedVectorSetAllInvalid_Sycl(vec));
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
static int CeedVectorGetArrayWrite_Sycl(const CeedVector vec, const CeedMemType mem_type, CeedScalar **array) {
  CeedVector_Sycl *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  bool has_array_of_type = true;
  CeedCallBackend(CeedVectorHasArrayOfType_Sycl(vec, mem_type, &has_array_of_type));
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

  return CeedVectorGetArray_Sycl(vec, mem_type, array);
}

//------------------------------------------------------------------------------
// Get the norm of a CeedVector
//------------------------------------------------------------------------------
static int CeedVectorNorm_Sycl(CeedVector vec, CeedNormType type, CeedScalar *norm) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedVector_Sycl *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedSize length;
  CeedCallBackend(CeedVectorGetLength(vec, &length));
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  // Compute norm
  const CeedScalar *d_array;
  CeedCallBackend(CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE, &d_array));

  switch (type) {
    case CEED_NORM_1: {
      // Order queue
      sycl::event e            = data->sycl_queue.ext_oneapi_submit_barrier();
      auto        sumReduction = sycl::reduction(impl->reduction_norm, sycl::plus<>());
      data->sycl_queue.parallel_for(length, {e}, sumReduction, [=](sycl::id<1> i, auto &sum) { sum += abs(d_array[i]); }).wait_and_throw();
    } break;
    case CEED_NORM_2: {
      // Order queue
      sycl::event e            = data->sycl_queue.ext_oneapi_submit_barrier();
      auto        sumReduction = sycl::reduction(impl->reduction_norm, sycl::plus<>());
      data->sycl_queue.parallel_for(length, {e}, sumReduction, [=](sycl::id<1> i, auto &sum) { sum += (d_array[i] * d_array[i]); }).wait_and_throw();
    } break;
    case CEED_NORM_MAX: {
      // Order queue
      sycl::event e            = data->sycl_queue.ext_oneapi_submit_barrier();
      auto        maxReduction = sycl::reduction(impl->reduction_norm, sycl::maximum<>());
      data->sycl_queue.parallel_for(length, {e}, maxReduction, [=](sycl::id<1> i, auto &max) { max.combine(abs(d_array[i])); }).wait_and_throw();
    } break;
  }
  // L2 norm - square root over reduced value
  if (type == CEED_NORM_2) *norm = sqrt(*impl->reduction_norm);
  else *norm = *impl->reduction_norm;

  CeedCallBackend(CeedVectorRestoreArrayRead(vec, &d_array));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Take reciprocal of a vector on host
//------------------------------------------------------------------------------
static int CeedHostReciprocal_Sycl(CeedScalar *h_array, CeedInt length) {
  for (int i = 0; i < length; i++) {
    if (std::fabs(h_array[i]) > CEED_EPSILON) h_array[i] = 1. / h_array[i];
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Take reciprocal of a vector on device
//------------------------------------------------------------------------------
static int CeedDeviceReciprocal_Sycl(sycl::queue &sycl_queue, CeedScalar *d_array, CeedInt length) {
  // Order queue
  sycl::event e = sycl_queue.ext_oneapi_submit_barrier();
  sycl_queue.parallel_for(length, {e}, [=](sycl::id<1> i) {
    if (std::fabs(d_array[i]) > CEED_EPSILON) d_array[i] = 1. / d_array[i];
  });
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Take reciprocal of a vector
//------------------------------------------------------------------------------
static int CeedVectorReciprocal_Sycl(CeedVector vec) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedVector_Sycl *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedSize length;
  CeedCallBackend(CeedVectorGetLength(vec, &length));
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  // Set value for synced device/host array
  if (impl->d_array) CeedCallBackend(CeedDeviceReciprocal_Sycl(data->sycl_queue, impl->d_array, length));
  if (impl->h_array) CeedCallBackend(CeedHostReciprocal_Sycl(impl->h_array, length));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute x = alpha x on the host
//------------------------------------------------------------------------------
static int CeedHostScale_Sycl(CeedScalar *x_array, CeedScalar alpha, CeedInt length) {
  for (int i = 0; i < length; i++) x_array[i] *= alpha;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute x = alpha x on device
//------------------------------------------------------------------------------
static int CeedDeviceScale_Sycl(sycl::queue &sycl_queue, CeedScalar *x_array, CeedScalar alpha, CeedInt length) {
  // Order queue
  sycl::event e = sycl_queue.ext_oneapi_submit_barrier();
  sycl_queue.parallel_for(length, {e}, [=](sycl::id<1> i) { x_array[i] *= alpha; });
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute x = alpha x
//------------------------------------------------------------------------------
static int CeedVectorScale_Sycl(CeedVector x, CeedScalar alpha) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(x, &ceed));
  CeedVector_Sycl *x_impl;
  CeedCallBackend(CeedVectorGetData(x, &x_impl));
  CeedSize length;
  CeedCallBackend(CeedVectorGetLength(x, &length));
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  // Set value for synced device/host array
  if (x_impl->d_array) CeedCallBackend(CeedDeviceScale_Sycl(data->sycl_queue, x_impl->d_array, alpha, length));
  if (x_impl->h_array) CeedCallBackend(CeedHostScale_Sycl(x_impl->h_array, alpha, length));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute y = alpha x + y on the host
//------------------------------------------------------------------------------
static int CeedHostAXPY_Sycl(CeedScalar *y_array, CeedScalar alpha, CeedScalar *x_array, CeedInt length) {
  for (int i = 0; i < length; i++) y_array[i] += alpha * x_array[i];
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute y = alpha x + y on device
//------------------------------------------------------------------------------
static int CeedDeviceAXPY_Sycl(sycl::queue &sycl_queue, CeedScalar *y_array, CeedScalar alpha, CeedScalar *x_array, CeedInt length) {
  // Order queue
  sycl::event e = sycl_queue.ext_oneapi_submit_barrier();
  sycl_queue.parallel_for(length, {e}, [=](sycl::id<1> i) { y_array[i] += alpha * x_array[i]; });
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute y = alpha x + y
//------------------------------------------------------------------------------
static int CeedVectorAXPY_Sycl(CeedVector y, CeedScalar alpha, CeedVector x) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(y, &ceed));
  CeedVector_Sycl *y_impl, *x_impl;
  CeedCallBackend(CeedVectorGetData(y, &y_impl));
  CeedCallBackend(CeedVectorGetData(x, &x_impl));
  CeedSize length;
  CeedCallBackend(CeedVectorGetLength(y, &length));
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  // Set value for synced device/host array
  if (y_impl->d_array) {
    CeedCallBackend(CeedVectorSyncArray(x, CEED_MEM_DEVICE));
    CeedCallBackend(CeedDeviceAXPY_Sycl(data->sycl_queue, y_impl->d_array, alpha, x_impl->d_array, length));
  }
  if (y_impl->h_array) {
    CeedCallBackend(CeedVectorSyncArray(x, CEED_MEM_HOST));
    CeedCallBackend(CeedHostAXPY_Sycl(y_impl->h_array, alpha, x_impl->h_array, length));
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute the pointwise multiplication w = x .* y on the host
//------------------------------------------------------------------------------
static int CeedHostPointwiseMult_Sycl(CeedScalar *w_array, CeedScalar *x_array, CeedScalar *y_array, CeedInt length) {
  for (int i = 0; i < length; i++) w_array[i] = x_array[i] * y_array[i];
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute the pointwise multiplication w = x .* y on device (impl in .cu file)
//------------------------------------------------------------------------------
static int CeedDevicePointwiseMult_Sycl(sycl::queue &sycl_queue, CeedScalar *w_array, CeedScalar *x_array, CeedScalar *y_array, CeedInt length) {
  // Order queue
  sycl::event e = sycl_queue.ext_oneapi_submit_barrier();
  sycl_queue.parallel_for(length, {e}, [=](sycl::id<1> i) { w_array[i] = x_array[i] * y_array[i]; });
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute the pointwise multiplication w = x .* y
//------------------------------------------------------------------------------
static int CeedVectorPointwiseMult_Sycl(CeedVector w, CeedVector x, CeedVector y) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(w, &ceed));
  CeedVector_Sycl *w_impl, *x_impl, *y_impl;
  CeedCallBackend(CeedVectorGetData(w, &w_impl));
  CeedCallBackend(CeedVectorGetData(x, &x_impl));
  CeedCallBackend(CeedVectorGetData(y, &y_impl));
  CeedSize length;
  CeedCallBackend(CeedVectorGetLength(w, &length));
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  // Set value for synced device/host array
  if (!w_impl->d_array && !w_impl->h_array) {
    CeedCallBackend(CeedVectorSetValue(w, 0.0));
  }
  if (w_impl->d_array) {
    CeedCallBackend(CeedVectorSyncArray(x, CEED_MEM_DEVICE));
    CeedCallBackend(CeedVectorSyncArray(y, CEED_MEM_DEVICE));
    CeedCallBackend(CeedDevicePointwiseMult_Sycl(data->sycl_queue, w_impl->d_array, x_impl->d_array, y_impl->d_array, length));
  }
  if (w_impl->h_array) {
    CeedCallBackend(CeedVectorSyncArray(x, CEED_MEM_HOST));
    CeedCallBackend(CeedVectorSyncArray(y, CEED_MEM_HOST));
    CeedCallBackend(CeedHostPointwiseMult_Sycl(w_impl->h_array, x_impl->h_array, y_impl->h_array, length));
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy the vector
//------------------------------------------------------------------------------
static int CeedVectorDestroy_Sycl(const CeedVector vec) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedVector_Sycl *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  // Wait for all work to finish before freeing memory
  CeedCallSycl(ceed, data->sycl_queue.wait_and_throw());
  CeedCallSycl(ceed, sycl::free(impl->d_array_owned, data->sycl_context));
  CeedCallSycl(ceed, sycl::free(impl->reduction_norm, data->sycl_context));

  CeedCallBackend(CeedFree(&impl->h_array_owned));
  CeedCallBackend(CeedFree(&impl));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create a vector of the specified length (does not allocate memory)
//------------------------------------------------------------------------------
int CeedVectorCreate_Sycl(CeedSize n, CeedVector vec) {
  CeedVector_Sycl *impl;
  Ceed             ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallSycl(ceed, impl->reduction_norm = sycl::malloc_host<CeedScalar>(1, data->sycl_context));

  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Vector", vec, "HasValidArray", CeedVectorHasValidArray_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Vector", vec, "HasBorrowedArrayOfType", CeedVectorHasBorrowedArrayOfType_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Vector", vec, "SetArray", CeedVectorSetArray_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Vector", vec, "TakeArray", CeedVectorTakeArray_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Vector", vec, "SetValue", CeedVectorSetValue_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Vector", vec, "SyncArray", CeedVectorSyncArray_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Vector", vec, "GetArray", CeedVectorGetArray_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Vector", vec, "GetArrayRead", CeedVectorGetArrayRead_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Vector", vec, "GetArrayWrite", CeedVectorGetArrayWrite_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Vector", vec, "Norm", CeedVectorNorm_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Vector", vec, "Reciprocal", CeedVectorReciprocal_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Vector", vec, "AXPY", CeedVectorAXPY_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Vector", vec, "Scale", CeedVectorScale_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Vector", vec, "PointwiseMult", CeedVectorPointwiseMult_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Vector", vec, "Destroy", CeedVectorDestroy_Sycl));

  CeedCallBackend(CeedVectorSetData(vec, impl));

  return CEED_ERROR_SUCCESS;
}
