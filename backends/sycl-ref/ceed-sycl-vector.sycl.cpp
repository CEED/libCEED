// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
// #include <cublas_v2.h>
#include <cmath>
#include <string>

#include "ceed-sycl-ref.hpp"

//------------------------------------------------------------------------------
// Check if host/device sync is needed
//------------------------------------------------------------------------------
static inline int CeedVectorNeedSync_Sycl(const CeedVector vec, CeedMemType mem_type, bool *need_sync) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Sync host to device
//------------------------------------------------------------------------------
static inline int CeedVectorSyncH2D_Sycl(const CeedVector vec) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Sync device to host
//------------------------------------------------------------------------------
static inline int CeedVectorSyncD2H_Sycl(const CeedVector vec) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
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
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Check if CeedVector has any valid pointer
//------------------------------------------------------------------------------
static inline int CeedVectorHasValidArray_Sycl(const CeedVector vec, bool *has_valid_array) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Check if has array of given type
//------------------------------------------------------------------------------
static inline int CeedVectorHasArrayOfType_Sycl(const CeedVector vec, CeedMemType mem_type, bool *has_array_of_type) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Check if has borrowed array of given type
//------------------------------------------------------------------------------
static inline int CeedVectorHasBorrowedArrayOfType_Sycl(const CeedVector vec, CeedMemType mem_type, bool *has_borrowed_array_of_type) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Set array from host
//------------------------------------------------------------------------------
static int CeedVectorSetArrayHost_Sycl(const CeedVector vec, const CeedCopyMode copy_mode, CeedScalar *array) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Set array from device
//------------------------------------------------------------------------------
static int CeedVectorSetArrayDevice_Sycl(const CeedVector vec, const CeedCopyMode copy_mode, CeedScalar *array) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Set the array used by a vector,
//   freeing any previously allocated array if applicable
//------------------------------------------------------------------------------
static int CeedVectorSetArray_Sycl(const CeedVector vec, const CeedMemType mem_type, const CeedCopyMode copy_mode, CeedScalar *array) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
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
int CeedDeviceSetValue_Sycl(CeedScalar *d_array, CeedInt length, CeedScalar val) {
  return CeedError(NULL, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Set a vector to a value,
//------------------------------------------------------------------------------
static int CeedVectorSetValue_Sycl(CeedVector vec, CeedScalar val) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Vector Take Array
//------------------------------------------------------------------------------
static int CeedVectorTakeArray_Sycl(CeedVector vec, CeedMemType mem_type, CeedScalar **array) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Core logic for array syncronization for GetArray.
//   If a different memory type is most up to date, this will perform a copy
//------------------------------------------------------------------------------
static int CeedVectorGetArrayCore_Sycl(const CeedVector vec, const CeedMemType mem_type, CeedScalar **array) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
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
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Get write access to a vector via the specified mem_type
//------------------------------------------------------------------------------
static int CeedVectorGetArrayWrite_Sycl(const CeedVector vec, const CeedMemType mem_type, CeedScalar **array) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Get the norm of a CeedVector
//------------------------------------------------------------------------------
static int CeedVectorNorm_Sycl(CeedVector vec, CeedNormType type, CeedScalar *norm) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Take reciprocal of a vector on host
//------------------------------------------------------------------------------
static int CeedHostReciprocal_Sycl(CeedScalar *h_array, CeedInt length) {
  for (int i = 0; i < length; i++) {
    if (fabs(h_array[i]) > CEED_EPSILON) h_array[i] = 1. / h_array[i];
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Take reciprocal of a vector on device
//------------------------------------------------------------------------------
int CeedDeviceReciprocal_Sycl(CeedScalar *d_array, CeedInt length) {
  return CeedError(NULL, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Take reciprocal of a vector
//------------------------------------------------------------------------------
static int CeedVectorReciprocal_Sycl(CeedVector vec) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
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
int CeedDeviceScale_Sycl(CeedScalar *x_array, CeedScalar alpha, CeedInt length) {
  return CeedError(NULL, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Compute x = alpha x
//------------------------------------------------------------------------------
static int CeedVectorScale_Sycl(CeedVector x, CeedScalar alpha) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(x, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
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
int CeedDeviceAXPY_Sycl(CeedScalar *y_array, CeedScalar alpha, CeedScalar *x_array, CeedInt length) {
  return CeedError(NULL, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Compute y = alpha x + y
//------------------------------------------------------------------------------
static int CeedVectorAXPY_Sycl(CeedVector y, CeedScalar alpha, CeedVector x) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(y, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
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
int CeedDevicePointwiseMult_Sycl(CeedScalar *w_array, CeedScalar *x_array, CeedScalar *y_array, CeedInt length) {
  return CeedError(NULL, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Compute the pointwise multiplication w = x .* y
//------------------------------------------------------------------------------
static int CeedVectorPointwiseMult_Sycl(CeedVector w, CeedVector x, CeedVector y) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(w, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Destroy the vector
//------------------------------------------------------------------------------
static int CeedVectorDestroy_Sycl(const CeedVector vec) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Create a vector of the specified length (does not allocate memory)
//------------------------------------------------------------------------------
int CeedVectorCreate_Sycl(CeedSize n, CeedVector vec) {
  CeedVector_Sycl *impl;
  Ceed             ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}
