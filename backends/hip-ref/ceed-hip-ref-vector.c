// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <hip/hip_runtime.h>
#include <hipblas.h>
#include <math.h>
#include <string.h>
#include "ceed-hip-ref.h"


//------------------------------------------------------------------------------
// Get size of the scalar type
// TODO: move to interface level for all backends?
//------------------------------------------------------------------------------
static inline int CeedScalarTypeGetSize_Hip(Ceed ceed, CeedScalarType prec,
    size_t *size) {
  switch(prec) {
  case CEED_SCALAR_FP32:
    *size = sizeof(float);
    break;
  case CEED_SCALAR_FP64:
    *size = sizeof(double);
    break;
  default:
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Invalid scalar precision type specified");
    // LCOV_EXCL_STOP
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get info about the current status of the different precisions in the
// valid, borrowed, and owned arrays, for a specific mem_type
//------------------------------------------------------------------------------
static int CeedVectorCheckArrayStatus_Hip(CeedVector vec,
    CeedMemType mem_type,
    unsigned int *valid_status,
    unsigned int *borrowed_status,
    unsigned int *owned_status) {

  int ierr;
  CeedVector_Hip *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);
  *valid_status = 0;
  *borrowed_status = 0;
  *owned_status = 0;
  switch(mem_type) {
  case CEED_MEM_HOST:
    for (int i = 0; i < CEED_NUM_PRECISIONS; i++) {
      if (!!impl->h_array.values[i])
        *valid_status += 1 << i;
      if (!!impl->h_array_borrowed.values[i])
        *borrowed_status += 1 << i;
      if (!!impl->h_array_owned.values[i])
        *owned_status += 1 << i;
    }
    break;
  case CEED_MEM_DEVICE:
    for (int i = 0; i < CEED_NUM_PRECISIONS; i++) {
      if (!!impl->d_array.values[i])
        *valid_status += 1 << i;
      if (!!impl->d_array_borrowed.values[i])
        *borrowed_status += 1 << i;
      if (!!impl->d_array_owned.values[i])
        *owned_status += 1 << i;
    }
    break;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set all pointers as invalid
//------------------------------------------------------------------------------
static inline int CeedVectorSetAllInvalid_Hip(const CeedVector vec) {
  int ierr;
  CeedVector_Hip *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  for (int i = 0; i < CEED_NUM_PRECISIONS; i++) {
    impl->h_array.values[i] = NULL;
    impl->d_array.values[i] = NULL;
  }

  return CEED_ERROR_SUCCESS;
}


//------------------------------------------------------------------------------
// Check if CeedVector has any valid pointers
//------------------------------------------------------------------------------
static inline int CeedVectorHasValidArray_Hip(const CeedVector vec,
    bool *has_valid_array) {
  int ierr;
  CeedVector_Hip *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  *has_valid_array = false;
  for (int i = 0; i < CEED_NUM_PRECISIONS; i++) {
    *has_valid_array = *has_valid_array ||
                       (!!impl->h_array.values[i] || !!impl->d_array.values[i]);
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if has valid array of given memory type
//------------------------------------------------------------------------------
static inline int CeedVectorHasValidArrayOfMemType_Hip(const CeedVector vec,
    CeedMemType mem_type, bool *has_valid_array_of_mem_type) {
  int ierr;
  CeedVector_Hip *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  *has_valid_array_of_mem_type = false;
  switch (mem_type) {
  case CEED_MEM_HOST:
    for (int i = 0; i < CEED_NUM_PRECISIONS; i++) {
      *has_valid_array_of_mem_type = *has_valid_array_of_mem_type ||
                                     !!impl->h_array.values[i];
    }
    break;
  case CEED_MEM_DEVICE:
    for (int i = 0; i < CEED_NUM_PRECISIONS; i++) {
      *has_valid_array_of_mem_type = *has_valid_array_of_mem_type ||
                                     !!impl->d_array.values[i];
    }
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if has any array of given type
//------------------------------------------------------------------------------
static inline int CeedVectorHasArrayOfType_Hip(const CeedVector vec,
    CeedMemType mem_type, CeedScalarType prec, bool *has_array_of_type) {
  int ierr;
  CeedVector_Hip *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  switch (mem_type) {
  case CEED_MEM_HOST:
    *has_array_of_type = !!impl->h_array_borrowed.values[prec] ||
                         !!impl->h_array_owned.values[prec];
    break;
  case CEED_MEM_DEVICE:
    *has_array_of_type = !!impl->d_array_borrowed.values[prec] ||
                         !!impl->d_array_owned.values[prec];
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if has borrowed array of given type
//------------------------------------------------------------------------------
static inline int CeedVectorHasBorrowedArrayOfType_Hip(const CeedVector vec,
    CeedMemType mem_type, CeedScalarType prec,
    bool *has_borrowed_array_of_type) {
  int ierr;
  CeedVector_Hip *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  switch (mem_type) {
  case CEED_MEM_HOST:
    *has_borrowed_array_of_type = !!impl->h_array_borrowed.values[prec];
    break;
  case CEED_MEM_DEVICE:
    *has_borrowed_array_of_type = !!impl->d_array_borrowed.values[prec];
    break;
  }

  return CEED_ERROR_SUCCESS;
}


//------------------------------------------------------------------------------
// Return the scalar type of the valid array on mem_type, or the "preferred
//   precision" for copying, if more than one precision is valid. If no
//   precisions are valid on the specified mem_type, it will return
//   CEED_SCALAR_TYPE (default precision); you should check for a valid array
//   separately.
//------------------------------------------------------------------------------
static inline int CeedVectorGetPrecision_Hip(const CeedVector vec,
    const CeedMemType mem_type, CeedScalarType *preferred_precision) {

  int ierr;
  CeedVector_Hip *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  *preferred_precision = CEED_SCALAR_TYPE;
  // Check for valid precisions, from most to least precise precise (we want
  // the most precision if multiple arrays are valid)
  switch (mem_type) {
  case CEED_MEM_HOST:
    if (!!impl->h_array.values[CEED_SCALAR_FP64])
      *preferred_precision = CEED_SCALAR_FP64;
    else if (!!impl->h_array.values[CEED_SCALAR_FP32])
      *preferred_precision = CEED_SCALAR_FP32;
    break;
  case CEED_MEM_DEVICE:
    if (!!impl->d_array.values[CEED_SCALAR_FP64])
      *preferred_precision = CEED_SCALAR_FP64;
    else if (!!impl->d_array.values[CEED_SCALAR_FP32])
      *preferred_precision = CEED_SCALAR_FP32;
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Convert vector's host array to new precision.
//------------------------------------------------------------------------------
static int CeedVectorConvertArrayHost_Hip(CeedVector vec,
    const CeedScalarType from_prec, const CeedScalarType to_prec) {
  CeedInt ierr;
  CeedSize length;
  ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);

  switch (from_prec) {

  case CEED_SCALAR_FP64:
    switch (to_prec) {
    case CEED_SCALAR_FP64:
      // No conversion needed
      break;
    case CEED_SCALAR_FP32:
      if (!data->h_array.values[CEED_SCALAR_FP32]) {
        // Use borrowed memory, if we have it for this precision
        if (data->h_array_borrowed.values[CEED_SCALAR_FP32]) {
          data->h_array.values[CEED_SCALAR_FP32] =
            data->h_array_borrowed.values[CEED_SCALAR_FP32];
        } else {
          // Use owned memory
          if (!data->h_array_owned.values[CEED_SCALAR_FP32]) {
            ierr = CeedMalloc(length,
                              (float **) &data->h_array_owned.values[CEED_SCALAR_FP32]);
            CeedChkBackend(ierr);
          }
          data->h_array.values[CEED_SCALAR_FP32] =
            data->h_array_owned.values[CEED_SCALAR_FP32];
        }
      }
      float *float_data = (float *) data->h_array.values[CEED_SCALAR_FP32];
      double *double_data = (double *) data->h_array.values[CEED_SCALAR_FP64];
      for (int i = 0; i < length; i++)
        float_data[i] = (float) double_data[i];
      break;
    }
    break;

  case CEED_SCALAR_FP32:
    switch (to_prec) {
    case CEED_SCALAR_FP64:
      if (!data->h_array.values[CEED_SCALAR_FP64]) {
        // Use borrowed memory, if we have it for this precision
        if (data->h_array_borrowed.values[CEED_SCALAR_FP64]) {
          data->h_array.values[CEED_SCALAR_FP64] =
            data->h_array_borrowed.values[CEED_SCALAR_FP64];
        } else {
          // Use owned memory
          if (!data->h_array_owned.values[CEED_SCALAR_FP64]) {
            ierr = CeedMalloc(length,
                              (double **) &data->h_array_owned.values[CEED_SCALAR_FP64]);
            CeedChkBackend(ierr);
          }
          data->h_array.values[CEED_SCALAR_FP64] =
            data->h_array_owned.values[CEED_SCALAR_FP64];
        }
      }
      float *float_data = (float *) data->h_array.values[CEED_SCALAR_FP32];
      double *double_data = (double *) data->h_array.values[CEED_SCALAR_FP64];
      for (int i = 0; i < length; i++)
        double_data[i] = (double) float_data[i];
      break;
    case CEED_SCALAR_FP32:
      // No conversion needed
      break;
    }
    break;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Convert a double-precision array to single precision
//------------------------------------------------------------------------------
int CeedDeviceConvertArray_Hip_Fp64_Fp32(CeedInt length,
    double *double_data, float *float_data);

//------------------------------------------------------------------------------
// Convert a single-precision array to double precision
//------------------------------------------------------------------------------
int CeedDeviceConvertArray_Hip_Fp32_Fp64(CeedInt length,
    float *float_data, double *double_data);

//------------------------------------------------------------------------------
// Convert device array to new precision(impl of individual functions/kernels in
// .hip.cpp file)
//------------------------------------------------------------------------------
static int CeedVectorConvertArrayDevice_Hip(CeedVector vec,
    const CeedScalarType from_prec, const CeedScalarType to_prec) {

  CeedSize length;
  CeedInt ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);
  switch (from_prec) {

  case CEED_SCALAR_FP64:
    switch (to_prec) {
    case CEED_SCALAR_FP64:
      // No conversion needed
      break;
    case CEED_SCALAR_FP32:
      if (!data->d_array.values[CEED_SCALAR_FP32]) {
        // Use borrowed memory, if we have it for this precision
        if (data->d_array_borrowed.values[CEED_SCALAR_FP32]) {
          data->d_array.values[CEED_SCALAR_FP32] =
            data->d_array_borrowed.values[CEED_SCALAR_FP32];
        } else {
          // Use owned memory
          if (!data->d_array_owned.values[CEED_SCALAR_FP32]) {
            size_t bytes = length * sizeof(float);
            ierr = hipMalloc((void **)&data->d_array_owned.values[CEED_SCALAR_FP32],
                             bytes);
            CeedChk_Hip(ceed, ierr);
          }
          data->d_array.values[CEED_SCALAR_FP32] =
            data->d_array_owned.values[CEED_SCALAR_FP32];
        }
      }
      ierr = CeedDeviceConvertArray_Hip_Fp64_Fp32(length,
             (double *) data->d_array.values[CEED_SCALAR_FP64],
             (float *) data->d_array.values[CEED_SCALAR_FP32]);
      CeedChkBackend(ierr);
      break;
    }
    break;

  case CEED_SCALAR_FP32:
    switch (to_prec) {
    case CEED_SCALAR_FP64:
      if (!data->d_array.values[CEED_SCALAR_FP64]) {
        // Use borrowed memory, if we have it for this precision
        if (data->d_array_borrowed.values[CEED_SCALAR_FP64]) {
          data->d_array.values[CEED_SCALAR_FP64] =
            data->d_array_borrowed.values[CEED_SCALAR_FP64];
        } else {
          // Use owned memory
          if (!data->d_array_owned.values[CEED_SCALAR_FP64]) {
            size_t bytes = length * sizeof(double);
            ierr = hipMalloc((void **)&data->d_array_owned.values[CEED_SCALAR_FP64],
                             bytes);
            CeedChk_Hip(ceed, ierr);
          }
          data->d_array.values[CEED_SCALAR_FP64] =
            data->d_array_owned.values[CEED_SCALAR_FP64];
        }
      }
      ierr = CeedDeviceConvertArray_Hip_Fp32_Fp64(length,
             (float *) data->d_array.values[CEED_SCALAR_FP32],
             (double *) data->d_array.values[CEED_SCALAR_FP64]);
      CeedChkBackend(ierr);
      break;
    case CEED_SCALAR_FP32:
      // No conversion needed
      break;
    }
    break;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Convert data array from one precision to another (through copy/cast).
//------------------------------------------------------------------------------
static int CeedVectorConvertArray_Hip(CeedVector vec,
                                      const CeedMemType mem_type,
                                      const CeedScalarType from_prec,
                                      const CeedScalarType to_prec) {

  switch (mem_type) {
  case CEED_MEM_HOST: return CeedVectorConvertArrayHost_Hip(vec, from_prec,
                               to_prec);
  case CEED_MEM_DEVICE: return CeedVectorConvertArrayDevice_Hip(vec, from_prec,
                                 to_prec);
  }
  return CEED_ERROR_UNSUPPORTED;
}

//------------------------------------------------------------------------------
// Sync host to device
//------------------------------------------------------------------------------
static inline int CeedVectorSyncH2D_Hip(const CeedVector vec,
                                        const CeedScalarType prec) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Hip *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  CeedSize length;
  ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
  size_t prec_size;
  ierr = CeedScalarTypeGetSize_Hip(ceed, prec, &prec_size);
  CeedChkBackend(ierr);
  size_t bytes = length * prec_size;

  if (!impl->h_array.values[prec])
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "No valid host data to sync to device");
  // LCOV_EXCL_STOP

  if (impl->d_array_borrowed.values[prec]) {
    impl->d_array.values[prec] = impl->d_array_borrowed.values[prec];
  } else if (impl->d_array_owned.values[prec]) {
    impl->d_array.values[prec] = impl->d_array_owned.values[prec];
  } else {
    ierr = hipMalloc((void **)&impl->d_array_owned.values[prec], bytes);
    CeedChk_Hip(ceed, ierr);
    impl->d_array.values[prec] = impl->d_array_owned.values[prec];
  }

  ierr = hipMemcpy(impl->d_array.values[prec], impl->h_array.values[prec],
                   bytes, hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync device to host
//------------------------------------------------------------------------------
static inline int CeedVectorSyncD2H_Hip(const CeedVector vec,
                                        const CeedScalarType prec) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Hip *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  if (!impl->d_array.values[prec])
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "No valid device data to sync to host");
  // LCOV_EXCL_STOP

  CeedSize length;
  ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
  size_t prec_size;
  ierr = CeedScalarTypeGetSize_Hip(ceed, prec, &prec_size);
  CeedChkBackend(ierr);
  size_t bytes = length * prec_size;

  if (impl->h_array_borrowed.values[prec]) {
    impl->h_array.values[prec] = impl->h_array_borrowed.values[prec];
  } else if (impl->h_array_owned.values[prec]) {
    impl->h_array.values[prec] = impl->h_array_owned.values[prec];
  } else {
    CeedSize length;
    ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
    ierr = CeedCallocArray(length, prec_size,&impl->h_array_owned.values[prec]);
    CeedChkBackend(ierr);
    impl->h_array.values[prec] = impl->h_array_owned.values[prec];
  }

  ierr = hipMemcpy(impl->h_array.values[prec], impl->d_array.values[prec],
                   bytes, hipMemcpyDeviceToHost); CeedChk_Hip(ceed, ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if the only current valid array is on another MemType than mem_type
//------------------------------------------------------------------------------
static inline int CeedVectorNeedSync_Hip(const CeedVector vec,
    CeedMemType mem_type, bool *need_sync) {
  int ierr;
  CeedVector_Hip *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  bool has_valid_array = false;
  ierr = CeedVectorHasValidArray(vec, &has_valid_array); CeedChkBackend(ierr);
  bool has_valid_array_of_mem_type = false;
  ierr = CeedVectorHasValidArrayOfMemType_Hip(vec, mem_type,
         &has_valid_array_of_mem_type);
  CeedChkBackend(ierr);

  // Check if we have a valid array, but not for the correct memory type
  *need_sync = has_valid_array && !has_valid_array_of_mem_type;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync arrays between host and device
//------------------------------------------------------------------------------
static int CeedVectorSyncArrayGeneric_Hip(const CeedVector vec,
    CeedMemType mem_type,
    CeedScalarType prec) {

  int ierr;

  // Check whether device/host sync is needed
  bool need_sync = false;
  ierr = CeedVectorNeedSync_Hip(vec, mem_type, &need_sync);
  CeedChkBackend(ierr);
  if (!need_sync)
    return CEED_ERROR_SUCCESS;

  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedMemType source_mem_type = CEED_MEM_HOST;
  if (mem_type == CEED_MEM_HOST) source_mem_type = CEED_MEM_DEVICE;
  // Sync array to requested mem_type
  // Figure out which current precision we have to convert from
  CeedScalarType source_cur_prec;
  ierr = CeedVectorGetPrecision_Hip(vec, source_mem_type, &source_cur_prec);
  CeedChkBackend(ierr);
  bool need_convert = false;
  CeedScalarType sync_prec = prec;
  if (source_cur_prec != prec) {
    size_t cur_prec_size, prec_size;
    ierr = CeedScalarTypeGetSize_Hip(ceed, source_cur_prec, &cur_prec_size);
    CeedChkBackend(ierr);
    size_t ierr = CeedScalarTypeGetSize_Hip(ceed, prec, &prec_size);
    CeedChkBackend(ierr);

    // If the size of the current precision's data type is greater than
    // the destination precision, we want to convert first, then sync,
    // to reduce size of memory movement between host and device
    if (cur_prec_size > prec_size) {
      ierr = CeedVectorConvertArray_Hip(vec, source_mem_type, source_cur_prec, prec);
      CeedChkBackend(ierr);
    }
    // Else, we will sync first, then convert
    else {
      sync_prec = source_cur_prec;
      need_convert = true;
    }
  }

  // Perform sync between host and device in destination precision
  switch (mem_type) {
  case CEED_MEM_HOST:
    ierr = CeedVectorSyncD2H_Hip(vec, sync_prec); CeedChkBackend(ierr);
    break;
  case CEED_MEM_DEVICE:
    ierr = CeedVectorSyncH2D_Hip(vec, sync_prec); CeedChkBackend(ierr);
    break;
  default:
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Invalid memory type specified");
    // LCOV_EXCL_STOP
  }

  // Perform conversion, if still necessary
  if (need_convert) {
    ierr = CeedVectorConvertArray_Hip(vec, mem_type, source_cur_prec, prec);
    CeedChkBackend(ierr);
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set array from host
//------------------------------------------------------------------------------
static int CeedVectorSetArrayHost_Hip(const CeedVector vec,
                                      const CeedScalarType prec,
                                      const CeedCopyMode copy_mode, void *array) {
  int ierr;
  CeedVector_Hip *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  size_t prec_size;
  ierr = CeedScalarTypeGetSize_Hip(ceed, prec, &prec_size);
  CeedChkBackend(ierr);

  switch (copy_mode) {
  case CEED_COPY_VALUES: {
    CeedSize length;
    if (!impl->h_array_owned.values[prec]) {
      ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
      ierr = CeedMallocArray(length, prec_size, &impl->h_array_owned.values[prec]);
      CeedChkBackend(ierr);
    }
    impl->h_array_borrowed.values[prec] = NULL;
    impl->h_array.values[prec] = impl->h_array_owned.values[prec];
    if (array) {
      CeedSize length;
      ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
      size_t bytes = length * prec_size;
      memcpy(impl->h_array.values[prec], array, bytes);
    }
  } break;
  case CEED_OWN_POINTER:
    ierr = CeedFree(&impl->h_array_owned.values[prec]); CeedChkBackend(ierr);
    impl->h_array_owned.values[prec] = array;
    impl->h_array_borrowed.values[prec] = NULL;
    impl->h_array.values[prec] = array;
    break;
  case CEED_USE_POINTER:
    ierr = CeedFree(&impl->h_array_owned.values[prec]); CeedChkBackend(ierr);
    impl->h_array_borrowed.values[prec] = array;
    impl->h_array.values[prec] = array;
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set array from device
//------------------------------------------------------------------------------
static int CeedVectorSetArrayDevice_Hip(const CeedVector vec,
                                        const CeedScalarType prec,
                                        const CeedCopyMode copy_mode, void *array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Hip *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  size_t prec_size;
  ierr = CeedScalarTypeGetSize_Hip(ceed, prec, &prec_size);
  CeedChkBackend(ierr);

  switch (copy_mode) {
  case CEED_COPY_VALUES: {
    CeedSize length;
    ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
    size_t bytes = length * prec_size;
    if (!impl->d_array_owned.values[prec]) {
      ierr = hipMalloc((void **)&impl->d_array_owned.values[prec], bytes);
      CeedChk_Hip(ceed, ierr);
    }
    impl->d_array_borrowed.values[prec] = NULL;
    impl->d_array.values[prec] = impl->d_array_owned.values[prec];
    if (array) {
      ierr = hipMemcpy(impl->d_array.values[prec], array, bytes,
                       hipMemcpyDeviceToDevice); CeedChk_Hip(ceed, ierr);
    }
  } break;
  case CEED_OWN_POINTER:
    ierr = hipFree(impl->d_array_owned.values[prec]); CeedChk_Hip(ceed, ierr);
    impl->d_array_owned.values[prec] = array;
    impl->d_array_borrowed.values[prec] = NULL;
    impl->d_array.values[prec] = array;
    break;
  case CEED_USE_POINTER:
    ierr = hipFree(impl->d_array_owned.values[prec]); CeedChk_Hip(ceed, ierr);
    impl->d_array_owned.values[prec] = NULL;
    impl->d_array_borrowed.values[prec] = array;
    impl->d_array.values[prec] = array;
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set the array used by a vector,
//   freeing any previously allocated array if applicable
//------------------------------------------------------------------------------
static int CeedVectorSetArrayGeneric_Hip(const CeedVector vec,
    const CeedMemType mem_type,
    const CeedScalarType prec,
    const CeedCopyMode copy_mode, void *array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Hip *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  ierr = CeedVectorSetAllInvalid_Hip(vec); CeedChkBackend(ierr);
  switch (mem_type) {
  case CEED_MEM_HOST:
    return CeedVectorSetArrayHost_Hip(vec, prec, copy_mode, array);
  case CEED_MEM_DEVICE:
    return CeedVectorSetArrayDevice_Hip(vec, prec, copy_mode, array);
  }

  return CEED_ERROR_UNSUPPORTED;
}

//------------------------------------------------------------------------------
// Set host array to value
//------------------------------------------------------------------------------
static int CeedHostSetValue_Hip(CeedScalar *h_array, CeedInt length,
                                CeedScalar val) {
  for (int i = 0; i < length; i++)
    h_array[i] = val;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set device array to value (impl in .hip file)
//------------------------------------------------------------------------------
int CeedDeviceSetValue_Hip(CeedScalar *d_array, CeedInt length, CeedScalar val);

//------------------------------------------------------------------------------
// Set a vector to a value
//------------------------------------------------------------------------------
static int CeedVectorSetValue_Hip(CeedVector vec, CeedScalar val) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Hip *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);
  CeedSize length;
  ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);

  // Set value for synced device/host array
  if (!impl->d_array.values[CEED_SCALAR_TYPE] &&
      !impl->h_array.values[CEED_SCALAR_TYPE]) {
    if (impl->d_array_borrowed.values[CEED_SCALAR_TYPE]) {
      impl->d_array.values[CEED_SCALAR_TYPE] =
        impl->d_array_borrowed.values[CEED_SCALAR_TYPE];
    } else if (impl->h_array_borrowed.values[CEED_SCALAR_TYPE]) {
      impl->h_array.values[CEED_SCALAR_TYPE] =
        impl->h_array_borrowed.values[CEED_SCALAR_TYPE];
    } else if (impl->d_array_owned.values[CEED_SCALAR_TYPE]) {
      impl->d_array.values[CEED_SCALAR_TYPE] =
        impl->d_array_owned.values[CEED_SCALAR_TYPE];
    } else if (impl->h_array_owned.values[CEED_SCALAR_TYPE]) {
      impl->h_array.values[CEED_SCALAR_TYPE] =
        impl->h_array_owned.values[CEED_SCALAR_TYPE];
    } else {
      ierr = CeedVectorSetArray(vec, CEED_MEM_DEVICE, CEED_COPY_VALUES, NULL);
      CeedChkBackend(ierr);
    }
  }
  if (impl->d_array.values[CEED_SCALAR_TYPE]) {
    ierr = CeedDeviceSetValue_Hip(impl->d_array.values[CEED_SCALAR_TYPE], length,
                                  val);
    CeedChkBackend(ierr);
  }
  if (impl->h_array.values[CEED_SCALAR_TYPE]) {
    ierr = CeedHostSetValue_Hip(impl->h_array.values[CEED_SCALAR_TYPE], length,
                                val);
    CeedChkBackend(ierr);
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Take Array
//------------------------------------------------------------------------------
static int CeedVectorTakeArrayGeneric_Hip(CeedVector vec, CeedMemType mem_type,
    CeedScalarType prec,
    void **array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Hip *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  // Sync host/device (if necessary, otherwise the function will return)
  ierr = CeedVectorSyncArrayGeneric_Hip(vec, mem_type, prec);
  CeedChkBackend(ierr);

  // Check if we need to convert from another precision
  CeedScalarType cur_prec;
  ierr = CeedVectorGetPrecision_Hip(vec, mem_type, &cur_prec);
  CeedChkBackend(ierr);
  if (cur_prec != prec) {
    ierr = CeedVectorConvertArray_Hip(vec, mem_type, cur_prec, prec);
    CeedChkBackend(ierr);
  }

  // Update pointer
  switch (mem_type) {
  case CEED_MEM_HOST:
    (*array) = impl->h_array_borrowed.values[prec];
    impl->h_array_borrowed.values[prec] = NULL;
    impl->h_array.values[prec] = NULL;
    break;
  case CEED_MEM_DEVICE:
    (*array) = impl->d_array_borrowed.values[prec];
    impl->d_array_borrowed.values[prec] = NULL;
    impl->d_array.values[prec] = NULL;
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Core logic for array syncronization for GetArray.
//   If a different memory type is most up to date, this will perform a copy
//------------------------------------------------------------------------------
static int CeedVectorGetArrayCore_Hip(const CeedVector vec,
                                      const CeedMemType mem_type,
                                      const CeedScalarType prec,
                                      void **array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Hip *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  // Sync host/device (if necessary, otherwise the function will return)
  ierr = CeedVectorSyncArrayGeneric_Hip(vec, mem_type, prec);
  CeedChkBackend(ierr);

  // Check if we need to convert from another precision
  CeedScalarType cur_prec;
  ierr = CeedVectorGetPrecision_Hip(vec, mem_type, &cur_prec);
  CeedChkBackend(ierr);
  if (cur_prec != prec) {
    ierr = CeedVectorConvertArray_Hip(vec, mem_type, cur_prec, prec);
    CeedChkBackend(ierr);
  }

  // Update pointer
  switch (mem_type) {
  case CEED_MEM_HOST:
    *array = impl->h_array.values[prec];
    break;
  case CEED_MEM_DEVICE:
    *array = impl->d_array.values[prec];
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get read-only access to a vector via the specified mem_type and precision
//------------------------------------------------------------------------------
static int CeedVectorGetArrayReadGeneric_Hip(const CeedVector vec,
    const CeedMemType mem_type,
    const CeedScalarType prec,
    const void **array) {
  return CeedVectorGetArrayCore_Hip(vec, mem_type, prec, (void **)array);
}

//------------------------------------------------------------------------------
// Get read/write access to a vector via the specified mem_type
//------------------------------------------------------------------------------
static int CeedVectorGetArrayGeneric_Hip(const CeedVector vec,
    const CeedMemType mem_type,
    const CeedScalarType prec,
    void **array) {
  int ierr;
  CeedVector_Hip *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  ierr = CeedVectorGetArrayCore_Hip(vec, mem_type, prec, array);
  CeedChkBackend(ierr);

  ierr = CeedVectorSetAllInvalid_Hip(vec); CeedChkBackend(ierr);
  switch (mem_type) {
  case CEED_MEM_HOST:
    impl->h_array.values[prec] = *array;
    break;
  case CEED_MEM_DEVICE:
    impl->d_array.values[prec] = *array;
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get write access to a vector via the specified mem_type and precision
//------------------------------------------------------------------------------
static int CeedVectorGetArrayWriteGeneric_Hip(const CeedVector vec,
    const CeedMemType mem_type,
    const CeedScalarType prec,
    void **array) {
  int ierr;
  CeedVector_Hip *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  bool has_array_of_type = true;
  ierr = CeedVectorHasArrayOfType_Hip(vec, mem_type, prec, &has_array_of_type);
  CeedChkBackend(ierr);
  if (!has_array_of_type) {
    // Allocate if array is not yet allocated
    ierr = CeedVectorSetArrayGeneric(vec, mem_type, prec, CEED_COPY_VALUES, NULL);
    CeedChkBackend(ierr);
  } else {
    // Select dirty array
    switch (mem_type) {
    case CEED_MEM_HOST:
      if (impl->h_array_borrowed.values[prec])
        impl->h_array.values[prec] = impl->h_array_borrowed.values[prec];
      else
        impl->h_array.values[prec] = impl->h_array_owned.values[prec];
      break;
    case CEED_MEM_DEVICE:
      if (impl->d_array_borrowed.values[prec])
        impl->d_array.values[prec] = impl->d_array_borrowed.values[prec];
      else
        impl->d_array.values[prec] = impl->d_array_owned.values[prec];
    }
  }

  return CeedVectorGetArrayGeneric_Hip(vec, mem_type, prec, array);
}

//------------------------------------------------------------------------------
// Get the norm of a CeedVector
//------------------------------------------------------------------------------
static int CeedVectorNorm_Hip(CeedVector vec, CeedNormType type,
                              CeedScalar *norm) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Hip *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);
  CeedSize length;
  ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
  hipblasHandle_t handle;
  ierr = CeedHipGetHipblasHandle(ceed, &handle); CeedChkBackend(ierr);

  // Compute norm
  const CeedScalar *d_array;
  ierr = CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE, &d_array);
  CeedChkBackend(ierr);
  switch (type) {
  case CEED_NORM_1: {
    if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
      ierr = hipblasSasum(handle, length, (float *) d_array, 1, (float *) norm);
    } else {
      ierr = hipblasDasum(handle, length, (double *) d_array, 1, (double *) norm);
    }
    CeedChk_Hipblas(ceed, ierr);
    break;
  }
  case CEED_NORM_2: {
    if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
      ierr = hipblasSnrm2(handle, length, (float *) d_array, 1, (float *) norm);
    } else {
      ierr = hipblasDnrm2(handle, length, (double *) d_array, 1, (double *) norm);
    }
    CeedChk_Hipblas(ceed, ierr);
    break;
  }
  case CEED_NORM_MAX: {
    CeedInt indx;
    if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
      ierr = hipblasIsamax(handle, length, (float *) d_array, 1, &indx);
    } else {
      ierr = hipblasIdamax(handle, length, (double *) d_array, 1, &indx);
    }
    CeedChk_Hipblas(ceed, ierr);
    CeedScalar normNoAbs;
    ierr = hipMemcpy(&normNoAbs,
                     (CeedScalar *)(impl->d_array.values[CEED_SCALAR_TYPE])+indx-1,
                     sizeof(CeedScalar),
                     hipMemcpyDeviceToHost); CeedChk_Hip(ceed, ierr);
    *norm = fabs(normNoAbs);
    break;
  }
  }
  ierr = CeedVectorRestoreArrayRead(vec, &d_array); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Take reciprocal of a vector on host
//------------------------------------------------------------------------------
static int CeedHostReciprocal_Hip(CeedScalar *h_array, CeedInt length) {
  for (int i = 0; i < length; i++)
    if (fabs(h_array[i]) > CEED_EPSILON)
      h_array[i] = 1./h_array[i];
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
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Hip *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);
  CeedSize length;
  ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);

  // Set value for synced device/host array
  if (impl->d_array.values[CEED_SCALAR_TYPE]) {
    ierr = CeedDeviceReciprocal_Hip((CeedScalar *)
                                    impl->d_array.values[CEED_SCALAR_TYPE],
                                    length); CeedChkBackend(ierr);
  }
  if (impl->h_array.values[CEED_SCALAR_TYPE]) {
    ierr = CeedHostReciprocal_Hip((CeedScalar *)
                                  impl->h_array.values[CEED_SCALAR_TYPE],
                                  length); CeedChkBackend(ierr);
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute x = alpha x on the host
//------------------------------------------------------------------------------
static int CeedHostScale_Hip(CeedScalar *x_array, CeedScalar alpha,
                             CeedInt length) {
  for (int i = 0; i < length; i++)
    x_array[i] *= alpha;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute x = alpha x on device (impl in .cu file)
//------------------------------------------------------------------------------
int CeedDeviceScale_Hip(CeedScalar *x_array, CeedScalar alpha,
                        CeedInt length);

//------------------------------------------------------------------------------
// Compute x = alpha x
//------------------------------------------------------------------------------
static int CeedVectorScale_Hip(CeedVector x, CeedScalar alpha) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(x, &ceed); CeedChkBackend(ierr);
  CeedVector_Hip *x_impl;
  ierr = CeedVectorGetData(x, &x_impl); CeedChkBackend(ierr);
  CeedSize length;
  ierr = CeedVectorGetLength(x, &length); CeedChkBackend(ierr);

  // Set value for synced device/host array
  if (x_impl->d_array.values[CEED_SCALAR_TYPE]) {
    ierr = CeedDeviceScale_Hip((CeedScalar *)
                               x_impl->d_array.values[CEED_SCALAR_TYPE],
                               alpha, length);
    CeedChkBackend(ierr);
  }
  if (x_impl->h_array.values[CEED_SCALAR_TYPE]) {
    ierr = CeedHostScale_Hip((CeedScalar *)
                             x_impl->h_array.values[CEED_SCALAR_TYPE],
                             alpha, length); CeedChkBackend(ierr);
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute y = alpha x + y on the host
//------------------------------------------------------------------------------
static int CeedHostAXPY_Hip(CeedScalar *y_array, CeedScalar alpha,
                            CeedScalar *x_array, CeedInt length) {
  for (int i = 0; i < length; i++)
    y_array[i] += alpha * x_array[i];
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute y = alpha x + y on device (impl in .cu file)
//------------------------------------------------------------------------------
int CeedDeviceAXPY_Hip(CeedScalar *y_array, CeedScalar alpha,
                       CeedScalar *x_array, CeedInt length);

//------------------------------------------------------------------------------
// Compute y = alpha x + y
//------------------------------------------------------------------------------
static int CeedVectorAXPY_Hip(CeedVector y, CeedScalar alpha, CeedVector x) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(y, &ceed); CeedChkBackend(ierr);
  CeedVector_Hip *y_impl, *x_impl;
  ierr = CeedVectorGetData(y, &y_impl); CeedChkBackend(ierr);
  ierr = CeedVectorGetData(x, &x_impl); CeedChkBackend(ierr);
  CeedSize length;
  ierr = CeedVectorGetLength(y, &length); CeedChkBackend(ierr);

  // Set value for synced device/host array
  if (y_impl->d_array.values[CEED_SCALAR_TYPE]) {
    ierr = CeedVectorSyncArray(x, CEED_MEM_DEVICE); CeedChkBackend(ierr);
    ierr = CeedDeviceAXPY_Hip((CeedScalar *)
                              y_impl->d_array.values[CEED_SCALAR_TYPE],
                              alpha, (CeedScalar *) x_impl->d_array.values[CEED_SCALAR_TYPE],
                              length);
    CeedChkBackend(ierr);
  }
  if (y_impl->h_array.values[CEED_SCALAR_TYPE]) {
    ierr = CeedVectorSyncArray(x, CEED_MEM_HOST); CeedChkBackend(ierr);
    ierr = CeedHostAXPY_Hip((CeedScalar *) y_impl->h_array.values[CEED_SCALAR_TYPE],
                            alpha, (CeedScalar *) x_impl->h_array.values[CEED_SCALAR_TYPE], length);
    CeedChkBackend(ierr);
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute the pointwise multiplication w = x .* y on the host
//------------------------------------------------------------------------------
static int CeedHostPointwiseMult_Hip(CeedScalar *w_array, CeedScalar *x_array,
                                     CeedScalar *y_array, CeedInt length) {
  for (int i = 0; i < length; i++)
    w_array[i] = x_array[i] * y_array[i];
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute the pointwise multiplication w = x .* y on device (impl in .cu file)
//------------------------------------------------------------------------------
int CeedDevicePointwiseMult_Hip(CeedScalar *w_array, CeedScalar *x_array,
                                CeedScalar *y_array, CeedInt length);

//------------------------------------------------------------------------------
// Compute the pointwise multiplication w = x .* y
//------------------------------------------------------------------------------
static int CeedVectorPointwiseMult_Hip(CeedVector w, CeedVector x,
                                       CeedVector y) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(w, &ceed); CeedChkBackend(ierr);
  CeedVector_Hip *w_impl, *x_impl, *y_impl;
  ierr = CeedVectorGetData(w, &w_impl); CeedChkBackend(ierr);
  ierr = CeedVectorGetData(x, &x_impl); CeedChkBackend(ierr);
  ierr = CeedVectorGetData(y, &y_impl); CeedChkBackend(ierr);
  CeedSize length;
  ierr = CeedVectorGetLength(w, &length); CeedChkBackend(ierr);

  // Set value for synced device/host array
  if (!w_impl->d_array.values[CEED_SCALAR_TYPE] &&
      !w_impl->h_array.values[CEED_SCALAR_TYPE]) {
    ierr = CeedVectorSetValue(w, 0.0); CeedChkBackend(ierr);
  }
  if (w_impl->d_array.values[CEED_SCALAR_TYPE]) {
    ierr = CeedVectorSyncArray(x, CEED_MEM_DEVICE); CeedChkBackend(ierr);
    ierr = CeedVectorSyncArray(y, CEED_MEM_DEVICE); CeedChkBackend(ierr);
    ierr = CeedDevicePointwiseMult_Hip((CeedScalar *)
                                       w_impl->d_array.values[CEED_SCALAR_TYPE],
                                       (CeedScalar *) x_impl->d_array.values[CEED_SCALAR_TYPE],
                                       (CeedScalar *) y_impl->d_array.values[CEED_SCALAR_TYPE], length);
    CeedChkBackend(ierr);
  }
  if (w_impl->h_array.values[CEED_SCALAR_TYPE]) {
    ierr = CeedVectorSyncArray(x, CEED_MEM_HOST); CeedChkBackend(ierr);
    ierr = CeedVectorSyncArray(y, CEED_MEM_HOST); CeedChkBackend(ierr);
    ierr = CeedHostPointwiseMult_Hip((CeedScalar *)
                                     w_impl->h_array.values[CEED_SCALAR_TYPE],
                                     (CeedScalar *) x_impl->h_array.values[CEED_SCALAR_TYPE],
                                     (CeedScalar *) y_impl->h_array.values[CEED_SCALAR_TYPE], length);
    CeedChkBackend(ierr);
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy the vector
//------------------------------------------------------------------------------
static int CeedVectorDestroy_Hip(const CeedVector vec) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Hip *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  for (int i = 0; i < CEED_NUM_PRECISIONS; i++) {
    if (impl->d_array_owned.values[i]) {
      ierr = hipFree(impl->d_array_owned.values[i]); CeedChk_Hip(ceed, ierr);
    }
    if (impl->h_array_owned.values[i]) {
      ierr = CeedFree(&impl->h_array_owned.values[i]); CeedChkBackend(ierr);
    }
  }
  ierr = CeedFree(&impl); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create a vector of the specified length (does not allocate memory)
//------------------------------------------------------------------------------
int CeedVectorCreate_Hip(CeedSize n, CeedVector vec) {
  CeedVector_Hip *impl;
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "HasValidArray",
                                CeedVectorHasValidArray_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "HasBorrowedArrayOfType",
                                CeedVectorHasBorrowedArrayOfType_Hip);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "CheckArrayStatus",
                                CeedVectorCheckArrayStatus_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "SetArrayGeneric",
                                CeedVectorSetArrayGeneric_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "TakeArrayGeneric",
                                CeedVectorTakeArrayGeneric_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "SetValue",
                                (int (*)())(CeedVectorSetValue_Hip)); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "SyncArrayGeneric",
                                CeedVectorSyncArrayGeneric_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayGeneric",
                                CeedVectorGetArrayGeneric_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayReadGeneric",
                                CeedVectorGetArrayReadGeneric_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayWriteGeneric",
                                CeedVectorGetArrayWriteGeneric_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "Norm",
                                CeedVectorNorm_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "Reciprocal",
                                CeedVectorReciprocal_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "Scale",
                                (int (*)())(CeedVectorScale_Hip)); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "AXPY",
                                (int (*)())(CeedVectorAXPY_Hip)); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "PointwiseMult",
                                CeedVectorPointwiseMult_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "Destroy",
                                CeedVectorDestroy_Hip); CeedChkBackend(ierr);

  ierr = CeedCalloc(1, &impl); CeedChkBackend(ierr);
  ierr = CeedVectorSetData(vec, impl); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}
