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

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <string.h>
#include "ceed-cuda-ref.h"

//------------------------------------------------------------------------------
// * Bytes used
//------------------------------------------------------------------------------
static inline size_t bytes(const CeedVector vec, CeedScalarType prec) {
  int ierr;
  CeedInt length;
  ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
  CeedInt mem_size = 0;
  if (prec == CEED_SCALAR_FP64) {
    mem_size = length * sizeof(double);
  } else if (prec == CEED_SCALAR_FP32) {
    mem_size = length * sizeof(float);
  }
  return mem_size;
}

//------------------------------------------------------------------------------
// Sync host to device for specified precision
//------------------------------------------------------------------------------
static inline int CeedVectorSyncH2D_Cuda(const CeedVector vec,
    const CeedScalarType prec) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  if (!impl->h_array.values[prec])
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "No valid host data to sync to device");
  // LCOV_EXCL_STOP

  if (impl->d_array_borrowed.values[prec]) {
    impl->d_array.values[prec] =
      impl->d_array_borrowed.values[prec];
  } else if (impl->d_array_owned.values[prec]) {
    impl->d_array.values[prec] =
      impl->d_array_owned.values[prec];
  } else {
    ierr = cudaMalloc((void **)&impl->d_array_owned.values[prec],
                      bytes(vec, prec));
    CeedChk_Cu(ceed, ierr);
    impl->d_array.values[prec] =
      impl->d_array_owned.values[prec];
  }

  ierr = cudaMemcpy(impl->d_array.values[prec],
                    impl->h_array.values[prec],
                    bytes(vec, prec),
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync device to host for specified precision
//------------------------------------------------------------------------------
static inline int CeedVectorSyncD2H_Cuda(const CeedVector vec,
    const CeedScalarType prec) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  if (!impl->d_array.values[prec])
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "No valid device data to sync to host");
  // LCOV_EXCL_STOP

  if (impl->h_array_borrowed.values[prec]) {
    impl->h_array.values[prec] =
      impl->h_array_borrowed.values[prec];
  } else if (impl->h_array_owned.values[prec]) {
    impl->h_array.values[prec] =
      impl->h_array_owned.values[prec];
  } else {
    CeedInt length;
    ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
    if (prec == CEED_SCALAR_FP32) {
      ierr = CeedCalloc(length,
                        (float **) &impl->h_array_owned.values[prec]);
    } else if (prec == CEED_SCALAR_FP64) {
      ierr = CeedCalloc(length,
                        (double **) &impl->h_array_owned.values[prec]);
    } else {
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "Invalid scalar precision type specified in CeedVectorSyncD2H");
      // LCOV_EXCL_STOP
    }
    CeedChkBackend(ierr);
    impl->h_array.values[prec] =
      impl->h_array_owned.values[prec];
  }

  ierr = cudaMemcpy(impl->h_array.values[prec],
                    impl->d_array.values[prec],
                    bytes(vec, prec),
                    cudaMemcpyDeviceToHost); CeedChk_Cu(ceed, ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync arrays
//------------------------------------------------------------------------------
static inline int CeedVectorSync_Cuda(const CeedVector vec,
                                      CeedMemType mem_type,
                                      CeedScalarType prec_type) {
  switch (mem_type) {
  case CEED_MEM_HOST: return CeedVectorSyncD2H_Cuda(vec, prec_type);
  case CEED_MEM_DEVICE: return CeedVectorSyncH2D_Cuda(vec, prec_type);
  }
  return CEED_ERROR_UNSUPPORTED;
}

//TODO: check convert code against new owned/borrowed/valid logic...
// convert values only in owned??

//------------------------------------------------------------------------------
// Convert host array to new precision. from_array and to_array *must* be
//   on host already.
//------------------------------------------------------------------------------
static int CeedVectorConvertArrayHost_Cuda(CeedVector vec,
    const CeedScalarType from_prec, const CeedScalarType to_prec,
    CeedScalarArray *from_array, CeedScalarArray *to_array) {
  CeedInt length, ierr;
  ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
  CeedVector_Cuda *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);

  switch (from_prec) {

  case CEED_SCALAR_FP64:
    switch (to_prec) {
    case CEED_SCALAR_FP64:
      // No conversion needed
      break;
    case CEED_SCALAR_FP32:
      if (!to_array->values[CEED_SCALAR_FP32]) {
        if (!data->h_array_owned.values[CEED_SCALAR_FP32]) {
          ierr = CeedMalloc(length,
                            (float **) &data->h_array_owned.values[CEED_SCALAR_FP32]);
          CeedChkBackend(ierr);
        }
        // Use owned memory
        to_array->values[CEED_SCALAR_FP32] =
          data->h_array_owned.values[CEED_SCALAR_FP32];
      }
      float *float_data = (float *) to_array->values[CEED_SCALAR_FP32];
      double *double_data = (double *) from_array->values[CEED_SCALAR_FP64];
      for (int i = 0; i < length; i++)
        float_data[i] = (float) double_data[i];
      break;
    }
    break;

  case CEED_SCALAR_FP32:
    switch (to_prec) {
    case CEED_SCALAR_FP64:
      if (!to_array->values[CEED_SCALAR_FP64]) {
        if (!data->h_array_owned.values[CEED_SCALAR_FP64]) {
          ierr = CeedMalloc(length,
                            (double **) &data->h_array_owned.values[CEED_SCALAR_FP64]);
          CeedChkBackend(ierr);
        }
        // Use owned memory
        to_array->values[CEED_SCALAR_FP64] =
          data->h_array_owned.values[CEED_SCALAR_FP64];
      }
      float *float_data = (float *) from_array->values[CEED_SCALAR_FP32];
      double *double_data = (double *) to_array->values[CEED_SCALAR_FP64];
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
int CeedDeviceConvertArray_Cuda_Fp64_Fp32(CeedInt length,
    double *double_data, float *float_data);

//------------------------------------------------------------------------------
// Convert a single-precision array to double precision
//------------------------------------------------------------------------------
int CeedDeviceConvertArray_Cuda_Fp32_Fp64(CeedInt length,
    float *float_data, double *double_data);

//------------------------------------------------------------------------------
// Convert device array to new precision(impl of individual functions/kernels in
// .cu file)
//------------------------------------------------------------------------------
static int CeedVectorConvertArrayDevice_Cuda(CeedVector vec,
    const CeedScalarType from_prec, const CeedScalarType to_prec,
    CeedScalarArray *from_array, CeedScalarArray *to_array) {

  CeedInt length, ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
  CeedVector_Cuda *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);
  switch (from_prec) {

  case CEED_SCALAR_FP64:
    switch (to_prec) {
    case CEED_SCALAR_FP64:
      // No conversion needed
      break;
    case CEED_SCALAR_FP32:
      if (!to_array->values[CEED_SCALAR_FP32]) {
        if (!data->d_array_owned.values[CEED_SCALAR_FP32]) {
          ierr = cudaMalloc((void **)&data->d_array_owned.values[CEED_SCALAR_FP32],
                            bytes(vec, CEED_SCALAR_FP32));
          CeedChk_Cu(ceed, ierr);
        }
        // Use owned memory
        to_array->values[CEED_SCALAR_FP32] =
          data->d_array_owned.values[CEED_SCALAR_FP32];
      }
      ierr = CeedDeviceConvertArray_Cuda_Fp64_Fp32(length,
             (double *) data->d_array.values[CEED_SCALAR_FP64],
             (float *) data->d_array.values[CEED_SCALAR_FP32]);
      CeedChkBackend(ierr);
      break;
    }
    break;

  case CEED_SCALAR_FP32:
    switch (to_prec) {
    case CEED_SCALAR_FP64:
      if (!to_array->values[CEED_SCALAR_FP64]) {
        if (!data->d_array_owned.values[CEED_SCALAR_FP64]) {
          ierr = cudaMalloc((void **)&data->d_array_owned.values[CEED_SCALAR_FP64],
                            bytes(vec, CEED_SCALAR_FP64));
          CeedChk_Cu(ceed, ierr);
        }
        // Use owned memory
        to_array->values[CEED_SCALAR_FP64] =
          data->d_array_owned.values[CEED_SCALAR_FP64];
      }
      ierr = CeedDeviceConvertArray_Cuda_Fp32_Fp64(length,
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
static int CeedVectorConvertArray_Cuda(CeedVector vec,
                                       const CeedMemType mem_type, const CeedScalarType from_prec,
                                       const CeedScalarType to_prec, CeedScalarArray *from_array,
                                       CeedScalarArray *to_array) {

  switch (mem_type) {
  case CEED_MEM_HOST: return CeedVectorConvertArrayHost_Cuda(vec, from_prec,
                               to_prec, from_array, to_array);
  case CEED_MEM_DEVICE: return CeedVectorConvertArrayDevice_Cuda(vec, from_prec,
                                 to_prec, from_array, to_array);
  }
  return CEED_ERROR_UNSUPPORTED;
}

//------------------------------------------------------------------------------
// Set all pointers as invalid
//------------------------------------------------------------------------------
static inline int CeedVectorSetAllInvalid_Cuda(const CeedVector vec) {
  int ierr;
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  for (int i = 0; i < CEED_NUM_PRECISIONS; i++) {
    impl->h_array.values[i] = NULL;
    impl->d_array.values[i] = NULL;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if CeedVector has any valid pointer
//------------------------------------------------------------------------------
static inline int CeedVectorHasValidArray_Cuda(const CeedVector vec,
    bool *has_valid_array) {
  int ierr;
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  *has_valid_array = false;
  for (int i = 0; i < CEED_NUM_PRECISIONS; i++) {
    *has_valid_array = *has_valid_array ||
                       (!!impl->h_array.values[i] || !!impl->d_array.values[i]);
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
static inline int CeedVectorGetPrecision_Cuda(const CeedVector vec,
    const CeedMemType mem_type, CeedScalarType *preferred_precision) {

  int ierr;
  CeedVector_Cuda *impl;
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
// Check if CeedVector has a valid pointer in the specified precision
//------------------------------------------------------------------------------
static inline int CeedVectorHasValidArrayOfPrecision_Cuda(const CeedVector vec,
    CeedScalarType prec_type, bool *has_valid_array) {
  int ierr;
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  *has_valid_array = !!impl->h_array.values[prec_type] ||
                     !!impl->d_array.values[prec_type];

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if has array of given memory type
//------------------------------------------------------------------------------
static inline int CeedVectorHasArrayOfType_Cuda(const CeedVector vec,
    CeedMemType mem_type, bool *has_array_of_type) {
  int ierr;
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  switch (mem_type) {
  case CEED_MEM_HOST:
    *has_array_of_type = !!impl->h_array_borrowed.values[CEED_SCALAR_TYPE] ||
                         !!impl->h_array_owned.values[CEED_SCALAR_TYPE];
    break;
  case CEED_MEM_DEVICE:
    *has_array_of_type = !!impl->d_array_borrowed.values[CEED_SCALAR_TYPE] ||
                         !!impl->d_array_owned.values[CEED_SCALAR_TYPE];
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if has array of given memory and precision type
//------------------------------------------------------------------------------
static inline int CeedVectorHasArrayOfTypeAndPrecision_Cuda(
  const CeedVector vec,
  CeedMemType mem_type, CeedScalarType prec_type, bool *has_array_of_type) {
  int ierr;
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  switch (mem_type) {
  case CEED_MEM_HOST:
    *has_array_of_type = !!impl->h_array_borrowed.values[prec_type] ||
                         !!impl->h_array_owned.values[prec_type];
    break;
  case CEED_MEM_DEVICE:
    *has_array_of_type = !!impl->d_array_borrowed.values[prec_type] ||
                         !!impl->d_array_owned.values[prec_type];
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if has valid array of given memory type
//------------------------------------------------------------------------------
static inline int CeedVectorHasValidArrayOfType_Cuda(const CeedVector vec,
    CeedMemType mem_type, bool *has_valid_array_of_type) {
  int ierr;
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  *has_valid_array_of_type = false;
  switch (mem_type) {
  case CEED_MEM_HOST:
    for (int i = 0; i < CEED_NUM_PRECISIONS; i++) {
      *has_valid_array_of_type = *has_valid_array_of_type ||
                                 !!impl->h_array.values[i];
    }
    break;
  case CEED_MEM_DEVICE:
    for (int i = 0; i < CEED_NUM_PRECISIONS; i++) {
      *has_valid_array_of_type = *has_valid_array_of_type ||
                                 !!impl->d_array.values[i];
    }
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if has borrowed array of given memory type and default precision
//------------------------------------------------------------------------------
static inline int CeedVectorHasBorrowedArrayOfType_Cuda(const CeedVector vec,
    CeedMemType mem_type, bool *has_borrowed_array_of_type) {
  int ierr;
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  switch (mem_type) {
  case CEED_MEM_HOST:
    *has_borrowed_array_of_type = !!impl->h_array_borrowed.values[CEED_SCALAR_TYPE];
    break;
  case CEED_MEM_DEVICE:
    *has_borrowed_array_of_type = !!impl->d_array_borrowed.values[CEED_SCALAR_TYPE];
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if has borrowed array of given memory type and specified precision
//------------------------------------------------------------------------------
static inline int CeedVectorHasBorrowedArrayOfTypeAndPrecision_Cuda(
  const CeedVector vec,
  CeedMemType mem_type, CeedScalarType prec_type,
  bool *has_borrowed_array_of_type) {
  int ierr;
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  switch (mem_type) {
  case CEED_MEM_HOST:
    *has_borrowed_array_of_type = !!impl->h_array_borrowed.values[prec_type];
    break;
  case CEED_MEM_DEVICE:
    *has_borrowed_array_of_type = !!impl->d_array_borrowed.values[prec_type];
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if is synchronization is needed from other memory
//------------------------------------------------------------------------------
static inline int CeedVectorNeedSync_Cuda(const CeedVector vec,
    CeedMemType mem_type, bool *need_sync) {
  int ierr;
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  bool has_valid_array = false;
  ierr = CeedVectorHasValidArray(vec, &has_valid_array); CeedChkBackend(ierr);
  bool has_valid_array_of_type = false;
  ierr = CeedVectorHasValidArrayOfType_Cuda(vec, mem_type,
         &has_valid_array_of_type);
  CeedChkBackend(ierr);

  // Check if we have a valid array, but not for the correct memory type
  *need_sync = has_valid_array && !has_valid_array_of_type;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set array from host
//------------------------------------------------------------------------------
static int CeedVectorSetArrayHost_Cuda(const CeedVector vec,
                                       const CeedScalarType prec_type, const CeedCopyMode copy_mode, void *array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  switch (copy_mode) {
  case CEED_COPY_VALUES: {
    CeedInt length;
    if (!impl->h_array_owned.values[prec_type]) {
      ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
      if (prec_type == CEED_SCALAR_FP32) {
        ierr = CeedMalloc(length,
                          (float **) &impl->h_array_owned.values[CEED_SCALAR_FP32]);
      } else if (prec_type == CEED_SCALAR_FP64) {
        ierr = CeedMalloc(length,
                          (double **) &impl->h_array_owned.values[CEED_SCALAR_FP64]);
      } else {
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_BACKEND,
                         "Invalid scalar precision type specified in CeedVectorSetArray");
        // LCOV_EXCL_STOP
      }
      CeedChkBackend(ierr);
    }
    impl->h_array_borrowed.values[prec_type] = NULL;
    impl->h_array.values[prec_type] =
      impl->h_array_owned.values[prec_type];
    if (array)
      memcpy(impl->h_array.values[prec_type], array, bytes(vec,
             prec_type));
  } break;
  case CEED_OWN_POINTER:
    ierr = CeedFree(&impl->h_array_owned.values[prec_type]);
    CeedChkBackend(ierr);
    impl->h_array_owned.values[prec_type] = array;
    impl->h_array_borrowed.values[prec_type] = NULL;
    impl->h_array.values[prec_type] = array;
    break;
  case CEED_USE_POINTER:
    ierr = CeedFree(&impl->h_array_owned.values[prec_type]);
    CeedChkBackend(ierr);
    impl->h_array_borrowed.values[prec_type] = array;
    impl->h_array.values[prec_type] = array;
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set array from device
//------------------------------------------------------------------------------
static int CeedVectorSetArrayDevice_Cuda(const CeedVector vec,
    const CeedScalarType prec_type, const CeedCopyMode copy_mode, void *array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  switch (copy_mode) {
  case CEED_COPY_VALUES:
    if (!impl->d_array_owned.values[prec_type]) {
      ierr = cudaMalloc((void **)&impl->d_array_owned.values[prec_type],
                        bytes(vec, prec_type));
      CeedChk_Cu(ceed, ierr);
      impl->d_array.values[prec_type] =
        impl->d_array_owned.values[prec_type];
    }
    if (array) {
      ierr = cudaMemcpy(impl->d_array.values[prec_type], array, bytes(vec,
                        prec_type),
                        cudaMemcpyDeviceToDevice); CeedChk_Cu(ceed, ierr);
    }
    break;
  case CEED_OWN_POINTER:
    ierr = cudaFree(impl->d_array_owned.values[prec_type]);
    CeedChk_Cu(ceed, ierr);
    impl->d_array_owned.values[prec_type] = array;
    impl->d_array_borrowed.values[prec_type] = NULL;
    impl->d_array.values[prec_type] = array;
    break;
  case CEED_USE_POINTER:
    ierr = cudaFree(impl->d_array_owned.values[prec_type]);
    CeedChk_Cu(ceed, ierr);
    impl->d_array_owned.values[prec_type] = NULL;
    impl->d_array_borrowed.values[prec_type] = array;
    impl->d_array.values[prec_type] = array;
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set the array used by a vector,
//   freeing any previously allocated array if applicable
//------------------------------------------------------------------------------
static int CeedVectorSetArray_Cuda(const CeedVector vec,
                                   const CeedMemType mem_type,
                                   const CeedCopyMode copy_mode, CeedScalar *array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  ierr = CeedVectorSetAllInvalid_Cuda(vec); CeedChkBackend(ierr);
  switch (mem_type) {
  case CEED_MEM_HOST:
    return CeedVectorSetArrayHost_Cuda(vec, CEED_SCALAR_TYPE, copy_mode, array);
  case CEED_MEM_DEVICE:
    return CeedVectorSetArrayDevice_Cuda(vec, CEED_SCALAR_TYPE, copy_mode, array);
  }

  return CEED_ERROR_UNSUPPORTED;
}

//------------------------------------------------------------------------------
// Set the array used by a vector, with specified precision,
//   freeing any previously allocated array if applicable
//------------------------------------------------------------------------------
static int CeedVectorSetArrayTyped_Cuda(const CeedVector vec,
                                        const CeedMemType mem_type,
                                        const CeedScalarType prec_type,
                                        const CeedCopyMode copy_mode,
                                        CeedScalar *array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  ierr = CeedVectorSetAllInvalid_Cuda(vec); CeedChkBackend(ierr);
  switch (mem_type) {
  case CEED_MEM_HOST:
    return CeedVectorSetArrayHost_Cuda(vec, prec_type, copy_mode, array);
  case CEED_MEM_DEVICE:
    return CeedVectorSetArrayDevice_Cuda(vec, prec_type, copy_mode, array);
  }

  return CEED_ERROR_UNSUPPORTED;
}

//------------------------------------------------------------------------------
// Set host array to value
//------------------------------------------------------------------------------
static int CeedHostSetValue_Cuda(CeedScalar *h_array, CeedInt length,
                                 CeedScalar val) {
  for (int i = 0; i < length; i++)
    h_array[i] = val;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set device array to value (impl in .cu file)
//------------------------------------------------------------------------------
int CeedDeviceSetValue_Cuda(CeedScalar *d_array, CeedInt length,
                            CeedScalar val);

//------------------------------------------------------------------------------
// Set a vector to a value,
//------------------------------------------------------------------------------
static int CeedVectorSetValue_Cuda(CeedVector vec, CeedScalar val) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);
  CeedInt length;
  ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);

  // Set value for synced device/host array
  if (!impl->d_array.values[CEED_SCALAR_TYPE]
      && !impl->h_array.values[CEED_SCALAR_TYPE]) {
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
    ierr = CeedDeviceSetValue_Cuda((CeedScalar *)
                                   impl->d_array.values[CEED_SCALAR_TYPE], length, val);
    CeedChkBackend(ierr);
    impl->h_array.values[CEED_SCALAR_TYPE] = NULL;
  }
  if (impl->h_array.values[CEED_SCALAR_TYPE]) {
    ierr = CeedHostSetValue_Cuda((CeedScalar *)
                                 impl->h_array.values[CEED_SCALAR_TYPE], length, val); CeedChkBackend(ierr);
    impl->d_array.values[CEED_SCALAR_TYPE] = NULL;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Core logic for VectorTakeArray.  If a different memory type or precision
//   is most up to date, this will perform a copy and/or conversion. The order
//   of sync or convert is chosen to minimize data movement to/from the GPU.
//------------------------------------------------------------------------------
static int CeedVectorTakeArrayCore_Cuda(CeedVector vec, CeedMemType mem_type,
                                        CeedScalarType prec_type, void **array) {

  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  // TODO: clean up logic, make more general in case other precisions are added

  // Check if we have a memory type mismatch between
  //  valid array(s) and requested mem_type.
  bool need_sync = false;
  ierr = CeedVectorNeedSync_Cuda(vec, mem_type, &need_sync);
  CeedChkBackend(ierr);
  switch (mem_type) {
  case CEED_MEM_HOST:
    // Do we need to sync from the device?
    if (need_sync) {
      if (prec_type == CEED_SCALAR_FP32) {
        // Check for device array in matching precision
        if (!impl->d_array.values[CEED_SCALAR_FP32]) {
          // Convert on device
          ierr = CeedVectorConvertArray_Cuda(vec, CEED_MEM_DEVICE,
                                             CEED_SCALAR_FP64, CEED_SCALAR_FP32,
                                             &impl->d_array, &impl->d_array);
          CeedChkBackend(ierr);
        }
        // Sync array
        ierr = CeedVectorSync_Cuda(vec, mem_type, prec_type);
        CeedChkBackend(ierr);
      } else {
        // Check for device array in matching precision
        if (!impl->d_array.values[CEED_SCALAR_FP64]) {
          // Sync first, then convert on host
          ierr = CeedVectorSync_Cuda(vec, mem_type, CEED_SCALAR_FP32);
          CeedChkBackend(ierr);
        } else {
          // Sync double precision
          ierr = CeedVectorSync_Cuda(vec, mem_type, CEED_SCALAR_FP64);
          CeedChkBackend(ierr);
        }
      }
    }
    // Convert host array to new precision, if required
    if (!impl->h_array.values[prec_type]) {
      // Get current precision to convert from
      CeedScalarType cur_prec;
      ierr = CeedVectorGetPrecision_Cuda(vec, CEED_MEM_HOST, &cur_prec);
      CeedChkBackend(ierr);
      ierr = CeedVectorConvertArray_Cuda(vec, CEED_MEM_HOST, cur_prec,
                                         prec_type, &impl->h_array, &impl->h_array);
      CeedChkBackend(ierr);
    }
    break;

  case CEED_MEM_DEVICE:
    // Do we need to sync to the device?
    if (need_sync) {
      if (prec_type == CEED_SCALAR_FP32) {
        // Check for host array in matching precision
        if (!impl->h_array.values[CEED_SCALAR_FP32]) {
          // Convert on host
          ierr = CeedVectorConvertArray_Cuda(vec, CEED_MEM_HOST,
                                             CEED_SCALAR_FP64, CEED_SCALAR_FP32,
                                             &impl->h_array, &impl->h_array);
          CeedChkBackend(ierr);
        }
        // Sync array
        ierr = CeedVectorSync_Cuda(vec, mem_type, prec_type);
        CeedChkBackend(ierr);
      } else {
        // Check for host array in matching precision
        if (!impl->h_array.values[CEED_SCALAR_FP64]) {
          // Sync first, then convert on host
          ierr = CeedVectorSync_Cuda(vec, mem_type, CEED_SCALAR_FP32);
          CeedChkBackend(ierr);
        } else {
          // Sync double precision
          ierr = CeedVectorSync_Cuda(vec, mem_type, CEED_SCALAR_FP64);
          CeedChkBackend(ierr);
        }
      }
    }
    // Convert device array to new precision, if required
    if (!impl->d_array.values[prec_type]) {
      // Get current precision to convert from
      CeedScalarType cur_prec;
      ierr = CeedVectorGetPrecision_Cuda(vec, CEED_MEM_DEVICE, &cur_prec);
      CeedChkBackend(ierr);
      ierr = CeedVectorConvertArray_Cuda(vec, CEED_MEM_DEVICE, cur_prec,
                                         prec_type, &impl->d_array, &impl->d_array);
      CeedChkBackend(ierr);
    }
    break;
  }

  // Update pointer
  switch (mem_type) {
  case CEED_MEM_HOST:
    (*array) = impl->h_array_borrowed.values[prec_type];
    impl->h_array_borrowed.values[prec_type] = NULL;
    impl->h_array.values[prec_type] = NULL;
    break;
  case CEED_MEM_DEVICE:
    (*array) = impl->d_array_borrowed.values[prec_type];
    impl->d_array_borrowed.values[prec_type] = NULL;
    impl->d_array.values[prec_type] = NULL;
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Take Array
//------------------------------------------------------------------------------
static int CeedVectorTakeArray_Cuda(CeedVector vec, CeedMemType mem_type,
                                    CeedScalar **array) {
  return CeedVectorTakeArrayCore_Cuda(vec, mem_type, CEED_SCALAR_TYPE,
                                      (void **) array);
}

//------------------------------------------------------------------------------
// Vector Take Array in a specific parameter
//------------------------------------------------------------------------------
static int CeedVectorTakeArrayTyped_Cuda(CeedVector vec, CeedMemType mem_type,
    CeedScalarType prec_type, void **array) {
  return CeedVectorTakeArrayCore_Cuda(vec, mem_type, prec_type, array);
}

//------------------------------------------------------------------------------
// Core logic for array syncronization for GetArray.
//   If a different memory type is most up to date, this will perform a copy
//------------------------------------------------------------------------------
static int CeedVectorGetArrayCore_Cuda(const CeedVector vec,
                                       const CeedMemType mem_type, const CeedScalarType prec_type, void **array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  bool need_sync = false;
  ierr = CeedVectorNeedSync_Cuda(vec, mem_type, &need_sync);
  CeedChkBackend(ierr);
  switch (mem_type) {
  case CEED_MEM_HOST:
    // Do we need to sync from the device?
    if (need_sync) {
      if (prec_type == CEED_SCALAR_FP32) {
        // Check for device array in matching precision
        if (!impl->d_array.values[CEED_SCALAR_FP32]) {
          // Convert on device
          ierr = CeedVectorConvertArray_Cuda(vec, CEED_MEM_DEVICE,
                                             CEED_SCALAR_FP64, CEED_SCALAR_FP32,
                                             &impl->d_array, &impl->d_array);
          CeedChkBackend(ierr);
        }
        // Sync array
        ierr = CeedVectorSync_Cuda(vec, mem_type, prec_type);
        CeedChkBackend(ierr);
      } else {
        // Check for device array in matching precision
        if (!impl->d_array.values[CEED_SCALAR_FP64]) {
          // Sync first, then convert on host
          ierr = CeedVectorSync_Cuda(vec, mem_type, CEED_SCALAR_FP32);
          CeedChkBackend(ierr);
        } else {
          // Sync double precision
          ierr = CeedVectorSync_Cuda(vec, mem_type, CEED_SCALAR_FP64);
          CeedChkBackend(ierr);
        }
      }
    }
    // Convert host array to new precision, if required
    if (!impl->h_array.values[prec_type]) {
      // Get current precision to convert from
      CeedScalarType cur_prec;
      ierr = CeedVectorGetPrecision_Cuda(vec, CEED_MEM_HOST, &cur_prec);
      CeedChkBackend(ierr);
      ierr = CeedVectorConvertArray_Cuda(vec, CEED_MEM_HOST, cur_prec,
                                         prec_type, &impl->h_array, &impl->h_array);
      CeedChkBackend(ierr);
    }
    break;

  case CEED_MEM_DEVICE:
    // Do we need to sync to the device?
    if (need_sync) {
      if (prec_type == CEED_SCALAR_FP32) {
        // Check for host array in matching precision
        if (!impl->h_array.values[CEED_SCALAR_FP32]) {
          // Convert on host
          ierr = CeedVectorConvertArray_Cuda(vec, CEED_MEM_HOST,
                                             CEED_SCALAR_FP64, CEED_SCALAR_FP32,
                                             &impl->h_array, &impl->h_array);
          CeedChkBackend(ierr);
        }
        // Sync array
        ierr = CeedVectorSync_Cuda(vec, mem_type, prec_type);
        CeedChkBackend(ierr);
      } else {
        // Check for host array in matching precision
        if (!impl->h_array.values[CEED_SCALAR_FP64]) {
          // Sync first, then convert on host
          ierr = CeedVectorSync_Cuda(vec, mem_type, CEED_SCALAR_FP32);
          CeedChkBackend(ierr);
        } else {
          // Sync double precision
          ierr = CeedVectorSync_Cuda(vec, mem_type, CEED_SCALAR_FP64);
          CeedChkBackend(ierr);
        }
      }
    }
    // Convert device array to new precision, if required
    if (!impl->d_array.values[prec_type]) {
      // Get current precision to convert from
      CeedScalarType cur_prec;
      ierr = CeedVectorGetPrecision_Cuda(vec, CEED_MEM_DEVICE, &cur_prec);
      CeedChkBackend(ierr);
      ierr = CeedVectorConvertArray_Cuda(vec, CEED_MEM_DEVICE, cur_prec,
                                         prec_type, &impl->d_array, &impl->d_array);
      CeedChkBackend(ierr);
    }
    break;
  }

  // Update pointer
  switch (mem_type) {
  case CEED_MEM_HOST:
    *array = impl->h_array.values[prec_type];
    break;
  case CEED_MEM_DEVICE:
    *array = impl->d_array.values[prec_type];
    break;
  }

  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
// Get read-only access to a vector via the specified mem_type
//------------------------------------------------------------------------------
static int CeedVectorGetArrayRead_Cuda(const CeedVector vec,
                                       const CeedMemType mem_type, const CeedScalar **array) {
  return CeedVectorGetArrayCore_Cuda(vec, mem_type, CEED_SCALAR_TYPE,
                                     (void **)array);
}

//------------------------------------------------------------------------------
// Get read-only access to a vector via the specified mem_type and prec_type
//------------------------------------------------------------------------------
static int CeedVectorGetArrayReadTyped_Cuda(const CeedVector vec,
    const CeedMemType mem_type, const CeedScalarType prec_type,
    const void **array) {
  return CeedVectorGetArrayCore_Cuda(vec, mem_type, prec_type, (void **)array);
}

//------------------------------------------------------------------------------
// Get read/write access to a vector via the specified mem_type
//------------------------------------------------------------------------------
static int CeedVectorGetArray_Cuda(const CeedVector vec,
                                   const CeedMemType mem_type, CeedScalar **array) {
  int ierr;
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  ierr = CeedVectorGetArrayCore_Cuda(vec, mem_type, CEED_SCALAR_TYPE,
                                     (void **) array);
  CeedChkBackend(ierr);

  ierr = CeedVectorSetAllInvalid_Cuda(vec); CeedChkBackend(ierr);
  switch (mem_type) {
  case CEED_MEM_HOST:
    impl->h_array.values[CEED_SCALAR_TYPE] = (void *) *array;
    break;
  case CEED_MEM_DEVICE:
    impl->d_array.values[CEED_SCALAR_TYPE] = (void *) *array;
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get read/write access to a vector via the specified mem_type and precision
//------------------------------------------------------------------------------
static int CeedVectorGetArrayTyped_Cuda(const CeedVector vec,
                                        const CeedMemType mem_type, const CeedScalarType prec_type, void **array) {
  int ierr;
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  ierr = CeedVectorGetArrayCore_Cuda(vec, mem_type, prec_type, array);
  CeedChkBackend(ierr);

  ierr = CeedVectorSetAllInvalid_Cuda(vec); CeedChkBackend(ierr);
  switch (mem_type) {
  case CEED_MEM_HOST:
    impl->h_array.values[prec_type] = (void *) *array;
    break;
  case CEED_MEM_DEVICE:
    impl->d_array.values[prec_type] = (void *) *array;
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get write access to a vector via the specified mem_type
//------------------------------------------------------------------------------
static int CeedVectorGetArrayWrite_Cuda(const CeedVector vec,
                                        const CeedMemType mem_type, CeedScalar **array) {
  int ierr;
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  bool has_array_of_type = true;
  ierr = CeedVectorHasArrayOfType_Cuda(vec, mem_type, &has_array_of_type);
  CeedChkBackend(ierr);
  if (!has_array_of_type) {
    // Allocate if array is not yet allocated
    ierr = CeedVectorSetArray(vec, mem_type, CEED_COPY_VALUES, NULL);
    CeedChkBackend(ierr);
  } else {
    // Select dirty array
    switch (mem_type) {
    case CEED_MEM_HOST:
      if (impl->h_array_borrowed.values[CEED_SCALAR_TYPE])
        impl->h_array.values[CEED_SCALAR_TYPE] =
          impl->h_array_borrowed.values[CEED_SCALAR_TYPE];
      else
        impl->h_array.values[CEED_SCALAR_TYPE] =
          impl->h_array_owned.values[CEED_SCALAR_TYPE];
      break;
    case CEED_MEM_DEVICE:
      if (impl->d_array_borrowed.values[CEED_SCALAR_TYPE])
        impl->d_array.values[CEED_SCALAR_TYPE] =
          impl->d_array_borrowed.values[CEED_SCALAR_TYPE];
      else
        impl->d_array.values[CEED_SCALAR_TYPE] =
          impl->d_array_owned.values[CEED_SCALAR_TYPE];
    }
  }

  return CeedVectorGetArray_Cuda(vec, mem_type, array);
}

//------------------------------------------------------------------------------
// Get write access to a vector via the specified mem_type and precision
//------------------------------------------------------------------------------
static int CeedVectorGetArrayWriteTyped_Cuda(const CeedVector vec,
    const CeedMemType mem_type, const CeedScalarType prec_type, void **array) {
  int ierr;
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  bool has_array_of_type = true;
  ierr = CeedVectorHasArrayOfTypeAndPrecision_Cuda(vec, mem_type, prec_type,
         &has_array_of_type);
  CeedChkBackend(ierr);
  if (!has_array_of_type) {
    // Allocate if array is not yet allocated
    ierr = CeedVectorSetArrayTyped(vec, mem_type, prec_type, CEED_COPY_VALUES,
                                   NULL);
    CeedChkBackend(ierr);
  } else {
    // Select dirty array
    switch (mem_type) {
    case CEED_MEM_HOST:
      if (impl->h_array_borrowed.values[prec_type])
        impl->h_array.values[prec_type] =
          impl->h_array_borrowed.values[prec_type];
      else
        impl->h_array.values[prec_type] =
          impl->h_array_owned.values[prec_type];
      break;
    case CEED_MEM_DEVICE:
      if (impl->d_array_borrowed.values[prec_type])
        impl->d_array.values[prec_type] =
          impl->d_array_borrowed.values[prec_type];
      else
        impl->d_array.values[prec_type] =
          impl->d_array_owned.values[prec_type];
    }
  }

  return CeedVectorGetArrayTyped_Cuda(vec, mem_type, prec_type, (void **) array);
}

//------------------------------------------------------------------------------
// Restore an array obtained using CeedVectorGetArrayRead()
//------------------------------------------------------------------------------
static int CeedVectorRestoreArrayRead_Cuda(const CeedVector vec) {
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Restore an array obtained using CeedVectorGetArray()
//------------------------------------------------------------------------------
static int CeedVectorRestoreArray_Cuda(const CeedVector vec) {
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get the norm of a CeedVector
//------------------------------------------------------------------------------
static int CeedVectorNorm_Cuda(CeedVector vec, CeedNormType type,
                               CeedScalar *norm) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);
  CeedInt length;
  ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
  cublasHandle_t handle;
  ierr = CeedCudaGetCublasHandle(ceed, &handle); CeedChkBackend(ierr);

  // Compute norm
  const CeedScalar *d_array;
  ierr = CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE, &d_array);
  CeedChkBackend(ierr);
  switch (type) {
  case CEED_NORM_1: {
    if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
      ierr = cublasSasum(handle, length, (float *) d_array, 1, (float *) norm);
    } else {
      ierr = cublasDasum(handle, length, (double *) d_array, 1, (double *) norm);
    }
    CeedChk_Cublas(ceed, ierr);
    break;
  }
  case CEED_NORM_2: {
    if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
      ierr = cublasSnrm2(handle, length, (float *) d_array, 1, (float *) norm);
    } else {
      ierr = cublasDnrm2(handle, length, (double *) d_array, 1, (double *) norm);
    }
    CeedChk_Cublas(ceed, ierr);
    break;
  }
  case CEED_NORM_MAX: {
    CeedInt indx;
    if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
      ierr = cublasIsamax(handle, length, (float *) d_array, 1, &indx);
    } else {
      ierr = cublasIdamax(handle, length, (double *) d_array, 1, &indx);
    }
    CeedChk_Cublas(ceed, ierr);
    CeedScalar normNoAbs;
    ierr = cudaMemcpy(&normNoAbs,
                      (CeedScalar *)(impl->d_array.values[CEED_SCALAR_TYPE])+indx-1,
                      sizeof(CeedScalar),
                      cudaMemcpyDeviceToHost); CeedChk_Cu(ceed, ierr);
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
static int CeedHostReciprocal_Cuda(CeedScalar *h_array, CeedInt length) {
  for (int i = 0; i < length; i++)
    if (fabs(h_array[i]) > CEED_EPSILON)
      h_array[i] = 1./h_array[i];
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Take reciprocal of a vector on device (impl in .cu file)
//------------------------------------------------------------------------------
int CeedDeviceReciprocal_Cuda(CeedScalar *d_array, CeedInt length);

//------------------------------------------------------------------------------
// Take reciprocal of a vector
//------------------------------------------------------------------------------
static int CeedVectorReciprocal_Cuda(CeedVector vec) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);
  CeedInt length;
  ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);

  // Set value for synced device/host array
  if (impl->d_array.values[CEED_SCALAR_TYPE]) {
    ierr = CeedDeviceReciprocal_Cuda((CeedScalar *)
                                     impl->d_array.values[CEED_SCALAR_TYPE], length); CeedChkBackend(ierr);
  }
  if (impl->h_array.values[CEED_SCALAR_TYPE]) {
    ierr = CeedHostReciprocal_Cuda((CeedScalar *)
                                   impl->h_array.values[CEED_SCALAR_TYPE], length); CeedChkBackend(ierr);
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute x = alpha x on the host
//------------------------------------------------------------------------------
static int CeedHostScale_Cuda(CeedScalar *x_array, CeedScalar alpha,
                              CeedInt length) {
  for (int i = 0; i < length; i++)
    x_array[i] *= alpha;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute x = alpha x on device (impl in .cu file)
//------------------------------------------------------------------------------
int CeedDeviceScale_Cuda(CeedScalar *x_array, CeedScalar alpha,
                         CeedInt length);

//------------------------------------------------------------------------------
// Compute x = alpha x
//------------------------------------------------------------------------------
static int CeedVectorScale_Cuda(CeedVector x, CeedScalar alpha) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(x, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *x_impl;
  ierr = CeedVectorGetData(x, &x_impl); CeedChkBackend(ierr);
  CeedInt length;
  ierr = CeedVectorGetLength(x, &length); CeedChkBackend(ierr);

  // Set value for synced device/host array
  if (x_impl->d_array.values[CEED_SCALAR_TYPE]) {
    ierr = CeedDeviceScale_Cuda((CeedScalar *)
                                x_impl->d_array.values[CEED_SCALAR_TYPE], alpha, length);
    CeedChkBackend(ierr);
  }
  if (x_impl->h_array.values[CEED_SCALAR_TYPE]) {
    ierr = CeedHostScale_Cuda((CeedScalar *)
                              x_impl->h_array.values[CEED_SCALAR_TYPE], alpha, length); CeedChkBackend(ierr);
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute y = alpha x + y on the host
//------------------------------------------------------------------------------
static int CeedHostAXPY_Cuda(CeedScalar *y_array, CeedScalar alpha,
                             CeedScalar *x_array, CeedInt length) {
  for (int i = 0; i < length; i++)
    y_array[i] += alpha * x_array[i];
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute y = alpha x + y on device (impl in .cu file)
//------------------------------------------------------------------------------
int CeedDeviceAXPY_Cuda(CeedScalar *y_array, CeedScalar alpha,
                        CeedScalar *x_array, CeedInt length);

//------------------------------------------------------------------------------
// Compute y = alpha x + y
//------------------------------------------------------------------------------
static int CeedVectorAXPY_Cuda(CeedVector y, CeedScalar alpha, CeedVector x) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(y, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *y_impl, *x_impl;
  ierr = CeedVectorGetData(y, &y_impl); CeedChkBackend(ierr);
  ierr = CeedVectorGetData(x, &x_impl); CeedChkBackend(ierr);
  CeedInt length;
  ierr = CeedVectorGetLength(y, &length); CeedChkBackend(ierr);

  // Set value for synced device/host array
  if (y_impl->d_array.values[CEED_SCALAR_TYPE]) {
    ierr = CeedVectorSyncArray(x, CEED_MEM_DEVICE); CeedChkBackend(ierr);
    ierr = CeedDeviceAXPY_Cuda((CeedScalar *)
                               y_impl->d_array.values[CEED_SCALAR_TYPE],
                               alpha, (CeedScalar *) x_impl->d_array.values[CEED_SCALAR_TYPE], length);
    CeedChkBackend(ierr);
  }
  if (y_impl->h_array.values[CEED_SCALAR_TYPE]) {
    ierr = CeedVectorSyncArray(x, CEED_MEM_HOST); CeedChkBackend(ierr);
    ierr = CeedHostAXPY_Cuda((CeedScalar *)
                             y_impl->h_array.values[CEED_SCALAR_TYPE],
                             alpha, (CeedScalar *) x_impl->h_array.values[CEED_SCALAR_TYPE], length);
    CeedChkBackend(ierr);
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute the pointwise multiplication w = x .* y on the host
//------------------------------------------------------------------------------
static int CeedHostPointwiseMult_Cuda(CeedScalar *w_array, CeedScalar *x_array,
                                      CeedScalar *y_array, CeedInt length) {
  for (int i = 0; i < length; i++)
    w_array[i] = x_array[i] * y_array[i];
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute the pointwise multiplication w = x .* y on device (impl in .cu file)
//------------------------------------------------------------------------------
int CeedDevicePointwiseMult_Cuda(CeedScalar *w_array, CeedScalar *x_array,
                                 CeedScalar *y_array, CeedInt length);

//------------------------------------------------------------------------------
// Compute the pointwise multiplication w = x .* y
//------------------------------------------------------------------------------
static int CeedVectorPointwiseMult_Cuda(CeedVector w, CeedVector x,
                                        CeedVector y) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(w, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *w_impl, *x_impl, *y_impl;
  ierr = CeedVectorGetData(w, &w_impl); CeedChkBackend(ierr);
  ierr = CeedVectorGetData(x, &x_impl); CeedChkBackend(ierr);
  ierr = CeedVectorGetData(y, &y_impl); CeedChkBackend(ierr);
  CeedInt length;
  ierr = CeedVectorGetLength(w, &length); CeedChkBackend(ierr);

  // Set value for synced device/host array
  if (!w_impl->d_array.values[CEED_SCALAR_TYPE]
      && !w_impl->h_array.values[CEED_SCALAR_TYPE]) {
    ierr = CeedVectorSetValue(w, 0.0); CeedChkBackend(ierr);
  }
  if (w_impl->d_array.values[CEED_SCALAR_TYPE]) {
    ierr = CeedVectorSyncArray(x, CEED_MEM_DEVICE); CeedChkBackend(ierr);
    ierr = CeedVectorSyncArray(y, CEED_MEM_DEVICE); CeedChkBackend(ierr);
    ierr = CeedDevicePointwiseMult_Cuda((CeedScalar *)
                                        w_impl->d_array.values[CEED_SCALAR_TYPE],
                                        (CeedScalar *) x_impl->d_array.values[CEED_SCALAR_TYPE],
                                        (CeedScalar *) y_impl->d_array.values[CEED_SCALAR_TYPE], length);
    CeedChkBackend(ierr);
  }
  if (w_impl->h_array.values[CEED_SCALAR_TYPE]) {
    ierr = CeedVectorSyncArray(x, CEED_MEM_HOST); CeedChkBackend(ierr);
    ierr = CeedVectorSyncArray(y, CEED_MEM_HOST); CeedChkBackend(ierr);
    ierr = CeedHostPointwiseMult_Cuda((CeedScalar *)
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
static int CeedVectorDestroy_Cuda(const CeedVector vec) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  for (int i = 0; i < CEED_NUM_PRECISIONS; i++) {
    if (impl->d_array_owned.values[i]) {
      ierr = cudaFree(impl->d_array_owned.values[i]); CeedChk_Cu(ceed, ierr);
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
int CeedVectorCreate_Cuda(CeedInt n, CeedVector vec) {
  CeedVector_Cuda *impl;
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "HasValidArray",
                                CeedVectorHasValidArray_Cuda); CeedChkBackend(ierr);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "HasBorrowedArrayOfType",
                                CeedVectorHasBorrowedArrayOfType_Cuda);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec,
                                "HasBorrowedArrayOfTypeAndPrecision",
                                CeedVectorHasBorrowedArrayOfTypeAndPrecision_Cuda);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "SetArray",
                                CeedVectorSetArray_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "SetArrayTyped",
                                CeedVectorSetArrayTyped_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "TakeArray",
                                CeedVectorTakeArray_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "TakeArrayTyped",
                                CeedVectorTakeArrayTyped_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "SetValue",
                                (int (*)())(CeedVectorSetValue_Cuda));
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArray",
                                CeedVectorGetArray_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayTyped",
                                CeedVectorGetArrayTyped_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayRead",
                                CeedVectorGetArrayRead_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayReadTyped",
                                CeedVectorGetArrayReadTyped_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayWrite",
                                CeedVectorGetArrayWrite_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayWriteTyped",
                                CeedVectorGetArrayWriteTyped_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "RestoreArray",
                                CeedVectorRestoreArray_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "RestoreArrayRead",
                                CeedVectorRestoreArrayRead_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "Norm",
                                CeedVectorNorm_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "Reciprocal",
                                CeedVectorReciprocal_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "AXPY",
                                (int (*)())(CeedVectorAXPY_Cuda)); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "Scale",
                                (int (*)())(CeedVectorScale_Cuda)); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "PointwiseMult",
                                CeedVectorPointwiseMult_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "Destroy",
                                CeedVectorDestroy_Cuda); CeedChkBackend(ierr);

  ierr = CeedCalloc(1, &impl); CeedChkBackend(ierr);
  ierr = CeedVectorSetData(vec, impl); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}
