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
static inline size_t bytes(const CeedVector vec) {
  int ierr;
  CeedInt length;
  ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
  return length * sizeof(CeedScalar);
}

//------------------------------------------------------------------------------
// Sync host to device
//------------------------------------------------------------------------------
static inline int CeedVectorSyncH2D_Cuda(const CeedVector vec) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  if (!impl->h_array)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "No valid host data to sync to device");
  // LCOV_EXCL_STOP

  if (impl->d_array_borrowed) {
    impl->d_array = impl->d_array_borrowed;
  } else if (impl->d_array_owned) {
    impl->d_array = impl->d_array_owned;
  } else {
    ierr = cudaMalloc((void **)&impl->d_array_owned, bytes(vec));
    CeedChk_Cu(ceed, ierr);
    impl->d_array = impl->d_array_owned;
  }

  ierr = cudaMemcpy(impl->d_array, impl->h_array, bytes(vec),
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync device to host
//------------------------------------------------------------------------------
static inline int CeedVectorSyncD2H_Cuda(const CeedVector vec) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  if (!impl->d_array)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "No valid device data to sync to host");
  // LCOV_EXCL_STOP

  if (impl->h_array_borrowed) {
    impl->h_array = impl->h_array_borrowed;
  } else if (impl->h_array_owned) {
    impl->h_array = impl->h_array_owned;
  } else {
    CeedInt length;
    ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
    ierr = CeedCalloc(length, &impl->h_array_owned);  CeedChkBackend(ierr);
    impl->h_array = impl->h_array_owned;
  }

  ierr = cudaMemcpy(impl->h_array, impl->d_array, bytes(vec),
                    cudaMemcpyDeviceToHost); CeedChk_Cu(ceed, ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync arrays
//------------------------------------------------------------------------------
static inline int CeedVectorSync_Cuda(const CeedVector vec,
                                      CeedMemType mem_type) {
  switch (mem_type) {
  case CEED_MEM_HOST: return CeedVectorSyncD2H_Cuda(vec);
  case CEED_MEM_DEVICE: return CeedVectorSyncH2D_Cuda(vec);
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

  impl->h_array = NULL;
  impl->d_array = NULL;

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

  *has_valid_array = !!impl->h_array || !!impl->d_array;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if has array of given type
//------------------------------------------------------------------------------
static inline int CeedVectorHasArrayOfType_Cuda(const CeedVector vec,
    CeedMemType mem_type, bool *has_array_of_type) {
  int ierr;
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

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
static inline int CeedVectorHasBorrowedArrayOfType_Cuda(const CeedVector vec,
    CeedMemType mem_type, bool *has_borrowed_array_of_type) {
  int ierr;
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

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
// Check if is any array of given type
//------------------------------------------------------------------------------
static inline int CeedVectorNeedSync_Cuda(const CeedVector vec,
    CeedMemType mem_type, bool *need_sync) {
  int ierr;
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  bool has_valid_array = false;
  ierr = CeedVectorHasValidArray(vec, &has_valid_array); CeedChkBackend(ierr);
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
// Set array from host
//------------------------------------------------------------------------------
static int CeedVectorSetArrayHost_Cuda(const CeedVector vec,
                                       const CeedCopyMode copy_mode, CeedScalar *array) {
  int ierr;
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  switch (copy_mode) {
  case CEED_COPY_VALUES: {
    CeedInt length;
    if (!impl->h_array_owned) {
      ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
      ierr = CeedMalloc(length, &impl->h_array_owned); CeedChkBackend(ierr);
    }
    impl->h_array_borrowed = NULL;
    impl->h_array = impl->h_array_owned;
    if (array)
      memcpy(impl->h_array, array, bytes(vec));
  } break;
  case CEED_OWN_POINTER:
    ierr = CeedFree(&impl->h_array_owned); CeedChkBackend(ierr);
    impl->h_array_owned = array;
    impl->h_array_borrowed = NULL;
    impl->h_array = array;
    break;
  case CEED_USE_POINTER:
    ierr = CeedFree(&impl->h_array_owned); CeedChkBackend(ierr);
    impl->h_array_borrowed = array;
    impl->h_array = array;
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set array from device
//------------------------------------------------------------------------------
static int CeedVectorSetArrayDevice_Cuda(const CeedVector vec,
    const CeedCopyMode copy_mode, CeedScalar *array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  switch (copy_mode) {
  case CEED_COPY_VALUES:
    if (!impl->d_array_owned) {
      ierr = cudaMalloc((void **)&impl->d_array_owned, bytes(vec));
      CeedChk_Cu(ceed, ierr);
      impl->d_array = impl->d_array_owned;
    }
    if (array) {
      ierr = cudaMemcpy(impl->d_array, array, bytes(vec),
                        cudaMemcpyDeviceToDevice); CeedChk_Cu(ceed, ierr);
    }
    break;
  case CEED_OWN_POINTER:
    ierr = cudaFree(impl->d_array_owned); CeedChk_Cu(ceed, ierr);
    impl->d_array_owned = array;
    impl->d_array_borrowed = NULL;
    impl->d_array = array;
    break;
  case CEED_USE_POINTER:
    ierr = cudaFree(impl->d_array_owned); CeedChk_Cu(ceed, ierr);
    impl->d_array_owned = NULL;
    impl->d_array_borrowed = array;
    impl->d_array = array;
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
    return CeedVectorSetArrayHost_Cuda(vec, copy_mode, array);
  case CEED_MEM_DEVICE:
    return CeedVectorSetArrayDevice_Cuda(vec, copy_mode, array);
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
      ierr = CeedVectorSetArray(vec, CEED_MEM_DEVICE, CEED_COPY_VALUES, NULL);
      CeedChkBackend(ierr);
    }
  }
  if (impl->d_array) {
    ierr = CeedDeviceSetValue_Cuda(impl->d_array, length, val);
    CeedChkBackend(ierr);
    impl->h_array = NULL;
  }
  if (impl->h_array) {
    ierr = CeedHostSetValue_Cuda(impl->h_array, length, val); CeedChkBackend(ierr);
    impl->d_array = NULL;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Take Array
//------------------------------------------------------------------------------
static int CeedVectorTakeArray_Cuda(CeedVector vec, CeedMemType mem_type,
                                    CeedScalar **array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  // Sync array to requested mem_type
  bool need_sync = false;
  ierr = CeedVectorNeedSync_Cuda(vec, mem_type, &need_sync); CeedChkBackend(ierr);
  if (need_sync) {
    ierr = CeedVectorSync_Cuda(vec, mem_type); CeedChkBackend(ierr);
  }

  // Update pointer
  switch (mem_type) {
  case CEED_MEM_HOST:
    (*array) = impl->h_array_borrowed;
    impl->h_array_borrowed = NULL;
    impl->h_array = NULL;
    break;
  case CEED_MEM_DEVICE:
    (*array) = impl->d_array_borrowed;
    impl->d_array_borrowed = NULL;
    impl->d_array = NULL;
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Core logic for array syncronization for GetArray.
//   If a different memory type is most up to date, this will perform a copy
//------------------------------------------------------------------------------
static int CeedVectorGetArrayCore_Cuda(const CeedVector vec,
                                       const CeedMemType mem_type, CeedScalar **array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  bool need_sync = false, has_array_of_type = true;
  ierr = CeedVectorNeedSync_Cuda(vec, mem_type, &need_sync); CeedChkBackend(ierr);
  ierr = CeedVectorHasArrayOfType_Cuda(vec, mem_type, &has_array_of_type);
  CeedChkBackend(ierr);
  if (need_sync) {
    // Sync array to requested mem_type
    ierr = CeedVectorSync_Cuda(vec, mem_type); CeedChkBackend(ierr);
  }

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
static int CeedVectorGetArrayRead_Cuda(const CeedVector vec,
                                       const CeedMemType mem_type, const CeedScalar **array) {
  return CeedVectorGetArrayCore_Cuda(vec, mem_type, (CeedScalar **)array);
}

//------------------------------------------------------------------------------
// Get read/write access to a vector via the specified mem_type
//------------------------------------------------------------------------------
static int CeedVectorGetArray_Cuda(const CeedVector vec,
                                   const CeedMemType mem_type, CeedScalar **array) {
  int ierr;
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  ierr = CeedVectorGetArrayCore_Cuda(vec, mem_type, array); CeedChkBackend(ierr);

  ierr = CeedVectorSetAllInvalid_Cuda(vec); CeedChkBackend(ierr);
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
      if (impl->h_array_borrowed)
        impl->h_array = impl->h_array_borrowed;
      else
        impl->h_array = impl->h_array_owned;
      break;
    case CEED_MEM_DEVICE:
      if (impl->d_array_borrowed)
        impl->d_array = impl->d_array_borrowed;
      else
        impl->d_array = impl->d_array_owned;
    }
  }

  return CeedVectorGetArray_Cuda(vec, mem_type, array);
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
    ierr = cudaMemcpy(&normNoAbs, impl->d_array+indx-1, sizeof(CeedScalar),
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
  if (impl->d_array) {
    ierr = CeedDeviceReciprocal_Cuda(impl->d_array, length); CeedChkBackend(ierr);
  }
  if (impl->h_array) {
    ierr = CeedHostReciprocal_Cuda(impl->h_array, length); CeedChkBackend(ierr);
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
  if (x_impl->d_array) {
    ierr = CeedDeviceScale_Cuda(x_impl->d_array, alpha, length);
    CeedChkBackend(ierr);
  }
  if (x_impl->h_array) {
    ierr = CeedHostScale_Cuda(x_impl->h_array, alpha, length); CeedChkBackend(ierr);
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
  if (y_impl->d_array) {
    ierr = CeedVectorSyncArray(x, CEED_MEM_DEVICE); CeedChkBackend(ierr);
    ierr = CeedDeviceAXPY_Cuda(y_impl->d_array, alpha, x_impl->d_array, length);
    CeedChkBackend(ierr);
  }
  if (y_impl->h_array) {
    ierr = CeedVectorSyncArray(x, CEED_MEM_HOST); CeedChkBackend(ierr);
    ierr = CeedHostAXPY_Cuda(y_impl->h_array, alpha, x_impl->h_array, length);
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
  if (!w_impl->d_array && !w_impl->h_array) {
    ierr = CeedVectorSetValue(w, 0.0); CeedChkBackend(ierr);
  }
  if (w_impl->d_array) {
    ierr = CeedVectorSyncArray(x, CEED_MEM_DEVICE); CeedChkBackend(ierr);
    ierr = CeedVectorSyncArray(y, CEED_MEM_DEVICE); CeedChkBackend(ierr);
    ierr = CeedDevicePointwiseMult_Cuda(w_impl->d_array, x_impl->d_array,
                                        y_impl->d_array, length);
    CeedChkBackend(ierr);
  }
  if (w_impl->h_array) {
    ierr = CeedVectorSyncArray(x, CEED_MEM_HOST); CeedChkBackend(ierr);
    ierr = CeedVectorSyncArray(y, CEED_MEM_HOST); CeedChkBackend(ierr);
    ierr = CeedHostPointwiseMult_Cuda(w_impl->h_array, x_impl->h_array,
                                      y_impl->h_array, length);
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

  ierr = cudaFree(impl->d_array_owned); CeedChk_Cu(ceed, ierr);
  ierr = CeedFree(&impl->h_array_owned); CeedChkBackend(ierr);
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
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "HasBorrowedArrayOfType",
                                CeedVectorHasBorrowedArrayOfType_Cuda);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "SetArray",
                                CeedVectorSetArray_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "TakeArray",
                                CeedVectorTakeArray_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "SetValue",
                                (int (*)())(CeedVectorSetValue_Cuda));
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArray",
                                CeedVectorGetArray_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayRead",
                                CeedVectorGetArrayRead_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayWrite",
                                CeedVectorGetArrayWrite_Cuda); CeedChkBackend(ierr);
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
