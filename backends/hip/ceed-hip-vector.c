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
#include <hip/hip_runtime.h>
#include <hipblas.h>
#include <math.h>
#include <string.h>
#include "ceed-hip.h"

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
static inline int CeedVectorSyncH2D_Hip(const CeedVector vec) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);

  if (!data->h_array)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "No valid host data to sync to device");
  // LCOV_EXCL_STOP

  if (data->d_array_borrowed) {
    data->d_array = data->d_array_borrowed;
  } else if (data->d_array_owned) {
    data->d_array = data->d_array_owned;
  } else {
    ierr = hipMalloc((void **)&data->d_array_owned, bytes(vec));
    CeedChk_Hip(ceed, ierr);
    data->d_array = data->d_array_owned;
  }

  ierr = hipMemcpy(data->d_array, data->h_array, bytes(vec),
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync device to host
//------------------------------------------------------------------------------
static inline int CeedVectorSyncD2H_Hip(const CeedVector vec) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);

  if (!data->d_array)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "No valid device data to sync to host");
  // LCOV_EXCL_STOP

  if (data->h_array_borrowed) {
    data->h_array = data->h_array_borrowed;
  } else if (data->h_array_owned) {
    data->h_array = data->h_array_owned;
  } else {
    CeedInt length;
    ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
    ierr = CeedCalloc(length, &data->h_array_owned); CeedChkBackend(ierr);
    data->h_array = data->h_array_owned;
  }

  ierr = hipMemcpy(data->h_array, data->d_array, bytes(vec),
                   hipMemcpyDeviceToHost); CeedChk_Hip(ceed, ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync arrays
//------------------------------------------------------------------------------
static inline int CeedVectorSync_Hip(const CeedVector vec, CeedMemType mtype) {
  switch (mtype) {
  case CEED_MEM_HOST: return CeedVectorSyncD2H_Hip(vec);
  case CEED_MEM_DEVICE: return CeedVectorSyncH2D_Hip(vec);
  }
  return CEED_ERROR_UNSUPPORTED;
}

//------------------------------------------------------------------------------
// Set all pointers as stale
//------------------------------------------------------------------------------
static inline int CeedVectorSetAllStale_Hip(const CeedVector vec) {
  int ierr;
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);

  data->h_array = NULL;
  data->d_array = NULL;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if all pointers are stale
//------------------------------------------------------------------------------
static inline int CeedVectorIsAllStale_Hip(const CeedVector vec,
    bool *is_all_stale) {
  int ierr;
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);

  *is_all_stale = !data->h_array && !data->d_array;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if is any array of given type
//------------------------------------------------------------------------------
static inline int CeedVectorIsArrayOfType_Hip(const CeedVector vec,
    CeedMemType mtype, bool *is_array) {
  int ierr;
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);

  switch (mtype) {
  case CEED_MEM_HOST:
    *is_array = !!data->h_array_borrowed || !!data->h_array_owned;
    break;
  case CEED_MEM_DEVICE:
    *is_array = !!data->d_array_borrowed || !!data->d_array_owned;
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync array of given type
//------------------------------------------------------------------------------
static inline int CeedVectorNeedSync_Hip(const CeedVector vec,
    CeedMemType mtype, bool *need_sync) {
  int ierr;
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);

  bool is_all_stale = false;
  ierr = CeedVectorIsAllStale_Hip(vec, &is_all_stale); CeedChkBackend(ierr);
  switch (mtype) {
  case CEED_MEM_HOST:
    *need_sync = !is_all_stale && !data->h_array;
    break;
  case CEED_MEM_DEVICE:
    *need_sync = !is_all_stale && !data->d_array;
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set array from host
//------------------------------------------------------------------------------
static int CeedVectorSetArrayHost_Hip(const CeedVector vec,
                                      const CeedCopyMode cmode, CeedScalar *array) {
  int ierr;
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);

  switch (cmode) {
  case CEED_COPY_VALUES: {
    CeedInt length;
    if (!data->h_array_owned) {
      ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
      ierr = CeedMalloc(length, &data->h_array_owned); CeedChkBackend(ierr);
    }
    data->h_array_borrowed = NULL;
    data->h_array = data->h_array_owned;
    if (array)
      memcpy(data->h_array, array, bytes(vec));
  } break;
  case CEED_OWN_POINTER:
    ierr = CeedFree(&data->h_array_owned); CeedChkBackend(ierr);
    data->h_array_owned = array;
    data->h_array_borrowed = NULL;
    data->h_array = array;
    break;
  case CEED_USE_POINTER:
    ierr = CeedFree(&data->h_array_owned); CeedChkBackend(ierr);
    data->h_array_borrowed = array;
    data->h_array = array;
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set array from device
//------------------------------------------------------------------------------
static int CeedVectorSetArrayDevice_Hip(const CeedVector vec,
                                        const CeedCopyMode cmode, CeedScalar *array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);

  switch (cmode) {
  case CEED_COPY_VALUES:
    if (!data->d_array_owned) {
      ierr = hipMalloc((void **)&data->d_array_owned, bytes(vec));
      CeedChk_Hip(ceed, ierr);
    }
    data->d_array_borrowed = NULL;
    data->d_array = data->d_array_owned;
    if (array) {
      ierr = hipMemcpy(data->d_array, array, bytes(vec),
                       hipMemcpyDeviceToDevice); CeedChk_Hip(ceed, ierr);
    }
    break;
  case CEED_OWN_POINTER:
    ierr = hipFree(data->d_array_owned); CeedChk_Hip(ceed, ierr);
    data->d_array_owned = array;
    data->d_array_borrowed = NULL;
    data->d_array = array;
    break;
  case CEED_USE_POINTER:
    ierr = hipFree(data->d_array_owned); CeedChk_Hip(ceed, ierr);
    data->d_array_owned = NULL;
    data->d_array_borrowed = array;
    data->d_array = array;
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set the array used by a vector,
//   freeing any previously allocated array if applicable
//------------------------------------------------------------------------------
static int CeedVectorSetArray_Hip(const CeedVector vec, const CeedMemType mtype,
                                  const CeedCopyMode cmode, CeedScalar *array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);

  ierr = CeedVectorSetAllStale_Hip(vec); CeedChkBackend(ierr);
  switch (mtype) {
  case CEED_MEM_HOST:
    return CeedVectorSetArrayHost_Hip(vec, cmode, array);
  case CEED_MEM_DEVICE:
    return CeedVectorSetArrayDevice_Hip(vec, cmode, array);
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
// Set a vector to a value,
//------------------------------------------------------------------------------
static int CeedVectorSetValue_Hip(CeedVector vec, CeedScalar val) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);
  CeedInt length;
  ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);

  // Set value for synced device/host array
  if (!data->d_array && !data->h_array) {
    if (data->d_array_borrowed) {
      data->d_array = data->d_array_borrowed;
    } else if (data->h_array_borrowed) {
      data->h_array = data->h_array_borrowed;
    } else if (data->d_array_owned) {
      data->d_array = data->d_array_owned;
    } else if (data->h_array_owned) {
      data->h_array = data->h_array_owned;
    } else {
      ierr = CeedVectorSetArray(vec, CEED_MEM_DEVICE, CEED_COPY_VALUES, NULL);
      CeedChkBackend(ierr);
    }
  }
  if (data->d_array) {
    ierr = CeedDeviceSetValue_Hip(data->d_array, length, val); CeedChkBackend(ierr);
  }
  if (data->h_array) {
    ierr = CeedHostSetValue_Hip(data->h_array, length, val); CeedChkBackend(ierr);
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Take Array
//------------------------------------------------------------------------------
static int CeedVectorTakeArray_Hip(CeedVector vec, CeedMemType mtype,
                                   CeedScalar **array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Hip *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  bool is_all_stale = false;
  ierr = CeedVectorIsAllStale_Hip(vec, &is_all_stale); CeedChkBackend(ierr);
  if (is_all_stale)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Invalid data in array; must set vector with CeedVectorSetValue or CeedVectorSetArray before calling CeedVectorTakeArray");
  // LCOV_EXCL_STOP

  // Sync array to requested memtype
  bool need_sync = false;
  ierr = CeedVectorNeedSync_Hip(vec, mtype, &need_sync); CeedChkBackend(ierr);
  if (need_sync) {
    ierr = CeedVectorSync_Hip(vec, mtype); CeedChkBackend(ierr);
  }

  // Update pointer
  switch(mtype) {
  case CEED_MEM_HOST:
    if (!impl->h_array_borrowed)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "Must set HOST array with CeedVectorSetArray and CEED_USE_POINTER before calling CeedVectorTakeArray");
    // LCOV_EXCL_STOP

    (*array) = impl->h_array_borrowed;
    impl->h_array_borrowed = NULL;
    impl->h_array = NULL;
    break;
  case CEED_MEM_DEVICE:
    if (!impl->d_array_borrowed)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "Must set DEVICE array with CeedVectorSetArray and CEED_USE_POINTER before calling CeedVectorTakeArray");
    // LCOV_EXCL_STOP

    (*array) = impl->d_array_borrowed;
    impl->d_array_borrowed = NULL;
    impl->d_array = NULL;
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get read-only access to a vector via the specified mtype memory type
//   on which to access the array. If the backend uses a different memory type,
//   this will perform a copy (possibly cached).
//------------------------------------------------------------------------------
static int CeedVectorGetArrayRead_Hip(const CeedVector vec,
                                      const CeedMemType mtype, const CeedScalar **array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);

  bool is_all_stale = false;
  ierr = CeedVectorIsAllStale_Hip(vec, &is_all_stale); CeedChkBackend(ierr);
  bool is_array_of_type = true;
  ierr = CeedVectorIsArrayOfType_Hip(vec, mtype, &is_array_of_type);
  CeedChkBackend(ierr);
  if (is_all_stale && is_array_of_type) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Invalid data in array; must set vector with CeedVectorSetValue or CeedVectorSetArray before calling CeedVectorGetArray");
    // LCOV_EXCL_STOP
  }

  // Sync array to requested memtype
  bool need_sync = false;
  ierr = CeedVectorNeedSync_Hip(vec, mtype, &need_sync); CeedChkBackend(ierr);
  if (need_sync) {
    ierr = CeedVectorSync_Hip(vec, mtype); CeedChkBackend(ierr);
  } else if (!is_array_of_type) {
    ierr = CeedVectorSetArray(vec, mtype, CEED_COPY_VALUES, NULL);
    CeedChkBackend(ierr);
  }

  // Update pointer
  switch (mtype) {
  case CEED_MEM_HOST:
    *array = data->h_array;
    break;
  case CEED_MEM_DEVICE:
    *array = data->d_array;
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get array
//------------------------------------------------------------------------------
static int CeedVectorGetArray_Hip(const CeedVector vec, const CeedMemType mtype,
                                  CeedScalar **array) {
  int ierr;
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);

  ierr = CeedVectorGetArrayRead_Hip(vec, mtype, (const CeedScalar **)array);
  CeedChkBackend(ierr);

  ierr = CeedVectorSetAllStale_Hip(vec); CeedChkBackend(ierr);
  switch (mtype) {
  case CEED_MEM_HOST:
    data->h_array = *array;
    break;
  case CEED_MEM_DEVICE:
    data->d_array = *array;
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Restore an array obtained using CeedVectorGetArrayRead()
//------------------------------------------------------------------------------
static int CeedVectorRestoreArrayRead_Hip(const CeedVector vec) {
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Restore an array obtained using CeedVectorGetArray()
//------------------------------------------------------------------------------
static int CeedVectorRestoreArray_Hip(const CeedVector vec) {
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get the norm of a CeedVector
//------------------------------------------------------------------------------
static int CeedVectorNorm_Hip(CeedVector vec, CeedNormType type,
                              CeedScalar *norm) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);
  CeedInt length;
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
    ierr = hipMemcpy(&normNoAbs, data->d_array+indx-1, sizeof(CeedScalar),
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
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);
  CeedInt length;
  ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);

  bool is_all_stale = false;
  ierr = CeedVectorIsAllStale_Hip(vec, &is_all_stale); CeedChkBackend(ierr);
  if (is_all_stale)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Invalid data in array; must set vector with CeedVectorSetValue or CeedVectorSetArray before calling CeedVectorGetArray");
  // LCOV_EXCL_STOP

  // Set value for synced device/host array
  if (data->d_array) {
    ierr = CeedDeviceReciprocal_Hip(data->d_array, length); CeedChkBackend(ierr);
  }
  if (data->h_array) {
    ierr = CeedHostReciprocal_Hip(data->h_array, length); CeedChkBackend(ierr);
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
  CeedVector_Hip *x_data;
  ierr = CeedVectorGetData(x, &x_data); CeedChkBackend(ierr);
  CeedInt length;
  ierr = CeedVectorGetLength(x, &length); CeedChkBackend(ierr);

  bool is_all_stale = false;
  ierr = CeedVectorIsAllStale_Hip(x, &is_all_stale); CeedChkBackend(ierr);
  if (is_all_stale)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Invalid data in array; must set vector with CeedVectorSetValue or CeedVectorSetArray before calling CeedVectorGetArray");
  // LCOV_EXCL_STOP

  // Set value for synced device/host array
  if (x_data->d_array) {
    ierr = CeedDeviceScale_Hip(x_data->d_array, alpha, length);
    CeedChkBackend(ierr);
  }
  if (x_data->h_array) {
    ierr = CeedHostScale_Hip(x_data->h_array, alpha, length); CeedChkBackend(ierr);
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
  CeedVector_Hip *y_data, *x_data;
  ierr = CeedVectorGetData(y, &y_data); CeedChkBackend(ierr);
  ierr = CeedVectorGetData(x, &x_data); CeedChkBackend(ierr);
  CeedInt length;
  ierr = CeedVectorGetLength(y, &length); CeedChkBackend(ierr);

  bool is_all_stale_x = false, is_all_stale_y = false;
  ierr = CeedVectorIsAllStale_Hip(x, &is_all_stale_x); CeedChkBackend(ierr);
  ierr = CeedVectorIsAllStale_Hip(y, &is_all_stale_y); CeedChkBackend(ierr);
  if (is_all_stale_x || is_all_stale_y)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Invalid data in array; must set vector with CeedVectorSetValue or CeedVectorSetArray before calling CeedVectorGetArray");
  // LCOV_EXCL_STOP

  // Set value for synced device/host array
  if (y_data->d_array) {
    ierr = CeedVectorSyncArray(x, CEED_MEM_DEVICE); CeedChkBackend(ierr);
    ierr = CeedDeviceAXPY_Hip(y_data->d_array, alpha, x_data->d_array, length);
    CeedChkBackend(ierr);
  }
  if (y_data->h_array) {
    ierr = CeedVectorSyncArray(x, CEED_MEM_HOST); CeedChkBackend(ierr);
    ierr = CeedHostAXPY_Hip(y_data->h_array, alpha, x_data->h_array, length);
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
  CeedVector_Hip *w_data, *x_data, *y_data;
  ierr = CeedVectorGetData(w, &w_data); CeedChkBackend(ierr);
  ierr = CeedVectorGetData(x, &x_data); CeedChkBackend(ierr);
  ierr = CeedVectorGetData(y, &y_data); CeedChkBackend(ierr);
  CeedInt length;
  ierr = CeedVectorGetLength(w, &length); CeedChkBackend(ierr);

  bool is_all_stale_x = false, is_all_stale_y = false;
  ierr = CeedVectorIsAllStale_Hip(x, &is_all_stale_x); CeedChkBackend(ierr);
  ierr = CeedVectorIsAllStale_Hip(y, &is_all_stale_y); CeedChkBackend(ierr);
  if (is_all_stale_x || is_all_stale_y)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Invalid data in array; must set vector with CeedVectorSetValue or CeedVectorSetArray before calling CeedVectorGetArray");
  // LCOV_EXCL_STOP

  // Set value for synced device/host array
  if (!w_data->d_array && !w_data->h_array) {
    ierr = CeedVectorSetValue(w, 0.0); CeedChkBackend(ierr);
  }
  if (w_data->d_array) {
    ierr = CeedVectorSyncArray(x, CEED_MEM_DEVICE); CeedChkBackend(ierr);
    ierr = CeedVectorSyncArray(y, CEED_MEM_DEVICE); CeedChkBackend(ierr);
    ierr = CeedDevicePointwiseMult_Hip(w_data->d_array, x_data->d_array,
                                       y_data->d_array, length);
    CeedChkBackend(ierr);
  }
  if (w_data->h_array) {
    ierr = CeedVectorSyncArray(x, CEED_MEM_HOST); CeedChkBackend(ierr);
    ierr = CeedVectorSyncArray(y, CEED_MEM_HOST); CeedChkBackend(ierr);
    ierr = CeedHostPointwiseMult_Hip(w_data->h_array, x_data->h_array,
                                     y_data->h_array, length);
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
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);

  ierr = hipFree(data->d_array_owned); CeedChk_Hip(ceed, ierr);
  ierr = CeedFree(&data->h_array_owned); CeedChkBackend(ierr);
  ierr = CeedFree(&data); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create a vector of the specified length (does not allocate memory)
//------------------------------------------------------------------------------
int CeedVectorCreate_Hip(CeedInt n, CeedVector vec) {
  CeedVector_Hip *data;
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "SetArray",
                                CeedVectorSetArray_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "TakeArray",
                                CeedVectorTakeArray_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "SetValue",
                                (int (*)())(CeedVectorSetValue_Hip)); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArray",
                                CeedVectorGetArray_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayRead",
                                CeedVectorGetArrayRead_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "RestoreArray",
                                CeedVectorRestoreArray_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "RestoreArrayRead",
                                CeedVectorRestoreArrayRead_Hip); CeedChkBackend(ierr);
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

  ierr = CeedCalloc(1, &data); CeedChkBackend(ierr);
  ierr = CeedVectorSetData(vec, data); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}
