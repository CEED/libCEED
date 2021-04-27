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

  ierr = hipMemcpy(data->h_array, data->d_array, bytes(vec),
                   hipMemcpyDeviceToHost); CeedChk_Hip(ceed, ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set array from host
//------------------------------------------------------------------------------
static int CeedVectorSetArrayHost_Hip(const CeedVector vec,
                                      const CeedCopyMode cmode,
                                      CeedScalar *array) {
  int ierr;
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);

  switch (cmode) {
  case CEED_COPY_VALUES: {
    CeedInt length;
    if(!data->h_array) {
      ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
      ierr = CeedMalloc(length, &data->h_array_allocated); CeedChkBackend(ierr);
      data->h_array = data->h_array_allocated;
    }
    if (array)
      memcpy(data->h_array, array, bytes(vec));
  } break;
  case CEED_OWN_POINTER:
    ierr = CeedFree(&data->h_array_allocated); CeedChkBackend(ierr);
    data->h_array_allocated = array;
    data->h_array = array;
    break;
  case CEED_USE_POINTER:
    ierr = CeedFree(&data->h_array_allocated); CeedChkBackend(ierr);
    data->h_array = array;
    break;
  }
  data->memState = CEED_HIP_HOST_SYNC;
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
    if (!data->d_array) {
      ierr = hipMalloc((void **)&data->d_array_allocated, bytes(vec));
      CeedChk_Hip(ceed, ierr);
      data->d_array = data->d_array_allocated;
    }
    if (array) {
      ierr = hipMemcpy(data->d_array, array, bytes(vec),
                       hipMemcpyDeviceToDevice); CeedChk_Hip(ceed, ierr);
    }
    break;
  case CEED_OWN_POINTER:
    ierr = hipFree(data->d_array_allocated); CeedChk_Hip(ceed, ierr);
    data->d_array_allocated = array;
    data->d_array = array;
    break;
  case CEED_USE_POINTER:
    ierr = hipFree(data->d_array_allocated); CeedChk_Hip(ceed, ierr);
    data->d_array_allocated = NULL;
    data->d_array = array;
    break;
  }
  data->memState = CEED_HIP_DEVICE_SYNC;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set the array used by a vector,
//   freeing any previously allocated array if applicable
//------------------------------------------------------------------------------
static int CeedVectorSetArray_Hip(const CeedVector vec,
                                  const CeedMemType mtype,
                                  const CeedCopyMode cmode,
                                  CeedScalar *array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);

  switch (mtype) {
  case CEED_MEM_HOST:
    return CeedVectorSetArrayHost_Hip(vec, cmode, array);
  case CEED_MEM_DEVICE:
    return CeedVectorSetArrayDevice_Hip(vec, cmode, array);
  }
  return 1;
}

//------------------------------------------------------------------------------
// Vector Take Array
//------------------------------------------------------------------------------
static int CeedVectorTakeArray_Hip(CeedVector vec, CeedMemType mtype,
                                   CeedScalar **array) {
  int ierr;
  CeedVector_Hip *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  switch(mtype) {
  case CEED_MEM_HOST:
    if (impl->memState == CEED_HIP_DEVICE_SYNC) {
      ierr = CeedVectorSyncD2H_Hip(vec); CeedChkBackend(ierr);
    }
    (*array) = impl->h_array;
    impl->h_array = NULL;
    impl->h_array_allocated = NULL;
    impl->memState = CEED_HIP_HOST_SYNC;
    break;
  case CEED_MEM_DEVICE:
    if (impl->memState == CEED_HIP_HOST_SYNC) {
      ierr = CeedVectorSyncH2D_Hip(vec); CeedChkBackend(ierr);
    }
    (*array) = impl->d_array;
    impl->d_array = NULL;
    impl->d_array_allocated = NULL;
    impl->memState = CEED_HIP_DEVICE_SYNC;
    break;
  }

  return CEED_ERROR_SUCCESS;
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
  switch(data->memState) {
  case CEED_HIP_HOST_SYNC:
    ierr = CeedHostSetValue_Hip(data->h_array, length, val); CeedChkBackend(ierr);
    break;
  case CEED_HIP_NONE_SYNC:
    /*
      Handles the case where SetValue is used without SetArray.
      Default allocation then happens on the GPU.
    */
    if (data->d_array == NULL) {
      ierr = hipMalloc((void **)&data->d_array_allocated, bytes(vec));
      CeedChk_Hip(ceed, ierr);
      data->d_array = data->d_array_allocated;
    }
    data->memState = CEED_HIP_DEVICE_SYNC;
    ierr = CeedDeviceSetValue_Hip(data->d_array, length, val); CeedChkBackend(ierr);
    break;
  case CEED_HIP_DEVICE_SYNC:
    ierr = CeedDeviceSetValue_Hip(data->d_array, length, val); CeedChkBackend(ierr);
    break;
  case CEED_HIP_BOTH_SYNC:
    ierr = CeedHostSetValue_Hip(data->h_array, length, val); CeedChkBackend(ierr);
    ierr = CeedDeviceSetValue_Hip(data->d_array, length, val); CeedChkBackend(ierr);
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
                                      const CeedMemType mtype,
                                      const CeedScalar **array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);

  // Sync array to requested memtype and update pointer
  switch (mtype) {
  case CEED_MEM_HOST:
    if(data->h_array==NULL) {
      CeedInt length;
      ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
      ierr = CeedMalloc(length, &data->h_array_allocated);
      CeedChkBackend(ierr);
      data->h_array = data->h_array_allocated;
    }
    if(data->memState==CEED_HIP_DEVICE_SYNC) {
      ierr = CeedVectorSyncD2H_Hip(vec);
      CeedChkBackend(ierr);
      data->memState = CEED_HIP_BOTH_SYNC;
    }
    *array = data->h_array;
    break;
  case CEED_MEM_DEVICE:
    if (data->d_array==NULL) {
      ierr = hipMalloc((void **)&data->d_array_allocated, bytes(vec));
      CeedChk_Hip(ceed, ierr);
      data->d_array = data->d_array_allocated;
    }
    if (data->memState==CEED_HIP_HOST_SYNC) {
      ierr = CeedVectorSyncH2D_Hip(vec);
      CeedChkBackend(ierr);
      data->memState = CEED_HIP_BOTH_SYNC;
    }
    *array = data->d_array;
    break;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get array
//------------------------------------------------------------------------------
static int CeedVectorGetArray_Hip(const CeedVector vec,
                                  const CeedMemType mtype,
                                  CeedScalar **array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);

  // Sync array to requested memtype and update pointer
  switch (mtype) {
  case CEED_MEM_HOST:
    if(data->h_array==NULL) {
      CeedInt length;
      ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
      ierr = CeedMalloc(length, &data->h_array_allocated);
      CeedChkBackend(ierr);
      data->h_array = data->h_array_allocated;
    }
    if(data->memState==CEED_HIP_DEVICE_SYNC) {
      ierr = CeedVectorSyncD2H_Hip(vec); CeedChkBackend(ierr);
    }
    data->memState = CEED_HIP_HOST_SYNC;
    *array = data->h_array;
    break;
  case CEED_MEM_DEVICE:
    if (data->d_array==NULL) {
      ierr = hipMalloc((void **)&data->d_array_allocated, bytes(vec));
      CeedChk_Hip(ceed, ierr);
      data->d_array = data->d_array_allocated;
    }
    if (data->memState==CEED_HIP_HOST_SYNC) {
      ierr = CeedVectorSyncH2D_Hip(vec); CeedChkBackend(ierr);
    }
    data->memState = CEED_HIP_DEVICE_SYNC;
    *array = data->d_array;
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
    ierr = hipblasDasum(handle, length, d_array, 1, norm);
    CeedChk_Hipblas(ceed, ierr);
    break;
  }
  case CEED_NORM_2: {
    ierr = hipblasDnrm2(handle, length, d_array, 1, norm);
    CeedChk_Hipblas(ceed, ierr);
    break;
  }
  case CEED_NORM_MAX: {
    CeedInt indx;
    ierr = hipblasIdamax(handle, length, d_array, 1, &indx);
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

  // Set value for synced device/host array
  switch(data->memState) {
  case CEED_HIP_HOST_SYNC:
    ierr = CeedHostReciprocal_Hip(data->h_array, length); CeedChkBackend(ierr);
    break;
  case CEED_HIP_DEVICE_SYNC:
    ierr = CeedDeviceReciprocal_Hip(data->d_array, length); CeedChkBackend(ierr);
    break;
  case CEED_HIP_BOTH_SYNC:
    ierr = CeedDeviceReciprocal_Hip(data->d_array, length); CeedChkBackend(ierr);
    data->memState = CEED_HIP_DEVICE_SYNC;
    break;
  // LCOV_EXCL_START
  case CEED_HIP_NONE_SYNC:
    break; // Not possible, but included for completness
    // LCOV_EXCL_STOP
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

  // Set value for synced device/host array
  switch(y_data->memState) {
  case CEED_HIP_HOST_SYNC:
    ierr = CeedVectorSyncArray(x, CEED_MEM_HOST); CeedChkBackend(ierr);
    ierr = CeedHostAXPY_Hip(y_data->h_array, alpha, x_data->h_array, length);
    CeedChkBackend(ierr);
    break;
  case CEED_HIP_DEVICE_SYNC:
    ierr = CeedVectorSyncArray(x, CEED_MEM_DEVICE); CeedChkBackend(ierr);
    ierr = CeedDeviceAXPY_Hip(y_data->d_array, alpha, x_data->d_array, length);
    CeedChkBackend(ierr);
    break;
  case CEED_HIP_BOTH_SYNC:
    ierr = CeedVectorSyncArray(x, CEED_MEM_DEVICE); CeedChkBackend(ierr);
    ierr = CeedDeviceAXPY_Hip(y_data->d_array, alpha, x_data->d_array, length);
    CeedChkBackend(ierr);
    y_data->memState = CEED_HIP_DEVICE_SYNC;
    break;
  // LCOV_EXCL_START
  case CEED_HIP_NONE_SYNC:
    break; // Not possible, but included for completness
    // LCOV_EXCL_STOP
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

  // Set value for synced device/host array
  switch(w_data->memState) {
  case CEED_HIP_HOST_SYNC:
    ierr = CeedVectorSyncArray(x, CEED_MEM_HOST); CeedChkBackend(ierr);
    ierr = CeedVectorSyncArray(y, CEED_MEM_HOST); CeedChkBackend(ierr);
    ierr = CeedHostPointwiseMult_Hip(w_data->h_array, x_data->h_array,
                                     y_data->h_array, length);
    CeedChkBackend(ierr);
    break;
  case CEED_HIP_DEVICE_SYNC:
    ierr = CeedVectorSyncArray(x, CEED_MEM_DEVICE); CeedChkBackend(ierr);
    ierr = CeedVectorSyncArray(y, CEED_MEM_DEVICE); CeedChkBackend(ierr);
    ierr = CeedDevicePointwiseMult_Hip(w_data->d_array, x_data->d_array,
                                       y_data->d_array, length);
    CeedChkBackend(ierr);
    break;
  case CEED_HIP_BOTH_SYNC:
    ierr = CeedVectorSyncArray(x, CEED_MEM_DEVICE); CeedChkBackend(ierr);
    ierr = CeedVectorSyncArray(y, CEED_MEM_DEVICE); CeedChkBackend(ierr);
    ierr = CeedDevicePointwiseMult_Hip(w_data->d_array, x_data->d_array,
                                       y_data->d_array, length);
    CeedChkBackend(ierr);
    w_data->memState = CEED_HIP_DEVICE_SYNC;
    break;
  case CEED_HIP_NONE_SYNC:
    ierr = CeedVectorSetValue(w, 0.0); CeedChkBackend(ierr);
    ierr = CeedVectorSyncArray(x, CEED_MEM_DEVICE); CeedChkBackend(ierr);
    ierr = CeedVectorSyncArray(y, CEED_MEM_DEVICE); CeedChkBackend(ierr);
    ierr = CeedDevicePointwiseMult_Hip(w_data->d_array, x_data->d_array,
                                       y_data->d_array, length);
    CeedChkBackend(ierr);
    break;
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

  ierr = hipFree(data->d_array_allocated); CeedChk_Hip(ceed, ierr);
  ierr = CeedFree(&data->h_array_allocated); CeedChkBackend(ierr);
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
                                CeedVectorSetValue_Hip); CeedChkBackend(ierr);
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
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "AXPY",
                                CeedVectorAXPY_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "PointwiseMult",
                                CeedVectorPointwiseMult_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "Destroy",
                                CeedVectorDestroy_Hip); CeedChkBackend(ierr);

  ierr = CeedCalloc(1, &data); CeedChkBackend(ierr);
  ierr = CeedVectorSetData(vec, data); CeedChkBackend(ierr);
  data->memState = CEED_HIP_NONE_SYNC;
  return CEED_ERROR_SUCCESS;
}
