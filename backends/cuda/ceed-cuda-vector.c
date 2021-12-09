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
#include "ceed-cuda.h"

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
// Sync host to device
//------------------------------------------------------------------------------
static inline int CeedVectorSyncH2D_Cuda(const CeedVector vec) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);

  if (data->d_array_borrowed) {
    data->d_array = data->d_array_borrowed;
  } else if (data->d_array_owned) {
    data->d_array = data->d_array_owned;
  } else {
    ierr = cudaMalloc((void **)&data->d_array_owned, bytes(vec));
    CeedChk_Cu(ceed, ierr);
    data->d_array = data->d_array_owned;
  }

  if (data->h_array) {
    ierr = cudaMemcpy(data->d_array, data->h_array, bytes(vec),
                      cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);
  } else {
    CeedInt length;
    ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
    ierr = CeedDeviceSetValue_Cuda(data->d_array, length, 0.0);
    CeedChkBackend(ierr);
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync device to host
//------------------------------------------------------------------------------
static inline int CeedVectorSyncD2H_Cuda(const CeedVector vec) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);

  if (data->h_array_borrowed) {
    data->h_array = data->h_array_borrowed;
  } else if (data->h_array_owned) {
    data->h_array = data->h_array_owned;
  } else {
    CeedInt length;
    ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
    ierr = CeedMalloc(length, &data->h_array_owned);  CeedChkBackend(ierr);
    data->h_array = data->h_array_owned;
  }

  if (data->d_array) {
    ierr = cudaMemcpy(data->h_array, data->d_array, bytes(vec),
                      cudaMemcpyDeviceToHost); CeedChk_Cu(ceed, ierr);
  } else {
    CeedInt length;
    ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
    ierr = CeedHostSetValue_Cuda(data->h_array, length, 0.0); CeedChkBackend(ierr);
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set all pointers as stale
//------------------------------------------------------------------------------
static inline int CeedVectorSetAllStale_Cuda(const CeedVector vec) {
  int ierr;
  CeedVector_Cuda *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);

  data->h_array = NULL;
  data->d_array = NULL;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set array from host
//------------------------------------------------------------------------------
static int CeedVectorSetArrayHost_Cuda(const CeedVector vec,
                                       const CeedCopyMode cmode, CeedScalar *array) {
  int ierr;
  CeedVector_Cuda *data;
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
static int CeedVectorSetArrayDevice_Cuda(const CeedVector vec,
    const CeedCopyMode cmode, CeedScalar *array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);

  switch (cmode) {
  case CEED_COPY_VALUES:
    if (!data->d_array_owned) {
      ierr = cudaMalloc((void **)&data->d_array_owned, bytes(vec));
      CeedChk_Cu(ceed, ierr);
      data->d_array = data->d_array_owned;
    }
    if (array) {
      ierr = cudaMemcpy(data->d_array, array, bytes(vec),
                        cudaMemcpyDeviceToDevice); CeedChk_Cu(ceed, ierr);
    }
    break;
  case CEED_OWN_POINTER:
    ierr = cudaFree(data->d_array_owned); CeedChk_Cu(ceed, ierr);
    data->d_array_owned = array;
    data->d_array_borrowed = NULL;
    data->d_array = array;
    break;
  case CEED_USE_POINTER:
    ierr = cudaFree(data->d_array_owned); CeedChk_Cu(ceed, ierr);
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
static int CeedVectorSetArray_Cuda(const CeedVector vec,
                                   const CeedMemType mtype, const CeedCopyMode cmode, CeedScalar *array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);

  ierr = CeedVectorSetAllStale_Cuda(vec); CeedChkBackend(ierr);
  switch (mtype) {
  case CEED_MEM_HOST:
    return CeedVectorSetArrayHost_Cuda(vec, cmode, array);
  case CEED_MEM_DEVICE:
    return CeedVectorSetArrayDevice_Cuda(vec, cmode, array);
  }

  return CEED_ERROR_UNSUPPORTED;
}

//------------------------------------------------------------------------------
// Vector Take Array
//------------------------------------------------------------------------------
static int CeedVectorTakeArray_Cuda(CeedVector vec, CeedMemType mtype,
                                    CeedScalar **array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  switch(mtype) {
  case CEED_MEM_HOST:
    if (!impl->h_array_borrowed)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "Must set HOST array with CeedVectorSetArray and CEED_USE_POINTER before calling CeedVectorTakeArray");
    // LCOV_EXCL_STOP

    if (!impl->h_array && impl->d_array) {
      ierr = CeedVectorSyncD2H_Cuda(vec); CeedChkBackend(ierr);
    }
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

    if (!impl->d_array && impl->h_array) {
      ierr = CeedVectorSyncH2D_Cuda(vec); CeedChkBackend(ierr);
    }
    (*array) = impl->d_array_borrowed;
    impl->d_array_borrowed = NULL;
    impl->d_array = NULL;
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set a vector to a value,
//------------------------------------------------------------------------------
static int CeedVectorSetValue_Cuda(CeedVector vec, CeedScalar val) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *data;
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
    ierr = CeedDeviceSetValue_Cuda(data->d_array, length, val);
    CeedChkBackend(ierr);
    data->h_array = NULL;
  }
  if (data->h_array) {
    ierr = CeedHostSetValue_Cuda(data->h_array, length, val); CeedChkBackend(ierr);
    data->d_array = NULL;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get read-only access to a vector via the specified mtype memory type
//   on which to access the array. If the backend uses a different memory type,
//   this will perform a copy (possibly cached).
//------------------------------------------------------------------------------
static int CeedVectorGetArrayRead_Cuda(const CeedVector vec,
                                       const CeedMemType mtype, const CeedScalar **array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);
  CeedVector_Cuda *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);

  // Sync array to requested memtype and update pointer
  switch (mtype) {
  case CEED_MEM_HOST:
    if (!data->h_array) {
      ierr = CeedVectorSyncD2H_Cuda(vec); CeedChkBackend(ierr);
    }
    *array = data->h_array;
    break;
  case CEED_MEM_DEVICE:
    if (!data->d_array) {
      ierr = CeedVectorSyncH2D_Cuda(vec); CeedChkBackend(ierr);
    }
    *array = data->d_array;
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get array
//------------------------------------------------------------------------------
static int CeedVectorGetArray_Cuda(const CeedVector vec,
                                   const CeedMemType mtype, CeedScalar **array) {
  int ierr;
  CeedVector_Cuda *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);

  ierr = CeedVectorGetArrayRead_Cuda(vec, mtype, (const CeedScalar **)array);
  CeedChkBackend(ierr);

  ierr = CeedVectorSetAllStale_Cuda(vec); CeedChkBackend(ierr);
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
  CeedVector_Cuda *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);
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
    ierr = cudaMemcpy(&normNoAbs, data->d_array+indx-1, sizeof(CeedScalar),
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
  CeedVector_Cuda *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);
  CeedInt length;
  ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);

  // Set value for synced device/host array
  if (data->d_array) {
    ierr = CeedDeviceReciprocal_Cuda(data->d_array, length); CeedChkBackend(ierr);
  }
  if (data->h_array) {
    ierr = CeedHostReciprocal_Cuda(data->h_array, length); CeedChkBackend(ierr);
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
  CeedVector_Cuda *x_data;
  ierr = CeedVectorGetData(x, &x_data); CeedChkBackend(ierr);
  CeedInt length;
  ierr = CeedVectorGetLength(x, &length); CeedChkBackend(ierr);

  // Set value for synced device/host array
  if (x_data->d_array) {
    ierr = CeedDeviceScale_Cuda(x_data->d_array, alpha, length);
    CeedChkBackend(ierr);
  }
  if (x_data->h_array) {
    ierr = CeedHostScale_Cuda(x_data->h_array, alpha, length); CeedChkBackend(ierr);
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
  CeedVector_Cuda *y_data, *x_data;
  ierr = CeedVectorGetData(y, &y_data); CeedChkBackend(ierr);
  ierr = CeedVectorGetData(x, &x_data); CeedChkBackend(ierr);
  CeedInt length;
  ierr = CeedVectorGetLength(y, &length); CeedChkBackend(ierr);

  // Set value for synced device/host array
  if (y_data->d_array) {
    ierr = CeedVectorSyncArray(x, CEED_MEM_DEVICE); CeedChkBackend(ierr);
    ierr = CeedDeviceAXPY_Cuda(y_data->d_array, alpha, x_data->d_array, length);
    CeedChkBackend(ierr);
  }
  if (y_data->h_array) {
    ierr = CeedVectorSyncArray(x, CEED_MEM_HOST); CeedChkBackend(ierr);
    ierr = CeedHostAXPY_Cuda(y_data->h_array, alpha, x_data->h_array, length);
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
  CeedVector_Cuda *w_data, *x_data, *y_data;
  ierr = CeedVectorGetData(w, &w_data); CeedChkBackend(ierr);
  ierr = CeedVectorGetData(x, &x_data); CeedChkBackend(ierr);
  ierr = CeedVectorGetData(y, &y_data); CeedChkBackend(ierr);
  CeedInt length;
  ierr = CeedVectorGetLength(w, &length); CeedChkBackend(ierr);

  // Set value for synced device/host array
  if (!w_data->d_array && w_data->h_array) {
    ierr = CeedVectorSetValue(w, 0.0); CeedChkBackend(ierr);
  }
  if (w_data->d_array) {
    ierr = CeedVectorSyncArray(x, CEED_MEM_DEVICE); CeedChkBackend(ierr);
    ierr = CeedVectorSyncArray(y, CEED_MEM_DEVICE); CeedChkBackend(ierr);
    ierr = CeedDevicePointwiseMult_Cuda(w_data->d_array, x_data->d_array,
                                        y_data->d_array, length);
    CeedChkBackend(ierr);
  }
  if (w_data->h_array) {
    ierr = CeedVectorSyncArray(x, CEED_MEM_HOST); CeedChkBackend(ierr);
    ierr = CeedVectorSyncArray(y, CEED_MEM_HOST); CeedChkBackend(ierr);
    ierr = CeedHostPointwiseMult_Cuda(w_data->h_array, x_data->h_array,
                                      y_data->h_array, length);
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
  CeedVector_Cuda *data;
  ierr = CeedVectorGetData(vec, &data); CeedChkBackend(ierr);

  ierr = cudaFree(data->d_array_owned); CeedChk_Cu(ceed, ierr);
  ierr = CeedFree(&data->h_array_owned); CeedChkBackend(ierr);
  ierr = CeedFree(&data); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create a vector of the specified length (does not allocate memory)
//------------------------------------------------------------------------------
int CeedVectorCreate_Cuda(CeedInt n, CeedVector vec) {
  CeedVector_Cuda *data;
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "SetArray",
                                CeedVectorSetArray_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "TakeArray",
                                CeedVectorTakeArray_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "SetValue",
                                (int (*)())(CeedVectorSetValue_Cuda)); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArray",
                                CeedVectorGetArray_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayRead",
                                CeedVectorGetArrayRead_Cuda); CeedChkBackend(ierr);
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

  ierr = CeedCalloc(1, &data); CeedChkBackend(ierr);
  ierr = CeedVectorSetData(vec, data); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}
