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

#include "ceed-hip.h"
#include <string.h>

//------------------------------------------------------------------------------
// * Bytes used
//------------------------------------------------------------------------------
static inline size_t bytes(const CeedVector vec) {
  int ierr;
  CeedInt length;
  ierr = CeedVectorGetLength(vec, &length); CeedChk(ierr);
  return length * sizeof(CeedScalar);
}

//------------------------------------------------------------------------------
// Sync host to device
//------------------------------------------------------------------------------
static inline int CeedSyncH2D_Hip(const CeedVector vec) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, (void *)&data); CeedChk(ierr);

  ierr = hipMemcpy(data->d_array, data->h_array, bytes(vec),
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);
  return 0;
}

//------------------------------------------------------------------------------
// Sync device to host
//------------------------------------------------------------------------------
static inline int CeedSyncD2H_Hip(const CeedVector vec) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, (void *)&data); CeedChk(ierr);

  ierr = hipMemcpy(data->h_array, data->d_array, bytes(vec),
                   hipMemcpyDeviceToHost); CeedChk_Hip(ceed, ierr);
  return 0;
}

//------------------------------------------------------------------------------
// Set array from host
//------------------------------------------------------------------------------
static int CeedVectorSetArrayHost_Hip(const CeedVector vec,
                                      const CeedCopyMode cmode,
                                      CeedScalar *array) {
  int ierr;
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, (void *)&data); CeedChk(ierr);

  switch (cmode) {
  case CEED_COPY_VALUES: {
    CeedInt length;
    if(!data->h_array) {
      ierr = CeedVectorGetLength(vec, &length); CeedChk(ierr);
      ierr = CeedMalloc(length, &data->h_array_allocated); CeedChk(ierr);
      data->h_array = data->h_array_allocated;
    }
    if (array)
      memcpy(data->h_array, array, bytes(vec));
  } break;
  case CEED_OWN_POINTER:
    ierr = CeedFree(&data->h_array_allocated); CeedChk(ierr);
    data->h_array_allocated = array;
    data->h_array = array;
    break;
  case CEED_USE_POINTER:
    ierr = CeedFree(&data->h_array_allocated); CeedChk(ierr);
    data->h_array = array;
    break;
  }
  data->memState = CEED_HIP_HOST_SYNC;
  return 0;
}

//------------------------------------------------------------------------------
// Set array from device
//------------------------------------------------------------------------------
static int CeedVectorSetArrayDevice_Hip(const CeedVector vec,
                                        const CeedCopyMode cmode, CeedScalar *array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, (void *)&data); CeedChk(ierr);

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
  return 0;
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
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, (void *)&data); CeedChk(ierr);

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
  ierr = CeedVectorGetData(vec, (void *)&impl); CeedChk(ierr);

  switch(mtype) {
  case CEED_MEM_HOST:
    if (impl->memState == CEED_HIP_DEVICE_SYNC) {
      ierr = CeedSyncD2H_Hip(vec); CeedChk(ierr);
    }
    (*array) = impl->h_array;
    impl->h_array = NULL;
    impl->h_array_allocated = NULL;
    impl->memState = CEED_HIP_HOST_SYNC;
    break;
  case CEED_MEM_DEVICE:
    if (impl->memState == CEED_HIP_HOST_SYNC) {
      ierr = CeedSyncH2D_Hip(vec); CeedChk(ierr);
    }
    (*array) = impl->d_array;
    impl->d_array = NULL;
    impl->d_array_allocated = NULL;
    impl->memState = CEED_HIP_DEVICE_SYNC;
    break;
  }

  return 0;
}

//------------------------------------------------------------------------------
// Set host array to value
//------------------------------------------------------------------------------
static int CeedHostSetValue(CeedScalar *h_array, CeedInt length,
                            CeedScalar val) {
  for (int i = 0; i < length; i++)
    h_array[i] = val;
  return 0;
}

//------------------------------------------------------------------------------
// Set device array to value (impl in .hip file)
//------------------------------------------------------------------------------
int CeedDeviceSetValue(CeedScalar *d_array, CeedInt length, CeedScalar val);

//------------------------------------------------------------------------------
// Set a vector to a value,
//------------------------------------------------------------------------------
static int CeedVectorSetValue_Hip(CeedVector vec, CeedScalar val) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, (void *)&data); CeedChk(ierr);
  CeedInt length;
  ierr = CeedVectorGetLength(vec, &length); CeedChk(ierr);

  // Set value for synced device/host array
  switch(data->memState) {
  case CEED_HIP_HOST_SYNC:
    ierr = CeedHostSetValue(data->h_array, length, val); CeedChk(ierr);
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
    ierr = CeedDeviceSetValue(data->d_array, length, val); CeedChk(ierr);
    break;
  case CEED_HIP_DEVICE_SYNC:
    ierr = CeedDeviceSetValue(data->d_array, length, val); CeedChk(ierr);
    break;
  case CEED_HIP_BOTH_SYNC:
    ierr = CeedHostSetValue(data->h_array, length, val); CeedChk(ierr);
    ierr = CeedDeviceSetValue(data->d_array, length, val); CeedChk(ierr);
    break;
  }
  return 0;
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
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, (void *)&data); CeedChk(ierr);

  // Sync array to requested memtype and update pointer
  switch (mtype) {
  case CEED_MEM_HOST:
    if(data->h_array==NULL) {
      CeedInt length;
      ierr = CeedVectorGetLength(vec, &length); CeedChk(ierr);
      ierr = CeedMalloc(length, &data->h_array_allocated);
      CeedChk(ierr);
      data->h_array = data->h_array_allocated;
    }
    if(data->memState==CEED_HIP_DEVICE_SYNC) {
      ierr = CeedSyncD2H_Hip(vec);
      CeedChk(ierr);
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
      ierr = CeedSyncH2D_Hip(vec);
      CeedChk(ierr);
      data->memState = CEED_HIP_BOTH_SYNC;
    }
    *array = data->d_array;
    break;
  }
  return 0;
}

//------------------------------------------------------------------------------
// Get array
//------------------------------------------------------------------------------
static int CeedVectorGetArray_Hip(const CeedVector vec,
                                  const CeedMemType mtype,
                                  CeedScalar **array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, (void *)&data); CeedChk(ierr);

  // Sync array to requested memtype and update pointer
  switch (mtype) {
  case CEED_MEM_HOST:
    if(data->h_array==NULL) {
      CeedInt length;
      ierr = CeedVectorGetLength(vec, &length); CeedChk(ierr);
      ierr = CeedMalloc(length, &data->h_array_allocated);
      CeedChk(ierr);
      data->h_array = data->h_array_allocated;
    }
    if(data->memState==CEED_HIP_DEVICE_SYNC) {
      ierr = CeedSyncD2H_Hip(vec); CeedChk(ierr);
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
      ierr = CeedSyncH2D_Hip(vec); CeedChk(ierr);
    }
    data->memState = CEED_HIP_DEVICE_SYNC;
    *array = data->d_array;
    break;
  }
  return 0;
}

//------------------------------------------------------------------------------
// Restore an array obtained using CeedVectorGetArrayRead()
//------------------------------------------------------------------------------
static int CeedVectorRestoreArrayRead_Hip(const CeedVector vec) {
  return 0;
}

//------------------------------------------------------------------------------
// Restore an array obtained using CeedVectorGetArray()
//------------------------------------------------------------------------------
static int CeedVectorRestoreArray_Hip(const CeedVector vec) {
  return 0;
}

//------------------------------------------------------------------------------
// Destroy the vector
//------------------------------------------------------------------------------
static int CeedVectorDestroy_Hip(const CeedVector vec) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);
  CeedVector_Hip *data;
  ierr = CeedVectorGetData(vec, (void *)&data); CeedChk(ierr);

  ierr = hipFree(data->d_array_allocated); CeedChk_Hip(ceed, ierr);
  ierr = CeedFree(&data->h_array_allocated); CeedChk(ierr);
  ierr = CeedFree(&data); CeedChk(ierr);
  return 0;
}

//------------------------------------------------------------------------------
// Create a vector of the specified length (does not allocate memory)
//------------------------------------------------------------------------------
int CeedVectorCreate_Hip(CeedInt n, CeedVector vec) {
  CeedVector_Hip *data;
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "SetArray",
                                CeedVectorSetArray_Hip); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "TakeArray",
                                CeedVectorTakeArray_Hip); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "SetValue",
                                CeedVectorSetValue_Hip); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArray",
                                CeedVectorGetArray_Hip); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayRead",
                                CeedVectorGetArrayRead_Hip); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "RestoreArray",
                                CeedVectorRestoreArray_Hip); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "RestoreArrayRead",
                                CeedVectorRestoreArrayRead_Hip); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "Destroy",
                                CeedVectorDestroy_Hip); CeedChk(ierr);

  ierr = CeedCalloc(1, &data); CeedChk(ierr);
  ierr = CeedVectorSetData(vec, (void *)&data); CeedChk(ierr);
  data->memState = CEED_HIP_NONE_SYNC;
  return 0;
}
