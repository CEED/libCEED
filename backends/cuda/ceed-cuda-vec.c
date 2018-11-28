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
#include "ceed-cuda.h"
#include "string.h"
// #include "ceed-cuda.cuh"
#include <cuda.h>

// *****************************************************************************
// * Bytes used
// *****************************************************************************
static inline size_t bytes(const CeedVector vec) {
  return vec->length * sizeof(CeedScalar);
}

static inline void CeedSyncH2D_Cuda(const CeedVector vec) {
  CeedVector_Cuda *data = (CeedVector_Cuda*)vec->data;
  cudaMemcpy(data->d_array, data->h_array, bytes(vec), cudaMemcpyHostToDevice);
}

static inline void CeedSyncD2H_Cuda(const CeedVector vec) {
  CeedVector_Cuda *data = (CeedVector_Cuda*)vec->data;
  cudaMemcpy(data->h_array, data->d_array, bytes(vec), cudaMemcpyDeviceToHost);
}

static int CeedVectorSetArrayHost_Cuda(const CeedVector vec,
                                       const CeedCopyMode cmode, CeedScalar *array) {
  int ierr;
  CeedVector_Cuda *data = (CeedVector_Cuda*)vec->data;

  switch (cmode) {
  case CEED_COPY_VALUES:
    ierr = CeedMalloc(vec->length, &data->h_array_allocated); CeedChk(ierr);
    data->h_array = data->h_array_allocated;

    if (array) memcpy(data->h_array, array, bytes(vec));
    break;
  case CEED_OWN_POINTER:
    data->h_array_allocated = array;
    data->h_array = array;
    break;
  case CEED_USE_POINTER:
    data->h_array = array;
    break;
  }
  data->memState = HOST_SYNC;
  return 0;
}

static int CeedVectorSetArrayDevice_Cuda(const CeedVector vec,
    const CeedCopyMode cmode, CeedScalar *array) {
  int ierr;
  CeedVector_Cuda *data = (CeedVector_Cuda*)vec->data;

  switch (cmode) {
  case CEED_COPY_VALUES:
    ierr = cudaMalloc((void**)&data->d_array_allocated, bytes(vec)); CeedChk_Cu(vec->ceed, ierr);
    data->d_array = data->d_array_allocated;

    if (array) cudaMemcpy(data->d_array, array, bytes(vec),
                            cudaMemcpyDeviceToDevice);
    break;
  case CEED_OWN_POINTER:
    data->d_array_allocated = array;
    data->d_array = array;
    break;
  case CEED_USE_POINTER:
    data->d_array = array;
    break;
  }
  data->memState = DEVICE_SYNC;
  return 0;
}

// *****************************************************************************
// * Set the array used by a vector,
// * freeing any previously allocated array if applicable
// *****************************************************************************
static int CeedVectorSetArray_Cuda(const CeedVector vec,
                                   const CeedMemType mtype,
                                   const CeedCopyMode cmode,
                                   CeedScalar *array) {
  int ierr;
  CeedVector_Cuda *data = (CeedVector_Cuda*)vec->data;

  ierr = CeedFree(&data->h_array_allocated); CeedChk(ierr);
  ierr = cudaFree(data->d_array_allocated); CeedChk_Cu(vec->ceed, ierr);

  switch (mtype) {
  case CEED_MEM_HOST:
    return CeedVectorSetArrayHost_Cuda(vec, cmode, array);
  case CEED_MEM_DEVICE:
    return CeedVectorSetArrayDevice_Cuda(vec, cmode, array);
  }
  return 1;
}

// *****************************************************************************
static int HostSetValue(CeedScalar* h_array, CeedInt length, CeedScalar val) {
  for (int i=0; i<length; i++) h_array[i] = val;
  return 0;
}

int DeviceSetValue(CeedScalar* d_array, CeedInt length, CeedScalar val);

// *****************************************************************************
// * Set a vector to a value,
// *****************************************************************************
static int CeedVectorSetValue_Cuda(CeedVector vec, CeedScalar val) {
  int ierr;
  CeedVector_Cuda *data = (CeedVector_Cuda*)vec->data;
  switch(data->memState){
  case HOST_SYNC:
    ierr = HostSetValue(data->h_array, vec->length, val);
    CeedChk(ierr);
    break;
  case DEVICE_SYNC:
    /*
      Handles the case where SetValue is used without SetArray.
      Default allocation then happens on the GPU.
    */
    if (data->d_array==NULL)
    {
      ierr = cudaMalloc((void**)&data->d_array_allocated, bytes(vec));
      CeedChk_Cu(vec->ceed, ierr);
      data->d_array = data->d_array_allocated;
    }
    ierr = DeviceSetValue(data->d_array, vec->length, val);
    CeedChk(ierr);
    break;
  case BOTH_SYNC:
    ierr = HostSetValue(data->h_array, vec->length, val);
    CeedChk(ierr);
    ierr = DeviceSetValue(data->d_array, vec->length, val);
    CeedChk(ierr);
    break;
  }
  return 0;
}


// *****************************************************************************
// * Get read-only access to a vector via the specified mtype memory type
// * on which to access the array. If the backend uses a different memory type,
// * this will perform a copy (possibly cached).
// *****************************************************************************
static int CeedVectorGetArrayRead_Cuda(const CeedVector vec,
                                       const CeedMemType mtype,
                                       const CeedScalar **array) {
  int ierr;
  CeedVector_Cuda *data = (CeedVector_Cuda*)vec->data;

  switch (mtype) {
  case CEED_MEM_HOST:
    if(data->memState==DEVICE_SYNC){
      if(data->h_array==NULL){
        ierr = CeedMalloc(vec->length, &data->h_array_allocated);
        CeedChk(ierr);
        data->h_array = data->h_array_allocated;
      }
      CeedSyncD2H_Cuda(vec);
      data->memState = BOTH_SYNC;
    }
    *array = data->h_array;
    break;
  case CEED_MEM_DEVICE:
    if (data->memState==HOST_SYNC){
      if (data->d_array==NULL)
      {
        ierr = cudaMalloc((void**)&data->d_array_allocated, bytes(vec));
        CeedChk_Cu(vec->ceed, ierr);
        data->d_array = data->d_array_allocated;
      }
      CeedSyncH2D_Cuda(vec);
      data->memState = BOTH_SYNC;
    }
    *array = data->d_array;
    break;
  }
  return 0;
}

// *****************************************************************************
static int CeedVectorGetArray_Cuda(const CeedVector vec,
                                   const CeedMemType mtype,
                                   CeedScalar **array) {
  int ierr;
  CeedVector_Cuda *data = (CeedVector_Cuda*)vec->data;

  switch (mtype) {
  case CEED_MEM_HOST:
    if(data->memState==DEVICE_SYNC){
      if(data->h_array==NULL){
        ierr = CeedMalloc(vec->length, &data->h_array_allocated);
        CeedChk(ierr);
        data->h_array = data->h_array_allocated;
      }
      CeedSyncD2H_Cuda(vec);
      data->memState = HOST_SYNC;
    }
    *array = data->h_array;
    break;
  case CEED_MEM_DEVICE:
    if (data->memState==HOST_SYNC){
      if (data->d_array==NULL)
      {
        ierr = cudaMalloc((void**)&data->d_array_allocated, bytes(vec));
        CeedChk_Cu(vec->ceed, ierr);
        data->d_array = data->d_array_allocated;
      }
      CeedSyncH2D_Cuda(vec);
      data->memState = DEVICE_SYNC;
    }
    *array = data->d_array;
    break;
  }
  return 0;  
}

// *****************************************************************************
// * Restore an array obtained using CeedVectorGetArray()
// *****************************************************************************
static int CeedVectorRestoreArrayRead_Cuda(const CeedVector vec,
    const CeedScalar **array) {
  CeedVector_Cuda *data = (CeedVector_Cuda*)vec->data;
  if ((*array != data->h_array) && (*array != data->d_array)) {
    return CeedError(vec->ceed, 1, "Invalid restore array");
  }
  *array = NULL;
  return 0;
}
// *****************************************************************************
static int CeedVectorRestoreArray_Cuda(const CeedVector vec,
                                       CeedScalar **array) {
  CeedVector_Cuda *data = (CeedVector_Cuda*)vec->data;
  if (*array == data->h_array) {
    data->memState = HOST_SYNC;
  } else if (*array == data->d_array) {
    data->memState = DEVICE_SYNC;
  } else {
    return CeedError(vec->ceed, 1, "Invalid restore array");
  }
  *array = NULL;
  return 0;
}

// *****************************************************************************
// * Destroy the vector
// *****************************************************************************
static int CeedVectorDestroy_Cuda(const CeedVector vec) {
  int ierr;
  CeedVector_Cuda *data = (CeedVector_Cuda*)vec->data;
  ierr = cudaFree(data->d_array_allocated); CeedChk_Cu(vec->ceed, ierr);
  ierr = CeedFree(&data->h_array_allocated); CeedChk(ierr);
  ierr = CeedFree(&data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * Create a vector of the specified length (does not allocate memory)
// *****************************************************************************
int CeedVectorCreate_Cuda(CeedInt n, CeedVector vec) {
  CeedVector_Cuda *data;
  int ierr;

  vec->SetArray = CeedVectorSetArray_Cuda;
  vec->SetValue = CeedVectorSetValue_Cuda;
  vec->GetArray = CeedVectorGetArray_Cuda;
  vec->GetArrayRead = CeedVectorGetArrayRead_Cuda;
  vec->RestoreArray = CeedVectorRestoreArray_Cuda;
  vec->RestoreArrayRead = CeedVectorRestoreArrayRead_Cuda;
  vec->Destroy = CeedVectorDestroy_Cuda;
  // ***************************************************************************
  ierr = CeedCalloc(1, &data); CeedChk(ierr);
  vec->data = data;
  data->memState = DEVICE_SYNC; //Synchronized with the Device by default
  return 0;
}
