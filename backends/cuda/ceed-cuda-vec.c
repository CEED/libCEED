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
  ierr = cudaMalloc((void**)&data->d_array_allocated, bytes(vec)); CeedChk(ierr);
  data->d_array = data->d_array_allocated;

  switch (cmode) {
    case CEED_COPY_VALUES:
      ierr = cudaMallocHost((void**)&data->h_array_allocated, bytes(vec)); CeedChk(ierr);
      data->h_array = data->h_array_allocated;

      if (array) memcpy(data->h_array, array, bytes(vec));
      if (array) CeedSyncH2D_Cuda(vec);
      break;
    case CEED_OWN_POINTER:
      data->h_array_allocated = array;
      data->h_array = array;
      CeedSyncH2D_Cuda(vec);
      break;
    case CEED_USE_POINTER:
      data->h_array = array;
      CeedSyncH2D_Cuda(vec);
      break;
  }
  return 0;
}

static int CeedVectorSetArrayDevice_Cuda(const CeedVector vec,
    const CeedCopyMode cmode, CeedScalar *array) {
  int ierr;
  CeedVector_Cuda *data = (CeedVector_Cuda*)vec->data;
  ierr = cudaMallocHost((void**)&data->h_array_allocated, bytes(vec)); CeedChk(ierr);
  data->h_array = data->h_array_allocated;

  switch (cmode) {
    case CEED_COPY_VALUES:
      ierr = cudaMalloc((void**)&data->d_array_allocated, bytes(vec)); CeedChk(ierr);
      data->d_array = data->d_array_allocated;

      if (array) cudaMemcpy(data->d_array, array, bytes(vec),
          cudaMemcpyDeviceToDevice);
      if (array) CeedSyncD2H_Cuda(vec);
      break;
    case CEED_OWN_POINTER:
      data->d_array_allocated = array;
      data->d_array = array;
      CeedSyncD2H_Cuda(vec);
      break;
    case CEED_USE_POINTER:
      data->d_array = array;
      CeedSyncD2H_Cuda(vec);
      break;
  }
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
  ierr = cudaFree(data->d_array_allocated); CeedChk(ierr);

  switch (mtype) {
    case CEED_MEM_HOST:
      return CeedVectorSetArrayHost_Cuda(vec, cmode, array);
    case CEED_MEM_DEVICE:
      return CeedVectorSetArrayDevice_Cuda(vec, cmode, array);
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

  if (!data->h_array || !data->d_array) {
    ierr = CeedVectorSetArray(vec, CEED_MEM_HOST, CEED_COPY_VALUES, NULL);
    CeedChk(ierr);
  }

  switch (mtype) {
    case CEED_MEM_HOST:
      CeedSyncD2H_Cuda(vec);
      *array = data->h_array;
      break;
    case CEED_MEM_DEVICE:
      // TODO: Should a copy occur?
      // CeedSyncH2D_Cuda(vec);
      *array = data->d_array;
      break;
  }
  return 0;
}

// *****************************************************************************
static int CeedVectorGetArray_Cuda(const CeedVector vec,
                                   const CeedMemType mtype,
                                   CeedScalar **array) {
  return CeedVectorGetArrayRead_Cuda(vec,mtype,(const CeedScalar**)array);
}

// *****************************************************************************
// * Restore an array obtained using CeedVectorGetArray()
// *****************************************************************************
static int CeedVectorRestoreArrayRead_Cuda(const CeedVector vec,
    const CeedScalar **array) {
  *array = NULL;
  return 0;
}
// *****************************************************************************
static int CeedVectorRestoreArray_Cuda(const CeedVector vec,
                                       CeedScalar **array) {
  CeedVector_Cuda *data = (CeedVector_Cuda*)vec->data;
  if (*array == data->h_array) {
    CeedSyncH2D_Cuda(vec);
  } else if (*array == data->d_array) {
    CeedSyncD2H_Cuda(vec);
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
  ierr = cudaFree(data->d_array_allocated); CeedChk(ierr);
  ierr = cudaFreeHost(data->h_array_allocated); CeedChk(ierr);
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
  vec->GetArray = CeedVectorGetArray_Cuda;
  vec->GetArrayRead = CeedVectorGetArrayRead_Cuda;
  vec->RestoreArray = CeedVectorRestoreArray_Cuda;
  vec->RestoreArrayRead = CeedVectorRestoreArrayRead_Cuda;
  vec->Destroy = CeedVectorDestroy_Cuda;
  // ***************************************************************************
  ierr = CeedCalloc(1, &data); CeedChk(ierr);
  vec->data = data;
  return 0;
}
