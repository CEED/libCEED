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
#include "ceed-cuda.cuh"

// *****************************************************************************
// * Bytes used
// *****************************************************************************
static inline size_t bytes(const CeedVector vec) {
  return vec->length * sizeof(CeedScalar);
}

// *****************************************************************************
// * OCCA SYNC functions
// *****************************************************************************
static inline void CeedSyncH2D_Cuda(const CeedVector vec) {
  const CeedVector_Cuda *data = (CeedVector_Cuda*)vec->data;
  assert(data);
  cudaMemcpy(data->d_array, data->h_array, bytes(vec), cudaMemcpyHostToDevice);
}
// *****************************************************************************
static inline void CeedSyncD2H_Cuda(const CeedVector vec) {
  const CeedVector_Cuda *data = (CeedVector_Cuda*)vec->data;
  assert(data);
  cudaMemcpy(data->h_array, data->d_array, bytes(vec), cudaMemcpyDeviceToHost);
}

// *****************************************************************************
// * Set the array used by a vector,
// * freeing any previously allocated array if applicable
// *****************************************************************************
static int CeedVectorSetArray_Cuda(const CeedVector vec,
                                   const CeedMemType mtype,
                                   const CeedCopyMode cmode,
                                   CeedScalar *array) {
  CeedVector_Cuda *data = (CeedVector_Cuda*)vec->data;
  int ierr;
  if (mtype != CEED_MEM_HOST)
    return CeedError(vec->ceed, 1, "Only MemType = HOST supported");
  ierr = CeedFree(&data->h_array); CeedChk(ierr);
  switch (cmode) {
  // Implementation will copy the values and not store the passed pointer.
  case CEED_COPY_VALUES:
    ierr = CeedMalloc(vec->length, &data->h_array); CeedChk(ierr);
    if (array) memcpy(data->h_array, array, bytes(vec));
    if (array) CeedSyncH2D_Cuda(vec);
    break;
  // Implementation takes ownership of the pointer
  // and will free using CeedFree() when done using it
  case CEED_OWN_POINTER:
    data->h_array = array;
    CeedSyncH2D_Cuda(vec);
    break;
  // Implementation can use and modify the data provided by the user
  case CEED_USE_POINTER:
    data->h_array = array;
    data->used_pointer = array;
    CeedSyncH2D_Cuda(vec);
    data->h_array = NULL; // but does not take ownership.
    break;
  default: CeedError(vec->ceed,1," Cuda backend no default error");
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
  CeedVector_Cuda *data = (CeedVector_Cuda*)vec->data;
  int ierr;
  if (mtype != CEED_MEM_HOST)
    return CeedError(vec->ceed, 1, "Can only provide to HOST memory");
  if (!data->h_array) { // Allocate if array was not allocated yet
    ierr = CeedVectorSetArray(vec, CEED_MEM_HOST, CEED_COPY_VALUES, NULL);
    CeedChk(ierr);
  }
  CeedSyncD2H_Cuda(vec);
  *array = data->h_array;
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
  assert(((CeedVector_Cuda *)vec->data)->h_array);
  assert(*array);
  CeedSyncH2D_Cuda(vec); // sync Host to Device
  *array = NULL;
  return 0;
}
// *****************************************************************************
static int CeedVectorRestoreArray_Cuda(const CeedVector vec,
                                       CeedScalar **array) {
  return CeedVectorRestoreArrayRead_Cuda(vec,(const CeedScalar**)array);
}

// *****************************************************************************
// * Destroy the vector
// *****************************************************************************
static int CeedVectorDestroy_Cuda(const CeedVector vec) {
  int ierr;
  CeedVector_Cuda *data = (CeedVector_Cuda*)vec->data;
  ierr = cudaFree(data->d_array); CeedChk(ierr);
  ierr = CeedFree(&data->h_array); CeedChk(ierr);
  ierr = CeedFree(&data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * Create a vector of the specified length (does not allocate memory)
// *****************************************************************************
int CeedVectorCreate_Cuda(const Ceed ceed, const CeedInt n, CeedVector vec) {
  int ierr;
  CeedVector_Cuda *data;
  vec->SetArray = CeedVectorSetArray_Cuda;
  vec->GetArray = CeedVectorGetArray_Cuda;
  vec->GetArrayRead = CeedVectorGetArrayRead_Cuda;
  vec->RestoreArray = CeedVectorRestoreArray_Cuda;
  vec->RestoreArrayRead = CeedVectorRestoreArrayRead_Cuda;
  vec->Destroy = CeedVectorDestroy_Cuda;
  // ***************************************************************************
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  vec->data = data;
  // ***************************************************************************
  data->used_pointer = NULL;
  cudaMalloc(&data->d_array, bytes(vec));
  // Flush device memory *******************************************************
  ierr=CeedCalloc(vec->length, &data->h_array); CeedChk(ierr);
  CeedSyncH2D_Cuda(vec);
  ierr = CeedFree(&data->h_array); CeedChk(ierr);
  return 0;
}
