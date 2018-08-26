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
#define CEED_DEBUG_COLOR 11
#include "ceed-occa.h"

// *****************************************************************************
// * Bytes used
// *****************************************************************************
static inline size_t bytes(const CeedVector vec) {
  return vec->length * sizeof(CeedScalar);
}

// *****************************************************************************
// * OCCA SYNC functions
// *****************************************************************************
static inline void CeedSyncH2D_Occa(const CeedVector vec) {
  const CeedVector_Occa *data = vec->data;
  assert(data);
  assert(data->h_array);
  occaCopyPtrToMem(data->d_array, data->h_array, bytes(vec), NO_OFFSET,
                   NO_PROPS);
}
// *****************************************************************************
static inline void CeedSyncD2H_Occa(const CeedVector vec) {
  const CeedVector_Occa *data = vec->data;
  assert(data);
  assert(data->h_array);
  occaCopyMemToPtr(data->h_array,data->d_array, bytes(vec), NO_OFFSET, NO_PROPS);
}

// *****************************************************************************
// * Set the array used by a vector,
// * freeing any previously allocated array if applicable
// *****************************************************************************
static int CeedVectorSetArray_Occa(const CeedVector vec,
                                   const CeedMemType mtype,
                                   const CeedCopyMode cmode,
                                   CeedScalar *array) {
  const Ceed ceed = vec->ceed;
  CeedVector_Occa *data = vec->data;
  int ierr;
  dbg("[CeedVector][Set]");
  if (mtype != CEED_MEM_HOST)
    return CeedError(vec->ceed, 1, "Only MemType = HOST supported");
  ierr = CeedFree(&data->h_array_allocated); CeedChk(ierr);
  switch (cmode) {
  // Implementation will copy the values and not store the passed pointer.
  case CEED_COPY_VALUES:
    dbg("\t[CeedVector][Set] CEED_COPY_VALUES");
    ierr = CeedMalloc(vec->length, &data->h_array); CeedChk(ierr);
    data->h_array_allocated = data->h_array;
    if (array) memcpy(data->h_array, array, bytes(vec));
    if (array) CeedSyncH2D_Occa(vec);
    break;
  // Implementation takes ownership of the pointer
  // and will free using CeedFree() when done using it
  case CEED_OWN_POINTER:
    dbg("\t[CeedVector][Set] CEED_OWN_POINTER");
    data->h_array = array;
    data->h_array_allocated = array;
    CeedSyncH2D_Occa(vec);
    break;
  // Implementation can use and modify the data provided by the user
  case CEED_USE_POINTER:
    dbg("\t[CeedVector][Set] CEED_USE_POINTER");
    data->h_array = array;
    CeedSyncH2D_Occa(vec);
    break;
  default: CeedError(ceed,1," OCCA backend no default error");
  }
  dbg("\t[CeedVector][Set] done");
  return 0;
}

// *****************************************************************************
// * Get read-only access to a vector via the specified mtype memory type
// * on which to access the array. If the backend uses a different memory type,
// * this will perform a copy (possibly cached).
// *****************************************************************************
static int CeedVectorGetArrayRead_Occa(const CeedVector vec,
                                       const CeedMemType mtype,
                                       const CeedScalar **array) {
  const Ceed ceed = vec->ceed;
  dbg("[CeedVector][Get]");
  CeedVector_Occa *data = vec->data;
  int ierr;
  if (mtype != CEED_MEM_HOST)
    return CeedError(vec->ceed, 1, "Can only provide to HOST memory");
  if (!data->h_array) { // Allocate if array was not allocated yet
    dbg("[CeedVector][Get] Allocating");
    ierr = CeedVectorSetArray(vec, CEED_MEM_HOST, CEED_COPY_VALUES, NULL);
    CeedChk(ierr);
  }
  dbg("[CeedVector][Get] CeedSyncD2H_Occa");
  CeedSyncD2H_Occa(vec);
  *array = data->h_array;
  return 0;
}
// *****************************************************************************
static int CeedVectorGetArray_Occa(const CeedVector vec,
                                   const CeedMemType mtype,
                                   CeedScalar **array) {
  return CeedVectorGetArrayRead_Occa(vec,mtype,(const CeedScalar**)array);
}

// *****************************************************************************
// * Restore an array obtained using CeedVectorGetArray()
// *****************************************************************************
static int CeedVectorRestoreArrayRead_Occa(const CeedVector vec,
    const CeedScalar **array) {
  const Ceed ceed = vec->ceed;
  dbg("[CeedVector][Restore]");
  assert(((CeedVector_Occa *)vec->data)->h_array);
  assert(*array);
  CeedSyncH2D_Occa(vec); // sync Host to Device
  *array = NULL;
  return 0;
}
// *****************************************************************************
static int CeedVectorRestoreArray_Occa(const CeedVector vec,
                                       CeedScalar **array) {
  return CeedVectorRestoreArrayRead_Occa(vec,(const CeedScalar**)array);
}

// *****************************************************************************
// * Destroy the vector
// *****************************************************************************
static int CeedVectorDestroy_Occa(const CeedVector vec) {
  int ierr;
  const Ceed ceed = vec->ceed;
  CeedVector_Occa *data = vec->data;
  dbg("[CeedVector][Destroy]");
  ierr = CeedFree(&data->h_array_allocated); CeedChk(ierr);
  occaFree(data->d_array);
  ierr = CeedFree(&data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * Create a vector of the specified length (does not allocate memory)
// *****************************************************************************
int CeedVectorCreate_Occa(const Ceed ceed, const CeedInt n, CeedVector vec) {
  int ierr;
  const Ceed_Occa *ceed_data = ceed->data;
  CeedVector_Occa *data;
  dbg("[CeedVector][Create] n=%d", n);
  vec->SetArray = CeedVectorSetArray_Occa;
  vec->GetArray = CeedVectorGetArray_Occa;
  vec->GetArrayRead = CeedVectorGetArrayRead_Occa;
  vec->RestoreArray = CeedVectorRestoreArray_Occa;
  vec->RestoreArrayRead = CeedVectorRestoreArrayRead_Occa;
  vec->Destroy = CeedVectorDestroy_Occa;
  // ***************************************************************************
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  vec->data = data;
  data->d_array = occaDeviceMalloc(ceed_data->device, bytes(vec),NULL,NO_PROPS);
  return 0;
}
