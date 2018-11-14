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
  CeedInt length;
  CeedVectorGetLength(vec, &length);

  return length * sizeof(CeedScalar);
}

// *****************************************************************************
// * OCCA SYNC functions
// *****************************************************************************
static inline void CeedSyncH2D_Occa(const CeedVector vec) {
  CeedVector_Occa *data;
  CeedVectorGetData(vec, (void*)&data);

  assert(data);
  assert(data->h_array);
  occaCopyPtrToMem(data->d_array, data->h_array, bytes(vec), NO_OFFSET,
                   NO_PROPS);
}
// *****************************************************************************
static inline void CeedSyncD2H_Occa(const CeedVector vec) {
  CeedVector_Occa *data;
  CeedVectorGetData(vec, (void*)&data);

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
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);
  CeedInt length;
  ierr = CeedVectorGetLength(vec, &length); CeedChk(ierr);
  CeedVector_Occa *data;
  ierr = CeedVectorGetData(vec, (void*)&data); CeedChk(ierr);
  dbg("[CeedVector][Set]");
  if (mtype != CEED_MEM_HOST)
    return CeedError(ceed, 1, "Only MemType = HOST supported");
  ierr = CeedFree(&data->h_array_allocated); CeedChk(ierr);
  switch (cmode) {
  // Implementation will copy the values and not store the passed pointer.
  case CEED_COPY_VALUES:
    dbg("\t[CeedVector][Set] CEED_COPY_VALUES");
    ierr = CeedMalloc(length, &data->h_array); CeedChk(ierr);
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
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);
  dbg("[CeedVector][Get]");
  CeedVector_Occa *data;
  ierr = CeedVectorGetData(vec, (void*)&data); CeedChk(ierr);
  if (mtype != CEED_MEM_HOST)
    return CeedError(ceed, 1, "Can only provide to HOST memory");
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
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);
  dbg("[CeedVector][Restore]");
  CeedVector_Occa *data;
  ierr = CeedVectorGetData(vec, (void*)&data); CeedChk(ierr);
  assert((data)->h_array);
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
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);
  CeedVector_Occa *data;
  ierr = CeedVectorGetData(vec, (void*)&data); CeedChk(ierr);
  dbg("[CeedVector][Destroy]");
  ierr = CeedFree(&data->h_array_allocated); CeedChk(ierr);
  occaFree(data->d_array);
  ierr = CeedFree(&data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * Create a vector of the specified length (does not allocate memory)
// *****************************************************************************
int CeedVectorCreate_Occa(const CeedInt n, CeedVector vec) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);
  Ceed_Occa *ceed_data;
  ierr = CeedGetData(ceed, (void*)&ceed_data); CeedChk(ierr);
  CeedVector_Occa *data;
  dbg("[CeedVector][Create] n=%d", n);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "SetArray",
                                CeedVectorSetArray_Occa); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArray",
                                CeedVectorGetArray_Occa); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayRead",
                                CeedVectorGetArrayRead_Occa); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "RestoreArray",
                                CeedVectorRestoreArray_Occa); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "RestoreArrayRead",
                                CeedVectorRestoreArrayRead_Occa); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "Destroy",
                                CeedVectorDestroy_Occa); CeedChk(ierr);
  // ***************************************************************************
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  data->d_array = occaDeviceMalloc(ceed_data->device, bytes(vec),NULL,NO_PROPS);
  ierr = CeedVectorSetData(vec, (void *)&data); CeedChk(ierr);
  return 0;
}
