// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
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

#include "ceed-occa.h"

// *****************************************************************************
// * VECTORS: - Create, Destroy,
// *          - Restore w/ & w/o const
// *          - Set, Get w/ & w/o const
// *****************************************************************************

// *****************************************************************************
// * Bytes used
// *****************************************************************************
static inline size_t bytes(const CeedVector vec) {
  return vec->length * sizeof(CeedScalar);
}

// *****************************************************************************
// * OCCA SYNC functions
// * Ptr == void*, Mem == device
// * occaCopyPtrToMem(occaMemory dest, const void *src,
// * occaCopyMemToPtr(void *dest, occaMemory src,
// *****************************************************************************
static inline void occaSyncH2D(const CeedVector vec) {
  const CeedVector_Occa *occa = vec->data;
  assert(occa);
  assert(occa->h_array);
  assert(occa->d_array);
  occaCopyPtrToMem(*occa->d_array, occa->h_array, bytes(vec), NO_OFFSET, NO_PROPS);
}
static inline void occaSyncD2H(const CeedVector vec) {
  const CeedVector_Occa *occa = vec->data;
  assert(occa);
  assert(occa->h_array);
  assert(occa->d_array);
  occaCopyMemToPtr(occa->h_array,*occa->d_array, bytes(vec), NO_OFFSET, NO_PROPS);
}

// *****************************************************************************
// * Set the array used by a vector,
// * freeing any previously allocated array if applicable
// *****************************************************************************
static int CeedVectorSetArray_Occa(const CeedVector x,
                                   const CeedMemType mtype,
                                   const CeedCopyMode cmode,
                                   CeedScalar *array) {
  CeedVector_Occa *data = x->data;
  int ierr;

  CeedDebug("\033[33m[CeedVectorOcca][Set]");
  if (mtype != CEED_MEM_HOST)
    return CeedError(x->ceed, 1, "Only MemType = HOST supported");
  ierr = CeedFree(&data->h_array); CeedChk(ierr);
  switch (cmode) {
    // Implementation will copy the values and not store the passed pointer.
  case CEED_COPY_VALUES:
    CeedDebug("\t\033[33m[CeedVectorOcca][Set] CEED_COPY_VALUES");
    ierr = CeedMalloc(x->length, &data->h_array); CeedChk(ierr);
    assert(data->h_array);
    if (array) memcpy(data->h_array, array, bytes(x));
    if (array) occaSyncH2D(x);
    break;
    // Implementation takes ownership of the pointer
    // and will free using CeedFree() when done using it
  case CEED_OWN_POINTER:
    CeedDebug("\t\033[33m[CeedVectorOcca][Set] CEED_OWN_POINTER");
    data->h_array = array;
    occaSyncH2D(x);
    break;
    // Implementation can use and modify the data provided by the user
  case CEED_USE_POINTER: 
    CeedDebug("\t\033[33m[CeedVectorOcca][Set] CEED_USE_POINTER");
    data->h_array = array;
    occaSyncH2D(x);
    data->h_array = NULL; // but does not take ownership.
    break;
  default: CeedError(x->ceed,1," OCCA backend no default error");
  }
  CeedDebug("\t\033[33m[CeedVectorOcca][Set] done");
  return 0;
}

// *****************************************************************************
// * Get read-only access to a vector via the specified mtype memory type
// * on which to access the array. If the backend uses a different memory type,
// * this will perform a copy (possibly cached).
// *****************************************************************************
static int CeedVectorGetArrayRead_Occa(const CeedVector x,
                                   const CeedMemType mtype,
                                   const CeedScalar **array) {
  CeedDebug("\033[33m[CeedVector][Get]");
  CeedVector_Occa *occa = x->data;
  int ierr;

  if (mtype != CEED_MEM_HOST)
    return CeedError(x->ceed, 1, "Can only provide to HOST memory");
  if (!occa->h_array) { // Allocate if array was not allocated yet
    CeedDebug("\033[33m[CeedVector][Get] Allocating");
    ierr = CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, NULL);
    CeedChk(ierr);
  }
  CeedDebug("\033[33m[CeedVector][Get] occaSyncH2D");
  occaSyncD2H(x); // sync Device to Host
  *array = occa->h_array;
  return 0;
}
// *****************************************************************************
static int CeedVectorGetArray_Occa(const CeedVector x,
                                   const CeedMemType mtype,
                                   CeedScalar **array) {
  return CeedVectorGetArrayRead_Occa(x,mtype,(const CeedScalar**)array);
}

// *****************************************************************************
// * Restore an array obtained using CeedVectorGetArray()
// *****************************************************************************
static int CeedVectorRestoreArrayRead_Occa(const CeedVector x,
                                           const CeedScalar **array) {
  CeedDebug("\033[33m[CeedVector][Restore]");
  CeedVector_Occa *occa = x->data;
  assert(occa->h_array);
  assert(*array);
  occaSyncH2D(x); // sync Host to Device
  *array = NULL;
  return 0;
}
// *****************************************************************************
static int CeedVectorRestoreArray_Occa(const CeedVector x,
                                       CeedScalar **array) {
  return CeedVectorRestoreArrayRead_Occa(x,(const CeedScalar**)array);
}

// *****************************************************************************
// * Destroy the vector
// *****************************************************************************
static int CeedVectorDestroy_Occa(const CeedVector vec) {
  int ierr;
  CeedVector_Occa *occa = vec->data;
  CeedDebug("\033[33m[CeedVector][Destroy]");
  occaMemoryFree(*occa->d_array);
  ierr = CeedFree(&occa->h_array); CeedChk(ierr);
  ierr = CeedFree(&occa->d_array); CeedChk(ierr);
  ierr = CeedFree(&occa); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * Create a vector of the specified length (does not allocate memory)
// *****************************************************************************
int CeedVectorCreate_Occa(const Ceed ceed, const CeedInt n, CeedVector vec) {
  int ierr;
  CeedVector_Occa *data;
  const Ceed_Occa *occa=ceed->data;
  CeedDebug("\033[33m[CeedVector][Create] n=%d", n);
  vec->SetArray = CeedVectorSetArray_Occa;
  vec->GetArray = CeedVectorGetArray_Occa;
  vec->GetArrayRead = CeedVectorGetArrayRead_Occa;
  vec->RestoreArray = CeedVectorRestoreArray_Occa;
  vec->RestoreArrayRead = CeedVectorRestoreArrayRead_Occa;
  vec->Destroy = CeedVectorDestroy_Occa;
  // ***************************************************************************
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  vec->data = data;
  // ***************************************************************************
  ierr = CeedCalloc(1,&data->d_array); CeedChk(ierr);
  *data->d_array = occaDeviceMalloc(*occa->device, bytes(vec), NULL, NO_PROPS);
  // Flush device memory *******************************************************
  ierr=CeedCalloc(vec->length, &data->h_array); CeedChk(ierr);
  assert(data->h_array);
  occaSyncH2D(vec);
  ierr = CeedFree(&data->h_array); CeedChk(ierr);
  return 0;
}
