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
  assert(occa->array_device);
  occaCopyPtrToMem(*occa->array_device, occa->array, bytes(vec), NO_OFFSET, NO_PROPS);
}
/*static inline void occaSyncD2H(const CeedVector vec) {
  const CeedVector_Occa *impl = vec->data;
  assert(impl);
  assert(impl->array);
  assert(impl->device);
  occaCopyMemToPtr(impl->array, *impl->array_device, bytes(vec), NO_OFFSET, NO_PROPS);
  }*/

// *****************************************************************************
// * OCCA COPY functions
// *****************************************************************************
//static inline void occaCopyH2D(const CeedVector vec, void *from) {
//  const CeedVector_Occa *impl = vec->data;
//  assert(from);
//  assert(impl);
//  assert(impl->device);
//  occaCopyPtrToMem(*impl->array_device, from, bytes(vec), NO_OFFSET, NO_PROPS);
//}
//static inline void occaCopyD2H(const CeedVector vec, void *to) {
// const CeedVector_Occa *impl = vec->data;
//  assert(to);
//  assert(impl);
//  occaCopyMemToPtr(to, *impl->device, bytes(vec), NO_OFFSET, NO_PROPS);
//}

// *****************************************************************************
// * Set
// *****************************************************************************
static int CeedVectorSetArray_Occa(const CeedVector vec,
                                   const CeedMemType mtype,
                                   const CeedCopyMode cmode,
                                   CeedScalar *array) {
  CeedVector_Occa *occa = vec->data;
  int ierr;

  CeedDebug("\033[33m[CeedVector][Set]");
  if (mtype != CEED_MEM_HOST)
    return CeedError(vec->ceed, 1, "Only MemType = HOST supported");
  ierr = CeedFree(&occa->array_allocated); CeedChk(ierr);
  switch (cmode) {
  case CEED_COPY_VALUES:
    CeedDebug("\t\033[33m[CeedVector][Set] CEED_COPY_VALUES");
    ierr = CeedMalloc(vec->length, &occa->array_allocated); CeedChk(ierr);
    if (array) memcpy(occa->array_allocated, array, bytes(vec));
    occa->array = occa->array_allocated;
    if (array) occaSyncH2D(vec);
    break;
  case CEED_OWN_POINTER:
    CeedDebug("\t\033[33m[CeedVector][Set] CEED_OWN_POINTER");
    occa->array_allocated = array;
    occa->array = array;
    occaSyncH2D(vec);
    break;
  case CEED_USE_POINTER:
    CeedDebug("\t\033[33m[CeedVector][Set] CEED_USE_POINTER");
    occa->array = array;
    occaSyncH2D(vec);
    break;
  default: CeedError(vec->ceed,1," OCCA backend no default error");
  }
  return 0;
}

// *****************************************************************************
// * Get/Get+Const
// *****************************************************************************
static int CeedVectorGetArrayRead_Occa(const CeedVector vec,
                                   const CeedMemType mtype,
                                   const CeedScalar **array) {
  CeedDebug("\033[33m[CeedVector][Get]");
  CeedVector_Occa *occa = vec->data;
  int ierr;

  if (mtype != CEED_MEM_HOST)
    return CeedError(vec->ceed, 1, "Can only provide to HOST memory");
  if (!occa->array) { // Allocate if array was not allocated yet
    ierr = CeedVectorSetArray(vec, CEED_MEM_HOST, CEED_COPY_VALUES, NULL);
    CeedChk(ierr);
  }
  occaSyncH2D(vec); // sync Host to Device
  *array = occa->array;
  return 0;
}
// *****************************************************************************
static int CeedVectorGetArray_Occa(const CeedVector vec,
                                   const CeedMemType mtype,
                                   CeedScalar **array) {
  return CeedVectorGetArrayRead_Occa(vec,mtype,(const CeedScalar**)array);
}

// *****************************************************************************
// * Restore/Restore+Const
// *****************************************************************************
static int CeedVectorRestoreArrayRead_Occa(const CeedVector vec,
                                           const CeedScalar **array) {
  CeedDebug("\033[33m[CeedVector][Restore]");
  //CeedVector_Occa *occa = vec->data;
  // free memory we used for the view
  //CeedChk(CeedFree(&occa->array));
  *array = NULL;
  return 0;
}
// *****************************************************************************
static int CeedVectorRestoreArray_Occa(const CeedVector vec,
                                       CeedScalar **array) {
  return CeedVectorRestoreArrayRead_Occa(vec,(const CeedScalar**)array);
}

// *****************************************************************************
// * Destroy
// *****************************************************************************
static int CeedVectorDestroy_Occa(const CeedVector vec) {
  int ierr;
  CeedVector_Occa *occa = vec->data;
  CeedDebug("\033[33m[CeedVector][Destroy]");
  occaMemoryFree(*occa->array_device);
  ierr = CeedFree(&occa->array_allocated); CeedChk(ierr);
  ierr = CeedFree(&occa->array_device); CeedChk(ierr);
  ierr = CeedFree(&vec->data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * Create
// *****************************************************************************
int CeedVectorCreate_Occa(const Ceed ceed, const CeedInt n, CeedVector vec) {
  int ierr;
  CeedVector_Occa *vector_st;
  const Ceed_Occa *occa=ceed->data;

  CeedDebug("\033[33m[CeedVector][Create] n=%d", n);
  // ***************************************************************************
  vec->SetArray = CeedVectorSetArray_Occa;
  vec->GetArray = CeedVectorGetArray_Occa;
  vec->GetArrayRead = CeedVectorGetArrayRead_Occa;
  vec->RestoreArray = CeedVectorRestoreArray_Occa;
  vec->RestoreArrayRead = CeedVectorRestoreArrayRead_Occa;
  vec->Destroy = CeedVectorDestroy_Occa;
  // Allocating vector_st, host & device
  ierr = CeedCalloc(1,&vector_st); CeedChk(ierr);
  ierr = CeedCalloc(1,&vector_st->array_device); CeedChk(ierr);
  *vector_st->array_device = occaDeviceMalloc(occa->device, bytes(vec), NULL, NO_PROPS);
  vec->data = vector_st;
  return 0;
}
