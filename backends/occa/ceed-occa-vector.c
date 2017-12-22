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
  const CeedVector_Occa *impl = vec->data;
  assert(impl);
  assert(impl->device);
  occaCopyPtrToMem(*impl->device, impl->host, bytes(vec), NO_OFFSET, NO_PROPS);
}
static inline void occaSyncD2H(const CeedVector vec) {
  const CeedVector_Occa *impl = vec->data;
  assert(impl);
  assert(impl->host);
  occaCopyMemToPtr(impl->host, *impl->device, bytes(vec), NO_OFFSET, NO_PROPS);
}

// *****************************************************************************
// * OCCA COPY functions
// *****************************************************************************
static inline void occaCopyH2D(const CeedVector vec, void *from) {
  const CeedVector_Occa *impl = vec->data;
  assert(from);
  assert(impl);
  assert(impl->device);
  occaCopyPtrToMem(*impl->device, from, bytes(vec), NO_OFFSET, NO_PROPS);
}
static inline void occaCopyD2H(const CeedVector vec, void *to) {
  const CeedVector_Occa *impl = vec->data;
  assert(to);
  assert(impl);
  occaCopyMemToPtr(to, *impl->device, bytes(vec), NO_OFFSET, NO_PROPS);
}

// *****************************************************************************
// * Destroy
// *****************************************************************************
static int CeedVectorDestroy_Occa(const CeedVector vec) {
  CeedVector_Occa *impl = vec->data;

  CeedDebug("\033[33m[CeedVector][Destroy]");
  // free device memory
  occaMemoryFree(*impl->device);
  // free device object
  CeedChk(CeedFree(&impl->device));
  // free our CeedVector_Occa struct
  CeedChk(CeedFree(&vec->data));
  return 0;
}

// *****************************************************************************
// * Set
// *****************************************************************************
static int CeedVectorSetArray_Occa(const CeedVector vec,
                                   const CeedMemType mtype,
                                   const CeedCopyMode cmode,
                                   CeedScalar *array) {
  CeedVector_Occa *impl = vec->data;

  CeedDebug("\033[33m[CeedVector][SetArray]");
  if (mtype != CEED_MEM_HOST)
    return CeedError(vec->ceed, 1, "Only MemType = HOST supported");
  // ***************************************************************************
  // free previous allocated array
  //occaMemoryFree(*impl->device);
  // free device object
  //ierr = CeedFree(&impl->device); CeedChk(ierr);
  // and rallocate everything
  //CeedChk(CeedCalloc(1,&impl->device));
  // Allocating memory on device
  //*impl->device = occaDeviceMalloc(device, bytes(vec), NULL, occaDefault);
  // ***************************************************************************
  switch (cmode) {
  case CEED_COPY_VALUES:
    CeedDebug("\t\033[33m[CeedVector][SetArray] CEED_COPY_VALUES");
    if (array) memcpy(impl->host, array, bytes(vec));
    break;
  case CEED_OWN_POINTER:
    CeedDebug("\t\033[33m[CeedVector][SetArray] CEED_OWN_POINTER");
    impl->host = array;
    break;
  case CEED_USE_POINTER:
    CeedDebug("\t\033[33m[CeedVector][SetArray] CEED_USE_POINTER");
    impl->host = array;
    break;
  default: CeedError(vec->ceed,1," OCCA backend no default error");
  }
  // now sync the host to the device
  if (array) {
    CeedDebug("\033[33m[CeedVector][SetArray] occaSyncH2D");
    occaSyncH2D(vec);
  }
  return 0;
}

// *****************************************************************************
// * Get
// *****************************************************************************
static int CeedVectorGetArray_Occa(const CeedVector vec,
                                   CeedMemType mtype,
                                   CeedScalar **array) {
  CeedVector_Occa *impl = vec->data;

  CeedDebug("\033[33m[CeedVector][GetArray]");
  if (mtype != CEED_MEM_HOST)
    return CeedError(vec->ceed, 1, "Can only provide to HOST memory");
  // Allocating space on host to allow the view
  CeedChk(CeedCalloc(vec->length,&impl->host));
  // sync'ing back device to the host
  occaSyncD2H(vec);
  *array = impl->host;
  return 0;
}

// *****************************************************************************
// * Get + Const
// *****************************************************************************
static int CeedVectorGetArrayRead_Occa(const CeedVector vec,
                                       const CeedMemType mtype,
                                       const CeedScalar **array) {
  CeedVector_Occa *impl = vec->data;

  CeedDebug("\033[33m[CeedVector][GetArray][Const]");
  if (mtype != CEED_MEM_HOST)
    return CeedError(vec->ceed, 1, "Can only provide to HOST memory");
  // Allocating space on host to allow the view
  CeedChk(CeedCalloc(vec->length,&impl->host));
  // sync'ing back device to the host
  occaSyncD2H(vec);
  // providing the view
  *array = impl->host;
  return 0;
}

// *****************************************************************************
// * Restore + Const
// *****************************************************************************
static int CeedVectorRestoreArrayRead_Occa(const CeedVector vec,
                                           const CeedScalar **array) {
  CeedVector_Occa *impl = vec->data;

  CeedDebug("\033[33m[CeedVector][RestoreArray][Const]");
  // free memory we used for the view
  CeedChk(CeedFree(&impl->host));
  *array = NULL;
  return 0;
}

// *****************************************************************************
// * Restore
// *****************************************************************************
static int CeedVectorRestoreArray_Occa(const CeedVector vec,
                                       CeedScalar **array) {
  CeedVector_Occa *impl = vec->data;

  CeedDebug("\033[33m[CeedVector][RestoreArray]");
  impl->host = NULL;
  *array = NULL;
  return 0;
}

// *****************************************************************************
// * Create
// *****************************************************************************
int CeedVectorCreate_Occa(const Ceed ceed, const CeedInt n, CeedVector vec) {
  CeedVector_Occa *impl;
  Ceed_Occa *ceed_data=ceed->data;

  CeedDebug("\033[33m[CeedVector][Create] n=%d", n);
  // ***************************************************************************
  vec->SetArray = CeedVectorSetArray_Occa;
  vec->GetArray = CeedVectorGetArray_Occa;
  vec->GetArrayRead = CeedVectorGetArrayRead_Occa;
  vec->RestoreArray = CeedVectorRestoreArray_Occa;
  vec->RestoreArrayRead = CeedVectorRestoreArrayRead_Occa;
  vec->Destroy = CeedVectorDestroy_Occa;
  // Allocating impl, host & device
  CeedChk(CeedCalloc(1,&impl));
  CeedChk(CeedCalloc(1,&impl->device));
  *impl->device = occaDeviceMalloc(ceed_data->device, bytes(vec), NULL, NO_PROPS);
  vec->data = impl;
  CeedDebug("\033[33m[CeedVector][Create] done");
  return 0;
}
