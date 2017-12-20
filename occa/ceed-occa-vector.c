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
// *          - RestoreArrayRead, RestoreArray
// *          - GetArrayRead, GetArray, SetArray
// *****************************************************************************
typedef struct {
  CeedScalar* array;
  occaMemory* array_allocated;
} CeedVectorOcca;

// *****************************************************************************
int CeedVectorDestroyOcca(CeedVector vec) {
  CeedVectorOcca* impl = vec->data;
  int ierr;

  dbg("\033[33m[CeedVector][Destroy][Occa]");
  occaMemoryFree(*impl->array_allocated);
  ierr = CeedFree(&impl->array_allocated); CeedChk(ierr);
  ierr = CeedFree(&vec->data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
int CeedVectorSetArrayOcca(CeedVector vec, CeedMemType mtype,
                              CeedCopyMode cmode, CeedScalar* array) {
  CeedVectorOcca* impl = vec->data;
  int ierr;

  dbg("\033[33m[CeedVector][SetArray][Occa]");
  if (mtype != CEED_MEM_HOST)
    return CeedError(vec->ceed, 1, "Only MemType = HOST supported");
  // Freeing previous allocated array
  //occaMemoryFree(*impl->array_allocated);
  //ierr = CeedFree(&impl->array_allocated); CeedChk(ierr);
  // and rallocating everything
  //ierr = CeedCalloc(1,&impl->array_allocated); CeedChk(ierr);
  const size_t bytes = vec->length * sizeof(array[0]);
  // ***************************************************************************
  switch (cmode) {
    case CEED_COPY_VALUES:
      dbg("\t\033[33m[CeedVector][SetArray][Occa] CEED_COPY_VALUES");
      // Allocating space for our occaMemory
      ierr = CeedCalloc(1,&impl->array_allocated); CeedChk(ierr);
      // Allocating memory on device
      *impl->array_allocated = occaDeviceMalloc(device, bytes, NULL, occaDefault);
      if (array) occaCopyPtrToMem(*impl->array_allocated, array, bytes, 0, occaDefault);
      //impl->array = impl->array_allocated;
      break;
    case CEED_OWN_POINTER:
      dbg("\t\033[33m[CeedVector][SetArray][Occa] CEED_OWN_POINTER");
      // Allocating space for our occaMemory
      ierr = CeedCalloc(1,&impl->array_allocated); CeedChk(ierr);
      // Allocating memory on device
      *impl->array_allocated = occaDeviceMalloc(device, bytes, NULL, occaDefault);
      occaCopyPtrToMem(*impl->array_allocated, array, bytes, 0, occaDefault);
      impl->array = array;
      break;
    case CEED_USE_POINTER:
      dbg("\t\033[33m[CeedVector][SetArray][Occa] CEED_USE_POINTER");
      impl->array = array;
  }
  return 0;
}

// *****************************************************************************
int CeedVectorGetArrayOcca(CeedVector vec, CeedMemType mtype,
                              CeedScalar** array) {
  CeedVectorOcca* impl = vec->data;

  dbg("\033[33m[CeedVector][GetArray][Occa]");
  if (mtype != CEED_MEM_HOST)
    return CeedError(vec->ceed, 1, "Can only provide to HOST memory");
  *array = impl->array;
  return 0;
}

// *****************************************************************************
int CeedVectorGetArrayReadOcca(CeedVector vec, CeedMemType mtype,
                                  const CeedScalar** array) {
  CeedVectorOcca* impl = vec->data;

  dbg("\033[33m[CeedVector][GetArray][Const][Occa]");
  if (mtype != CEED_MEM_HOST)
    return CeedError(vec->ceed, 1, "Can only provide to HOST memory");
  // CEED_OWN_POINTER
  //occaCopyMemToPtr(impl->array, *impl->array_allocated, impl->size*sizeof(CeedScalar), 0, occaDefault);
  *array = impl->array;
  return 0;
}

// *****************************************************************************
static int CeedVectorRestoreArrayOcca(CeedVector vec, CeedScalar** array) {
  dbg("\033[33m[CeedVector][RestoreArray][Occa]");
  *array = NULL;
  return 0;
}

// *****************************************************************************
static int CeedVectorRestoreArrayReadOcca(CeedVector vec,
                                             const CeedScalar** array) {
  dbg("\033[33m[CeedVector][RestoreArray][Const][Occa]");
  *array = NULL;
  return 0;
}

// *****************************************************************************
int CeedVectorCreateOcca(Ceed ceed, CeedInt n, CeedVector vec) {
  CeedVectorOcca* impl;
  int ierr;

  dbg("\033[33m[CeedVector][Create][Occa] n=%d", n);
  vec->SetArray = CeedVectorSetArrayOcca;
  vec->GetArray = CeedVectorGetArrayOcca;
  vec->GetArrayRead = CeedVectorGetArrayReadOcca;
  vec->RestoreArray = CeedVectorRestoreArrayOcca;
  vec->RestoreArrayRead = CeedVectorRestoreArrayReadOcca;
  vec->Destroy = CeedVectorDestroyOcca;
  ierr = CeedCalloc(1,&impl); CeedChk(ierr);
  ierr = CeedCalloc(1,&impl->array_allocated); CeedChk(ierr);
  // Allocating on device
  //*impl->array_allocated = occaDeviceMalloc(device, n*sizeof(CeedScalar), NULL, occaDefault);
  vec->length = n;
  vec->data = impl;
  dbg("\033[33m[CeedVector][Create][Occa] done");
  return 0;
}
