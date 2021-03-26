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

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <string.h>
#include "ceed-ref.h"

//------------------------------------------------------------------------------
// Vector Set Array
//------------------------------------------------------------------------------
static int CeedVectorSetArray_Ref(CeedVector vec, CeedMemType mtype,
                                  CeedCopyMode cmode, CeedScalar *array) {
  int ierr;
  CeedVector_Ref *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);
  CeedInt length;
  ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);

  if (mtype != CEED_MEM_HOST)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Only MemType = HOST supported");
  // LCOV_EXCL_STOP
  ierr = CeedFree(&impl->array_allocated); CeedChkBackend(ierr);
  switch (cmode) {
  case CEED_COPY_VALUES:
    ierr = CeedMalloc(length, &impl->array_allocated); CeedChkBackend(ierr);
    impl->array = impl->array_allocated;
    if (array) memcpy(impl->array, array, length * sizeof(array[0]));
    break;
  case CEED_OWN_POINTER:
    impl->array_allocated = array;
    impl->array = array;
    break;
  case CEED_USE_POINTER:
    impl->array = array;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Take Array
//------------------------------------------------------------------------------
static int CeedVectorTakeArray_Ref(CeedVector vec, CeedMemType mtype,
                                   CeedScalar **array) {
  int ierr;
  CeedVector_Ref *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);

  if (mtype != CEED_MEM_HOST)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Only MemType = HOST supported");
  // LCOV_EXCL_STOP

  (*array) = impl->array;
  impl->array = NULL;
  impl->array_allocated = NULL;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Get Array
//------------------------------------------------------------------------------
static int CeedVectorGetArray_Ref(CeedVector vec, CeedMemType mtype,
                                  CeedScalar **array) {
  int ierr;
  CeedVector_Ref *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);

  if (mtype != CEED_MEM_HOST)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Can only provide to HOST memory");
  // LCOV_EXCL_STOP
  if (!impl->array) { // Allocate if array is not yet allocated
    ierr = CeedVectorSetArray(vec, CEED_MEM_HOST, CEED_COPY_VALUES, NULL);
    CeedChkBackend(ierr);
  }
  *array = impl->array;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Get Array Read
//------------------------------------------------------------------------------
static int CeedVectorGetArrayRead_Ref(CeedVector vec, CeedMemType mtype,
                                      const CeedScalar **array) {
  int ierr;
  CeedVector_Ref *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);

  if (mtype != CEED_MEM_HOST)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Can only provide to HOST memory");
  // LCOV_EXCL_STOP
  if (!impl->array) { // Allocate if array is not yet allocated
    ierr = CeedVectorSetArray(vec, CEED_MEM_HOST, CEED_COPY_VALUES, NULL);
    CeedChkBackend(ierr);
  }
  *array = impl->array;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Restore Array
//------------------------------------------------------------------------------
static int CeedVectorRestoreArray_Ref(CeedVector vec) {
  return CEED_ERROR_SUCCESS;
}

static int CeedVectorRestoreArrayRead_Ref(CeedVector vec) {
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Destroy
//------------------------------------------------------------------------------
static int CeedVectorDestroy_Ref(CeedVector vec) {
  int ierr;
  CeedVector_Ref *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  ierr = CeedFree(&impl->array_allocated); CeedChkBackend(ierr);
  ierr = CeedFree(&impl); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Create
//------------------------------------------------------------------------------
int CeedVectorCreate_Ref(CeedInt n, CeedVector vec) {
  int ierr;
  CeedVector_Ref *impl;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "SetArray",
                                CeedVectorSetArray_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "TakeArray",
                                CeedVectorTakeArray_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArray",
                                CeedVectorGetArray_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayRead",
                                CeedVectorGetArrayRead_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "RestoreArray",
                                CeedVectorRestoreArray_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "RestoreArrayRead",
                                CeedVectorRestoreArrayRead_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "Destroy",
                                CeedVectorDestroy_Ref); CeedChkBackend(ierr);
  ierr = CeedCalloc(1,&impl); CeedChkBackend(ierr);
  ierr = CeedVectorSetData(vec, impl); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
