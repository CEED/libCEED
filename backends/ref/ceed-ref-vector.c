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
static int CeedVectorSetArray_Ref(CeedVector vec, CeedMemType mem_type,
                                  CeedCopyMode copy_mode, CeedScalar *array) {
  int ierr;
  CeedVector_Ref *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);
  CeedInt length;
  ierr = CeedVectorGetLength(vec, &length); CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);

  if (mem_type != CEED_MEM_HOST)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Can only set HOST memory for this backend");
  // LCOV_EXCL_STOP

  switch (copy_mode) {
  case CEED_COPY_VALUES:
    if (!impl->array_owned) {
      ierr = CeedMalloc(length, &impl->array_owned); CeedChkBackend(ierr);
    }
    impl->array_borrowed = NULL;
    impl->array = impl->array_owned;
    if (array)
      memcpy(impl->array, array, length * sizeof(array[0]));
    break;
  case CEED_OWN_POINTER:
    ierr = CeedFree(&impl->array_owned); CeedChkBackend(ierr);
    impl->array_owned = array;
    impl->array_borrowed = NULL;
    impl->array = array;
    break;
  case CEED_USE_POINTER:
    ierr = CeedFree(&impl->array_owned); CeedChkBackend(ierr);
    impl->array_borrowed = array;
    impl->array = array;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Take Array
//------------------------------------------------------------------------------
static int CeedVectorTakeArray_Ref(CeedVector vec, CeedMemType mem_type,
                                   CeedScalar **array) {
  int ierr;
  CeedVector_Ref *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);

  if (mem_type != CEED_MEM_HOST)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Can only provide HOST memory for this backend");
  // LCOV_EXCL_STOP

  if (!impl->array_borrowed)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Must set array with CeedVectorSetArray and CEED_USE_POINTER before calling CeedVectorTakeArray");
  // LCOV_EXCL_STOP

  (*array) = impl->array_borrowed;
  impl->array_borrowed = NULL;
  impl->array = NULL;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Get Array
//------------------------------------------------------------------------------
static int CeedVectorGetArray_Ref(CeedVector vec, CeedMemType mem_type,
                                  CeedScalar **array) {
  int ierr;
  CeedVector_Ref *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);

  if (mem_type != CEED_MEM_HOST)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Can only provide HOST memory for this backend");
  // LCOV_EXCL_STOP

  if (!impl->array) {
    // Allocate if array is not yet allocated
    if (!impl->array_owned && !impl->array_borrowed) {
      ierr = CeedVectorSetArray(vec, CEED_MEM_HOST, CEED_COPY_VALUES, NULL);
      CeedChkBackend(ierr);
    } else {
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "Invalid data in array; must set vector with CeedVectorSetValue or CeedVectorSetArray before calling CeedVectorGetArray");
      // LCOV_EXCL_STOP
    }
  }

  *array = impl->array;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Get Array Read
//------------------------------------------------------------------------------
static int CeedVectorGetArrayRead_Ref(CeedVector vec, CeedMemType mem_type,
                                      const CeedScalar **array) {
  return CeedVectorGetArray_Ref(vec, mem_type, (CeedScalar **)array);
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

  ierr = CeedFree(&impl->array_owned); CeedChkBackend(ierr);
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

  ierr = CeedCalloc(1, &impl); CeedChkBackend(ierr);
  ierr = CeedVectorSetData(vec, impl); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
