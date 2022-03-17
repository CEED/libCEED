// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <string.h>
#include "ceed-ref.h"

//------------------------------------------------------------------------------
// Has Valid Array
//------------------------------------------------------------------------------
static int CeedVectorHasValidArray_Ref(CeedVector vec, bool *has_valid_array) {
  int ierr;
  CeedVector_Ref *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  *has_valid_array = !!impl->array;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if has borrowed array of given type
//------------------------------------------------------------------------------
static inline int CeedVectorHasBorrowedArrayOfType_Ref(const CeedVector vec,
    CeedMemType mem_type, bool *has_borrowed_array_of_type) {
  int ierr;
  CeedVector_Ref *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);

  switch (mem_type) {
  case CEED_MEM_HOST:
    *has_borrowed_array_of_type = !!impl->array_borrowed;
    break;
  default:
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Can only set HOST memory for this backend");
    // LCOV_EXCL_STOP
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Set Array
//------------------------------------------------------------------------------
static int CeedVectorSetArray_Ref(CeedVector vec, CeedMemType mem_type,
                                  CeedCopyMode copy_mode, CeedScalar *array) {
  int ierr;
  CeedVector_Ref *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);
  CeedSize length;
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
      ierr = CeedCalloc(length, &impl->array_owned); CeedChkBackend(ierr);
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

  (*array) = impl->array_borrowed;
  impl->array_borrowed = NULL;
  impl->array = NULL;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Get Array
//------------------------------------------------------------------------------
static int CeedVectorGetArrayCore_Ref(CeedVector vec, CeedMemType mem_type,
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

  *array = impl->array;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Get Array Read
//------------------------------------------------------------------------------
static int CeedVectorGetArrayRead_Ref(CeedVector vec, CeedMemType mem_type,
                                      const CeedScalar **array) {
  return CeedVectorGetArrayCore_Ref(vec, mem_type, (CeedScalar **)array);
}

//------------------------------------------------------------------------------
// Vector Get Array
//------------------------------------------------------------------------------
static int CeedVectorGetArray_Ref(CeedVector vec, CeedMemType mem_type,
                                  CeedScalar **array) {
  return CeedVectorGetArrayCore_Ref(vec, mem_type, array);
}

//------------------------------------------------------------------------------
// Vector Get Array Write
//------------------------------------------------------------------------------
static int CeedVectorGetArrayWrite_Ref(CeedVector vec, CeedMemType mem_type,
                                       const CeedScalar **array) {
  int ierr;
  CeedVector_Ref *impl;
  ierr = CeedVectorGetData(vec, &impl); CeedChkBackend(ierr);

  if (!impl->array) {
    if (!impl->array_owned && !impl->array_borrowed) {
      // Allocate if array is not yet allocated
      ierr = CeedVectorSetArray(vec, CEED_MEM_HOST, CEED_COPY_VALUES, NULL);
      CeedChkBackend(ierr);
    } else {
      // Select dirty array for GetArrayWrite
      if (impl->array_borrowed)
        impl->array = impl->array_borrowed;
      else
        impl->array = impl->array_owned;
    }
  }

  return CeedVectorGetArrayCore_Ref(vec, mem_type, (CeedScalar **)array);
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
int CeedVectorCreate_Ref(CeedSize n, CeedVector vec) {
  int ierr;
  CeedVector_Ref *impl;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "HasValidArray",
                                CeedVectorHasValidArray_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "HasBorrowedArrayOfType",
                                CeedVectorHasBorrowedArrayOfType_Ref);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "SetArray",
                                CeedVectorSetArray_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "TakeArray",
                                CeedVectorTakeArray_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArray",
                                CeedVectorGetArray_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayRead",
                                CeedVectorGetArrayRead_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayWrite",
                                CeedVectorGetArrayWrite_Ref); CeedChkBackend(ierr);
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
