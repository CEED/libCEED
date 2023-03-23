// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <valgrind/memcheck.h>

#include "ceed-memcheck.h"

//------------------------------------------------------------------------------
// Has Valid Array
//------------------------------------------------------------------------------
static int CeedVectorHasValidArray_Memcheck(CeedVector vec, bool *has_valid_array) {
  CeedVector_Memcheck *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  *has_valid_array = !!impl->array;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if has borrowed array of given type
//------------------------------------------------------------------------------
static inline int CeedVectorHasBorrowedArrayOfType_Memcheck(const CeedVector vec, CeedMemType mem_type, bool *has_borrowed_array_of_type) {
  CeedVector_Memcheck *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));

  switch (mem_type) {
    case CEED_MEM_HOST:
      *has_borrowed_array_of_type = !!impl->array_borrowed;
      break;
    default:
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_BACKEND, "Can only set HOST memory for this backend");
      // LCOV_EXCL_STOP
      break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Set Array
//------------------------------------------------------------------------------
static int CeedVectorSetArray_Memcheck(CeedVector vec, CeedMemType mem_type, CeedCopyMode copy_mode, CeedScalar *array) {
  CeedVector_Memcheck *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedSize length;
  CeedCallBackend(CeedVectorGetLength(vec, &length));
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));

  if (mem_type != CEED_MEM_HOST) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Can only set HOST memory for this backend");
    // LCOV_EXCL_STOP
  }

  CeedCallBackend(CeedFree(&impl->array_allocated));
  CeedCallBackend(CeedFree(&impl->array_owned));
  switch (copy_mode) {
    case CEED_COPY_VALUES:
      CeedCallBackend(CeedCalloc(length, &impl->array_owned));
      impl->array_borrowed = NULL;
      impl->array          = impl->array_owned;
      if (array) {
        memcpy(impl->array, array, length * sizeof(array[0]));
      } else {
        for (CeedInt i = 0; i < length; i++) impl->array[i] = NAN;
      }
      break;
    case CEED_OWN_POINTER:
      impl->array_owned    = array;
      impl->array_borrowed = NULL;
      impl->array          = array;
      break;
    case CEED_USE_POINTER:
      impl->array_borrowed = array;
      impl->array          = array;
  }
  // Copy data to check access
  CeedCallBackend(CeedCalloc(length, &impl->array_allocated));
  memcpy(impl->array_allocated, impl->array, length * sizeof(array[0]));
  impl->array = impl->array_allocated;
  VALGRIND_DISCARD(impl->mem_block_id);
  impl->mem_block_id = VALGRIND_CREATE_BLOCK(impl->array, length * sizeof(array[0]), "'Vector backend array data copy'");

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Take Array
//------------------------------------------------------------------------------
static int CeedVectorTakeArray_Memcheck(CeedVector vec, CeedMemType mem_type, CeedScalar **array) {
  CeedVector_Memcheck *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));

  if (mem_type != CEED_MEM_HOST) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Can only provide HOST memory for this backend");
    // LCOV_EXCL_STOP
  }

  (*array)             = impl->array_borrowed;
  impl->array_borrowed = NULL;
  impl->array          = NULL;
  VALGRIND_DISCARD(impl->mem_block_id);
  CeedCallBackend(CeedFree(&impl->array_allocated));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Get Array
//------------------------------------------------------------------------------
static int CeedVectorGetArray_Memcheck(CeedVector vec, CeedMemType mem_type, CeedScalar **array) {
  CeedVector_Memcheck *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));

  if (mem_type != CEED_MEM_HOST) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Can only provide HOST memory for this backend");
    // LCOV_EXCL_STOP
  }

  *array = impl->array;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Get Array Read
//------------------------------------------------------------------------------
static int CeedVectorGetArrayRead_Memcheck(CeedVector vec, CeedMemType mem_type, const CeedScalar **array) {
  CeedVector_Memcheck *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedSize length;
  CeedCallBackend(CeedVectorGetLength(vec, &length));
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));

  CeedCallBackend(CeedVectorGetArray_Memcheck(vec, mem_type, (CeedScalar **)array));

  // Make copy to verify no write occurred
  CeedCallBackend(CeedCalloc(length, &impl->array_read_only_copy));
  memcpy(impl->array_read_only_copy, *array, length * sizeof((*array)[0]));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Get Array Write
//------------------------------------------------------------------------------
static int CeedVectorGetArrayWrite_Memcheck(CeedVector vec, CeedMemType mem_type, CeedScalar **array) {
  CeedVector_Memcheck *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedSize length;
  CeedCallBackend(CeedVectorGetLength(vec, &length));
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));

  // Invalidate data to make sure no read occurs
  if (!impl->array) CeedCallBackend(CeedVectorSetArray_Memcheck(vec, mem_type, CEED_COPY_VALUES, NULL));
  CeedCallBackend(CeedVectorGetArray_Memcheck(vec, mem_type, array));
  for (CeedInt i = 0; i < length; i++) (*array)[i] = NAN;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Restore Array
//------------------------------------------------------------------------------
static int CeedVectorRestoreArray_Memcheck(CeedVector vec) {
  CeedVector_Memcheck *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedSize length;
  CeedCallBackend(CeedVectorGetLength(vec, &length));
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));

  if (impl->array_borrowed) {
    memcpy(impl->array_borrowed, impl->array, length * sizeof(impl->array[0]));
  }
  if (impl->array_owned) {
    memcpy(impl->array_owned, impl->array, length * sizeof(impl->array[0]));
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Restore Array Read-Only
//------------------------------------------------------------------------------
static int CeedVectorRestoreArrayRead_Memcheck(CeedVector vec) {
  CeedVector_Memcheck *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedSize length;
  CeedCallBackend(CeedVectorGetLength(vec, &length));
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));

  if (memcmp(impl->array, impl->array_read_only_copy, length * sizeof(impl->array[0]))) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Array data changed while accessed in read-only mode");
    // LCOV_EXCL_STOP
  }

  CeedCallBackend(CeedFree(&impl->array_read_only_copy));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Destroy
//------------------------------------------------------------------------------
static int CeedVectorDestroy_Memcheck(CeedVector vec) {
  CeedVector_Memcheck *impl;
  CeedCallBackend(CeedVectorGetData(vec, &impl));

  VALGRIND_DISCARD(impl->mem_block_id);
  CeedCallBackend(CeedFree(&impl->array_allocated));
  CeedCallBackend(CeedFree(&impl->array_owned));
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Create
//------------------------------------------------------------------------------
int CeedVectorCreate_Memcheck(CeedSize n, CeedVector vec) {
  CeedVector_Memcheck *impl;
  Ceed                 ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "HasValidArray", CeedVectorHasValidArray_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "HasBorrowedArrayOfType", CeedVectorHasBorrowedArrayOfType_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "SetArray", CeedVectorSetArray_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "TakeArray", CeedVectorTakeArray_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "GetArray", CeedVectorGetArray_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayRead", CeedVectorGetArrayRead_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayWrite", CeedVectorGetArrayWrite_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "RestoreArray", CeedVectorRestoreArray_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "RestoreArrayRead", CeedVectorRestoreArrayRead_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "Destroy", CeedVectorDestroy_Memcheck));

  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedVectorSetData(vec, impl));

  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
