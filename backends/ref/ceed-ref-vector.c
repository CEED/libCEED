// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <string.h>

#include "ceed-ref.h"

//------------------------------------------------------------------------------
// Has Valid Array
//------------------------------------------------------------------------------
static int CeedVectorHasValidArray_Ref(CeedVector vec, bool *has_valid_array) {
  CeedVector_Ref *impl;

  CeedCallBackend(CeedVectorGetData(vec, &impl));

  *has_valid_array = impl->array;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if has borrowed array of given type
//------------------------------------------------------------------------------
static inline int CeedVectorHasBorrowedArrayOfType_Ref(const CeedVector vec, CeedMemType mem_type, bool *has_borrowed_array_of_type) {
  CeedVector_Ref *impl;

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCheck(mem_type == CEED_MEM_HOST, CeedVectorReturnCeed(vec), CEED_ERROR_BACKEND, "Can only set HOST memory for this backend");
  *has_borrowed_array_of_type = impl->array_borrowed;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Set Array
//------------------------------------------------------------------------------
static int CeedVectorSetArray_Ref(CeedVector vec, CeedMemType mem_type, CeedCopyMode copy_mode, CeedScalar *array) {
  CeedSize        length;
  CeedVector_Ref *impl;

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedVectorGetLength(vec, &length));

  CeedCheck(mem_type == CEED_MEM_HOST, CeedVectorReturnCeed(vec), CEED_ERROR_BACKEND, "Can only set HOST memory for this backend");

  CeedCallBackend(CeedSetHostCeedScalarArray(array, copy_mode, length, (const CeedScalar **)&impl->array_owned,
                                             (const CeedScalar **)&impl->array_borrowed, (const CeedScalar **)&impl->array));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Take Array
//------------------------------------------------------------------------------
static int CeedVectorTakeArray_Ref(CeedVector vec, CeedMemType mem_type, CeedScalar **array) {
  CeedVector_Ref *impl;

  CeedCallBackend(CeedVectorGetData(vec, &impl));

  CeedCheck(mem_type == CEED_MEM_HOST, CeedVectorReturnCeed(vec), CEED_ERROR_BACKEND, "Can only provide HOST memory for this backend");

  (*array)             = impl->array_borrowed;
  impl->array_borrowed = NULL;
  impl->array          = NULL;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Get Array
//------------------------------------------------------------------------------
static int CeedVectorGetArrayCore_Ref(CeedVector vec, CeedMemType mem_type, CeedScalar **array) {
  CeedVector_Ref *impl;

  CeedCallBackend(CeedVectorGetData(vec, &impl));

  CeedCheck(mem_type == CEED_MEM_HOST, CeedVectorReturnCeed(vec), CEED_ERROR_BACKEND, "Can only provide HOST memory for this backend");

  *array = impl->array;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Get Array Read
//------------------------------------------------------------------------------
static int CeedVectorGetArrayRead_Ref(CeedVector vec, CeedMemType mem_type, const CeedScalar **array) {
  return CeedVectorGetArrayCore_Ref(vec, mem_type, (CeedScalar **)array);
}

//------------------------------------------------------------------------------
// Vector Get Array
//------------------------------------------------------------------------------
static int CeedVectorGetArray_Ref(CeedVector vec, CeedMemType mem_type, CeedScalar **array) {
  return CeedVectorGetArrayCore_Ref(vec, mem_type, array);
}

//------------------------------------------------------------------------------
// Vector Get Array Write
//------------------------------------------------------------------------------
static int CeedVectorGetArrayWrite_Ref(CeedVector vec, CeedMemType mem_type, CeedScalar **array) {
  CeedVector_Ref *impl;

  CeedCallBackend(CeedVectorGetData(vec, &impl));

  if (!impl->array) CeedCallBackend(CeedVectorSetArray(vec, CEED_MEM_HOST, CEED_COPY_VALUES, NULL));
  return CeedVectorGetArrayCore_Ref(vec, mem_type, (CeedScalar **)array);
}

//------------------------------------------------------------------------------
// Vector Restore Array
//------------------------------------------------------------------------------
static int CeedVectorRestoreArray_Ref(CeedVector vec) { return CEED_ERROR_SUCCESS; }

//------------------------------------------------------------------------------
// Vector Restore Array Read
//------------------------------------------------------------------------------
static int CeedVectorRestoreArrayRead_Ref(CeedVector vec) { return CEED_ERROR_SUCCESS; }

//------------------------------------------------------------------------------
// Vector Destroy
//------------------------------------------------------------------------------
static int CeedVectorDestroy_Ref(CeedVector vec) {
  CeedVector_Ref *impl;

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedFree(&impl->array_owned));
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Create
//------------------------------------------------------------------------------
int CeedVectorCreate_Ref(CeedSize n, CeedVector vec) {
  Ceed            ceed;
  CeedVector_Ref *impl;

  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "HasValidArray", CeedVectorHasValidArray_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "HasBorrowedArrayOfType", CeedVectorHasBorrowedArrayOfType_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "SetArray", CeedVectorSetArray_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "TakeArray", CeedVectorTakeArray_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "GetArray", CeedVectorGetArray_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayRead", CeedVectorGetArrayRead_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayWrite", CeedVectorGetArrayWrite_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "RestoreArray", CeedVectorRestoreArray_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "RestoreArrayRead", CeedVectorRestoreArrayRead_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "Destroy", CeedVectorDestroy_Ref));
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedVectorSetData(vec, impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
