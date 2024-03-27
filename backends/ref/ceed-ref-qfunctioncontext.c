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
// QFunctionContext has valid data
//------------------------------------------------------------------------------
static int CeedQFunctionContextHasValidData_Ref(CeedQFunctionContext ctx, bool *has_valid_data) {
  CeedQFunctionContext_Ref *impl;

  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));
  *has_valid_data = impl->data;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext has borrowed data
//------------------------------------------------------------------------------
static int CeedQFunctionContextHasBorrowedDataOfType_Ref(CeedQFunctionContext ctx, CeedMemType mem_type, bool *has_borrowed_data_of_type) {
  CeedQFunctionContext_Ref *impl;

  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));
  CeedCheck(mem_type == CEED_MEM_HOST, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_BACKEND, "Can only set HOST memory for this backend");
  *has_borrowed_data_of_type = impl->data_borrowed;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Set Data
//------------------------------------------------------------------------------
static int CeedQFunctionContextSetData_Ref(CeedQFunctionContext ctx, CeedMemType mem_type, CeedCopyMode copy_mode, void *data) {
  size_t                    ctx_size;
  CeedQFunctionContext_Ref *impl;

  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));
  CeedCallBackend(CeedQFunctionContextGetContextSize(ctx, &ctx_size));

  CeedCheck(mem_type == CEED_MEM_HOST, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_BACKEND, "Can only set HOST memory for this backend");

  CeedCallBackend(CeedFree(&impl->data_owned));
  switch (copy_mode) {
    case CEED_COPY_VALUES:
      CeedCallBackend(CeedMallocArray(1, ctx_size, &impl->data_owned));
      impl->data_borrowed = NULL;
      impl->data          = impl->data_owned;
      memcpy(impl->data, data, ctx_size);
      break;
    case CEED_OWN_POINTER:
      impl->data_owned    = data;
      impl->data_borrowed = NULL;
      impl->data          = data;
      break;
    case CEED_USE_POINTER:
      impl->data_borrowed = data;
      impl->data          = data;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Take Data
//------------------------------------------------------------------------------
static int CeedQFunctionContextTakeData_Ref(CeedQFunctionContext ctx, CeedMemType mem_type, void *data) {
  CeedQFunctionContext_Ref *impl;

  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));

  CeedCheck(mem_type == CEED_MEM_HOST, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_BACKEND, "Can only provide HOST memory for this backend");

  *(void **)data      = impl->data;
  impl->data_borrowed = NULL;
  impl->data          = NULL;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Get Data
//------------------------------------------------------------------------------
static int CeedQFunctionContextGetData_Ref(CeedQFunctionContext ctx, CeedMemType mem_type, void *data) {
  CeedQFunctionContext_Ref *impl;

  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));

  CeedCheck(mem_type == CEED_MEM_HOST, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_BACKEND, "Can only provide HOST memory for this backend");

  *(void **)data = impl->data;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Restore Data
//------------------------------------------------------------------------------
static int CeedQFunctionContextRestoreData_Ref(CeedQFunctionContext ctx) { return CEED_ERROR_SUCCESS; }

//------------------------------------------------------------------------------
// QFunctionContext Destroy
//------------------------------------------------------------------------------
static int CeedQFunctionContextDestroy_Ref(CeedQFunctionContext ctx) {
  CeedQFunctionContext_Ref *impl;

  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));
  CeedCallBackend(CeedFree(&impl->data_owned));
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Create
//------------------------------------------------------------------------------
int CeedQFunctionContextCreate_Ref(CeedQFunctionContext ctx) {
  Ceed                      ceed;
  CeedQFunctionContext_Ref *impl;

  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "HasValidData", CeedQFunctionContextHasValidData_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "HasBorrowedDataOfType", CeedQFunctionContextHasBorrowedDataOfType_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "SetData", CeedQFunctionContextSetData_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "TakeData", CeedQFunctionContextTakeData_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "GetData", CeedQFunctionContextGetData_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "GetDataRead", CeedQFunctionContextGetData_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "RestoreData", CeedQFunctionContextRestoreData_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "RestoreDataRead", CeedQFunctionContextRestoreData_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "Destroy", CeedQFunctionContextDestroy_Ref));
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedQFunctionContextSetBackendData(ctx, impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
