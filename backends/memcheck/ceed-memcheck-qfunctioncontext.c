// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <string.h>
#include <valgrind/memcheck.h>

#include "ceed-memcheck.h"

//------------------------------------------------------------------------------
// QFunctionContext has valid data
//------------------------------------------------------------------------------
static int CeedQFunctionContextHasValidData_Memcheck(CeedQFunctionContext ctx, bool *has_valid_data) {
  CeedQFunctionContext_Memcheck *impl;
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, (void *)&impl));

  *has_valid_data = !!impl->data;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext has borrowed data
//------------------------------------------------------------------------------
static int CeedQFunctionContextHasBorrowedDataOfType_Memcheck(CeedQFunctionContext ctx, CeedMemType mem_type, bool *has_borrowed_data_of_type) {
  CeedQFunctionContext_Memcheck *impl;
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, (void *)&impl));
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));

  switch (mem_type) {
    case CEED_MEM_HOST:
      *has_borrowed_data_of_type = !!impl->data_borrowed;
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
// QFunctionContext Set Data
//------------------------------------------------------------------------------
static int CeedQFunctionContextSetData_Memcheck(CeedQFunctionContext ctx, CeedMemType mem_type, CeedCopyMode copy_mode, void *data) {
  CeedQFunctionContext_Memcheck *impl;
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, (void *)&impl));
  size_t ctx_size;
  CeedCallBackend(CeedQFunctionContextGetContextSize(ctx, &ctx_size));
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));

  if (mem_type != CEED_MEM_HOST) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Can only set HOST memory for this backend");
    // LCOV_EXCL_STOP
  }

  CeedCallBackend(CeedFree(&impl->data_allocated));
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
  // Copy data to check ctx_size bounds
  CeedCallBackend(CeedMallocArray(1, ctx_size, &impl->data_allocated));
  memcpy(impl->data_allocated, impl->data, ctx_size);
  impl->data = impl->data_allocated;
  VALGRIND_DISCARD(impl->mem_block_id);
  impl->mem_block_id = VALGRIND_CREATE_BLOCK(impl->data, ctx_size, "'QFunction backend context data copy'");

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Take Data
//------------------------------------------------------------------------------
static int CeedQFunctionContextTakeData_Memcheck(CeedQFunctionContext ctx, CeedMemType mem_type, void *data) {
  CeedQFunctionContext_Memcheck *impl;
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, (void *)&impl));
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));

  if (mem_type != CEED_MEM_HOST) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Can only provide HOST memory for this backend");
    // LCOV_EXCL_STOP
  }

  *(void **)data      = impl->data_borrowed;
  impl->data_borrowed = NULL;
  impl->data          = NULL;
  VALGRIND_DISCARD(impl->mem_block_id);
  CeedCallBackend(CeedFree(&impl->data_allocated));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Get Data
//------------------------------------------------------------------------------
static int CeedQFunctionContextGetData_Memcheck(CeedQFunctionContext ctx, CeedMemType mem_type, void *data) {
  CeedQFunctionContext_Memcheck *impl;
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, (void *)&impl));
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));

  if (mem_type != CEED_MEM_HOST) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Can only provide HOST memory for this backend");
    // LCOV_EXCL_STOP
  }

  *(void **)data = impl->data;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Get Data Read-Only
//------------------------------------------------------------------------------
static int CeedQFunctionContextGetDataRead_Memcheck(CeedQFunctionContext ctx, CeedMemType mem_type, void *data) {
  CeedQFunctionContext_Memcheck *impl;
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, (void *)&impl));
  size_t ctx_size;
  CeedCallBackend(CeedQFunctionContextGetContextSize(ctx, &ctx_size));
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));

  CeedCallBackend(CeedQFunctionContextGetData_Memcheck(ctx, mem_type, data));

  // Make copy to verify no write occured
  CeedCallBackend(CeedMallocArray(1, ctx_size, &impl->data_read_only_copy));
  memcpy(impl->data_read_only_copy, *(void **)data, ctx_size);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Restore Data
//------------------------------------------------------------------------------
static int CeedQFunctionContextRestoreData_Memcheck(CeedQFunctionContext ctx) {
  size_t ctx_size;
  CeedCallBackend(CeedQFunctionContextGetContextSize(ctx, &ctx_size));
  CeedQFunctionContext_Memcheck *impl;
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, (void *)&impl));

  if (impl->data_borrowed) {
    memcpy(impl->data_borrowed, impl->data, ctx_size);
  }
  if (impl->data_owned) {
    memcpy(impl->data_owned, impl->data, ctx_size);
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Restore Data Read-Only
//------------------------------------------------------------------------------
static int CeedQFunctionContextRestoreDataRead_Memcheck(CeedQFunctionContext ctx) {
  size_t ctx_size;
  CeedCallBackend(CeedQFunctionContextGetContextSize(ctx, &ctx_size));
  CeedQFunctionContext_Memcheck *impl;
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, (void *)&impl));
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));

  if (memcmp(impl->data, impl->data_read_only_copy, ctx_size)) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Context data changed while accessed in read-only mode");
    // LCOV_EXCL_STOP
  }

  CeedCallBackend(CeedFree(&impl->data_read_only_copy));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext destroy user data
//------------------------------------------------------------------------------
static int CeedQFunctionContextDataDestroy_Memcheck(CeedQFunctionContext ctx) {
  CeedQFunctionContext_Memcheck *impl;
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));
  CeedQFunctionContextDataDestroyUser data_destroy_function;
  CeedMemType                         data_destroy_mem_type;
  CeedCallBackend(CeedQFunctionContextGetDataDestroy(ctx, &data_destroy_mem_type, &data_destroy_function));
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));

  if (data_destroy_mem_type != CEED_MEM_HOST) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Can only destroy HOST memory for this backend");
    // LCOV_EXCL_STOP
  }

  if (data_destroy_function) {
    CeedCallBackend(data_destroy_function(impl->data_borrowed ? impl->data_borrowed : impl->data_owned));
  }
  CeedCallBackend(CeedFree(&impl->data_allocated));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Destroy
//------------------------------------------------------------------------------
static int CeedQFunctionContextDestroy_Memcheck(CeedQFunctionContext ctx) {
  CeedQFunctionContext_Memcheck *impl;
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));

  CeedCallBackend(CeedFree(&impl->data_allocated));
  CeedCallBackend(CeedFree(&impl->data_owned));
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Create
//------------------------------------------------------------------------------
int CeedQFunctionContextCreate_Memcheck(CeedQFunctionContext ctx) {
  CeedQFunctionContext_Memcheck *impl;
  Ceed                           ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));

  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "HasValidData", CeedQFunctionContextHasValidData_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "HasBorrowedDataOfType", CeedQFunctionContextHasBorrowedDataOfType_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "SetData", CeedQFunctionContextSetData_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "TakeData", CeedQFunctionContextTakeData_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "GetData", CeedQFunctionContextGetData_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "GetDataRead", CeedQFunctionContextGetDataRead_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "RestoreData", CeedQFunctionContextRestoreData_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "RestoreDataRead", CeedQFunctionContextRestoreDataRead_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "DataDestroy", CeedQFunctionContextDataDestroy_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "Destroy", CeedQFunctionContextDestroy_Memcheck));

  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedQFunctionContextSetBackendData(ctx, impl));

  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
