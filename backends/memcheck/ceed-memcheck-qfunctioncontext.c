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
#include <valgrind/memcheck.h>

#include "ceed-memcheck.h"

//------------------------------------------------------------------------------
// QFunctionContext has valid data
//------------------------------------------------------------------------------
static int CeedQFunctionContextHasValidData_Memcheck(CeedQFunctionContext ctx, bool *has_valid_data) {
  CeedQFunctionContext_Memcheck *impl;

  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));
  *has_valid_data = !!impl->data_allocated;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext has borrowed data
//------------------------------------------------------------------------------
static int CeedQFunctionContextHasBorrowedDataOfType_Memcheck(CeedQFunctionContext ctx, CeedMemType mem_type, bool *has_borrowed_data_of_type) {
  CeedQFunctionContext_Memcheck *impl;

  CeedCheck(mem_type == CEED_MEM_HOST, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_BACKEND, "Can only set HOST memory for this backend");

  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));
  *has_borrowed_data_of_type = !!impl->data_borrowed;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Set Data
//------------------------------------------------------------------------------
static int CeedQFunctionContextSetData_Memcheck(CeedQFunctionContext ctx, CeedMemType mem_type, CeedCopyMode copy_mode, void *data) {
  size_t                         ctx_size;
  CeedQFunctionContext_Memcheck *impl;

  CeedCheck(mem_type == CEED_MEM_HOST, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_BACKEND, "Can only set HOST memory for this backend");

  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));
  CeedCallBackend(CeedQFunctionContextGetContextSize(ctx, &ctx_size));

  // Clear previous owned data buffers
  if (impl->data_allocated) {
    memset(impl->data_allocated, -42, ctx_size);
    VALGRIND_DISCARD(impl->allocated_block_id);
  }
  CeedCallBackend(CeedFree(&impl->data_allocated));
  if (impl->data_owned) {
    memset(impl->data_owned, -42, ctx_size);
    VALGRIND_DISCARD(impl->owned_block_id);
  }
  CeedCallBackend(CeedFree(&impl->data_owned));

  // Clear borrowed block id, if present
  if (impl->data_borrowed) VALGRIND_DISCARD(impl->borrowed_block_id);

  // Set internal pointers to external buffers
  switch (copy_mode) {
    case CEED_COPY_VALUES:
      impl->data_owned    = NULL;
      impl->data_borrowed = NULL;
      break;
    case CEED_OWN_POINTER:
      impl->data_owned     = data;
      impl->data_borrowed  = NULL;
      impl->owned_block_id = VALGRIND_CREATE_BLOCK(impl->data_owned, ctx_size, "Owned external data buffer");
      break;
    case CEED_USE_POINTER:
      impl->data_owned     = NULL;
      impl->data_borrowed  = data;
      impl->owned_block_id = VALGRIND_CREATE_BLOCK(impl->data_borrowed, ctx_size, "Borrowed external data buffer");
  }

  // Create internal data buffer
  CeedCallBackend(CeedMallocArray(1, ctx_size, &impl->data_allocated));
  impl->allocated_block_id = VALGRIND_CREATE_BLOCK(impl->data_allocated, ctx_size, "'Allocated internal context data buffer");
  memcpy(impl->data_allocated, data, ctx_size);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync data
//------------------------------------------------------------------------------
static int CeedQFunctionContextSyncData_Memcheck(CeedQFunctionContext ctx, CeedMemType mem_type) {
  size_t                         ctx_size;
  CeedQFunctionContext_Memcheck *impl;

  CeedCheck(mem_type == CEED_MEM_HOST, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_BACKEND, "Can only set HOST memory for this backend");

  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));
  CeedCallBackend(CeedQFunctionContextGetContextSize(ctx, &ctx_size));

  // Copy internal buffer back to owned or borrowed data buffer
  if (impl->data_owned) {
    memcpy(impl->data_owned, impl->data_allocated, ctx_size);
  }
  if (impl->data_borrowed) {
    memcpy(impl->data_borrowed, impl->data_allocated, ctx_size);
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Take Data
//------------------------------------------------------------------------------
static int CeedQFunctionContextTakeData_Memcheck(CeedQFunctionContext ctx, CeedMemType mem_type, void *data) {
  size_t                         ctx_size;
  CeedQFunctionContext_Memcheck *impl;

  CeedCheck(mem_type == CEED_MEM_HOST, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_BACKEND, "Can only provide HOST memory for this backend");

  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));
  CeedCallBackend(CeedQFunctionContextGetContextSize(ctx, &ctx_size));

  // Synchronize memory
  CeedCallBackend(CeedQFunctionContextSyncData_Memcheck(ctx, CEED_MEM_HOST));

  // Return borrowed buffer
  *(void **)data      = impl->data_borrowed;
  impl->data_borrowed = NULL;
  VALGRIND_DISCARD(impl->borrowed_block_id);

  // De-allocate internal memory
  if (impl->data_allocated) {
    memset(impl->data_allocated, -42, ctx_size);
    VALGRIND_DISCARD(impl->allocated_block_id);
  }
  CeedCallBackend(CeedFree(&impl->data_allocated));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Get Data
//------------------------------------------------------------------------------
static int CeedQFunctionContextGetData_Memcheck(CeedQFunctionContext ctx, CeedMemType mem_type, void *data) {
  size_t                         ctx_size;
  CeedQFunctionContext_Memcheck *impl;

  CeedCheck(mem_type == CEED_MEM_HOST, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_BACKEND, "Can only set HOST memory for this backend");

  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));
  CeedCallBackend(CeedQFunctionContextGetContextSize(ctx, &ctx_size));

  // Create and return writable buffer
  CeedCallBackend(CeedMallocArray(1, ctx_size, &impl->data_writable_copy));
  impl->writable_block_id = VALGRIND_CREATE_BLOCK(impl->data_writable_copy, ctx_size, "Allocated writeable data buffer copy");
  memcpy(impl->data_writable_copy, impl->data_allocated, ctx_size);
  *(void **)data = impl->data_writable_copy;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Get Data Read-Only
//------------------------------------------------------------------------------
static int CeedQFunctionContextGetDataRead_Memcheck(CeedQFunctionContext ctx, CeedMemType mem_type, void *data) {
  size_t                         ctx_size;
  CeedQFunctionContext_Memcheck *impl;

  CeedCheck(mem_type == CEED_MEM_HOST, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_BACKEND, "Can only set HOST memory for this backend");

  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));
  CeedCallBackend(CeedQFunctionContextGetContextSize(ctx, &ctx_size));

  // Create and return read-only buffer
  if (!impl->data_read_only_copy) {
    CeedCallBackend(CeedMallocArray(1, ctx_size, &impl->data_read_only_copy));
    impl->writable_block_id = VALGRIND_CREATE_BLOCK(impl->data_read_only_copy, ctx_size, "Allocated read-only data buffer copy");
    memcpy(impl->data_read_only_copy, impl->data_allocated, ctx_size);
  }
  *(void **)data = impl->data_read_only_copy;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Restore Data
//------------------------------------------------------------------------------
static int CeedQFunctionContextRestoreData_Memcheck(CeedQFunctionContext ctx) {
  size_t                         ctx_size;
  CeedQFunctionContext_Memcheck *impl;

  CeedCallBackend(CeedQFunctionContextGetContextSize(ctx, &ctx_size));
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));

  // Copy back to internal buffer and sync
  memcpy(impl->data_allocated, impl->data_writable_copy, ctx_size);
  CeedCallBackend(CeedQFunctionContextSyncData_Memcheck(ctx, CEED_MEM_HOST));

  // Invalidate writable buffer
  memset(impl->data_writable_copy, -42, ctx_size);
  CeedCallBackend(CeedFree(&impl->data_writable_copy));
  VALGRIND_DISCARD(impl->writable_block_id);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Restore Data Read-Only
//------------------------------------------------------------------------------
static int CeedQFunctionContextRestoreDataRead_Memcheck(CeedQFunctionContext ctx) {
  Ceed                           ceed;
  size_t                         ctx_size;
  CeedQFunctionContext_Memcheck *impl;

  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  CeedCallBackend(CeedQFunctionContextGetContextSize(ctx, &ctx_size));
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));

  // Verify no changes made during read-only access
  bool is_changed = memcmp(impl->data_allocated, impl->data_read_only_copy, ctx_size);

  CeedCheck(!is_changed, ceed, CEED_ERROR_BACKEND, "Context data changed while accessed in read-only mode");

  // Invalidate read-only buffer
  memset(impl->data_read_only_copy, -42, ctx_size);
  CeedCallBackend(CeedFree(&impl->data_read_only_copy));
  VALGRIND_DISCARD(impl->read_only_block_id);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext destroy user data
//------------------------------------------------------------------------------
static int CeedQFunctionContextDataDestroy_Memcheck(CeedQFunctionContext ctx) {
  Ceed                                ceed;
  CeedMemType                         data_destroy_mem_type;
  CeedQFunctionContextDataDestroyUser data_destroy_function;
  CeedQFunctionContext_Memcheck      *impl;

  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));

  CeedCallBackend(CeedQFunctionContextGetDataDestroy(ctx, &data_destroy_mem_type, &data_destroy_function));
  CeedCheck(data_destroy_mem_type == CEED_MEM_HOST, ceed, CEED_ERROR_BACKEND, "Can only destroy HOST memory for this backend");

  // Run user destroy routine
  if (data_destroy_function) {
    bool is_borrowed = !!impl->data_borrowed;

    CeedCallBackend(data_destroy_function(is_borrowed ? impl->data_borrowed : impl->data_owned));
    if (is_borrowed) VALGRIND_DISCARD(impl->borrowed_block_id);
    else VALGRIND_DISCARD(impl->owned_block_id);
  }
  // Free allocations and discard block ids
  if (impl->data_allocated) {
    CeedCallBackend(CeedFree(&impl->data_allocated));
    VALGRIND_DISCARD(impl->allocated_block_id);
  }
  if (impl->data_owned) {
    CeedCallBackend(CeedFree(&impl->data_owned));
    VALGRIND_DISCARD(impl->owned_block_id);
  }
  if (impl->data_borrowed) {
    VALGRIND_DISCARD(impl->borrowed_block_id);
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Destroy
//------------------------------------------------------------------------------
static int CeedQFunctionContextDestroy_Memcheck(CeedQFunctionContext ctx) {
  CeedQFunctionContext_Memcheck *impl;

  // Free allocations and discard block ids
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));
  if (impl->data_allocated) {
    CeedCallBackend(CeedFree(&impl->data_allocated));
    VALGRIND_DISCARD(impl->allocated_block_id);
  }
  if (impl->data_owned) {
    CeedCallBackend(CeedFree(&impl->data_owned));
    VALGRIND_DISCARD(impl->owned_block_id);
  }
  if (impl->data_borrowed) {
    VALGRIND_DISCARD(impl->borrowed_block_id);
  }
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Create
//------------------------------------------------------------------------------
int CeedQFunctionContextCreate_Memcheck(CeedQFunctionContext ctx) {
  Ceed                           ceed;
  CeedQFunctionContext_Memcheck *impl;

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
