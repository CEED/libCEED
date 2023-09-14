// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>

#include <string>
#include <sycl/sycl.hpp>

#include "ceed-sycl-ref.hpp"

//------------------------------------------------------------------------------
// Sync host to device
//------------------------------------------------------------------------------
static inline int CeedQFunctionContextSyncH2D_Sycl(const CeedQFunctionContext ctx) {
  Ceed                       ceed;
  Ceed_Sycl                 *sycl_data;
  size_t                     ctx_size;
  CeedQFunctionContext_Sycl *impl;

  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  CeedCallBackend(CeedGetData(ceed, &sycl_data));
  CeedCheck(impl->h_data, ceed, CEED_ERROR_BACKEND, "No valid host data to sync to device");

  CeedCallBackend(CeedQFunctionContextGetContextSize(ctx, &ctx_size));

  if (impl->d_data_borrowed) {
    impl->d_data = impl->d_data_borrowed;
  } else if (impl->d_data_owned) {
    impl->d_data = impl->d_data_owned;
  } else {
    CeedCallSycl(ceed, impl->d_data_owned = sycl::malloc_device(ctx_size, sycl_data->sycl_device, sycl_data->sycl_context));
    impl->d_data = impl->d_data_owned;
  }
  std::vector<sycl::event> e;
  if (!sycl_data->sycl_queue.is_in_order()) e = {sycl_data->sycl_queue.ext_oneapi_submit_barrier()};
  sycl::event copy_event = sycl_data->sycl_queue.memcpy(impl->d_data, impl->h_data, ctx_size, {e});
  CeedCallSycl(ceed, copy_event.wait_and_throw());
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync device to host
//------------------------------------------------------------------------------
static inline int CeedQFunctionContextSyncD2H_Sycl(const CeedQFunctionContext ctx) {
  Ceed                       ceed;
  Ceed_Sycl                 *sycl_data;
  size_t                     ctx_size;
  CeedQFunctionContext_Sycl *impl;

  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  CeedCallBackend(CeedGetData(ceed, &sycl_data));
  CeedCheck(impl->d_data, ceed, CEED_ERROR_BACKEND, "No valid device data to sync to host");

  CeedCallBackend(CeedQFunctionContextGetContextSize(ctx, &ctx_size));

  if (impl->h_data_borrowed) {
    impl->h_data = impl->h_data_borrowed;
  } else if (impl->h_data_owned) {
    impl->h_data = impl->h_data_owned;
  } else {
    CeedCallBackend(CeedMallocArray(1, ctx_size, &impl->h_data_owned));
    impl->h_data = impl->h_data_owned;
  }

  std::vector<sycl::event> e;
  if (!sycl_data->sycl_queue.is_in_order()) e = {sycl_data->sycl_queue.ext_oneapi_submit_barrier()};
  sycl::event copy_event = sycl_data->sycl_queue.memcpy(impl->h_data, impl->d_data, ctx_size, {e});
  CeedCallSycl(ceed, copy_event.wait_and_throw());
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync data of type
//------------------------------------------------------------------------------
static inline int CeedQFunctionContextSync_Sycl(const CeedQFunctionContext ctx, CeedMemType mem_type) {
  switch (mem_type) {
    case CEED_MEM_HOST:
      return CeedQFunctionContextSyncD2H_Sycl(ctx);
    case CEED_MEM_DEVICE:
      return CeedQFunctionContextSyncH2D_Sycl(ctx);
  }
  return CEED_ERROR_UNSUPPORTED;
}

//------------------------------------------------------------------------------
// Set all pointers as invalid
//------------------------------------------------------------------------------
static inline int CeedQFunctionContextSetAllInvalid_Sycl(const CeedQFunctionContext ctx) {
  CeedQFunctionContext_Sycl *impl;

  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));
  impl->h_data = NULL;
  impl->d_data = NULL;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if ctx has valid data
//------------------------------------------------------------------------------
static inline int CeedQFunctionContextHasValidData_Sycl(const CeedQFunctionContext ctx, bool *has_valid_data) {
  CeedQFunctionContext_Sycl *impl;

  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));
  *has_valid_data = impl && (impl->h_data || impl->d_data);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if ctx has borrowed data
//------------------------------------------------------------------------------
static inline int CeedQFunctionContextHasBorrowedDataOfType_Sycl(const CeedQFunctionContext ctx, CeedMemType mem_type,
                                                                 bool *has_borrowed_data_of_type) {
  CeedQFunctionContext_Sycl *impl;

  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));
  switch (mem_type) {
    case CEED_MEM_HOST:
      *has_borrowed_data_of_type = impl->h_data_borrowed;
      break;
    case CEED_MEM_DEVICE:
      *has_borrowed_data_of_type = impl->d_data_borrowed;
      break;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if data of given type needs sync
//------------------------------------------------------------------------------
static inline int CeedQFunctionContextNeedSync_Sycl(const CeedQFunctionContext ctx, CeedMemType mem_type, bool *need_sync) {
  bool                       has_valid_data = true;
  CeedQFunctionContext_Sycl *impl;

  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));
  CeedCallBackend(CeedQFunctionContextHasValidData(ctx, &has_valid_data));
  switch (mem_type) {
    case CEED_MEM_HOST:
      *need_sync = has_valid_data && !impl->h_data;
      break;
    case CEED_MEM_DEVICE:
      *need_sync = has_valid_data && !impl->d_data;
      break;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set data from host
//------------------------------------------------------------------------------
static int CeedQFunctionContextSetDataHost_Sycl(const CeedQFunctionContext ctx, const CeedCopyMode copy_mode, void *data) {
  CeedQFunctionContext_Sycl *impl;

  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));
  CeedCallBackend(CeedFree(&impl->h_data_owned));
  switch (copy_mode) {
    case CEED_COPY_VALUES:
      size_t ctx_size;

      CeedCallBackend(CeedQFunctionContextGetContextSize(ctx, &ctx_size));
      CeedCallBackend(CeedMallocArray(1, ctx_size, &impl->h_data_owned));
      impl->h_data_borrowed = NULL;
      impl->h_data          = impl->h_data_owned;
      memcpy(impl->h_data, data, ctx_size);
      break;
    case CEED_OWN_POINTER:
      impl->h_data_owned    = data;
      impl->h_data_borrowed = NULL;
      impl->h_data          = data;
      break;
    case CEED_USE_POINTER:
      impl->h_data_borrowed = data;
      impl->h_data          = data;
      break;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set data from device
//------------------------------------------------------------------------------
static int CeedQFunctionContextSetDataDevice_Sycl(const CeedQFunctionContext ctx, const CeedCopyMode copy_mode, void *data) {
  Ceed                       ceed;
  Ceed_Sycl                 *sycl_data;
  CeedQFunctionContext_Sycl *impl;

  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  CeedCallBackend(CeedGetData(ceed, &sycl_data));

  std::vector<sycl::event> e;
  if (!sycl_data->sycl_queue.is_in_order()) e = {sycl_data->sycl_queue.ext_oneapi_submit_barrier()};

  // Wait for all work to finish before freeing memory
  if (impl->d_data_owned) {
    CeedCallSycl(ceed, sycl_data->sycl_queue.wait_and_throw());
    CeedCallSycl(ceed, sycl::free(impl->d_data_owned, sycl_data->sycl_context));
    impl->d_data_owned = NULL;
  }

  switch (copy_mode) {
    case CEED_COPY_VALUES: {
      size_t ctx_size;

      CeedCallBackend(CeedQFunctionContextGetContextSize(ctx, &ctx_size));
      CeedCallSycl(ceed, impl->d_data_owned = sycl::malloc_device(ctx_size, sycl_data->sycl_device, sycl_data->sycl_context));
      impl->d_data_borrowed  = NULL;
      impl->d_data           = impl->d_data_owned;
      sycl::event copy_event = sycl_data->sycl_queue.memcpy(impl->d_data, data, ctx_size, {e});
      CeedCallSycl(ceed, copy_event.wait_and_throw());
    } break;
    case CEED_OWN_POINTER: {
      impl->d_data_owned    = data;
      impl->d_data_borrowed = NULL;
      impl->d_data          = data;
    } break;
    case CEED_USE_POINTER: {
      impl->d_data_owned    = NULL;
      impl->d_data_borrowed = data;
      impl->d_data          = data;
    } break;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set the data used by a user context,
//   freeing any previously allocated data if applicable
//------------------------------------------------------------------------------
static int CeedQFunctionContextSetData_Sycl(const CeedQFunctionContext ctx, const CeedMemType mem_type, const CeedCopyMode copy_mode, void *data) {
  Ceed ceed;

  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  CeedCallBackend(CeedQFunctionContextSetAllInvalid_Sycl(ctx));
  switch (mem_type) {
    case CEED_MEM_HOST:
      return CeedQFunctionContextSetDataHost_Sycl(ctx, copy_mode, data);
    case CEED_MEM_DEVICE:
      return CeedQFunctionContextSetDataDevice_Sycl(ctx, copy_mode, data);
  }
  return CEED_ERROR_UNSUPPORTED;
}

//------------------------------------------------------------------------------
// Take data
//------------------------------------------------------------------------------
static int CeedQFunctionContextTakeData_Sycl(const CeedQFunctionContext ctx, const CeedMemType mem_type, void *data) {
  Ceed                       ceed;
  Ceed_Sycl                 *ceedSycl;
  bool                       need_sync = false;
  CeedQFunctionContext_Sycl *impl;

  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));
  CeedCallBackend(CeedGetData(ceed, &ceedSycl));

  // Order queue if needed
  if (!ceedSycl->sycl_queue.is_in_order()) ceedSycl->sycl_queue.ext_oneapi_submit_barrier();

  // Sync data to requested mem_type
  CeedCallBackend(CeedQFunctionContextNeedSync_Sycl(ctx, mem_type, &need_sync));
  if (need_sync) CeedCallBackend(CeedQFunctionContextSync_Sycl(ctx, mem_type));

  // Update pointer
  switch (mem_type) {
    case CEED_MEM_HOST:
      *(void **)data        = impl->h_data_borrowed;
      impl->h_data_borrowed = NULL;
      impl->h_data          = NULL;
      break;
    case CEED_MEM_DEVICE:
      *(void **)data        = impl->d_data_borrowed;
      impl->d_data_borrowed = NULL;
      impl->d_data          = NULL;
      break;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Core logic for GetData.
//   If a different memory type is most up to date, this will perform a copy
//------------------------------------------------------------------------------
static int CeedQFunctionContextGetDataCore_Sycl(const CeedQFunctionContext ctx, const CeedMemType mem_type, void *data) {
  Ceed                       ceed;
  bool                       need_sync = false;
  CeedQFunctionContext_Sycl *impl;

  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));

  // Sync data to requested mem_type
  CeedCallBackend(CeedQFunctionContextNeedSync_Sycl(ctx, mem_type, &need_sync));
  if (need_sync) CeedCallBackend(CeedQFunctionContextSync_Sycl(ctx, mem_type));

  // Update pointer
  switch (mem_type) {
    case CEED_MEM_HOST:
      *(void **)data = impl->h_data;
      break;
    case CEED_MEM_DEVICE:
      *(void **)data = impl->d_data;
      break;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get read-only access to the data
//------------------------------------------------------------------------------
static int CeedQFunctionContextGetDataRead_Sycl(const CeedQFunctionContext ctx, const CeedMemType mem_type, void *data) {
  return CeedQFunctionContextGetDataCore_Sycl(ctx, mem_type, data);
}

//------------------------------------------------------------------------------
// Get read/write access to the data
//------------------------------------------------------------------------------
static int CeedQFunctionContextGetData_Sycl(const CeedQFunctionContext ctx, const CeedMemType mem_type, void *data) {
  Ceed                       ceed;
  CeedQFunctionContext_Sycl *impl;

  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  CeedCallBackend(CeedQFunctionContextGetDataCore_Sycl(ctx, mem_type, data));

  // Mark only pointer for requested memory as valid
  CeedCallBackend(CeedQFunctionContextSetAllInvalid_Sycl(ctx));
  switch (mem_type) {
    case CEED_MEM_HOST:
      impl->h_data = *(void **)data;
      break;
    case CEED_MEM_DEVICE:
      impl->d_data = *(void **)data;
      break;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy the user context
//------------------------------------------------------------------------------
static int CeedQFunctionContextDestroy_Sycl(const CeedQFunctionContext ctx) {
  Ceed                       ceed;
  Ceed_Sycl                 *sycl_data;
  CeedQFunctionContext_Sycl *impl;

  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));
  CeedCallBackend(CeedGetData(ceed, &sycl_data));

  // Wait for all work to finish before freeing memory
  CeedCallSycl(ceed, sycl_data->sycl_queue.wait_and_throw());
  CeedCallSycl(ceed, sycl::free(impl->d_data_owned, sycl_data->sycl_context));
  CeedCallBackend(CeedFree(&impl->h_data_owned));
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Create
//------------------------------------------------------------------------------
int CeedQFunctionContextCreate_Sycl(CeedQFunctionContext ctx) {
  Ceed                       ceed;
  CeedQFunctionContext_Sycl *impl;

  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "QFunctionContext", ctx, "HasValidData", CeedQFunctionContextHasValidData_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "QFunctionContext", ctx, "HasBorrowedDataOfType", CeedQFunctionContextHasBorrowedDataOfType_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "QFunctionContext", ctx, "SetData", CeedQFunctionContextSetData_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "QFunctionContext", ctx, "TakeData", CeedQFunctionContextTakeData_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "QFunctionContext", ctx, "GetData", CeedQFunctionContextGetData_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "QFunctionContext", ctx, "GetDataRead", CeedQFunctionContextGetDataRead_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "QFunctionContext", ctx, "Destroy", CeedQFunctionContextDestroy_Sycl));
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedQFunctionContextSetBackendData(ctx, impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
