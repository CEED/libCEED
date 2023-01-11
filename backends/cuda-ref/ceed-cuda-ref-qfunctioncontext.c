// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <cuda_runtime.h>
#include <string.h>

#include "ceed-cuda-ref.h"

//------------------------------------------------------------------------------
// Sync host to device
//------------------------------------------------------------------------------
static inline int CeedQFunctionContextSyncH2D_Cuda(const CeedQFunctionContext ctx) {
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  CeedQFunctionContext_Cuda *impl;
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));

  if (!impl->h_data) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "No valid host data to sync to device");
    // LCOV_EXCL_STOP
  }

  size_t ctxsize;
  CeedCallBackend(CeedQFunctionContextGetContextSize(ctx, &ctxsize));

  if (impl->d_data_borrowed) {
    impl->d_data = impl->d_data_borrowed;
  } else if (impl->d_data_owned) {
    impl->d_data = impl->d_data_owned;
  } else {
    CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_data_owned, ctxsize));
    impl->d_data = impl->d_data_owned;
  }

  CeedCallCuda(ceed, cudaMemcpy(impl->d_data, impl->h_data, ctxsize, cudaMemcpyHostToDevice));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync device to host
//------------------------------------------------------------------------------
static inline int CeedQFunctionContextSyncD2H_Cuda(const CeedQFunctionContext ctx) {
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  CeedQFunctionContext_Cuda *impl;
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));

  if (!impl->d_data) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "No valid device data to sync to host");
    // LCOV_EXCL_STOP
  }

  size_t ctxsize;
  CeedCallBackend(CeedQFunctionContextGetContextSize(ctx, &ctxsize));

  if (impl->h_data_borrowed) {
    impl->h_data = impl->h_data_borrowed;
  } else if (impl->h_data_owned) {
    impl->h_data = impl->h_data_owned;
  } else {
    CeedCallBackend(CeedMallocArray(1, ctxsize, &impl->h_data_owned));
    impl->h_data = impl->h_data_owned;
  }

  CeedCallCuda(ceed, cudaMemcpy(impl->h_data, impl->d_data, ctxsize, cudaMemcpyDeviceToHost));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync data of type
//------------------------------------------------------------------------------
static inline int CeedQFunctionContextSync_Cuda(const CeedQFunctionContext ctx, CeedMemType mem_type) {
  switch (mem_type) {
    case CEED_MEM_HOST:
      return CeedQFunctionContextSyncD2H_Cuda(ctx);
    case CEED_MEM_DEVICE:
      return CeedQFunctionContextSyncH2D_Cuda(ctx);
  }
  return CEED_ERROR_UNSUPPORTED;
}

//------------------------------------------------------------------------------
// Set all pointers as invalid
//------------------------------------------------------------------------------
static inline int CeedQFunctionContextSetAllInvalid_Cuda(const CeedQFunctionContext ctx) {
  CeedQFunctionContext_Cuda *impl;
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));

  impl->h_data = NULL;
  impl->d_data = NULL;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if ctx has valid data
//------------------------------------------------------------------------------
static inline int CeedQFunctionContextHasValidData_Cuda(const CeedQFunctionContext ctx, bool *has_valid_data) {
  CeedQFunctionContext_Cuda *impl;
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));

  *has_valid_data = impl && (!!impl->h_data || !!impl->d_data);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if ctx has borrowed data
//------------------------------------------------------------------------------
static inline int CeedQFunctionContextHasBorrowedDataOfType_Cuda(const CeedQFunctionContext ctx, CeedMemType mem_type,
                                                                 bool *has_borrowed_data_of_type) {
  CeedQFunctionContext_Cuda *impl;
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));

  switch (mem_type) {
    case CEED_MEM_HOST:
      *has_borrowed_data_of_type = !!impl->h_data_borrowed;
      break;
    case CEED_MEM_DEVICE:
      *has_borrowed_data_of_type = !!impl->d_data_borrowed;
      break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if data of given type needs sync
//------------------------------------------------------------------------------
static inline int CeedQFunctionContextNeedSync_Cuda(const CeedQFunctionContext ctx, CeedMemType mem_type, bool *need_sync) {
  CeedQFunctionContext_Cuda *impl;
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));

  bool has_valid_data = true;
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
static int CeedQFunctionContextSetDataHost_Cuda(const CeedQFunctionContext ctx, const CeedCopyMode copy_mode, void *data) {
  CeedQFunctionContext_Cuda *impl;
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));

  CeedCallBackend(CeedFree(&impl->h_data_owned));
  switch (copy_mode) {
    case CEED_COPY_VALUES: {
      size_t ctxsize;
      CeedCallBackend(CeedQFunctionContextGetContextSize(ctx, &ctxsize));
      CeedCallBackend(CeedMallocArray(1, ctxsize, &impl->h_data_owned));
      impl->h_data_borrowed = NULL;
      impl->h_data          = impl->h_data_owned;
      memcpy(impl->h_data, data, ctxsize);
    } break;
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
static int CeedQFunctionContextSetDataDevice_Cuda(const CeedQFunctionContext ctx, const CeedCopyMode copy_mode, void *data) {
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  CeedQFunctionContext_Cuda *impl;
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));

  CeedCallCuda(ceed, cudaFree(impl->d_data_owned));
  impl->d_data_owned = NULL;
  switch (copy_mode) {
    case CEED_COPY_VALUES: {
      size_t ctxsize;
      CeedCallBackend(CeedQFunctionContextGetContextSize(ctx, &ctxsize));
      CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_data_owned, ctxsize));
      impl->d_data_borrowed = NULL;
      impl->d_data          = impl->d_data_owned;
      CeedCallCuda(ceed, cudaMemcpy(impl->d_data, data, ctxsize, cudaMemcpyDeviceToDevice));
    } break;
    case CEED_OWN_POINTER:
      impl->d_data_owned    = data;
      impl->d_data_borrowed = NULL;
      impl->d_data          = data;
      break;
    case CEED_USE_POINTER:
      impl->d_data_owned    = NULL;
      impl->d_data_borrowed = data;
      impl->d_data          = data;
      break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set the data used by a user context,
//   freeing any previously allocated data if applicable
//------------------------------------------------------------------------------
static int CeedQFunctionContextSetData_Cuda(const CeedQFunctionContext ctx, const CeedMemType mem_type, const CeedCopyMode copy_mode, void *data) {
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));

  CeedCallBackend(CeedQFunctionContextSetAllInvalid_Cuda(ctx));
  switch (mem_type) {
    case CEED_MEM_HOST:
      return CeedQFunctionContextSetDataHost_Cuda(ctx, copy_mode, data);
    case CEED_MEM_DEVICE:
      return CeedQFunctionContextSetDataDevice_Cuda(ctx, copy_mode, data);
  }

  return CEED_ERROR_UNSUPPORTED;
}

//------------------------------------------------------------------------------
// Take data
//------------------------------------------------------------------------------
static int CeedQFunctionContextTakeData_Cuda(const CeedQFunctionContext ctx, const CeedMemType mem_type, void *data) {
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  CeedQFunctionContext_Cuda *impl;
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));

  // Sync data to requested mem_type
  bool need_sync = false;
  CeedCallBackend(CeedQFunctionContextNeedSync_Cuda(ctx, mem_type, &need_sync));
  if (need_sync) CeedCallBackend(CeedQFunctionContextSync_Cuda(ctx, mem_type));

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
static int CeedQFunctionContextGetDataCore_Cuda(const CeedQFunctionContext ctx, const CeedMemType mem_type, void *data) {
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  CeedQFunctionContext_Cuda *impl;
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));

  // Sync data to requested mem_type
  bool need_sync = false;
  CeedCallBackend(CeedQFunctionContextNeedSync_Cuda(ctx, mem_type, &need_sync));
  if (need_sync) CeedCallBackend(CeedQFunctionContextSync_Cuda(ctx, mem_type));

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
static int CeedQFunctionContextGetDataRead_Cuda(const CeedQFunctionContext ctx, const CeedMemType mem_type, void *data) {
  return CeedQFunctionContextGetDataCore_Cuda(ctx, mem_type, data);
}

//------------------------------------------------------------------------------
// Get read/write access to the data
//------------------------------------------------------------------------------
static int CeedQFunctionContextGetData_Cuda(const CeedQFunctionContext ctx, const CeedMemType mem_type, void *data) {
  CeedQFunctionContext_Cuda *impl;
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));

  CeedCallBackend(CeedQFunctionContextGetDataCore_Cuda(ctx, mem_type, data));

  // Mark only pointer for requested memory as valid
  CeedCallBackend(CeedQFunctionContextSetAllInvalid_Cuda(ctx));
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
static int CeedQFunctionContextDestroy_Cuda(const CeedQFunctionContext ctx) {
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  CeedQFunctionContext_Cuda *impl;
  CeedCallBackend(CeedQFunctionContextGetBackendData(ctx, &impl));

  CeedCallCuda(ceed, cudaFree(impl->d_data_owned));
  CeedCallBackend(CeedFree(&impl->h_data_owned));
  CeedCallBackend(CeedFree(&impl));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Create
//------------------------------------------------------------------------------
int CeedQFunctionContextCreate_Cuda(CeedQFunctionContext ctx) {
  CeedQFunctionContext_Cuda *impl;
  Ceed                       ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));

  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "HasValidData", CeedQFunctionContextHasValidData_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "HasBorrowedDataOfType", CeedQFunctionContextHasBorrowedDataOfType_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "SetData", CeedQFunctionContextSetData_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "TakeData", CeedQFunctionContextTakeData_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "GetData", CeedQFunctionContextGetData_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "GetDataRead", CeedQFunctionContextGetDataRead_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "Destroy", CeedQFunctionContextDestroy_Cuda));

  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedQFunctionContextSetBackendData(ctx, impl));

  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
