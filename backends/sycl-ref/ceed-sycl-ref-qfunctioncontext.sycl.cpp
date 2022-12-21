// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
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
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Sync device to host
//------------------------------------------------------------------------------
static inline int CeedQFunctionContextSyncD2H_Sycl(const CeedQFunctionContext ctx) {
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Sync data of type
//------------------------------------------------------------------------------
static inline int CeedQFunctionContextSync_Sycl(const CeedQFunctionContext ctx, CeedMemType mem_type) {
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Set all pointers as invalid
//------------------------------------------------------------------------------
static inline int CeedQFunctionContextSetAllInvalid_Sycl(const CeedQFunctionContext ctx) {
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Check if ctx has valid data
//------------------------------------------------------------------------------
static inline int CeedQFunctionContextHasValidData_Sycl(const CeedQFunctionContext ctx, bool *has_valid_data) {
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Check if ctx has borrowed data
//------------------------------------------------------------------------------
static inline int CeedQFunctionContextHasBorrowedDataOfType_Sycl(const CeedQFunctionContext ctx, CeedMemType mem_type,
                                                                 bool *has_borrowed_data_of_type) {
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Check if data of given type needs sync
//------------------------------------------------------------------------------
static inline int CeedQFunctionContextNeedSync_Sycl(const CeedQFunctionContext ctx, CeedMemType mem_type, bool *need_sync) {
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Set data from host
//------------------------------------------------------------------------------
static int CeedQFunctionContextSetDataHost_Sycl(const CeedQFunctionContext ctx, const CeedCopyMode copy_mode, void *data) {
  return CeedError(NULL, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Set data from device
//------------------------------------------------------------------------------
static int CeedQFunctionContextSetDataDevice_Sycl(const CeedQFunctionContext ctx, const CeedCopyMode copy_mode, void *data) {
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Set the data used by a user context,
//   freeing any previously allocated data if applicable
//------------------------------------------------------------------------------
static int CeedQFunctionContextSetData_Sycl(const CeedQFunctionContext ctx, const CeedMemType mem_type, const CeedCopyMode copy_mode, void *data) {
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Take data
//------------------------------------------------------------------------------
static int CeedQFunctionContextTakeData_Sycl(const CeedQFunctionContext ctx, const CeedMemType mem_type, void *data) {
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Core logic for GetData.
//   If a different memory type is most up to date, this will perform a copy
//------------------------------------------------------------------------------
static int CeedQFunctionContextGetDataCore_Sycl(const CeedQFunctionContext ctx, const CeedMemType mem_type, void *data) {
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Get read-only access to the data
//------------------------------------------------------------------------------
static int CeedQFunctionContextGetDataRead_Sycl(const CeedQFunctionContext ctx, const CeedMemType mem_type, void *data) {
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Get read/write access to the data
//------------------------------------------------------------------------------
static int CeedQFunctionContextGetData_Sycl(const CeedQFunctionContext ctx, const CeedMemType mem_type, void *data) {
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Destroy the user context
//------------------------------------------------------------------------------
static int CeedQFunctionContextDestroy_Sycl(const CeedQFunctionContext ctx) {
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// QFunctionContext Create
//------------------------------------------------------------------------------
int CeedQFunctionContextCreate_Sycl(CeedQFunctionContext ctx) {
  CeedQFunctionContext_Sycl *impl;
  Ceed                       ceed;
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}
//------------------------------------------------------------------------------
