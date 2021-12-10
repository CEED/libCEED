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
#include <hip/hip_runtime.h>
#include <string.h>
#include "ceed-hip.h"

//------------------------------------------------------------------------------
// * Bytes used
//------------------------------------------------------------------------------
static inline size_t bytes(const CeedQFunctionContext ctx) {
  int ierr;
  size_t ctxsize;
  ierr = CeedQFunctionContextGetContextSize(ctx, &ctxsize); CeedChkBackend(ierr);
  return ctxsize;
}

//------------------------------------------------------------------------------
// Sync host to device
//------------------------------------------------------------------------------
static inline int CeedQFunctionContextSyncH2D_Hip(
  const CeedQFunctionContext ctx) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionContextGetCeed(ctx, &ceed); CeedChkBackend(ierr);
  CeedQFunctionContext_Hip *impl;
  ierr = CeedQFunctionContextGetBackendData(ctx, &impl); CeedChkBackend(ierr);

  if (impl->d_data_borrowed) {
    impl->d_data = impl->d_data_borrowed;
  } else if (impl->d_data_owned) {
    impl->d_data = impl->d_data_owned;
  } else {
    ierr = hipMalloc((void **)&impl->d_data_owned, bytes(ctx));
    CeedChk_Hip(ceed, ierr);
    impl->d_data = impl->d_data_owned;
  }

  ierr = hipMemcpy(impl->d_data, impl->h_data, bytes(ctx),
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync device to host
//------------------------------------------------------------------------------
static inline int CeedQFunctionContextSyncD2H_Hip(
  const CeedQFunctionContext ctx) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionContextGetCeed(ctx, &ceed); CeedChkBackend(ierr);
  CeedQFunctionContext_Hip *impl;
  ierr = CeedQFunctionContextGetBackendData(ctx, &impl); CeedChkBackend(ierr);

  if (impl->h_data_borrowed) {
    impl->h_data = impl->h_data_borrowed;
  } else if (impl->h_data_owned) {
    impl->h_data = impl->h_data_owned;
  } else {
    ierr = CeedMalloc(bytes(ctx), &impl->h_data_owned);
    CeedChkBackend(ierr);
    impl->h_data = impl->h_data_owned;
  }

  ierr = hipMemcpy(impl->h_data, impl->d_data, bytes(ctx),
                   hipMemcpyDeviceToHost); CeedChk_Hip(ceed, ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync data of type
//------------------------------------------------------------------------------
static inline int CeedQFunctionContextSync_Hip(const CeedQFunctionContext ctx,
    CeedMemType mtype) {
  switch (mtype) {
  case CEED_MEM_HOST: return CeedQFunctionContextSyncD2H_Hip(ctx);
  case CEED_MEM_DEVICE: return CeedQFunctionContextSyncH2D_Hip(ctx);
  }
  return CEED_ERROR_UNSUPPORTED;
}

//------------------------------------------------------------------------------
// Set all pointers as stale
//------------------------------------------------------------------------------
static inline int CeedQFunctionContextSetAllStale_Hip(
  const CeedQFunctionContext ctx) {
  int ierr;
  CeedQFunctionContext_Hip *data;
  ierr = CeedQFunctionContextGetBackendData(ctx, &data); CeedChkBackend(ierr);

  data->h_data = NULL;
  data->d_data = NULL;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if all pointers are stale
//------------------------------------------------------------------------------
static inline int CeedQFunctionContextIsAllStale_Hip(
  const CeedQFunctionContext ctx, bool *is_all_stale) {
  int ierr;
  CeedQFunctionContext_Hip *data;
  ierr = CeedQFunctionContextGetBackendData(ctx, &data); CeedChkBackend(ierr);

  *is_all_stale = !data->h_data && !data->d_data;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if data of given type needs sync
//------------------------------------------------------------------------------
static inline int CeedQFunctionContextNeedSync_Hip(
  const CeedQFunctionContext ctx, CeedMemType mtype, bool *need_sync) {
  int ierr;
  CeedQFunctionContext_Hip *data;
  ierr = CeedQFunctionContextGetBackendData(ctx, &data); CeedChkBackend(ierr);

  bool is_all_stale = false;
  ierr = CeedQFunctionContextIsAllStale_Hip(ctx, &is_all_stale);
  CeedChkBackend(ierr);
  switch (mtype) {
  case CEED_MEM_HOST:
    *need_sync = !is_all_stale && !data->h_data;
    break;
  case CEED_MEM_DEVICE:
    *need_sync = !is_all_stale && !data->d_data;
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set data from host
//------------------------------------------------------------------------------
static int CeedQFunctionContextSetDataHost_Hip(const CeedQFunctionContext ctx,
    const CeedCopyMode cmode, void *data) {
  int ierr;
  CeedQFunctionContext_Hip *impl;
  ierr = CeedQFunctionContextGetBackendData(ctx, &impl); CeedChkBackend(ierr);

  ierr = CeedFree(&impl->h_data_owned); CeedChkBackend(ierr);
  switch (cmode) {
  case CEED_COPY_VALUES: {
    ierr = CeedMalloc(bytes(ctx), &impl->h_data_owned); CeedChkBackend(ierr);
    impl->h_data_borrowed = NULL;
    impl->h_data = impl->h_data_owned;
    memcpy(impl->h_data, data, bytes(ctx));
  } break;
  case CEED_OWN_POINTER:
    impl->h_data_owned = data;
    impl->h_data_borrowed = NULL;
    impl->h_data = data;
    break;
  case CEED_USE_POINTER:
    impl->h_data_borrowed = data;
    impl->h_data = data;
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set data from device
//------------------------------------------------------------------------------
static int CeedQFunctionContextSetDataDevice_Hip(const CeedQFunctionContext ctx,
    const CeedCopyMode cmode, void *data) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionContextGetCeed(ctx, &ceed); CeedChkBackend(ierr);
  CeedQFunctionContext_Hip *impl;
  ierr = CeedQFunctionContextGetBackendData(ctx, &impl); CeedChkBackend(ierr);

  ierr = hipFree(impl->d_data_owned); CeedChk_Hip(ceed, ierr);
  impl->d_data_owned = NULL;
  switch (cmode) {
  case CEED_COPY_VALUES:
    ierr = hipMalloc((void **)&impl->d_data_owned, bytes(ctx));
    CeedChk_Hip(ceed, ierr);
    impl->d_data_borrowed = NULL;
    impl->d_data = impl->d_data_owned;
    ierr = hipMemcpy(impl->d_data, data, bytes(ctx),
                     hipMemcpyDeviceToDevice); CeedChk_Hip(ceed, ierr);
    break;
  case CEED_OWN_POINTER:
    impl->d_data_owned = data;
    impl->d_data_borrowed = NULL;
    impl->d_data = data;
    break;
  case CEED_USE_POINTER:
    impl->d_data_owned = NULL;
    impl->d_data_borrowed = data;
    impl->d_data = data;
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set the data used by a user context,
//   freeing any previously allocated data if applicable
//------------------------------------------------------------------------------
static int CeedQFunctionContextSetData_Hip(const CeedQFunctionContext ctx,
    const CeedMemType mtype, const CeedCopyMode cmode, void *data) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionContextGetCeed(ctx, &ceed); CeedChkBackend(ierr);

  ierr = CeedQFunctionContextSetAllStale_Hip(ctx); CeedChkBackend(ierr);
  switch (mtype) {
  case CEED_MEM_HOST:
    return CeedQFunctionContextSetDataHost_Hip(ctx, cmode, data);
  case CEED_MEM_DEVICE:
    return CeedQFunctionContextSetDataDevice_Hip(ctx, cmode, data);
  }

  return CEED_ERROR_UNSUPPORTED;
}

//------------------------------------------------------------------------------
// Take data
//------------------------------------------------------------------------------
static int CeedQFunctionContextTakeData_Hip(const CeedQFunctionContext ctx,
    const CeedMemType mtype, void *data) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionContextGetCeed(ctx, &ceed); CeedChkBackend(ierr);
  CeedQFunctionContext_Hip *impl;
  ierr = CeedQFunctionContextGetBackendData(ctx, &impl); CeedChkBackend(ierr);

  bool is_all_stale = false;
  ierr = CeedQFunctionContextIsAllStale_Hip(ctx, &is_all_stale);
  CeedChkBackend(ierr);
  if (is_all_stale)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "No valid context data set");
  // LCOV_EXCL_STOP

  // Sync data to requested memtype
  bool need_sync = false;
  ierr = CeedQFunctionContextNeedSync_Hip(ctx, mtype, &need_sync);
  CeedChkBackend(ierr);
  if (need_sync) {
    ierr = CeedQFunctionContextSync_Hip(ctx, mtype); CeedChkBackend(ierr);
  }

  // Update pointer
  switch (mtype) {
  case CEED_MEM_HOST:
    if (!impl->h_data_borrowed)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "Must set HOST context data with CeedQFunctionContextSetData and CEED_USE_POINTER before calling CeedQFunctionContextTakeData");
    // LCOV_EXCL_STOP

    *(void **)data = impl->h_data_borrowed;
    impl->h_data_borrowed = NULL;
    impl->h_data = NULL;
    break;
  case CEED_MEM_DEVICE:
    if (!impl->d_data_borrowed)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "Must set DEVICE context data with CeedQFunctionContextSetData and CEED_USE_POINTER before calling CeedQFunctionContextTakeData");
    // LCOV_EXCL_STOP

    *(void **)data = impl->d_data_borrowed;
    impl->d_data_borrowed = NULL;
    impl->d_data = NULL;
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get data
//------------------------------------------------------------------------------
static int CeedQFunctionContextGetData_Hip(const CeedQFunctionContext ctx,
    const CeedMemType mtype, void *data) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionContextGetCeed(ctx, &ceed); CeedChkBackend(ierr);
  CeedQFunctionContext_Hip *impl;
  ierr = CeedQFunctionContextGetBackendData(ctx, &impl); CeedChkBackend(ierr);

  bool is_all_stale = false;
  ierr = CeedQFunctionContextIsAllStale_Hip(ctx, &is_all_stale);
  CeedChkBackend(ierr);
  if (is_all_stale)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "No valid context data set");
  // LCOV_EXCL_STOP

  // Sync data to requested memtype
  bool need_sync = false;
  ierr = CeedQFunctionContextNeedSync_Hip(ctx, mtype, &need_sync);
  CeedChkBackend(ierr);
  if (need_sync) {
    ierr = CeedQFunctionContextSync_Hip(ctx, mtype); CeedChkBackend(ierr);
  }

  // Sync data to requested memtype and update pointer
  switch (mtype) {
  case CEED_MEM_HOST:
    *(void **)data = impl->h_data;
    break;
  case CEED_MEM_DEVICE:
    *(void **)data = impl->d_data;
    break;
  }

  // Mark only pointer for requested memory as valid
  ierr = CeedQFunctionContextSetAllStale_Hip(ctx); CeedChkBackend(ierr);
  switch (mtype) {
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
// Restore data obtained using CeedQFunctionContextGetData()
//------------------------------------------------------------------------------
static int CeedQFunctionContextRestoreData_Hip(const CeedQFunctionContext ctx) {
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy the user context
//------------------------------------------------------------------------------
static int CeedQFunctionContextDestroy_Hip(const CeedQFunctionContext ctx) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionContextGetCeed(ctx, &ceed); CeedChkBackend(ierr);
  CeedQFunctionContext_Hip *impl;
  ierr = CeedQFunctionContextGetBackendData(ctx, &impl); CeedChkBackend(ierr);

  ierr = hipFree(impl->d_data_owned); CeedChk_Hip(ceed, ierr);
  ierr = CeedFree(&impl->h_data_owned); CeedChkBackend(ierr);
  ierr = CeedFree(&impl); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Create
//------------------------------------------------------------------------------
int CeedQFunctionContextCreate_Hip(CeedQFunctionContext ctx) {
  int ierr;
  CeedQFunctionContext_Hip *impl;
  Ceed ceed;
  ierr = CeedQFunctionContextGetCeed(ctx, &ceed); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "SetData",
                                CeedQFunctionContextSetData_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "TakeData",
                                CeedQFunctionContextTakeData_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "GetData",
                                CeedQFunctionContextGetData_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "RestoreData",
                                CeedQFunctionContextRestoreData_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "Destroy",
                                CeedQFunctionContextDestroy_Hip); CeedChkBackend(ierr);

  ierr = CeedCalloc(1, &impl); CeedChkBackend(ierr);
  ierr = CeedQFunctionContextSetBackendData(ctx, impl); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
