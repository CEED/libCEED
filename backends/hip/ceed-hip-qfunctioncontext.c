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

  ierr = hipMemcpy(impl->h_data, impl->d_data, bytes(ctx),
                   hipMemcpyDeviceToHost); CeedChk_Hip(ceed, ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set data from host
//------------------------------------------------------------------------------
static int CeedQFunctionContextSetDataHost_Hip(const CeedQFunctionContext ctx,
    const CeedCopyMode cmode, CeedScalar *data) {
  int ierr;
  CeedQFunctionContext_Hip *impl;
  ierr = CeedQFunctionContextGetBackendData(ctx, &impl); CeedChkBackend(ierr);

  switch (cmode) {
  case CEED_COPY_VALUES: {
    if(!impl->h_data) {
      ierr = CeedMalloc(bytes(ctx), &impl->h_data_allocated); CeedChkBackend(ierr);
      impl->h_data = impl->h_data_allocated;
    }
    memcpy(impl->h_data, data, bytes(ctx));
  } break;
  case CEED_OWN_POINTER:
    ierr = CeedFree(&impl->h_data_allocated); CeedChkBackend(ierr);
    impl->h_data_allocated = data;
    impl->h_data = data;
    break;
  case CEED_USE_POINTER:
    ierr = CeedFree(&impl->h_data_allocated); CeedChkBackend(ierr);
    impl->h_data = data;
    break;
  }
  impl->memState = CEED_HIP_HOST_SYNC;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set data from device
//------------------------------------------------------------------------------
static int CeedQFunctionContextSetDataDevice_Hip(const CeedQFunctionContext ctx,
    const CeedCopyMode cmode, CeedScalar *data) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionContextGetCeed(ctx, &ceed); CeedChkBackend(ierr);
  CeedQFunctionContext_Hip *impl;
  ierr = CeedQFunctionContextGetBackendData(ctx, &impl); CeedChkBackend(ierr);

  switch (cmode) {
  case CEED_COPY_VALUES:
    if (!impl->d_data) {
      ierr = hipMalloc((void **)&impl->d_data_allocated, bytes(ctx));
      CeedChk_Hip(ceed, ierr);
      impl->d_data = impl->d_data_allocated;
    }
    ierr = hipMemcpy(impl->d_data, data, bytes(ctx),
                     hipMemcpyDeviceToDevice); CeedChk_Hip(ceed, ierr);
    break;
  case CEED_OWN_POINTER:
    ierr = hipFree(impl->d_data_allocated); CeedChk_Hip(ceed, ierr);
    impl->d_data_allocated = data;
    impl->d_data = data;
    break;
  case CEED_USE_POINTER:
    ierr = hipFree(impl->d_data_allocated); CeedChk_Hip(ceed, ierr);
    impl->d_data_allocated = NULL;
    impl->d_data = data;
    break;
  }
  impl->memState = CEED_HIP_DEVICE_SYNC;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set the array used by a user context,
//   freeing any previously allocated array if applicable
//------------------------------------------------------------------------------
static int CeedQFunctionContextSetData_Hip(const CeedQFunctionContext ctx,
    const CeedMemType mtype, const CeedCopyMode cmode, CeedScalar *data) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionContextGetCeed(ctx, &ceed); CeedChkBackend(ierr);

  switch (mtype) {
  case CEED_MEM_HOST:
    return CeedQFunctionContextSetDataHost_Hip(ctx, cmode, data);
  case CEED_MEM_DEVICE:
    return CeedQFunctionContextSetDataDevice_Hip(ctx, cmode, data);
  }
  return 1;
}

//------------------------------------------------------------------------------
// Take data
//------------------------------------------------------------------------------
static int CeedQFunctionContextTakeData_Hip(const CeedQFunctionContext ctx,
    const CeedMemType mtype, CeedScalar *data) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionContextGetCeed(ctx, &ceed); CeedChkBackend(ierr);
  CeedQFunctionContext_Hip *impl;
  ierr = CeedQFunctionContextGetBackendData(ctx, &impl); CeedChkBackend(ierr);
  if(impl->h_data == NULL && impl->d_data == NULL)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "No context data set");
  // LCOV_EXCL_STOP

  // Sync array to requested memtype and update pointer
  switch (mtype) {
  case CEED_MEM_HOST:
    if (impl->h_data == NULL) {
      ierr = CeedMalloc(bytes(ctx), &impl->h_data_allocated);
      CeedChkBackend(ierr);
      impl->h_data = impl->h_data_allocated;
    }
    if (impl->memState == CEED_HIP_DEVICE_SYNC) {
      ierr = CeedQFunctionContextSyncD2H_Hip(ctx); CeedChkBackend(ierr);
    }
    impl->memState = CEED_HIP_HOST_SYNC;
    *(void **)data = impl->h_data;
    impl->h_data = NULL;
    impl->h_data_allocated = NULL;
    break;
  case CEED_MEM_DEVICE:
    if (impl->d_data == NULL) {
      ierr = hipMalloc((void **)&impl->d_data_allocated, bytes(ctx));
      CeedChk_Hip(ceed, ierr);
      impl->d_data = impl->d_data_allocated;
    }
    if (impl->memState == CEED_HIP_HOST_SYNC) {
      ierr = CeedQFunctionContextSyncH2D_Hip(ctx); CeedChkBackend(ierr);
    }
    impl->memState = CEED_HIP_DEVICE_SYNC;
    *(void **)data = impl->d_data;
    impl->d_data = NULL;
    impl->d_data_allocated = NULL;
    break;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get data
//------------------------------------------------------------------------------
static int CeedQFunctionContextGetData_Hip(const CeedQFunctionContext ctx,
    const CeedMemType mtype, CeedScalar *data) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionContextGetCeed(ctx, &ceed); CeedChkBackend(ierr);
  CeedQFunctionContext_Hip *impl;
  ierr = CeedQFunctionContextGetBackendData(ctx, &impl); CeedChkBackend(ierr);
  if(impl->h_data == NULL && impl->d_data == NULL)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "No context data set");
  // LCOV_EXCL_STOP

  // Sync array to requested memtype and update pointer
  switch (mtype) {
  case CEED_MEM_HOST:
    if (impl->h_data == NULL) {
      ierr = CeedMalloc(bytes(ctx), &impl->h_data_allocated);
      CeedChkBackend(ierr);
      impl->h_data = impl->h_data_allocated;
    }
    if (impl->memState == CEED_HIP_DEVICE_SYNC) {
      ierr = CeedQFunctionContextSyncD2H_Hip(ctx); CeedChkBackend(ierr);
    }
    impl->memState = CEED_HIP_HOST_SYNC;
    *(void **)data = impl->h_data;
    break;
  case CEED_MEM_DEVICE:
    if (impl->d_data == NULL) {
      ierr = hipMalloc((void **)&impl->d_data_allocated, bytes(ctx));
      CeedChk_Hip(ceed, ierr);
      impl->d_data = impl->d_data_allocated;
    }
    if (impl->memState == CEED_HIP_HOST_SYNC) {
      ierr = CeedQFunctionContextSyncH2D_Hip(ctx); CeedChkBackend(ierr);
    }
    impl->memState = CEED_HIP_DEVICE_SYNC;
    *(void **)data = impl->d_data;
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

  ierr = hipFree(impl->d_data_allocated); CeedChk_Hip(ceed, ierr);
  ierr = CeedFree(&impl->h_data_allocated); CeedChkBackend(ierr);
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
  impl->memState = CEED_HIP_NONE_SYNC;
  ierr = CeedQFunctionContextSetBackendData(ctx, impl); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
