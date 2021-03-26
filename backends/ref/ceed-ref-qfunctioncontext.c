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
#include "ceed-ref.h"

//------------------------------------------------------------------------------
// QFunctionContext Set Data
//------------------------------------------------------------------------------
static int CeedQFunctionContextSetData_Ref(CeedQFunctionContext ctx,
    CeedMemType mtype,
    CeedCopyMode cmode, CeedScalar *data) {
  int ierr;
  CeedQFunctionContext_Ref *impl;
  ierr = CeedQFunctionContextGetBackendData(ctx, (void *)&impl);
  CeedChkBackend(ierr);
  size_t ctxsize;
  ierr = CeedQFunctionContextGetContextSize(ctx, &ctxsize); CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedQFunctionContextGetCeed(ctx, &ceed); CeedChkBackend(ierr);

  if (mtype != CEED_MEM_HOST)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Only MemType = HOST supported");
  // LCOV_EXCL_STOP
  ierr = CeedFree(&impl->data_allocated); CeedChkBackend(ierr);
  switch (cmode) {
  case CEED_COPY_VALUES:
    ierr = CeedMallocArray(1, ctxsize, &impl->data_allocated); CeedChkBackend(ierr);
    impl->data = impl->data_allocated;
    memcpy(impl->data, data, ctxsize);
    break;
  case CEED_OWN_POINTER:
    impl->data_allocated = data;
    impl->data = data;
    break;
  case CEED_USE_POINTER:
    impl->data = data;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Get Data
//------------------------------------------------------------------------------
static int CeedQFunctionContextGetData_Ref(CeedQFunctionContext ctx,
    CeedMemType mtype, CeedScalar *data) {
  int ierr;
  CeedQFunctionContext_Ref *impl;
  ierr = CeedQFunctionContextGetBackendData(ctx, (void *)&impl);
  CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedQFunctionContextGetCeed(ctx, &ceed); CeedChkBackend(ierr);

  if (mtype != CEED_MEM_HOST)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Can only provide to HOST memory");
  // LCOV_EXCL_STOP
  if (!impl->data)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "No context data set");
  // LCOV_EXCL_STOP
  *(void **)data = impl->data;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Restore Data
//------------------------------------------------------------------------------
static int CeedQFunctionContextRestoreData_Ref(CeedQFunctionContext ctx) {
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Destroy
//------------------------------------------------------------------------------
static int CeedQFunctionContextDestroy_Ref(CeedQFunctionContext ctx) {
  int ierr;
  CeedQFunctionContext_Ref *impl;
  ierr = CeedQFunctionContextGetBackendData(ctx, &impl); CeedChkBackend(ierr);

  ierr = CeedFree(&impl->data_allocated); CeedChkBackend(ierr);
  ierr = CeedFree(&impl); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Create
//------------------------------------------------------------------------------
int CeedQFunctionContextCreate_Ref(CeedQFunctionContext ctx) {
  int ierr;
  CeedQFunctionContext_Ref *impl;
  Ceed ceed;
  ierr = CeedQFunctionContextGetCeed(ctx, &ceed); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "SetData",
                                CeedQFunctionContextSetData_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "GetData",
                                CeedQFunctionContextGetData_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "RestoreData",
                                CeedQFunctionContextRestoreData_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "Destroy",
                                CeedQFunctionContextDestroy_Ref); CeedChkBackend(ierr);
  ierr = CeedCalloc(1, &impl); CeedChkBackend(ierr);
  ierr = CeedQFunctionContextSetBackendData(ctx, impl); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
