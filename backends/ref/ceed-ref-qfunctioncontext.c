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
// QFunctionContext Has Valid Data
//------------------------------------------------------------------------------
static int CeedQFunctionContextHasValidData_Ref(CeedQFunctionContext ctx,
    bool *has_valid_data) {
  int ierr;
  CeedQFunctionContext_Ref *impl;
  ierr = CeedQFunctionContextGetBackendData(ctx, (void *)&impl);
  CeedChkBackend(ierr);

  *has_valid_data = !!impl->data;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Set Data
//------------------------------------------------------------------------------
static int CeedQFunctionContextSetData_Ref(CeedQFunctionContext ctx,
    CeedMemType mem_type, CeedCopyMode copy_mode, void *data) {
  int ierr;
  CeedQFunctionContext_Ref *impl;
  ierr = CeedQFunctionContextGetBackendData(ctx, (void *)&impl);
  CeedChkBackend(ierr);
  size_t ctx_size;
  ierr = CeedQFunctionContextGetContextSize(ctx, &ctx_size); CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedQFunctionContextGetCeed(ctx, &ceed); CeedChkBackend(ierr);

  if (mem_type != CEED_MEM_HOST)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Can only set HOST memory for this backend");
  // LCOV_EXCL_STOP

  ierr = CeedFree(&impl->data_owned); CeedChkBackend(ierr);
  switch (copy_mode) {
  case CEED_COPY_VALUES:
    ierr = CeedMallocArray(1, ctx_size, &impl->data_owned); CeedChkBackend(ierr);
    impl->data_borrowed = NULL;
    impl->data = impl->data_owned;
    memcpy(impl->data, data, ctx_size);
    break;
  case CEED_OWN_POINTER:
    impl->data_owned = data;
    impl->data_borrowed = NULL;
    impl->data = data;
    break;
  case CEED_USE_POINTER:
    impl->data_borrowed = data;
    impl->data = data;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Take Data
//------------------------------------------------------------------------------
static int CeedQFunctionContextTakeData_Ref(CeedQFunctionContext ctx,
    CeedMemType mem_type, void *data) {
  int ierr;
  CeedQFunctionContext_Ref *impl;
  ierr = CeedQFunctionContextGetBackendData(ctx, (void *)&impl);
  CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedQFunctionContextGetCeed(ctx, &ceed); CeedChkBackend(ierr);

  if (mem_type != CEED_MEM_HOST)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Can only provide HOST memory for this backend");
  // LCOV_EXCL_STOP

  if (!impl->data_borrowed)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Must set context data with CeedQFunctionContextSetData and CEED_USE_POINTER before calling CeedQFunctionContextTakeData");
  // LCOV_EXCL_STOP

  *(void **)data = impl->data;
  impl->data_borrowed = NULL;
  impl->data = NULL;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunctionContext Get Data
//------------------------------------------------------------------------------
static int CeedQFunctionContextGetData_Ref(CeedQFunctionContext ctx,
    CeedMemType mem_type, void *data) {
  int ierr;
  CeedQFunctionContext_Ref *impl;
  ierr = CeedQFunctionContextGetBackendData(ctx, (void *)&impl);
  CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedQFunctionContextGetCeed(ctx, &ceed); CeedChkBackend(ierr);

  if (mem_type != CEED_MEM_HOST)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Can only provide HOST memory for this backend");
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

  ierr = CeedFree(&impl->data_owned); CeedChkBackend(ierr);
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

  ierr = CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "HasValidData",
                                CeedQFunctionContextHasValidData_Ref);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "SetData",
                                CeedQFunctionContextSetData_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "TakeData",
                                CeedQFunctionContextTakeData_Ref); CeedChkBackend(ierr);
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
