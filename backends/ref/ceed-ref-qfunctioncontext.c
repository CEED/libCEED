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

#include "ceed-ref.h"

//------------------------------------------------------------------------------
// QFunctionContext Set Data
//------------------------------------------------------------------------------
static int CeedQFunctionContextSetData_Ref(CeedQFunctionContext ctx,
    CeedMemType mtype,
    CeedCopyMode cmode, CeedScalar *data) {
  int ierr;
  CeedQFunctionContext_Ref *impl;
  ierr = CeedQFunctionContextGetBackendData(ctx, (void *)&impl); CeedChk(ierr);
  size_t ctxsize;
  ierr = CeedQFunctionContextGetContextSize(ctx, &ctxsize); CeedChk(ierr);
  Ceed ceed;
  ierr = CeedQFunctionContextGetCeed(ctx, &ceed); CeedChk(ierr);

  if (mtype != CEED_MEM_HOST)
    // LCOV_EXCL_START
    return CeedError(ceed, 1, "Only MemType = HOST supported");
  // LCOV_EXCL_STOP
  ierr = CeedFree(&impl->data_allocated); CeedChk(ierr);
  switch (cmode) {
  case CEED_COPY_VALUES:
    ierr = CeedMallocArray(1, ctxsize, &impl->data_allocated); CeedChk(ierr);
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
  return 0;
}

//------------------------------------------------------------------------------
// QFunctionContext Get Data
//------------------------------------------------------------------------------
static int CeedQFunctionContextGetData_Ref(CeedQFunctionContext ctx,
    CeedMemType mtype, CeedScalar *data) {
  int ierr;
  CeedQFunctionContext_Ref *impl;
  ierr = CeedQFunctionContextGetBackendData(ctx, (void *)&impl); CeedChk(ierr);
  Ceed ceed;
  ierr = CeedQFunctionContextGetCeed(ctx, &ceed); CeedChk(ierr);

  if (mtype != CEED_MEM_HOST)
    // LCOV_EXCL_START
    return CeedError(ceed, 1, "Can only provide to HOST memory");
  // LCOV_EXCL_STOP
  if (!impl->data)
    // LCOV_EXCL_START
    return CeedError(ceed, 1, "No context data set");
  // LCOV_EXCL_STOP
  *(void **)data = impl->data;
  return 0;
}

//------------------------------------------------------------------------------
// QFunctionContext Restore Data
//------------------------------------------------------------------------------
static int CeedQFunctionContextRestoreData_Ref(CeedQFunctionContext ctx) {
  return 0;
}

//------------------------------------------------------------------------------
// QFunctionContext Destroy
//------------------------------------------------------------------------------
static int CeedQFunctionContextDestroy_Ref(CeedQFunctionContext ctx) {
  int ierr;
  CeedQFunctionContext_Ref *impl;
  ierr = CeedQFunctionContextGetBackendData(ctx, &impl); CeedChk(ierr);

  ierr = CeedFree(&impl->data_allocated); CeedChk(ierr);
  ierr = CeedFree(&impl); CeedChk(ierr);
  return 0;
}

//------------------------------------------------------------------------------
// QFunctionContext Create
//------------------------------------------------------------------------------
int CeedQFunctionContextCreate_Ref(CeedQFunctionContext ctx) {
  int ierr;
  CeedQFunctionContext_Ref *impl;
  Ceed ceed;
  ierr = CeedQFunctionContextGetCeed(ctx, &ceed); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "SetData",
                                CeedQFunctionContextSetData_Ref); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "GetData",
                                CeedQFunctionContextGetData_Ref); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "RestoreData",
                                CeedQFunctionContextRestoreData_Ref); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "Destroy",
                                CeedQFunctionContextDestroy_Ref); CeedChk(ierr);
  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  ierr = CeedQFunctionContextSetBackendData(ctx, impl); CeedChk(ierr);
  return 0;
}
//------------------------------------------------------------------------------
