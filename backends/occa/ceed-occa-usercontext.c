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
#define CEED_DEBUG_COLOR 11
#include "ceed-occa.h"

// *****************************************************************************
// * Bytes used
// *****************************************************************************
static inline size_t bytes(const CeedUserContext ctx) {
  size_t size;
  CeedUserContextGetContextSize(ctx, &size);
  return size;
}

// *****************************************************************************
// * Set the data used by a user context,
// * freeing any previously allocated data if applicable
// *****************************************************************************
static int CeedUserContextSetData_Occa(const CeedUserContext ctx,
                                       const CeedMemType mtype,
                                       const CeedCopyMode cmode,
                                       CeedScalar *data) {
  int ierr;
  Ceed ceed;
  ierr = CeedUserContextGetCeed(ctx, &ceed); CeedChk(ierr);
  CeedUserContext_Occa *impl;
  ierr = CeedUserContextGetBackendData(ctx, (void *)&impl); CeedChk(ierr);
  CeedDebug("[CeedUserContext][Set]");
  if (mtype != CEED_MEM_HOST)
    return CeedError(ceed, 1, "Only MemType = HOST supported");
  switch (cmode) {
  // Implementation will copy the values and not store the passed pointer.
  case CEED_COPY_VALUES:
    CeedDebug("\t[CeedUserContext][Set] CEED_COPY_VALUES");
    if (!impl->h_data) {
      ierr = CeedMalloc(bytes(ctx), &impl->h_data_allocated); CeedChk(ierr);
      impl->h_data = impl->h_data_allocated;
    }
    memcpy(impl->h_data, data, bytes(ctx));
    break;
  // Implementation takes ownership of the pointer
  // and will free using CeedFree() when done using it
  case CEED_OWN_POINTER:
    CeedDebug("\t[CeedUserContext][Set] CEED_OWN_POINTER");
    ierr = CeedFree(&impl->h_data_allocated); CeedChk(ierr);
    impl->h_data = data;
    impl->h_data_allocated = data;
    break;
  // Implementation can use and modify the data provided by the user
  case CEED_USE_POINTER:
    CeedDebug("\t[CeedUserContext][Set] CEED_USE_POINTER");
    ierr = CeedFree(&impl->h_data_allocated); CeedChk(ierr);
    impl->h_data = data;
    break;
  default: CeedError(ceed,1," OCCA backend no default error");
  }

  CeedDebug("\t[CeedUserContext][Set] done");
  return 0;
}

// *****************************************************************************
// * Get access to user context via the specified mtype memory type
// *****************************************************************************
static int CeedUserContextGetData_Occa(const CeedUserContext ctx,
                                       const CeedMemType mtype,
                                       const CeedScalar **data) {
  int ierr;
  Ceed ceed;
  ierr = CeedUserContextGetCeed(ctx, &ceed); CeedChk(ierr);
  CeedDebug("[CeedUserContext][Get]");
  CeedUserContext_Occa *impl;
  ierr = CeedUserContextGetBackendData(ctx, (void *)&impl); CeedChk(ierr);
  if (mtype != CEED_MEM_HOST)
    return CeedError(ceed, 1, "Can only provide to HOST memory");
  if (!impl->h_data)
    return CeedError (ceed, 1, "No context data set");
  *data = impl->h_data;
  return 0;
}

// *****************************************************************************
static int CeedUserContextRestoreData_Occa(const CeedUserContext ctx) {
  int ierr;
  Ceed ceed;
  ierr = CeedUserContextGetCeed(ctx, &ceed); CeedChk(ierr);
  CeedDebug("[CeedUserContext][Restore]");
  CeedUserContext_Occa *impl;
  ierr = CeedUserContextGetBackendData(ctx, (void *)&impl); CeedChk(ierr);
  assert(impl->h_data);
  return 0;
}

// *****************************************************************************
// * Destroy the user context
// *****************************************************************************
static int CeedUserContextDestroy_Occa(const CeedUserContext ctx) {
  int ierr;
  Ceed ceed;
  ierr = CeedUserContextGetCeed(ctx, &ceed); CeedChk(ierr);
  CeedUserContext_Occa *impl;
  ierr = CeedUserContextGetBackendData(ctx, (void *)&impl); CeedChk(ierr);
  CeedDebug("[CeedUserContext][Destroy]");
  ierr = CeedFree(&impl->h_data_allocated); CeedChk(ierr);
  ierr = CeedFree(&impl); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * Create a user context
// *****************************************************************************
int CeedUserContextCreate_Occa(CeedUserContext ctx) {
  int ierr;
  Ceed ceed;
  ierr = CeedUserContextGetCeed(ctx, &ceed); CeedChk(ierr);
  Ceed_Occa *ceed_data;
  ierr = CeedGetData(ceed, (void *)&ceed_data); CeedChk(ierr);
  CeedUserContext_Occa *impl;
  CeedDebug("[CeedUserContext][Create]");
  ierr = CeedSetBackendFunction(ceed, "UserContext", ctx, "SetData",
                                CeedUserContextSetData_Occa); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "UserContext", ctx, "GetData",
                                CeedUserContextGetData_Occa); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "UserContext", ctx, "RestoreData",
                                CeedUserContextRestoreData_Occa); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "UserContext", ctx, "Destroy",
                                CeedUserContextDestroy_Occa); CeedChk(ierr);
  // ***************************************************************************
  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  ierr = CeedUserContextSetBackendData(ctx, (void *)&impl); CeedChk(ierr);
  return 0;
}
