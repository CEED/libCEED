// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
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

#include <ceed-impl.h>
#include <ceed-backend.h>
#include <limits.h>

/// @file
/// Implementation of public CeedUserContext interfaces

/// ----------------------------------------------------------------------------
/// CeedUserContext Backend API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedQFunctionBackend
/// @{

/**
  @brief Get the Ceed associated with a CeedUserContext

  @param ctx             CeedUserContext
  @param[out] ceed       Variable to store Ceed

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedUserContextGetCeed(CeedUserContext ctx, Ceed *ceed) {
  *ceed = ctx->ceed;
  return 0;
}

/**
  @brief Get the state of a CeedUserContext

  @param ctx           CeedUserContext to retrieve state
  @param[out] state    Variable to store state

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedUserContextGetState(CeedUserContext ctx, uint64_t *state) {
  *state = ctx->state;
  return 0;
}

/**
  @brief Get data size for a Context

  @param ctx             CeedUserContext
  @param[out] ctxsize    Variable to store size of context data values

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedUserContextGetContextSize(CeedUserContext ctx, size_t *ctxsize) {
  *ctxsize = ctx->ctxsize;
  return 0;
}

/**
  @brief Get backend data of a CeedUserContext

  @param ctx             CeedUserContext
  @param[out] data       Variable to store data

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedUserContextGetBackendData(CeedUserContext ctx, void **data) {
  *data = ctx->data;
  return 0;
}

/**
  @brief Set backend data of a CeedUserContext

  @param[out] ctx        CeedUserContext
  @param data            Data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedUserContextSetBackendData(CeedUserContext ctx, void **data) {
  ctx->data = *data;
  return 0;
}

/// @}

/// ----------------------------------------------------------------------------
/// CeedUserContext Public API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedQFunctionUser
/// @{

/**
  @brief Create a CeedUserContext for storing CeedQFunction user context data

  @param ceed       A Ceed object where the CeedUserContext will be created
  @param[out] ctx   Address of the variable where the newly created
                      CeedUserContext will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedUserContextCreate(Ceed ceed, CeedUserContext *ctx) {
  int ierr;

  if (!ceed->UserContextCreate) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "Context"); CeedChk(ierr);

    if (!delegate)
      // LCOV_EXCL_START
      return CeedError(ceed, 1, "Backend does not support ContextCreate");
    // LCOV_EXCL_STOP

    ierr = CeedUserContextCreate(delegate, ctx); CeedChk(ierr);
    return 0;
  }

  ierr = CeedCalloc(1, ctx); CeedChk(ierr);
  (*ctx)->ceed = ceed;
  ceed->refcount++;
  (*ctx)->refcount = 1;
  ierr = ceed->UserContextCreate(*ctx); CeedChk(ierr);
  return 0;
}

/**
  @brief Set the data used by a CeedUserContext, freeing any previously allocated
           data if applicable. The backend may copy values to a different
           memtype, such as during @ref CeedQFunctionApply().
           See also @ref CeedUserContextTakeData().

  @param ctx   CeedUserContext
  @param mtype Memory type of the data being passed
  @param cmode Copy mode for the data
  @param data  Data to be used

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedUserContextSetData(CeedUserContext ctx, CeedMemType mtype,
                           CeedCopyMode cmode,
                           size_t size, void *data) {
  int ierr;

  if (!ctx->SetData)
    // LCOV_EXCL_START
    return CeedError(ctx->ceed, 1, "Backend does not support ContextSetData");
  // LCOV_EXCL_STOP

  if (ctx->state % 2 == 1)
    return CeedError(ctx->ceed, 1, "Cannot grant CeedUserContext data access, the "
                     "access lock is already in use");

  ctx->ctxsize = size;
  ierr = ctx->SetData(ctx, mtype, cmode, data); CeedChk(ierr);
  ctx->state += 2;

  return 0;
}

/**
  @brief Get read/write access to a CeedUserContext via the specified memory type.
           Restore access with @ref CeedUserContextRestoreData().

  @param ctx        CeedUserContext to access
  @param mtype      Memory type on which to access the data. If the backend
                    uses a different memory type, this will perform a copy.
  @param[out] data  Data on memory type mtype

  @note The CeedUserContextGetData() and @ref CeedUserContextRestoreData() functions
    provide access to array pointers in the desired memory space. Pairing
    get/restore allows the Context to track access.

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedUserContextGetData(CeedUserContext ctx, CeedMemType mtype,
                           void **data) {
  int ierr;

  if (!ctx->GetData)
    // LCOV_EXCL_START
    return CeedError(ctx->ceed, 1, "Backend does not support GetData");
  // LCOV_EXCL_STOP

  if (ctx->state % 2 == 1)
    return CeedError(ctx->ceed, 1, "Cannot grant CeedUserContext data access, the "
                     "access lock is already in use");

  ierr = ctx->GetData(ctx, mtype, data); CeedChk(ierr);
  ctx->state += 1;

  return 0;
}

/**
  @brief Restore data obtained using @ref CeedUserContextGetData()

  @param ctx     CeedUserContext to restore
  @param data    Data to restore

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedUserContextRestoreData(CeedUserContext ctx, void **data) {
  int ierr;

  if (!ctx->RestoreData)
    // LCOV_EXCL_START
    return CeedError(ctx->ceed, 1, "Backend does not support RestoreData");
  // LCOV_EXCL_STOP

  if (ctx->state % 2 != 1)
    return CeedError(ctx->ceed, 1, "Cannot restore CeedUserContext array access, "
                     "access was not granted");

  ierr = ctx->RestoreData(ctx); CeedChk(ierr);
  *data = NULL;
  ctx->state += 1;

  return 0;
}

/**
  @brief View a CeedUserContext

  @param[in] ctx           CeedUserContext to view
  @param[in] stream        Filestream to write to

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedUserContextView(CeedUserContext ctx, FILE *stream) {
  fprintf(stream, "CeedUserContext\n");
  fprintf(stream, "  Context Data Size: %ld\n", ctx->ctxsize);
  return 0;
}

/**
  @brief Destroy a CeedUserContext

  @param ctx   CeedUserContext to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedUserContextDestroy(CeedUserContext *ctx) {
  int ierr;

  if (!*ctx || --(*ctx)->refcount > 0)
    return 0;

  if ((*ctx) && ((*ctx)->state % 2) == 1)
    return CeedError((*ctx)->ceed, 1, "Cannot destroy CeedUserContext, the access "
                     "lock is in use");

  if ((*ctx)->Destroy) {
    ierr = (*ctx)->Destroy(*ctx); CeedChk(ierr);
  }

  ierr = CeedDestroy(&(*ctx)->ceed); CeedChk(ierr);
  ierr = CeedFree(ctx); CeedChk(ierr);
  return 0;
}

/// @}
