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

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <ceed-impl.h>
#include <stdint.h>
#include <stdio.h>

/// @file
/// Implementation of public CeedQFunctionContext interfaces

/// ----------------------------------------------------------------------------
/// CeedQFunctionContext Backend API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedQFunctionBackend
/// @{

/**
  @brief Get the Ceed associated with a CeedQFunctionContext

  @param ctx        CeedQFunctionContext
  @param[out] ceed  Variable to store Ceed

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextGetCeed(CeedQFunctionContext ctx, Ceed *ceed) {
  *ceed = ctx->ceed;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the state of a CeedQFunctionContext

  @param ctx         CeedQFunctionContext to retrieve state
  @param[out] state  Variable to store state

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextGetState(CeedQFunctionContext ctx, uint64_t *state) {
  *state = ctx->state;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get data size for a Context

  @param ctx            CeedQFunctionContext
  @param[out] ctx_size  Variable to store size of context data values

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextGetContextSize(CeedQFunctionContext ctx,
                                       size_t *ctx_size) {
  *ctx_size = ctx->ctx_size;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get backend data of a CeedQFunctionContext

  @param ctx        CeedQFunctionContext
  @param[out] data  Variable to store data

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextGetBackendData(CeedQFunctionContext ctx, void *data) {
  *(void **)data = ctx->data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set backend data of a CeedQFunctionContext

  @param[out] ctx  CeedQFunctionContext
  @param data      Data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextSetBackendData(CeedQFunctionContext ctx, void *data) {
  ctx->data = data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Increment the reference counter for a CeedQFunctionContext

  @param ctx  CeedQFunctionContext to increment the reference counter

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextReference(CeedQFunctionContext ctx) {
  ctx->ref_count++;
  return CEED_ERROR_SUCCESS;
}

/// @}

/// ----------------------------------------------------------------------------
/// CeedQFunctionContext Public API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedQFunctionUser
/// @{

/**
  @brief Create a CeedQFunctionContext for storing CeedQFunction user context data

  @param ceed      A Ceed object where the CeedQFunctionContext will be created
  @param[out] ctx  Address of the variable where the newly created
                     CeedQFunctionContext will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextCreate(Ceed ceed, CeedQFunctionContext *ctx) {
  int ierr;

  if (!ceed->QFunctionContextCreate) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "Context"); CeedChk(ierr);

    if (!delegate)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                       "Backend does not support ContextCreate");
    // LCOV_EXCL_STOP

    ierr = CeedQFunctionContextCreate(delegate, ctx); CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  }

  ierr = CeedCalloc(1, ctx); CeedChk(ierr);
  (*ctx)->ceed = ceed;
  ierr = CeedReference(ceed); CeedChk(ierr);
  (*ctx)->ref_count = 1;
  ierr = ceed->QFunctionContextCreate(*ctx); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Copy the pointer to a CeedQFunctionContext. Both pointers should
           be destroyed with `CeedQFunctionContextDestroy()`;
           Note: If `*ctx_copy` is non-NULL, then it is assumed that
           `*ctx_copy` is a pointer to a CeedQFunctionContext. This
           CeedQFunctionContext will be destroyed if `*ctx_copy` is the
           only reference to this CeedQFunctionContext.

  @param ctx            CeedQFunctionContext to copy reference to
  @param[out] ctx_copy  Variable to store copied reference

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextReferenceCopy(CeedQFunctionContext ctx,
                                      CeedQFunctionContext *ctx_copy) {
  int ierr;

  ierr = CeedQFunctionContextReference(ctx); CeedChk(ierr);
  ierr = CeedQFunctionContextDestroy(ctx_copy); CeedChk(ierr);
  *ctx_copy = ctx;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the data used by a CeedQFunctionContext, freeing any previously allocated
           data if applicable. The backend may copy values to a different
           memtype, such as during @ref CeedQFunctionApply().
           See also @ref CeedQFunctionContextTakeData().

  @param ctx        CeedQFunctionContext
  @param mem_type   Memory type of the data being passed
  @param copy_mode  Copy mode for the data
  @param size       Size of data, in bytes
  @param data       Data to be used

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextSetData(CeedQFunctionContext ctx, CeedMemType mem_type,
                                CeedCopyMode copy_mode,
                                size_t size, void *data) {
  int ierr;

  if (!ctx->SetData)
    // LCOV_EXCL_START
    return CeedError(ctx->ceed, CEED_ERROR_UNSUPPORTED,
                     "Backend does not support ContextSetData");
  // LCOV_EXCL_STOP

  if (ctx->state % 2 == 1)
    // LCOV_EXCL_START
    return CeedError(ctx->ceed, 1,
                     "Cannot grant CeedQFunctionContext data access, the "
                     "access lock is already in use");
  // LCOV_EXCL_STOP

  ctx->ctx_size = size;
  ierr = ctx->SetData(ctx, mem_type, copy_mode, data); CeedChk(ierr);
  ctx->state += 2;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Take ownership of the data in a CeedQFunctionContext via the specified memory type.
           The caller is responsible for managing and freeing the memory.

  @param ctx        CeedQFunctionContext to access
  @param mem_type   Memory type on which to access the data. If the backend
                      uses a different memory type, this will perform a copy.
  @param[out] data  Data on memory type mem_type

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextTakeData(CeedQFunctionContext ctx, CeedMemType mem_type,
                                 void *data) {
  int ierr;

  if (!ctx->TakeData)
    // LCOV_EXCL_START
    return CeedError(ctx->ceed, CEED_ERROR_UNSUPPORTED,
                     "Backend does not support TakeData");
  // LCOV_EXCL_STOP

  if (ctx->state % 2 == 1)
    // LCOV_EXCL_START
    return CeedError(ctx->ceed, 1,
                     "Cannot grant CeedQFunctionContext data access, the "
                     "access lock is already in use");
  // LCOV_EXCL_STOP

  void *temp_data = NULL;
  ierr = ctx->TakeData(ctx, mem_type, &temp_data); CeedChk(ierr);
  if (data) (*(void **)data) = temp_data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get read/write access to a CeedQFunctionContext via the specified memory type.
           Restore access with @ref CeedQFunctionContextRestoreData().

  @param ctx        CeedQFunctionContext to access
  @param mem_type   Memory type on which to access the data. If the backend
                      uses a different memory type, this will perform a copy.
  @param[out] data  Data on memory type mem_type

  @note The CeedQFunctionContextGetData() and @ref CeedQFunctionContextRestoreData() functions
    provide access to array pointers in the desired memory space. Pairing
    get/restore allows the Context to track access.

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextGetData(CeedQFunctionContext ctx, CeedMemType mem_type,
                                void *data) {
  int ierr;

  if (!ctx->GetData)
    // LCOV_EXCL_START
    return CeedError(ctx->ceed, CEED_ERROR_UNSUPPORTED,
                     "Backend does not support GetData");
  // LCOV_EXCL_STOP

  if (ctx->state % 2 == 1)
    // LCOV_EXCL_START
    return CeedError(ctx->ceed, 1,
                     "Cannot grant CeedQFunctionContext data access, the "
                     "access lock is already in use");
  // LCOV_EXCL_STOP

  ierr = ctx->GetData(ctx, mem_type, data); CeedChk(ierr);
  ctx->state += 1;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore data obtained using @ref CeedQFunctionContextGetData()

  @param ctx   CeedQFunctionContext to restore
  @param data  Data to restore

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextRestoreData(CeedQFunctionContext ctx, void *data) {
  int ierr;

  if (!ctx->RestoreData)
    // LCOV_EXCL_START
    return CeedError(ctx->ceed, CEED_ERROR_UNSUPPORTED,
                     "Backend does not support RestoreData");
  // LCOV_EXCL_STOP

  if (ctx->state % 2 != 1)
    // LCOV_EXCL_START
    return CeedError(ctx->ceed, 1,
                     "Cannot restore CeedQFunctionContext array access, "
                     "access was not granted");
  // LCOV_EXCL_STOP

  ierr = ctx->RestoreData(ctx); CeedChk(ierr);
  *(void **)data = NULL;
  ctx->state += 1;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief View a CeedQFunctionContext

  @param[in] ctx     CeedQFunctionContext to view
  @param[in] stream  Filestream to write to

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextView(CeedQFunctionContext ctx, FILE *stream) {
  fprintf(stream, "CeedQFunctionContext\n");
  fprintf(stream, "  Context Data Size: %ld\n", ctx->ctx_size);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Destroy a CeedQFunctionContext

  @param ctx  CeedQFunctionContext to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextDestroy(CeedQFunctionContext *ctx) {
  int ierr;

  if (!*ctx || --(*ctx)->ref_count > 0)
    return CEED_ERROR_SUCCESS;

  if ((*ctx) && ((*ctx)->state % 2) == 1)
    // LCOV_EXCL_START
    return CeedError((*ctx)->ceed, 1,
                     "Cannot destroy CeedQFunctionContext, the access "
                     "lock is in use");
  // LCOV_EXCL_STOP

  if ((*ctx)->Destroy) {
    ierr = (*ctx)->Destroy(*ctx); CeedChk(ierr);
  }
  ierr = CeedDestroy(&(*ctx)->ceed); CeedChk(ierr);
  ierr = CeedFree(ctx); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/// @}
