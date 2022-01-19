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
#include <string.h>

/// @file
/// Implementation of public CeedQFunctionContext interfaces

/// ----------------------------------------------------------------------------
/// CeedQFunctionContext Library Internal Functions
/// ----------------------------------------------------------------------------
/// @addtogroup CeedQFunctionDeveloper
/// @{

/**
  @brief Get index for QFunctionContext field

  @param ctx         CeedQFunctionContext
  @param field_name  Name of field
  @param field_index Index of field, or -1 if field is not registered

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedQFunctionContextGetFieldIndex(CeedQFunctionContext ctx,
                                      const char *field_name, CeedInt *field_index) {
  *field_index = -1;
  for (CeedInt i=0; i<ctx->num_fields; i++)
    if (!strcmp(ctx->field_descriptions[i].name, field_name))
      *field_index = i;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Common function for registering QFunctionContext fields

  @param ctx               CeedQFunctionContext
  @param field_name        Name of field to register
  @param field_offset      Offset of field to register
  @param field_description Description of field, or NULL for none
  @param field_type        Field data type, such as double or int32
  @param field_size        Size of field, in bytes

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedQFunctionContextRegisterGeneric(CeedQFunctionContext ctx,
                                        const char *field_name, size_t field_offset,
                                        const char *field_description,
                                        CeedContextFieldType field_type,
                                        size_t field_size) {
  int ierr;

  // Check for duplicate
  CeedInt field_index = -1;
  ierr = CeedQFunctionContextGetFieldIndex(ctx, field_name, &field_index);
  CeedChk(ierr);
  if (field_index != -1)
    // LCOV_EXCL_START
    return CeedError(ctx->ceed, CEED_ERROR_UNSUPPORTED,
                     "QFunctionContext field with name \"%s\" already registered",
                     field_name);
  // LCOV_EXCL_STOP

  // Allocate space for field data
  if (ctx->num_fields == 0) {
    ierr = CeedCalloc(1, &ctx->field_descriptions); CeedChk(ierr);
    ctx->max_fields = 1;
  } else if (ctx->num_fields == ctx->max_fields) {
    ierr = CeedRealloc(2*ctx->max_fields, &ctx->field_descriptions);
    CeedChk(ierr);
    ctx->max_fields *= 2;
  }

  // Copy field data
  ierr = CeedStringAllocCopy(field_name,
                             (char **)&ctx->field_descriptions[ctx->num_fields].name);
  CeedChk(ierr);
  ierr = CeedStringAllocCopy(field_description,
                             (char **)&ctx->field_descriptions[ctx->num_fields].description);
  CeedChk(ierr);
  ctx->field_descriptions[ctx->num_fields].type = field_type;
  ctx->field_descriptions[ctx->num_fields].offset = field_offset;
  ctx->field_descriptions[ctx->num_fields].size = field_size;
  ctx->num_fields++;
  return CEED_ERROR_SUCCESS;
}

/// @}

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
  @brief Check for valid data in a CeedQFunctionContext

  @param ctx                  CeedQFunctionContext to check validity
  @param[out] has_valid_data  Variable to store validity

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextHasValidData(CeedQFunctionContext ctx,
                                     bool *has_valid_data) {
  int ierr;

  if (!ctx->HasValidData)
    // LCOV_EXCL_START
    return CeedError(ctx->ceed, CEED_ERROR_UNSUPPORTED,
                     "Backend does not support HasValidData");
  // LCOV_EXCL_STOP

  ierr = ctx->HasValidData(ctx, has_valid_data); CeedChk(ierr);

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Check for borrowed data of a specific CeedMemType in a
           CeedQFunctionContext

  @param ctx                             CeedQFunctionContext to check
  @param mem_type                        Memory type to check
  @param[out] has_borrowed_data_of_type  Variable to store result

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextHasBorrowedDataOfType(CeedQFunctionContext ctx,
    CeedMemType mem_type, bool *has_borrowed_data_of_type) {
  int ierr;

  if (!ctx->HasBorrowedDataOfType)
    // LCOV_EXCL_START
    return CeedError(ctx->ceed, CEED_ERROR_UNSUPPORTED,
                     "Backend does not support HasBorrowedDataOfType");
  // LCOV_EXCL_STOP

  ierr = ctx->HasBorrowedDataOfType(ctx, mem_type, has_borrowed_data_of_type);
  CeedChk(ierr);

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
  @brief Set QFunctionContext field

  @param ctx        CeedQFunctionContext
  @param field_name Name of field to set
  @param field_type Type of field to set
  @param is_set     Boolean flag if value was set
  @param value      Value to set

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextSetGeneric(CeedQFunctionContext ctx,
                                   const char *field_name,
                                   CeedContextFieldType field_type,
                                   bool *is_set, void *value) {
  int ierr;

  // Check field index
  *is_set = false;
  CeedInt field_index = -1;
  ierr = CeedQFunctionContextGetFieldIndex(ctx, field_name, &field_index);
  CeedChk(ierr);
  if (field_index == -1)
    return CEED_ERROR_SUCCESS;

  if (ctx->field_descriptions[field_index].type != field_type)
    // LCOV_EXCL_START
    return CeedError(ctx->ceed, CEED_ERROR_UNSUPPORTED,
                     "QFunctionContext field with name \"%s\" registered as %s, "
                     "not registered as %s", field_name,
                     CeedContextFieldTypes[ctx->field_descriptions[field_index].type],
                     CeedContextFieldTypes[field_type]);
  // LCOV_EXCL_STOP

  char *data;
  ierr = CeedQFunctionContextGetData(ctx, CEED_MEM_HOST, &data); CeedChk(ierr);
  memcpy(&data[ctx->field_descriptions[field_index].offset], value,
         ctx->field_descriptions[field_index].size);
  ierr = CeedQFunctionContextRestoreData(ctx, &data); CeedChk(ierr);
  *is_set = true;

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

  bool has_valid_data = true;
  ierr = CeedQFunctionContextHasValidData(ctx, &has_valid_data); CeedChk(ierr);
  if (!has_valid_data)
    // LCOV_EXCL_START
    return CeedError(ctx->ceed, CEED_ERROR_BACKEND,
                     "CeedQFunctionContext has no valid data to take, must set data");
  // LCOV_EXCL_STOP

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

  bool has_borrowed_data_of_type = true;
  ierr = CeedQFunctionContextHasBorrowedDataOfType(ctx, mem_type,
         &has_borrowed_data_of_type); CeedChk(ierr);
  if (!has_borrowed_data_of_type)
    // LCOV_EXCL_START
    return CeedError(ctx->ceed, CEED_ERROR_BACKEND,
                     "CeedQFunctionContext has no borowed %s data, "
                     "must set data with CeedQFunctionContextSetData",
                     CeedMemTypes[mem_type]);
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

  bool has_valid_data = true;
  ierr = CeedQFunctionContextHasValidData(ctx, &has_valid_data); CeedChk(ierr);
  if (!has_valid_data)
    // LCOV_EXCL_START
    return CeedError(ctx->ceed, CEED_ERROR_BACKEND,
                     "CeedQFunctionContext has no valid data to get, must set data");
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

  if (ctx->state % 2 != 1)
    // LCOV_EXCL_START
    return CeedError(ctx->ceed, 1,
                     "Cannot restore CeedQFunctionContext array access, "
                     "access was not granted");
  // LCOV_EXCL_STOP

  if (ctx->RestoreData) {
    ierr = ctx->RestoreData(ctx); CeedChk(ierr);
  }
  *(void **)data = NULL;
  ctx->state += 1;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Register QFunctionContext a field holding a double precision value

  @param ctx               CeedQFunctionContext
  @param field_name        Name of field to register
  @param field_offset      Offset of field to register
  @param field_description Description of field, or NULL for none

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextRegisterDouble(CeedQFunctionContext ctx,
                                       const char *field_name, size_t field_offset,
                                       const char *field_description) {
  return CeedQFunctionContextRegisterGeneric(ctx, field_name, field_offset,
         field_description, CEED_CONTEXT_FIELD_DOUBLE, sizeof(double));
}

/**
  @brief Register QFunctionContext a field holding a int32 value

  @param ctx               CeedQFunctionContext
  @param field_name        Name of field to register
  @param field_offset      Offset of field to register
  @param field_description Description of field, or NULL for none

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextRegisterInt32(CeedQFunctionContext ctx,
                                      const char *field_name, size_t field_offset,
                                      const char *field_description) {
  return CeedQFunctionContextRegisterGeneric(ctx, field_name, field_offset,
         field_description, CEED_CONTEXT_FIELD_INT32, sizeof(int));
}

/**
  @brief Get descriptions for registered QFunctionContext fields

  @param ctx                     CeedQFunctionContext
  @param[out] field_descriptions Variable to hold array of field descriptions
  @param[out] num_fields         Length of field descriptions array

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextGetFieldDescriptions(CeedQFunctionContext ctx,
    const CeedQFunctionContextFieldDescription **field_descriptions,
    CeedInt *num_fields) {
  *field_descriptions = ctx->field_descriptions;
  *num_fields = ctx->num_fields;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set QFunctionContext field holding a double precision value

  @param ctx        CeedQFunctionContext
  @param field_name Name of field to register
  @param value      Value to set

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextSetDouble(CeedQFunctionContext ctx,
                                  const char *field_name, double value) {
  int ierr;
  bool is_set = false;

  ierr = CeedQFunctionContextSetGeneric(ctx, field_name,
                                        CEED_CONTEXT_FIELD_DOUBLE,
                                        &is_set, &value); CeedChk(ierr);
  if (!is_set)
    // LCOV_EXCL_START
    return CeedError(ctx->ceed, CEED_ERROR_UNSUPPORTED,
                     "QFunctionContext field with name \"%s\" not registered",
                     field_name);
  // LCOV_EXCL_STOP

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set QFunctionContext field holding an int32 value

  @param ctx        CeedQFunctionContext
  @param field_name Name of field to set
  @param value      Value to set

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextSetInt32(CeedQFunctionContext ctx,
                                 const char *field_name, int value) {
  int ierr;
  bool is_set = false;

  ierr = CeedQFunctionContextSetGeneric(ctx, field_name,
                                        CEED_CONTEXT_FIELD_INT32,
                                        &is_set, &value); CeedChk(ierr);
  if (!is_set)
    // LCOV_EXCL_START
    return CeedError(ctx->ceed, CEED_ERROR_UNSUPPORTED,
                     "QFunctionContext field with name \"%s\" not registered",
                     field_name);
  // LCOV_EXCL_STOP

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get data size for a Context

  @param ctx            CeedQFunctionContext
  @param[out] ctx_size  Variable to store size of context data values

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextGetContextSize(CeedQFunctionContext ctx,
                                       size_t *ctx_size) {
  *ctx_size = ctx->ctx_size;
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
  for (CeedInt i=0; i<(*ctx)->num_fields; i++) {
    ierr = CeedFree(&(*ctx)->field_descriptions[i].name); CeedChk(ierr);
    ierr = CeedFree(&(*ctx)->field_descriptions[i].description); CeedChk(ierr);
  }
  ierr = CeedFree(&(*ctx)->field_descriptions); CeedChk(ierr);
  ierr = CeedDestroy(&(*ctx)->ceed); CeedChk(ierr);
  ierr = CeedFree(ctx); CeedChk(ierr);

  return CEED_ERROR_SUCCESS;
}

/// @}
