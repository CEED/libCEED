// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed-impl.h>
#include <ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
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
  @brief Get index for `CeedQFunctionContext` field

  @param[in]  ctx         `CeedQFunctionContext`
  @param[in]  field_name  Name of field
  @param[out] field_index Index of field, or `-1` if field is not registered

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedQFunctionContextGetFieldIndex(CeedQFunctionContext ctx, const char *field_name, CeedInt *field_index) {
  *field_index = -1;
  for (CeedInt i = 0; i < ctx->num_fields; i++) {
    if (!strcmp(ctx->field_labels[i]->name, field_name)) *field_index = i;
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Common function for registering `CeedQFunctionContext` fields

  @param[in,out] ctx               `CeedQFunctionContext`
  @param[in]     field_name        Name of field to register
  @param[in]     field_offset      Offset of field to register
  @param[in]     field_description Description of field, or `NULL` for none
  @param[in]     field_type        @ref CeedContextFieldType
  @param[in]     num_values        Number of values to register, must be contiguous in memory

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedQFunctionContextRegisterGeneric(CeedQFunctionContext ctx, const char *field_name, size_t field_offset, const char *field_description,
                                        CeedContextFieldType field_type, size_t num_values) {
  size_t  field_size  = 0;
  CeedInt field_index = -1;

  // Check for duplicate
  CeedCall(CeedQFunctionContextGetFieldIndex(ctx, field_name, &field_index));
  CeedCheck(field_index == -1, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_UNSUPPORTED,
            "QFunctionContext field with name \"%s\" already registered", field_name);

  // Allocate space for field data
  if (ctx->num_fields == 0) {
    CeedCall(CeedCalloc(1, &ctx->field_labels));
    ctx->max_fields = 1;
  } else if (ctx->num_fields == ctx->max_fields) {
    CeedCall(CeedRealloc(2 * ctx->max_fields, &ctx->field_labels));
    ctx->max_fields *= 2;
  }
  CeedCall(CeedCalloc(1, &ctx->field_labels[ctx->num_fields]));

  // Compute field size
  switch (field_type) {
    case CEED_CONTEXT_FIELD_DOUBLE:
      field_size = sizeof(double);
      break;
    case CEED_CONTEXT_FIELD_INT32:
      field_size = sizeof(int);
      break;
    case CEED_CONTEXT_FIELD_BOOL:
      field_size = sizeof(bool);
      break;
  }

  // Copy field data
  CeedCall(CeedStringAllocCopy(field_name, (char **)&ctx->field_labels[ctx->num_fields]->name));
  CeedCall(CeedStringAllocCopy(field_description, (char **)&ctx->field_labels[ctx->num_fields]->description));
  ctx->field_labels[ctx->num_fields]->type       = field_type;
  ctx->field_labels[ctx->num_fields]->offset     = field_offset;
  ctx->field_labels[ctx->num_fields]->size       = field_size * num_values;
  ctx->field_labels[ctx->num_fields]->num_values = num_values;
  ctx->num_fields++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Destroy user data held by `CeedQFunctionContext`, using function set by @ref CeedQFunctionContextSetDataDestroy(), if applicable

  @param[in,out] ctx `CeedQFunctionContext` to destroy user data

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedQFunctionContextDestroyData(CeedQFunctionContext ctx) {
  if (ctx->DataDestroy) {
    CeedCall(ctx->DataDestroy(ctx));
  } else {
    CeedMemType                         data_destroy_mem_type;
    CeedQFunctionContextDataDestroyUser data_destroy_function;

    CeedCall(CeedQFunctionContextGetDataDestroy(ctx, &data_destroy_mem_type, &data_destroy_function));
    if (data_destroy_function) {
      void *data;

      CeedCall(CeedQFunctionContextGetData(ctx, data_destroy_mem_type, &data));
      CeedCall(data_destroy_function(data));
      CeedCall(CeedQFunctionContextRestoreData(ctx, &data));
    }
  }
  return CEED_ERROR_SUCCESS;
}

/// @}

/// ----------------------------------------------------------------------------
/// CeedQFunctionContext Backend API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedQFunctionBackend
/// @{

/**
  @brief Get the `Ceed` associated with a `CeedQFunctionContext`

  @param[in]  ctx  `CeedQFunctionContext`
  @param[out] ceed Variable to store `Ceed`

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextGetCeed(CeedQFunctionContext ctx, Ceed *ceed) {
  *ceed = NULL;
  CeedCall(CeedReferenceCopy(CeedQFunctionContextReturnCeed(ctx), ceed));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Return the `Ceed` associated with a `CeedQFunctionContext`

  @param[in]  ctx  `CeedQFunctionContext`

  @return `Ceed` associated with `ctx`

  @ref Backend
**/
Ceed CeedQFunctionContextReturnCeed(CeedQFunctionContext ctx) { return ctx->ceed; }

/**
  @brief Check for valid data in a `CeedQFunctionContext`

  @param[in]  ctx            `CeedQFunctionContext` to check validity
  @param[out] has_valid_data Variable to store validity

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextHasValidData(CeedQFunctionContext ctx, bool *has_valid_data) {
  CeedCheck(ctx->HasValidData, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_UNSUPPORTED,
            "Backend does not support CeedQFunctionContextHasValidData");
  CeedCall(ctx->HasValidData(ctx, has_valid_data));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Check for borrowed data of a specific @ref CeedMemType in a `CeedQFunctionContext`

  @param[in]  ctx                       `CeedQFunctionContext` to check
  @param[in]  mem_type                  Memory type to check
  @param[out] has_borrowed_data_of_type Variable to store result

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextHasBorrowedDataOfType(CeedQFunctionContext ctx, CeedMemType mem_type, bool *has_borrowed_data_of_type) {
  CeedCheck(ctx->HasBorrowedDataOfType, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_UNSUPPORTED,
            "Backend does not support CeedQFunctionContextHasBorrowedDataOfType");
  CeedCall(ctx->HasBorrowedDataOfType(ctx, mem_type, has_borrowed_data_of_type));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the state of a `CeedQFunctionContext`

  @param[in]  ctx   `CeedQFunctionContext` to retrieve state
  @param[out] state Variable to store state

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextGetState(CeedQFunctionContext ctx, uint64_t *state) {
  *state = ctx->state;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get backend data of a `CeedQFunctionContext`

  @param[in]  ctx  `CeedQFunctionContext`
  @param[out] data Variable to store data

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextGetBackendData(CeedQFunctionContext ctx, void *data) {
  *(void **)data = ctx->data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set backend data of a `CeedQFunctionContext`

  @param[in,out] ctx  `CeedQFunctionContext`
  @param[in]     data Data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextSetBackendData(CeedQFunctionContext ctx, void *data) {
  ctx->data = data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get label for a registered `CeedQFunctionContext` field, or `NULL` if no field has been registered with this `field_name`

  @param[in]  ctx         `CeedQFunctionContext`
  @param[in]  field_name  Name of field to retrieve label
  @param[out] field_label Variable to field label

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextGetFieldLabel(CeedQFunctionContext ctx, const char *field_name, CeedContextFieldLabel *field_label) {
  CeedInt field_index;

  CeedCall(CeedQFunctionContextGetFieldIndex(ctx, field_name, &field_index));

  if (field_index != -1) {
    *field_label = ctx->field_labels[field_index];
  } else {
    *field_label = NULL;
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set `CeedQFunctionContext` field

  @param[in,out] ctx         `CeedQFunctionContext`
  @param[in]     field_label Label of field to set
  @param[in]     field_type  Type of field to set
  @param[in]     values      Value to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextSetGeneric(CeedQFunctionContext ctx, CeedContextFieldLabel field_label, CeedContextFieldType field_type, void *values) {
  bool  is_different;
  char *data;

  // Check field type
  CeedCheck(field_label->type == field_type, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_UNSUPPORTED,
            "QFunctionContext field with name \"%s\" registered as %s, not registered as %s", field_label->name,
            CeedContextFieldTypes[field_label->type], CeedContextFieldTypes[field_type]);

  CeedCall(CeedQFunctionContextGetDataRead(ctx, CEED_MEM_HOST, &data));
  is_different = memcmp(&data[field_label->offset], values, field_label->size);
  CeedCall(CeedQFunctionContextRestoreDataRead(ctx, &data));
  if (is_different) {
    CeedCall(CeedQFunctionContextGetData(ctx, CEED_MEM_HOST, &data));
    memcpy(&data[field_label->offset], values, field_label->size);
    CeedCall(CeedQFunctionContextRestoreData(ctx, &data));
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get `CeedQFunctionContext` field data, read-only

  @param[in]  ctx         `CeedQFunctionContext`
  @param[in]  field_label Label of field to read
  @param[in]  field_type  Type of field to read
  @param[out] num_values  Number of values in the field label
  @param[out] values      Pointer to context values

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextGetGenericRead(CeedQFunctionContext ctx, CeedContextFieldLabel field_label, CeedContextFieldType field_type,
                                       size_t *num_values, void *values) {
  char *data;

  // Check field type
  CeedCheck(field_label->type == field_type, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_UNSUPPORTED,
            "QFunctionContext field with name \"%s\" registered as %s, not registered as %s", field_label->name,
            CeedContextFieldTypes[field_label->type], CeedContextFieldTypes[field_type]);

  CeedCall(CeedQFunctionContextGetDataRead(ctx, CEED_MEM_HOST, &data));
  *(void **)values = &data[field_label->offset];
  switch (field_type) {
    case CEED_CONTEXT_FIELD_INT32:
      *num_values = field_label->size / sizeof(int);
      break;
    case CEED_CONTEXT_FIELD_DOUBLE:
      *num_values = field_label->size / sizeof(double);
      break;
    case CEED_CONTEXT_FIELD_BOOL:
      *num_values = field_label->size / sizeof(bool);
      break;
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore `CeedQFunctionContext` field data, read-only

  @param[in]  ctx         `CeedQFunctionContext`
  @param[in]  field_label Label of field to restore
  @param[in]  field_type  Type of field to restore
  @param[out] values      Pointer to context values

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextRestoreGenericRead(CeedQFunctionContext ctx, CeedContextFieldLabel field_label, CeedContextFieldType field_type,
                                           void *values) {
  // Check field type
  CeedCheck(field_label->type == field_type, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_UNSUPPORTED,
            "QFunctionContext field with name \"%s\" registered as %s, not registered as %s", field_label->name,
            CeedContextFieldTypes[field_label->type], CeedContextFieldTypes[field_type]);

  CeedCall(CeedQFunctionContextRestoreDataRead(ctx, values));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set `CeedQFunctionContext` field holding double precision values

  @param[in,out] ctx         `CeedQFunctionContext`
  @param[in]     field_label Label for field to set
  @param[in]     values      Values to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextSetDouble(CeedQFunctionContext ctx, CeedContextFieldLabel field_label, double *values) {
  CeedCheck(field_label, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_UNSUPPORTED, "Invalid field label");
  CeedCall(CeedQFunctionContextSetGeneric(ctx, field_label, CEED_CONTEXT_FIELD_DOUBLE, values));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get `CeedQFunctionContext` field holding double precision values, read-only

  @param[in]  ctx         `CeedQFunctionContext`
  @param[in]  field_label Label for field to get
  @param[out] num_values  Number of values in the field label
  @param[out] values      Pointer to context values

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextGetDoubleRead(CeedQFunctionContext ctx, CeedContextFieldLabel field_label, size_t *num_values, const double **values) {
  CeedCheck(field_label, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_UNSUPPORTED, "Invalid field label");
  CeedCall(CeedQFunctionContextGetGenericRead(ctx, field_label, CEED_CONTEXT_FIELD_DOUBLE, num_values, values));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore `CeedQFunctionContext` field holding double precision values, read-only

  @param[in]  ctx         `CeedQFunctionContext`
  @param[in]  field_label Label for field to restore
  @param[out] values      Pointer to context values

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextRestoreDoubleRead(CeedQFunctionContext ctx, CeedContextFieldLabel field_label, const double **values) {
  CeedCheck(field_label, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_UNSUPPORTED, "Invalid field label");
  CeedCall(CeedQFunctionContextRestoreGenericRead(ctx, field_label, CEED_CONTEXT_FIELD_DOUBLE, values));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set CeedQFunctionContext field holding `int32` values

  @param[in,out] ctx         CeedQFunctionContext
  @param[in]     field_label Label for field to set
  @param[in]     values      Values to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextSetInt32(CeedQFunctionContext ctx, CeedContextFieldLabel field_label, int32_t *values) {
  CeedCheck(field_label, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_UNSUPPORTED, "Invalid field label");
  CeedCall(CeedQFunctionContextSetGeneric(ctx, field_label, CEED_CONTEXT_FIELD_INT32, values));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get `CeedQFunctionContext` field holding `int32` values, read-only

  @param[in]  ctx         `CeedQFunctionContext`
  @param[in]  field_label Label for field to get
  @param[out] num_values  Number of values in the field label
  @param[out] values      Pointer to context values

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextGetInt32Read(CeedQFunctionContext ctx, CeedContextFieldLabel field_label, size_t *num_values, const int32_t **values) {
  CeedCheck(field_label, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_UNSUPPORTED, "Invalid field label");
  CeedCall(CeedQFunctionContextGetGenericRead(ctx, field_label, CEED_CONTEXT_FIELD_INT32, num_values, values));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore `CeedQFunctionContext` field holding `int32` values, read-only

  @param[in]  ctx         `CeedQFunctionContext`
  @param[in]  field_label Label for field to restore
  @param[out] values      Pointer to context values

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextRestoreInt32Read(CeedQFunctionContext ctx, CeedContextFieldLabel field_label, const int32_t **values) {
  CeedCheck(field_label, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_UNSUPPORTED, "Invalid field label");
  CeedCall(CeedQFunctionContextRestoreGenericRead(ctx, field_label, CEED_CONTEXT_FIELD_INT32, values));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set `CeedQFunctionContext` field holding boolean values

  @param[in,out] ctx         `CeedQFunctionContext`
  @param[in]     field_label Label for field to set
  @param[in]     values      Values to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextSetBoolean(CeedQFunctionContext ctx, CeedContextFieldLabel field_label, bool *values) {
  CeedCheck(field_label, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_UNSUPPORTED, "Invalid field label");
  CeedCall(CeedQFunctionContextSetGeneric(ctx, field_label, CEED_CONTEXT_FIELD_BOOL, values));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get `CeedQFunctionContext` field holding boolean values, read-only

  @param[in]  ctx         `CeedQFunctionContext`
  @param[in]  field_label Label for field to get
  @param[out] num_values  Number of values in the field label
  @param[out] values      Pointer to context values

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextGetBooleanRead(CeedQFunctionContext ctx, CeedContextFieldLabel field_label, size_t *num_values, const bool **values) {
  CeedCheck(field_label, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_UNSUPPORTED, "Invalid field label");
  CeedCall(CeedQFunctionContextGetGenericRead(ctx, field_label, CEED_CONTEXT_FIELD_BOOL, num_values, values));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore `CeedQFunctionContext` field holding boolean values, read-only

  @param[in]  ctx         `CeedQFunctionContext`
  @param[in]  field_label Label for field to restore
  @param[out] values      Pointer to context values

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextRestoreBooleanRead(CeedQFunctionContext ctx, CeedContextFieldLabel field_label, const bool **values) {
  CeedCheck(field_label, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_UNSUPPORTED, "Invalid field label");
  CeedCall(CeedQFunctionContextRestoreGenericRead(ctx, field_label, CEED_CONTEXT_FIELD_BOOL, values));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get additional destroy routine for `CeedQFunctionContext` user data

  @param[in] ctx         `CeedQFunctionContext` to get user destroy function
  @param[out] f_mem_type Memory type to use when passing data into `f`
  @param[out] f          Additional routine to use to destroy user data

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionContextGetDataDestroy(CeedQFunctionContext ctx, CeedMemType *f_mem_type, CeedQFunctionContextDataDestroyUser *f) {
  if (f_mem_type) *f_mem_type = ctx->data_destroy_mem_type;
  if (f) *f = ctx->data_destroy_function;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Increment the reference counter for a `CeedQFunctionContext`

  @param[in,out] ctx `CeedQFunctionContext` to increment the reference counter

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
  @brief Create a `CeedQFunctionContext` for storing `CeedQFunctionContext` user context data

  @param[in]  ceed `Ceed` object used to create the `CeedQFunctionContext`
  @param[out] ctx  Address of the variable where the newly created `CeedQFunctionContext` will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextCreate(Ceed ceed, CeedQFunctionContext *ctx) {
  if (!ceed->QFunctionContextCreate) {
    Ceed delegate;

    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "Context"));
    CeedCheck(delegate, ceed, CEED_ERROR_UNSUPPORTED, "Backend does not implement CeedQFunctionContextCreate");
    CeedCall(CeedQFunctionContextCreate(delegate, ctx));
    CeedCall(CeedDestroy(&delegate));
    return CEED_ERROR_SUCCESS;
  }

  CeedCall(CeedCalloc(1, ctx));
  CeedCall(CeedReferenceCopy(ceed, &(*ctx)->ceed));
  (*ctx)->ref_count = 1;
  CeedCall(ceed->QFunctionContextCreate(*ctx));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Copy the pointer to a `CeedQFunctionContext`.

  Both pointers should be destroyed with @ref CeedQFunctionContextDestroy().

  Note: If the value of `*ctx_copy` passed to this function is non-`NULL`, then it is assumed that `*ctx_copy` is a pointer to a `CeedQFunctionContext`.
        This `CeedQFunctionContext` will be destroyed if `*ctx_copy` is the only reference to this `CeedQFunctionContext`.

  @param[in]     ctx      CeedQFunctionContext to copy reference to
  @param[in,out] ctx_copy Variable to store copied reference

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextReferenceCopy(CeedQFunctionContext ctx, CeedQFunctionContext *ctx_copy) {
  CeedCall(CeedQFunctionContextReference(ctx));
  CeedCall(CeedQFunctionContextDestroy(ctx_copy));
  *ctx_copy = ctx;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the data used by a `CeedQFunctionContext`, freeing any previously allocated data if applicable.

  The backend may copy values to a different @ref CeedMemType, such as during @ref CeedQFunctionApply().
  See also @ref CeedQFunctionContextTakeData().

  @param[in,out] ctx       `CeedQFunctionContext`
  @param[in]     mem_type  Memory type of the data being passed
  @param[in]     copy_mode Copy mode for the data
  @param[in]     size      Size of data, in bytes
  @param[in]     data      Data to be used

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextSetData(CeedQFunctionContext ctx, CeedMemType mem_type, CeedCopyMode copy_mode, size_t size, void *data) {
  CeedCheck(ctx->SetData, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_UNSUPPORTED, "Backend does not support CeedQFunctionContextSetData");
  CeedCheck(ctx->state % 2 == 0, CeedQFunctionContextReturnCeed(ctx), 1,
            "Cannot grant CeedQFunctionContext data access, the access lock is already in use");

  CeedCall(CeedQFunctionContextDestroyData(ctx));
  ctx->ctx_size = size;
  CeedCall(ctx->SetData(ctx, mem_type, copy_mode, data));
  ctx->state += 2;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Take ownership of the data in a `CeedQFunctionContext` via the specified memory type.

  The caller is responsible for managing and freeing the memory.

  @param[in]  ctx      `CeedQFunctionContext` to access
  @param[in]  mem_type Memory type on which to access the data.
                         If the backend uses a different memory type, this will perform a copy.
  @param[out] data     Data on memory type mem_type

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextTakeData(CeedQFunctionContext ctx, CeedMemType mem_type, void *data) {
  void *temp_data      = NULL;
  bool  has_valid_data = true, has_borrowed_data_of_type = true;

  CeedCall(CeedQFunctionContextHasValidData(ctx, &has_valid_data));
  CeedCheck(has_valid_data, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_BACKEND, "CeedQFunctionContext has no valid data to take, must set data");

  CeedCheck(ctx->TakeData, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_UNSUPPORTED, "Backend does not support CeedQFunctionContextTakeData");
  CeedCheck(ctx->state % 2 == 0, CeedQFunctionContextReturnCeed(ctx), 1,
            "Cannot grant CeedQFunctionContext data access, the access lock is already in use");

  CeedCall(CeedQFunctionContextHasBorrowedDataOfType(ctx, mem_type, &has_borrowed_data_of_type));
  CeedCheck(has_borrowed_data_of_type, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_BACKEND,
            "CeedQFunctionContext has no borrowed %s data, must set data with CeedQFunctionContextSetData", CeedMemTypes[mem_type]);

  CeedCall(ctx->TakeData(ctx, mem_type, &temp_data));
  if (data) (*(void **)data) = temp_data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get read/write access to a `CeedQFunctionContext` via the specified memory type.

  Restore access with @ref CeedQFunctionContextRestoreData().

  @param[in]  ctx      `CeedQFunctionContext` to access
  @param[in]  mem_type Memory type on which to access the data.
                         If the backend uses a different memory type, this will perform a copy.
  @param[out] data     Data on memory type mem_type

  @note The @ref CeedQFunctionContextGetData() and @ref CeedQFunctionContextRestoreData() functions provide access to array pointers in the desired memory space.
        Pairing get/restore allows the `CeedQFunctionContext` to track access.

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextGetData(CeedQFunctionContext ctx, CeedMemType mem_type, void *data) {
  bool has_valid_data = true;

  CeedCheck(ctx->GetData, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_UNSUPPORTED, "Backend does not support CeedQFunctionContextGetData");
  CeedCheck(ctx->state % 2 == 0, CeedQFunctionContextReturnCeed(ctx), 1,
            "Cannot grant CeedQFunctionContext data access, the access lock is already in use");
  CeedCheck(ctx->num_readers == 0, CeedQFunctionContextReturnCeed(ctx), 1,
            "Cannot grant CeedQFunctionContext data access, a process has read access");

  CeedCall(CeedQFunctionContextHasValidData(ctx, &has_valid_data));
  CeedCheck(has_valid_data, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_BACKEND, "CeedQFunctionContext has no valid data to get, must set data");

  CeedCall(ctx->GetData(ctx, mem_type, data));
  ctx->state++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get read only access to a `CeedQFunctionContext` via the specified memory type.

  Restore access with @ref CeedQFunctionContextRestoreData().

  @param[in]  ctx      `CeedQFunctionContext` to access
  @param[in]  mem_type Memory type on which to access the data.
                         If the backend uses a different memory type, this will perform a copy.
  @param[out] data     Data on memory type mem_type

  @note The @ref CeedQFunctionContextGetDataRead() and @ref CeedQFunctionContextRestoreDataRead() functions provide access to array pointers in the desired memory space.
        Pairing get/restore allows the `CeedQFunctionContext` to track access.

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextGetDataRead(CeedQFunctionContext ctx, CeedMemType mem_type, void *data) {
  bool has_valid_data = true;

  CeedCheck(ctx->GetDataRead, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_UNSUPPORTED,
            "Backend does not support CeedQFunctionContextGetDataRead");
  CeedCheck(ctx->state % 2 == 0, CeedQFunctionContextReturnCeed(ctx), 1,
            "Cannot grant CeedQFunctionContext data access, the access lock is already in use");

  CeedCall(CeedQFunctionContextHasValidData(ctx, &has_valid_data));
  CeedCheck(has_valid_data, CeedQFunctionContextReturnCeed(ctx), CEED_ERROR_BACKEND, "CeedQFunctionContext has no valid data to get, must set data");

  CeedCall(ctx->GetDataRead(ctx, mem_type, data));
  ctx->num_readers++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore data obtained using @ref CeedQFunctionContextGetData()

  @param[in]     ctx  `CeedQFunctionContext` to restore
  @param[in,out] data Data to restore

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextRestoreData(CeedQFunctionContext ctx, void *data) {
  CeedCheck(ctx->state % 2 == 1, CeedQFunctionContextReturnCeed(ctx), 1, "Cannot restore CeedQFunctionContext array access, access was not granted");

  if (ctx->RestoreData) CeedCall(ctx->RestoreData(ctx));
  *(void **)data = NULL;
  ctx->state++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore data obtained using @ref CeedQFunctionContextGetDataRead()

  @param[in]     ctx  `CeedQFunctionContext` to restore
  @param[in,out] data Data to restore

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextRestoreDataRead(CeedQFunctionContext ctx, void *data) {
  CeedCheck(ctx->num_readers > 0, CeedQFunctionContextReturnCeed(ctx), 1, "Cannot restore CeedQFunctionContext array access, access was not granted");

  ctx->num_readers--;
  if (ctx->num_readers == 0 && ctx->RestoreDataRead) CeedCall(ctx->RestoreDataRead(ctx));
  *(void **)data = NULL;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Register a `CeedQFunctionContext` field holding double precision values

  @param[in,out] ctx               `CeedQFunctionContext`
  @param[in]     field_name        Name of field to register
  @param[in]     field_offset      Offset of field to register
  @param[in]     num_values        Number of values to register, must be contiguous in memory
  @param[in]     field_description Description of field, or `NULL` for none

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextRegisterDouble(CeedQFunctionContext ctx, const char *field_name, size_t field_offset, size_t num_values,
                                       const char *field_description) {
  return CeedQFunctionContextRegisterGeneric(ctx, field_name, field_offset, field_description, CEED_CONTEXT_FIELD_DOUBLE, num_values);
}

/**
  @brief Register a `CeedQFunctionContext` field holding `int32` values

  @param[in,out] ctx               CeedQFunctionContext
  @param[in]     field_name        Name of field to register
  @param[in]     field_offset      Offset of field to register
  @param[in]     num_values        Number of values to register, must be contiguous in memory
  @param[in]     field_description Description of field, or `NULL` for none

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextRegisterInt32(CeedQFunctionContext ctx, const char *field_name, size_t field_offset, size_t num_values,
                                      const char *field_description) {
  return CeedQFunctionContextRegisterGeneric(ctx, field_name, field_offset, field_description, CEED_CONTEXT_FIELD_INT32, num_values);
}

/**
  @brief Register a `CeedQFunctionContext` field holding boolean values

  @param[in,out] ctx               `CeedQFunctionContext`
  @param[in]     field_name        Name of field to register
  @param[in]     field_offset      Offset of field to register
  @param[in]     num_values        Number of values to register, must be contiguous in memory
  @param[in]     field_description Description of field, or `NULL` for none

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextRegisterBoolean(CeedQFunctionContext ctx, const char *field_name, size_t field_offset, size_t num_values,
                                        const char *field_description) {
  return CeedQFunctionContextRegisterGeneric(ctx, field_name, field_offset, field_description, CEED_CONTEXT_FIELD_BOOL, num_values);
}

/**
  @brief Get labels for all registered `CeedQFunctionContext` fields

  @param[in]  ctx          `CeedQFunctionContext`
  @param[out] field_labels Variable to hold array of field labels
  @param[out] num_fields   Length of field descriptions array

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextGetAllFieldLabels(CeedQFunctionContext ctx, const CeedContextFieldLabel **field_labels, CeedInt *num_fields) {
  *field_labels = ctx->field_labels;
  *num_fields   = ctx->num_fields;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the descriptive information about a `CeedContextFieldLabel`

  @param[in]  label             @ref CeedContextFieldLabel
  @param[out] field_name        Name of labeled field
  @param[out] field_offset      Offset of field registered
  @param[out] num_values        Number of values registered
  @param[out] field_description Description of field, or NULL for none
  @param[out] field_type        @ref CeedContextFieldType

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedContextFieldLabelGetDescription(CeedContextFieldLabel label, const char **field_name, size_t *field_offset, size_t *num_values,
                                        const char **field_description, CeedContextFieldType *field_type) {
  if (field_name) *field_name = label->name;
  if (field_offset) *field_offset = label->offset;
  if (num_values) *num_values = label->num_values;
  if (field_description) *field_description = label->description;
  if (field_type) *field_type = label->type;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get data size for a Context

  @param[in]  ctx      `CeedQFunctionContext`
  @param[out] ctx_size Variable to store size of context data values

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextGetContextSize(CeedQFunctionContext ctx, size_t *ctx_size) {
  *ctx_size = ctx->ctx_size;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief View a `CeedQFunctionContext`

  @param[in] ctx    `CeedQFunctionContext` to view
  @param[in] stream Filestream to write to

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextView(CeedQFunctionContext ctx, FILE *stream) {
  fprintf(stream, "CeedQFunctionContext\n");
  fprintf(stream, "  Context Data Size: %zu\n", ctx->ctx_size);
  for (CeedInt i = 0; i < ctx->num_fields; i++) {
    fprintf(stream, "  Labeled %s field: %s\n", CeedContextFieldTypes[ctx->field_labels[i]->type], ctx->field_labels[i]->name);
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set additional destroy routine for `CeedQFunctionContext` user data

  @param[in,out] ctx        `CeedQFunctionContext` to set user destroy function
  @param[in]     f_mem_type Memory type to use when passing data into `f`
  @param[in]     f          Additional routine to use to destroy user data

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextSetDataDestroy(CeedQFunctionContext ctx, CeedMemType f_mem_type, CeedQFunctionContextDataDestroyUser f) {
  CeedCheck(f, CeedQFunctionContextReturnCeed(ctx), 1, "Must provide valid callback function for destroying user data");
  ctx->data_destroy_mem_type = f_mem_type;
  ctx->data_destroy_function = f;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Destroy a `CeedQFunctionContext`

  @param[in,out] ctx `CeedQFunctionContext` to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionContextDestroy(CeedQFunctionContext *ctx) {
  if (!*ctx || --(*ctx)->ref_count > 0) {
    *ctx = NULL;
    return CEED_ERROR_SUCCESS;
  }
  CeedCheck(((*ctx)->state % 2) == 0, (*ctx)->ceed, 1, "Cannot destroy CeedQFunctionContext, the access lock is in use");

  CeedCall(CeedQFunctionContextDestroyData(*ctx));
  if ((*ctx)->Destroy) CeedCall((*ctx)->Destroy(*ctx));
  for (CeedInt i = 0; i < (*ctx)->num_fields; i++) {
    CeedCall(CeedFree(&(*ctx)->field_labels[i]->name));
    CeedCall(CeedFree(&(*ctx)->field_labels[i]->description));
    CeedCall(CeedFree(&(*ctx)->field_labels[i]));
  }
  CeedCall(CeedFree(&(*ctx)->field_labels));
  CeedCall(CeedDestroy(&(*ctx)->ceed));
  CeedCall(CeedFree(ctx));
  return CEED_ERROR_SUCCESS;
}

/// @}
