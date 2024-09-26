// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <valgrind/memcheck.h>

#include "ceed-memcheck.h"

//------------------------------------------------------------------------------
// Has Valid Array
//------------------------------------------------------------------------------
static int CeedVectorHasValidArray_Memcheck(CeedVector vec, bool *has_valid_array) {
  CeedVector_Memcheck *impl;

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  *has_valid_array = !!impl->array_allocated;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Check if has borrowed array of given type
//------------------------------------------------------------------------------
static inline int CeedVectorHasBorrowedArrayOfType_Memcheck(const CeedVector vec, CeedMemType mem_type, bool *has_borrowed_array_of_type) {
  CeedVector_Memcheck *impl;

  CeedCheck(mem_type == CEED_MEM_HOST, CeedVectorReturnCeed(vec), CEED_ERROR_BACKEND, "Can only set HOST memory for this backend");

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  *has_borrowed_array_of_type = !!impl->array_borrowed;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Set Array
//------------------------------------------------------------------------------
static int CeedVectorSetArray_Memcheck(CeedVector vec, CeedMemType mem_type, CeedCopyMode copy_mode, CeedScalar *array) {
  CeedSize             length;
  CeedVector_Memcheck *impl;

  CeedCheck(mem_type == CEED_MEM_HOST, CeedVectorReturnCeed(vec), CEED_ERROR_BACKEND, "Can only set HOST memory for this backend");

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedVectorGetLength(vec, &length));

  // Clear previous owned arrays
  if (impl->array_allocated) {
    for (CeedSize i = 0; i < length; i++) impl->array_allocated[i] = NAN;
    VALGRIND_DISCARD(impl->allocated_block_id);
  }
  CeedCallBackend(CeedFree(&impl->array_allocated));
  if (impl->array_owned) {
    for (CeedSize i = 0; i < length; i++) impl->array_owned[i] = NAN;
    VALGRIND_DISCARD(impl->owned_block_id);
  }
  CeedCallBackend(CeedFree(&impl->array_owned));

  // Clear borrowed block id, if present
  if (impl->array_borrowed) VALGRIND_DISCARD(impl->borrowed_block_id);

  // Set internal pointers to external arrays
  switch (copy_mode) {
    case CEED_COPY_VALUES:
      impl->array_owned    = NULL;
      impl->array_borrowed = NULL;
      break;
    case CEED_OWN_POINTER:
      impl->array_owned    = array;
      impl->array_borrowed = NULL;
      impl->owned_block_id = VALGRIND_CREATE_BLOCK(impl->array_owned, length * sizeof(CeedScalar), "Owned external array buffer");
      break;
    case CEED_USE_POINTER:
      impl->array_owned       = NULL;
      impl->array_borrowed    = array;
      impl->borrowed_block_id = VALGRIND_CREATE_BLOCK(impl->array_borrowed, length * sizeof(CeedScalar), "Borrowed external array buffer");
      break;
  }

  // Create internal array data buffer
  CeedCallBackend(CeedCalloc(length, &impl->array_allocated));
  impl->allocated_block_id = VALGRIND_CREATE_BLOCK(impl->array_allocated, length * sizeof(CeedScalar), "Allocated internal array buffer");
  if (array) {
    memcpy(impl->array_allocated, array, length * sizeof(CeedScalar));
  } else {
    for (CeedInt i = 0; i < length; i++) impl->array_allocated[i] = NAN;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set internal array to value
//------------------------------------------------------------------------------
static int CeedVectorSetValue_Memcheck(CeedVector vec, CeedScalar value) {
  CeedSize             length;
  CeedVector_Memcheck *impl;

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedVectorGetLength(vec, &length));

  if (!impl->array_allocated) CeedCallBackend(CeedVectorSetArray_Memcheck(vec, CEED_MEM_HOST, CEED_COPY_VALUES, NULL));
  assert(impl->array_allocated);
  for (CeedSize i = 0; i < length; i++) impl->array_allocated[i] = value;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set internal array to value strided
//------------------------------------------------------------------------------
static int CeedVectorSetValueStrided_Memcheck(CeedVector vec, CeedSize start, CeedSize step, CeedScalar val) {
  CeedSize             length;
  CeedVector_Memcheck *impl;

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedVectorGetLength(vec, &length));

  if (!impl->array_allocated) CeedCallBackend(CeedVectorSetArray_Memcheck(vec, CEED_MEM_HOST, CEED_COPY_VALUES, NULL));
  assert(impl->array_allocated);
  for (CeedSize i = start; i < length; i += step) impl->array_allocated[i] = val;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Sync arrays
//------------------------------------------------------------------------------
static int CeedVectorSyncArray_Memcheck(const CeedVector vec, CeedMemType mem_type) {
  CeedSize             length;
  CeedVector_Memcheck *impl;

  CeedCheck(mem_type == CEED_MEM_HOST, CeedVectorReturnCeed(vec), CEED_ERROR_BACKEND, "Can only provide HOST memory for this backend");

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedVectorGetLength(vec, &length));

  // Copy internal buffer back to owned or borrowed array
  if (impl->array_owned) {
    memcpy(impl->array_owned, impl->array_allocated, length * sizeof(CeedScalar));
  }
  if (impl->array_borrowed) {
    memcpy(impl->array_borrowed, impl->array_allocated, length * sizeof(CeedScalar));
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Take Array
//------------------------------------------------------------------------------
static int CeedVectorTakeArray_Memcheck(CeedVector vec, CeedMemType mem_type, CeedScalar **array) {
  CeedSize             length;
  CeedVector_Memcheck *impl;

  CeedCheck(mem_type == CEED_MEM_HOST, CeedVectorReturnCeed(vec), CEED_ERROR_BACKEND, "Can only provide HOST memory for this backend");

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedVectorGetLength(vec, &length));

  // Synchronize memory
  CeedCallBackend(CeedVectorSyncArray_Memcheck(vec, CEED_MEM_HOST));

  // Return borrowed array
  (*array)             = impl->array_borrowed;
  impl->array_borrowed = NULL;
  VALGRIND_DISCARD(impl->borrowed_block_id);

  // De-allocate internal memory
  if (impl->array_allocated) {
    for (CeedSize i = 0; i < length; i++) impl->array_allocated[i] = NAN;
    VALGRIND_DISCARD(impl->allocated_block_id);
  }
  CeedCallBackend(CeedFree(&impl->array_allocated));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Get Array
//------------------------------------------------------------------------------
static int CeedVectorGetArray_Memcheck(CeedVector vec, CeedMemType mem_type, CeedScalar **array) {
  CeedSize             length;
  CeedVector_Memcheck *impl;

  CeedCheck(mem_type == CEED_MEM_HOST, CeedVectorReturnCeed(vec), CEED_ERROR_BACKEND, "Can only provide HOST memory for this backend");

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedVectorGetLength(vec, &length));

  // Create and return writable buffer
  CeedCallBackend(CeedCalloc(length, &impl->array_writable_copy));
  impl->writable_block_id = VALGRIND_CREATE_BLOCK(impl->array_writable_copy, length * sizeof(CeedScalar), "Allocated writeable array buffer copy");
  memcpy(impl->array_writable_copy, impl->array_allocated, length * sizeof(CeedScalar));
  *array = impl->array_writable_copy;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Get Array Read
//------------------------------------------------------------------------------
static int CeedVectorGetArrayRead_Memcheck(CeedVector vec, CeedMemType mem_type, const CeedScalar **array) {
  CeedSize             length;
  CeedVector_Memcheck *impl;

  CeedCheck(mem_type == CEED_MEM_HOST, CeedVectorReturnCeed(vec), CEED_ERROR_BACKEND, "Can only provide HOST memory for this backend");

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedVectorGetLength(vec, &length));

  // Create and return read-only buffer
  if (!impl->array_read_only_copy) {
    CeedCallBackend(CeedCalloc(length, &impl->array_read_only_copy));
    impl->writable_block_id = VALGRIND_CREATE_BLOCK(impl->array_read_only_copy, length * sizeof(CeedScalar), "Allocated read-only array buffer copy");
    memcpy(impl->array_read_only_copy, impl->array_allocated, length * sizeof(CeedScalar));
  }
  *array = impl->array_read_only_copy;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Get Array Write
//------------------------------------------------------------------------------
static int CeedVectorGetArrayWrite_Memcheck(CeedVector vec, CeedMemType mem_type, CeedScalar **array) {
  CeedSize             length;
  CeedVector_Memcheck *impl;

  CeedCheck(mem_type == CEED_MEM_HOST, CeedVectorReturnCeed(vec), CEED_ERROR_BACKEND, "Can only provide HOST memory for this backend");

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedVectorGetLength(vec, &length));

  // Allocate buffer if necessary
  if (!impl->array_allocated) CeedCallBackend(CeedVectorSetArray_Memcheck(vec, mem_type, CEED_COPY_VALUES, NULL));

  // Get writable buffer
  CeedCallBackend(CeedVectorGetArray_Memcheck(vec, mem_type, array));

  // Invalidate array data to prevent accidental reads
  for (CeedSize i = 0; i < length; i++) (*array)[i] = NAN;
  impl->is_write_only_access = true;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Restore Array
//------------------------------------------------------------------------------
static int CeedVectorRestoreArray_Memcheck(CeedVector vec) {
  Ceed                 ceed;
  CeedSize             length;
  CeedVector_Memcheck *impl;

  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedVectorGetLength(vec, &length));

  // Check for unset entries after write-only access
  if (impl->is_write_only_access) {
    for (CeedSize i = 0; i < length; i++) {
      if (isnan(impl->array_writable_copy[i])) {
        CeedDebug256(ceed, CEED_DEBUG_COLOR_WARNING, "WARNING: Vec entry %" CeedSize_FMT " is NaN after restoring write-only access", i);
      }
    }
    impl->is_write_only_access = false;
  }

  // Copy back to internal buffer and sync
  memcpy(impl->array_allocated, impl->array_writable_copy, length * sizeof(CeedScalar));
  CeedCallBackend(CeedVectorSyncArray_Memcheck(vec, CEED_MEM_HOST));

  // Invalidate writable buffer
  for (CeedSize i = 0; i < length; i++) impl->array_writable_copy[i] = NAN;
  CeedCallBackend(CeedFree(&impl->array_writable_copy));
  VALGRIND_DISCARD(impl->writable_block_id);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Restore Array Read-Only
//------------------------------------------------------------------------------
static int CeedVectorRestoreArrayRead_Memcheck(CeedVector vec) {
  Ceed                 ceed;
  CeedSize             length;
  CeedVector_Memcheck *impl;

  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedVectorGetLength(vec, &length));

  // Verify no changes made during read-only access
  bool is_changed = memcmp(impl->array_allocated, impl->array_read_only_copy, length * sizeof(CeedScalar));

  CeedCheck(!is_changed, ceed, CEED_ERROR_BACKEND, "Array data changed while accessed in read-only mode");

  // Invalidate read-only buffer
  for (CeedSize i = 0; i < length; i++) impl->array_read_only_copy[i] = NAN;
  CeedCallBackend(CeedFree(&impl->array_read_only_copy));
  VALGRIND_DISCARD(impl->read_only_block_id);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Take reciprocal of a vector
//------------------------------------------------------------------------------
static int CeedVectorReciprocal_Memcheck(CeedVector vec) {
  CeedSize             length;
  CeedVector_Memcheck *impl;

  CeedCallBackend(CeedVectorGetData(vec, &impl));
  CeedCallBackend(CeedVectorGetLength(vec, &length));

  for (CeedSize i = 0; i < length; i++) {
    if (fabs(impl->array_allocated[i]) > CEED_EPSILON) impl->array_allocated[i] = 1. / impl->array_allocated[i];
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute x = alpha x
//------------------------------------------------------------------------------
static int CeedVectorScale_Memcheck(CeedVector x, CeedScalar alpha) {
  CeedSize             length;
  CeedVector_Memcheck *impl;

  CeedCallBackend(CeedVectorGetData(x, &impl));
  CeedCallBackend(CeedVectorGetLength(x, &length));

  for (CeedSize i = 0; i < length; i++) impl->array_allocated[i] *= alpha;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute y = alpha x + y
//------------------------------------------------------------------------------
static int CeedVectorAXPY_Memcheck(CeedVector y, CeedScalar alpha, CeedVector x) {
  CeedSize             length;
  CeedVector_Memcheck *impl_x, *impl_y;

  CeedCallBackend(CeedVectorGetData(x, &impl_x));
  CeedCallBackend(CeedVectorGetData(y, &impl_y));
  CeedCallBackend(CeedVectorGetLength(y, &length));

  for (CeedSize i = 0; i < length; i++) impl_y->array_allocated[i] += alpha * impl_x->array_allocated[i];
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute y = alpha x + beta y
//------------------------------------------------------------------------------
static int CeedVectorAXPBY_Memcheck(CeedVector y, CeedScalar alpha, CeedScalar beta, CeedVector x) {
  CeedSize             length;
  CeedVector_Memcheck *impl_x, *impl_y;

  CeedCallBackend(CeedVectorGetData(x, &impl_x));
  CeedCallBackend(CeedVectorGetData(y, &impl_y));
  CeedCallBackend(CeedVectorGetLength(y, &length));

  for (CeedSize i = 0; i < length; i++) impl_y->array_allocated[i] = alpha * impl_x->array_allocated[i] + beta * impl_y->array_allocated[i];
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute the pointwise multiplication w = x .* y
//------------------------------------------------------------------------------
static int CeedVectorPointwiseMult_Memcheck(CeedVector w, CeedVector x, CeedVector y) {
  CeedSize             length;
  CeedVector_Memcheck *impl_x, *impl_y, *impl_w;

  CeedCallBackend(CeedVectorGetData(x, &impl_x));
  CeedCallBackend(CeedVectorGetData(y, &impl_y));
  CeedCallBackend(CeedVectorGetData(w, &impl_w));
  CeedCallBackend(CeedVectorGetLength(w, &length));

  if (!impl_w->array_allocated) CeedCallBackend(CeedVectorSetArray_Memcheck(w, CEED_MEM_HOST, CEED_COPY_VALUES, NULL));
  assert(impl_w->array_allocated);
  for (CeedSize i = 0; i < length; i++) impl_w->array_allocated[i] = impl_x->array_allocated[i] * impl_y->array_allocated[i];
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Destroy
//------------------------------------------------------------------------------
static int CeedVectorDestroy_Memcheck(CeedVector vec) {
  CeedVector_Memcheck *impl;

  // Free allocations and discard block ids
  CeedCallBackend(CeedVectorGetData(vec, &impl));
  if (impl->array_allocated) {
    CeedCallBackend(CeedFree(&impl->array_allocated));
    VALGRIND_DISCARD(impl->allocated_block_id);
  }
  if (impl->array_owned) {
    CeedCallBackend(CeedFree(&impl->array_owned));
    VALGRIND_DISCARD(impl->owned_block_id);
  }
  if (impl->array_borrowed) {
    VALGRIND_DISCARD(impl->borrowed_block_id);
  }
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Vector Create
//------------------------------------------------------------------------------
int CeedVectorCreate_Memcheck(CeedSize n, CeedVector vec) {
  Ceed                 ceed;
  CeedVector_Memcheck *impl;

  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedVectorSetData(vec, impl));

  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "HasValidArray", CeedVectorHasValidArray_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "HasBorrowedArrayOfType", CeedVectorHasBorrowedArrayOfType_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "SetArray", CeedVectorSetArray_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "SetValue", CeedVectorSetValue_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "SetValueStrided", CeedVectorSetValueStrided_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "SyncArray", CeedVectorSyncArray_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "TakeArray", CeedVectorTakeArray_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "GetArray", CeedVectorGetArray_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayRead", CeedVectorGetArrayRead_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayWrite", CeedVectorGetArrayWrite_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "RestoreArray", CeedVectorRestoreArray_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "RestoreArrayRead", CeedVectorRestoreArrayRead_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "Reciprocal", CeedVectorReciprocal_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "Scale", CeedVectorScale_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "AXPY", CeedVectorAXPY_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "AXPBY", CeedVectorAXPBY_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "PointwiseMult", CeedVectorPointwiseMult_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "Destroy", CeedVectorDestroy_Memcheck));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
