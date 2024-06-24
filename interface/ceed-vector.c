// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed-impl.h>
#include <ceed.h>
#include <ceed/backend.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

/// @file
/// Implementation of public CeedVector interfaces

/// @cond DOXYGEN_SKIP
static struct CeedVector_private ceed_vector_active;
static struct CeedVector_private ceed_vector_none;
/// @endcond

/// @addtogroup CeedVectorUser
/// @{

/// Indicate that vector will be provided as an explicit argument to @ref CeedOperatorApply().
const CeedVector CEED_VECTOR_ACTIVE = &ceed_vector_active;

/// Indicate that no vector is applicable (i.e., for @ref CEED_EVAL_WEIGHT).
const CeedVector CEED_VECTOR_NONE = &ceed_vector_none;

/// @}

/// ----------------------------------------------------------------------------
/// CeedVector Backend API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedVectorBackend
/// @{

/**
  @brief Check for valid data in a `CeedVector`

  @param[in]  vec             `CeedVector` to check validity
  @param[out] has_valid_array Variable to store validity

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedVectorHasValidArray(CeedVector vec, bool *has_valid_array) {
  CeedSize length;

  CeedCheck(vec->HasValidArray, CeedVectorReturnCeed(vec), CEED_ERROR_UNSUPPORTED, "Backend does not support CeedVectorHasValidArray");
  CeedCall(CeedVectorGetLength(vec, &length));
  if (length == 0) {
    *has_valid_array = true;
    return CEED_ERROR_SUCCESS;
  }
  CeedCall(vec->HasValidArray(vec, has_valid_array));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Check for borrowed array of a specific @ref CeedMemType in a `CeedVector`

  @param[in]  vec                        `CeedVector` to check
  @param[in]  mem_type                   Memory type to check
  @param[out] has_borrowed_array_of_type Variable to store result

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedVectorHasBorrowedArrayOfType(CeedVector vec, CeedMemType mem_type, bool *has_borrowed_array_of_type) {
  CeedCheck(vec->HasBorrowedArrayOfType, CeedVectorReturnCeed(vec), CEED_ERROR_UNSUPPORTED,
            "Backend does not support CeedVectorHasBorrowedArrayOfType");
  CeedCall(vec->HasBorrowedArrayOfType(vec, mem_type, has_borrowed_array_of_type));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the state of a `CeedVector`

  @param[in]  vec    `CeedVector` to retrieve state
  @param[out] state  Variable to store state

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedVectorGetState(CeedVector vec, uint64_t *state) {
  *state = vec->state;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the backend data of a `CeedVector`

  @param[in]  vec  `CeedVector` to retrieve state
  @param[out] data Variable to store data

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedVectorGetData(CeedVector vec, void *data) {
  *(void **)data = vec->data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the backend data of a `CeedVector`

  @param[in,out] vec  `CeedVector` to retrieve state
  @param[in]     data Data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedVectorSetData(CeedVector vec, void *data) {
  vec->data = data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Increment the reference counter for a `CeedVector`

  @param[in,out] vec `CeedVector` to increment the reference counter

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedVectorReference(CeedVector vec) {
  vec->ref_count++;
  return CEED_ERROR_SUCCESS;
}

/// @}

/// ----------------------------------------------------------------------------
/// CeedVector Public API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedVectorUser
/// @{

/**
  @brief Create a `CeedVector` of the specified length (does not allocate memory)

  @param[in]  ceed   `Ceed` object used to create the `CeedVector`
  @param[in]  length Length of vector
  @param[out] vec    Address of the variable where the newly created `CeedVector` will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorCreate(Ceed ceed, CeedSize length, CeedVector *vec) {
  if (!ceed->VectorCreate) {
    Ceed delegate;

    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "Vector"));
    CeedCheck(delegate, ceed, CEED_ERROR_UNSUPPORTED, "Backend does not implement VectorCreate");
    CeedCall(CeedVectorCreate(delegate, length, vec));
    return CEED_ERROR_SUCCESS;
  }

  CeedCall(CeedCalloc(1, vec));
  CeedCall(CeedReferenceCopy(ceed, &(*vec)->ceed));
  (*vec)->ref_count = 1;
  (*vec)->length    = length;
  (*vec)->state     = 0;
  CeedCall(ceed->VectorCreate(length, *vec));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Copy the pointer to a `CeedVector`.

  Both pointers should be destroyed with @ref CeedVectorDestroy().

  Note: If the value of `*vec_copy` passed to this function is non-`NULL`, then it is assumed that `*vec_copy` is a pointer to a `CeedVector`.
        This `CeedVector` will be destroyed if `*vec_copy` is the only reference to this `CeedVector`.

  @param[in]     vec      `CeedVector` to copy reference to
  @param[in,out] vec_copy Variable to store copied reference

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorReferenceCopy(CeedVector vec, CeedVector *vec_copy) {
  if (vec != CEED_VECTOR_ACTIVE && vec != CEED_VECTOR_NONE) CeedCall(CeedVectorReference(vec));
  CeedCall(CeedVectorDestroy(vec_copy));
  *vec_copy = vec;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Copy a `CeedVector` into a different `CeedVector`.

  @param[in]     vec      `CeedVector` to copy
  @param[in,out] vec_copy `CeedVector` to copy array into

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorCopy(CeedVector vec, CeedVector vec_copy) {
  Ceed        ceed;
  CeedMemType mem_type, mem_type_copy;
  CeedScalar *array;

  // Get the preferred memory type
  CeedCall(CeedVectorGetCeed(vec, &ceed));
  CeedCall(CeedGetPreferredMemType(ceed, &mem_type));

  // Get the preferred memory type
  CeedCall(CeedVectorGetCeed(vec_copy, &ceed));
  CeedCall(CeedGetPreferredMemType(ceed, &mem_type_copy));

  // Check that both have same memory type
  if (mem_type != mem_type_copy) mem_type = CEED_MEM_HOST;

  // Check compatible lengths
  {
    CeedSize length_vec, length_copy;

    CeedCall(CeedVectorGetLength(vec, &length_vec));
    CeedCall(CeedVectorGetLength(vec_copy, &length_copy));
    CeedCheck(length_vec == length_copy, ceed, CEED_ERROR_INCOMPATIBLE, "CeedVectors must have the same length to copy");
  }

  // Copy the values from vec to vec_copy
  CeedCall(CeedVectorGetArray(vec, mem_type, &array));
  CeedCall(CeedVectorSetArray(vec_copy, mem_type, CEED_COPY_VALUES, array));

  CeedCall(CeedVectorRestoreArray(vec, &array));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Copy a strided portion of `CeedVector` contents into a different `CeedVector`

  @param[in]     vec      `CeedVector` to copy
  @param[in]     start    First index to copy
  @param[in]     step     Stride between indices to copy
  @param[in,out] vec_copy `CeedVector` to copy values to

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorCopyStrided(CeedVector vec, CeedSize start, CeedInt step, CeedVector vec_copy) {
  CeedSize          length;
  const CeedScalar *array;
  CeedScalar       *array_copy;

  // Backend version
  if (vec->CopyStrided && vec_copy->CopyStrided) {
    CeedCall(vec->CopyStrided(vec, start, step, vec_copy));
    vec_copy->state += 2;
    return CEED_ERROR_SUCCESS;
  }

  // Get length
  {
    CeedSize length_vec, length_copy;

    CeedCall(CeedVectorGetLength(vec, &length_vec));
    CeedCall(CeedVectorGetLength(vec_copy, &length_copy));
    length = length_vec > length_copy ? length_vec : length_copy;
  }

  // Copy
  CeedCall(CeedVectorGetArrayRead(vec, CEED_MEM_HOST, &array));
  CeedCall(CeedVectorGetArray(vec_copy, CEED_MEM_HOST, &array_copy));
  for (CeedSize i = start; i < length; i += step) array_copy[i] = array[i];

  // Cleanup
  CeedCall(CeedVectorRestoreArrayRead(vec, &array));
  CeedCall(CeedVectorRestoreArray(vec_copy, &array_copy));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the array used by a `CeedVector`, freeing any previously allocated array if applicable.

  The backend may copy values to a different @ref CeedMemType, such as during @ref CeedOperatorApply().
  See also @ref CeedVectorSyncArray() and @ref CeedVectorTakeArray().

  @param[in,out] vec       `CeedVector`
  @param[in]     mem_type  Memory type of the array being passed
  @param[in]     copy_mode Copy mode for the array
  @param[in]     array     Array to be used, or `NULL` with @ref CEED_COPY_VALUES to have the library allocate

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorSetArray(CeedVector vec, CeedMemType mem_type, CeedCopyMode copy_mode, CeedScalar *array) {
  CeedSize length;
  Ceed     ceed;

  CeedCall(CeedVectorGetCeed(vec, &ceed));

  CeedCheck(vec->SetArray, ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support VectorSetArray");
  CeedCheck(vec->state % 2 == 0, ceed, CEED_ERROR_ACCESS, "Cannot grant CeedVector array access, the access lock is already in use");
  CeedCheck(vec->num_readers == 0, ceed, CEED_ERROR_ACCESS, "Cannot grant CeedVector array access, a process has read access");

  CeedCall(CeedVectorGetLength(vec, &length));
  if (length > 0) CeedCall(vec->SetArray(vec, mem_type, copy_mode, array));
  vec->state += 2;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the `CeedVector` to a constant value

  @param[in,out] vec   `CeedVector`
  @param[in]     value Value to be used

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorSetValue(CeedVector vec, CeedScalar value) {
  Ceed ceed;

  CeedCall(CeedVectorGetCeed(vec, &ceed));
  CeedCheck(vec->state % 2 == 0, ceed, CEED_ERROR_ACCESS, "Cannot grant CeedVector array access, the access lock is already in use");
  CeedCheck(vec->num_readers == 0, ceed, CEED_ERROR_ACCESS, "Cannot grant CeedVector array access, a process has read access");

  if (vec->SetValue) {
    CeedCall(vec->SetValue(vec, value));
    vec->state += 2;
  } else {
    CeedSize    length;
    CeedScalar *array;

    CeedCall(CeedVectorGetArrayWrite(vec, CEED_MEM_HOST, &array));
    CeedCall(CeedVectorGetLength(vec, &length));
    for (CeedSize i = 0; i < length; i++) array[i] = value;
    CeedCall(CeedVectorRestoreArray(vec, &array));
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set a portion of a `CeedVector` to a constant value.

  Note: The `CeedVector` must already have valid data set via @ref CeedVectorSetArray() or similar.

  @param[in,out] vec   `CeedVector`
  @param[in]     start First index to set
  @param[in]     step  Stride between indices to set
  @param[in]     value Value to be used

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorSetValueStrided(CeedVector vec, CeedSize start, CeedInt step, CeedScalar value) {
  Ceed ceed;

  CeedCall(CeedVectorGetCeed(vec, &ceed));
  CeedCheck(vec->state % 2 == 0, ceed, CEED_ERROR_ACCESS, "Cannot grant CeedVector array access, the access lock is already in use");
  CeedCheck(vec->num_readers == 0, ceed, CEED_ERROR_ACCESS, "Cannot grant CeedVector array access, a process has read access");

  if (vec->SetValueStrided) {
    CeedCall(vec->SetValueStrided(vec, start, step, value));
    vec->state += 2;
  } else {
    CeedSize    length;
    CeedScalar *array;

    CeedCall(CeedVectorGetArray(vec, CEED_MEM_HOST, &array));
    CeedCall(CeedVectorGetLength(vec, &length));
    for (CeedSize i = start; i < length; i += step) array[i] = value;
    CeedCall(CeedVectorRestoreArray(vec, &array));
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Sync the `CeedVector` to a specified `mem_type`.

  This function is used to force synchronization of arrays set with @ref CeedVectorSetArray().
  If the requested `mem_type` is already synchronized, this function results in a no-op.

  @param[in,out] vec      `CeedVector`
  @param[in]     mem_type @ref CeedMemType to be synced

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorSyncArray(CeedVector vec, CeedMemType mem_type) {
  CeedSize length;

  CeedCheck(vec->state % 2 == 0, CeedVectorReturnCeed(vec), CEED_ERROR_ACCESS, "Cannot sync CeedVector, the access lock is already in use");

  // Don't sync empty array
  CeedCall(CeedVectorGetLength(vec, &length));
  if (length == 0) return CEED_ERROR_SUCCESS;

  if (vec->SyncArray) {
    CeedCall(vec->SyncArray(vec, mem_type));
  } else {
    const CeedScalar *array;

    CeedCall(CeedVectorGetArrayRead(vec, mem_type, &array));
    CeedCall(CeedVectorRestoreArrayRead(vec, &array));
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Take ownership of the `CeedVector` array set by @ref CeedVectorSetArray() with @ref CEED_USE_POINTER and remove the array from the `CeedVector`.

  The caller is responsible for managing and freeing the array.
  This function will error if @ref CeedVectorSetArray() was not previously called with @ref CEED_USE_POINTER for the corresponding mem_type.

  @param[in,out] vec      `CeedVector`
  @param[in]     mem_type Memory type on which to take the array.
                            If the backend uses a different memory type, this will perform a copy.
  @param[out]    array    Array on memory type `mem_type`, or `NULL` if array pointer is not required

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorTakeArray(CeedVector vec, CeedMemType mem_type, CeedScalar **array) {
  CeedSize    length;
  CeedScalar *temp_array = NULL;
  Ceed        ceed;

  CeedCall(CeedVectorGetCeed(vec, &ceed));
  CeedCheck(vec->state % 2 == 0, ceed, CEED_ERROR_ACCESS, "Cannot take CeedVector array, the access lock is already in use");
  CeedCheck(vec->num_readers == 0, ceed, CEED_ERROR_ACCESS, "Cannot take CeedVector array, a process has read access");

  CeedCall(CeedVectorGetLength(vec, &length));
  if (length > 0) {
    bool has_borrowed_array_of_type = true, has_valid_array = true;

    CeedCall(CeedVectorHasBorrowedArrayOfType(vec, mem_type, &has_borrowed_array_of_type));
    CeedCheck(has_borrowed_array_of_type, ceed, CEED_ERROR_BACKEND, "CeedVector has no borrowed %s array, must set array with CeedVectorSetArray",
              CeedMemTypes[mem_type]);

    CeedCall(CeedVectorHasValidArray(vec, &has_valid_array));
    CeedCheck(has_valid_array, ceed, CEED_ERROR_BACKEND,
              "CeedVector has no valid data to take, must set data with CeedVectorSetValue or CeedVectorSetArray");

    CeedCall(vec->TakeArray(vec, mem_type, &temp_array));
  }
  if (array) (*array) = temp_array;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get read/write access to a `CeedVector` via the specified memory type.

  Restore access with @ref CeedVectorRestoreArray().

  @param[in,out] vec      `CeedVector` to access
  @param[in]     mem_type Memory type on which to access the array.
                            If the backend uses a different memory type, this will perform a copy.
  @param[out]    array    Array on memory type `mem_type`

  @note The @ref CeedVectorGetArray() and @ref CeedVectorRestoreArray() functions provide access to array pointers in the desired memory space.
        Pairing get/restore allows the `CeedVector` to track access, thus knowing if norms or other operations may need to be recomputed.

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorGetArray(CeedVector vec, CeedMemType mem_type, CeedScalar **array) {
  CeedSize length;
  Ceed     ceed;

  CeedCall(CeedVectorGetCeed(vec, &ceed));
  CeedCheck(vec->GetArray, ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support GetArray");
  CeedCheck(vec->state % 2 == 0, ceed, CEED_ERROR_ACCESS, "Cannot grant CeedVector array access, the access lock is already in use");
  CeedCheck(vec->num_readers == 0, ceed, CEED_ERROR_ACCESS, "Cannot grant CeedVector array access, a process has read access");

  CeedCall(CeedVectorGetLength(vec, &length));
  if (length > 0) {
    bool has_valid_array = true;

    CeedCall(CeedVectorHasValidArray(vec, &has_valid_array));
    CeedCheck(has_valid_array, ceed, CEED_ERROR_BACKEND,
              "CeedVector has no valid data to read, must set data with CeedVectorSetValue or CeedVectorSetArray");

    CeedCall(vec->GetArray(vec, mem_type, array));
  } else {
    *array = NULL;
  }
  vec->state++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get read-only access to a `CeedVector` via the specified memory type.

  Restore access with @ref CeedVectorRestoreArrayRead().

  @param[in]  vec      `CeedVector` to access
  @param[in]  mem_type Memory type on which to access the array.
                         If the backend uses a different memory type, this will perform a copy (possibly cached).
  @param[out] array    Array on memory type `mem_type`

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorGetArrayRead(CeedVector vec, CeedMemType mem_type, const CeedScalar **array) {
  CeedSize length;
  Ceed     ceed;

  CeedCall(CeedVectorGetCeed(vec, &ceed));
  CeedCheck(vec->GetArrayRead, ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support GetArrayRead");
  CeedCheck(vec->state % 2 == 0, ceed, CEED_ERROR_ACCESS, "Cannot grant CeedVector read-only array access, the access lock is already in use");

  CeedCall(CeedVectorGetLength(vec, &length));
  if (length > 0) {
    bool has_valid_array = true;

    CeedCall(CeedVectorHasValidArray(vec, &has_valid_array));
    CeedCheck(has_valid_array, ceed, CEED_ERROR_BACKEND,
              "CeedVector has no valid data to read, must set data with CeedVectorSetValue or CeedVectorSetArray");

    CeedCall(vec->GetArrayRead(vec, mem_type, array));
  } else {
    *array = NULL;
  }
  vec->num_readers++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get write access to a `CeedVector` via the specified memory type.

  Restore access with @ref CeedVectorRestoreArray().
  All old values should be assumed to be invalid.

  @param[in,out] vec      `CeedVector` to access
  @param[in]     mem_type Memory type on which to access the array.
  @param[out]    array    Array on memory type `mem_type`

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorGetArrayWrite(CeedVector vec, CeedMemType mem_type, CeedScalar **array) {
  CeedSize length;
  Ceed     ceed;

  CeedCall(CeedVectorGetCeed(vec, &ceed));
  CeedCheck(vec->GetArrayWrite, ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support CeedVectorGetArrayWrite");
  CeedCheck(vec->state % 2 == 0, ceed, CEED_ERROR_ACCESS, "Cannot grant CeedVector array access, the access lock is already in use");
  CeedCheck(vec->num_readers == 0, ceed, CEED_ERROR_ACCESS, "Cannot grant CeedVector array access, a process has read access");

  CeedCall(CeedVectorGetLength(vec, &length));
  if (length > 0) {
    CeedCall(vec->GetArrayWrite(vec, mem_type, array));
  } else {
    *array = NULL;
  }
  vec->state++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore an array obtained using @ref CeedVectorGetArray() or @ref CeedVectorGetArrayWrite()

  @param[in,out] vec   `CeedVector` to restore
  @param[in,out] array Array of vector data

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorRestoreArray(CeedVector vec, CeedScalar **array) {
  CeedSize length;

  CeedCheck(vec->state % 2 == 1, CeedVectorReturnCeed(vec), CEED_ERROR_ACCESS, "Cannot restore CeedVector array access, access was not granted");
  CeedCall(CeedVectorGetLength(vec, &length));
  if (length > 0 && vec->RestoreArray) CeedCall(vec->RestoreArray(vec));
  *array = NULL;
  vec->state++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore an array obtained using @ref CeedVectorGetArrayRead()

  @param[in]     vec   `CeedVector` to restore
  @param[in,out] array Array of vector data

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorRestoreArrayRead(CeedVector vec, const CeedScalar **array) {
  CeedSize length;

  CeedCheck(vec->num_readers > 0, CeedVectorReturnCeed(vec), CEED_ERROR_ACCESS,
            "Cannot restore CeedVector array read access, access was not granted");
  vec->num_readers--;
  CeedCall(CeedVectorGetLength(vec, &length));
  if (length > 0 && vec->num_readers == 0 && vec->RestoreArrayRead) CeedCall(vec->RestoreArrayRead(vec));
  *array = NULL;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the norm of a `CeedVector`.

  Note: This operation is local to the `CeedVector`.
        This function will likely not provide the desired results for the norm of the libCEED portion of a parallel vector or a `CeedVector` with duplicated or hanging nodes.

  @param[in]  vec       `CeedVector` to retrieve maximum value
  @param[in]  norm_type Norm type @ref CEED_NORM_1, @ref CEED_NORM_2, or @ref CEED_NORM_MAX
  @param[out] norm      Variable to store norm value

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorNorm(CeedVector vec, CeedNormType norm_type, CeedScalar *norm) {
  bool     has_valid_array = true;
  CeedSize length;

  CeedCall(CeedVectorHasValidArray(vec, &has_valid_array));
  CeedCheck(has_valid_array, CeedVectorReturnCeed(vec), CEED_ERROR_BACKEND,
            "CeedVector has no valid data to compute norm, must set data with CeedVectorSetValue or CeedVectorSetArray");

  CeedCall(CeedVectorGetLength(vec, &length));
  if (length == 0) {
    *norm = 0;
    return CEED_ERROR_SUCCESS;
  }

  // Backend impl for GPU, if added
  if (vec->Norm) {
    CeedCall(vec->Norm(vec, norm_type, norm));
    return CEED_ERROR_SUCCESS;
  }

  const CeedScalar *array;
  CeedCall(CeedVectorGetArrayRead(vec, CEED_MEM_HOST, &array));
  assert(array);

  *norm = 0.;
  switch (norm_type) {
    case CEED_NORM_1:
      for (CeedSize i = 0; i < length; i++) {
        *norm += fabs(array[i]);
      }
      break;
    case CEED_NORM_2:
      for (CeedSize i = 0; i < length; i++) {
        *norm += fabs(array[i]) * fabs(array[i]);
      }
      break;
    case CEED_NORM_MAX:
      for (CeedSize i = 0; i < length; i++) {
        const CeedScalar abs_v_i = fabs(array[i]);
        *norm                    = *norm > abs_v_i ? *norm : abs_v_i;
      }
  }
  if (norm_type == CEED_NORM_2) *norm = sqrt(*norm);

  CeedCall(CeedVectorRestoreArrayRead(vec, &array));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Compute `x = alpha x`

  @param[in,out] x     `CeedVector` for scaling
  @param[in]     alpha scaling factor

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorScale(CeedVector x, CeedScalar alpha) {
  bool        has_valid_array = true;
  CeedSize    length;
  CeedScalar *x_array = NULL;

  CeedCall(CeedVectorHasValidArray(x, &has_valid_array));
  CeedCheck(has_valid_array, CeedVectorReturnCeed(x), CEED_ERROR_BACKEND,
            "CeedVector has no valid data to scale, must set data with CeedVectorSetValue or CeedVectorSetArray");

  // Return early for empty vector
  CeedCall(CeedVectorGetLength(x, &length));
  if (length == 0) return CEED_ERROR_SUCCESS;

  // Backend implementation
  if (x->Scale) return x->Scale(x, alpha);

  // Default implementation
  CeedCall(CeedVectorGetArray(x, CEED_MEM_HOST, &x_array));
  assert(x_array);
  for (CeedSize i = 0; i < length; i++) x_array[i] *= alpha;
  CeedCall(CeedVectorRestoreArray(x, &x_array));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Compute `y = alpha x + y`

  @param[in,out] y     target `CeedVector` for sum
  @param[in]     alpha scaling factor
  @param[in]     x     second `CeedVector`, must be different than ``y`

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorAXPY(CeedVector y, CeedScalar alpha, CeedVector x) {
  bool              has_valid_array_x = true, has_valid_array_y = true;
  CeedSize          length_x, length_y;
  CeedScalar       *y_array = NULL;
  CeedScalar const *x_array = NULL;
  Ceed              ceed, ceed_parent_x, ceed_parent_y;

  CeedCall(CeedVectorGetCeed(y, &ceed));
  CeedCall(CeedVectorGetLength(y, &length_y));
  CeedCall(CeedVectorGetLength(x, &length_x));
  CeedCheck(length_x == length_y, ceed, CEED_ERROR_UNSUPPORTED, "Cannot add vector of different lengths");
  CeedCheck(x != y, ceed, CEED_ERROR_UNSUPPORTED, "Cannot use same vector for x and y in CeedVectorAXPY");

  CeedCall(CeedVectorHasValidArray(x, &has_valid_array_x));
  CeedCheck(has_valid_array_x, ceed, CEED_ERROR_BACKEND,
            "CeedVector x has no valid data, must set data with CeedVectorSetValue or CeedVectorSetArray");
  CeedCall(CeedVectorHasValidArray(y, &has_valid_array_y));
  CeedCheck(has_valid_array_y, ceed, CEED_ERROR_BACKEND,
            "CeedVector y has no valid data, must set data with CeedVectorSetValue or CeedVectorSetArray");

  CeedCall(CeedGetParent(x->ceed, &ceed_parent_x));
  CeedCall(CeedGetParent(y->ceed, &ceed_parent_y));
  CeedCheck(ceed_parent_x == ceed_parent_y, ceed, CEED_ERROR_INCOMPATIBLE, "Vectors x and y must be created by the same Ceed context");

  // Return early for empty vectors
  if (length_y == 0) return CEED_ERROR_SUCCESS;

  // Backend implementation
  if (y->AXPY) {
    CeedCall(y->AXPY(y, alpha, x));
    return CEED_ERROR_SUCCESS;
  }

  // Default implementation
  CeedCall(CeedVectorGetArray(y, CEED_MEM_HOST, &y_array));
  CeedCall(CeedVectorGetArrayRead(x, CEED_MEM_HOST, &x_array));

  assert(x_array);
  assert(y_array);

  for (CeedSize i = 0; i < length_y; i++) y_array[i] += alpha * x_array[i];

  CeedCall(CeedVectorRestoreArray(y, &y_array));
  CeedCall(CeedVectorRestoreArrayRead(x, &x_array));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Compute `y = alpha x + beta y`

  @param[in,out] y     target `CeedVector` for sum
  @param[in]     alpha first scaling factor
  @param[in]     beta  second scaling factor
  @param[in]     x     second `CeedVector`, must be different than `y`

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorAXPBY(CeedVector y, CeedScalar alpha, CeedScalar beta, CeedVector x) {
  bool              has_valid_array_x = true, has_valid_array_y = true;
  CeedSize          length_x, length_y;
  CeedScalar       *y_array = NULL;
  CeedScalar const *x_array = NULL;
  Ceed              ceed, ceed_parent_x, ceed_parent_y;

  CeedCall(CeedVectorGetCeed(y, &ceed));

  CeedCall(CeedVectorGetLength(y, &length_y));
  CeedCall(CeedVectorGetLength(x, &length_x));
  CeedCheck(length_x == length_y, ceed, CEED_ERROR_UNSUPPORTED, "Cannot add vector of different lengths");
  CeedCheck(x != y, ceed, CEED_ERROR_UNSUPPORTED, "Cannot use same vector for x and y in CeedVectorAXPBY");

  CeedCall(CeedVectorHasValidArray(x, &has_valid_array_x));
  CeedCheck(has_valid_array_x, ceed, CEED_ERROR_BACKEND,
            "CeedVector x has no valid data, must set data with CeedVectorSetValue or CeedVectorSetArray");
  CeedCall(CeedVectorHasValidArray(y, &has_valid_array_y));
  CeedCheck(has_valid_array_y, ceed, CEED_ERROR_BACKEND,
            "CeedVector y has no valid data, must set data with CeedVectorSetValue or CeedVectorSetArray");

  CeedCall(CeedGetParent(x->ceed, &ceed_parent_x));
  CeedCall(CeedGetParent(y->ceed, &ceed_parent_y));
  CeedCheck(ceed_parent_x == ceed_parent_y, ceed, CEED_ERROR_INCOMPATIBLE, "Vectors x and y must be created by the same Ceed context");

  // Return early for empty vectors
  if (length_y == 0) return CEED_ERROR_SUCCESS;

  // Backend implementation
  if (y->AXPBY) {
    CeedCall(y->AXPBY(y, alpha, beta, x));
    return CEED_ERROR_SUCCESS;
  }

  // Default implementation
  CeedCall(CeedVectorGetArray(y, CEED_MEM_HOST, &y_array));
  CeedCall(CeedVectorGetArrayRead(x, CEED_MEM_HOST, &x_array));

  assert(x_array);
  assert(y_array);

  for (CeedSize i = 0; i < length_y; i++) y_array[i] = alpha * x_array[i] + beta * y_array[i];

  CeedCall(CeedVectorRestoreArray(y, &y_array));
  CeedCall(CeedVectorRestoreArrayRead(x, &x_array));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Compute the pointwise multiplication \f$w = x .* y\f$.

  Any subset of `x`, `y`, and `w` may be the same `CeedVector`.

  @param[out] w target `CeedVector` for the product
  @param[in]  x first `CeedVector` for product
  @param[in]  y second `CeedVector` for the product

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorPointwiseMult(CeedVector w, CeedVector x, CeedVector y) {
  bool              has_valid_array_x = true, has_valid_array_y = true;
  CeedScalar       *w_array = NULL;
  CeedScalar const *x_array = NULL, *y_array = NULL;
  CeedSize          length_w, length_x, length_y;
  Ceed              ceed, ceed_parent_w, ceed_parent_x, ceed_parent_y;

  CeedCall(CeedVectorGetCeed(w, &ceed));
  CeedCall(CeedVectorGetLength(w, &length_w));
  CeedCall(CeedVectorGetLength(x, &length_x));
  CeedCall(CeedVectorGetLength(y, &length_y));
  CeedCheck(length_w == length_x && length_w == length_y, ceed, CEED_ERROR_UNSUPPORTED, "Cannot multiply vectors of different lengths");

  CeedCall(CeedGetParent(w->ceed, &ceed_parent_w));
  CeedCall(CeedGetParent(x->ceed, &ceed_parent_x));
  CeedCall(CeedGetParent(y->ceed, &ceed_parent_y));
  CeedCheck(ceed_parent_w == ceed_parent_x && ceed_parent_w == ceed_parent_y, ceed, CEED_ERROR_INCOMPATIBLE,
            "Vectors w, x, and y must be created by the same Ceed context");

  CeedCall(CeedVectorHasValidArray(x, &has_valid_array_x));
  CeedCheck(has_valid_array_x, ceed, CEED_ERROR_BACKEND,
            "CeedVector x has no valid data, must set data with CeedVectorSetValue or CeedVectorSetArray");
  CeedCall(CeedVectorHasValidArray(y, &has_valid_array_y));
  CeedCheck(has_valid_array_y, ceed, CEED_ERROR_BACKEND,
            "CeedVector y has no valid data, must set data with CeedVectorSetValue or CeedVectorSetArray");

  // Return early for empty vectors
  if (length_w == 0) return CEED_ERROR_SUCCESS;

  // Backend implementation
  if (w->PointwiseMult) {
    CeedCall(w->PointwiseMult(w, x, y));
    return CEED_ERROR_SUCCESS;
  }

  // Default implementation
  if (x == w || y == w) {
    CeedCall(CeedVectorGetArray(w, CEED_MEM_HOST, &w_array));
  } else {
    CeedCall(CeedVectorGetArrayWrite(w, CEED_MEM_HOST, &w_array));
  }
  if (x != w) {
    CeedCall(CeedVectorGetArrayRead(x, CEED_MEM_HOST, &x_array));
  } else {
    x_array = w_array;
  }
  if (y != w && y != x) {
    CeedCall(CeedVectorGetArrayRead(y, CEED_MEM_HOST, &y_array));
  } else if (y == x) {
    y_array = x_array;
  } else if (y == w) {
    y_array = w_array;
  }

  assert(w_array);
  assert(x_array);
  assert(y_array);

  for (CeedSize i = 0; i < length_w; i++) w_array[i] = x_array[i] * y_array[i];

  if (y != w && y != x) CeedCall(CeedVectorRestoreArrayRead(y, &y_array));
  if (x != w) CeedCall(CeedVectorRestoreArrayRead(x, &x_array));
  CeedCall(CeedVectorRestoreArray(w, &w_array));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Take the reciprocal of a `CeedVector`.

  @param[in,out] vec `CeedVector` to take reciprocal

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorReciprocal(CeedVector vec) {
  bool        has_valid_array = true;
  CeedSize    length;
  CeedScalar *array;
  Ceed        ceed;

  CeedCall(CeedVectorGetCeed(vec, &ceed));
  CeedCall(CeedVectorHasValidArray(vec, &has_valid_array));
  CeedCheck(has_valid_array, ceed, CEED_ERROR_BACKEND,
            "CeedVector has no valid data to compute reciprocal, must set data with CeedVectorSetValue or CeedVectorSetArray");

  // Check if vector data set
  CeedCheck(vec->state > 0, ceed, CEED_ERROR_INCOMPLETE, "CeedVector must have data set to take reciprocal");

  // Return early for empty vector
  CeedCall(CeedVectorGetLength(vec, &length));
  if (length == 0) return CEED_ERROR_SUCCESS;

  // Backend impl for GPU, if added
  if (vec->Reciprocal) {
    CeedCall(vec->Reciprocal(vec));
    return CEED_ERROR_SUCCESS;
  }

  CeedCall(CeedVectorGetArray(vec, CEED_MEM_HOST, &array));
  for (CeedSize i = 0; i < length; i++) {
    if (fabs(array[i]) > CEED_EPSILON) array[i] = 1. / array[i];
  }

  CeedCall(CeedVectorRestoreArray(vec, &array));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief View a `CeedVector`

  Note: It is safe to use any unsigned values for `start` or `stop` and any nonzero integer for `step`.
        Any portion of the provided range that is outside the range of valid indices for the `CeedVector` will be ignored.

  @param[in] vec    `CeedVector` to view
  @param[in] start  Index of first `CeedVector` entry to view
  @param[in] stop   Index of last `CeedVector` entry to view
  @param[in] step   Step between `CeedVector` entries to view
  @param[in] fp_fmt Printing format
  @param[in] stream Filestream to write to

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorViewRange(CeedVector vec, CeedSize start, CeedSize stop, CeedInt step, const char *fp_fmt, FILE *stream) {
  char              fmt[1024];
  CeedSize          length;
  const CeedScalar *x;

  CeedCheck(step != 0, CeedVectorReturnCeed(vec), CEED_ERROR_MINOR, "View range 'step' must be nonzero");

  CeedCall(CeedVectorGetLength(vec, &length));
  fprintf(stream, "CeedVector length %" CeedSize_FMT "\n", length);
  if (start != 0 || stop != length || step != 1) {
    fprintf(stream, "  start: %" CeedSize_FMT "\n  stop:  %" CeedSize_FMT "\n  step:  %" CeedInt_FMT "\n", start, stop, step);
  }
  if (start > length) start = length;
  if (stop > length) stop = length;

  snprintf(fmt, sizeof fmt, "  %s\n", fp_fmt ? fp_fmt : "%g");
  CeedCall(CeedVectorGetArrayRead(vec, CEED_MEM_HOST, &x));
  for (CeedSize i = start; step > 0 ? (i < stop) : (i > stop); i += step) fprintf(stream, fmt, x[i]);
  CeedCall(CeedVectorRestoreArrayRead(vec, &x));
  if (stop != length) fprintf(stream, "  ...\n");
  return CEED_ERROR_SUCCESS;
}

/**
  @brief View a `CeedVector`

  @param[in] vec    `CeedVector` to view
  @param[in] fp_fmt Printing format
  @param[in] stream Filestream to write to

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorView(CeedVector vec, const char *fp_fmt, FILE *stream) {
  CeedSize length;

  CeedCall(CeedVectorGetLength(vec, &length));
  CeedCall(CeedVectorViewRange(vec, 0, length, 1, fp_fmt, stream));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the `Ceed` associated with a `CeedVector`

  @param[in]  vec  `CeedVector` to retrieve state
  @param[out] ceed Variable to store `Ceed`

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedVectorGetCeed(CeedVector vec, Ceed *ceed) {
  *ceed = CeedVectorReturnCeed(vec);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Return the `Ceed` associated with a `CeedVector`

  @param[in]  vec  `CeedVector` to retrieve state

  @return `Ceed` associated with the `vec`

  @ref Advanced
**/
Ceed CeedVectorReturnCeed(CeedVector vec) { return vec->ceed; }

/**
  @brief Get the length of a `CeedVector`

  @param[in]  vec    `CeedVector` to retrieve length
  @param[out] length Variable to store length

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorGetLength(CeedVector vec, CeedSize *length) {
  *length = vec->length;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Destroy a `CeedVector`

  @param[in,out] vec `CeedVector` to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorDestroy(CeedVector *vec) {
  if (!*vec || *vec == CEED_VECTOR_ACTIVE || *vec == CEED_VECTOR_NONE || --(*vec)->ref_count > 0) {
    *vec = NULL;
    return CEED_ERROR_SUCCESS;
  }
  CeedCheck((*vec)->state % 2 == 0, (*vec)->ceed, CEED_ERROR_ACCESS, "Cannot destroy CeedVector, the writable access lock is in use");
  CeedCheck((*vec)->num_readers == 0, (*vec)->ceed, CEED_ERROR_ACCESS, "Cannot destroy CeedVector, a process has read access");

  if ((*vec)->Destroy) CeedCall((*vec)->Destroy(*vec));

  CeedCall(CeedDestroy(&(*vec)->ceed));
  CeedCall(CeedFree(vec));
  return CEED_ERROR_SUCCESS;
}

/// @}
