// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <assert.h>
#include <ceed-impl.h>
#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <math.h>
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

/// Indicate that vector will be provided as an explicit argument to
///   CeedOperatorApply().
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
  @brief Check for valid data in a CeedVector

  @param vec                   CeedVector to check validity
  @param[out] has_valid_array  Variable to store validity

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedVectorHasValidArray(CeedVector vec, bool *has_valid_array) {
  if (!vec->HasValidArray) {
    // LCOV_EXCL_START
    return CeedError(vec->ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support HasValidArray");
    // LCOV_EXCL_STOP
  }

  CeedCall(vec->HasValidArray(vec, has_valid_array));

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Check for borrowed array of a specific CeedMemType in a CeedVector

  @param vec                              CeedVector to check
  @param mem_type                         Memory type to check
  @param[out] has_borrowed_array_of_type  Variable to store result

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedVectorHasBorrowedArrayOfType(CeedVector vec, CeedMemType mem_type, bool *has_borrowed_array_of_type) {
  if (!vec->HasBorrowedArrayOfType) {
    // LCOV_EXCL_START
    return CeedError(vec->ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support HasBorrowedArrayOfType");
    // LCOV_EXCL_STOP
  }

  CeedCall(vec->HasBorrowedArrayOfType(vec, mem_type, has_borrowed_array_of_type));

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the state of a CeedVector

  @param vec         CeedVector to retrieve state
  @param[out] state  Variable to store state

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedVectorGetState(CeedVector vec, uint64_t *state) {
  *state = vec->state;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Add a reference to a CeedVector

  @param[out] vec  CeedVector to increment reference counter

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedVectorAddReference(CeedVector vec) {
  vec->ref_count++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the backend data of a CeedVector

  @param vec        CeedVector to retrieve state
  @param[out] data  Variable to store data

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedVectorGetData(CeedVector vec, void *data) {
  *(void **)data = vec->data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the backend data of a CeedVector

  @param[out] vec  CeedVector to retrieve state
  @param data      Data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedVectorSetData(CeedVector vec, void *data) {
  vec->data = data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Increment the reference counter for a CeedVector

  @param vec  CeedVector to increment the reference counter

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
  @brief Create a CeedVector of the specified length (does not allocate memory)

  @param ceed      Ceed object where the CeedVector will be created
  @param length    Length of vector
  @param[out] vec  Address of the variable where the newly created
                     CeedVector will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorCreate(Ceed ceed, CeedSize length, CeedVector *vec) {
  if (!ceed->VectorCreate) {
    Ceed delegate;
    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "Vector"));

    if (!delegate) {
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support VectorCreate");
      // LCOV_EXCL_STOP
    }

    CeedCall(CeedVectorCreate(delegate, length, vec));
    return CEED_ERROR_SUCCESS;
  }

  CeedCall(CeedCalloc(1, vec));
  (*vec)->ceed = ceed;
  CeedCall(CeedReference(ceed));
  (*vec)->ref_count = 1;
  (*vec)->length    = length;
  (*vec)->state     = 0;
  CeedCall(ceed->VectorCreate(length, *vec));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Copy the pointer to a CeedVector. Both pointers should
           be destroyed with `CeedVectorDestroy()`;
           Note: If `*vec_copy` is non-NULL, then it is assumed that
           `*vec_copy` is a pointer to a CeedVector. This
           CeedVector will be destroyed if `*vec_copy` is the only
           reference to this CeedVector.

  @param vec            CeedVector to copy reference to
  @param[out] vec_copy  Variable to store copied reference

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorReferenceCopy(CeedVector vec, CeedVector *vec_copy) {
  CeedCall(CeedVectorReference(vec));
  CeedCall(CeedVectorDestroy(vec_copy));
  *vec_copy = vec;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the array used by a CeedVector, freeing any previously allocated
           array if applicable. The backend may copy values to a different
           memtype, such as during @ref CeedOperatorApply().
           See also @ref CeedVectorSyncArray() and @ref CeedVectorTakeArray().

  @param vec        CeedVector
  @param mem_type   Memory type of the array being passed
  @param copy_mode  Copy mode for the array
  @param array      Array to be used, or NULL with @ref CEED_COPY_VALUES to have the
                      library allocate

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorSetArray(CeedVector vec, CeedMemType mem_type, CeedCopyMode copy_mode, CeedScalar *array) {
  if (!vec->SetArray) {
    // LCOV_EXCL_START
    return CeedError(vec->ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support VectorSetArray");
    // LCOV_EXCL_STOP
  }
  if (vec->state % 2 == 1) {
    return CeedError(vec->ceed, CEED_ERROR_ACCESS, "Cannot grant CeedVector array access, the access lock is already in use");
  }
  if (vec->num_readers > 0) {
    return CeedError(vec->ceed, CEED_ERROR_ACCESS, "Cannot grant CeedVector array access, a process has read access");
  }

  CeedCall(vec->SetArray(vec, mem_type, copy_mode, array));
  vec->state += 2;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the CeedVector to a constant value

  @param vec        CeedVector
  @param[in] value  Value to be used

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorSetValue(CeedVector vec, CeedScalar value) {
  if (vec->state % 2 == 1) {
    // LCOV_EXCL_START
    return CeedError(vec->ceed, CEED_ERROR_ACCESS, "Cannot grant CeedVector array access, the access lock is already in use");
    // LCOV_EXCL_STOP
  }
  if (vec->num_readers > 0) {
    // LCOV_EXCL_START
    return CeedError(vec->ceed, CEED_ERROR_ACCESS, "Cannot grant CeedVector array access, a process has read access");
    // LCOV_EXCL_STOP
  }

  if (vec->SetValue) {
    CeedCall(vec->SetValue(vec, value));
  } else {
    CeedScalar *array;
    CeedCall(CeedVectorGetArrayWrite(vec, CEED_MEM_HOST, &array));
    for (CeedInt i = 0; i < vec->length; i++) array[i] = value;
    CeedCall(CeedVectorRestoreArray(vec, &array));
  }
  vec->state += 2;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Sync the CeedVector to a specified memtype. This function is used to
           force synchronization of arrays set with @ref CeedVectorSetArray().
           If the requested memtype is already synchronized, this function
           results in a no-op.

  @param vec       CeedVector
  @param mem_type  Memtype to be synced

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorSyncArray(CeedVector vec, CeedMemType mem_type) {
  if (vec->state % 2 == 1) {
    return CeedError(vec->ceed, CEED_ERROR_ACCESS, "Cannot sync CeedVector, the access lock is already in use");
  }

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
  @brief Take ownership of the CeedVector array set by @ref CeedVectorSetArray()
           with @ref CEED_USE_POINTER and remove the array from the CeedVector.
           The caller is responsible for managing and freeing the array.
           This function will error if @ref CeedVectorSetArray() was not previously
           called with @ref CEED_USE_POINTER for the corresponding mem_type.

  @param vec         CeedVector
  @param mem_type    Memory type on which to take the array. If the backend
                       uses a different memory type, this will perform a copy.
  @param[out] array  Array on memory type mem_type, or NULL if array pointer is
                       not required

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorTakeArray(CeedVector vec, CeedMemType mem_type, CeedScalar **array) {
  if (vec->state % 2 == 1) {
    // LCOV_EXCL_START
    return CeedError(vec->ceed, CEED_ERROR_ACCESS, "Cannot take CeedVector array, the access lock is already in use");
    // LCOV_EXCL_STOP
  }
  if (vec->num_readers > 0) {
    // LCOV_EXCL_START
    return CeedError(vec->ceed, CEED_ERROR_ACCESS, "Cannot take CeedVector array, a process has read access");
    // LCOV_EXCL_STOP
  }

  CeedScalar *temp_array = NULL;
  if (vec->length > 0) {
    bool has_borrowed_array_of_type = true;
    CeedCall(CeedVectorHasBorrowedArrayOfType(vec, mem_type, &has_borrowed_array_of_type));
    if (!has_borrowed_array_of_type) {
      // LCOV_EXCL_START
      return CeedError(vec->ceed, CEED_ERROR_BACKEND, "CeedVector has no borrowed %s array, must set array with CeedVectorSetArray",
                       CeedMemTypes[mem_type]);
      // LCOV_EXCL_STOP
    }

    bool has_valid_array = true;
    CeedCall(CeedVectorHasValidArray(vec, &has_valid_array));
    if (!has_valid_array) {
      // LCOV_EXCL_START
      return CeedError(vec->ceed, CEED_ERROR_BACKEND,
                       "CeedVector has no valid data to take, must set data with CeedVectorSetValue or CeedVectorSetArray");
      // LCOV_EXCL_STOP
    }

    CeedCall(vec->TakeArray(vec, mem_type, &temp_array));
  }
  if (array) (*array) = temp_array;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get read/write access to a CeedVector via the specified memory type.
           Restore access with @ref CeedVectorRestoreArray().

  @param vec         CeedVector to access
  @param mem_type    Memory type on which to access the array. If the backend
                       uses a different memory type, this will perform a copy.
  @param[out] array  Array on memory type mem_type

  @note The CeedVectorGetArray* and CeedVectorRestoreArray* functions provide
    access to array pointers in the desired memory space. Pairing get/restore
    allows the Vector to track access, thus knowing if norms or other
    operations may need to be recomputed.

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorGetArray(CeedVector vec, CeedMemType mem_type, CeedScalar **array) {
  if (!vec->GetArray) {
    // LCOV_EXCL_START
    return CeedError(vec->ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support GetArray");
    // LCOV_EXCL_STOP
  }
  if (vec->state % 2 == 1) {
    return CeedError(vec->ceed, CEED_ERROR_ACCESS, "Cannot grant CeedVector array access, the access lock is already in use");
  }
  if (vec->num_readers > 0) {
    return CeedError(vec->ceed, CEED_ERROR_ACCESS, "Cannot grant CeedVector array access, a process has read access");
  }

  bool has_valid_array = true;
  CeedCall(CeedVectorHasValidArray(vec, &has_valid_array));
  if (!has_valid_array) {
    // LCOV_EXCL_START
    return CeedError(vec->ceed, CEED_ERROR_BACKEND,
                     "CeedVector has no valid data to read, must set data with CeedVectorSetValue or CeedVectorSetArray");
    // LCOV_EXCL_STOP
  }

  CeedCall(vec->GetArray(vec, mem_type, array));
  vec->state++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get read-only access to a CeedVector via the specified memory type.
           Restore access with @ref CeedVectorRestoreArrayRead().

  @param vec         CeedVector to access
  @param mem_type    Memory type on which to access the array.  If the backend
                       uses a different memory type, this will perform a copy
                       (possibly cached).
  @param[out] array  Array on memory type mem_type

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorGetArrayRead(CeedVector vec, CeedMemType mem_type, const CeedScalar **array) {
  if (!vec->GetArrayRead) {
    // LCOV_EXCL_START
    return CeedError(vec->ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support GetArrayRead");
    // LCOV_EXCL_STOP
  }
  if (vec->state % 2 == 1) {
    return CeedError(vec->ceed, CEED_ERROR_ACCESS, "Cannot grant CeedVector read-only array access, the access lock is already in use");
  }

  if (vec->length > 0) {
    bool has_valid_array = true;
    CeedCall(CeedVectorHasValidArray(vec, &has_valid_array));
    if (!has_valid_array) {
      // LCOV_EXCL_START
      return CeedError(vec->ceed, CEED_ERROR_BACKEND,
                       "CeedVector has no valid data to read, must set data with CeedVectorSetValue or CeedVectorSetArray");
      // LCOV_EXCL_STOP
    }

    CeedCall(vec->GetArrayRead(vec, mem_type, array));
  } else {
    *array = NULL;
  }
  vec->num_readers++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get write access to a CeedVector via the specified memory type.
           Restore access with @ref CeedVectorRestoreArray(). All old
           values should be assumed to be invalid.

  @param vec         CeedVector to access
  @param mem_type    Memory type on which to access the array.
  @param[out] array  Array on memory type mem_type

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorGetArrayWrite(CeedVector vec, CeedMemType mem_type, CeedScalar **array) {
  if (!vec->GetArrayWrite) {
    // LCOV_EXCL_START
    return CeedError(vec->ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support GetArrayWrite");
    // LCOV_EXCL_STOP
  }
  if (vec->state % 2 == 1) {
    // LCOV_EXCL_START
    return CeedError(vec->ceed, CEED_ERROR_ACCESS, "Cannot grant CeedVector array access, the access lock is already in use");
    // LCOV_EXCL_STOP
  }
  if (vec->num_readers > 0) {
    // LCOV_EXCL_START
    return CeedError(vec->ceed, CEED_ERROR_ACCESS, "Cannot grant CeedVector array access, a process has read access");
    // LCOV_EXCL_STOP
  }

  CeedCall(vec->GetArrayWrite(vec, mem_type, array));
  vec->state++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore an array obtained using @ref CeedVectorGetArray()
           or @ref CeedVectorGetArrayWrite()

  @param vec    CeedVector to restore
  @param array  Array of vector data

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorRestoreArray(CeedVector vec, CeedScalar **array) {
  if (vec->state % 2 != 1) {
    return CeedError(vec->ceed, CEED_ERROR_ACCESS, "Cannot restore CeedVector array access, access was not granted");
  }
  if (vec->RestoreArray) CeedCall(vec->RestoreArray(vec));
  *array = NULL;
  vec->state++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore an array obtained using @ref CeedVectorGetArrayRead()

  @param vec    CeedVector to restore
  @param array  Array of vector data

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorRestoreArrayRead(CeedVector vec, const CeedScalar **array) {
  if (vec->num_readers == 0) {
    // LCOV_EXCL_START
    return CeedError(vec->ceed, CEED_ERROR_ACCESS, "Cannot restore CeedVector array read access, access was not granted");
    // LCOV_EXCL_STOP
  }

  vec->num_readers--;
  if (vec->num_readers == 0 && vec->RestoreArrayRead) CeedCall(vec->RestoreArrayRead(vec));
  *array = NULL;

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the norm of a CeedVector.

  Note: This operation is local to the CeedVector. This function will likely
          not provide the desired results for the norm of the libCEED portion
          of a parallel vector or a CeedVector with duplicated or hanging nodes.

  @param vec        CeedVector to retrieve maximum value
  @param norm_type  Norm type @ref CEED_NORM_1, @ref CEED_NORM_2, or @ref CEED_NORM_MAX
  @param[out] norm  Variable to store norm value

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorNorm(CeedVector vec, CeedNormType norm_type, CeedScalar *norm) {
  bool has_valid_array = true;
  CeedCall(CeedVectorHasValidArray(vec, &has_valid_array));
  if (!has_valid_array) {
    // LCOV_EXCL_START
    return CeedError(vec->ceed, CEED_ERROR_BACKEND,
                     "CeedVector has no valid data to compute norm, must set data with CeedVectorSetValue or CeedVectorSetArray");
    // LCOV_EXCL_STOP
  }

  // Backend impl for GPU, if added
  if (vec->Norm) {
    CeedCall(vec->Norm(vec, norm_type, norm));
    return CEED_ERROR_SUCCESS;
  }

  const CeedScalar *array;
  CeedCall(CeedVectorGetArrayRead(vec, CEED_MEM_HOST, &array));

  *norm = 0.;
  switch (norm_type) {
    case CEED_NORM_1:
      for (CeedInt i = 0; i < vec->length; i++) {
        *norm += fabs(array[i]);
      }
      break;
    case CEED_NORM_2:
      for (CeedInt i = 0; i < vec->length; i++) {
        *norm += fabs(array[i]) * fabs(array[i]);
      }
      break;
    case CEED_NORM_MAX:
      for (CeedInt i = 0; i < vec->length; i++) {
        const CeedScalar abs_v_i = fabs(array[i]);
        *norm                    = *norm > abs_v_i ? *norm : abs_v_i;
      }
  }
  if (norm_type == CEED_NORM_2) *norm = sqrt(*norm);

  CeedCall(CeedVectorRestoreArrayRead(vec, &array));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Compute x = alpha x

  @param[in,out] x  vector for scaling
  @param[in] alpha  scaling factor

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorScale(CeedVector x, CeedScalar alpha) {
  CeedScalar *x_array = NULL;
  CeedSize    n_x;

  bool has_valid_array = true;
  CeedCall(CeedVectorHasValidArray(x, &has_valid_array));
  if (!has_valid_array) {
    // LCOV_EXCL_START
    return CeedError(x->ceed, CEED_ERROR_BACKEND,
                     "CeedVector has no valid data to scale, must set data with CeedVectorSetValue or CeedVectorSetArray");
    // LCOV_EXCL_STOP
  }

  CeedCall(CeedVectorGetLength(x, &n_x));

  // Backend implementation
  if (x->Scale) return x->Scale(x, alpha);

  // Default implementation
  CeedCall(CeedVectorGetArrayWrite(x, CEED_MEM_HOST, &x_array));
  for (CeedInt i = 0; i < n_x; i++) x_array[i] *= alpha;
  CeedCall(CeedVectorRestoreArray(x, &x_array));

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Compute y = alpha x + y

  @param[in,out] y  target vector for sum
  @param[in] alpha  scaling factor
  @param[in] x      second vector, must be different than y

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorAXPY(CeedVector y, CeedScalar alpha, CeedVector x) {
  CeedScalar       *y_array = NULL;
  CeedScalar const *x_array = NULL;
  CeedSize          n_x, n_y;

  CeedCall(CeedVectorGetLength(y, &n_y));
  CeedCall(CeedVectorGetLength(x, &n_x));
  if (n_x != n_y) {
    // LCOV_EXCL_START
    return CeedError(y->ceed, CEED_ERROR_UNSUPPORTED, "Cannot add vector of different lengths");
    // LCOV_EXCL_STOP
  }
  if (x == y) {
    // LCOV_EXCL_START
    return CeedError(y->ceed, CEED_ERROR_UNSUPPORTED, "Cannot use same vector for x and y in CeedVectorAXPY");
    // LCOV_EXCL_STOP
  }

  bool has_valid_array_x = true, has_valid_array_y = true;
  CeedCall(CeedVectorHasValidArray(x, &has_valid_array_x));
  if (!has_valid_array_x) {
    // LCOV_EXCL_START
    return CeedError(x->ceed, CEED_ERROR_BACKEND, "CeedVector x has no valid data, must set data with CeedVectorSetValue or CeedVectorSetArray");
    // LCOV_EXCL_STOP
  }
  CeedCall(CeedVectorHasValidArray(y, &has_valid_array_y));
  if (!has_valid_array_y) {
    // LCOV_EXCL_START
    return CeedError(y->ceed, CEED_ERROR_BACKEND, "CeedVector y has no valid data, must set data with CeedVectorSetValue or CeedVectorSetArray");
    // LCOV_EXCL_STOP
  }

  Ceed ceed_parent_x, ceed_parent_y;
  CeedCall(CeedGetParent(x->ceed, &ceed_parent_x));
  CeedCall(CeedGetParent(y->ceed, &ceed_parent_y));
  if (ceed_parent_x != ceed_parent_y) {
    // LCOV_EXCL_START
    return CeedError(y->ceed, CEED_ERROR_INCOMPATIBLE, "Vectors x and y must be created by the same Ceed context");
    // LCOV_EXCL_STOP
  }

  // Backend implementation
  if (y->AXPY) {
    CeedCall(y->AXPY(y, alpha, x));
    return CEED_ERROR_SUCCESS;
  }

  // Default implementation
  CeedCall(CeedVectorGetArrayWrite(y, CEED_MEM_HOST, &y_array));
  CeedCall(CeedVectorGetArrayRead(x, CEED_MEM_HOST, &x_array));

  assert(x_array);
  assert(y_array);

  for (CeedInt i = 0; i < n_y; i++) y_array[i] += alpha * x_array[i];

  CeedCall(CeedVectorRestoreArray(y, &y_array));
  CeedCall(CeedVectorRestoreArrayRead(x, &x_array));

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Compute the pointwise multiplication w = x .* y. Any
           subset of x, y, and w may be the same vector.

  @param[out] w  target vector for the product
  @param[in] x   first vector for product
  @param[in] y   second vector for the product

  @return An error code: 0 - success, otherwise - failure

  @ ref User
**/
int CeedVectorPointwiseMult(CeedVector w, CeedVector x, CeedVector y) {
  CeedScalar       *w_array = NULL;
  CeedScalar const *x_array = NULL, *y_array = NULL;
  CeedSize          n_w, n_x, n_y;

  CeedCall(CeedVectorGetLength(w, &n_w));
  CeedCall(CeedVectorGetLength(x, &n_x));
  CeedCall(CeedVectorGetLength(y, &n_y));
  if (n_w != n_x || n_w != n_y) {
    // LCOV_EXCL_START
    return CeedError(w->ceed, CEED_ERROR_UNSUPPORTED, "Cannot multiply vectors of different lengths");
    // LCOV_EXCL_STOP
  }

  Ceed ceed_parent_w, ceed_parent_x, ceed_parent_y;
  CeedCall(CeedGetParent(w->ceed, &ceed_parent_w));
  CeedCall(CeedGetParent(x->ceed, &ceed_parent_x));
  CeedCall(CeedGetParent(y->ceed, &ceed_parent_y));
  if ((ceed_parent_w != ceed_parent_x) || (ceed_parent_w != ceed_parent_y)) {
    // LCOV_EXCL_START
    return CeedError(w->ceed, CEED_ERROR_INCOMPATIBLE, "Vectors w, x, and y must be created by the same Ceed context");
    // LCOV_EXCL_STOP
  }

  bool has_valid_array_x = true, has_valid_array_y = true;
  CeedCall(CeedVectorHasValidArray(x, &has_valid_array_x));
  if (!has_valid_array_x) {
    // LCOV_EXCL_START
    return CeedError(x->ceed, CEED_ERROR_BACKEND, "CeedVector x has no valid data, must set data with CeedVectorSetValue or CeedVectorSetArray");
    // LCOV_EXCL_STOP
  }
  CeedCall(CeedVectorHasValidArray(y, &has_valid_array_y));
  if (!has_valid_array_y) {
    // LCOV_EXCL_START
    return CeedError(y->ceed, CEED_ERROR_BACKEND, "CeedVector y has no valid data, must set data with CeedVectorSetValue or CeedVectorSetArray");
    // LCOV_EXCL_STOP
  }

  // Backend implementation
  if (w->PointwiseMult) {
    CeedCall(w->PointwiseMult(w, x, y));
    return CEED_ERROR_SUCCESS;
  }

  // Default implementation
  CeedCall(CeedVectorGetArrayWrite(w, CEED_MEM_HOST, &w_array));
  if (x != w) {
    CeedCall(CeedVectorGetArrayRead(x, CEED_MEM_HOST, &x_array));
  } else {
    x_array = w_array;
  }

  if (y != w && y != x) {
    CeedCall(CeedVectorGetArrayRead(y, CEED_MEM_HOST, &y_array));
  } else if (y != x) {
    y_array = w_array;
  } else {
    y_array = x_array;
  }

  assert(w_array);
  assert(x_array);
  assert(y_array);

  for (CeedInt i = 0; i < n_w; i++) w_array[i] = x_array[i] * y_array[i];

  if (y != w && y != x) CeedCall(CeedVectorRestoreArrayRead(y, &y_array));
  if (x != w) CeedCall(CeedVectorRestoreArrayRead(x, &x_array));
  CeedCall(CeedVectorRestoreArray(w, &w_array));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Take the reciprocal of a CeedVector.

  @param vec  CeedVector to take reciprocal

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorReciprocal(CeedVector vec) {
  bool has_valid_array = true;
  CeedCall(CeedVectorHasValidArray(vec, &has_valid_array));

  if (!has_valid_array) {
    // LCOV_EXCL_START
    return CeedError(vec->ceed, CEED_ERROR_BACKEND,
                     "CeedVector has no valid data to compute reciprocal, must set data with CeedVectorSetValue or CeedVectorSetArray");
    // LCOV_EXCL_STOP
  }

  // Check if vector data set
  if (!vec->state) {
    // LCOV_EXCL_START
    return CeedError(vec->ceed, CEED_ERROR_INCOMPLETE, "CeedVector must have data set to take reciprocal");
    // LCOV_EXCL_STOP
  }

  // Backend impl for GPU, if added
  if (vec->Reciprocal) {
    CeedCall(vec->Reciprocal(vec));
    return CEED_ERROR_SUCCESS;
  }

  CeedSize len;
  CeedCall(CeedVectorGetLength(vec, &len));
  CeedScalar *array;
  CeedCall(CeedVectorGetArrayWrite(vec, CEED_MEM_HOST, &array));
  for (CeedInt i = 0; i < len; i++) {
    if (fabs(array[i]) > CEED_EPSILON) array[i] = 1. / array[i];
  }

  CeedCall(CeedVectorRestoreArray(vec, &array));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief View a CeedVector

  @param[in] vec     CeedVector to view
  @param[in] fp_fmt  Printing format
  @param[in] stream  Filestream to write to

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorView(CeedVector vec, const char *fp_fmt, FILE *stream) {
  const CeedScalar *x;
  CeedCall(CeedVectorGetArrayRead(vec, CEED_MEM_HOST, &x));

  char fmt[1024];
  fprintf(stream, "CeedVector length %ld\n", (long)vec->length);
  snprintf(fmt, sizeof fmt, "  %s\n", fp_fmt ? fp_fmt : "%g");
  for (CeedInt i = 0; i < vec->length; i++) fprintf(stream, fmt, x[i]);

  CeedCall(CeedVectorRestoreArrayRead(vec, &x));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the Ceed associated with a CeedVector

  @param vec        CeedVector to retrieve state
  @param[out] ceed  Variable to store ceed

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedVectorGetCeed(CeedVector vec, Ceed *ceed) {
  *ceed = vec->ceed;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the length of a CeedVector

  @param vec          CeedVector to retrieve length
  @param[out] length  Variable to store length

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorGetLength(CeedVector vec, CeedSize *length) {
  *length = vec->length;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Destroy a CeedVector

  @param vec  CeedVector to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedVectorDestroy(CeedVector *vec) {
  if (!*vec || --(*vec)->ref_count > 0) return CEED_ERROR_SUCCESS;

  if (((*vec)->state % 2) == 1) {
    return CeedError((*vec)->ceed, CEED_ERROR_ACCESS, "Cannot destroy CeedVector, the writable access lock is in use");
  }
  if ((*vec)->num_readers > 0) {
    // LCOV_EXCL_START
    return CeedError((*vec)->ceed, CEED_ERROR_ACCESS, "Cannot destroy CeedVector, a process has read access");
    // LCOV_EXCL_STOP
  }

  if ((*vec)->Destroy) CeedCall((*vec)->Destroy(*vec));

  CeedCall(CeedDestroy(&(*vec)->ceed));
  CeedCall(CeedFree(vec));
  return CEED_ERROR_SUCCESS;
}

/// @}
