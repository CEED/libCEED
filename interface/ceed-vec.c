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

/// @cond DOXYGEN_SKIP
static struct CeedVector_private ceed_vector_active;
static struct CeedVector_private ceed_vector_none;
/// @endcond

/// @file
/// Implementation of public CeedVector interfaces
///
/// @addtogroup CeedVector
/// @{

/**
  @brief Create a CeedVector of the specified length (does not allocate memory)

  @param ceed      Ceed object where the CeedVector will be created
  @param length    Length of vector
  @param[out] vec  Address of the variable where the newly created
                     CeedVector will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedVectorCreate(Ceed ceed, CeedInt length, CeedVector *vec) {
  int ierr;

  if (!ceed->VectorCreate) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "Vector"); CeedChk(ierr);

    if (!delegate)
      return CeedError(ceed, 1, "Backend does not support VectorCreate");

    ierr = CeedVectorCreate(delegate, length, vec); CeedChk(ierr);
    return 0;
  }

  ierr = CeedCalloc(1,vec); CeedChk(ierr);
  (*vec)->ceed = ceed;
  ceed->refcount++;
  (*vec)->refcount = 1;
  (*vec)->length = length;
  (*vec)->state = 0;
  ierr = ceed->VectorCreate(length, *vec); CeedChk(ierr);
  return 0;
}

/**
  @brief Set the array used by a CeedVector, freeing any previously allocated
    array if applicable

  @param vec   CeedVector
  @param mtype Memory type of the array being passed
  @param cmode Copy mode for the array
  @param array Array to be used, or NULL with CEED_COPY_VALUES to have the
                 library allocate

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedVectorSetArray(CeedVector vec, CeedMemType mtype, CeedCopyMode cmode,
                       CeedScalar *array) {
  int ierr;

  if (!vec->SetArray)
    return CeedError(vec->ceed, 1, "Not supported");

  if (vec->state % 2 == 1)
    return CeedError(vec->ceed, 1,
                     "Cannot grant CeedVector array access, the access lock is already in use");

  if (vec->numreaders > 0)
    return CeedError(vec->ceed, 1,
                     "Cannot grant CeedVector array access, a process has read access");

  ierr = vec->SetArray(vec, mtype, cmode, array); CeedChk(ierr);
  vec->state += 2;

  return 0;
}

/**
  @brief Set the CeedVector to a constant value

  @param vec        CeedVector
  @param[in] value  Value to be used

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedVectorSetValue(CeedVector vec, CeedScalar value) {
  int ierr;

  if (vec->state % 2 == 1)
    return CeedError(vec->ceed, 1,
                     "Cannot grant CeedVector array access, the access lock is already in use");

  if (vec->SetValue) {
    ierr = vec->SetValue(vec, value); CeedChk(ierr);
  } else {
    CeedScalar *array;
    ierr = CeedVectorGetArray(vec, CEED_MEM_HOST, &array); CeedChk(ierr);
    for (int i=0; i<vec->length; i++) array[i] = value;
    ierr = CeedVectorRestoreArray(vec, &array); CeedChk(ierr);
  }

  vec->state += 2;

  return 0;
}

/**
  @brief Sync the CeedVector to a specified memtype

  @param vec        CeedVector
  @param mtype      Memtype to be synced

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedVectorSyncArray(CeedVector vec, CeedMemType mtype) {
  int ierr;

  if (vec->state % 2 == 1)
    return CeedError(vec->ceed, 1,
                     "Cannot sync CeedVector, the access lock is already in use");

  if (vec->SyncArray) {
    ierr = vec->SyncArray(vec, mtype); CeedChk(ierr);
  } else {
    const CeedScalar *array;
    ierr = CeedVectorGetArrayRead(vec, mtype, &array); CeedChk(ierr);
    ierr = CeedVectorRestoreArrayRead(vec, &array); CeedChk(ierr);
  }

  return 0;
}

/**
  @brief Get read/write access to a CeedVector via the specified memory type

  @param vec        CeedVector to access
  @param mtype      Memory type on which to access the array.  If the backend
                    uses a different memory type, this will perform a copy and
                    CeedVectorRestoreArray() will copy back.
  @param[out] array Array on memory type mtype

  @note The CeedVectorGetArray* and CeedVectorRestoreArray* functions provide
    access to array pointers in the desired memory space. Pairing get/restore
    allows the Vector to track access, thus knowing if norms or other
    operations may need to be recomputed.

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedVectorGetArray(CeedVector vec, CeedMemType mtype, CeedScalar **array) {
  int ierr;

  if (!vec->GetArray)
    return CeedError(vec->ceed, 1, "Not supported");

  if (vec->state % 2 == 1)
    return CeedError(vec->ceed, 1,
                     "Cannot grant CeedVector array access, the access lock is already in use");

  if (vec->numreaders > 0)
    return CeedError(vec->ceed, 1,
                     "Cannot grant CeedVector array access, a process has read access");

  ierr = vec->GetArray(vec, mtype, array); CeedChk(ierr);
  vec->state += 1;

  return 0;
}

/**
  @brief Get read-only access to a CeedVector via the specified memory type

  @param vec        CeedVector to access
  @param mtype      Memory type on which to access the array.  If the backend
                    uses a different memory type, this will perform a copy
                    (possibly cached).
  @param[out] array Array on memory type mtype

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedVectorGetArrayRead(CeedVector vec, CeedMemType mtype,
                           const CeedScalar **array) {
  int ierr;

  if (!vec->GetArrayRead)
    return CeedError(vec->ceed, 1, "Not supported");

  if (vec->state % 2 == 1)
    return CeedError(vec->ceed, 1,
                     "Cannot grant CeedVector read-only array access, the access lock is already in use");

  ierr = vec->GetArrayRead(vec, mtype, array); CeedChk(ierr);
  vec->numreaders++;

  return 0;
}

/**
  @brief Restore an array obtained using CeedVectorGetArray()

  @param vec     CeedVector to restore
  @param array   Array of vector data

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedVectorRestoreArray(CeedVector vec, CeedScalar **array) {
  int ierr;

  if (!vec->RestoreArray)
    return CeedError(vec->ceed, 1, "Not supported");

  if (vec->state % 2 != 1)
    return CeedError(vec->ceed, 1,
                     "Cannot restore CeedVector array access, access was not granted");

  ierr = vec->RestoreArray(vec); CeedChk(ierr);
  *array = NULL;
  vec->state += 1;

  return 0;
}

/**
  @brief Restore an array obtained using CeedVectorGetArrayRead()

  @param vec     CeedVector to restore
  @param array   Array of vector data

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedVectorRestoreArrayRead(CeedVector vec, const CeedScalar **array) {
  int ierr;

  if (!vec->RestoreArrayRead)
    return CeedError(vec->ceed, 1, "Not supported");

  ierr = vec->RestoreArrayRead(vec); CeedChk(ierr);
  *array = NULL;
  vec->numreaders--;

  return 0;
}

/**
  @brief View a CeedVector

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedVectorView(CeedVector vec, const char *fpfmt, FILE *stream) {
  const CeedScalar *x;

  int ierr = CeedVectorGetArrayRead(vec, CEED_MEM_HOST, &x); CeedChk(ierr);

  char fmt[1024];
  fprintf(stream, "CeedVector length %ld\n", (long)vec->length);
  snprintf(fmt, sizeof fmt, "  %s\n", fpfmt ? fpfmt : "%g");
  for (CeedInt i=0; i<vec->length; i++) {
    fprintf(stream, fmt, x[i]);
  }

  ierr = CeedVectorRestoreArrayRead(vec, &x); CeedChk(ierr);

  return 0;
}

/**
  @brief Get the Ceed associated with a CeedVector

  @param vec           CeedVector to retrieve state
  @param[out] ceed     Variable to store ceed

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedVectorGetCeed(CeedVector vec, Ceed *ceed) {
  *ceed = vec->ceed;
  return 0;
}

/**
  @brief Get the length of a CeedVector

  @param vec           CeedVector to retrieve length
  @param[out] length   Variable to store length

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedVectorGetLength(CeedVector vec, CeedInt *length) {
  *length = vec->length;
  return 0;
}

/**
  @brief Get the state of a CeedVector

  @param vec           CeedVector to retrieve state
  @param[out] state    Variable to store state

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedVectorGetState(CeedVector vec, uint64_t *state) {
  *state = vec->state;
  return 0;
}

/**
  @brief Get the backend data of a CeedVector

  @param vec           CeedVector to retrieve state
  @param[out] data     Variable to store data

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedVectorGetData(CeedVector vec, void* *data) {
  *data = vec->data;
  return 0;
}

/**
  @brief Set the backend data of a CeedVector

  @param[out] vec     CeedVector to retrieve state
  @paramdata          Data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedVectorSetData(CeedVector vec, void* *data) {
  vec->data = *data;
  return 0;
}

/**
  @brief Destroy a CeedVector

  @param vec   CeedVector to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedVectorDestroy(CeedVector *vec) {
  int ierr;

  if (!*vec || --(*vec)->refcount > 0) return 0;

  if ((*vec) && ((*vec)->state % 2) == 1)
    return CeedError((*vec)->ceed, 1,
                     "Cannot destroy CeedVector, the access lock is in use");

  if ((*vec)->Destroy) {
    ierr = (*vec)->Destroy(*vec); CeedChk(ierr);
  }

  ierr = CeedDestroy(&(*vec)->ceed); CeedChk(ierr);
  ierr = CeedFree(vec); CeedChk(ierr);

  return 0;
}

/// @cond DOXYGEN_SKIP
// Indicate that vector will be provided as an explicit argument to
//   CeedOperatorApply().
CeedVector CEED_VECTOR_ACTIVE = &ceed_vector_active;

// Indicate that no vector is applicable (i.e., for CEED_EVAL_WEIGHTS).
CeedVector CEED_VECTOR_NONE = &ceed_vector_none;
/// @endcond
/// @}
