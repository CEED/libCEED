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

/// @cond DOXYGEN_SKIP
static struct CeedVector_private ceed_vector_active;
static struct CeedVector_private ceed_vector_none;

/// @file
/// Implementation of public CeedVector interfaces
///
/// @defgroup CeedVector CeedVector: storing and manipulating vectors
/// @{

/// Create a vector of the specified length (does not allocate memory)
///
/// @param ceed   Ceed
/// @param length Length of vector
/// @param vec    New vector
int CeedVectorCreate(Ceed ceed, CeedInt length, CeedVector *vec) {
  int ierr;

  if (!ceed->VecCreate)
    return CeedError(ceed, 1, "Backend does not support VecCreate");
  ierr = CeedCalloc(1,vec); CeedChk(ierr);
  (*vec)->ceed = ceed;
  ceed->refcount++;
  (*vec)->refcount = 1;
  (*vec)->length = length;
  ierr = ceed->VecCreate(ceed, length, *vec); CeedChk(ierr);
  return 0;
}

/// Set the array used by a vector, freeing any previously allocated array if applicable.
///
/// @param x Vector
/// @param mtype Memory type of the array being passed
/// @param cmode Copy mode for the array
/// @param array Array to be used, or NULL with CEED_COPY_VALUES to have the library allocate
int CeedVectorSetArray(CeedVector x, CeedMemType mtype, CeedCopyMode cmode,
                       CeedScalar *array) {
  int ierr;

  if (!x || !x->SetArray)
    return CeedError(x ? x->ceed : NULL, 1, "Not supported");
  ierr = x->SetArray(x, mtype, cmode, array); CeedChk(ierr);
  return 0;
}

/// Set the array used by a vector, freeing any previously allocated array if applicable.
///
/// @param x Vector
/// @param value to be used
int CeedVectorSetValue(CeedVector x, CeedScalar value) {
  int ierr;
  CeedScalar *array;

  if (x->SetValue) {
    ierr = x->SetValue(x, value); CeedChk(ierr);
  } else {
    ierr = CeedVectorGetArray(x, CEED_MEM_HOST, &array); CeedChk(ierr);
    for (int i=0; i<x->length; i++) array[i] = value;
    ierr = CeedVectorRestoreArray(x, &array); CeedChk(ierr);
  }
  return 0;
}

/// Get read/write access to a vector via the specified memory type
///
/// @param x Vector to access
/// @param mtype Memory type on which to access the array.  If the backend uses
///              a different memory type, this will perform a copy and
///              CeedVectorRestoreArray() will copy back.
/// @param[out] array Array on memory type mtype
///
/// @note The CeedVectorGetArray* and CeedVectorRestoreArray* functions provide
///   access to array pointers in the desired memory space. Pairing get/restore
///   allows the Vector to track access, thus knowing if norms or other
///   operations may need to be recomputed.
///
/// @sa CeedVectorRestoreArray()
int CeedVectorGetArray(CeedVector x, CeedMemType mtype, CeedScalar **array) {
  int ierr;
  if (!x || !x->GetArray)
    return CeedError(x ? x->ceed : NULL, 1, "Not supported");
  ierr = x->GetArray(x, mtype, array); CeedChk(ierr);
  return 0;
}

/// Get read-only access to a vector via the specified memory type
///
/// @param x Vector to access
/// @param mtype Memory type on which to access the array.  If the backend uses
///              a different memory type, this will perform a copy (possibly cached).
/// @param[out] array Array on memory type mtype
///
/// @sa CeedVectorRestoreArrayRead()
int CeedVectorGetArrayRead(CeedVector x, CeedMemType mtype,
                           const CeedScalar **array) {
  int ierr;

  if (!x || !x->GetArrayRead)
    return CeedError(x ? x->ceed : NULL, 1, "Not supported");
  ierr = x->GetArrayRead(x, mtype, array); CeedChk(ierr);
  return 0;
}

/// Restore an array obtained using CeedVectorGetArray()
int CeedVectorRestoreArray(CeedVector x, CeedScalar **array) {
  int ierr;

  if (!x || !x->RestoreArray)
    return CeedError(x ? x->ceed : NULL, 1, "Not supported");
  ierr = x->RestoreArray(x, array); CeedChk(ierr);
  return 0;
}

/// Restore an array obtained using CeedVectorGetArrayRead()
int CeedVectorRestoreArrayRead(CeedVector x, const CeedScalar **array) {
  int ierr;

  if (!x || !x->RestoreArrayRead)
    return CeedError(x ? x->ceed : NULL, 1, "Not supported");
  ierr = x->RestoreArrayRead(x, array); CeedChk(ierr);
  return 0;
}

/** View a vector
 */
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

/// Get the length of a vector
CEED_EXTERN int CeedVectorGetLength(CeedVector vec, CeedInt *length) {
  *length = vec->length;
  return 0;
}

/// Destroy a vector
int CeedVectorDestroy(CeedVector *x) {
  int ierr;

  if (!*x || --(*x)->refcount > 0) return 0;
  if ((*x)->Destroy) {
    ierr = (*x)->Destroy(*x); CeedChk(ierr);
  }
  ierr = CeedDestroy(&(*x)->ceed); CeedChk(ierr);
  ierr = CeedFree(x); CeedChk(ierr);
  return 0;
}

/// Indicate that vector will be provided as an explicit argument to CeedOperatorApply().
CeedVector CEED_VECTOR_ACTIVE = &ceed_vector_active;

/// Indicate that no vector is applicable (i.e., for CEED_EVAL_WEIGHTS).
CeedVector CEED_VECTOR_NONE = &ceed_vector_none;
