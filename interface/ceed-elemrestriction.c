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
#include <stdbool.h>
#include <stdio.h>

/// @file
/// Implementation of CeedElemRestriction interfaces

/// ----------------------------------------------------------------------------
/// CeedElemRestriction Library Internal Functions
/// ----------------------------------------------------------------------------
/// @addtogroup CeedElemRestrictionDeveloper
/// @{

/**
  @brief Permute and pad offsets for a blocked restriction

  @param offsets    Array of shape [@a nelem, @a elemsize]. Row i holds the
                      ordered list of the offsets (into the input CeedVector)
                      for the unknowns corresponding to element i, where
                      0 <= i < @a nelem. All offsets must be in the range
                      [0, @a lsize - 1].
  @param blkoffsets Array of permuted and padded offsets of
                      shape [@a nblk, @a elemsize, @a blksize].
  @param nblk       Number of blocks
  @param nelem      Number of elements
  @param blksize    Number of elements in a block
  @param elemsize   Size of each element

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedPermutePadOffsets(const CeedInt *offsets, CeedInt *blkoffsets,
                          CeedInt nblk, CeedInt nelem, CeedInt blksize,
                          CeedInt elemsize) {
  for (CeedInt e=0; e<nblk*blksize; e+=blksize)
    for (int j=0; j<blksize; j++)
      for (int k=0; k<elemsize; k++)
        blkoffsets[e*elemsize + k*blksize + j]
          = offsets[CeedIntMin(e+j,nelem-1)*elemsize + k];
  return CEED_ERROR_SUCCESS;
}

/// @}

/// ----------------------------------------------------------------------------
/// CeedElemRestriction Backend API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedElemRestrictionBackend
/// @{

/**
  @brief Get the Ceed associated with a CeedElemRestriction

  @param rstr             CeedElemRestriction
  @param[out] ceed        Variable to store Ceed

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetCeed(CeedElemRestriction rstr, Ceed *ceed) {
  *ceed = rstr->ceed;
  return CEED_ERROR_SUCCESS;
}

/**

  @brief Get the strides of a strided CeedElemRestriction

  @param rstr             CeedElemRestriction
  @param[out] strides     Variable to store strides array

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetStrides(CeedElemRestriction rstr,
                                  CeedInt (*strides)[3]) {
  if (!rstr->strides)
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_MINOR,
                     "ElemRestriction has no stride data");
  // LCOV_EXCL_STOP

  for (int i=0; i<3; i++)
    (*strides)[i] = rstr->strides[i];
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get read-only access to a CeedElemRestriction offsets array by memtype

  @param rstr         CeedElemRestriction to retrieve offsets
  @param mtype        Memory type on which to access the array.  If the backend
                        uses a different memory type, this will perform a copy
                        (possibly cached).
  @param[out] offsets Array on memory type mtype

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionGetOffsets(CeedElemRestriction rstr, CeedMemType mtype,
                                  const CeedInt **offsets) {
  int ierr;

  if (!rstr->GetOffsets)
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_UNSUPPORTED,
                     "Backend does not support GetOffsets");
  // LCOV_EXCL_STOP

  ierr = rstr->GetOffsets(rstr, mtype, offsets); CeedChk(ierr);
  rstr->numreaders++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore an offsets array obtained using CeedElemRestrictionGetOffsets()

  @param rstr    CeedElemRestriction to restore
  @param offsets Array of offset data

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionRestoreOffsets(CeedElemRestriction rstr,
                                      const CeedInt **offsets) {
  *offsets = NULL;
  rstr->numreaders--;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the strided status of a CeedElemRestriction

  @param rstr             CeedElemRestriction
  @param[out] isstrided   Variable to store strided status, 1 if strided else 0

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionIsStrided(CeedElemRestriction rstr, bool *isstrided) {
  *isstrided = rstr->strides ? true : false;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the backend stride status of a CeedElemRestriction

  @param rstr             CeedElemRestriction
  @param[out] status      Variable to store stride status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionHasBackendStrides(CeedElemRestriction rstr,
    bool *hasbackendstrides) {
  if (!rstr->strides)
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_MINOR,
                     "ElemRestriction has no stride data");
  // LCOV_EXCL_STOP

  *hasbackendstrides = ((rstr->strides[0] == CEED_STRIDES_BACKEND[0]) &&
                        (rstr->strides[1] == CEED_STRIDES_BACKEND[1]) &&
                        (rstr->strides[2] == CEED_STRIDES_BACKEND[2]));
  return CEED_ERROR_SUCCESS;
}

/**

  @brief Get the E-vector layout of a CeedElemRestriction

  @param rstr             CeedElemRestriction
  @param[out] layout      Variable to store layout array,
                            stored as [nodes, components, elements].
                            The data for node i, component j, element k in the
                            E-vector is given by
                            i*layout[0] + j*layout[1] + k*layout[2]

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetELayout(CeedElemRestriction rstr,
                                  CeedInt (*layout)[3]) {
  if (!rstr->layout[0])
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_MINOR,
                     "ElemRestriction has no layout data");
  // LCOV_EXCL_STOP

  for (int i=0; i<3; i++)
    (*layout)[i] = rstr->layout[i];
  return CEED_ERROR_SUCCESS;
}

/**

  @brief Set the E-vector layout of a CeedElemRestriction

  @param rstr             CeedElemRestriction
  @param layout           Variable to containing layout array,
                            stored as [nodes, components, elements].
                            The data for node i, component j, element k in the
                            E-vector is given by
                            i*layout[0] + j*layout[1] + k*layout[2]

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionSetELayout(CeedElemRestriction rstr,
                                  CeedInt layout[3]) {
  for (int i = 0; i<3; i++)
    rstr->layout[i] = layout[i];
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the backend data of a CeedElemRestriction

  @param rstr             CeedElemRestriction
  @param[out] data        Variable to store data

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetData(CeedElemRestriction rstr, void *data) {
  *(void **)data = rstr->data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the backend data of a CeedElemRestriction

  @param[out] rstr        CeedElemRestriction
  @param data             Data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionSetData(CeedElemRestriction rstr, void *data) {
  rstr->data = data;
  return CEED_ERROR_SUCCESS;
}

/// @}

/// @cond DOXYGEN_SKIP
static struct CeedElemRestriction_private ceed_elemrestriction_none;
/// @endcond

/// ----------------------------------------------------------------------------
/// CeedElemRestriction Public API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedElemRestrictionUser
/// @{

/// Indicate that the stride is determined by the backend
const CeedInt CEED_STRIDES_BACKEND[3] = {};

/// Indicate that no CeedElemRestriction is provided by the user
const CeedElemRestriction CEED_ELEMRESTRICTION_NONE =
  &ceed_elemrestriction_none;

/**
  @brief Create a CeedElemRestriction

  @param ceed       A Ceed object where the CeedElemRestriction will be created
  @param nelem      Number of elements described in the @a offsets array
  @param elemsize   Size (number of "nodes") per element
  @param ncomp      Number of field components per interpolation node
                      (1 for scalar fields)
  @param compstride Stride between components for the same L-vector "node".
                      Data for node i, component j, element k can be found in
                      the L-vector at index
                        offsets[i + k*elemsize] + j*compstride.
  @param lsize      The size of the L-vector. This vector may be larger than
                      the elements and fields given by this restriction.
  @param mtype      Memory type of the @a offsets array, see CeedMemType
  @param cmode      Copy mode for the @a offsets array, see CeedCopyMode
  @param offsets    Array of shape [@a nelem, @a elemsize]. Row i holds the
                      ordered list of the offsets (into the input CeedVector)
                      for the unknowns corresponding to element i, where
                      0 <= i < @a nelem. All offsets must be in the range
                      [0, @a lsize - 1].
  @param[out] rstr  Address of the variable where the newly created
                      CeedElemRestriction will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionCreate(Ceed ceed, CeedInt nelem, CeedInt elemsize,
                              CeedInt ncomp, CeedInt compstride,
                              CeedInt lsize, CeedMemType mtype,
                              CeedCopyMode cmode, const CeedInt *offsets,
                              CeedElemRestriction *rstr) {
  int ierr;

  if (!ceed->ElemRestrictionCreate) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction");
    CeedChk(ierr);

    if (!delegate)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                       "Backend does not support ElemRestrictionCreate");
    // LCOV_EXCL_STOP

    ierr = CeedElemRestrictionCreate(delegate, nelem, elemsize, ncomp,
                                     compstride, lsize, mtype, cmode,
                                     offsets, rstr); CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  }

  ierr = CeedCalloc(1, rstr); CeedChk(ierr);
  (*rstr)->ceed = ceed;
  ceed->refcount++;
  (*rstr)->refcount = 1;
  (*rstr)->nelem = nelem;
  (*rstr)->elemsize = elemsize;
  (*rstr)->ncomp = ncomp;
  (*rstr)->compstride = compstride;
  (*rstr)->lsize = lsize;
  (*rstr)->nblk = nelem;
  (*rstr)->blksize = 1;
  ierr = ceed->ElemRestrictionCreate(mtype, cmode, offsets, *rstr);
  CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a strided CeedElemRestriction

  @param ceed       A Ceed object where the CeedElemRestriction will be created
  @param nelem      Number of elements described by the restriction
  @param elemsize   Size (number of "nodes") per element
  @param ncomp      Number of field components per interpolation "node"
                      (1 for scalar fields)
  @param lsize      The size of the L-vector. This vector may be larger than
                      the elements and fields given by this restriction.
  @param strides    Array for strides between [nodes, components, elements].
                      Data for node i, component j, element k can be found in
                      the L-vector at index
                        i*strides[0] + j*strides[1] + k*strides[2].
                      @a CEED_STRIDES_BACKEND may be used with vectors created
                      by a Ceed backend.
  @param rstr       Address of the variable where the newly created
                      CeedElemRestriction will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionCreateStrided(Ceed ceed, CeedInt nelem, CeedInt elemsize,
                                     CeedInt ncomp, CeedInt lsize,
                                     const CeedInt strides[3],
                                     CeedElemRestriction *rstr) {
  int ierr;

  if (!ceed->ElemRestrictionCreate) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction");
    CeedChk(ierr);

    if (!delegate)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                       "Backend does not support ElemRestrictionCreate");
    // LCOV_EXCL_STOP

    ierr = CeedElemRestrictionCreateStrided(delegate, nelem, elemsize, ncomp,
                                            lsize, strides, rstr);
    CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  }

  ierr = CeedCalloc(1, rstr); CeedChk(ierr);
  (*rstr)->ceed = ceed;
  ceed->refcount++;
  (*rstr)->refcount = 1;
  (*rstr)->nelem = nelem;
  (*rstr)->elemsize = elemsize;
  (*rstr)->ncomp = ncomp;
  (*rstr)->lsize = lsize;
  (*rstr)->nblk = nelem;
  (*rstr)->blksize = 1;
  ierr = CeedMalloc(3, &(*rstr)->strides); CeedChk(ierr);
  for (int i=0; i<3; i++)
    (*rstr)->strides[i] = strides[i];
  ierr = ceed->ElemRestrictionCreate(CEED_MEM_HOST, CEED_OWN_POINTER, NULL,
                                     *rstr);
  CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a blocked CeedElemRestriction, typically only called by backends

  @param ceed       A Ceed object where the CeedElemRestriction will be created.
  @param nelem      Number of elements described in the @a offsets array.
  @param elemsize   Size (number of unknowns) per element
  @param blksize    Number of elements in a block
  @param ncomp      Number of field components per interpolation node
                      (1 for scalar fields)
  @param compstride Stride between components for the same L-vector "node".
                      Data for node i, component j, element k can be found in
                      the L-vector at index
                        offsets[i + k*elemsize] + j*compstride.
  @param lsize      The size of the L-vector. This vector may be larger than
                      the elements and fields given by this restriction.
  @param mtype      Memory type of the @a offsets array, see CeedMemType
  @param cmode      Copy mode for the @a offsets array, see CeedCopyMode
  @param offsets    Array of shape [@a nelem, @a elemsize]. Row i holds the
                      ordered list of the offsets (into the input CeedVector)
                      for the unknowns corresponding to element i, where
                      0 <= i < @a nelem. All offsets must be in the range
                      [0, @a lsize - 1]. The backend will permute and pad this
                      array to the desired ordering for the blocksize, which is
                      typically given by the backend. The default reordering is
                      to interlace elements.
  @param rstr       Address of the variable where the newly created
                      CeedElemRestriction will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
 **/
int CeedElemRestrictionCreateBlocked(Ceed ceed, CeedInt nelem, CeedInt elemsize,
                                     CeedInt blksize, CeedInt ncomp,
                                     CeedInt compstride, CeedInt lsize,
                                     CeedMemType mtype, CeedCopyMode cmode,
                                     const CeedInt *offsets,
                                     CeedElemRestriction *rstr) {
  int ierr;
  CeedInt *blkoffsets;
  CeedInt nblk = (nelem / blksize) + !!(nelem % blksize);

  if (!ceed->ElemRestrictionCreateBlocked) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction");
    CeedChk(ierr);

    if (!delegate)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support "
                       "ElemRestrictionCreateBlocked");
    // LCOV_EXCL_STOP

    ierr = CeedElemRestrictionCreateBlocked(delegate, nelem, elemsize, blksize,
                                            ncomp, compstride, lsize, mtype,
                                            cmode, offsets, rstr);
    CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  }

  ierr = CeedCalloc(1, rstr); CeedChk(ierr);

  ierr = CeedCalloc(nblk*blksize*elemsize, &blkoffsets); CeedChk(ierr);
  ierr = CeedPermutePadOffsets(offsets, blkoffsets, nblk, nelem, blksize,
                               elemsize);
  CeedChk(ierr);

  (*rstr)->ceed = ceed;
  ceed->refcount++;
  (*rstr)->refcount = 1;
  (*rstr)->nelem = nelem;
  (*rstr)->elemsize = elemsize;
  (*rstr)->ncomp = ncomp;
  (*rstr)->compstride = compstride;
  (*rstr)->lsize = lsize;
  (*rstr)->nblk = nblk;
  (*rstr)->blksize = blksize;
  ierr = ceed->ElemRestrictionCreateBlocked(CEED_MEM_HOST, CEED_OWN_POINTER,
         (const CeedInt *) blkoffsets, *rstr); CeedChk(ierr);
  if (cmode == CEED_OWN_POINTER) {
    ierr = CeedFree(&offsets); CeedChk(ierr);
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a blocked strided CeedElemRestriction

  @param ceed       A Ceed object where the CeedElemRestriction will be created
  @param nelem      Number of elements described by the restriction
  @param elemsize   Size (number of "nodes") per element
  @param blksize    Number of elements in a block
  @param ncomp      Number of field components per interpolation node
                      (1 for scalar fields)
  @param lsize      The size of the L-vector. This vector may be larger than
                      the elements and fields given by this restriction.
  @param strides    Array for strides between [nodes, components, elements].
                      Data for node i, component j, element k can be found in
                      the L-vector at index
                        i*strides[0] + j*strides[1] + k*strides[2].
                      @a CEED_STRIDES_BACKEND may be used with vectors created
                      by a Ceed backend.
  @param rstr       Address of the variable where the newly created
                      CeedElemRestriction will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionCreateBlockedStrided(Ceed ceed, CeedInt nelem,
    CeedInt elemsize, CeedInt blksize, CeedInt ncomp, CeedInt lsize,
    const CeedInt strides[3], CeedElemRestriction *rstr) {
  int ierr;
  CeedInt nblk = (nelem / blksize) + !!(nelem % blksize);

  if (!ceed->ElemRestrictionCreateBlocked) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction");
    CeedChk(ierr);

    if (!delegate)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support "
                       "ElemRestrictionCreateBlocked");
    // LCOV_EXCL_STOP

    ierr = CeedElemRestrictionCreateBlockedStrided(delegate, nelem, elemsize,
           blksize, ncomp, lsize, strides, rstr);
    CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  }

  ierr = CeedCalloc(1, rstr); CeedChk(ierr);

  (*rstr)->ceed = ceed;
  ceed->refcount++;
  (*rstr)->refcount = 1;
  (*rstr)->nelem = nelem;
  (*rstr)->elemsize = elemsize;
  (*rstr)->ncomp = ncomp;
  (*rstr)->lsize = lsize;
  (*rstr)->nblk = nblk;
  (*rstr)->blksize = blksize;
  ierr = CeedMalloc(3, &(*rstr)->strides); CeedChk(ierr);
  for (int i=0; i<3; i++)
    (*rstr)->strides[i] = strides[i];
  ierr = ceed->ElemRestrictionCreateBlocked(CEED_MEM_HOST, CEED_OWN_POINTER,
         NULL, *rstr); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create CeedVectors associated with a CeedElemRestriction

  @param rstr  CeedElemRestriction
  @param lvec  The address of the L-vector to be created, or NULL
  @param evec  The address of the E-vector to be created, or NULL

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionCreateVector(CeedElemRestriction rstr, CeedVector *lvec,
                                    CeedVector *evec) {
  int ierr;
  CeedInt n, m;
  m = rstr->lsize;
  n = rstr->nblk * rstr->blksize * rstr->elemsize * rstr->ncomp;
  if (lvec) {
    ierr = CeedVectorCreate(rstr->ceed, m, lvec); CeedChk(ierr);
  }
  if (evec) {
    ierr = CeedVectorCreate(rstr->ceed, n, evec); CeedChk(ierr);
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restrict an L-vector to an E-vector or apply its transpose

  @param rstr    CeedElemRestriction
  @param tmode   Apply restriction or transpose
  @param u       Input vector (of size @a lsize when tmode=@ref CEED_NOTRANSPOSE)
  @param ru      Output vector (of shape [@a nelem * @a elemsize] when
                   tmode=@ref CEED_NOTRANSPOSE). Ordering of the e-vector is decided
                   by the backend.
  @param request Request or @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionApply(CeedElemRestriction rstr, CeedTransposeMode tmode,
                             CeedVector u, CeedVector ru,
                             CeedRequest *request) {
  CeedInt m,n;
  int ierr;

  if (tmode == CEED_NOTRANSPOSE) {
    m = rstr->nblk * rstr->blksize * rstr->elemsize * rstr->ncomp;
    n = rstr->lsize;
  } else {
    m = rstr->lsize;
    n = rstr->nblk * rstr->blksize * rstr->elemsize * rstr->ncomp;
  }
  if (n != u->length)
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_DIMENSION,
                     "Input vector size %d not compatible with "
                     "element restriction (%d, %d)", u->length, m, n);
  // LCOV_EXCL_STOP
  if (m != ru->length)
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_DIMENSION,
                     "Output vector size %d not compatible with "
                     "element restriction (%d, %d)", ru->length, m, n);
  // LCOV_EXCL_STOP
  ierr = rstr->Apply(rstr, tmode, u, ru, request); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restrict an L-vector to a block of an E-vector or apply its transpose

  @param rstr    CeedElemRestriction
  @param block   Block number to restrict to/from, i.e. block=0 will handle
                   elements [0 : blksize] and block=3 will handle elements
                   [3*blksize : 4*blksize]
  @param tmode   Apply restriction or transpose
  @param u       Input vector (of size @a lsize when tmode=@ref CEED_NOTRANSPOSE)
  @param ru      Output vector (of shape [@a blksize * @a elemsize] when
                   tmode=@ref CEED_NOTRANSPOSE). Ordering of the e-vector is decided
                   by the backend.
  @param request Request or @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionApplyBlock(CeedElemRestriction rstr, CeedInt block,
                                  CeedTransposeMode tmode, CeedVector u,
                                  CeedVector ru, CeedRequest *request) {
  CeedInt m,n;
  int ierr;

  if (tmode == CEED_NOTRANSPOSE) {
    m = rstr->blksize * rstr->elemsize * rstr->ncomp;
    n = rstr->lsize;
  } else {
    m = rstr->lsize;
    n = rstr->blksize * rstr->elemsize * rstr->ncomp;
  }
  if (n != u->length)
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_DIMENSION,
                     "Input vector size %d not compatible with "
                     "element restriction (%d, %d)", u->length, m, n);
  // LCOV_EXCL_STOP
  if (m != ru->length)
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_DIMENSION,
                     "Output vector size %d not compatible with "
                     "element restriction (%d, %d)", ru->length, m, n);
  // LCOV_EXCL_STOP
  if (rstr->blksize*block > rstr->nelem)
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_DIMENSION,
                     "Cannot retrieve block %d, element %d > "
                     "total elements %d", block, rstr->blksize*block,
                     rstr->nelem);
  // LCOV_EXCL_STOP
  ierr = rstr->ApplyBlock(rstr, block, tmode, u, ru, request);
  CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the L-vector component stride

  @param rstr             CeedElemRestriction
  @param[out] compstride  Variable to store component stride

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetCompStride(CeedElemRestriction rstr,
                                     CeedInt *compstride) {
  *compstride = rstr->compstride;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the total number of elements in the range of a CeedElemRestriction

  @param rstr             CeedElemRestriction
  @param[out] numelem     Variable to store number of elements

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetNumElements(CeedElemRestriction rstr,
                                      CeedInt *numelem) {
  *numelem = rstr->nelem;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the size of elements in the CeedElemRestriction

  @param rstr             CeedElemRestriction
  @param[out] elemsize    Variable to store size of elements

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetElementSize(CeedElemRestriction rstr,
                                      CeedInt *elemsize) {
  *elemsize = rstr->elemsize;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the size of the l-vector for a CeedElemRestriction

  @param rstr             CeedElemRestriction
  @param[out] numnodes    Variable to store number of nodes

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetLVectorSize(CeedElemRestriction rstr,
                                      CeedInt *lsize) {
  *lsize = rstr->lsize;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the number of components in the elements of a
         CeedElemRestriction

  @param rstr             CeedElemRestriction
  @param[out] numcomp     Variable to store number of components

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetNumComponents(CeedElemRestriction rstr,
                                        CeedInt *numcomp) {
  *numcomp = rstr->ncomp;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the number of blocks in a CeedElemRestriction

  @param rstr             CeedElemRestriction
  @param[out] numblock    Variable to store number of blocks

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetNumBlocks(CeedElemRestriction rstr,
                                    CeedInt *numblock) {
  *numblock = rstr->nblk;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the size of blocks in the CeedElemRestriction

  @param rstr             CeedElemRestriction
  @param[out] blksize     Variable to store size of blocks

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetBlockSize(CeedElemRestriction rstr,
                                    CeedInt *blksize) {
  *blksize = rstr->blksize;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the multiplicity of nodes in a CeedElemRestriction

  @param rstr             CeedElemRestriction
  @param[out] mult        Vector to store multiplicity (of size lsize)

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionGetMultiplicity(CeedElemRestriction rstr,
                                       CeedVector mult) {
  int ierr;
  CeedVector evec;

  // Create and set evec
  ierr = CeedElemRestrictionCreateVector(rstr, NULL, &evec); CeedChk(ierr);
  ierr = CeedVectorSetValue(evec, 1.0); CeedChk(ierr);
  ierr = CeedVectorSetValue(mult, 0.0); CeedChk(ierr);

  // Apply to get multiplicity
  ierr = CeedElemRestrictionApply(rstr, CEED_TRANSPOSE, evec, mult,
                                  CEED_REQUEST_IMMEDIATE); CeedChk(ierr);

  // Cleanup
  ierr = CeedVectorDestroy(&evec); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief View a CeedElemRestriction

  @param[in] rstr    CeedElemRestriction to view
  @param[in] stream  Stream to write; typically stdout/stderr or a file

  @return Error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionView(CeedElemRestriction rstr, FILE *stream) {
  char stridesstr[500];
  if (rstr->strides)
    sprintf(stridesstr, "[%d, %d, %d]", rstr->strides[0], rstr->strides[1],
            rstr->strides[2]);
  else
    sprintf(stridesstr, "%d", rstr->compstride);

  fprintf(stream, "%sCeedElemRestriction from (%d, %d) to %d elements with %d "
          "nodes each and %s %s\n", rstr->blksize > 1 ? "Blocked " : "",
          rstr->lsize, rstr->ncomp, rstr->nelem, rstr->elemsize,
          rstr->strides ? "strides" : "component stride", stridesstr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Destroy a CeedElemRestriction

  @param rstr  CeedElemRestriction to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionDestroy(CeedElemRestriction *rstr) {
  int ierr;

  if (!*rstr || --(*rstr)->refcount > 0) return CEED_ERROR_SUCCESS;
  if ((*rstr)->numreaders)
    return CeedError((*rstr)->ceed, CEED_ERROR_ACCESS,
                     "Cannot destroy CeedElemRestriction, "
                     "a process has read access to the offset data");
  if ((*rstr)->Destroy) {
    ierr = (*rstr)->Destroy(*rstr); CeedChk(ierr);
  }
  ierr = CeedFree(&(*rstr)->strides); CeedChk(ierr);
  ierr = CeedDestroy(&(*rstr)->ceed); CeedChk(ierr);
  ierr = CeedFree(rstr); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/// @}
