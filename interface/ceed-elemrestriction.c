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

/// @file
/// Implementation of CeedElemRestriction interfaces

/// ----------------------------------------------------------------------------
/// CeedElemRestriction Library Internal Functions
/// ----------------------------------------------------------------------------
/// @addtogroup CeedElemRestrictionDeveloper
/// @{

/**
  @brief Permute and pad indices for a blocked restriction

  @param indices    Array of shape [@a nelem, @a elemsize]. Row i holds the
                      ordered list of the indices (into the input CeedVector)
                      for the unknowns corresponding to element i, where
                      0 <= i < @a nelem. All indices must be in the range
                      [0, @a nnodes).
  @param blkindices Array of permuted and padded indices of
                      shape [@a nblk, @a elemsize, @a blksize].
  @param nblk       Number of blocks
  @param nelem      Number of elements
  @param blksize    Number of elements in a block
  @param elemsize   Size of each element

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedPermutePadIndices(const CeedInt *indices, CeedInt *blkindices,
                          CeedInt nblk, CeedInt nelem, CeedInt blksize,
                          CeedInt elemsize) {
  for (CeedInt e = 0; e < nblk*blksize; e+=blksize)
    for (int j = 0; j < blksize; j++)
      for (int k = 0; k < elemsize; k++)
        blkindices[e*elemsize + k*blksize + j]
          = indices[CeedIntMin(e+j,nelem-1)*elemsize + k];
  return 0;
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
  return 0;
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
    return CeedError(rstr->ceed, 1, "ElemRestriction has no stride data");
  // LCOV_EXCL_STOP

  for (int i = 0; i<3; i++)
    (*strides)[i] = rstr->strides[i];
  return 0;
}

/**
  @brief Get the backend stride status of a CeedElemRestriction

  @param rstr             CeedElemRestriction
  @param[out] status      Variable to store stride status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetBackendStridesStatus(CeedElemRestriction rstr,
    bool *status) {
  if (!rstr->strides)
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, 1, "ElemRestriction has no stride data");
  // LCOV_EXCL_STOP

  *status = ((rstr->strides[0] == CEED_STRIDES_BACKEND[0]) &&
             (rstr->strides[1] == CEED_STRIDES_BACKEND[1]) &&
             (rstr->strides[2] == CEED_STRIDES_BACKEND[2]));
  return 0;
}

/**
  @brief Get the backend data of a CeedElemRestriction

  @param rstr             CeedElemRestriction
  @param[out] data        Variable to store data

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetData(CeedElemRestriction rstr, void **data) {
  *data = rstr->data;
  return 0;
}

/**
  @brief Set the backend data of a CeedElemRestriction

  @param[out] rstr        CeedElemRestriction
  @param data             Data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionSetData(CeedElemRestriction rstr, void **data) {
  rstr->data = *data;
  return 0;
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

/// Indicate that no ElemRestriction is provided by the user
const CeedElemRestriction CEED_ELEMRESTRICTION_NONE =
  &ceed_elemrestriction_none;

/**
  @brief Create a CeedElemRestriction

  @param ceed       A Ceed object where the CeedElemRestriction will be created
  @param imode      Ordering of the ncomp components, i.e. it specifies
                      the ordering of the components of the L-vector used
                      by this CeedElemRestriction. CEED_NONINTERLACED indicates
                      the component is the outermost index and CEED_INTERLACED
                      indicates the component is the innermost index in
                      ordering of the L-vector.
  @param nelem      Number of elements described in the @a indices array
  @param elemsize   Size (number of "nodes") per element
  @param nnodes     The number of nodes in the L-vector. The input CeedVector
                      to which the restriction will be applied is of size
                      @a nnodes * @a ncomp. This size may include data
                      used by other CeedElemRestriction objects describing
                      different types of elements.
  @param ncomp      Number of field components per interpolation node
                      (1 for scalar fields)
  @param mtype      Memory type of the @a indices array, see CeedMemType
  @param cmode      Copy mode for the @a indices array, see CeedCopyMode
  @param indices    Array of shape [@a nelem, @a elemsize]. Row i holds the
                      ordered list of the indices (into the input CeedVector)
                      for the unknowns corresponding to element i, where
                      0 <= i < @a nelem. All indices must be in the range
                      [0, @a nnodes - 1].
  @param[out] rstr  Address of the variable where the newly created
                      CeedElemRestriction will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionCreate(Ceed ceed, CeedInterlaceMode imode,
                              CeedInt nelem, CeedInt elemsize, CeedInt nnodes,
                              CeedInt ncomp, CeedMemType mtype,
                              CeedCopyMode cmode, const CeedInt *indices,
                              CeedElemRestriction *rstr) {
  int ierr;

  if (!ceed->ElemRestrictionCreate) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction");
    CeedChk(ierr);

    if (!delegate)
      // LCOV_EXCL_START
      return CeedError(ceed, 1, "Backend does not support ElemRestrictionCreate");
    // LCOV_EXCL_STOP

    ierr = CeedElemRestrictionCreate(delegate, imode, nelem, elemsize,
                                     nnodes, ncomp, mtype, cmode,
                                     indices, rstr); CeedChk(ierr);
    return 0;
  }

  ierr = CeedCalloc(1, rstr); CeedChk(ierr);
  (*rstr)->ceed = ceed;
  ceed->refcount++;
  (*rstr)->refcount = 1;
  (*rstr)->imode = imode;
  (*rstr)->nelem = nelem;
  (*rstr)->elemsize = elemsize;
  (*rstr)->nnodes = nnodes;
  (*rstr)->ncomp = ncomp;
  (*rstr)->nblk = nelem;
  (*rstr)->blksize = 1;
  ierr = ceed->ElemRestrictionCreate(mtype, cmode, indices, *rstr); CeedChk(ierr);
  return 0;
}

/**
  @brief Create a strided CeedElemRestriction

  @param ceed       A Ceed object where the CeedElemRestriction will be created
  @param nelem      Number of elements described by the restriction
  @param elemsize   Size (number of "nodes") per element
  @param nnodes     The number of nodes in the L-vector. The input CeedVector
                      to which the restriction will be applied is of size
                      @a nnodes * @a ncomp. This size may include data
                      used by other CeedElemRestriction objects describing
                      different types of elements.
  @param ncomp      Number of field components per interpolation node
                      (1 for scalar fields)
  @param strides    Array for strides between [nodes, components, elements].
                      The data for node i, component j, element k in the
                      L-vector is given by
                        i*strides[0] + j*strides[1] + k*strides[2]
  @param rstr       Address of the variable where the newly created
                      CeedElemRestriction will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionCreateStrided(Ceed ceed, CeedInt nelem, CeedInt elemsize,
                                     CeedInt nnodes, CeedInt ncomp,
                                     const CeedInt strides[3],
                                     CeedElemRestriction *rstr) {
  int ierr;

  if (!ceed->ElemRestrictionCreate) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction");
    CeedChk(ierr);

    if (!delegate)
      // LCOV_EXCL_START
      return CeedError(ceed, 1, "Backend does not support ElemRestrictionCreate");
    // LCOV_EXCL_STOP

    ierr = CeedElemRestrictionCreateStrided(delegate, nelem, elemsize, nnodes,
                                            ncomp, strides, rstr); CeedChk(ierr);
    return 0;
  }

  ierr = CeedCalloc(1, rstr); CeedChk(ierr);
  (*rstr)->ceed = ceed;
  ceed->refcount++;
  (*rstr)->refcount = 1;
  (*rstr)->nelem = nelem;
  (*rstr)->elemsize = elemsize;
  (*rstr)->nnodes = nnodes;
  (*rstr)->ncomp = ncomp;
  (*rstr)->nblk = nelem;
  (*rstr)->blksize = 1;
  ierr = CeedMalloc(3, &(*rstr)->strides); CeedChk(ierr);
  for (int i = 0; i<3; i++)
    (*rstr)->strides[i] = strides[i];
  ierr = ceed->ElemRestrictionCreate(CEED_MEM_HOST, CEED_OWN_POINTER, NULL,
                                     *rstr);
  CeedChk(ierr);
  return 0;
}

/**
  @brief Create a blocked CeedElemRestriction, typically only called by backends

  @param ceed       A Ceed object where the CeedElemRestriction will be created.
  @param imode      Ordering of the ncomp components, i.e. it specifies
                      the ordering of the components of the L-vector used
                      by this CeedElemRestriction. CEED_NONINTERLACED indicates
                      the component is the outermost index and CEED_INTERLACED
                      indicates the component is the innermost index in
                      ordering of the L-vector.
  @param nelem      Number of elements described in the @a indices array.
  @param elemsize   Size (number of unknowns) per element
  @param blksize    Number of elements in a block
  @param nnodes     The number of nodes in the L-vector. The input CeedVector
                      to which the restriction will be applied is of size
                      @a nnodes * @a ncomp. This size may include data
                      used by other CeedElemRestriction objects describing
                      different types of elements.
  @param ncomp      Number of field components per interpolation node
                      (1 for scalar fields)
  @param mtype      Memory type of the @a indices array, see CeedMemType
  @param cmode      Copy mode for the @a indices array, see CeedCopyMode
  @param indices    Array of shape [@a nelem, @a elemsize]. Row i holds the
                      ordered list of the indices (into the input CeedVector)
                      for the unknowns corresponding to element i, where
                      0 <= i < @a nelem. All indices must be in the range
                      [0, @a nnodes). The backend will permute and pad this
                      array to the desired ordering for the blocksize, which is
                      typically given by the backend. The default reordering is
                      to interlace elements.
  @param rstr       Address of the variable where the newly created
                      CeedElemRestriction will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
 **/
int CeedElemRestrictionCreateBlocked(Ceed ceed,  CeedInterlaceMode imode,
                                     CeedInt nelem, CeedInt elemsize,
                                     CeedInt blksize, CeedInt nnodes,
                                     CeedInt ncomp, CeedMemType mtype,
                                     CeedCopyMode cmode, const CeedInt *indices,
                                     CeedElemRestriction *rstr) {
  int ierr;
  CeedInt *blkindices;
  CeedInt nblk = (nelem / blksize) + !!(nelem % blksize);

  if (!ceed->ElemRestrictionCreateBlocked) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction");
    CeedChk(ierr);

    if (!delegate)
      // LCOV_EXCL_START
      return CeedError(ceed, 1, "Backend does not support "
                       "ElemRestrictionCreateBlocked");
    // LCOV_EXCL_STOP

    ierr = CeedElemRestrictionCreateBlocked(delegate, imode, nelem, elemsize,
                                            blksize, nnodes, ncomp, mtype, cmode,
                                            indices, rstr); CeedChk(ierr);
    return 0;
  }

  ierr = CeedCalloc(1, rstr); CeedChk(ierr);

  ierr = CeedCalloc(nblk*blksize*elemsize, &blkindices); CeedChk(ierr);
  ierr = CeedPermutePadIndices(indices, blkindices, nblk, nelem, blksize,
                               elemsize);
  CeedChk(ierr);

  (*rstr)->ceed = ceed;
  ceed->refcount++;
  (*rstr)->refcount = 1;
  (*rstr)->imode = imode;
  (*rstr)->nelem = nelem;
  (*rstr)->elemsize = elemsize;
  (*rstr)->nnodes = nnodes;
  (*rstr)->ncomp = ncomp;
  (*rstr)->nblk = nblk;
  (*rstr)->blksize = blksize;
  ierr = ceed->ElemRestrictionCreateBlocked(CEED_MEM_HOST, CEED_OWN_POINTER,
         (const CeedInt *) blkindices, *rstr); CeedChk(ierr);

  if (cmode == CEED_OWN_POINTER) {
    ierr = CeedFree(&indices); CeedChk(ierr);
  }

  return 0;
}

/**
  @brief Create a blocked strided CeedElemRestriction

  @param ceed       A Ceed object where the CeedElemRestriction will be created
  @param nelem      Number of elements described by the restriction
  @param elemsize   Size (number of "nodes") per element
  @param blksize    Number of elements in a block
  @param nnodes     The number of nodes in the L-vector. The input CeedVector
                      to which the restriction will be applied is of size
                      @a nnodes * @a ncomp. This size may include data
                      used by other CeedElemRestriction objects describing
                      different types of elements.
  @param ncomp      Number of field components per interpolation node
                      (1 for scalar fields)
  @param strides    Array for strides between [nodes, components, elements].
                      The data for node i, component j, element k in the
                      L-vector is given by
                        i*strides[0] + j*strides[1] + k*strides[2]
  @param rstr       Address of the variable where the newly created
                      CeedElemRestriction will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionCreateBlockedStrided(Ceed ceed, CeedInt nelem,
    CeedInt elemsize, CeedInt blksize, CeedInt nnodes, CeedInt ncomp,
    const CeedInt strides[3], CeedElemRestriction *rstr) {
  int ierr;
  CeedInt nblk = (nelem / blksize) + !!(nelem % blksize);

  if (!ceed->ElemRestrictionCreateBlocked) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction");
    CeedChk(ierr);

    if (!delegate)
      // LCOV_EXCL_START
      return CeedError(ceed, 1, "Backend does not support "
                       "ElemRestrictionCreateBlocked");
    // LCOV_EXCL_STOP

    ierr = CeedElemRestrictionCreateBlockedStrided(delegate, nelem, elemsize,
           blksize, nnodes, ncomp, strides, rstr);
    CeedChk(ierr);
    return 0;
  }

  ierr = CeedCalloc(1, rstr); CeedChk(ierr);

  (*rstr)->ceed = ceed;
  ceed->refcount++;
  (*rstr)->refcount = 1;
  (*rstr)->nelem = nelem;
  (*rstr)->elemsize = elemsize;
  (*rstr)->nnodes = nnodes;
  (*rstr)->ncomp = ncomp;
  (*rstr)->nblk = nblk;
  (*rstr)->blksize = blksize;
  ierr = CeedMalloc(3, &(*rstr)->strides); CeedChk(ierr);
  for (int i = 0; i<3; i++)
    (*rstr)->strides[i] = strides[i];
  ierr = ceed->ElemRestrictionCreateBlocked(CEED_MEM_HOST, CEED_OWN_POINTER,
         NULL, *rstr); CeedChk(ierr);

  return 0;
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
  m = rstr->nnodes * rstr->ncomp;
  n = rstr->nblk * rstr->blksize * rstr->elemsize * rstr->ncomp;
  if (lvec) {
    ierr = CeedVectorCreate(rstr->ceed, m, lvec); CeedChk(ierr);
  }
  if (evec) {
    ierr = CeedVectorCreate(rstr->ceed, n, evec); CeedChk(ierr);
  }
  return 0;
}

/**
  @brief Restrict an L-vector to an E-vector or apply its transpose

  @param rstr    CeedElemRestriction
  @param tmode   Apply restriction or transpose
  @param u       Input vector (of shape [@a nnodes, @a ncomp] when
                   tmode=CEED_NOTRANSPOSE, imode=CEED_INTERLACED)
  @param ru      Output vector (of shape [@a nelem * @a elemsize] when
                   tmode=CEED_NOTRANSPOSE). Ordering of the e-vector is decided
                   by the backend.
  @param request Request or CEED_REQUEST_IMMEDIATE

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
    n = rstr->nnodes * rstr->ncomp;
  } else {
    m = rstr->nnodes * rstr->ncomp;
    n = rstr->nblk * rstr->blksize * rstr->elemsize * rstr->ncomp;
  }
  if (n != u->length)
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, 2, "Input vector size %d not compatible with "
                     "element restriction (%d, %d)", u->length, m, n);
  // LCOV_EXCL_STOP
  if (m != ru->length)
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, 2, "Output vector size %d not compatible with "
                     "element restriction (%d, %d)", ru->length, m, n);
  // LCOV_EXCL_STOP
  ierr = rstr->Apply(rstr, tmode, u, ru, request); CeedChk(ierr);

  return 0;
}

/**
  @brief Restrict an L-vector to a block of an E-vector or apply its transpose

  @param rstr    CeedElemRestriction
  @param block   Block number to restrict to/from, i.e. block=0 will handle
                   elements [0 : blksize] and block=3 will handle elements
                   [3*blksize : 4*blksize]
  @param tmode   Apply restriction or transpose
  @param u       Input vector (of shape [@a nnodes, @a ncomp] when
                   tmode=CEED_NOTRANSPOSE, imode=CEED_INTERLACED)
  @param ru      Output vector (of shape [@a blksize * @a elemsize] when
                   tmode=CEED_NOTRANSPOSE). Ordering of the e-vector is decided
                   by the backend.
  @param request Request or CEED_REQUEST_IMMEDIATE

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
    n = rstr->nnodes * rstr->ncomp;
  } else {
    m = rstr->nnodes * rstr->ncomp;
    n = rstr->blksize * rstr->elemsize * rstr->ncomp;
  }
  if (n != u->length)
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, 2, "Input vector size %d not compatible with "
                     "element restriction (%d, %d)", u->length, m, n);
  // LCOV_EXCL_STOP
  if (m != ru->length)
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, 2, "Output vector size %d not compatible with "
                     "element restriction (%d, %d)", ru->length, m, n);
  // LCOV_EXCL_STOP
  if (rstr->blksize*block > rstr->nelem)
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, 2, "Cannot retrieve block %d, element %d > "
                     "total elements %d", block, rstr->blksize*block,
                     rstr->nelem);
  // LCOV_EXCL_STOP
  ierr = rstr->ApplyBlock(rstr, block, tmode, u, ru, request);
  CeedChk(ierr);

  return 0;
}

/**
  @brief Get the L-vector interlaced mode of a CeedElemRestriction

  @param rstr             CeedElemRestriction
  @param[out] imode       Variable to store imode

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetIMode(CeedElemRestriction rstr,
                                CeedInterlaceMode *imode) {
  *imode = rstr->imode;
  return 0;
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
  return 0;
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
  return 0;
}

/**
  @brief Get the number of degrees of freedom in the range of a
         CeedElemRestriction

  @param rstr             CeedElemRestriction
  @param[out] numnodes    Variable to store number of nodes

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetNumNodes(CeedElemRestriction rstr,
                                   CeedInt *numnodes) {
  *numnodes = rstr->nnodes;
  return 0;
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
  return 0;
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
  return 0;
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
  return 0;
}

/**
  @brief Get the multiplicity of nodes in a CeedElemRestriction

  @param rstr             CeedElemRestriction
  @param[out] mult        Vector to store multiplicity (of size nnodes*ncomp)

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

  return 0;
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

  fprintf(stream, "%sCeedElemRestriction from (%d, %d) to %d elements with %d "
          "nodes each and %s %s\n", rstr->blksize > 1 ? "Blocked " : "",
          rstr->nnodes, rstr->ncomp, rstr->nelem, rstr->elemsize,
          rstr->strides ? "strides" : "L-vector components",
          rstr->strides ? stridesstr : CeedInterlaceModes[rstr->imode]);
  return 0;
}

/**
  @brief Destroy a CeedElemRestriction

  @param rstr  CeedElemRestriction to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionDestroy(CeedElemRestriction *rstr) {
  int ierr;

  if (!*rstr || --(*rstr)->refcount > 0)
    return 0;
  if ((*rstr)->Destroy) {
    ierr = (*rstr)->Destroy(*rstr); CeedChk(ierr);
  }
  ierr = CeedFree(&(*rstr)->strides); CeedChk(ierr);
  ierr = CeedDestroy(&(*rstr)->ceed); CeedChk(ierr);
  ierr = CeedFree(rstr); CeedChk(ierr);
  return 0;
}

/// @}
