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

/// @file
/// Implementation of public CeedElemRestriction interfaces
///
/// @defgroup CeedElemRestriction CeedElemRestriction: restriction from vectors to elements
/// @{

/**
  @brief Create a CeedElemRestriction

  @param ceed       A Ceed object where the CeedElemRestriction will be created
  @param nelem      Number of elements described in the @a indices array
  @param elemsize   Size (number of "nodes") per element
  @param ndof       The total size of the input CeedVector to which the
                      restriction will be applied. This size may include data
                      used by other CeedElemRestriction objects describing
                      different types of elements.
  @param ncomp      Number of field components per interpolation node
  @param mtype      Memory type of the @a indices array, see CeedMemType
  @param cmode      Copy mode for the @a indices array, see CeedCopyMode
  @param indices    Array of dimensions @a nelem × @a elemsize) using
                      column-major storage layout. Row i holds the ordered list
                      of the indices (into the input CeedVector) for the unknowns
                      corresponding to element i, where 0 <= i < @a nelements.
                      All indices must be in the range [0, @a ndof).
  @param[out] r     Address of the variable where the newly created
                      CeedElemRestriction will be stored

  @return An error code: 0 - success, otherwise - failure
**/
int CeedElemRestrictionCreate(Ceed ceed, CeedInt nelem, CeedInt elemsize,
                              CeedInt ndof, CeedInt ncomp, CeedMemType mtype,
                              CeedCopyMode cmode, const CeedInt *indices,
                              CeedElemRestriction *r) {
  int ierr;

  if (!ceed->ElemRestrictionCreate)
    return CeedError(ceed, 1, "Backend does not support ElemRestrictionCreate");
  ierr = CeedCalloc(1, r); CeedChk(ierr);
  (*r)->ceed = ceed;
  ceed->refcount++;
  (*r)->refcount = 1;
  (*r)->nelem = nelem;
  (*r)->elemsize = elemsize;
  (*r)->ndof = ndof;
  (*r)->ncomp = ncomp;
  (*r)->nblk = nelem;
  (*r)->blksize = 1;
  ierr = ceed->ElemRestrictionCreate(*r, mtype, cmode, indices); CeedChk(ierr);
  return 0;
}

/**
  @brief Create an identity CeedElemRestriction

  @param ceed       A Ceed object where the CeedElemRestriction will be created
  @param nelem      Number of elements described in the @a indices array
  @param elemsize   Size (number of "nodes") per element
  @param ndof       The total size of the input CeedVector to which the
                      restriction will be applied. This size may include data
                      used by other CeedElemRestriction objects describing
                      different types of elements
  @param ncomp      Number of field components per interpolation node
  @param r          Address of the variable where the newly created
                      CeedElemRestriction will be stored

  @return An error code: 0 - success, otherwise - failure
**/
int CeedElemRestrictionCreateIdentity(Ceed ceed, CeedInt nelem, CeedInt elemsize,
                              CeedInt ndof, CeedInt ncomp, CeedElemRestriction *r) {
  int ierr;

  if (!ceed->ElemRestrictionCreate)
    return CeedError(ceed, 1, "Backend does not support ElemRestrictionCreateIdentity");
  ierr = CeedCalloc(1, r); CeedChk(ierr);
  (*r)->ceed = ceed;
  ceed->refcount++;
  (*r)->refcount = 1;
  (*r)->nelem = nelem;
  (*r)->elemsize = elemsize;
  (*r)->ndof = ndof;
  (*r)->ncomp = ncomp;
  (*r)->nblk = nelem;
  (*r)->blksize = 1;
  ierr = ceed->ElemRestrictionCreate(*r, CEED_MEM_HOST, CEED_OWN_POINTER, NULL); CeedChk(ierr);
  return 0;
}

/**
  @brief Permute and pad indices for a blocked restriction

  @param indices    Array of dimensions @a nelem × @a elemsize) using
                      row-major storage layout. Row i holds the ordered list
                      of the indices (into the input CeedVector) for the unknowns
                      corresponding to element i, where 0 <= i < @a nelements.
                      All indices must be in the range [0, @a ndof).
  @param blkindices Array of permuted and padded indicies size
                      @a nblk × @a blksize × @a elemsize using row-major ordering.
  @param nblk       Number of blocks
  @param nelem      Number of elements
  @param blksize    Number of elements in a block
  @param elemsize   Size of each element

  @return An error code: 0 - success, otherwise - failure

**/
void CeedPermutePadIndices(const CeedInt *indices, CeedInt *blkindices,
                           CeedInt nblk, CeedInt nelem,
                           CeedInt blksize, CeedInt elemsize) {
  for (CeedInt e = 0; e < nblk*blksize; e+=blksize)
    for (int j = 0; j < blksize; j++)
      for (int k = 0; k < elemsize; k++)
        blkindices[e*elemsize + k*blksize + j]
          = indices[CeedIntMin(e+j,nelem-1)*elemsize + k];
}

/**
  @brief Create a blocked CeedElemRestriction, typically only called by backends

  @param ceed       A Ceed object where the CeedElemRestriction will be created.
  @param nelem      Number of elements described in the @a indices array.
  @param elemsize   Size (number of unknowns) per element
  @param blksize    Number of elements in a block
  @param ndof       The total size of the input CeedVector to which the
                      restriction will be applied. This size may include data
                      used by other CeedElemRestriction objects describing
                      different types of elements.
  @param ncomp      Number of components stored at each node
  @param mtype      Memory type of the @a indices array, see CeedMemType
  @param cmode      Copy mode for the @a indices array, see CeedCopyMode
  @param indices    Array of dimensions @a nelem × @a elemsize) using
                      column-major storage layout. Row i holds the ordered list
                      of the indices (into the input CeedVector) for the unknowns
                      corresponding to element i, where 0 <= i < @a nelements.
                      All indices must be in the range [0, @a ndof). The
                      backend will permute and pad this array to the desired
                      ordering for the blocksize, which is typically given by the
                      backend. The default reordering is to interlace elements.
  @param r          Address of the variable where the newly created
                      CeedElemRestriction will be stored

  @return An error code: 0 - success, otherwise - failure
 **/
int CeedElemRestrictionCreateBlocked(Ceed ceed, CeedInt nelem, CeedInt elemsize,
                                     CeedInt blksize, CeedInt ndof, CeedInt ncomp,
                                     CeedMemType mtype, CeedCopyMode cmode,
                                     const CeedInt *indices, CeedElemRestriction *r) {
  int ierr;
  CeedInt *blkindices;
  CeedInt nblk = (nelem / blksize) + !!(nelem % blksize);

  if (!ceed->ElemRestrictionCreateBlocked)
    return CeedError(ceed, 1,
                     "Backend does not support ElemRestrictionCreateBlocked");
  if (mtype != CEED_MEM_HOST)
    return CeedError(ceed, 1, "Only MemType = HOST supported");

  ierr = CeedCalloc(1, r); CeedChk(ierr);

  if (indices) {
    ierr = CeedCalloc(nblk*blksize*elemsize, &blkindices);
    CeedPermutePadIndices(indices, blkindices, nblk, nelem, blksize, elemsize);
  } else {
    blkindices = NULL;
  }  

  (*r)->ceed = ceed;
  ceed->refcount++;
  (*r)->refcount = 1;
  (*r)->nelem = nelem;
  (*r)->elemsize = elemsize;
  (*r)->ndof = ndof;
  (*r)->ncomp = ncomp;
  (*r)->nblk = nblk;
  (*r)->blksize = blksize;
  ierr = ceed->ElemRestrictionCreateBlocked(*r, CEED_MEM_HOST, CEED_OWN_POINTER,
         (const CeedInt *) blkindices);
  CeedChk(ierr);

  if (cmode == CEED_OWN_POINTER)
    ierr = CeedFree(&indices); CeedChk(ierr);

  return 0;
}

/**
  @brief Create CeedVectors associated with a CeedElemRestriction

  @param r     CeedElemRestriction
  @param lvec  The address of the L-vector to be created, or NULL 
  @param evec  The address of the E-vector to be created, or NULL

  @return An error code: 0 - success, otherwise - failure
**/
int CeedElemRestrictionCreateVector(CeedElemRestriction r, CeedVector *lvec,
                                    CeedVector *evec) {
  int ierr;
  CeedInt n, m;
  m = r->ndof * r->ncomp;
  n = r->nblk * r->blksize * r->elemsize * r->ncomp;
  if (lvec) {
    ierr = CeedVectorCreate(r->ceed, m, lvec); CeedChk(ierr);
  }
  if (evec) {
    ierr = CeedVectorCreate(r->ceed, n, evec); CeedChk(ierr);
  }
  return 0;
}

/**
  @brief Restrict an L-vector to an E-vector or apply transpose

  @param r       CeedElemRestriction
  @param tmode   Apply restriction or transpose
  @param lmode   Ordering of the ncomp components
  @param u       Input vector (of size @a ndof when tmode=CEED_NOTRANSPOSE)
  @param v       Output vector (of size @a nelem * @a elemsize when tmode=CEED_NOTRANSPOSE)
  @param request Request or CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure
**/
int CeedElemRestrictionApply(CeedElemRestriction r, CeedTransposeMode tmode,
                             CeedTransposeMode lmode,
                             CeedVector u, CeedVector v, CeedRequest *request) {
  CeedInt m,n;
  int ierr;

  if (tmode == CEED_NOTRANSPOSE) {
    m = r->nblk * r->blksize * r->elemsize * r->ncomp;
    n = r->ndof * r->ncomp;
  } else {
    m = r->ndof * r->ncomp;
    n = r->nblk * r->blksize * r->elemsize * r->ncomp;
  }
  if (n != u->length)
    return CeedError(r->ceed, 2,
                     "Input vector size %d not compatible with element restriction (%d, %d)",
                     u->length, m, n);
  if (m != v->length)
    return CeedError(r->ceed, 2,
                     "Output vector size %d not compatible with element restriction (%d, %d)",
                     v->length, m, n);
  ierr = r->Apply(r, tmode, lmode, u, v, request); CeedChk(ierr);

  return 0;
}

/**
  @brief Get the total number of elements in the range of a CeedElemRestriction

  @param r                CeedElemRestriction
  @param[out] numelements Number of elements

  @return An error code: 0 - success, otherwise - failure
**/
int CeedElemRestrictionGetNumElements(CeedElemRestriction r,
                                      CeedInt *numelements) {
  *numelements = r->nelem;
  return 0;
}

/**
  @brief Destroy a CeedElemRestriction

  @param r CeedElemRestriction to destroy

  @return An error code: 0 - success, otherwise - failure
**/
int CeedElemRestrictionDestroy(CeedElemRestriction *r) {
  int ierr;

  if (!*r || --(*r)->refcount > 0) return 0;
  if ((*r)->Destroy) {
    ierr = (*r)->Destroy(*r); CeedChk(ierr);
  }
  ierr = CeedDestroy(&(*r)->ceed); CeedChk(ierr);
  ierr = CeedFree(r); CeedChk(ierr);
  return 0;
}

/// @}
