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
  Create a CeedElemRestriction

  @param ceed       A Ceed object where the CeedElemRestriction will be created.
  @param nelem      Number of elements described in the @a indices array.
  @param elemsize   Size (number of unknowns) per element.
  @param ndof       The total size of the input CeedVector to which the
                    restriction will be applied. This size may include data
                    used by other CeedElemRestriction objects describing
                    different types of elements.
  @param mtype      Memory type of the @a indices array, see CeedMemType.
  @param cmode      Copy mode for the @a indices array, see CeedCopyMode.
  @param indices    Array of dimensions @a nelem × @a elemsize) using
                    column-major storage layout. Row i holds the ordered list
                    of the indices (into the input CeedVector) for the unknowns
                    corresponding to element i, where 0 <= i < @a nelements.
                    All indices must be in the range [0, @a ndof).
  @param r          The address of the variable where the newly created
                    CeedElemRestriction will be stored.

  @return An error code: 0 - success, otherwise - failure.
*/
int CeedElemRestrictionCreate(Ceed ceed, CeedInt nelem, CeedInt elemsize,
                              CeedInt ndof, CeedMemType mtype, CeedCopyMode cmode,
                              const CeedInt *indices, CeedElemRestriction *r) {
  int ierr;

  if (!ceed->ElemRestrictionCreate)
    return CeedError(ceed, 1, "Backend does not support ElemRestrictionCreate");
  ierr = CeedCalloc(1,r); CeedChk(ierr);
  (*r)->ceed = ceed;
  (*r)->nelem = nelem;
  (*r)->elemsize = elemsize;
  (*r)->ndof = ndof;
  ierr = ceed->ElemRestrictionCreate(*r, mtype, cmode, indices); CeedChk(ierr);
  return 0;
}

/**
  Create a blocked CeedElemRestriction

  @param ceed        A Ceed object where the CeedElemRestriction will be created.
  @param nelements   Number of elements described ...
  @param esize       Size (number of unknowns) per element.
  @param blocksize   ...
  @param mtype       Memory type of the @a blkindices array, see CeedMemType.
  @param cmode       Copy mode for the @a blkindices array, see CeedCopyMode.
  @param blkindices  ...
  @param r           The address of the variable where the newly created
                     CeedElemRestriction will be stored.

  @return An error code: 0 - success, otherwise - failure.
 */
int CeedElemRestrictionCreateBlocked(Ceed ceed, CeedInt nelements,
                                     CeedInt esize, CeedInt blocksize,
                                     CeedMemType mtype, CeedCopyMode cmode,
                                     CeedInt *blkindices, CeedElemRestriction *r) {
  return CeedError(ceed, 1, "Not implemented");
}

/**
  Restrict an L-vector to an E-vector or apply transpose

  @param r Restriction
  @param tmode Apply restriction or transpose
  @param ncomp Number of components [FIXME: include in CeedElemRestriction]
  @param lmode Ordering of the ncomp components
  @param u Input vector (of size @a ndof when tmode=CEED_NOTRANSPOSE)
  @param v Output vector (of size @a nelem * @a elemsize when tmode=CEED_NOTRANSPOSE)
  @param request Request or CEED_REQUEST_IMMEDIATE
*/
int CeedElemRestrictionApply(CeedElemRestriction r, CeedTransposeMode tmode,
                             CeedInt ncomp, CeedTransposeMode lmode,
                             CeedVector u, CeedVector v, CeedRequest *request) {
  CeedInt m,n;
  int ierr;

  if (tmode == CEED_NOTRANSPOSE) {
    m = r->nelem * r->elemsize * ncomp;
    n = r->ndof * ncomp;
  } else {
    m = r->ndof * ncomp;
    n = r->nelem * r->elemsize * ncomp;
  }
  if (n != u->length)
    return CeedError(r->ceed, 2,
                     "Input vector size %d not compatible with element restriction (%d,%d)",
                     u->length, m, n);
  if (m != v->length)
    return CeedError(r->ceed, 2,
                     "Output vector size %d not compatible with element restriction (%d,%d)",
                     v->length, m, n);
  ierr = r->Apply(r, tmode, ncomp, lmode, u, v, request); CeedChk(ierr);
  return 0;
}

/**
  Destroy a CeedElemRestriction
*/
int CeedElemRestrictionDestroy(CeedElemRestriction *r) {
  int ierr;
  if (!*r) return 0;
  if ((*r)->Destroy) {
    ierr = (*r)->Destroy(*r); CeedChk(ierr);
  }
  ierr = CeedFree(r); CeedChk(ierr);
  return 0;
}
