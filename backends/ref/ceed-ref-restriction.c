// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
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
#include <string.h>
#include "ceed-ref.h"

static int CeedElemRestrictionApply_Ref(CeedElemRestriction r,
                                        CeedTransposeMode tmode,
                                        CeedTransposeMode lmode, CeedVector u,
                                        CeedVector v, CeedRequest *request) {
  CeedElemRestriction_Ref *impl = r->data;
  int ierr;
  const CeedScalar *uu;
  CeedScalar *vv;
  CeedInt nblk = r->nblk, blksize = r->blksize, elemsize = r->elemsize,
          ncomp=r->ncomp;

  ierr = CeedVectorGetArrayRead(u, CEED_MEM_HOST, &uu); CeedChk(ierr);
  ierr = CeedVectorGetArray(v, CEED_MEM_HOST, &vv); CeedChk(ierr);
  if (tmode == CEED_NOTRANSPOSE) {
    // Perform: v = r * u
    if (!impl->indices) {
      for (CeedInt e = 0; e < nblk*blksize; e+=blksize)
        for (CeedInt j = 0; j < blksize; j++)
          for (CeedInt k = 0; k < ncomp*elemsize; k++)
            vv[e*elemsize*ncomp + k*blksize + j] =
              uu[CeedIntMin(e+j,r->nelem-1)*ncomp*elemsize + k];
    } else {
      // vv is (elemsize x ncomp x nelem), column-major
      if (lmode == CEED_NOTRANSPOSE) { // u is (ndof x ncomp), column-major
        for (CeedInt e = 0; e < nblk*blksize; e++)
          for (CeedInt d = 0; d < ncomp; d++)
            for (CeedInt i = 0; i < r->elemsize; i++)
              vv[i+r->elemsize*(d+ncomp*e)] =
                uu[impl->indices[i+r->elemsize*e]+r->ndof*d];
      } else { // u is (ncomp x ndof), column-major
        for (CeedInt e = 0; e < nblk*blksize; e++)
          for (CeedInt d = 0; d < ncomp; d++)
            for (CeedInt i = 0; i < r->elemsize; i++)
              vv[i+r->elemsize*(d+ncomp*e)] =
                uu[d+ncomp*impl->indices[i+r->elemsize*e]];
      }
    }
  } else {
    // Note: in transpose mode, we perform: v += r^t * u
    if (!impl->indices) {
      for (CeedInt e = 0; e < nblk*blksize; e+=blksize) {
        CeedInt maxj = ((e<(nblk-1)*blksize)
                        ||!(r->nelem%blksize))?blksize:r->nelem%blksize;
        for (CeedInt j = 0; j < maxj; j++)
          for (CeedInt k = 0; k < ncomp*elemsize; k++)
            vv[(e+j)*ncomp*elemsize + k] += uu[e*elemsize*ncomp + k*blksize + j];
      }
    } else {
      // u is (elemsize x ncomp x nelem)
      if (lmode == CEED_NOTRANSPOSE) { // vv is (ndof x ncomp), column-major
        for (CeedInt e = 0; e < nblk; e++) {
          CeedInt nblkelems = ((e==nblk-1)&&(r->nelem%blksize))?r->nelem%blksize:blksize;
          for (CeedInt d = 0; d < ncomp; d++)
            for (CeedInt i = 0; i < elemsize*blksize; i++)
              if ((i%blksize)<nblkelems) vv[impl->indices[i+e*blksize*elemsize]+r->ndof*d] +=
                  uu[i+elemsize*(d+e*blksize*ncomp)];
        }
      } else { // vv is (ncomp x ndof), column-major
        for (CeedInt e = 0; e < nblk; e++) {
          CeedInt nblkelems = ((e==nblk-1)&&(r->nelem%blksize))?r->nelem%blksize:blksize;
          for (CeedInt d = 0; d < ncomp; d++)
            for (CeedInt i = 0; i < elemsize*blksize; i++)
              if ((i%blksize)<nblkelems)  vv[d+ncomp*impl->indices[i+e*blksize*elemsize]] +=
                  uu[i+r->elemsize*(d+e*blksize*ncomp)];
        }
      }
    }
  }
  ierr = CeedVectorRestoreArrayRead(u, &uu); CeedChk(ierr);
  ierr = CeedVectorRestoreArray(v, &vv); CeedChk(ierr);
  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED)
    *request = NULL;
  return 0;
}

static int CeedElemRestrictionDestroy_Ref(CeedElemRestriction r) {
  CeedElemRestriction_Ref *impl = r->data;
  int ierr;

  ierr = CeedFree(&impl->indices_allocated); CeedChk(ierr);
  ierr = CeedFree(&r->data); CeedChk(ierr);
  return 0;
}

int CeedElemRestrictionCreate_Ref(CeedElemRestriction r,
                                  CeedMemType mtype,
                                  CeedCopyMode cmode, const CeedInt *indices) {
  int ierr;
  CeedElemRestriction_Ref *impl;

  if (mtype != CEED_MEM_HOST)
    return CeedError(r->ceed, 1, "Only MemType = HOST supported");
  ierr = CeedCalloc(1,&impl); CeedChk(ierr);
  switch (cmode) {
  case CEED_COPY_VALUES:
    ierr = CeedMalloc(r->nelem*r->elemsize, &impl->indices_allocated);
    CeedChk(ierr);
    memcpy(impl->indices_allocated, indices,
           r->nelem * r->elemsize * sizeof(indices[0]));
    impl->indices = impl->indices_allocated;
    break;
  case CEED_OWN_POINTER:
    impl->indices_allocated = (CeedInt *)indices;
    impl->indices = impl->indices_allocated;
    break;
  case CEED_USE_POINTER:
    impl->indices = indices;
  }
  r->data = impl;
  r->Apply = CeedElemRestrictionApply_Ref;
  r->Destroy = CeedElemRestrictionDestroy_Ref;
  return 0;
}
