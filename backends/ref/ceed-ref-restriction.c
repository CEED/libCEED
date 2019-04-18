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

#include "ceed-ref.h"

static int CeedElemRestrictionApply_Ref(CeedElemRestriction r,
                                        CeedTransposeMode tmode,
                                        CeedTransposeMode lmode, CeedVector u,
                                        CeedVector v, CeedRequest *request) {
  int ierr;
  CeedElemRestriction_Ref *impl;
  ierr = CeedElemRestrictionGetData(r, (void *)&impl); CeedChk(ierr);;
  const CeedScalar *uu;
  CeedScalar *vv;
  CeedInt nblk, blksize, nelem, elemsize, ndof, ncomp;
  ierr = CeedElemRestrictionGetNumBlocks(r, &nblk); CeedChk(ierr);
  ierr = CeedElemRestrictionGetBlockSize(r, &blksize); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumDoF(r, &ndof); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumComponents(r, &ncomp); CeedChk(ierr);

  ierr = CeedVectorGetArrayRead(u, CEED_MEM_HOST, &uu); CeedChk(ierr);
  ierr = CeedVectorGetArray(v, CEED_MEM_HOST, &vv); CeedChk(ierr);
  // Restriction from lvector to evector
  // Perform: v = r * u
  if (tmode == CEED_NOTRANSPOSE) {
    // No indicies provided, Identity Restriction
    if (!impl->indices) {
      for (CeedInt e = 0; e < nblk*blksize; e+=blksize)
        for (CeedInt j = 0; j < blksize; j++)
          for (CeedInt k = 0; k < ncomp*elemsize; k++)
            vv[e*elemsize*ncomp + k*blksize + j]
              = uu[CeedIntMin(e+j,nelem-1)*ncomp*elemsize + k];
    } else {
      // Indicies provided, standard or blocked restriction
      // vv has shape [elemsize, ncomp, nelem], row-major
      // uu has shape [ndof, ncomp]
      for (CeedInt e = 0; e < nblk*blksize; e+=blksize)
        for (CeedInt d = 0; d < ncomp; d++)
          for (CeedInt i = 0; i < elemsize*blksize; i++)
            vv[i+elemsize*(d*blksize+ncomp*e)]
              = uu[lmode == CEED_NOTRANSPOSE
                         ? impl->indices[i+elemsize*e]+ndof*d
                         : d+ncomp*impl->indices[i+elemsize*e]];
    }
  } else {
    // Restriction from evector to lvector
    // Performing v += r^T * u
    // No indicies provided, Identity Restriction
    if (!impl->indices) {
      for (CeedInt e = 0; e < nblk*blksize; e+=blksize)
        for (CeedInt j = 0; j < CeedIntMin(blksize, nelem-e); j++)
          for (CeedInt k = 0; k < ncomp*elemsize; k++)
            vv[(e+j)*ncomp*elemsize + k] += uu[e*elemsize*ncomp + k*blksize + j];
    } else {
      // Indicies provided, standard or blocked restriction
      // uu has shape [elemsize, ncomp, nelem]
      // vv has shape [ndof, ncomp]
      for (CeedInt e = 0; e < nblk*blksize; e+=blksize) {
        for (CeedInt d = 0; d < ncomp; d++)
          for (CeedInt i = 0; i < elemsize*blksize; i+=blksize)
            // Iteration bound set to discard padding elements
            for (CeedInt j = i; j < i+CeedIntMin(blksize, nelem-e); j++)
              vv[lmode == CEED_NOTRANSPOSE
                       ? impl->indices[j+e*elemsize]+ndof*d
                       : d+ncomp*impl->indices[j+e*elemsize]]
              += uu[j+elemsize*(d*blksize+ncomp*e)];
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
  int ierr;
  CeedElemRestriction_Ref *impl;
  ierr = CeedElemRestrictionGetData(r, (void *)&impl); CeedChk(ierr);

  ierr = CeedFree(&impl->indices_allocated); CeedChk(ierr);
  ierr = CeedFree(&impl); CeedChk(ierr);
  return 0;
}

int CeedElemRestrictionCreate_Ref(CeedMemType mtype, CeedCopyMode cmode,
                                  const CeedInt *indices, CeedElemRestriction r) {
  int ierr;
  CeedElemRestriction_Ref *impl;
  CeedInt elemsize, nelem;
  ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChk(ierr);
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);

  if (mtype != CEED_MEM_HOST)
    return CeedError(ceed, 1, "Only MemType = HOST supported");
  ierr = CeedCalloc(1,&impl); CeedChk(ierr);
  switch (cmode) {
  case CEED_COPY_VALUES:
    ierr = CeedMalloc(nelem*elemsize, &impl->indices_allocated);
    CeedChk(ierr);
    memcpy(impl->indices_allocated, indices,
           nelem * elemsize * sizeof(indices[0]));
    impl->indices = impl->indices_allocated;
    break;
  case CEED_OWN_POINTER:
    impl->indices_allocated = (CeedInt *)indices;
    impl->indices = impl->indices_allocated;
    break;
  case CEED_USE_POINTER:
    impl->indices = indices;
  }

  ierr = CeedElemRestrictionSetData(r, (void *)&impl); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Apply",
                                CeedElemRestrictionApply_Ref); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Destroy",
                                CeedElemRestrictionDestroy_Ref); CeedChk(ierr);
  return 0;
}
