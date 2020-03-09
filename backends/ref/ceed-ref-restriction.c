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

//------------------------------------------------------------------------------
// Core ElemRestriction Apply Code
//------------------------------------------------------------------------------
static inline int CeedElemRestrictionApply_Ref_Core(CeedElemRestriction r,
    const CeedInt blksize, const CeedInt ncomp, CeedInt start, CeedInt stop,
    CeedTransposeMode tmode, CeedInterlaceMode imode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  int ierr;
  CeedElemRestriction_Ref *impl;
  ierr = CeedElemRestrictionGetData(r, (void *)&impl); CeedChk(ierr);
  const CeedScalar *uu;
  CeedScalar *vv;
  CeedInt nelem, elemsize, nnodes, voffset;
  ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumNodes(r, &nnodes); CeedChk(ierr);
  voffset = start*blksize*elemsize*ncomp;

  ierr = CeedVectorGetArrayRead(u, CEED_MEM_HOST, &uu); CeedChk(ierr);
  ierr = CeedVectorGetArray(v, CEED_MEM_HOST, &vv); CeedChk(ierr);
  // Restriction from L-vector to E-vector
  // Perform: v = r * u
  if (tmode == CEED_NOTRANSPOSE) {
    // No indices provided, Identity Restriction
    if (!impl->indices) {
      bool backendstrides;
      ierr = CeedElemRestrictionGetBackendStridesStatus(r, &backendstrides);
      CeedChk(ierr);
      if (backendstrides) {
        // CPU backend strides are {1, elemsize, elemsize*ncomp}
        // This if branch is left separate to allow better inlining
        for (CeedInt e = start*blksize; e < stop*blksize; e+=blksize)
          CeedPragmaSIMD
          for (CeedInt k = 0; k < ncomp; k++)
            CeedPragmaSIMD
            for (CeedInt n = 0; n < elemsize; n++)
              CeedPragmaSIMD
              for (CeedInt j = 0; j < blksize; j++)
                vv[e*elemsize*ncomp + (k*elemsize+n)*blksize + j - voffset]
                  = uu[n + k*elemsize +
                         CeedIntMin(e+j, nelem-1)*elemsize*ncomp];
      } else {
        // User provided strides
        CeedInt strides[3];
        ierr = CeedElemRestrictionGetStrides(r, &strides); CeedChk(ierr);
        for (CeedInt e = start*blksize; e < stop*blksize; e+=blksize)
          CeedPragmaSIMD
          for (CeedInt k = 0; k < ncomp; k++)
            CeedPragmaSIMD
            for (CeedInt n = 0; n < elemsize; n++)
              CeedPragmaSIMD
              for (CeedInt j = 0; j < blksize; j++)
                vv[e*elemsize*ncomp + (k*elemsize+n)*blksize + j - voffset]
                  = uu[n*strides[0] + k*strides[1] +
                                    CeedIntMin(e+j, nelem-1)*strides[2]];
      }
    } else {
      // Indices provided, standard or blocked restriction
      // vv has shape [elemsize, ncomp, nelem], row-major
      // uu has shape [nnodes, ncomp]
      for (CeedInt e = start*blksize; e < stop*blksize; e+=blksize)
        CeedPragmaSIMD
        for (CeedInt d = 0; d < ncomp; d++)
          CeedPragmaSIMD
          for (CeedInt i = 0; i < elemsize*blksize; i++)
            vv[elemsize*(d*blksize+ncomp*e) + i - voffset]
              = uu[imode == CEED_NONINTERLACED
                         ? impl->indices[i+elemsize*e]+nnodes*d
                         : d+ncomp*impl->indices[i+elemsize*e]];
    }
  } else {
    // Restriction from E-vector to L-vector
    // Performing v += r^T * u
    // No indices provided, Identity Restriction
    if (!impl->indices) {
      bool backendstrides;
      ierr = CeedElemRestrictionGetBackendStridesStatus(r, &backendstrides);
      CeedChk(ierr);
      if (backendstrides) {
        // CPU backend strides are {1, elemsize, elemsize*ncomp}
        // This if brach is left separate to allow better inlining
        for (CeedInt e = start*blksize; e < stop*blksize; e+=blksize)
          CeedPragmaSIMD
          for (CeedInt k = 0; k < ncomp; k++)
            CeedPragmaSIMD
            for (CeedInt n = 0; n < elemsize; n++)
              CeedPragmaSIMD
              for (CeedInt j = 0; j < CeedIntMin(blksize, nelem-e); j++)
                vv[n + k*elemsize + (e+j)*elemsize*ncomp]
                += uu[e*elemsize*ncomp + (k*elemsize+n)*blksize + j - voffset];
      } else {
        // User provided strides
        CeedInt strides[3];
        ierr = CeedElemRestrictionGetStrides(r, &strides); CeedChk(ierr);
        for (CeedInt e = start*blksize; e < stop*blksize; e+=blksize)
          CeedPragmaSIMD
          for (CeedInt k = 0; k < ncomp; k++)
            CeedPragmaSIMD
            for (CeedInt n = 0; n < elemsize; n++)
              CeedPragmaSIMD
              for (CeedInt j = 0; j < CeedIntMin(blksize, nelem-e); j++)
                vv[n*strides[0] + k*strides[1] + (e+j)*strides[2]]
                += uu[e*elemsize*ncomp + (k*elemsize+n)*blksize + j - voffset];
      }
    } else {
      // Indices provided, standard or blocked restriction
      // uu has shape [elemsize, ncomp, nelem]
      // vv has shape [nnodes, ncomp]
      for (CeedInt e = start*blksize; e < stop*blksize; e+=blksize)
        for (CeedInt d = 0; d < ncomp; d++)
          for (CeedInt i = 0; i < elemsize*blksize; i+=blksize)
            // Iteration bound set to discard padding elements
            for (CeedInt j = i; j < i+CeedIntMin(blksize, nelem-e); j++)
              vv[imode == CEED_NONINTERLACED
                       ? impl->indices[j+e*elemsize]+nnodes*d
                       : d+ncomp*impl->indices[j+e*elemsize]]
              += uu[elemsize*(d*blksize+ncomp*e) + j - voffset];
    }
  }
  ierr = CeedVectorRestoreArrayRead(u, &uu); CeedChk(ierr);
  ierr = CeedVectorRestoreArray(v, &vv); CeedChk(ierr);
  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED)
    *request = NULL;
  return 0;
}

//------------------------------------------------------------------------------
// ElemRestriction Apply - Common Sizes
//------------------------------------------------------------------------------
static int CeedElemRestrictionApply_Ref_11(CeedElemRestriction r,
    CeedInt start, CeedInt stop, CeedTransposeMode tmode,
    CeedInterlaceMode imode, CeedVector u, CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 1, 1, start, stop, tmode, imode,
         u, v, request);
}

static int CeedElemRestrictionApply_Ref_18(CeedElemRestriction r,
    CeedInt start, CeedInt stop, CeedTransposeMode tmode,
    CeedInterlaceMode imode, CeedVector u, CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 8, 1, start, stop, tmode, imode,
         u, v, request);

}

static int CeedElemRestrictionApply_Ref_31(CeedElemRestriction r,
    CeedInt start, CeedInt stop, CeedTransposeMode tmode,
    CeedInterlaceMode imode, CeedVector u, CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 1, 3, start, stop, tmode, imode,
         u, v, request);
}

static int CeedElemRestrictionApply_Ref_38(CeedElemRestriction r,
    CeedInt start, CeedInt stop, CeedTransposeMode tmode,
    CeedInterlaceMode imode, CeedVector u, CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 8, 3, start, stop, tmode, imode,
         u, v, request);
}

//------------------------------------------------------------------------------
// ElemRestriction Apply
//------------------------------------------------------------------------------
static int CeedElemRestrictionApply_Ref(CeedElemRestriction r,
                                        CeedTransposeMode tmode, CeedVector u,
                                        CeedVector v, CeedRequest *request) {
  int ierr;
  CeedInt numblk, ncomp, blksize;
  CeedInterlaceMode imode = CEED_NONINTERLACED;
  ierr = CeedElemRestrictionGetNumBlocks(r, &numblk); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumComponents(r, &ncomp); CeedChk(ierr);
  ierr = CeedElemRestrictionGetBlockSize(r, &blksize); CeedChk(ierr);
  CeedElemRestriction_Ref *impl;
  ierr = CeedElemRestrictionGetData(r, (void *)&impl); CeedChk(ierr);
  if (impl->indices)
    ierr = CeedElemRestrictionGetIMode(r, &imode); CeedChk(ierr);

  CeedInt idx = -1;
  if (blksize < 10)
    idx = 10*ncomp + blksize;
  switch (idx) {
  case 11:
    return CeedElemRestrictionApply_Ref_11(r, 0, numblk, tmode, imode,
                                           u, v, request);
    break;
  case 18:
    return CeedElemRestrictionApply_Ref_18(r, 0, numblk, tmode, imode,
                                           u, v, request);
    break;
  case 31:
    return CeedElemRestrictionApply_Ref_31(r, 0, numblk, tmode, imode,
                                           u, v, request);
    break;
  case 38:
    return CeedElemRestrictionApply_Ref_38(r, 0, numblk, tmode, imode,
                                           u, v, request);
    break;
  default:
    // LCOV_EXCL_START
    return CeedElemRestrictionApply_Ref_Core(r, blksize, ncomp, 0, numblk,
           tmode, imode, u, v, request);
    // LCOV_EXCL_STOP
  }
}

//------------------------------------------------------------------------------
// ElemRestriction Apply Block
//------------------------------------------------------------------------------
static int CeedElemRestrictionApplyBlock_Ref(CeedElemRestriction r,
    CeedInt block, CeedTransposeMode tmode, CeedVector u, CeedVector v,
    CeedRequest *request) {
  int ierr;
  CeedInt ncomp, blksize;
  CeedInterlaceMode imode = CEED_NONINTERLACED;
  ierr = CeedElemRestrictionGetNumComponents(r, &ncomp); CeedChk(ierr);
  ierr = CeedElemRestrictionGetBlockSize(r, &blksize); CeedChk(ierr);
  CeedElemRestriction_Ref *impl;
  ierr = CeedElemRestrictionGetData(r, (void *)&impl); CeedChk(ierr);
  if (impl->indices)
    ierr = CeedElemRestrictionGetIMode(r, &imode); CeedChk(ierr);

  CeedInt idx = -1;
  if (blksize < 10)
    idx = 10*ncomp + blksize;
  switch (idx) {
  case 11:
    return CeedElemRestrictionApply_Ref_11(r, block, block+1, tmode, imode,
                                           u, v, request);
    break;
  case 18:
    return CeedElemRestrictionApply_Ref_18(r, block, block+1, tmode, imode,
                                           u, v, request);
    break;
  case 31:
    return CeedElemRestrictionApply_Ref_31(r, block, block+1, tmode, imode,
                                           u, v, request);
    break;
  case 38:
    return CeedElemRestrictionApply_Ref_38(r, block, block+1, tmode, imode,
                                           u, v, request);
    break;
  default:
    // LCOV_EXCL_START
    return CeedElemRestrictionApply_Ref_Core(r, blksize, ncomp, block, block+1,
           tmode, imode, u, v, request);
    // LCOV_EXCL_STOP
  }
}

//------------------------------------------------------------------------------
// ElemRestriction Destroy
//------------------------------------------------------------------------------
static int CeedElemRestrictionDestroy_Ref(CeedElemRestriction r) {
  int ierr;
  CeedElemRestriction_Ref *impl;
  ierr = CeedElemRestrictionGetData(r, (void *)&impl); CeedChk(ierr);

  ierr = CeedFree(&impl->indices_allocated); CeedChk(ierr);
  ierr = CeedFree(&impl); CeedChk(ierr);
  return 0;
}

//------------------------------------------------------------------------------
// ElemRestriction Create
//------------------------------------------------------------------------------
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
    // LCOV_EXCL_START
    return CeedError(ceed, 1, "Only MemType = HOST supported");
  // LCOV_EXCL_STOP
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
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "ApplyBlock",
                                CeedElemRestrictionApplyBlock_Ref);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Destroy",
                                CeedElemRestrictionDestroy_Ref); CeedChk(ierr);
  return 0;
}
//------------------------------------------------------------------------------

