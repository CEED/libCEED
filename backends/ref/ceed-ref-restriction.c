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

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <string.h>
#include "ceed-ref.h"

//------------------------------------------------------------------------------
// Core ElemRestriction Apply Code
//------------------------------------------------------------------------------
static inline int CeedElemRestrictionApply_Ref_Core(CeedElemRestriction r,
    const CeedInt ncomp, const CeedInt blksize, const CeedInt compstride,
    CeedInt start, CeedInt stop, CeedTransposeMode tmode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  int ierr;
  CeedElemRestriction_Ref *impl;
  ierr = CeedElemRestrictionGetData(r, &impl); CeedChkBackend(ierr);
  const CeedScalar *uu;
  CeedScalar *vv;
  CeedInt nelem, elemsize, voffset;
  ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChkBackend(ierr);
  voffset = start*blksize*elemsize*ncomp;

  ierr = CeedVectorGetArrayRead(u, CEED_MEM_HOST, &uu); CeedChkBackend(ierr);
  ierr = CeedVectorGetArray(v, CEED_MEM_HOST, &vv); CeedChkBackend(ierr);
  // Restriction from L-vector to E-vector
  // Perform: v = r * u
  if (tmode == CEED_NOTRANSPOSE) {
    // No offsets provided, Identity Restriction
    if (!impl->offsets) {
      bool backendstrides;
      ierr = CeedElemRestrictionHasBackendStrides(r, &backendstrides);
      CeedChkBackend(ierr);
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
        ierr = CeedElemRestrictionGetStrides(r, &strides); CeedChkBackend(ierr);
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
      // Offsets provided, standard or blocked restriction
      // vv has shape [elemsize, ncomp, nelem], row-major
      // uu has shape [nnodes, ncomp]
      for (CeedInt e = start*blksize; e < stop*blksize; e+=blksize)
        CeedPragmaSIMD
        for (CeedInt k = 0; k < ncomp; k++)
          CeedPragmaSIMD
          for (CeedInt i = 0; i < elemsize*blksize; i++)
            vv[elemsize*(k*blksize+ncomp*e) + i - voffset]
              = uu[impl->offsets[i+elemsize*e] + k*compstride];
    }
  } else {
    // Restriction from E-vector to L-vector
    // Performing v += r^T * u
    // No offsets provided, Identity Restriction
    if (!impl->offsets) {
      bool backendstrides;
      ierr = CeedElemRestrictionHasBackendStrides(r, &backendstrides);
      CeedChkBackend(ierr);
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
        ierr = CeedElemRestrictionGetStrides(r, &strides); CeedChkBackend(ierr);
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
      // Offsets provided, standard or blocked restriction
      // uu has shape [elemsize, ncomp, nelem]
      // vv has shape [nnodes, ncomp]
      for (CeedInt e = start*blksize; e < stop*blksize; e+=blksize)
        for (CeedInt k = 0; k < ncomp; k++)
          for (CeedInt i = 0; i < elemsize*blksize; i+=blksize)
            // Iteration bound set to discard padding elements
            for (CeedInt j = i; j < i+CeedIntMin(blksize, nelem-e); j++)
              vv[impl->offsets[j+e*elemsize] + k*compstride]
              += uu[elemsize*(k*blksize+ncomp*e) + j - voffset];
    }
  }
  ierr = CeedVectorRestoreArrayRead(u, &uu); CeedChkBackend(ierr);
  ierr = CeedVectorRestoreArray(v, &vv); CeedChkBackend(ierr);
  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED)
    *request = NULL;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// ElemRestriction Apply - Common Sizes
//------------------------------------------------------------------------------
static int CeedElemRestrictionApply_Ref_110(CeedElemRestriction r,
    const CeedInt ncomp, const CeedInt blksize, const CeedInt compstride,
    CeedInt start, CeedInt stop, CeedTransposeMode tmode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 1, 1, compstride, start, stop,
         tmode, u, v, request);
}

static int CeedElemRestrictionApply_Ref_111(CeedElemRestriction r,
    const CeedInt ncomp, const CeedInt blksize, const CeedInt compstride,
    CeedInt start, CeedInt stop, CeedTransposeMode tmode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 1, 1, 1, start, stop, tmode,
         u, v, request);
}

static int CeedElemRestrictionApply_Ref_180(CeedElemRestriction r,
    const CeedInt ncomp, const CeedInt blksize, const CeedInt compstride,
    CeedInt start, CeedInt stop, CeedTransposeMode tmode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 1, 8, compstride, start, stop,
         tmode, u, v, request);
}

static int CeedElemRestrictionApply_Ref_181(CeedElemRestriction r,
    const CeedInt ncomp, const CeedInt blksize, const CeedInt compstride,
    CeedInt start, CeedInt stop, CeedTransposeMode tmode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 1, 8, 1, start, stop, tmode,
         u, v, request);
}

static int CeedElemRestrictionApply_Ref_310(CeedElemRestriction r,
    const CeedInt ncomp, const CeedInt blksize, const CeedInt compstride,
    CeedInt start, CeedInt stop, CeedTransposeMode tmode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 3, 1, compstride, start, stop,
         tmode, u, v, request);
}

static int CeedElemRestrictionApply_Ref_311(CeedElemRestriction r,
    const CeedInt ncomp, const CeedInt blksize, const CeedInt compstride,
    CeedInt start, CeedInt stop, CeedTransposeMode tmode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 3, 1, 1, start, stop, tmode,
         u, v, request);
}

static int CeedElemRestrictionApply_Ref_380(CeedElemRestriction r,
    const CeedInt ncomp, const CeedInt blksize, const CeedInt compstride,
    CeedInt start, CeedInt stop, CeedTransposeMode tmode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 3, 8, compstride, start, stop,
         tmode, u, v, request);
}

static int CeedElemRestrictionApply_Ref_381(CeedElemRestriction r,
    const CeedInt ncomp, const CeedInt blksize, const CeedInt compstride,
    CeedInt start, CeedInt stop, CeedTransposeMode tmode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 3, 8, 1, start, stop, tmode,
         u, v, request);
}

// LCOV_EXCL_START
static int CeedElemRestrictionApply_Ref_510(CeedElemRestriction r,
    const CeedInt ncomp, const CeedInt blksize, const CeedInt compstride,
    CeedInt start, CeedInt stop, CeedTransposeMode tmode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 5, 1, compstride, start, stop,
         tmode, u, v, request);
}
// LCOV_EXCL_STOP

static int CeedElemRestrictionApply_Ref_511(CeedElemRestriction r,
    const CeedInt ncomp, const CeedInt blksize, const CeedInt compstride,
    CeedInt start, CeedInt stop, CeedTransposeMode tmode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 5, 1, 1, start, stop, tmode,
         u, v, request);
}

// LCOV_EXCL_START
static int CeedElemRestrictionApply_Ref_580(CeedElemRestriction r,
    const CeedInt ncomp, const CeedInt blksize, const CeedInt compstride,
    CeedInt start, CeedInt stop, CeedTransposeMode tmode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 5, 8, compstride, start, stop,
         tmode, u, v, request);
}
// LCOV_EXCL_STOP

static int CeedElemRestrictionApply_Ref_581(CeedElemRestriction r,
    const CeedInt ncomp, const CeedInt blksize, const CeedInt compstride,
    CeedInt start, CeedInt stop, CeedTransposeMode tmode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 5, 8, 1, start, stop, tmode,
         u, v, request);
}

//------------------------------------------------------------------------------
// ElemRestriction Apply
//------------------------------------------------------------------------------
static int CeedElemRestrictionApply_Ref(CeedElemRestriction r,
                                        CeedTransposeMode tmode, CeedVector u,
                                        CeedVector v, CeedRequest *request) {
  int ierr;
  CeedInt numblk, blksize, ncomp, compstride;
  ierr = CeedElemRestrictionGetNumBlocks(r, &numblk); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetBlockSize(r, &blksize); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetNumComponents(r, &ncomp); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetCompStride(r, &compstride); CeedChkBackend(ierr);
  CeedElemRestriction_Ref *impl;
  ierr = CeedElemRestrictionGetData(r, &impl); CeedChkBackend(ierr);

  return impl->Apply(r, ncomp, blksize, compstride, 0, numblk, tmode, u, v,
                     request);
}

//------------------------------------------------------------------------------
// ElemRestriction Apply Block
//------------------------------------------------------------------------------
static int CeedElemRestrictionApplyBlock_Ref(CeedElemRestriction r,
    CeedInt block, CeedTransposeMode tmode, CeedVector u, CeedVector v,
    CeedRequest *request) {
  int ierr;
  CeedInt blksize, ncomp, compstride;
  ierr = CeedElemRestrictionGetBlockSize(r, &blksize); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetNumComponents(r, &ncomp); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetCompStride(r, &compstride); CeedChkBackend(ierr);
  CeedElemRestriction_Ref *impl;
  ierr = CeedElemRestrictionGetData(r, &impl); CeedChkBackend(ierr);

  return impl->Apply(r, ncomp, blksize, compstride, block, block+1, tmode, u, v,
                     request);
}

//------------------------------------------------------------------------------
// ElemRestriction Get Offsets
//------------------------------------------------------------------------------
static int CeedElemRestrictionGetOffsets_Ref(CeedElemRestriction rstr,
    CeedMemType mtype, const CeedInt **offsets) {
  int ierr;
  CeedElemRestriction_Ref *impl;
  ierr = CeedElemRestrictionGetData(rstr, &impl); CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(rstr, &ceed); CeedChkBackend(ierr);

  if (mtype != CEED_MEM_HOST)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Can only provide to HOST memory");
  // LCOV_EXCL_STOP

  *offsets = impl->offsets;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// ElemRestriction Destroy
//------------------------------------------------------------------------------
static int CeedElemRestrictionDestroy_Ref(CeedElemRestriction r) {
  int ierr;
  CeedElemRestriction_Ref *impl;
  ierr = CeedElemRestrictionGetData(r, &impl); CeedChkBackend(ierr);

  ierr = CeedFree(&impl->offsets_allocated); CeedChkBackend(ierr);
  ierr = CeedFree(&impl); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// ElemRestriction Create
//------------------------------------------------------------------------------
int CeedElemRestrictionCreate_Ref(CeedMemType mtype, CeedCopyMode cmode,
                                  const CeedInt *offsets,
                                  CeedElemRestriction r) {
  int ierr;
  CeedElemRestriction_Ref *impl;
  CeedInt nelem, elemsize, numblk, blksize, ncomp, compstride;
  ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetNumBlocks(r, &numblk); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetBlockSize(r, &blksize); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetNumComponents(r, &ncomp); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetCompStride(r, &compstride); CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChkBackend(ierr);

  if (mtype != CEED_MEM_HOST)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Only MemType = HOST supported");
  // LCOV_EXCL_STOP
  ierr = CeedCalloc(1, &impl); CeedChkBackend(ierr);

  // Offsets data
  bool isStrided;
  ierr = CeedElemRestrictionIsStrided(r, &isStrided); CeedChkBackend(ierr);
  if (!isStrided) {
    // Check indices for ref or memcheck backends
    Ceed parentCeed = ceed, currCeed = NULL;
    while (parentCeed != currCeed) {
      currCeed = parentCeed;
      ierr = CeedGetParent(currCeed, &parentCeed); CeedChkBackend(ierr);
    }
    const char *resource;
    ierr = CeedGetResource(parentCeed, &resource); CeedChkBackend(ierr);
    if (!strcmp(resource, "/cpu/self/ref/serial")
        || !strcmp(resource, "/cpu/self/ref/blocked")
        || !strcmp(resource, "/cpu/self/memcheck/serial")
        || !strcmp(resource, "/cpu/self/memcheck/blocked")) {
      CeedInt lsize;
      ierr = CeedElemRestrictionGetLVectorSize(r, &lsize); CeedChkBackend(ierr);

      for (CeedInt i = 0; i < nelem*elemsize; i++)
        if (offsets[i] < 0 || lsize <= offsets[i] + (ncomp - 1) * compstride)
          // LCOV_EXCL_START
          return CeedError(ceed, CEED_ERROR_BACKEND,
                           "Restriction offset %d (%d) out of range "
                           "[0, %d]", i, offsets[i], lsize);
      // LCOV_EXCL_STOP
    }

    // Copy data
    switch (cmode) {
    case CEED_COPY_VALUES:
      ierr = CeedMalloc(nelem*elemsize, &impl->offsets_allocated);
      CeedChkBackend(ierr);
      memcpy(impl->offsets_allocated, offsets,
             nelem * elemsize * sizeof(offsets[0]));
      impl->offsets = impl->offsets_allocated;
      break;
    case CEED_OWN_POINTER:
      impl->offsets_allocated = (CeedInt *)offsets;
      impl->offsets = impl->offsets_allocated;
      break;
    case CEED_USE_POINTER:
      impl->offsets = offsets;
    }
  }

  ierr = CeedElemRestrictionSetData(r, impl); CeedChkBackend(ierr);
  CeedInt layout[3] = {1, elemsize, elemsize*ncomp};
  ierr = CeedElemRestrictionSetELayout(r, layout); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Apply",
                                CeedElemRestrictionApply_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "ApplyBlock",
                                CeedElemRestrictionApplyBlock_Ref);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "GetOffsets",
                                CeedElemRestrictionGetOffsets_Ref);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Destroy",
                                CeedElemRestrictionDestroy_Ref); CeedChkBackend(ierr);

  // Set apply function based upon ncomp, blksize, and compstride
  CeedInt idx = -1;
  if (blksize < 10)
    idx = 100*ncomp + 10*blksize + (compstride == 1);
  switch (idx) {
  case 110:
    impl->Apply = CeedElemRestrictionApply_Ref_110;
    break;
  case 111:
    impl->Apply = CeedElemRestrictionApply_Ref_111;
    break;
  case 180:
    impl->Apply = CeedElemRestrictionApply_Ref_180;
    break;
  case 181:
    impl->Apply = CeedElemRestrictionApply_Ref_181;
    break;
  case 310:
    impl->Apply = CeedElemRestrictionApply_Ref_310;
    break;
  case 311:
    impl->Apply = CeedElemRestrictionApply_Ref_311;
    break;
  case 380:
    impl->Apply = CeedElemRestrictionApply_Ref_380;
    break;
  case 381:
    impl->Apply = CeedElemRestrictionApply_Ref_381;
    break;
  // LCOV_EXCL_START
  case 510:
    impl->Apply = CeedElemRestrictionApply_Ref_510;
    break;
  // LCOV_EXCL_STOP
  case 511:
    impl->Apply = CeedElemRestrictionApply_Ref_511;
    break;
  // LCOV_EXCL_START
  case 580:
    impl->Apply = CeedElemRestrictionApply_Ref_580;
    break;
  // LCOV_EXCL_STOP
  case 581:
    impl->Apply = CeedElemRestrictionApply_Ref_581;
    break;
  default:
    impl->Apply = CeedElemRestrictionApply_Ref_Core;
    break;
  }

  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
