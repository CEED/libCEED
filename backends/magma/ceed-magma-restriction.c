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

#include "ceed-magma.h"

static int CeedElemRestrictionApply_Magma(CeedElemRestriction r,
					  CeedTransposeMode tmode,
					  CeedTransposeMode lmode, CeedVector u,
					  CeedVector v, CeedRequest *request) {
    int ierr;
    CeedElemRestriction_Magma *impl;
    ierr = CeedElemRestrictionGetData(r, (void *)&impl); CeedChk(ierr);

    CeedInt nelem;
    CeedElemRestrictionGetNumElements(r, &nelem);

    CeedInt esize;
    CeedElemRestrictionGetElementSize(r, &esize);

    CeedInt nnodes;
    CeedElemRestrictionGetNumNodes(r, &nnodes);

    CeedInt NCOMP;
    CeedElemRestrictionGetNumComponents(r, &NCOMP);

    const CeedScalar *du;
    CeedScalar *dv;
    ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &du); CeedChk(ierr);
    ierr = CeedVectorGetArray(v, CEED_MEM_DEVICE, &dv); CeedChk(ierr);

    if (tmode == CEED_TRANSPOSE) { 
      if (lmode == CEED_TRANSPOSE)
          magma_writeDofsTranspose(NCOMP, nnodes, esize, nelem, impl->dindices, du, dv);
      else
          magma_writeDofs(NCOMP, nnodes, esize, nelem, impl->dindices, du, dv);
    } else {
      if (lmode == CEED_TRANSPOSE)
          magma_readDofsTranspose(NCOMP, nnodes, esize, nelem, impl->dindices, du, dv);
      else
          magma_readDofs(NCOMP, nnodes, esize, nelem, impl->dindices, du, dv);
    }
    ierr = CeedVectorRestoreArrayRead(u, &du); CeedChk(ierr);
    ierr = CeedVectorRestoreArray(v, &dv); CeedChk(ierr);

    return 0;
}

int CeedElemRestrictionApplyBlock_Magma(CeedElemRestriction r,
                                       CeedInt block, CeedTransposeMode tmode,
                                       CeedTransposeMode lmode, CeedVector u,
                                       CeedVector v, CeedRequest *request) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
    // LCOV_EXCL_START
  return CeedError(ceed, 1, "Backend does not implement blocked restrictions");
  // LCOV_EXCL_STOP
}

static int CeedElemRestrictionDestroy_Magma(CeedElemRestriction r) {
  int ierr;
  CeedElemRestriction_Magma *impl;
  ierr = CeedElemRestrictionGetData(r, (void *)&impl); CeedChk(ierr);

  // Free if we own the data
  if (impl->own_) {
    ierr = magma_free_pinned(impl->indices); CeedChk(ierr);
    ierr = magma_free(impl->dindices);       CeedChk(ierr);
  } else if (impl->down_) {
    ierr = magma_free(impl->dindices);       CeedChk(ierr);
  }
  ierr = CeedFree(&impl); CeedChk(ierr);
  return 0;
}

int CeedElemRestrictionCreate_Magma(CeedMemType mtype, CeedCopyMode cmode,
                                  const CeedInt *indices, CeedElemRestriction r) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);

  CeedInt elemsize, nelem;
  ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChk(ierr);
  CeedInt size = elemsize * nelem;

  CeedElemRestriction_Magma *impl;
  ierr = CeedCalloc(1, &impl); CeedChk(ierr);

  impl->dindices = NULL;
  impl->indices  = NULL;
  impl->own_ = 0;
  impl->down_= 0;

  if (mtype == CEED_MEM_HOST) {
    // memory is on the host; own_ = 0
    switch (cmode) {
    case CEED_COPY_VALUES:
      impl->own_ = 1;

      if (indices != NULL) {

        ierr = magma_malloc( (void**)&impl->dindices,
                             size * sizeof(CeedInt)); CeedChk(ierr);
        ierr = magma_malloc_pinned( (void**)&impl->indices,
                                    size * sizeof(CeedInt)); CeedChk(ierr);
        memcpy(impl->indices, indices, size * sizeof(CeedInt));
        // TODO: magma calls
        ierr = cudaMemcpy(impl->dindices, indices, size * sizeof(CeedInt),
                        cudaMemcpyHostToDevice);
      }
      break;
    case CEED_OWN_POINTER:
      impl->own_ = 1;

      if (indices != NULL) {
          ierr = magma_malloc( (void**)&impl->dindices,
                           size * sizeof(CeedInt)); CeedChk(ierr);
      // TODO: possible problem here is if we are passed non-pinned memory;
      //       (as we own it, lter in destroy, we use free for pinned memory).
        impl->indices = (CeedInt *)indices;
        // TODO: magma calls
        ierr = cudaMemcpy(impl->dindices, indices, size * sizeof(CeedInt),
                        cudaMemcpyHostToDevice);
      }
      break;
    case CEED_USE_POINTER:
      if (indices != NULL) {
        ierr = magma_malloc( (void**)&impl->dindices,
                             size * sizeof(CeedInt)); CeedChk(ierr);
        // TODO: magma calls
        ierr = cudaMemcpy(impl->dindices, indices, size * sizeof(CeedInt),
                          cudaMemcpyHostToDevice);
      }
      impl->down_ = 1;
        impl->indices  = (CeedInt *)indices;
    }
  } else if (mtype == CEED_MEM_DEVICE) {
    // memory is on the device; own = 0
    switch (cmode) {
    case CEED_COPY_VALUES:
      ierr = magma_malloc( (void**)&impl->dindices,
                           size * sizeof(CeedInt)); CeedChk(ierr);
      ierr = magma_malloc_pinned( (void**)&impl->indices,
                                  size * sizeof(CeedInt)); CeedChk(ierr);
      impl->own_ = 1;

      if (indices)
        ierr = cudaMemcpy(impl->dindices, indices, size * sizeof(CeedInt),
                          cudaMemcpyDeviceToDevice);
        // TODO: magma calls
      break;
    case CEED_OWN_POINTER:
      impl->dindices = (CeedInt *)indices;
      ierr = magma_malloc_pinned( (void**)&impl->indices,
                                  size * sizeof(CeedInt)); CeedChk(ierr);
      impl->own_ = 1;

      break;
    case CEED_USE_POINTER:
      impl->dindices = (CeedInt *)indices;
      impl->indices  = NULL;
    }

  } else
    return CeedError(ceed, 1, "Only MemType = HOST or DEVICE supported");

  ierr = CeedElemRestrictionSetData(r, (void *)&impl); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Apply",
                                CeedElemRestrictionApply_Magma); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "ApplyBlock",
                                CeedElemRestrictionApplyBlock_Magma);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Destroy",
                                CeedElemRestrictionDestroy_Magma); CeedChk(ierr);
  return 0;
}

int CeedElemRestrictionCreateBlocked_Magma(const CeedMemType mtype,
    const CeedCopyMode cmode,
    const CeedInt *indices,
    const CeedElemRestriction r) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
    // LCOV_EXCL_START
  return CeedError(ceed, 1, "Backend does not implement blocked restrictions");
  // LCOV_EXCL_STOP

  return 0;
}
