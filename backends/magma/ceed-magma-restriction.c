// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>
#include <string.h>
#include "ceed-magma.h"
#ifdef CEED_MAGMA_USE_HIP
#include "../hip/ceed-hip-common.h"
#include "../hip/ceed-hip-compile.h"
#else
#include "../cuda/ceed-cuda-common.h"
#include "../cuda/ceed-cuda-compile.h"
#endif

static int CeedElemRestrictionApply_Magma(CeedElemRestriction r,
    CeedTransposeMode tmode, CeedVector u, CeedVector v, CeedRequest *request) {

  int ierr;
  CeedElemRestriction_Magma *impl;
  ierr = CeedElemRestrictionGetData(r, &impl); CeedChkBackend(ierr);

  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChkBackend(ierr);

  Ceed_Magma *data;
  ierr = CeedGetData(ceed, &data); CeedChkBackend(ierr);

  CeedInt nelem;
  CeedElemRestrictionGetNumElements(r, &nelem);

  CeedInt esize;
  CeedElemRestrictionGetElementSize(r, &esize);

  CeedInt ncomp;
  CeedElemRestrictionGetNumComponents(r, &ncomp);

  const CeedScalar *du;
  CeedScalar *dv;

  ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &du); CeedChkBackend(ierr);
  if (tmode == CEED_TRANSPOSE) {
    // Sum into for transpose mode, e-vec to l-vec
    ierr = CeedVectorGetArray(v, CEED_MEM_DEVICE, &dv); CeedChkBackend(ierr);
  } else {
    // Overwrite for notranspose mode, l-vec to e-vec
    ierr = CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &dv); CeedChkBackend(ierr);
  }

  bool isStrided;
  ierr = CeedElemRestrictionIsStrided(r, &isStrided); CeedChkBackend(ierr);

  if (isStrided) {  // Strided Restriction

    CeedInt strides[3];
    CeedInt *dstrides;
    ierr = magma_malloc( (void **)&dstrides,
                         3 * sizeof(CeedInt)); CeedChkBackend(ierr);

    // Check to see if we should use magma Q-/E-Vector layout
    //  (dimension = slowest index, then component, then element,
    //    then node)
    bool backendstrides;
    ierr = CeedElemRestrictionHasBackendStrides(r, &backendstrides);
    CeedChkBackend(ierr);

    if (backendstrides) {

      strides[0] = 1;             // node stride
      strides[1] = esize * nelem; //component stride
      strides[2] = esize;         //element stride
      magma_setvector(3, sizeof(CeedInt), strides, 1, dstrides, 1, data->queue);

    } else {

      // Get the new strides
      ierr = CeedElemRestrictionGetStrides(r, &strides); CeedChkBackend(ierr);
      magma_setvector(3, sizeof(CeedInt), strides, 1, dstrides, 1, data->queue);
    }

    void *args[] = {&ncomp, &esize, &nelem, &dstrides, &du, &dv};
    CeedInt grid = nelem;
    CeedInt blocksize = 256;
    // Perform strided restriction with dstrides
    if (tmode == CEED_TRANSPOSE) {
      ierr = MAGMA_RTC_RUN_KERNEL(ceed, impl->StridedTranspose,
                                  grid, blocksize, args); CeedChkBackend(ierr);
    } else {
      ierr = MAGMA_RTC_RUN_KERNEL(ceed, impl->StridedNoTranspose,
                                  grid, blocksize, args); CeedChkBackend(ierr);
    }

    ierr = magma_free(dstrides);  CeedChkBackend(ierr);

  } else { // Offsets array provided, standard restriction

    CeedInt compstride;
    ierr = CeedElemRestrictionGetCompStride(r, &compstride); CeedChkBackend(ierr);
    void *args[] = {&ncomp, &compstride, &esize, &nelem, &impl->doffsets, &du, &dv};
    CeedInt grid = nelem;
    CeedInt blocksize = 256;

    if (tmode == CEED_TRANSPOSE) {
      ierr = MAGMA_RTC_RUN_KERNEL(ceed, impl->OffsetTranspose,
                                  grid, blocksize, args); CeedChkBackend(ierr);
    } else {
      ierr = MAGMA_RTC_RUN_KERNEL(ceed, impl->OffsetNoTranspose,
                                  grid, blocksize, args); CeedChkBackend(ierr);
    }

  }

  ierr = CeedVectorRestoreArrayRead(u, &du); CeedChkBackend(ierr);
  ierr = CeedVectorRestoreArray(v, &dv); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

int CeedElemRestrictionApplyBlock_Magma(CeedElemRestriction r, CeedInt block,
                                        CeedTransposeMode tmode, CeedVector u,
                                        CeedVector v, CeedRequest *request) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChkBackend(ierr);
  // LCOV_EXCL_START
  return CeedError(ceed, CEED_ERROR_BACKEND,
                   "Backend does not implement blocked restrictions");
  // LCOV_EXCL_STOP
}

static int CeedElemRestrictionGetOffsets_Magma(CeedElemRestriction rstr,
    CeedMemType mtype, const CeedInt **offsets) {
  int ierr;
  CeedElemRestriction_Magma *impl;
  ierr = CeedElemRestrictionGetData(rstr, &impl); CeedChkBackend(ierr);

  switch (mtype) {
  case CEED_MEM_HOST:
    *offsets = impl->offsets;
    break;
  case CEED_MEM_DEVICE:
    *offsets = impl->doffsets;
    break;
  }
  return CEED_ERROR_SUCCESS;
}

static int CeedElemRestrictionDestroy_Magma(CeedElemRestriction r) {
  int ierr;
  CeedElemRestriction_Magma *impl;
  ierr = CeedElemRestrictionGetData(r, &impl); CeedChkBackend(ierr);

  // Free if we own the data
  if (impl->own_) {
    if (impl->own_ == OWNED_PINNED) {
      ierr = magma_free_pinned(impl->offsets); CeedChkBackend(ierr);
    } else if (impl->own_ == OWNED_UNPINNED) {
      free(impl->offsets);
    }
    ierr = magma_free(impl->doffsets);       CeedChkBackend(ierr);
  } else if (impl->down_) {
    ierr = magma_free(impl->doffsets);       CeedChkBackend(ierr);
  }
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChkBackend(ierr);
  #ifdef CEED_MAGMA_USE_HIP
  ierr = hipModuleUnload(impl->module); CeedChk_Hip(ceed, ierr);
  #else
  ierr = cuModuleUnload(impl->module); CeedChk_Cu(ceed, ierr);
  #endif
  ierr = CeedFree(&impl); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

int CeedElemRestrictionCreate_Magma(CeedMemType mtype, CeedCopyMode cmode,
                                    const CeedInt *offsets, CeedElemRestriction r) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChkBackend(ierr);

  Ceed_Magma *data;
  ierr = CeedGetData(ceed, &data); CeedChkBackend(ierr);

  CeedInt elemsize, nelem;
  ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChkBackend(ierr);
  CeedInt size = elemsize * nelem;

  CeedElemRestriction_Magma *impl;
  ierr = CeedCalloc(1, &impl); CeedChkBackend(ierr);

  impl->doffsets = NULL;
  impl->offsets  = NULL;
  impl->own_ = OWNED_NONE;
  impl->down_= 0;

  if (mtype == CEED_MEM_HOST) {
    // memory is on the host; own_ = 0
    switch (cmode) {
    case CEED_COPY_VALUES:
      impl->own_ = OWNED_PINNED;

      if (offsets != NULL) {

        ierr = magma_malloc( (void **)&impl->doffsets,
                             size * sizeof(CeedInt)); CeedChkBackend(ierr);
        ierr = magma_malloc_pinned( (void **)&impl->offsets,
                                    size * sizeof(CeedInt)); CeedChkBackend(ierr);
        memcpy(impl->offsets, offsets, size * sizeof(CeedInt));

        magma_setvector(size, sizeof(CeedInt), offsets, 1, impl->doffsets, 1,
                        data->queue);
      }
      break;
    case CEED_OWN_POINTER:
      impl->own_ = OWNED_UNPINNED;

      if (offsets != NULL) {
        ierr = magma_malloc( (void **)&impl->doffsets,
                             size * sizeof(CeedInt)); CeedChkBackend(ierr);
        impl->offsets = (CeedInt *)offsets;

        magma_setvector(size, sizeof(CeedInt), offsets, 1, impl->doffsets, 1,
                        data->queue);
      }
      break;
    case CEED_USE_POINTER:
      if (offsets != NULL) {
        ierr = magma_malloc( (void **)&impl->doffsets,
                             size * sizeof(CeedInt)); CeedChkBackend(ierr);
        magma_setvector(size, sizeof(CeedInt), offsets, 1, impl->doffsets, 1,
                        data->queue);
      }
      impl->down_ = 1;
      impl->offsets  = (CeedInt *)offsets;
    }
  } else if (mtype == CEED_MEM_DEVICE) {
    // memory is on the device; own = 0
    switch (cmode) {
    case CEED_COPY_VALUES:
      ierr = magma_malloc( (void **)&impl->doffsets,
                           size * sizeof(CeedInt)); CeedChkBackend(ierr);
      ierr = magma_malloc_pinned( (void **)&impl->offsets,
                                  size * sizeof(CeedInt)); CeedChkBackend(ierr);
      impl->own_ = OWNED_PINNED;

      if (offsets)
        magma_getvector(size, sizeof(CeedInt), impl->doffsets, 1, (void *)offsets, 1,
                        data->queue);
      break;
    case CEED_OWN_POINTER:
      impl->doffsets = (CeedInt *)offsets;
      ierr = magma_malloc_pinned( (void **)&impl->offsets,
                                  size * sizeof(CeedInt)); CeedChkBackend(ierr);
      impl->own_ = OWNED_PINNED;

      break;
    case CEED_USE_POINTER:
      impl->doffsets = (CeedInt *)offsets;
      impl->offsets  = NULL;
    }

  } else
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Only MemType = HOST or DEVICE supported");
  // Compile kernels
  char *magma_common_path;
  char *restriction_kernel_path, *restriction_kernel_source;
  ierr = CeedGetJitAbsolutePath(ceed,
                                "ceed/jit-source/magma/magma_common_device.h",
                                &magma_common_path); CeedChkBackend(ierr);
  CeedDebug256(ceed, 2, "----- Loading Restriction Kernel Source -----\n");
  ierr = CeedLoadSourceToBuffer(ceed, magma_common_path,
                                &restriction_kernel_source);
  CeedChkBackend(ierr);
  ierr = CeedGetJitAbsolutePath(ceed,
                                "ceed/jit-source/magma/elem_restriction.h",
                                &restriction_kernel_path); CeedChkBackend(ierr);
  ierr = CeedLoadSourceToInitializedBuffer(ceed, restriction_kernel_path,
         &restriction_kernel_source);
  CeedChkBackend(ierr);
  CeedDebug256(ceed, 2,
               "----- Loading Restriction Kernel Source Complete! -----\n");
  // The RTC compilation code expects a Ceed with the common Ceed_Cuda or Ceed_Hip
  // data
  Ceed delegate;
  ierr = CeedGetDelegate(ceed, &delegate); CeedChkBackend(ierr);
  ierr = MAGMA_RTC_COMPILE(delegate, restriction_kernel_source, &impl->module, 0);
  CeedChkBackend(ierr);

  // Kernel setup
  ierr = MAGMA_RTC_GET_KERNEL(ceed, impl->module, "magma_readDofsStrided_kernel",
                              &impl->StridedNoTranspose);
  CeedChkBackend(ierr);
  ierr = MAGMA_RTC_GET_KERNEL(ceed, impl->module, "magma_readDofsOffset_kernel",
                              &impl->OffsetNoTranspose);
  CeedChkBackend(ierr);
  ierr = MAGMA_RTC_GET_KERNEL(ceed, impl->module, "magma_writeDofsStrided_kernel",
                              &impl->StridedTranspose);
  CeedChkBackend(ierr);
  ierr = MAGMA_RTC_GET_KERNEL(ceed, impl->module, "magma_writeDofsOffset_kernel",
                              &impl->OffsetTranspose);
  CeedChkBackend(ierr);

  ierr = CeedElemRestrictionSetData(r, impl); CeedChkBackend(ierr);
  CeedInt layout[3] = {1, elemsize*nelem, elemsize};
  ierr = CeedElemRestrictionSetELayout(r, layout); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Apply",
                                CeedElemRestrictionApply_Magma); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "ApplyBlock",
                                CeedElemRestrictionApplyBlock_Magma);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "GetOffsets",
                                CeedElemRestrictionGetOffsets_Magma);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Destroy",
                                CeedElemRestrictionDestroy_Magma); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

int CeedElemRestrictionCreateBlocked_Magma(const CeedMemType mtype,
    const CeedCopyMode cmode, const CeedInt *offsets,
    const CeedElemRestriction r) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChkBackend(ierr);
  // LCOV_EXCL_START
  return CeedError(ceed, CEED_ERROR_BACKEND,
                   "Backend does not implement blocked restrictions");
  // LCOV_EXCL_STOP

  return CEED_ERROR_SUCCESS;
}
