// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
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

static int CeedElemRestrictionApply_Magma(CeedElemRestriction r, CeedTransposeMode tmode, CeedVector u, CeedVector v, CeedRequest *request) {
  CeedElemRestriction_Magma *impl;
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));

  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));

  Ceed_Magma *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  CeedInt nelem;
  CeedElemRestrictionGetNumElements(r, &nelem);

  CeedInt esize;
  CeedElemRestrictionGetElementSize(r, &esize);

  CeedInt ncomp;
  CeedElemRestrictionGetNumComponents(r, &ncomp);

  const CeedScalar *du;
  CeedScalar       *dv;

  CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &du));
  if (tmode == CEED_TRANSPOSE) {
    // Sum into for transpose mode, e-vec to l-vec
    CeedCallBackend(CeedVectorGetArray(v, CEED_MEM_DEVICE, &dv));
  } else {
    // Overwrite for notranspose mode, l-vec to e-vec
    CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &dv));
  }

  bool isStrided;
  CeedCallBackend(CeedElemRestrictionIsStrided(r, &isStrided));

  if (isStrided) {  // Strided Restriction

    CeedInt  strides[3];
    CeedInt *dstrides;
    CeedCallBackend(magma_malloc((void **)&dstrides, 3 * sizeof(CeedInt)));

    // Check to see if we should use magma Q-/E-Vector layout
    //  (dimension = slowest index, then component, then element,
    //    then node)
    bool backendstrides;
    CeedCallBackend(CeedElemRestrictionHasBackendStrides(r, &backendstrides));

    if (backendstrides) {
      strides[0] = 1;              // node stride
      strides[1] = esize * nelem;  // component stride
      strides[2] = esize;          // element stride
      magma_setvector(3, sizeof(CeedInt), strides, 1, dstrides, 1, data->queue);

    } else {
      // Get the new strides
      CeedCallBackend(CeedElemRestrictionGetStrides(r, &strides));
      magma_setvector(3, sizeof(CeedInt), strides, 1, dstrides, 1, data->queue);
    }

    void   *args[]    = {&ncomp, &esize, &nelem, &dstrides, &du, &dv};
    CeedInt grid      = nelem;
    CeedInt blocksize = 256;
    // Perform strided restriction with dstrides
    if (tmode == CEED_TRANSPOSE) {
      CeedCallBackend(CeedRunKernelMagma(ceed, impl->StridedTranspose, grid, blocksize, args));
    } else {
      CeedCallBackend(CeedRunKernelMagma(ceed, impl->StridedNoTranspose, grid, blocksize, args));
    }

    CeedCallBackend(magma_free(dstrides));

  } else {  // Offsets array provided, standard restriction

    CeedInt compstride;
    CeedCallBackend(CeedElemRestrictionGetCompStride(r, &compstride));
    void   *args[]    = {&ncomp, &compstride, &esize, &nelem, &impl->doffsets, &du, &dv};
    CeedInt grid      = nelem;
    CeedInt blocksize = 256;

    if (tmode == CEED_TRANSPOSE) {
      CeedCallBackend(CeedRunKernelMagma(ceed, impl->OffsetTranspose, grid, blocksize, args));
    } else {
      CeedCallBackend(CeedRunKernelMagma(ceed, impl->OffsetNoTranspose, grid, blocksize, args));
    }
  }

  CeedCallBackend(CeedVectorRestoreArrayRead(u, &du));
  CeedCallBackend(CeedVectorRestoreArray(v, &dv));

  return CEED_ERROR_SUCCESS;
}

int CeedElemRestrictionApplyBlock_Magma(CeedElemRestriction r, CeedInt block, CeedTransposeMode tmode, CeedVector u, CeedVector v,
                                        CeedRequest *request) {
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  // LCOV_EXCL_START
  return CeedError(ceed, CEED_ERROR_BACKEND, "Backend does not implement blocked restrictions");
  // LCOV_EXCL_STOP
}

static int CeedElemRestrictionGetOffsets_Magma(CeedElemRestriction rstr, CeedMemType mtype, const CeedInt **offsets) {
  CeedElemRestriction_Magma *impl;
  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));

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
  CeedElemRestriction_Magma *impl;
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));

  // Free if we own the data
  if (impl->own_) {
    if (impl->own_ == OWNED_PINNED) {
      CeedCallBackend(magma_free_pinned(impl->offsets));
    } else if (impl->own_ == OWNED_UNPINNED) {
      free(impl->offsets);
    }
    CeedCallBackend(magma_free(impl->doffsets));
  } else if (impl->down_) {
    CeedCallBackend(magma_free(impl->doffsets));
  }
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
#ifdef CEED_MAGMA_USE_HIP
  CeedCallHip(ceed, hipModuleUnload(impl->module));
#else
  CeedCallCuda(ceed, cuModuleUnload(impl->module));
#endif
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

int CeedElemRestrictionCreate_Magma(CeedMemType mtype, CeedCopyMode cmode, const CeedInt *offsets, CeedElemRestriction r) {
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));

  Ceed_Magma *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  CeedInt elemsize, nelem;
  CeedCallBackend(CeedElemRestrictionGetNumElements(r, &nelem));
  CeedCallBackend(CeedElemRestrictionGetElementSize(r, &elemsize));
  CeedInt size = elemsize * nelem;

  CeedElemRestriction_Magma *impl;
  CeedCallBackend(CeedCalloc(1, &impl));

  impl->doffsets = NULL;
  impl->offsets  = NULL;
  impl->own_     = OWNED_NONE;
  impl->down_    = 0;

  if (mtype == CEED_MEM_HOST) {
    // memory is on the host; own_ = 0
    switch (cmode) {
      case CEED_COPY_VALUES:
        impl->own_ = OWNED_PINNED;

        if (offsets != NULL) {
          CeedCallBackend(magma_malloc((void **)&impl->doffsets, size * sizeof(CeedInt)));
          CeedCallBackend(magma_malloc_pinned((void **)&impl->offsets, size * sizeof(CeedInt)));
          memcpy(impl->offsets, offsets, size * sizeof(CeedInt));

          magma_setvector(size, sizeof(CeedInt), offsets, 1, impl->doffsets, 1, data->queue);
        }
        break;
      case CEED_OWN_POINTER:
        impl->own_ = OWNED_UNPINNED;

        if (offsets != NULL) {
          CeedCallBackend(magma_malloc((void **)&impl->doffsets, size * sizeof(CeedInt)));
          impl->offsets = (CeedInt *)offsets;

          magma_setvector(size, sizeof(CeedInt), offsets, 1, impl->doffsets, 1, data->queue);
        }
        break;
      case CEED_USE_POINTER:
        if (offsets != NULL) {
          CeedCallBackend(magma_malloc((void **)&impl->doffsets, size * sizeof(CeedInt)));
          magma_setvector(size, sizeof(CeedInt), offsets, 1, impl->doffsets, 1, data->queue);
        }
        impl->down_   = 1;
        impl->offsets = (CeedInt *)offsets;
    }
  } else if (mtype == CEED_MEM_DEVICE) {
    // memory is on the device; own = 0
    switch (cmode) {
      case CEED_COPY_VALUES:
        CeedCallBackend(magma_malloc((void **)&impl->doffsets, size * sizeof(CeedInt)));
        CeedCallBackend(magma_malloc_pinned((void **)&impl->offsets, size * sizeof(CeedInt)));
        impl->own_ = OWNED_PINNED;

        if (offsets) magma_getvector(size, sizeof(CeedInt), impl->doffsets, 1, (void *)offsets, 1, data->queue);
        break;
      case CEED_OWN_POINTER:
        impl->doffsets = (CeedInt *)offsets;
        CeedCallBackend(magma_malloc_pinned((void **)&impl->offsets, size * sizeof(CeedInt)));
        impl->own_ = OWNED_PINNED;

        break;
      case CEED_USE_POINTER:
        impl->doffsets = (CeedInt *)offsets;
        impl->offsets  = NULL;
    }

  } else return CeedError(ceed, CEED_ERROR_BACKEND, "Only MemType = HOST or DEVICE supported");
  // Compile kernels
  char *magma_common_path;
  char *restriction_kernel_path, *restriction_kernel_source;
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/magma/magma_common_device.h", &magma_common_path));
  CeedDebug256(ceed, 2, "----- Loading Restriction Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, magma_common_path, &restriction_kernel_source));
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/magma/elem_restriction.h", &restriction_kernel_path));
  CeedCallBackend(CeedLoadSourceToInitializedBuffer(ceed, restriction_kernel_path, &restriction_kernel_source));
  CeedDebug256(ceed, 2, "----- Loading Restriction Kernel Source Complete! -----\n");
  // The RTC compilation code expects a Ceed with the common Ceed_Cuda or Ceed_Hip
  // data
  Ceed delegate;
  CeedCallBackend(CeedGetDelegate(ceed, &delegate));
  CeedCallBackend(CeedCompileMagma(delegate, restriction_kernel_source, &impl->module, 0));

  // Kernel setup
  CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_readDofsStrided_kernel", &impl->StridedNoTranspose));
  CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_readDofsOffset_kernel", &impl->OffsetNoTranspose));
  CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_writeDofsStrided_kernel", &impl->StridedTranspose));
  CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_writeDofsOffset_kernel", &impl->OffsetTranspose));

  CeedCallBackend(CeedElemRestrictionSetData(r, impl));
  CeedInt layout[3] = {1, elemsize * nelem, elemsize};
  CeedCallBackend(CeedElemRestrictionSetELayout(r, layout));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", r, "Apply", CeedElemRestrictionApply_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", r, "ApplyBlock", CeedElemRestrictionApplyBlock_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", r, "GetOffsets", CeedElemRestrictionGetOffsets_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", r, "Destroy", CeedElemRestrictionDestroy_Magma));
  CeedCallBackend(CeedFree(&restriction_kernel_path));
  CeedCallBackend(CeedFree(&restriction_kernel_source));

  return CEED_ERROR_SUCCESS;
}

int CeedElemRestrictionCreateBlocked_Magma(const CeedMemType mtype, const CeedCopyMode cmode, const CeedInt *offsets, const CeedElemRestriction r) {
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  // LCOV_EXCL_START
  return CeedError(ceed, CEED_ERROR_BACKEND, "Backend does not implement blocked restrictions");
  // LCOV_EXCL_STOP

  return CEED_ERROR_SUCCESS;
}
