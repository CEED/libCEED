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

// *****************************************************************************
// * Apply restriction operator r to u: v = r(rmode) u
// *****************************************************************************
static int CeedElemRestrictionApply_Magma(CeedElemRestriction r,
    CeedTransposeMode tmode, CeedTransposeMode lmode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  int ierr;
  ierr = CeedElemRestrictionGetData(r, (void *)&impl); CeedChk(ierr);

  const CeedScalar *uu;
  CeedScalar *vv;
  CeedInt nblk, blksize, nelem, elemsize, ndof, ncomp;
  ierr = CeedElemRestrictionGetNumBlocks(r, &nblk); CeedChk(ierr);
  ierr = CeedElemRestrictionGetBlockSize(r, &blksize); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumDoF(r, &ndof); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumComponents(r, &ncomp); CeedChk(ierr);
  CeedInt esize = nelem * elemsize;

#ifdef USE_MAGMA_BATCH2
  CeedInt *dindices = impl->dindices;
  // Get pointers on the device
  ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &uu); CeedChk(ierr);
  ierr = CeedVectorGetArray(v, CEED_MEM_DEVICE, &vv); CeedChk(ierr);
#else
  CeedInt *indices = impl->indices;
  ierr = CeedVectorGetArrayRead(u, CEED_MEM_HOST, &uu); CeedChk(ierr);
  ierr = CeedVectorGetArray(v, CEED_MEM_HOST, &vv); CeedChk(ierr);
#endif

  if (tmode == CEED_NOTRANSPOSE) {
    // Perform: v = r * u
    if (!impl->indices) {
        int esize_ncomp = esize*ncomp;
        #ifdef USE_MAGMA_BATCH2
           magma_template<<i=0:esize_ncomp>>
               (const CeedScalar *uu, CeedScalar *vv) {
               vv[i] = uu[i];
        }
        #else
           for (CeedInt i=0; i<esize*ncomp; i++) vv[i] = uu[i];
        #endif
    } else if (ncomp == 1) {
#ifdef USE_MAGMA_BATCH2
magma_template<<i=0:esize>>
      (const CeedScalar *uu, CeedScalar *vv, CeedInt *dindices) {
        vv[i] = uu[dindices[i]];
      }
#else
      for (CeedInt i=0; i<esize; i++) vv[i] = uu[indices[i]];
#endif
    } else {
      // vv is (elemsize x ncomp x nelem), column-major
      if (lmode == CEED_NOTRANSPOSE) { // u is (ndof x ncomp), column-major
#ifdef USE_MAGMA_BATCH2
magma_template<<e=0:nelem, d=0:ncomp, i=0:elemsize>>
        (const CeedScalar *uu, CeedScalar *vv, CeedInt *dindices, int ndof) {
          vv[i + iend*(d+dend*e)] = uu[dindices[i+iend*e]+ndof*d];
        }
#else
        for (CeedInt e = 0; e < nelem; e++)
          for (CeedInt d = 0; d < ncomp; d++)
            for (CeedInt i=0; i < elemsize; i++) {
              vv[i + elemsize*(d+ncomp*e)] =
                uu[indices[i+elemsize*e]+ndof*d];
            }
#endif
      } else { // u is (ncomp x ndof), column-major
#ifdef USE_MAGMA_BATCH2
magma_template<<e=0:nelem, d=0:ncomp, i=0:elemsize>>
        (const CeedScalar *uu, CeedScalar *vv, CeedInt *dindices) {
          vv[i + iend*(d+dend*e)] = uu[d+dend*dindices[i + iend*e]];
        }
#else
        for (CeedInt e = 0; e < nelem; e++)
          for (CeedInt d = 0; d < ncomp; d++)
            for (CeedInt i=0; i< elemsize; i++) {
              vv[i + elemsize*(d+ncomp*e)] =
                uu[d+ncomp*indices[i+elemsize*e]];
            }
#endif
      }
    }
  } else {
    // Note: in transpose mode, we perform: v += r^t * u
    if (!impl->indices) {
#ifdef USE_MAGMA_BATCH2
        magma_template<<i=0:esize>>(const CeedScalar *uu, CeedScalar *vv) {
            magmablas_datomic_add( &vv[i], uu[i]);
        }
#else
      for (CeedInt i=0; i<esize; i++) vv[i] += uu[i];
#endif
    } else if (ncomp == 1) {
      // fprintf(stderr,"3 ---------\n");
#ifdef USE_MAGMA_BATCH2
magma_template<<i=0:esize>>
      (const CeedScalar *uu, CeedScalar *vv, CeedInt *dindices) {
        magmablas_datomic_add( &vv[dindices[i]], uu[i]);
      }
#else
      for (CeedInt i=0; i<esize; i++) vv[indices[i]] += uu[i];
#endif
    } else { // u is (elemsize x ncomp x nelem)

      if (lmode == CEED_NOTRANSPOSE) { // vv is (ndof x ncomp), column-major
#ifdef USE_MAGMA_BATCH2
magma_template<<e=0:nelem, d=0:ncomp, i=0:elemsize>>
        (const CeedScalar *uu, CeedScalar *vv, CeedInt *dindices, CeedInt ndof) {
          magmablas_datomic_add( &vv[dindices[i+iend*e]+ndof*d],
                                 uu[i+iend*(d+e*dend)]);
        }
#else
        for (CeedInt e = 0; e < nelem; e++)
          for (CeedInt d = 0; d < ncomp; d++)
            for (CeedInt i=0; i < elemsize; i++) {
              vv[indices[i + elemsize*e]+ndof*d] +=
                uu[i + elemsize*(d+e*ncomp)];
            }
#endif
      } else { // vv is (ncomp x ndof), column-major
#ifdef USE_MAGMA_BATCH2
magma_template<<e=0:nelem, d=0:ncomp, i=0:elemsize>>
        (const CeedScalar *uu, CeedScalar *vv, CeedInt *dindices) {
          magmablas_datomic_add( &vv[d+dend*dindices[i + iend*e]],
                                 uu[i+iend*(d+e*dend)]);
        }
#else
        for (CeedInt e = 0; e < nelem; e++)
          for (CeedInt d = 0; d < ncomp; d++)
            for (CeedInt i=0; i < elemsize; i++) {
              vv[d+ncomp*indices[i + elemsize*e]] +=
                uu[i + elemsize*(d+e*ncomp)];
            }
#endif
      }
    }
  }

  ierr = CeedVectorRestoreArrayRead(u, &uu); CeedChk(ierr);
  ierr = CeedVectorRestoreArray(v, &vv); CeedChk(ierr);

  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED)
    *request = NULL;
  return 0;
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

static int CeedElemRestrictionCreate_Magma(CeedMemType mtype,
    CeedCopyMode cmode, const CeedInt *indices, CeedElemRestriction r) {
  int ierr;
  CeedInt nelem, elemsize, size;

  size = nelem*elemsize;
  ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChk(ierr);
  CeedElemRestriction_Magma *impl;

  // Allocate memory for the MAGMA Restricton and initializa pointers to NULL
  ierr = CeedCalloc(1,&impl); CeedChk(ierr);
  impl->dindices = NULL;
  impl->indices  = NULL;
  impl->own_ = 0;
  impl->down_= 0;

  if (mtype == CEED_MEM_HOST) {
    // memory is on the host; own_ = 0
    switch (cmode) {
    case CEED_COPY_VALUES:
      ierr = magma_malloc((void**)&impl->dindices,
                          size * sizeof(CeedInt)); CeedChk(ierr);
      ierr = magma_malloc_pinned((void**)&impl->indices,
                                 size * sizeof(CeedInt)); CeedChk(ierr);
      impl->own_ = 1;

      if (indices != NULL) {
        memcpy(impl->indices, indices, size * sizeof(indices[0]));
        magma_setvector(size, sizeof(CeedInt),
                        impl->indices, 1, impl->dindices, 1);
      }
      break;
    case CEED_OWN_POINTER:
      ierr = magma_malloc( (void**)&impl->dindices,
                           size * sizeof(CeedInt)); CeedChk(ierr);
      // TODO: possible problem here is if we are passed non-pinned memory;
      //       (as we own it, lter in destroy, we use free for pinned memory).
      impl->indices = (CeedInt *)indices;
      impl->own_ = 1;

      if (indices != NULL)
        magma_setvector(size, sizeof(CeedInt),
                        indices, 1, impl->dindices, 1);
      break;
    case CEED_USE_POINTER:
      ierr = magma_malloc( (void**)&impl->dindices,
                           size * sizeof(CeedInt)); CeedChk(ierr);
      magma_setvector(size, sizeof(CeedInt),
                      indices, 1, impl->dindices, 1);
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
        magma_copyvector(size, sizeof(CeedInt),
                         indices, 1, impl->dindices, 1);
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

  } else {
    Ceed ceed;
    ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
    return CeedError(ceed, 1, "Only MemType = HOST or DEVICE supported");
  }

  ierr = CeedElemRestrictionSetData(r, (void *)&impl); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Apply",
                                CeedElemRestrictionApply_Magma); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Destroy",
                                CeedElemRestrictionDestroy_Magma); CeedChk(ierr);
  return 0;
}
