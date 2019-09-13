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

#include "ceed-magma.h"
#include <string.h>

typedef struct {
  CeedScalar *array;
  CeedScalar *darray;
  int  own_;
  int down_;
} CeedVector_Magma;

typedef struct {
  CeedInt *indices;
  CeedInt *dindices;
  int  own_;
  int down_;            // cover a case where we own Device memory
} CeedElemRestriction_Magma;

typedef struct {
  const CeedScalar **inputs;
  CeedScalar **outputs;
} CeedQFunction_Magma;

typedef struct {
  CeedVector *Evecs; /// E-vectors needed to apply operator (in followed by out)
  CeedScalar **Edata;
  CeedVector *evecsin;   /// Input E-vectors needed to apply operator
  CeedVector *evecsout;   /// Output E-vectors needed to apply operator
  CeedVector *qvecsin;   /// Input Q-vectors needed to apply operator
  CeedVector *qvecsout;   /// Output Q-vectors needed to apply operator
  CeedInt    numein;
  CeedInt    numeout;
} CeedOperator_Magma;

// *****************************************************************************
// * Initialize vector vec (after free mem) with values from array based on cmode
// *   CEED_COPY_VALUES: memory is allocated in vec->array_allocated, made equal
// *                     to array, and data is copied (not store passed pointer)
// *   CEED_OWN_POINTER: vec->data->array_allocated and vec->data->array = array
// *   CEED_USE_POINTER: vec->data->array = array (can modify; no ownership)
// * mtype: CEED_MEM_HOST or CEED_MEM_DEVICE
// *****************************************************************************
static int CeedVectorSetArray_Magma(CeedVector vec, CeedMemType mtype,
                                    CeedCopyMode cmode, CeedScalar *array) {
  CeedVector_Magma *impl = vec->data;
  int ierr;

  // If own data, free the "old" data, e.g., as it may be of different size
  if (impl->own_) {
    magma_free( impl->darray );
    magma_free_pinned( impl->array );
    impl->darray = NULL;
    impl->array  = NULL;
    impl->own_ = 0;
    impl->down_= 0;
  }

  if (mtype == CEED_MEM_HOST) {
    // memory is on the host; own_ = 0
    switch (cmode) {
    case CEED_COPY_VALUES:
      ierr = magma_malloc( (void **)&impl->darray,
                           vec->length * sizeof(CeedScalar)); CeedChk(ierr);
      ierr = magma_malloc_pinned( (void **)&impl->array,
                                  vec->length * sizeof(CeedScalar)); CeedChk(ierr);
      impl->own_ = 1;

      if (array != NULL)
        magma_setvector(vec->length, sizeof(array[0]),
                        array, 1, impl->darray, 1);
      break;
    case CEED_OWN_POINTER:
      ierr = magma_malloc( (void **)&impl->darray,
                           vec->length * sizeof(CeedScalar)); CeedChk(ierr);
      // TODO: possible problem here is if we are passed non-pinned memory;
      //       (as we own it, lter in destroy, we use free for pinned memory).
      impl->array = array;
      impl->own_ = 1;

      if (array != NULL)
        magma_setvector(vec->length, sizeof(array[0]),
                        array, 1, impl->darray, 1);
      break;
    case CEED_USE_POINTER:
      ierr = magma_malloc( (void **)&impl->darray,
                           vec->length * sizeof(CeedScalar)); CeedChk(ierr);
      magma_setvector(vec->length, sizeof(array[0]),
                      array, 1, impl->darray, 1);

      impl->down_  = 1;
      impl->array  = array;
    }
  } else if (mtype == CEED_MEM_DEVICE) {
    // memory is on the device; own = 0
    switch (cmode) {
    case CEED_COPY_VALUES:
      ierr = magma_malloc( (void **)&impl->darray,
                           vec->length * sizeof(CeedScalar)); CeedChk(ierr);
      ierr = magma_malloc_pinned( (void **)&impl->array,
                                  vec->length * sizeof(CeedScalar)); CeedChk(ierr);
      impl->own_ = 1;

      if (array)
        magma_copyvector(vec->length, sizeof(array[0]),
                         array, 1, impl->darray, 1);
      else
        // t30 assumes allocation initializes with 0s
        magma_setvector(vec->length, sizeof(array[0]),
                        impl->array, 1, impl->darray, 1);
      break;
    case CEED_OWN_POINTER:
      impl->darray = array;
      ierr = magma_malloc_pinned( (void **)&impl->array,
                                  vec->length * sizeof(CeedScalar)); CeedChk(ierr);
      impl->own_ = 1;

      break;
    case CEED_USE_POINTER:
      impl->darray = array;
      impl->array  = NULL;
    }

  } else
    return CeedError(vec->ceed, 1, "Only MemType = HOST or DEVICE supported");

  return 0;
}

// *****************************************************************************
// * Give data pointer from vector vec to array (on HOST or DEVICE)
// *****************************************************************************
static int CeedVectorGetArray_Magma(CeedVector vec, CeedMemType mtype,
                                    CeedScalar **array) {
  CeedVector_Magma *impl = vec->data;
  int ierr;

  if (mtype == CEED_MEM_HOST) {
    if (impl->own_) {
      // data is owned so GPU had the most up-to-date version; copy it
      // TTT - apparantly it doesn't have most up to date data
      magma_getvector(vec->length, sizeof(*array[0]),
                      impl->darray, 1, impl->array, 1);
      CeedDebug("\033[31m[CeedVectorGetArray_Magma]");
      //fprintf(stderr,"rrrrrrrrrrrrrrr\n");
    } else if (impl->array == NULL) {
      // Vector doesn't own the data and was set on GPU
      if (impl->darray == NULL) {
        // call was made just to allocate memory
        ierr = CeedVectorSetArray(vec, mtype, CEED_COPY_VALUES, NULL);
        CeedChk(ierr);
      } else
        return CeedError(vec->ceed, 1, "Can not access DEVICE vector on HOST");
    }
    *array = impl->array;
  } else if (mtype == CEED_MEM_DEVICE) {
    if (impl->darray == NULL) {
      // Vector doesn't own the data and was set on the CPU
      if (impl->array == NULL) {
        // call was made just to allocate memory
        ierr = CeedVectorSetArray(vec, mtype, CEED_COPY_VALUES, NULL);
        CeedChk(ierr);
      } else
        return CeedError(vec->ceed, 1, "Can not access HOST vector on DEVICE");
    }
    *array = impl->darray;
  } else
    return CeedError(vec->ceed, 1, "Can only provide to HOST or DEVICE memory");

  return 0;
}

// *****************************************************************************
// * Give data pointer from vector vec to array (on HOST or DEVICE) to read it
// *****************************************************************************
static int CeedVectorGetArrayRead_Magma(CeedVector vec, CeedMemType mtype,
                                        const CeedScalar **array) {
  CeedVector_Magma *impl = vec->data;
  int ierr;

  if (mtype == CEED_MEM_HOST) {
    if (impl->own_) {
      // data is owned so GPU had the most up-to-date version; copy it
      magma_getvector(vec->length, sizeof(*array[0]),
                      impl->darray, 1, impl->array, 1);
    } else if (impl->array == NULL) {
      // Vector doesn't own the data and was set on GPU
      if (impl->darray == NULL) {
        // call was made just to allocate memory
        ierr = CeedVectorSetArray(vec, mtype, CEED_COPY_VALUES, NULL);
        CeedChk(ierr);
      } else
        return CeedError(vec->ceed, 1, "Can not access DEVICE vector on HOST");
    }
    *array = impl->array;
  } else if (mtype == CEED_MEM_DEVICE) {
    if (impl->darray == NULL) {
      // Vector doesn't own the data and was set on the CPU
      if (impl->array == NULL) {
        // call was made just to allocate memory
        ierr = CeedVectorSetArray(vec, mtype, CEED_COPY_VALUES, NULL);
        CeedChk(ierr);
      } else
        return CeedError(vec->ceed, 1, "Can not access HOST vector on DEVICE");
    }
    *array = impl->darray;
  } else
    return CeedError(vec->ceed, 1, "Can only provide to HOST or DEVICE memory");

  return 0;
}

// *****************************************************************************
// * There is no mtype here for array so it is not clear if we restore from HOST
// * memory or from DEVICE memory. We assume that it is CPU memory because if
// * it was GPU memory we would not call this routine at all.
// * Restore vector vec with values from array, where array received its values
// * from vec and possibly modified them.
// *****************************************************************************
static int CeedVectorRestoreArray_Magma(CeedVector vec) {
  CeedVector_Magma *impl = vec->data;

  if (impl->down_) {
    // nothing to do if array is on GPU, except if down_=1(case CPU use pointer)
    magma_getvector(vec->length, sizeof(*array[0]),
                    impl->darray, 1, impl->array, 1);
  }
  return 0;
}

// *****************************************************************************
// * There is no mtype here for array so it is not clear if we restore from HOST
// * memory or from DEVICE memory. We assume that it is CPU memory because if
// * it was GPU memory we would not call this routine at all.
// * Restore vector vec with values from array, where array received its values
// * from vec to only read them; in this case vec may have been modified meanwhile
// * and needs to be restored here.
// *****************************************************************************
static int CeedVectorRestoreArrayRead_Magma(CeedVector vec) {
  CeedVector_Magma *impl = vec->data;

  if (impl->down_) {
    // nothing to do if array is on GPU, except if down_=1(case CPU use pointer)
    magma_getvector(vec->length, sizeof(*array[0]),
                    impl->darray, 1, impl->array, 1);
  }
  return 0;
}

static int CeedVectorDestroy_Magma(CeedVector vec) {
  CeedVector_Magma *impl = vec->data;
  int ierr;

  // Free if we own the data
  if (impl->own_) {
    ierr = magma_free_pinned(impl->array); CeedChk(ierr);
    ierr = magma_free(impl->darray);       CeedChk(ierr);
  } else if (impl->down_) {
    ierr = magma_free(impl->darray);       CeedChk(ierr);
  }
  ierr = CeedFree(&vec->data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * Create vector vec of size n
// *****************************************************************************
static int CeedVectorCreate_Magma(CeedInt n, CeedVector vec) {
  CeedVector_Magma *impl;
  int ierr;

  vec->SetArray = CeedVectorSetArray_Magma;
  vec->GetArray = CeedVectorGetArray_Magma;
  vec->GetArrayRead = CeedVectorGetArrayRead_Magma;
  vec->RestoreArray = CeedVectorRestoreArray_Magma;
  vec->RestoreArrayRead = CeedVectorRestoreArrayRead_Magma;
  vec->Destroy = CeedVectorDestroy_Magma;
  ierr = CeedCalloc(1,&impl); CeedChk(ierr);
  impl->darray = NULL;
  impl->array  = NULL;
  impl->own_ = 0;
  impl->down_= 0;
  vec->data = impl;
  return 0;
}


// *****************************************************************************
// * Apply restriction operator r to u: v = r(rmode) u
// *****************************************************************************
static int CeedElemRestrictionApply_Magma(CeedElemRestriction r,
    CeedTransposeMode tmode,
    CeedTransposeMode lmode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  CeedElemRestriction_Magma *impl = r->data;
  int ierr;
  const CeedScalar *uu;
  CeedScalar *vv;
  CeedInt nelem = r->nelem, elemsize = r->elemsize, ndof = r->ndof,
          ncomp=r->ncomp;
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
      for (CeedInt i=0; i<esize*ncomp; i++) vv[i] = uu[i];
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
      for (CeedInt i=0; i<esize; i++) vv[i] += uu[i];
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
      fprintf(stderr,"2 ---------\n");

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

int CeedElemRestrictionApplyBlock_Magma(CeedElemRestriction r,
                                        CeedInt block, CeedTransposeMode tmode,
                                        CeedTransposeMode lmode, CeedVector u,
                                        CeedVector v, CeedRequest *request) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  return CeedError(ceed, 1, "Backend does not implement blocked restrictions");
}

static int CeedElemRestrictionDestroy_Magma(CeedElemRestriction r) {
  CeedElemRestriction_Magma *impl = r->data;
  int ierr;

  // Free if we own the data
  if (impl->own_) {
    ierr = magma_free_pinned(impl->indices); CeedChk(ierr);
    ierr = magma_free(impl->dindices);       CeedChk(ierr);
  } else if (impl->down_) {
    ierr = magma_free(impl->dindices);       CeedChk(ierr);
  }
  ierr = CeedFree(&r->data); CeedChk(ierr);
  return 0;
}

static int CeedElemRestrictionCreate_Magma(CeedMemType mtype,
    CeedCopyMode cmode,
    const CeedInt *indices, CeedElemRestriction r) {
  int ierr, size = r->nelem*r->elemsize;
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
      ierr = magma_malloc( (void **)&impl->dindices,
                           size * sizeof(CeedInt)); CeedChk(ierr);
      ierr = magma_malloc_pinned( (void **)&impl->indices,
                                  size * sizeof(CeedInt)); CeedChk(ierr);
      impl->own_ = 1;

      if (indices != NULL) {
        memcpy(impl->indices, indices, size * sizeof(indices[0]));
        magma_setvector(size, sizeof(CeedInt),
                        impl->indices, 1, impl->dindices, 1);
      }
      break;
    case CEED_OWN_POINTER:
      ierr = magma_malloc( (void **)&impl->dindices,
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
      ierr = magma_malloc( (void **)&impl->dindices,
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
      ierr = magma_malloc( (void **)&impl->dindices,
                           size * sizeof(CeedInt)); CeedChk(ierr);
      ierr = magma_malloc_pinned( (void **)&impl->indices,
                                  size * sizeof(CeedInt)); CeedChk(ierr);
      impl->own_ = 1;

      if (indices)
        magma_copyvector(size, sizeof(CeedInt),
                         indices, 1, impl->dindices, 1);
      break;
    case CEED_OWN_POINTER:
      impl->dindices = (CeedInt *)indices;
      ierr = magma_malloc_pinned( (void **)&impl->indices,
                                  size * sizeof(CeedInt)); CeedChk(ierr);
      impl->own_ = 1;

      break;
    case CEED_USE_POINTER:
      impl->dindices = (CeedInt *)indices;
      impl->indices  = NULL;
    }

  } else
    return CeedError(r->ceed, 1, "Only MemType = HOST or DEVICE supported");

  r->data    = impl;
  r->Apply   = CeedElemRestrictionApply_Magma;
  r->ApplyBlock = CeedElemRestrictionApplyBlock_Magma;
  r->Destroy = CeedElemRestrictionDestroy_Magma;

  return 0;
}

static int CeedElemRestrictionCreateBlocked_Magma(CeedMemType mtype,
    CeedCopyMode cmode,
    const CeedInt *indices, CeedElemRestriction r) {
  return CeedError(r->ceed, 1, "Backend does not implement blocked restrictions");
}

// Contracts on the middle index
// NOTRANSPOSE: V_ajc = T_jb U_abc
// TRANSPOSE:   V_ajc = T_bj U_abc
// If Add != 0, "=" is replaced by "+="
static int CeedTensorContract_Magma(Ceed ceed,
                                    CeedInt A, CeedInt B, CeedInt C, CeedInt J,
                                    const CeedScalar *t, CeedTransposeMode tmode,
                                    const CeedInt Add,
                                    const CeedScalar *u, CeedScalar *v) {
  #ifdef USE_MAGMA_BATCH
  magma_dtensor_contract(ceed, A, B, C, J, t, tmode, Add, u, v);
  #else
  CeedInt tstride0 = B, tstride1 = 1;
  if (tmode == CEED_TRANSPOSE) {
    tstride0 = 1; tstride1 = J;
  }
  CeedDebug("\033[31m[CeedTensorContract] A=%d, J=%d, C=%d, B=%d: %d %d %d",
            A,J,C,B,A*J*B*C, C*J*A, C*B*A);
  for (CeedInt a=0; a<A; a++) {
    for (CeedInt j=0; j<J; j++) {
      if (!Add) {
        for (CeedInt c=0; c<C; c++)
          v[(a*J+j)*C+c] = 0;
      }
      for (CeedInt b=0; b<B; b++) {
        for (CeedInt c=0; c<C; c++) {
          v[(a*J+j)*C+c] += t[j*tstride0 + b*tstride1] * u[(a*B+b)*C+c];
        }
      }
    }
  }
  #endif
  return 0;
}

static int CeedBasisApply_Magma(CeedBasis basis, CeedInt nelem,
                                CeedTransposeMode tmode, CeedEvalMode emode,
                                CeedVector U, CeedVector V) {
  int ierr;
  const CeedInt dim = basis->dim;
  const CeedInt ncomp = basis->ncomp;
  const CeedInt nqpt = ncomp*CeedPowInt(basis->Q1d, dim);
  const CeedInt add = (tmode == CEED_TRANSPOSE);
  const CeedScalar *u;
  CeedScalar *v;
  if (U) {
    ierr = CeedVectorGetArrayRead(U, CEED_MEM_HOST, &u); CeedChk(ierr);
  } else if (emode != CEED_EVAL_WEIGHT) {
    return CeedError(ceed, 1,
                     "An input vector is required for this CeedEvalMode");
  }
  ierr = CeedVectorGetArray(V, CEED_MEM_HOST, &v); CeedChk(ierr);

  if (nelem != 1)
    return CeedError(basis->ceed, 1,
                     "This backend does not support BasisApply for multiple elements");

  CeedDebug("\033[01m[CeedBasisApply_Magma] vsize=%d",
            ncomp*CeedPowInt(basis->P1d, dim));

  if (tmode == CEED_TRANSPOSE) {
    const CeedInt vsize = ncomp*CeedPowInt(basis->P1d, dim);
    for (CeedInt i = 0; i < vsize; i++)
      v[i] = (CeedScalar) 0;
  }
  if (emode & CEED_EVAL_INTERP) {
    CeedInt P = basis->P1d, Q = basis->Q1d;
    if (tmode == CEED_TRANSPOSE) {
      P = basis->Q1d; Q = basis->P1d;
    }
    CeedInt pre = ncomp*CeedPowInt(P, dim-1), post = 1;
    CeedScalar tmp[2][ncomp*Q*CeedPowInt(P>Q?P:Q, dim-1)];
    CeedDebug("\033[01m[CeedBasisApply_Magma] tmpsize = %d",
              ncomp*Q*CeedPowInt(P>Q?P:Q, dim-1));
    for (CeedInt d=0; d<dim; d++) {
      ierr = CeedTensorContract_Magma(basis->ceed, pre, P, post, Q, basis->interp1d,
                                      tmode, add&&(d==dim-1),
                                      d==0?u:tmp[d%2], d==dim-1?v:tmp[(d+1)%2]);
      CeedChk(ierr);
      pre /= P;
      post *= Q;
    }
    if (tmode == CEED_NOTRANSPOSE) {
      v += nqpt;
    } else {
      u += nqpt;
    }
  }
  if (emode & CEED_EVAL_GRAD) {
    CeedInt P = basis->P1d, Q = basis->Q1d;
    // In CEED_NOTRANSPOSE mode:
    // u is (P^dim x nc), column-major layout (nc = ncomp)
    // v is (Q^dim x nc x dim), column-major layout (nc = ncomp)
    // In CEED_TRANSPOSE mode, the sizes of u and v are switched.
    if (tmode == CEED_TRANSPOSE) {
      P = basis->Q1d, Q = basis->P1d;
    }
    CeedScalar tmp[2][ncomp*Q*CeedPowInt(P>Q?P:Q, dim-1)];
    CeedDebug("\033[01m[CeedBasisApply_Magma] tmpsize = %d",
              ncomp*Q*CeedPowInt(P>Q?P:Q, dim-1));
    for (CeedInt p = 0; p < dim; p++) {
      CeedInt pre = ncomp*CeedPowInt(P, dim-1), post = 1;
      for (CeedInt d=0; d<dim; d++) {
        ierr = CeedTensorContract_Magma(basis->ceed, pre, P, post, Q,
                                        (p==d)?basis->grad1d:basis->interp1d,
                                        tmode, add&&(d==dim-1),
                                        d==0?u:tmp[d%2], d==dim-1?v:tmp[(d+1)%2]);
        CeedChk(ierr);
        pre /= P;
        post *= Q;
      }
      if (tmode == CEED_NOTRANSPOSE) {
        v += nqpt;
      } else {
        u += nqpt;
      }
    }
  }
  if (emode & CEED_EVAL_WEIGHT) {
    if (tmode == CEED_TRANSPOSE)
      return CeedError(basis->ceed, 1,
                       "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
    CeedInt Q = basis->Q1d;
    for (CeedInt d=0; d<dim; d++) {
      CeedInt pre = CeedPowInt(Q, dim-d-1), post = CeedPowInt(Q, d);
      for (CeedInt i=0; i<pre; i++) {
        for (CeedInt j=0; j<Q; j++) {
          for (CeedInt k=0; k<post; k++) {
            v[(i*Q + j)*post + k] = basis->qweight1d[j]
                                    * (d == 0 ? 1 : v[(i*Q + j)*post + k]);
          }
        }
      }
    }
  }
  if (U) {
    ierr = CeedVectorRestoreArrayRead(U, &u); CeedChk(ierr);
  }
  ierr = CeedVectorRestoreArray(V, &v); CeedChk(ierr);
  return 0;
}

static int CeedBasisDestroy_Magma(CeedBasis basis) {
  return 0;
}

static int CeedBasisCreateTensorH1_Magma(CeedInt dim, CeedInt P1d,
    CeedInt Q1d, const CeedScalar *interp1d,
    const CeedScalar *grad1d,
    const CeedScalar *qref1d,
    const CeedScalar *qweight1d,
    CeedBasis basis) {
  basis->Apply = CeedBasisApply_Magma;
  basis->Destroy = CeedBasisDestroy_Magma;
  return 0;
}

static int CeedBasisCreateH1_Magma(CeedElemTopology topo, CeedInt dim,
                                   CeedInt ndof, CeedInt nqpts,
                                   const CeedScalar *interp,
                                   const CeedScalar *grad,
                                   const CeedScalar *qref,
                                   const CeedScalar *qweight,
                                   CeedBasis basis) {
  return CeedError(basis->ceed, 1, "Backend does not implement non-tensor bases");
}

static int CeedQFunctionApply_Magma(CeedQFunction qf, CeedInt Q,
                                    CeedVector *U, CeedVector *V) {
  int ierr;
  CeedQFunction_Ref *impl;
  ierr = CeedQFunctionGetData(qf, (void *)&impl); CeedChk(ierr);

  void *ctx;
  ierr = CeedQFunctionGetContext(qf, &ctx); CeedChk(ierr);

  int (*f)() = NULL;
  ierr = CeedQFunctionGetUserFunction(qf, (int (* *)())&f); CeedChk(ierr);

  CeedInt nIn, nOut;
  ierr = CeedQFunctionGetNumArgs(qf, &nIn, &nOut); CeedChk(ierr);

  for (int i = 0; i<nIn; i++) {
    if (U[i]) {
      ierr = CeedVectorGetArrayRead(U[i], CEED_MEM_HOST, &impl->inputs[i]);
      CeedChk(ierr);
    }
  }
  for (int i = 0; i<nOut; i++) {
    if (U[i]) {
      ierr = CeedVectorGetArray(V[i], CEED_MEM_HOST, &impl->outputs[i]);
      CeedChk(ierr);
    }
  }

  ierr = f(ctx, Q, impl->inputs, impl->outputs); CeedChk(ierr);

  for (int i = 0; i<nIn; i++) {
    if (U[i]) {
      ierr = CeedVectorRestoreArrayRead(U[i], &impl->inputs[i]); CeedChk(ierr);
    }
  }
  for (int i = 0; i<nOut; i++) {
    if (U[i]) {
      ierr = CeedVectorRestoreArray(V[i], &impl->outputs[i]); CeedChk(ierr);
    }
  }
  return 0;
}

static int CeedQFunctionDestroy_Magma(CeedQFunction qf) {
  int ierr;
  CeedQFunction_Magma *impl;
  ierr = CeedQFunctionGetData(qf, (void *)&impl); CeedChk(ierr);

  ierr = CeedFree(&impl->inputs); CeedChk(ierr);
  ierr = CeedFree(&impl->outputs); CeedChk(ierr);
  ierr = CeedFree(&impl); CeedChk(ierr);

  return 0;
}

static int CeedQFunctionCreate_Magma(CeedQFunction qf) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);

  CeedQFunction_Magma *impl;
  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  ierr = CeedCalloc(16, &impl->inputs); CeedChk(ierr);
  ierr = CeedCalloc(16, &impl->outputs); CeedChk(ierr);
  ierr = CeedQFunctionSetData(qf, (void *)&impl); CeedChk(ierr);

  qf->Apply = CeedQFunctionApply_Magma;
  qf->Destroy = CeedQFunctionDestroy_Magma;
  return 0;
}

static int CeedOperatorDestroy_Magma(CeedOperator op) {
  int ierr;
  CeedOperator_Magma *impl;
  ierr = CeedOperatorGetData(op, (void *)&impl); CeedChk(ierr);

  for (CeedInt i=0; i<impl->numein+impl->numeout; i++) {
    if (impl->Evecs[i]) {
      ierr = CeedVectorDestroy(&impl->Evecs[i]); CeedChk(ierr);
    }
  }
  ierr = CeedFree(&impl->Evecs); CeedChk(ierr);
  ierr = CeedFree(&impl->Edata); CeedChk(ierr);

  for (CeedInt i=0; i<impl->numein; i++) {
    ierr = CeedVectorDestroy(&impl->evecsin[i]); CeedChk(ierr);
    ierr = CeedVectorDestroy(&impl->qvecsin[i]); CeedChk(ierr);
  }
  ierr = CeedFree(&impl->evecsin); CeedChk(ierr);
  ierr = CeedFree(&impl->qvecsin); CeedChk(ierr);

  for (CeedInt i=0; i<impl->numeout; i++) {
    ierr = CeedVectorDestroy(&impl->evecsout[i]); CeedChk(ierr);
    ierr = CeedVectorDestroy(&impl->qvecsout[i]); CeedChk(ierr);
  }
  ierr = CeedFree(&impl->evecsout); CeedChk(ierr);
  ierr = CeedFree(&impl->qvecsout); CeedChk(ierr);


  ierr = CeedFree(&impl); CeedChk(ierr);
  return 0;
}


/*
  Setup infields or outfields
 */
static int CeedOperatorSetupFields_Magma(CeedQFunction qf, CeedOperator op,
    bool inOrOut,
    CeedVector *fullevecs, CeedVector *evecs,
    CeedVector *qvecs, CeedInt starte,
    CeedInt numfields, CeedInt Q) {
  CeedInt dim = 1, ierr, ncomp;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  CeedQFunction_Magma *qf_data;
  ierr = CeedQFunctionGetData(qf, (void *)&qf_data); CeedChk(ierr);
  CeedBasis basis;
  CeedElemRestriction Erestrict;
  CeedOperatorField *opfields;
  CeedQFunctionField *qffields;
  if (inOrOut) {
    ierr = CeedOperatorGetFields(op, NULL, &opfields);
    CeedChk(ierr);
    ierr = CeedQFunctionGetFields(qf, NULL, &qffields);
    CeedChk(ierr);
  } else {
    ierr = CeedOperatorGetFields(op, &opfields, NULL);
    CeedChk(ierr);
    ierr = CeedQFunctionGetFields(qf, &qffields, NULL);
    CeedChk(ierr);
  }

  // Loop over fields
  for (CeedInt i=0; i<numfields; i++) {
    CeedEvalMode emode;
    ierr = CeedQFunctionFieldGetEvalMode(qffields[i], &emode); CeedChk(ierr);
    if (emode != CEED_EVAL_WEIGHT) {
      ierr = CeedOperatorFieldGetElemRestriction(opfields[i], &Erestrict);
      CeedChk(ierr);
      ierr = CeedElemRestrictionCreateVector(Erestrict, NULL, &fullevecs[i+starte]);
      CeedChk(ierr);
    } else {
    }
    switch(emode) {
    case CEED_EVAL_NONE:
      ierr = CeedQFunctionFieldGetNumComponents(qffields[i], &ncomp);
      CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, Q*ncomp, &evecs[i]); CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, Q*ncomp, &qvecs[i]); CeedChk(ierr);
      break; // No action
    case CEED_EVAL_INTERP:
      ierr = CeedOperatorFieldGetBasis(opfields[i], &basis); CeedChk(ierr);
      ierr = CeedQFunctionFieldGetNumComponents(qffields[i], &ncomp);
      CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, Q*ncomp, &evecs[i]); CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, Q*ncomp, &qvecs[i]); CeedChk(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(opfields[i], &basis); CeedChk(ierr);
      ierr = CeedQFunctionFieldGetNumComponents(qffields[i], &ncomp);
      ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
      ierr = CeedQFunctionFieldGetNumComponents(qffields[i], &ncomp);
      CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, Q*ncomp, &evecs[i]); CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, Q*ncomp*dim, &qvecs[i]); CeedChk(ierr);
      break;
    case CEED_EVAL_WEIGHT: // Only on input fields
      ierr = CeedOperatorFieldGetBasis(opfields[i], &basis); CeedChk(ierr);
      ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, Q, &qvecs[i]); CeedChk(ierr);
      ierr = CeedBasisApply(basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT,
                            NULL, qvecs[i]); CeedChk(ierr);
      assert(starte==0);
      break;
    case CEED_EVAL_DIV: break; // Not implemented
    case CEED_EVAL_CURL: break; // Not implemented
    }
  }
  return 0;
}

/*
  CeedOperator needs to connect all the named fields (be they active or passive)
  to the named inputs and outputs of its CeedQFunction.
 */
static int CeedOperatorSetup_Magma(CeedOperator op) {
  int ierr;
  bool setupdone;
  ierr = CeedOperatorGetSetupStatus(op, &setupdone); CeedChk(ierr);
  if (setupdone) return 0;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  CeedOperator_Magma *data;
  ierr = CeedOperatorGetData(op, (void *)&data); CeedChk(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
  CeedInt Q, numinputfields, numoutputfields;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChk(ierr);
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &opinputfields, &opoutputfields);
  CeedChk(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, &qfinputfields, &qfoutputfields);
  CeedChk(ierr);

  data->numein = numinputfields;
  data->numeout = numoutputfields;

  // Allocate
  const CeedInt numIO = numinputfields + numoutputfields;

  ierr = CeedCalloc(numinputfields + numoutputfields, &data->Evecs);
  CeedChk(ierr);
  ierr = CeedCalloc(numinputfields + numoutputfields, &data->Edata);
  CeedChk(ierr);

  ierr = CeedCalloc(16, &data->evecsin); CeedChk(ierr);
  ierr = CeedCalloc(16, &data->evecsout); CeedChk(ierr);
  ierr = CeedCalloc(16, &data->qvecsin); CeedChk(ierr);
  ierr = CeedCalloc(16, &data->qvecsout); CeedChk(ierr);

  // Set up infield and outfield pointer arrays
  // Infields
  ierr = CeedOperatorSetupFields_Magma(qf, op, 0, data->Evecs,
                                       data->evecsin, data->qvecsin, 0,
                                       numinputfields, Q);
  CeedChk(ierr);
  // Outfields
  ierr = CeedOperatorSetupFields_Magma(qf, op, 1, data->Evecs,
                                       data->evecsout, data->qvecsout,
                                       numinputfields, numoutputfields, Q);
  CeedChk(ierr);
  ierr = CeedOperatorSetSetupDone(op); CeedChk(ierr);
  return 0;
}

static int CeedOperatorApply_Magma(CeedOperator op, CeedVector invec,
                                   CeedVector outvec, CeedRequest *request) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  CeedOperator_Magma *data;
  ierr = CeedOperatorGetData(op, (void *)&data); CeedChk(ierr);
  //CeedVector *E = data->Evecs, *D = data->D, outvec;
  CeedInt Q, elemsize, numelements, numinputfields, numoutputfields, ncomp;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChk(ierr);
  ierr = CeedOperatorGetNumElements(op, &numelements); CeedChk(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
  CeedTransposeMode lmode;
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &opinputfields, &opoutputfields);
  CeedChk(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, &qfinputfields, &qfoutputfields);
  CeedChk(ierr);
  CeedEvalMode emode;
  CeedVector vec;
  CeedBasis basis;
  CeedElemRestriction Erestrict;

  ierr = CeedOperatorSetup_Magma(op); CeedChk(ierr);
  ierr= CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChk(ierr);

  // Input Evecs and Restriction
  for (CeedInt i=0; i<numinputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChk(ierr);
    if (emode & CEED_EVAL_WEIGHT) {
    } else { // Restriction
      // Get input vector
      ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
        vec = invec;
      // Restrict
      ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict);
      CeedChk(ierr);
      ierr = CeedOperatorFieldGetLMode(opinputfields[i], &lmode); CeedChk(ierr);
      ierr = CeedElemRestrictionApply(Erestrict, CEED_NOTRANSPOSE,
                                      lmode, vec, data->Evecs[i],
                                      request); CeedChk(ierr);
      // Get evec
      ierr = CeedVectorGetArrayRead(data->Evecs[i], CEED_MEM_HOST,
                                    (const CeedScalar **) &data->Edata[i]);
      CeedChk(ierr);
    }
  }

  // Output Evecs
  for (CeedInt i=0; i<numoutputfields; i++) {
    ierr = CeedVectorGetArray(data->Evecs[i+data->numein], CEED_MEM_HOST,
                              &data->Edata[i + numinputfields]); CeedChk(ierr);
  }

  // Loop through elements
  for (CeedInt e=0; e<numelements; e++) {
    // Input basis apply if needed
    for (CeedInt i=0; i<numinputfields; i++) {
      // Get elemsize, emode, ncomp
      ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict);
      CeedChk(ierr);
      ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
      CeedChk(ierr);
      ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
      CeedChk(ierr);
      ierr = CeedQFunctionFieldGetNumComponents(qfinputfields[i], &ncomp);
      CeedChk(ierr);
      // Basis action
      switch(emode) {
      case CEED_EVAL_NONE:
        ierr = CeedVectorSetArray(data->qvecsin[i], CEED_MEM_HOST,
                                  CEED_USE_POINTER,
                                  &data->Edata[i][e*Q*ncomp]); CeedChk(ierr);
        break;
      case CEED_EVAL_INTERP:
        ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChk(ierr);
        ierr = CeedVectorSetArray(data->evecsin[i], CEED_MEM_HOST,
                                  CEED_USE_POINTER,
                                  &data->Edata[i][e*elemsize*ncomp]);
        CeedChk(ierr);
        ierr = CeedBasisApply(basis, 1, CEED_NOTRANSPOSE,
                              CEED_EVAL_INTERP, data->evecsin[i],
                              data->qvecsin[i]); CeedChk(ierr);
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChk(ierr);
        ierr = CeedVectorSetArray(data->evecsin[i], CEED_MEM_HOST,
                                  CEED_USE_POINTER,
                                  &data->Edata[i][e*elemsize*ncomp]);
        CeedChk(ierr);
        ierr = CeedBasisApply(basis, 1, CEED_NOTRANSPOSE,
                              CEED_EVAL_GRAD, data->evecsin[i],
                              data->qvecsin[i]); CeedChk(ierr);
        break;
      case CEED_EVAL_WEIGHT:
        break;  // No action
      case CEED_EVAL_DIV:
        break; // Not implemented
      case CEED_EVAL_CURL:
        break; // Not implemented
      }
    }
    // Output pointers
    for (CeedInt i=0; i<numoutputfields; i++) {
      ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
      CeedChk(ierr);
      if (emode == CEED_EVAL_NONE) {
        ierr = CeedQFunctionFieldGetNumComponents(qfoutputfields[i], &ncomp);
        CeedChk(ierr);
        ierr = CeedVectorSetArray(data->qvecsout[i], CEED_MEM_HOST,
                                  CEED_USE_POINTER,
                                  &data->Edata[i + numinputfields][e*Q*ncomp]);
        CeedChk(ierr);
      }
      if (emode == CEED_EVAL_INTERP) {
      }
      if (emode == CEED_EVAL_GRAD) {
      }
      if (emode == CEED_EVAL_WEIGHT) {
      }
    }

    // Q function
    ierr = CeedQFunctionApply(qf, Q, data->qvecsin, data->qvecsout); CeedChk(ierr);

    // Output basis apply if needed
    for (CeedInt i=0; i<numoutputfields; i++) {
      // Get elemsize, emode, ncomp
      ierr = CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict);
      CeedChk(ierr);
      ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
      CeedChk(ierr);
      ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
      CeedChk(ierr);
      ierr = CeedQFunctionFieldGetNumComponents(qfoutputfields[i], &ncomp);
      CeedChk(ierr);
      // Basis action
      switch(emode) {
      case CEED_EVAL_NONE:
        break; // No action
      case CEED_EVAL_INTERP:
        ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis);
        CeedChk(ierr);
        ierr = CeedVectorSetArray(data->evecsout[i], CEED_MEM_HOST,
                                  CEED_USE_POINTER,
                                  &data->Edata[i + numinputfields][e*elemsize*ncomp]);
        ierr = CeedBasisApply(basis, 1, CEED_TRANSPOSE,
                              CEED_EVAL_INTERP, data->qvecsout[i],
                              data->evecsout[i]); CeedChk(ierr);
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis);
        CeedChk(ierr);
        ierr = CeedVectorSetArray(data->evecsout[i], CEED_MEM_HOST,
                                  CEED_USE_POINTER,
                                  &data->Edata[i + numinputfields][e*elemsize*ncomp]);
        ierr = CeedBasisApply(basis, 1, CEED_TRANSPOSE,
                              CEED_EVAL_GRAD, data->qvecsout[i],
                              data->evecsout[i]); CeedChk(ierr);
        break;
      case CEED_EVAL_WEIGHT: break; // Should not occur
      case CEED_EVAL_DIV: break; // Not implemented
      case CEED_EVAL_CURL: break; // Not implemented
      }
    }
  } // numelements

  // Zero lvecs
  for (CeedInt i=0; i<numoutputfields; i++) {
    ierr = CeedOperatorFieldGetVector(opoutputfields[i], &vec); CeedChk(ierr);
    if (vec == CEED_VECTOR_ACTIVE)
      vec = outvec;
    ierr = CeedVectorSetValue(vec, 0.0); CeedChk(ierr);
  }

  // Output restriction
  for (CeedInt i=0; i<numoutputfields; i++) {
    // Restore evec
    ierr = CeedVectorRestoreArray(data->Evecs[i+data->numein],
                                  &data->Edata[i + numinputfields]);
    CeedChk(ierr);
    // Get output vector
    ierr = CeedOperatorFieldGetVector(opoutputfields[i], &vec); CeedChk(ierr);
    // Active
    if (vec == CEED_VECTOR_ACTIVE)
      vec = outvec;
    // Restrict
    ierr = CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict);
    CeedChk(ierr);
    ierr = CeedOperatorFieldGetLMode(opoutputfields[i], &lmode); CeedChk(ierr);
    ierr = CeedElemRestrictionApply(Erestrict, CEED_TRANSPOSE,
                                    lmode, data->Evecs[i+data->numein], vec,
                                    request); CeedChk(ierr);
  }

  // Restore input arrays
  for (CeedInt i=0; i<numinputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChk(ierr);
    if (emode & CEED_EVAL_WEIGHT) {
    } else {
      // Restriction
      ierr = CeedVectorRestoreArrayRead(data->Evecs[i],
                                        (const CeedScalar **) &data->Edata[i]);
      CeedChk(ierr);
    }
  }
  return 0;
}

static int CeedOperatorCreate_Magma(CeedOperator op) {
  int ierr;
  CeedOperator_Magma *impl;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);

  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  ierr = CeedOperatorSetData(op, (void *)&impl);

  ierr = CeedSetBackendFunction(ceed, "Operator", op, "Apply",
                                CeedOperatorApply_Magma); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "Destroy",
                                CeedOperatorDestroy_Magma); CeedChk(ierr);
  return 0;
}

int CeedCompositeOperatorCreate_Magma(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  return CeedError(ceed, 1, "Backend does not support composite operators");
}

// *****************************************************************************
// * INIT
// *****************************************************************************
static int CeedInit_Magma(const char *resource, Ceed ceed) {
  int ierr;
  if (strcmp(resource, "/gpu/magma"))
    return CeedError(ceed, 1, "MAGMA backend cannot use resource: %s", resource);

  ierr = magma_init();
  if (ierr) return CeedError(ceed, 1, "error in magma_init(): %d\n", ierr);
  //magma_print_environment();

  ceed->VectorCreate = CeedVectorCreate_Magma;
  ceed->BasisCreateTensorH1 = CeedBasisCreateTensorH1_Magma;
  ceed->BasisCreateH1 = CeedBasisCreateH1_Magma;
  ceed->ElemRestrictionCreate = CeedElemRestrictionCreate_Magma;
  ceed->ElemRestrictionCreateBlocked = CeedElemRestrictionCreateBlocked_Magma;
  ceed->QFunctionCreate = CeedQFunctionCreate_Magma;
  ceed->OperatorCreate = CeedOperatorCreate_Magma;
  ceed->CompositeOperatorCreate = CeedCompositeOperatorCreate_Magma;
  return 0;
}

// *****************************************************************************
// * REGISTER
// *****************************************************************************
__attribute__((constructor))
static void Register(void) {
  CeedRegister("/gpu/magma", CeedInit_Magma, 20);
}
