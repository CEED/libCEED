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
  CeedVector
  *evecs;   /// E-vectors needed to apply operator (input followed by outputs)
  CeedScalar **edata;
  CeedScalar **qdata; /// Inputs followed by outputs
  CeedScalar
  **qdata_alloc; /// Allocated quadrature data arrays (to be freed by us)
  CeedScalar **indata;
  CeedScalar **outdata;
  CeedInt    numein;
  CeedInt    numeout;
  CeedInt    numqin;
  CeedInt    numqout;
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
      ierr = magma_malloc( (void**)&impl->darray,
                           vec->length * sizeof(CeedScalar)); CeedChk(ierr);
      ierr = magma_malloc_pinned( (void**)&impl->array,
                                  vec->length * sizeof(CeedScalar)); CeedChk(ierr);
      impl->own_ = 1;

      if (array != NULL)
        magma_setvector(vec->length, sizeof(array[0]),
                        array, 1, impl->darray, 1);
      break;
    case CEED_OWN_POINTER:
      ierr = magma_malloc( (void**)&impl->darray,
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
      ierr = magma_malloc( (void**)&impl->darray,
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
      ierr = magma_malloc( (void**)&impl->darray,
                           vec->length * sizeof(CeedScalar)); CeedChk(ierr);
      ierr = magma_malloc_pinned( (void**)&impl->array,
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
      ierr = magma_malloc_pinned( (void**)&impl->array,
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
static int CeedVectorRestoreArray_Magma(CeedVector vec, CeedScalar **array) {
  CeedVector_Magma *impl = vec->data;

  // Check if the array is a CPU pointer
  if (*array == impl->array) {
    // Update device, if the device pointer is not NULL
    if (impl->darray != NULL) {
      magma_setvector(vec->length, sizeof(*array[0]),
                      *array, 1, impl->darray, 1);
    } else {
      // nothing to do (case of CPU use pointer)
    }

  } else if (impl->down_) {
    // nothing to do if array is on GPU, except if down_=1(case CPU use pointer)
    magma_getvector(vec->length, sizeof(*array[0]),
                    impl->darray, 1, impl->array, 1);
  }

  *array = NULL;
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
static int CeedVectorRestoreArrayRead_Magma(CeedVector vec,
    const CeedScalar **array) {
  CeedVector_Magma *impl = vec->data;

  // Check if the array is a CPU pointer
  if (*array == impl->array) {
    // Update device, if the device pointer is not NULL
    if (impl->darray != NULL) {
      magma_setvector(vec->length, sizeof(*array[0]),
                      *array, 1, impl->darray, 1);
    } else {
      // nothing to do (case of CPU use pointer)
    }

  } else if (impl->down_) {
    // nothing to do if array is on GPU, except if down_=1(case CPU use pointer)
    magma_getvector(vec->length, sizeof(*array[0]),
                    impl->darray, 1, impl->array, 1);
  }

  *array = NULL;
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
      ierr = magma_malloc( (void**)&impl->dindices,
                           size * sizeof(CeedInt)); CeedChk(ierr);
      ierr = magma_malloc_pinned( (void**)&impl->indices,
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

  } else
    return CeedError(r->ceed, 1, "Only MemType = HOST or DEVICE supported");

  r->data    = impl;
  r->Apply   = CeedElemRestrictionApply_Magma;
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
                                const CeedScalar *u, CeedScalar *v) {
  int ierr;
  const CeedInt dim = basis->dim;
  const CeedInt ncomp = basis->ncomp;
  const CeedInt nqpt = ncomp*CeedPowInt(basis->Q1d, dim);
  const CeedInt add = (tmode == CEED_TRANSPOSE);

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
                                    const CeedScalar *const *u,
                                    CeedScalar *const *v) {
  int ierr;
  ierr = qf->function(qf->ctx, Q, u, v); CeedChk(ierr);
  return 0;
}

static int CeedQFunctionDestroy_Magma(CeedQFunction qf) {
  return 0;
}

static int CeedQFunctionCreate_Magma(CeedQFunction qf) {
  qf->Apply = CeedQFunctionApply_Magma;
  qf->Destroy = CeedQFunctionDestroy_Magma;
  return 0;
}

static int CeedOperatorDestroy_Magma(CeedOperator op) {
  CeedOperator_Magma *impl = op->data;
  int ierr;

  for (CeedInt i=0; i<impl->numein+impl->numeout; i++) {
    ierr = CeedVectorDestroy(&impl->evecs[i]); CeedChk(ierr);
  }

  ierr = CeedFree(&impl->evecs); CeedChk(ierr);
  ierr = CeedFree(&impl->edata); CeedChk(ierr);

  for (CeedInt i=0; i<impl->numqin+impl->numqout; i++) {
    ierr = CeedFree(&impl->qdata_alloc[i]); CeedChk(ierr);
  }

  ierr = CeedFree(&impl->qdata_alloc); CeedChk(ierr);
  ierr = CeedFree(&impl->qdata); CeedChk(ierr);

  ierr = CeedFree(&impl->indata); CeedChk(ierr);
  ierr = CeedFree(&impl->outdata); CeedChk(ierr);

  ierr = CeedFree(&op->data); CeedChk(ierr);
  return 0;
}


/*
  Setup infields or outfields
 */
static int CeedOperatorSetupFields_Magma(struct CeedQFunctionField qfields[16],
                                       struct CeedOperatorField ofields[16],
                                       CeedVector *evecs, CeedScalar **qdata,
                                       CeedScalar **qdata_alloc, CeedScalar **indata,
                                       CeedInt starti, CeedInt startq,
                                       CeedInt numfields, CeedInt Q) {
  CeedInt dim, ierr, iq=startq, ncomp;

  // Loop over fields
  for (CeedInt i=0; i<numfields; i++) {
    CeedEvalMode emode = qfields[i].emode;
    if (emode != CEED_EVAL_WEIGHT) {
      ierr = CeedElemRestrictionCreateVector(ofields[i].Erestrict, NULL, &evecs[i]);
      CeedChk(ierr);
    }
    switch(emode) {
    case CEED_EVAL_NONE:
      break; // No action
    case CEED_EVAL_INTERP:
      ncomp = qfields[i].ncomp;
      ierr = CeedMalloc(Q*ncomp, &qdata_alloc[iq]); CeedChk(ierr);
      qdata[i + starti] = qdata_alloc[iq];
      iq++;
      break;
    case CEED_EVAL_GRAD:
      ncomp = qfields[i].ncomp;
      dim = ofields[i].basis->dim;
      ierr = CeedMalloc(Q*ncomp*dim, &qdata_alloc[iq]); CeedChk(ierr);
      qdata[i + starti] = qdata_alloc[iq];
      iq++;
      break;
    case CEED_EVAL_WEIGHT: // Only on input fields
      ierr = CeedMalloc(Q, &qdata_alloc[iq]); CeedChk(ierr);
      ierr = CeedBasisApply(ofields[iq].basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT,
                            NULL, qdata_alloc[iq]); CeedChk(ierr);
      qdata[i] = qdata_alloc[iq];
      indata[i] = qdata[i];
      iq++;
      break;
    case CEED_EVAL_DIV:
      break; // Not implimented
    case CEED_EVAL_CURL:
      break; // Not implimented
    }
  }
  return 0;
}

/*
  CeedOperator needs to connect all the named fields (be they active or passive)
  to the named inputs and outputs of its CeedQFunction.
 */
static int CeedOperatorSetup_Magma(CeedOperator op) {
  if (op->setupdone) return 0;
  CeedOperator_Magma *opmagma = op->data;
  CeedQFunction qf = op->qf;
  CeedInt Q = op->numqpoints;
  int ierr;

  // Count infield and outfield array sizes and evectors
  opmagma->numein = qf->numinfutfields;
  for (CeedInt i=0; i<qf->numinputfields; i++) {
    CeedEvalMode emode = qf->inputfields[i].emode;
    opmagma->numqin += !!(emode & CEED_EVAL_INTERP) + !!(emode & CEED_EVAL_GRAD) + !!
                     (emode & CEED_EVAL_WEIGHT);
  }
  qpmagma->numeout = qf->numoutputfields;
  for (CeedInt i=0; i<qf->numoutputfields; i++) {
    CeedEvalMode emode = qf->outputfields[i].emode;
    opmagma->numqout += !!(emode & CEED_EVAL_INTERP) + !!(emode & CEED_EVAL_GRAD);
  }

  // Allocate
  ierr = CeedCalloc(opmagma->numein + opmagma->numeout, &opmagma->evecs); CeedChk(ierr);
  ierr = CeedCalloc(opmagma->numein + opmagma->numeout, &opmagma->edata);
  CeedChk(ierr);

  ierr = CeedCalloc(opmagma->numqin + opmagma->numqout, &opmagma->qdata_alloc);
  CeedChk(ierr);
  ierr = CeedCalloc(qf->numinputfields + qf->numoutputfields, &opmagma->qdata);
  CeedChk(ierr);

  ierr = CeedCalloc(16, &opmagma->indata); CeedChk(ierr);
  ierr = CeedCalloc(16, &opmagma->outdata); CeedChk(ierr);

  // Set up infield and outfield pointer arrays
  // Infields
  ierr = CeedOperatorSetupFields_Magma(qf->inputfields, op->inputfields,
                                     opmagma->evecs, opmagma->qdata, opmagma->qdata_alloc,
                                     opmagma->indata, 0, 0,
                                     qf->numinputfields, Q); CeedChk(ierr);

  // Outfields
  ierr = CeedOperatorSetupFields_Magma(qf->outputfields, op->outputfields,
                                     opmagma->evecs, opmagma->qdata, opmagma->qdata_alloc,
                                     opmagma->indata, qf->numinputfields,
                                     opmagma->numqin, qf->numoutputfields, Q); CeedChk(ierr);

  // Output Qvecs
  for (CeedInt i=0; i<qf->numoutputfields; i++) {
    CeedEvalMode emode = qf->outputfields[i].emode;
    if (emode != CEED_EVAL_NONE) {
      opmagma->outdata[i] =  opmagma->qdata[i + qf->numinputfields];
    }
  }

  op->setupdone = 1;

  return 0;
}

static int CeedOperatorApply_Magma(CeedOperator op, CeedVector invec,
                                 CeedVector outvec, CeedRequest *request) {
  CeedOperator_Magma *opmagma = op->data;
  CeedInt Q = op->numqpoints, elemsize;
  int ierr;
  CeedQFunction qf = op->qf;
  CeedTransposeMode lmode = CEED_NOTRANSPOSE;
  CeedScalar *vec_temp;

  // Setup
  ierr = CeedOperatorSetup_Magma(op); CeedChk(ierr);

  // Input Evecs and Restriction
  for (CeedInt i=0; i<qf->numinputfields; i++) {
    CeedEvalMode emode = qf->inputfields[i].emode;
    if (emode & CEED_EVAL_WEIGHT) { // Skip
    } else {
      // Zero evec
      ierr = CeedVectorGetArray(opmagma->evecs[i], CEED_MEM_HOST, &vec_temp);
      CeedChk(ierr);
      for (CeedInt j=0; j<opmagma->evecs[i]->length; j++)
        vec_temp[j] = 0.;
      ierr = CeedVectorRestoreArray(opmagma->evecs[i], &vec_temp); CeedChk(ierr);
      // Active
      if (op->inputfields[i].vec == CEED_VECTOR_ACTIVE) {
        // Restrict
        ierr = CeedElemRestrictionApply(op->inputfields[i].Erestrict, CEED_NOTRANSPOSE,
                                        lmode, invec, opmagma->evecs[ieiin],
                                        request); CeedChk(ierr);
        // Get evec
        ierr = CeedVectorGetArrayRead(opmagma->evecs[i], CEED_MEM_HOST,
                                      (const CeedScalar **) &opmagma->edata[i]); CeedChk(ierr);
      } else {
        // Passive
        // Restrict
        ierr = CeedElemRestrictionApply(op->inputfields[i].Erestrict, CEED_NOTRANSPOSE,
                                        lmode, op->inputfields[i].vec, opmagma->evecs[i],
                                        request); CeedChk(ierr);
        // Get evec
        ierr = CeedVectorGetArrayRead(opmagma->evecs[i], CEED_MEM_HOST,
                                      (const CeedScalar **) &opmagma->edata[i]); CeedChk(ierr);
      }
    }
  }

  // Output Evecs
  for (CeedInt i=0; i<qf->numoutputfields; i++) {
    ierr = CeedVectorGetArray(opmagma->evecs[i+opmagma->numein], CEED_MEM_HOST,
                              &opmagma->edata[i + qf->numinputfields]);
    CeedChk(ierr);
  }

  // Loop through elements
  for (CeedInt e=0; e<op->numelements; e++) {
    // Input basis apply if needed
    for (CeedInt i=0; i<qf->numinputfields; i++) {
      // Get elemsize, emode, ncomp
      elemsize = op->inputfields[i].Erestrict->elemsize;
      CeedEvalMode emode = qf->inputfields[i].emode;
      CeedInt ncomp = qf->inputfields[i].ncomp;
      // Basis action
      switch(emode) {
      case CEED_EVAL_NONE:
        opmagma->indata[i] = &opmagma->edata[i][e*Q*ncomp];
        break;
      case CEED_EVAL_INTERP:
        ierr = CeedBasisApply(op->inputfields[i].basis, 1, CEED_NOTRANSPOSE,
                              CEED_EVAL_INTERP, &opmagma->edata[i][e*elemsize*ncomp], opmagma->qdata[i]);
        CeedChk(ierr);
        opmagma->indata[i] = opmagma->qdata[i];
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedBasisApply(op->inputfields[i].basis, 1, CEED_NOTRANSPOSE,
                              CEED_EVAL_GRAD, &opmagma->edata[i][e*elemsize*ncomp], opmagma->qdata[i]);
        CeedChk(ierr);
        opmagma->indata[i] = opmagma->qdata[i];
        break;
      case CEED_EVAL_WEIGHT:
        break;  // No action
      case CEED_EVAL_DIV:
        break; // Not implimented
      case CEED_EVAL_CURL:
        break; // Not implimented
      }
    }
    // Output pointers
    for (CeedInt i=0; i<qf->numoutputfields; i++) {
      CeedEvalMode emode = qf->outputfields[i].emode;
      if (emode == CEED_EVAL_NONE) {
        CeedInt ncomp = qf->outputfields[i].ncomp;
        opmagma->outdata[i] = &opmagma->edata[i + qf->numinputfields][e*Q*ncomp];
      }
    }
    // Q function
    ierr = CeedQFunctionApply(op->qf, Q, (const CeedScalar * const*) opmagma->indata,
                              opmagma->outdata); CeedChk(ierr);

    // Output basis apply if needed
    for (CeedInt i=0; i<qf->numoutputfields; i++) {
      // Get elemsize, emode, ncomp
      elemsize = op->outputfields[i].Erestrict->elemsize;
      CeedInt ncomp = qf->outputfields[i].ncomp;
      CeedEvalMode emode = qf->outputfields[i].emode;
      // Basis action
      switch(emode) {
      case CEED_EVAL_NONE:
        break; // No action
      case CEED_EVAL_INTERP:
        ierr = CeedBasisApply(op->outputfields[i].basis, 1, CEED_TRANSPOSE,
                              CEED_EVAL_INTERP, opmagma->outdata[i],
                              &opmagma->edata[i + qf->numinputfields][e*elemsize*ncomp]); CeedChk(ierr);
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedBasisApply(op->outputfields[i].basis, 1, CEED_TRANSPOSE, CEED_EVAL_GRAD,
                              opmagma->outdata[i], &opmagma->edata[i + qf->numinputfields][e*elemsize*ncomp]);
        CeedChk(ierr);
        break;
      case CEED_EVAL_WEIGHT:
        break; // Should not occur
      case CEED_EVAL_DIV:
        break; // Not implimented
      case CEED_EVAL_CURL:
        break; // Not implimented
      }
    }
  }

  // Output restriction
  for (CeedInt i=0; i<qf->numoutputfields; i++) {
    // Active
    if (op->outputfields[i].vec == CEED_VECTOR_ACTIVE) {
      // Restore evec
      ierr = CeedVectorRestoreArray(opmagma->evecs[i+opmagma->numein],
                                    &opmagma->edata[i + qf->numinputfields]); CeedChk(ierr);
      // Zero lvec
      ierr = CeedVectorGetArray(outvec, CEED_MEM_HOST, &vec_temp); CeedChk(ierr);
      for (CeedInt j=0; j<outvec->length; j++)
        vec_temp[j] = 0.;
      ierr = CeedVectorRestoreArray(outvec, &vec_temp); CeedChk(ierr);
      // Restrict
      ierr = CeedElemRestrictionApply(op->outputfields[i].Erestrict, CEED_TRANSPOSE,
                                      lmode, opmagma->evecs[i+opmagma->numein], outvec, request); CeedChk(ierr);
    } else {
      // Passive
      // Restore evec
      ierr = CeedVectorRestoreArray(opmagma->evecs[i+opmagma->numein],
                                    &opmagma->edata[i + qf->numinputfields]); CeedChk(ierr);
      // Zero lvec
      ierr = CeedVectorGetArray(op->outputfields[i].vec, CEED_MEM_HOST, &vec_temp);
      CeedChk(ierr);
      for (CeedInt j=0; j<op->outputfields[i].vec->length; j++)
        vec_temp[j] = 0.;
      ierr = CeedVectorRestoreArray(op->outputfields[i].vec, &vec_temp);
      CeedChk(ierr);
      // Restrict
      ierr = CeedElemRestrictionApply(op->outputfields[i].Erestrict, CEED_TRANSPOSE,
                                      lmode, opmagma->evecs[i+opmagma->numein], op->outputfields[i].vec,
                                      request); CeedChk(ierr);
    }
  }

  // Restore input arrays
  for (CeedInt i=0; i<qf->numinputfields; i++) {
    CeedEvalMode emode = qf->inputfields[i].emode;
    if (emode & CEED_EVAL_WEIGHT) {
    } else {
      ierr = CeedVectorRestoreArrayRead(opmagma->evecs[i],
                                        (const CeedScalar **) &opmagma->edata[i]); CeedChk(ierr);
    }
  }

  return 0;
}

static int CeedOperatorCreate_Magma(CeedOperator op) {
  CeedOperator_Magma *impl;
  int ierr;

  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  op->data = impl;
  op->Destroy  = CeedOperatorDestroy_Magma;
  op->Apply    = CeedOperatorApply_Magma;
  return 0;
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

  ceed->VecCreate = CeedVectorCreate_Magma;
  ceed->BasisCreateTensorH1 = CeedBasisCreateTensorH1_Magma;
  ceed->BasisCreateH1 = CeedBasisCreateH1_Magma;
  ceed->ElemRestrictionCreate = CeedElemRestrictionCreate_Magma;
  ceed->ElemRestrictionCreateBlocked = CeedElemRestrictionCreateBlocked_Magma;
  ceed->QFunctionCreate = CeedQFunctionCreate_Magma;
  ceed->OperatorCreate = CeedOperatorCreate_Magma;
  return 0;
}

// *****************************************************************************
// * REGISTER
// *****************************************************************************
__attribute__((constructor))
static void Register(void) {
  CeedRegister("/gpu/magma", CeedInit_Magma, 20);
}
