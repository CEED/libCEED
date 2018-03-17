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

#include <ceed-impl.h>
#include <string.h>
#include "magma.h"

typedef struct {
    CeedScalar *array;
    CeedScalar *darray;
    int own_;
} CeedVector_Magma;

typedef struct {
    const CeedInt *indices;
    CeedInt *indices_allocated;
} CeedElemRestriction_Magma;

typedef struct {
    CeedVector etmp;
    CeedVector qdata;
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

    // If own data, free that "old" data, e.g., as it may be of different size 
    if (impl->own_){
        magma_free( impl->darray );
        magma_free_pinned( impl->array );
        impl->darray = NULL;
        impl->array  = NULL;
        impl->own_ = 0;
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
            
            if (array)
                magma_setvector(vec->length, sizeof(array[0]), 
                                array, 1, impl->darray, 1);
            break;
        case CEED_OWN_POINTER:
            ierr = magma_malloc( (void**)&impl->darray,
                                 vec->length * sizeof(CeedScalar)); CeedChk(ierr);
            impl->array = array;
            impl->own_ = 1;

            if (array)
                magma_setvector(vec->length, sizeof(array[0]),
                                array, 1, impl->darray, 1);
            break;
        case CEED_USE_POINTER:
            impl->darray = NULL;
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
        if (!impl->array){
            // Allocate data if array is NULL
            ierr = CeedVectorSetArray(vec, mtype, CEED_COPY_VALUES, NULL);
            CeedChk(ierr);
        } else if (impl->own_) { 
            // data is owned so GPU had the most up-to-date version; copy it 
            magma_getvector(vec->length, sizeof(*array[0]),
                            impl->darray, 1, impl->array, 1);
        }
        *array = impl->array;
    } else if (mtype == CEED_MEM_DEVICE) {
        if (!impl->darray){
            // Allocate data if darray is NULL 
            ierr = CeedVectorSetArray(vec, mtype, CEED_COPY_VALUES, NULL);
            CeedChk(ierr);
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
        if (!impl->array){
            // Allocate data if array is NULL
            ierr = CeedVectorSetArray(vec, mtype, CEED_COPY_VALUES, NULL);
            CeedChk(ierr);
        } else if (impl->own_) {
            // data is owned so GPU had the most up-to-date version; copy it
            magma_getvector(vec->length, sizeof(*array[0]),
                            impl->darray, 1, impl->array, 1);
        }
        *array = impl->array;
    } else if (mtype == CEED_MEM_DEVICE) {
        if (!impl->darray){
            // Allocate data if darray is NULL
            ierr = CeedVectorSetArray(vec, mtype, CEED_COPY_VALUES, NULL);
            CeedChk(ierr);
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

    if (impl->darray!=NULL)
        magma_setvector(vec->length, sizeof(*array[0]),
                        *array, 1, impl->darray, 1);

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

    if (impl->darray!=NULL)
        magma_setvector(vec->length, sizeof(*array[0]),
                        *array, 1, impl->darray, 1);
    
    *array = NULL;
    return 0;
}

static int CeedVectorDestroy_Magma(CeedVector vec) {
    CeedVector_Magma *impl = vec->data;
    int ierr;

    // Free if we own the data
    if (impl->own_){
        ierr = magma_free_pinned(impl->array); CeedChk(ierr);
        ierr = magma_free(impl->darray);       CeedChk(ierr);
    }
    ierr = CeedFree(&vec->data); CeedChk(ierr);
    return 0;
}

// *****************************************************************************
// * Create vector vec of size n
// *****************************************************************************  
static int CeedVectorCreate_Magma(Ceed ceed, CeedInt n, CeedVector vec) {
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
    vec->data = impl;
    return 0;
}

// *****************************************************************************
// * Apply restriction operator r to u: v = r(rmode) u 
// *****************************************************************************
static int CeedElemRestrictionApply_Magma(CeedElemRestriction r,
                                        CeedTransposeMode tmode, CeedInt ncomp,
                                        CeedTransposeMode lmode, CeedVector u,
                                        CeedVector v, CeedRequest *request) {
    CeedElemRestriction_Magma *impl = r->data;
    int ierr;
    const CeedScalar *uu;
    CeedScalar *vv;
    CeedInt esize = r->nelem*r->elemsize;
    //printf("HELLOOOOOOOOO=======================\n");

    ierr = CeedVectorGetArrayRead(u, CEED_MEM_HOST, &uu); CeedChk(ierr);
    ierr = CeedVectorGetArray(v, CEED_MEM_HOST, &vv); CeedChk(ierr);
    if (tmode == CEED_NOTRANSPOSE) {
        // Perform: v = r * u
        if (ncomp == 1) {
            for (CeedInt i=0; i<esize; i++) vv[i] = uu[impl->indices[i]];
            
            /*
            // Works - in t05 x has to be with CEED_COPY_VALUES
            CeedVector_Magma *uimpl = u->data, *vimpl = v->data;
            uu = uimpl->darray;
            vv = vimpl->darray;
            CeedInt *indices;// = (int *)impl->indices;
            magma_malloc( (void**)&indices, esize * sizeof(CeedInt)); 
            magma_setvector(esize, sizeof(CeedInt),
                            (int *)impl->indices, 1, indices, 1);
            magma_template<<i=0:esize>>(const CeedScalar *uu, CeedScalar *vv, CeedInt *indices)
              {
                 vv[i] = uu[indices[i]];  
              }
             
            // printf("HELLOOOOOOOOO....... size = %d\n", esize);
            //
            
            if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED)
                *request = NULL;
            return 0;
            */
 
        } else {
            //printf("HELLOOOOOOOOO-------------\n");
            // vv is (elemsize x ncomp x nelem), column-major
            if (lmode == CEED_NOTRANSPOSE) { // u is (ndof x ncomp), column-major
                for (CeedInt e = 0; e < r->nelem; e++)
                    for (CeedInt d = 0; d < ncomp; d++)
                        for (CeedInt i=0; i<r->elemsize; i++) {
                            vv[i+r->elemsize*(d+ncomp*e)] =
                                uu[impl->indices[i+r->elemsize*e]+r->ndof*d];
                        }
            } else { // u is (ncomp x ndof), column-major
                for (CeedInt e = 0; e < r->nelem; e++)
                    for (CeedInt d = 0; d < ncomp; d++)
                        for (CeedInt i=0; i<r->elemsize; i++) {
                            vv[i+r->elemsize*(d+ncomp*e)] =
                                uu[d+ncomp*impl->indices[i+r->elemsize*e]];
                        }
            }
        }
    } else {
        // Note: in transpose mode, we perform: v += r^t * u
        if (ncomp == 1) {
            for (CeedInt i=0; i<esize; i++) vv[impl->indices[i]] += uu[i];
        } else {
            //printf("HELLOOOOOOOOO+++++++++++++\n");
            // u is (elemsize x ncomp x nelem)
            if (lmode == CEED_NOTRANSPOSE) { // vv is (ndof x ncomp), column-major
                for (CeedInt e = 0; e < r->nelem; e++)
                    for (CeedInt d = 0; d < ncomp; d++)
                        for (CeedInt i=0; i<r->elemsize; i++) {
                            vv[impl->indices[i+r->elemsize*e]+r->ndof*d] +=
                                uu[i+r->elemsize*(d+e*ncomp)];
                        }
            } else { // vv is (ncomp x ndof), column-major
                for (CeedInt e = 0; e < r->nelem; e++)
                    for (CeedInt d = 0; d < ncomp; d++)
                        for (CeedInt i=0; i<r->elemsize; i++) {
                            vv[d+ncomp*impl->indices[i+r->elemsize*e]] +=
                                uu[i+r->elemsize*(d+e*ncomp)];
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

static int CeedElemRestrictionDestroy_Magma(CeedElemRestriction r) {
  CeedElemRestriction_Magma *impl = r->data;
  int ierr;

  ierr = CeedFree(&impl->indices_allocated); CeedChk(ierr);
  ierr = CeedFree(&r->data); CeedChk(ierr);
  return 0;
}

static int CeedElemRestrictionCreate_Magma(CeedElemRestriction r,
                                           CeedMemType mtype,
                                           CeedCopyMode cmode, 
                                           const CeedInt *indices) {
    int ierr;
    CeedElemRestriction_Magma *impl;

    if (mtype != CEED_MEM_HOST)
        return CeedError(r->ceed, 1, "Only MemType = HOST supported");
    ierr = CeedCalloc(1,&impl); CeedChk(ierr);
    switch (cmode) {
    case CEED_COPY_VALUES:
        //printf("CeedElemRestriction_Magma COPY\n");
        ierr = CeedMalloc(r->nelem*r->elemsize, &impl->indices_allocated);
        CeedChk(ierr);
        memcpy(impl->indices_allocated, indices,
               r->nelem * r->elemsize * sizeof(indices[0]));
        impl->indices = impl->indices_allocated;
        break;
    case CEED_OWN_POINTER:
        //printf("CeedElemRestriction_Magma OWN\n");
        impl->indices_allocated = (CeedInt *)indices;
        impl->indices = impl->indices_allocated;
        break;
    case CEED_USE_POINTER:
        //printf("CeedElemRestriction_Magma USE\n");
        impl->indices = indices;
    }
    r->data = impl;
    r->Apply = CeedElemRestrictionApply_Magma;
    r->Destroy = CeedElemRestrictionDestroy_Magma;
    return 0;
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
    CeedInt tstride0 = B, tstride1 = 1;
    if (tmode == CEED_TRANSPOSE) {
        tstride0 = 1; tstride1 = J;
    }

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
    return 0;
}

static int CeedBasisApply_Magma(CeedBasis basis, CeedTransposeMode tmode,
                                CeedEvalMode emode,
                                const CeedScalar *u, CeedScalar *v) {
    int ierr;
    const CeedInt dim = basis->dim;
    const CeedInt ndof = basis->ndof;
    const CeedInt nqpt = ndof*CeedPowInt(basis->Q1d, dim);
    const CeedInt add = (tmode == CEED_TRANSPOSE);

    if (tmode == CEED_TRANSPOSE) {
        const CeedInt vsize = ndof*CeedPowInt(basis->P1d, dim);
        for (CeedInt i = 0; i < vsize; i++)
            v[i] = (CeedScalar) 0;
    }
    if (emode & CEED_EVAL_INTERP) {
        CeedInt P = basis->P1d, Q = basis->Q1d;
        if (tmode == CEED_TRANSPOSE) {
            P = basis->Q1d; Q = basis->P1d;
        }
        CeedInt pre = ndof*CeedPowInt(P, dim-1), post = 1;
        CeedScalar tmp[2][ndof*Q*CeedPowInt(P>Q?P:Q, dim-1)];
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
        // u is (P^dim x nc), column-major layout (nc = ndof)
        // v is (Q^dim x nc x dim), column-major layout (nc = ndof)
        // In CEED_TRANSPOSE mode, the sizes of u and v are switched.
        if (tmode == CEED_TRANSPOSE) {
            P = basis->Q1d, Q = basis->P1d;
        }
        CeedScalar tmp[2][ndof*Q*CeedPowInt(P>Q?P:Q, dim-1)];
        for (CeedInt p = 0; p < dim; p++) {
            CeedInt pre = ndof*CeedPowInt(P, dim-1), post = 1;
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

static int CeedBasisCreateTensorH1_Magma(Ceed ceed, CeedInt dim, CeedInt P1d,
                                         CeedInt Q1d, const CeedScalar *interp1d,
                                         const CeedScalar *grad1d,
                                         const CeedScalar *qref1d,
                                         const CeedScalar *qweight1d,
                                         CeedBasis basis) {
    basis->Apply = CeedBasisApply_Magma;
    basis->Destroy = CeedBasisDestroy_Magma;
    return 0;
}

static int CeedQFunctionApply_Magma(CeedQFunction qf, void *qdata, CeedInt Q,
                                    const CeedScalar *const *u,
                                    CeedScalar *const *v) {
    int ierr;
    ierr = qf->function(qf->ctx, qdata, Q, u, v); CeedChk(ierr);
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

    ierr = CeedVectorDestroy(&impl->etmp); CeedChk(ierr);
    ierr = CeedVectorDestroy(&impl->qdata); CeedChk(ierr);
    ierr = CeedFree(&op->data); CeedChk(ierr);
    return 0;
}

static int CeedOperatorApply_Magma(CeedOperator op, CeedVector qdata,
                                   CeedVector ustate,
                                   CeedVector residual, CeedRequest *request) {
    CeedOperator_Magma *impl = op->data;
    CeedVector etmp;
    CeedInt Q;
    const CeedInt nc = op->basis->ndof, dim = op->basis->dim;
    CeedScalar *Eu;
    char *qd;
    int ierr;
    CeedTransposeMode lmode = CEED_NOTRANSPOSE;
    
    if (!impl->etmp) {
        ierr = CeedVectorCreate(op->ceed,
                                nc * op->Erestrict->nelem * op->Erestrict->elemsize,
                                &impl->etmp); CeedChk(ierr);
        // etmp is allocated when CeedVectorGetArray is called below
    }
    etmp = impl->etmp;
    if (op->qf->inmode & ~CEED_EVAL_WEIGHT) {
        ierr = CeedElemRestrictionApply(op->Erestrict, CEED_NOTRANSPOSE,
                                        nc, lmode, ustate, etmp,
                                        CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
    }
    ierr = CeedBasisGetNumQuadraturePoints(op->basis, &Q); CeedChk(ierr);
    ierr = CeedVectorGetArray(etmp, CEED_MEM_HOST, &Eu); CeedChk(ierr);
    ierr = CeedVectorGetArray(qdata, CEED_MEM_HOST, (CeedScalar**)&qd);
    CeedChk(ierr);
    for (CeedInt e=0; e<op->Erestrict->nelem; e++) {
        CeedScalar BEu[Q*nc*(dim+2)], BEv[Q*nc*(dim+2)], *out[5] = {0,0,0,0,0};
        const CeedScalar *in[5] = {0,0,0,0,0};
        // TODO: quadrature weights can be computed just once
        ierr = CeedBasisApply(op->basis, CEED_NOTRANSPOSE, op->qf->inmode,
                              &Eu[e*op->Erestrict->elemsize*nc], BEu);
        CeedChk(ierr);
        CeedScalar *u_ptr = BEu, *v_ptr = BEv;
        if (op->qf->inmode & CEED_EVAL_INTERP) { in[0] = u_ptr; u_ptr += Q*nc; }
        if (op->qf->inmode & CEED_EVAL_GRAD) { in[1] = u_ptr; u_ptr += Q*nc*dim; }
        if (op->qf->inmode & CEED_EVAL_WEIGHT) { in[4] = u_ptr; u_ptr += Q; }
        if (op->qf->outmode & CEED_EVAL_INTERP) { out[0] = v_ptr; v_ptr += Q*nc; }
        if (op->qf->outmode & CEED_EVAL_GRAD) { out[1] = v_ptr; v_ptr += Q*nc*dim; }
        ierr = CeedQFunctionApply(op->qf, &qd[e*Q*op->qf->qdatasize], Q, in, out);
        CeedChk(ierr);
        ierr = CeedBasisApply(op->basis, CEED_TRANSPOSE, op->qf->outmode, BEv,
                              &Eu[e*op->Erestrict->elemsize*nc]);
        CeedChk(ierr);
    }
    ierr = CeedVectorRestoreArray(etmp, &Eu); CeedChk(ierr);
    if (residual) {
        CeedScalar *res;
        CeedVectorGetArray(residual, CEED_MEM_HOST, &res);
        for (int i = 0; i < residual->length; i++)
            res[i] = (CeedScalar)0;
        ierr = CeedElemRestrictionApply(op->Erestrict, CEED_TRANSPOSE,
                                        nc, lmode, etmp, residual,
                                        CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
    }
    if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED)
        *request = NULL;
    return 0;
}

static int CeedOperatorGetQData_Magma(CeedOperator op, CeedVector *qdata) {
    CeedOperator_Magma *impl = op->data;
    int ierr;

    if (!impl->qdata) {
        CeedInt Q;
        ierr = CeedBasisGetNumQuadraturePoints(op->basis, &Q); CeedChk(ierr);
        ierr = CeedVectorCreate(op->ceed,
                                op->Erestrict->nelem * Q
                                * op->qf->qdatasize / sizeof(CeedScalar),
                                &impl->qdata); CeedChk(ierr);
    }
    *qdata = impl->qdata;
    return 0;
}

static int CeedOperatorCreate_Magma(CeedOperator op) {
    CeedOperator_Magma *impl;
    int ierr;
    
    ierr = CeedCalloc(1, &impl); CeedChk(ierr);
    op->data = impl;
    op->Destroy = CeedOperatorDestroy_Magma;
    op->Apply = CeedOperatorApply_Magma;
    op->GetQData = CeedOperatorGetQData_Magma;
    return 0;
}

// *****************************************************************************
// * INIT
// *****************************************************************************
static int CeedInit_Magma(const char *resource, Ceed ceed) {
    if (strcmp(resource, "/cpu/magma")
        && strcmp(resource, "/gpu/magma"))
        return CeedError(ceed, 1, "MAGMA backend cannot use resource: %s", resource);

    magma_init();
    //magma_print_environment();

    ceed->VecCreate = CeedVectorCreate_Magma;
    ceed->BasisCreateTensorH1 = CeedBasisCreateTensorH1_Magma;
    ceed->ElemRestrictionCreate = CeedElemRestrictionCreate_Magma;
    ceed->QFunctionCreate = CeedQFunctionCreate_Magma;
    ceed->OperatorCreate = CeedOperatorCreate_Magma;
    return 0;
}

// *****************************************************************************
// * REGISTER
// *****************************************************************************
__attribute__((constructor))
static void Register(void) {
    CeedRegister("/cpu/magma", CeedInit_Magma);
    CeedRegister("/gpu/magma", CeedInit_Magma);
}
