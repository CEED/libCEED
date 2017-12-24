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

typedef struct {
  CeedScalar *array;
  CeedScalar *array_allocated;
} CeedVector_Ref;

typedef struct {
  const CeedInt *indices;
  CeedInt *indices_allocated;
} CeedElemRestriction_Ref;

typedef struct {
  CeedVector etmp;
  CeedVector qdata;
} CeedOperator_Ref;

static int CeedVectorSetArray_Ref(CeedVector vec, CeedMemType mtype,
                                  CeedCopyMode cmode, CeedScalar *array) {
  CeedVector_Ref *impl = vec->data;
  int ierr;

  if (mtype != CEED_MEM_HOST)
    return CeedError(vec->ceed, 1, "Only MemType = HOST supported");
  ierr = CeedFree(&impl->array_allocated); CeedChk(ierr);
  switch (cmode) {
  case CEED_COPY_VALUES:
    ierr = CeedMalloc(vec->length, &impl->array_allocated); CeedChk(ierr);
    impl->array = impl->array_allocated;
    if (array) memcpy(impl->array, array, vec->length * sizeof(array[0]));
    break;
  case CEED_OWN_POINTER:
    impl->array_allocated = array;
    impl->array = array;
    break;
  case CEED_USE_POINTER:
    impl->array = array;
  }
  return 0;
}

static int CeedVectorGetArray_Ref(CeedVector vec, CeedMemType mtype,
                                  CeedScalar **array) {
  CeedVector_Ref *impl = vec->data;
  int ierr;

  if (mtype != CEED_MEM_HOST)
    return CeedError(vec->ceed, 1, "Can only provide to HOST memory");
  if (!impl->array) { // Allocate if array is not yet allocated
    ierr = CeedVectorSetArray(vec, CEED_MEM_HOST, CEED_COPY_VALUES, NULL);
    CeedChk(ierr);
  }
  *array = impl->array;
  return 0;
}

static int CeedVectorGetArrayRead_Ref(CeedVector vec, CeedMemType mtype,
                                      const CeedScalar **array) {
  CeedVector_Ref *impl = vec->data;
  int ierr;

  if (mtype != CEED_MEM_HOST)
    return CeedError(vec->ceed, 1, "Can only provide to HOST memory");
  if (!impl->array) { // Allocate if array is not yet allocated
    ierr = CeedVectorSetArray(vec, CEED_MEM_HOST, CEED_COPY_VALUES, NULL);
    CeedChk(ierr);
  }
  *array = impl->array;
  return 0;
}

static int CeedVectorRestoreArray_Ref(CeedVector vec, CeedScalar **array) {
  *array = NULL;
  return 0;
}

static int CeedVectorRestoreArrayRead_Ref(CeedVector vec,
    const CeedScalar **array) {
  *array = NULL;
  return 0;
}

static int CeedVectorDestroy_Ref(CeedVector vec) {
  CeedVector_Ref *impl = vec->data;
  int ierr;

  ierr = CeedFree(&impl->array_allocated); CeedChk(ierr);
  ierr = CeedFree(&vec->data); CeedChk(ierr);
  return 0;
}

static int CeedVectorCreate_Ref(Ceed ceed, CeedInt n, CeedVector vec) {
  CeedVector_Ref *impl;
  int ierr;

  vec->SetArray = CeedVectorSetArray_Ref;
  vec->GetArray = CeedVectorGetArray_Ref;
  vec->GetArrayRead = CeedVectorGetArrayRead_Ref;
  vec->RestoreArray = CeedVectorRestoreArray_Ref;
  vec->RestoreArrayRead = CeedVectorRestoreArrayRead_Ref;
  vec->Destroy = CeedVectorDestroy_Ref;
  ierr = CeedCalloc(1,&impl); CeedChk(ierr);
  vec->data = impl;
  return 0;
}

static int CeedElemRestrictionApply_Ref(CeedElemRestriction r,
                                        CeedTransposeMode tmode, CeedInt ncomp,
                                        CeedTransposeMode lmode, CeedVector u,
                                        CeedVector v, CeedRequest *request) {
  CeedElemRestriction_Ref *impl = r->data;
  int ierr;
  const CeedScalar *uu;
  CeedScalar *vv;
  CeedInt esize = r->nelem*r->elemsize;

  ierr = CeedVectorGetArrayRead(u, CEED_MEM_HOST, &uu); CeedChk(ierr);
  ierr = CeedVectorGetArray(v, CEED_MEM_HOST, &vv); CeedChk(ierr);
  if (tmode == CEED_NOTRANSPOSE) {
    // Perform: v = r * u
    if (ncomp == 1) {
      for (CeedInt i=0; i<esize; i++) vv[i] = uu[impl->indices[i]];
    } else {
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
  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_NULL)
    *request = NULL;
  return 0;
}

static int CeedElemRestrictionDestroy_Ref(CeedElemRestriction r) {
  CeedElemRestriction_Ref *impl = r->data;
  int ierr;

  ierr = CeedFree(&impl->indices_allocated); CeedChk(ierr);
  ierr = CeedFree(&r->data); CeedChk(ierr);
  return 0;
}

static int CeedElemRestrictionCreate_Ref(CeedElemRestriction r,
    CeedMemType mtype,
    CeedCopyMode cmode, const CeedInt *indices) {
  int ierr;
  CeedElemRestriction_Ref *impl;

  if (mtype != CEED_MEM_HOST)
    return CeedError(r->ceed, 1, "Only MemType = HOST supported");
  ierr = CeedCalloc(1,&impl); CeedChk(ierr);
  switch (cmode) {
  case CEED_COPY_VALUES:
    ierr = CeedMalloc(r->nelem*r->elemsize, &impl->indices_allocated);
    CeedChk(ierr);
    memcpy(impl->indices_allocated, indices,
           r->nelem * r->elemsize * sizeof(indices[0]));
    impl->indices = impl->indices_allocated;
    break;
  case CEED_OWN_POINTER:
    impl->indices_allocated = (CeedInt *)indices;
    impl->indices = impl->indices_allocated;
    break;
  case CEED_USE_POINTER:
    impl->indices = indices;
  }
  r->data = impl;
  r->Apply = CeedElemRestrictionApply_Ref;
  r->Destroy = CeedElemRestrictionDestroy_Ref;
  return 0;
}

// Contracts on the middle index
// NOTRANSPOSE: V_ajc = T_jb U_abc
// TRANSPOSE:   V_ajc = T_bj U_abc
// If Add != 0, "=" is replaced by "+="
static int CeedTensorContract_Ref(Ceed ceed,
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

static int CeedBasisApply_Ref(CeedBasis basis, CeedTransposeMode tmode,
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
      ierr = CeedTensorContract_Ref(basis->ceed, pre, P, post, Q, basis->interp1d,
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
        ierr = CeedTensorContract_Ref(basis->ceed, pre, P, post, Q,
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

static int CeedBasisDestroy_Ref(CeedBasis basis) {
  return 0;
}

static int CeedBasisCreateTensorH1_Ref(Ceed ceed, CeedInt dim, CeedInt P1d,
                                       CeedInt Q1d, const CeedScalar *interp1d,
                                       const CeedScalar *grad1d,
                                       const CeedScalar *qref1d,
                                       const CeedScalar *qweight1d,
                                       CeedBasis basis) {
  basis->Apply = CeedBasisApply_Ref;
  basis->Destroy = CeedBasisDestroy_Ref;
  return 0;
}

static int CeedQFunctionApply_Ref(CeedQFunction qf, void *qdata, CeedInt Q,
                                  const CeedScalar *const *u,
                                  CeedScalar *const *v) {
  int ierr;
  ierr = qf->function(qf->ctx, qdata, Q, u, v); CeedChk(ierr);
  return 0;
}

static int CeedQFunctionDestroy_Ref(CeedQFunction qf) {
  return 0;
}

static int CeedQFunctionCreate_Ref(CeedQFunction qf) {
  qf->Apply = CeedQFunctionApply_Ref;
  qf->Destroy = CeedQFunctionDestroy_Ref;
  return 0;
}

static int CeedOperatorDestroy_Ref(CeedOperator op) {
  CeedOperator_Ref *impl = op->data;
  int ierr;

  ierr = CeedVectorDestroy(&impl->etmp); CeedChk(ierr);
  ierr = CeedVectorDestroy(&impl->qdata); CeedChk(ierr);
  ierr = CeedFree(&op->data); CeedChk(ierr);
  return 0;
}

static int CeedOperatorApply_Ref(CeedOperator op, CeedVector qdata,
                                 CeedVector ustate,
                                 CeedVector residual, CeedRequest *request) {
  CeedOperator_Ref *impl = op->data;
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
  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_NULL)
    *request = NULL;
  return 0;
}

static int CeedOperatorGetQData_Ref(CeedOperator op, CeedVector *qdata) {
  CeedOperator_Ref *impl = op->data;
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

static int CeedOperatorCreate_Ref(CeedOperator op) {
  CeedOperator_Ref *impl;
  int ierr;

  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  op->data = impl;
  op->Destroy = CeedOperatorDestroy_Ref;
  op->Apply = CeedOperatorApply_Ref;
  op->GetQData = CeedOperatorGetQData_Ref;
  return 0;
}

static int CeedInit_Ref(const char *resource, Ceed ceed) {
  if (strcmp(resource, "/cpu/self")
      && strcmp(resource, "/cpu/self/ref"))
    return CeedError(ceed, 1, "Ref backend cannot use resource: %s", resource);
  ceed->VecCreate = CeedVectorCreate_Ref;
  ceed->BasisCreateTensorH1 = CeedBasisCreateTensorH1_Ref;
  ceed->ElemRestrictionCreate = CeedElemRestrictionCreate_Ref;
  ceed->QFunctionCreate = CeedQFunctionCreate_Ref;
  ceed->OperatorCreate = CeedOperatorCreate_Ref;
  return 0;
}

__attribute__((constructor))
static void Register(void) {
  CeedRegister("/cpu/self/ref", CeedInit_Ref);
}
