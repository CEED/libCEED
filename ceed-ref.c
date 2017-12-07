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
} CeedOperator_Ref;

static int CeedVectorSetArray_Ref(CeedVector vec, CeedMemType mtype, CeedCopyMode cmode, CeedScalar *array) {
  CeedVector_Ref *impl = vec->data;
  int ierr;

  if (mtype != CEED_MEM_HOST) CeedError(vec->ceed, 1, "Only MemType = HOST supported");
  ierr = CeedFree(&impl->array_allocated);CeedChk(ierr);
  switch (cmode) {
  case CEED_COPY_VALUES:
    ierr = CeedMalloc(vec->length, &impl->array_allocated);CeedChk(ierr);
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

static int CeedVectorGetArray_Ref(CeedVector vec, CeedMemType mtype, CeedScalar **array) {
  CeedVector_Ref *impl = vec->data;

  if (mtype != CEED_MEM_HOST) CeedError(vec->ceed, 1, "Can only provide to HOST memory");
  *array = impl->array;
  return 0;
}

static int CeedVectorGetArrayRead_Ref(CeedVector vec, CeedMemType mtype, const CeedScalar **array) {
  CeedVector_Ref *impl = vec->data;

  if (mtype != CEED_MEM_HOST) CeedError(vec->ceed, 1, "Can only provide to HOST memory");
  *array = impl->array;
  return 0;
}

static int CeedVectorRestoreArray_Ref(CeedVector vec, CeedScalar **array) {
  *array = NULL;
  return 0;
}

static int CeedVectorRestoreArrayRead_Ref(CeedVector vec, const CeedScalar **array) {
  *array = NULL;
  return 0;
}

static int CeedVectorDestroy_Ref(CeedVector vec) {
  CeedVector_Ref *impl = vec->data;
  int ierr;

  ierr = CeedFree(&impl->array_allocated);CeedChk(ierr);
  ierr = CeedFree(&vec->data);CeedChk(ierr);
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
  ierr = CeedCalloc(1,&impl);CeedChk(ierr);
  vec->data = impl;
  return 0;
}

static int CeedElemRestrictionApply_Ref(CeedElemRestriction r, CeedTransposeMode tmode, CeedVector u, CeedVector v, CeedRequest *request) {
  CeedElemRestriction_Ref *impl = r->data;
  int ierr;
  const CeedScalar *uu;
  CeedScalar *vv;

  ierr = CeedVectorGetArrayRead(u, CEED_MEM_HOST, &uu);CeedChk(ierr);
  ierr = CeedVectorGetArray(v, CEED_MEM_HOST, &vv);CeedChk(ierr);
  if (tmode == CEED_NOTRANSPOSE) {
    for (CeedInt i=0; i<r->nelem*r->elemsize; i++) vv[i] = uu[impl->indices[i]];
  } else {
    for (CeedInt i=0; i<r->nelem*r->elemsize; i++) vv[impl->indices[i]] += uu[i];
  }
  ierr = CeedVectorRestoreArrayRead(u, &uu);CeedChk(ierr);
  ierr = CeedVectorRestoreArray(v, &vv);CeedChk(ierr);
  if (request != CEED_REQUEST_IMMEDIATE) *request = NULL;
  return 0;
}

static int CeedElemRestrictionDestroy_Ref(CeedElemRestriction r) {
  CeedElemRestriction_Ref *impl = r->data;
  int ierr;

  ierr = CeedFree(&impl->indices_allocated);CeedChk(ierr);
  ierr = CeedFree(&r->data);CeedChk(ierr);
  return 0;
}

static int CeedElemRestrictionCreate_Ref(CeedElemRestriction r, CeedMemType mtype, CeedCopyMode cmode, const CeedInt *indices) {
  int ierr;
  CeedElemRestriction_Ref *impl;

  if (mtype != CEED_MEM_HOST) CeedError(r->ceed, 1, "Only MemType = HOST supported");
  ierr = CeedCalloc(1,&impl);CeedChk(ierr);
  switch (cmode) {
  case CEED_COPY_VALUES:
    ierr = CeedMalloc(r->nelem*r->elemsize, &impl->indices_allocated);CeedChk(ierr);
    memcpy(impl->indices_allocated, indices, r->nelem * r->elemsize * sizeof(indices[0]));
    impl->indices = impl->indices_allocated;
    break;
  case CEED_OWN_POINTER:
    impl->indices_allocated = (CeedInt*)indices;
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
static int CeedTensorContract_Ref(Ceed ceed, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const CeedScalar *t, CeedTransposeMode tmode, const CeedScalar *u, CeedScalar *v) {
  CeedInt tstride0 = B, tstride1 = 1;
  if (tmode == CEED_TRANSPOSE) {
    tstride0 = 1; tstride1 = B;
  }

  for (CeedInt a=0; a<A; a++) {
    for (CeedInt j=0; j<J; j++) {
      for (CeedInt c=0; c<C; c++)
        v[(a*J+j)*C+c] = 0;
      for (CeedInt b=0; b<B; b++) {
        for (CeedInt c=0; c<C; c++) {
          v[(a*J+j)*C+c] += t[j*tstride0 + b*tstride1] * u[(a*B+b)*C+c];
        }
      }
    }
  }
  return 0;
}

static int CeedBasisApply_Ref(CeedBasis basis, CeedTransposeMode tmode, CeedEvalMode emode, const CeedScalar *u, CeedScalar *v) {
  int ierr;
  const CeedInt dim = basis->dim;
  const CeedInt ndof = basis->ndof;

  switch (emode) {
  case CEED_EVAL_INTERP: {
    CeedInt P = basis->P1d, Q = basis->Q1d;
    if (tmode == CEED_TRANSPOSE) {
      P = basis->Q1d; Q = basis->P1d;
    }
    CeedInt pre = ndof*CeedPowInt(P, dim-1), post = 1;
    CeedScalar tmp[2][Q*CeedPowInt(P>Q?P:Q, dim-1)];
    for (CeedInt d=0; d<dim; d++) {
      ierr = CeedTensorContract_Ref(basis->ceed, pre, P, post, Q, basis->interp1d, tmode, d==0?u:tmp[d%2], d==dim-1?v:tmp[(d+1)%2]);CeedChk(ierr);
      pre /= P;
      post *= Q;
    }
  } break;
  default:
    return CeedError(basis->ceed, 1, "EvalMode %d not supported", emode);
  }
  return 0;
}

static int CeedBasisDestroy_Ref(CeedBasis basis) {
  return 0;
}

static int CeedBasisCreateTensorH1_Ref(Ceed ceed, CeedInt dim, CeedInt P1d, CeedInt Q1d, const CeedScalar *interp1d, const CeedScalar *grad1d, const CeedScalar *qref1d, const CeedScalar *qweight1d, CeedBasis basis) {
  basis->Apply = CeedBasisApply_Ref;
  basis->Destroy = CeedBasisDestroy_Ref;
  return 0;
}

static int CeedQFunctionDestroy_Ref(CeedQFunction qf) {
  return 0;
}

static int CeedQFunctionCreate_Ref(CeedQFunction qf) {
  qf->Destroy = CeedQFunctionDestroy_Ref;
  return 0;
}

static int CeedOperatorDestroy_Ref(CeedOperator op) {
  CeedOperator_Ref *impl = op->data;
  int ierr;

  ierr = CeedVectorDestroy(&impl->etmp);CeedChk(ierr);
  ierr = CeedFree(&op->data);CeedChk(ierr);
  return 0;
}

static int CeedOperatorApply_Ref(CeedOperator op, CeedVector qdata, CeedVector ustate, CeedVector residual, CeedRequest *request) {
  CeedOperator_Ref *impl = op->data;
  CeedVector etmp;
  CeedScalar *Eu;
  int ierr;

  if (!impl->etmp) {
    ierr = CeedVectorCreate(op->ceed, op->Erestrict->nelem * op->Erestrict->elemsize, &impl->etmp);CeedChk(ierr);
  }
  etmp = impl->etmp;
  if (op->qf->inmode != CEED_EVAL_NONE) {
    ierr = CeedElemRestrictionApply(op->Erestrict, CEED_NOTRANSPOSE, ustate, etmp, CEED_REQUEST_IMMEDIATE);CeedChk(ierr);
  }
  ierr = CeedVectorGetArray(etmp, CEED_MEM_HOST, &Eu);CeedChk(ierr);
  for (CeedInt e=0; e<op->Erestrict->nelem; e++) {
    CeedScalar BEu[CeedPowInt(op->basis->Q1d, op->basis->dim)];
    CeedScalar BEv[CeedPowInt(op->basis->Q1d, op->basis->dim)];
    ierr = CeedBasisApply(op->basis, CEED_NOTRANSPOSE, op->qf->inmode, &Eu[e*op->Erestrict->elemsize], BEu);CeedChk(ierr);
    // qfunction
    ierr = CeedBasisApply(op->basis, CEED_TRANSPOSE, op->qf->outmode, BEv, &Eu[e*op->Erestrict->elemsize]);CeedChk(ierr);
  }
  ierr = CeedVectorRestoreArray(etmp, &Eu);CeedChk(ierr);
  ierr = CeedElemRestrictionApply(op->Erestrict, CEED_TRANSPOSE, etmp, residual, CEED_REQUEST_IMMEDIATE);CeedChk(ierr);
  if (request != CEED_REQUEST_IMMEDIATE) *request = NULL;
  return 0;
}

static int CeedOperatorCreate_Ref(CeedOperator op) {
  CeedOperator_Ref *impl;
  int ierr;

  ierr = CeedCalloc(1, &impl);CeedChk(ierr);
  op->data = impl;
  op->Destroy = CeedOperatorDestroy_Ref;
  op->Apply = CeedOperatorApply_Ref;
  return 0;
}

static int CeedInit_Ref(const char *resource, Ceed ceed) {
  if (strcmp(resource, "/cpu/self") && strcmp(resource, "/cpu/self/ref")) return CeedError(ceed, 1, "Ref backend cannot use resource: %s", resource);
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
