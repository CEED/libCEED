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

#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include <ceed-dbg.h>
#include <ceed-impl.h>

// *****************************************************************************
// * OCCA stuff
// *****************************************************************************
#include "occa.h"
static occaDevice device;
const char *deviceInfo = "mode: 'Serial'";
//static occaMemory *array_allocated;

// *****************************************************************************
// * RESTRICTIONS: Create, Apply, Destroy
// *****************************************************************************
typedef struct {
  const CeedInt* indices;
  CeedInt* indices_allocated;
} CeedElemRestrictionOccaCPU;


// *****************************************************************************
static int CeedElemRestrictionApplyOccaCPU(CeedElemRestriction r,
                                           CeedTransposeMode tmode, CeedVector u,
                                           CeedVector v, CeedRequest* request) {
  CeedElemRestrictionOccaCPU* impl = r->data;
  int ierr;
  const CeedScalar* uu;
  CeedScalar* vv;

  dbg("[CeedElemRestriction][Apply][OccaCPU]");
  ierr = CeedVectorGetArrayRead(u, CEED_MEM_HOST, &uu); CeedChk(ierr);
  ierr = CeedVectorGetArray(v, CEED_MEM_HOST, &vv); CeedChk(ierr);
  if (tmode == CEED_NOTRANSPOSE) {
    for (CeedInt i=0; i<r->nelem*r->elemsize; i++) vv[i] = uu[impl->indices[i]];
  } else {
    for (CeedInt i=0; i<r->nelem*r->elemsize; i++) vv[impl->indices[i]] += uu[i];
  }
  ierr = CeedVectorRestoreArrayRead(u, &uu); CeedChk(ierr);
  ierr = CeedVectorRestoreArray(v, &vv); CeedChk(ierr);
  if (request != CEED_REQUEST_IMMEDIATE) *request = NULL;
  return 0;
}

// *****************************************************************************
static int CeedElemRestrictionDestroyOccaCPU(CeedElemRestriction r) {
  CeedElemRestrictionOccaCPU* impl = r->data;
  int ierr;

  dbg("[CeedElemRestriction][Destroy][OccaCPU]");
  ierr = CeedFree(&impl->indices_allocated); CeedChk(ierr);
  ierr = CeedFree(&r->data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
static int CeedElemRestrictionCreateOccaCPU(CeedElemRestriction r,
    CeedMemType mtype,
    CeedCopyMode cmode, const CeedInt* indices) {
  int ierr;
  CeedElemRestrictionOccaCPU* impl;

  dbg("[CeedElemRestriction][Create][OccaCPU]");
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
      impl->indices_allocated = (CeedInt*)indices;
      impl->indices = impl->indices_allocated;
      break;
    case CEED_USE_POINTER:
      impl->indices = indices;
  }
  r->data = impl;
  r->Apply = CeedElemRestrictionApplyOccaCPU;
  r->Destroy = CeedElemRestrictionDestroyOccaCPU;
  return 0;
}


// *****************************************************************************
// * TENSORS: Contracts on the middle index
// *          NOTRANSPOSE: V_ajc = T_jb U_abc
// *          TRANSPOSE:   V_ajc = T_bj U_abc
// *****************************************************************************
static int CeedTensorContractOccaCPU(Ceed ceed,
                                     CeedInt A, CeedInt B, CeedInt C, CeedInt J,
                                     const CeedScalar* t, CeedTransposeMode tmode,
                                     const CeedScalar* u, CeedScalar* v) {
  CeedInt tstride0 = B, tstride1 = 1;
  dbg("[CeedTensorContract][OccaCPU]");
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


// *****************************************************************************
// * Q-functions: Apply, Destroy & Create
// *****************************************************************************
static int CeedQFunctionApplyOccaCPU(CeedQFunction qf, void* qdata, CeedInt Q,
                                     const CeedScalar* const* u,
                                     CeedScalar* const* v) {
  int ierr;
  dbg("[CeedQFunction][Apply][OccaCPU]");
  ierr = qf->function(qf->ctx, qdata, Q, u, v); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
static int CeedQFunctionDestroyOccaCPU(CeedQFunction qf) {
  dbg("[CeedQFunction][Destroy][OccaCPU]");
  return 0;
}

// *****************************************************************************
static int CeedQFunctionCreateOccaCPU(CeedQFunction qf) {
  dbg("[CeedQFunction][Create][OccaCPU]");
  qf->Apply = CeedQFunctionApplyOccaCPU;
  qf->Destroy = CeedQFunctionDestroyOccaCPU;
  return 0;
}


// *****************************************************************************
// * OPERATORS: Create, Apply & Destroy
// *****************************************************************************
typedef struct {
  CeedVector etmp;
} CeedOperatorOccaCPU;

// *****************************************************************************
static int CeedOperatorApplyOccaCPU(CeedOperator op, CeedVector qdata,
                                    CeedVector ustate,
                                    CeedVector residual, CeedRequest* request) {
  dbg("[CeedOperator][Apply][OccaCPU]");
  CeedOperatorOccaCPU* impl = op->data;
  CeedVector etmp;
  CeedScalar* Eu;
  int ierr;

  if (!impl->etmp) {
    ierr = CeedVectorCreate(op->ceed,
                            op->Erestrict->nelem * op->Erestrict->elemsize,
                            &impl->etmp); CeedChk(ierr);
  }
  etmp = impl->etmp;
  if (op->qf->inmode != CEED_EVAL_NONE || op->qf->inmode != CEED_EVAL_WEIGHT) {
    ierr = CeedElemRestrictionApply(op->Erestrict, CEED_NOTRANSPOSE, ustate, etmp,
                                    CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
  }
  ierr = CeedVectorGetArray(etmp, CEED_MEM_HOST, &Eu); CeedChk(ierr);
  for (CeedInt e=0; e<op->Erestrict->nelem; e++) {
    CeedScalar BEu[CeedPowInt(op->basis->Q1d, op->basis->dim)];
    CeedScalar BEv[CeedPowInt(op->basis->Q1d, op->basis->dim)];
    ierr = CeedBasisApply(op->basis, CEED_NOTRANSPOSE, op->qf->inmode,
                          &Eu[e*op->Erestrict->elemsize], BEu); CeedChk(ierr);
    // qfunction
    ierr = CeedBasisApply(op->basis, CEED_TRANSPOSE, op->qf->outmode, BEv,
                          &Eu[e*op->Erestrict->elemsize]); CeedChk(ierr);
  }
  ierr = CeedVectorRestoreArray(etmp, &Eu); CeedChk(ierr);
  ierr = CeedElemRestrictionApply(op->Erestrict, CEED_TRANSPOSE, etmp, residual,
                                  CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
  if (request != CEED_REQUEST_IMMEDIATE) *request = NULL;
  return 0;
}

// *****************************************************************************
static int CeedOperatorDestroyOccaCPU(CeedOperator op) {
  dbg("[CeedOperator][Destroy][OccaCPU]");
  CeedOperatorOccaCPU* impl = op->data;
  int ierr;

  ierr = CeedVectorDestroy(&impl->etmp); CeedChk(ierr);
  ierr = CeedFree(&op->data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
static int CeedOperatorCreateOccaCPU(CeedOperator op) {
  CeedOperatorOccaCPU* impl;
  int ierr;

  dbg("[CeedOperator][Create][OccaCPU]");
  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  op->data = impl;
  op->Destroy = CeedOperatorDestroyOccaCPU;
  op->Apply = CeedOperatorApplyOccaCPU;
  return 0;
}


// *****************************************************************************
// * BASIS: Apply, Destroy, CreateTensorH1
// *****************************************************************************
static int CeedBasisApplyOccaCPU(CeedBasis basis, CeedTransposeMode tmode,
                                 CeedEvalMode emode,
                                 const CeedScalar* u, CeedScalar* v) {
  int ierr;
  const CeedInt dim = basis->dim;
  const CeedInt ndof = basis->ndof;

  dbg("[CeedBasis][Apply][OccaCPU]");
  switch (emode) {
    case CEED_EVAL_INTERP: {
      CeedInt P = basis->P1d, Q = basis->Q1d;
      if (tmode == CEED_TRANSPOSE) {
        P = basis->Q1d; Q = basis->P1d;
      }
      CeedInt pre = ndof*CeedPowInt(P, dim-1), post = 1;
      CeedScalar tmp[2][Q*CeedPowInt(P>Q?P:Q, dim-1)];
      for (CeedInt d=0; d<dim; d++) {
        ierr = CeedTensorContractOccaCPU(basis->ceed, pre, P, post, Q, basis->interp1d,
                                         tmode,
                                         d==0?u:tmp[d%2], d==dim-1?v:tmp[(d+1)%2]); CeedChk(ierr);
        pre /= P;
        post *= Q;
      }
    } break;
    case CEED_EVAL_WEIGHT: {
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
    } break;
    default:
      return CeedError(basis->ceed, 1, "EvalMode %d not supported", emode);
  }
  dbg("[CeedBasis][Apply][OccaCPU] done");
  return 0;
}

// *****************************************************************************
static int CeedBasisDestroyOccaCPU(CeedBasis basis) {
  dbg("[CeedBasis][Destroy][OccaCPU] done");
  return 0;
}

// *****************************************************************************
static int CeedBasisCreateTensorH1OccaCPU(Ceed ceed, CeedInt dim, CeedInt P1d,
                                          CeedInt Q1d, const CeedScalar* interp1d,
                                          const CeedScalar* grad1d,
                                          const CeedScalar* qref1d,
                                          const CeedScalar* qweight1d,
                                          CeedBasis basis) {
  basis->Apply = CeedBasisApplyOccaCPU;
  basis->Destroy = CeedBasisDestroyOccaCPU;
  dbg("[CeedBasis][Create][TensorH1][OccaCPU] done");
  return 0;
}


// *****************************************************************************
// * VECTORS: - Create, Destroy,
// *          - RestoreArrayRead, RestoreArray
// *          - GetArrayRead, GetArray, SetArray
// *****************************************************************************
typedef struct {
  int size;
  CeedScalar* array;
  occaMemory* array_allocated;
} CeedVectorOccaCPU;

// *****************************************************************************
static int CeedVectorDestroyOccaCPU(CeedVector vec) {
  CeedVectorOccaCPU* impl = vec->data;
  int ierr;

  dbg("\033[33m[CeedVector][Destroy][OccaCPU]");
  occaMemoryFree(*impl->array_allocated);
  ierr = CeedFree(&impl->array_allocated); CeedChk(ierr);
  ierr = CeedFree(&vec->data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
static int CeedVectorSetArrayOccaCPU(CeedVector vec, CeedMemType mtype,
                                     CeedCopyMode cmode, CeedScalar* array) {
  CeedVectorOccaCPU* impl = vec->data;
  int ierr;

  dbg("\033[33m[CeedVector][SetArray][OccaCPU]");
  if (mtype != CEED_MEM_HOST)
    return CeedError(vec->ceed, 1, "Only MemType = HOST supported");
  // Freeing previous allocated array
  occaMemoryFree(*impl->array_allocated);
  ierr = CeedFree(&impl->array_allocated); CeedChk(ierr);
  // and rallocating everything
  ierr = CeedCalloc(1,&impl->array_allocated); CeedChk(ierr);
  *impl->array_allocated = occaDeviceMalloc(device, impl->size*sizeof(CeedScalar), NULL, occaDefault);
  // ***************************************************************************
  switch (cmode) {
    case CEED_COPY_VALUES:
      dbg("\t\033[33m[CeedVector][SetArray][OccaCPU] CEED_COPY_VALUES");
      assert(false);
      //ierr = CeedMalloc(vec->length, &impl->array_allocated); CeedChk(ierr);
      //occaCopyPtrToMem(impl->array, impl->array_allocated, occaAllBytes, 0, occaDefault);
      //impl->array = impl->array_allocated;
      if (array) memcpy(impl->array, array, vec->length * sizeof(array[0]));
      break;
    case CEED_OWN_POINTER:
      dbg("\t\033[33m[CeedVector][SetArray][OccaCPU] CEED_OWN_POINTER");
      assert(false);
      //occaCopyPtrToMem(*impl->array_allocated, array, occaAllBytes, 0, occaDefault);
      //impl->array_allocated = array;
      impl->array = array;
      break;
    case CEED_USE_POINTER:
      dbg("\t\033[33m[CeedVector][SetArray][OccaCPU] CEED_USE_POINTER");
      impl->array = array;
      occaCopyPtrToMem(*impl->array_allocated, array, impl->size*sizeof(CeedScalar), 0, occaDefault);
  }
  return 0;
}

// *****************************************************************************
static int CeedVectorGetArrayOccaCPU(CeedVector vec, CeedMemType mtype,
                                     CeedScalar** array) {
  CeedVectorOccaCPU* impl = vec->data;

  dbg("\033[33m[CeedVector][GetArray][OccaCPU]");
  if (mtype != CEED_MEM_HOST)
    return CeedError(vec->ceed, 1, "Can only provide to HOST memory");
  *array = impl->array;
  return 0;
}

// *****************************************************************************
static int CeedVectorGetArrayReadOccaCPU(CeedVector vec, CeedMemType mtype,
                                         const CeedScalar** array) {
  CeedVectorOccaCPU* impl = vec->data;

  dbg("\033[33m[CeedVector][GetArray][Const][OccaCPU]");
  if (mtype != CEED_MEM_HOST)
    return CeedError(vec->ceed, 1, "Can only provide to HOST memory");
  //occaCopyPtrToMem(impl->array, impl->array_allocated, occaAllBytes, 0, occaDefault);
  //occaCopyMemToPtr(impl->array, *impl->array_allocated, occaAllBytes, 0, occaDefault);
  *array = impl->array;
  return 0;
}

// *****************************************************************************
static int CeedVectorRestoreArrayOccaCPU(CeedVector vec, CeedScalar** array) {
  dbg("\033[33m[CeedVector][RestoreArray][OccaCPU]");
  *array = NULL;
  return 0;
}

// *****************************************************************************
static int CeedVectorRestoreArrayReadOccaCPU(CeedVector vec,
                                             const CeedScalar** array) {
  dbg("\033[33m[CeedVector][RestoreArray][Const][OccaCPU]");
  *array = NULL;
  return 0;
}

// *****************************************************************************
static int CeedVectorCreateOccaCPU(Ceed ceed, CeedInt n, CeedVector vec) {
  CeedVectorOccaCPU* impl;
  int ierr;

  dbg("\033[33m[CeedVector][Create][OccaCPU] n=%d", n);
  vec->SetArray = CeedVectorSetArrayOccaCPU;
  vec->GetArray = CeedVectorGetArrayOccaCPU;
  vec->GetArrayRead = CeedVectorGetArrayReadOccaCPU;
  vec->RestoreArray = CeedVectorRestoreArrayOccaCPU;
  vec->RestoreArrayRead = CeedVectorRestoreArrayReadOccaCPU;
  vec->Destroy = CeedVectorDestroyOccaCPU;
  ierr = CeedCalloc(1,&impl); CeedChk(ierr);
  ierr = CeedCalloc(1,&impl->array_allocated); CeedChk(ierr);
  // Allocating on device
  impl->size = n;
  *impl->array_allocated = occaDeviceMalloc(device, n*sizeof(CeedScalar), NULL, occaDefault);
  vec->data = impl;
  dbg("\033[33m[CeedVector][Create][OccaCPU] done");
  return 0;
}

// *****************************************************************************
int CeedErrorOccaCPU(Ceed ceed,
                     const char *file, int line,
                     const char *func, int code,
                     const char* format, va_list args){
  fprintf(stderr,"\033[31;1m");
  vfprintf(stderr, format, args);
  fprintf(stderr,"\033[m\n");
  fflush(stderr);
  return 0;
}

// *****************************************************************************
int CeedDestroyOccaCPU(Ceed ceed){
  dbg("\033[1m[CeedDestroy][OccaCPU]");
  occaDeviceFree(device);
  return 0;
}

// *****************************************************************************
// * INIT
// *****************************************************************************
static int CeedInitOccaCPU(const char* resource, Ceed ceed) {
  dbg("\033[1m[CeedInit][OccaCPU] resource='%s'", resource);
  if (strcmp(resource, "/cpu/occa"))
    return CeedError(ceed, 1, "Ref backend cannot use resource: %s", resource);
  ceed->Error = CeedErrorOccaCPU;
  ceed->Destroy = CeedDestroyOccaCPU;
  ceed->VecCreate = CeedVectorCreateOccaCPU;
  ceed->ElemRestrictionCreate = CeedElemRestrictionCreateOccaCPU;
  ceed->BasisCreateTensorH1 = CeedBasisCreateTensorH1OccaCPU;
  ceed->QFunctionCreate = CeedQFunctionCreateOccaCPU;
  ceed->OperatorCreate = CeedOperatorCreateOccaCPU;
  // Now creating OCCA device
  device = occaCreateDevice(occaString(deviceInfo));
  return 0;
} 


// *****************************************************************************
// * REGISTER
// *****************************************************************************
__attribute__((constructor))
static void Register(void) {
  dbg("\033[1m[Register] /cpu/occa");
  CeedRegister("/cpu/occa", CeedInitOccaCPU);
}
