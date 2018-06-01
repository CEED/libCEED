// Fortran interface
#include <ceed.h>
#include <ceed-impl.h>
#include <ceed-fortran-name.h>

#include <stdlib.h>
#include <string.h>

#define FORTRAN_REQUEST_IMMEDIATE -1
#define FORTRAN_REQUEST_ORDERED -2
#define FORTRAN_NULL -3

static Ceed *Ceed_dict = NULL;
static int Ceed_count = 0;
static int Ceed_n = 0;
static int Ceed_count_max = 0;

#define fCeedInit FORTRAN_NAME(ceedinit,CEEDINIT)
void fCeedInit(const char* resource, int *ceed, int *err) {
  if (Ceed_count == Ceed_count_max) {
    Ceed_count_max += Ceed_count_max/2 + 1;
    CeedRealloc(Ceed_count_max, &Ceed_dict);
  }

  Ceed *ceed_ = &Ceed_dict[Ceed_count];
  *err = CeedInit(resource, ceed_);

  if (*err == 0) {
    *ceed = Ceed_count++;
    Ceed_n++;
  }
}

#define fCeedDestroy FORTRAN_NAME(ceeddestroy,CEEDDESTROY)
void fCeedDestroy(int *ceed, int *err) {
  *err = CeedDestroy(&Ceed_dict[*ceed]);

  if (*err == 0) {
    Ceed_n--;
    if (Ceed_n == 0) {
      CeedFree(&Ceed_dict);
      Ceed_count = 0;
      Ceed_count_max = 0;
    }
  }
}

static CeedVector *CeedVector_dict = NULL;
static int CeedVector_count = 0;
static int CeedVector_n = 0;
static int CeedVector_count_max = 0;

#define fCeedVectorCreate FORTRAN_NAME(ceedvectorcreate,CEEDVECTORCREATE)
void fCeedVectorCreate(int *ceed, int *length, int *vec, int *err) {
  if (CeedVector_count == CeedVector_count_max) {
    CeedVector_count_max += CeedVector_count_max/2 + 1;
    CeedRealloc(CeedVector_count_max, &CeedVector_dict);
  }

  CeedVector* vec_ = &CeedVector_dict[CeedVector_count];
  *err = CeedVectorCreate(Ceed_dict[*ceed], *length, vec_);

  if (*err == 0) {
    *vec = CeedVector_count++;
    CeedVector_n++;
  }
}

#define fCeedVectorSetArray FORTRAN_NAME(ceedvectorsetarray,CEEDVECTORSETARRAY)
void fCeedVectorSetArray(int *vec, int *memtype, int *copymode,
                         CeedScalar *array, int *err) {
  *err = CeedVectorSetArray(CeedVector_dict[*vec], *memtype, *copymode, array);
}

#define fCeedVectorGetArray FORTRAN_NAME(ceedvectorgetarray,CEEDVECTORGETARRAY)
//TODO Need Fixing, double pointer
void fCeedVectorGetArray(int *vec, int *memtype, CeedScalar *array, int *err) {
  CeedScalar *b;
  CeedVector vec_ = CeedVector_dict[*vec];
  *err = CeedVectorGetArray(vec_, *memtype, &b);
  if (*err == 0)
    memcpy(array, b, sizeof(CeedScalar)*vec_->length);
}

#define fCeedVectorGetArrayRead \
    FORTRAN_NAME(ceedvectorgetarrayread,CEEDVECTORGETARRAYREAD)
//TODO Need Fixing, double pointer
void fCeedVectorGetArrayRead(int *vec, int *memtype, CeedScalar *array,
                             int *err) {
  const CeedScalar *b;
  *err = CeedVectorGetArrayRead(CeedVector_dict[*vec], *memtype, &b);
  CeedVector vec_ = CeedVector_dict[*vec];
  if (*err == 0)
    memcpy(array, b, sizeof(CeedScalar)*vec_->length);
}

#define fCeedVectorRestoreArray \
    FORTRAN_NAME(ceedvectorrestorearray,CEEDVECTORRESTOREARRAY)
void fCeedVectorRestoreArray(int *vec, CeedScalar *array, int *err) {
  *err = CeedVectorRestoreArray(CeedVector_dict[*vec], &array);
}

#define fCeedVectorRestoreArrayRead \
    FORTRAN_NAME(ceedvectorrestorearrayread,CEEDVECTORRESTOREARRAYREAD)
void fCeedVectorRestoreArrayRead(int *vec, const CeedScalar *array, int *err) {
  *err = CeedVectorRestoreArrayRead(CeedVector_dict[*vec], &array);
}

#define fCeedVectorDestroy FORTRAN_NAME(ceedvectordestroy,CEEDVECTORDESTROY)
void fCeedVectorDestroy(int *vec, int *err) {
  *err = CeedVectorDestroy(&CeedVector_dict[*vec]);

  if (*err == 0) {
    CeedVector_n--;
    if (CeedVector_n == 0) {
      CeedFree(&CeedVector_dict);
      CeedVector_count = 0;
      CeedVector_count_max = 0;
    }
  }
}

static CeedElemRestriction *CeedElemRestriction_dict = NULL;
static int CeedElemRestriction_count = 0;
static int CeedElemRestriction_n = 0;
static int CeedElemRestriction_count_max = 0;

#define fCeedElemRestrictionCreate \
    FORTRAN_NAME(ceedelemrestrictioncreate, CEEDELEMRESTRICTIONCREATE)
void fCeedElemRestrictionCreate(int *ceed, int *nelements,
                                int *esize, int *ndof, int *memtype, int *copymode,
                                const int *indices, int *elemrestriction, int *err) {
  if (CeedElemRestriction_count == CeedElemRestriction_count_max) {
    CeedElemRestriction_count_max += CeedElemRestriction_count_max/2 + 1;
    CeedRealloc(CeedElemRestriction_count_max, &CeedElemRestriction_dict);
  }

  CeedElemRestriction *elemrestriction_ =
    &CeedElemRestriction_dict[CeedElemRestriction_count];
  *err = CeedElemRestrictionCreate(Ceed_dict[*ceed], *nelements, *esize, *ndof,
                                   *memtype, *copymode, indices, elemrestriction_);

  if (*err == 0) {
    *elemrestriction = CeedElemRestriction_count++;
    CeedElemRestriction_n++;
  }
}

#define fCeedElemRestrictionCreateBlocked \
    FORTRAN_NAME(ceedelemrestrictioncreateblocked,CEEDELEMRESTRICTIONCREATEBLOCKED)
void fCeedElemRestrictionCreateBlocked(int *ceed, int *nelements,
                                       int *esize, int *blocksize, int *mtype, int *cmode,
                                       int *blkindices, int *elemr, int *err) {
  *err = CeedElemRestrictionCreateBlocked(Ceed_dict[*ceed], *nelements, *esize,
                                          *blocksize, *mtype, *cmode, blkindices, &CeedElemRestriction_dict[*elemr]);
}

static CeedRequest *CeedRequest_dict = NULL;
static int CeedRequest_count = 0;
static int CeedRequest_n = 0;
static int CeedRequest_count_max = 0;

#define fCeedElemRestrictionApply \
    FORTRAN_NAME(ceedelemrestrictionapply,CEEDELEMRESTRICTIONAPPLY)
void fCeedElemRestrictionApply(int *elemr, int *tmode, int *ncomp, int *lmode,
                               int *uvec, int *ruvec, int *rqst, int *err) {
  int createRequest = 1;
  // Check if input is CEED_REQUEST_ORDERED(-2) or CEED_REQUEST_IMMEDIATE(-1)
  if (*rqst == FORTRAN_REQUEST_IMMEDIATE || *rqst == FORTRAN_REQUEST_ORDERED)
    createRequest = 0;

  if (createRequest && CeedRequest_count == CeedRequest_count_max) {
    CeedRequest_count_max += CeedRequest_count_max/2 + 1;
    CeedRealloc(CeedRequest_count_max, &CeedRequest_dict);
  }

  CeedRequest *rqst_;
  if      (*rqst == FORTRAN_REQUEST_IMMEDIATE) rqst_ = CEED_REQUEST_IMMEDIATE;
  else if (*rqst == FORTRAN_REQUEST_ORDERED  ) rqst_ = CEED_REQUEST_ORDERED;
  else rqst_ = &CeedRequest_dict[CeedRequest_count];

  *err = CeedElemRestrictionApply(CeedElemRestriction_dict[*elemr], *tmode,
                                  *ncomp,
                                  *lmode, CeedVector_dict[*uvec], CeedVector_dict[*ruvec], rqst_);

  if (*err == 0 && createRequest) {
    *rqst = CeedRequest_count++;
    CeedRequest_n++;
  }
}

#define fCeedRequestWait FORTRAN_NAME(ceedrequestwait, CEEDREQUESTWAIT)
void fCeedRequestWait(int *rqst, int *err) {
  // TODO Uncomment this once CeedRequestWait is implemented
  //*err = CeedRequestWait(&CeedRequest_dict[*rqst]);

  if (*err == 0) {
    CeedRequest_n--;
    if (CeedRequest_n == 0) {
      CeedFree(&CeedRequest_dict);
      CeedRequest_count = 0;
      CeedRequest_count_max = 0;
    }
  }
}

#define fCeedElemRestrictionDestroy \
    FORTRAN_NAME(ceedelemrestrictiondestroy,CEEDELEMRESTRICTIONDESTROY)
void fCeedElemRestrictionDestroy(int *elem, int *err) {
  *err = CeedElemRestrictionDestroy(&CeedElemRestriction_dict[*elem]);

  if (*err == 0) {
    CeedElemRestriction_n--;
    if (CeedElemRestriction_n == 0) {
      CeedFree(&CeedElemRestriction_dict);
      CeedElemRestriction_count = 0;
      CeedElemRestriction_count_max = 0;
    }
  }
}

static CeedBasis *CeedBasis_dict = NULL;
static int CeedBasis_count = 0;
static int CeedBasis_n = 0;
static int CeedBasis_count_max = 0;

#define fCeedBasisCreateTensorH1Lagrange \
    FORTRAN_NAME(ceedbasiscreatetensorh1lagrange, CEEDBASISCREATETENSORH1LAGRANGE)
void fCeedBasisCreateTensorH1Lagrange(int *ceed, int *dim,
                                      int *ndof, int *P, int *Q, int *quadmode, int *basis,
                                      int *err) {
  if (CeedBasis_count == CeedBasis_count_max) {
    CeedBasis_count_max += CeedBasis_count_max/2 + 1;
    CeedRealloc(CeedBasis_count_max, &CeedBasis_dict);
  }

  *err = CeedBasisCreateTensorH1Lagrange(Ceed_dict[*ceed], *dim, *ndof, *P, *Q,
                                         *quadmode, &CeedBasis_dict[CeedBasis_count]);

  if (*err == 0) {
    *basis = CeedBasis_count++;
    CeedBasis_n++;
  }
}

#define fCeedBasisCreateTensorH1 \
    FORTRAN_NAME(ceedbasiscreatetensorh1, CEEDBASISCREATETENSORH1)
void fCeedBasisCreateTensorH1(int *ceed, int *dim, int *ndof, int *P1d,
                              int *Q1d, const CeedScalar *interp1d, const CeedScalar *grad1d,
                              const CeedScalar *qref1d, const CeedScalar *qweight1d, int *basis, int *err) {
  if (CeedBasis_count == CeedBasis_count_max) {
    CeedBasis_count_max += CeedBasis_count_max/2 + 1;
    CeedRealloc(CeedBasis_count_max, &CeedBasis_dict);
  }

  *err = CeedBasisCreateTensorH1(Ceed_dict[*ceed], *dim, *ndof, *P1d, *Q1d,
                                 interp1d, grad1d,
                                 qref1d, qweight1d, &CeedBasis_dict[CeedBasis_count]);

  if (*err == 0) {
    *basis = CeedBasis_count++;
    CeedBasis_n++;
  }
}

#define fCeedBasisView FORTRAN_NAME(ceedbasisview, CEEDBASISVIEW)
void fCeedBasisView(int *basis, int *err) {
  *err = CeedBasisView(CeedBasis_dict[*basis], stdout);
}

#define fCeedBasisApply FORTRAN_NAME(ceedbasisapply, CEEDBASISAPPLY)
void fCeedBasisApply(int *basis, int *tmode, int *emode, const CeedScalar *u,
                     CeedScalar *v, int *err) {
  *err = CeedBasisApply(CeedBasis_dict[*basis], *tmode, *emode, u, v);
}

#define fCeedBasisGetNumNodes \
    FORTRAN_NAME(ceedbasisgetnumnodes, CEEDBASISGETNUMNODES)
void fCeedBasisGetNumNodes(int *basis, int *P, int *err) {
  *err = CeedBasisGetNumNodes(CeedBasis_dict[*basis], P);
}

#define fCeedBasisGetNumQuadraturePoints \
    FORTRAN_NAME(ceedbasisgetnumquadraturepoints, CEEDBASISGETNUMQUADRATUREPOINTS)
void fCeedBasisGetNumQuadraturePoints(int *basis, int *Q, int *err) {
  *err = CeedBasisGetNumQuadraturePoints(CeedBasis_dict[*basis], Q);
}

#define fCeedBasisDestroy FORTRAN_NAME(ceedbasisdestroy,CEEDBASISDESTROY)
void fCeedBasisDestroy(int *basis, int *err) {
  *err = CeedBasisDestroy(&CeedBasis_dict[*basis]);

  if (*err == 0) {
    CeedBasis_n--;
    if (CeedBasis_n == 0) {
      CeedFree(&CeedBasis_dict);
      CeedBasis_count = 0;
      CeedBasis_count_max = 0;
    }
  }
}

#define fCeedGaussQuadrature FORTRAN_NAME(ceedgaussquadrature, CEEDGAUSSQUADRATURE)
void fCeedGaussQuadrature(int *Q, CeedScalar *qref1d, CeedScalar *qweight1d,
                          int *err) {
  *err = CeedGaussQuadrature(*Q, qref1d, qweight1d);
}

#define fCeedLobattoQuadrature \
    FORTRAN_NAME(ceedlobattoquadrature, CEEDLOBATTOQUADRATURE)
void fCeedLobattoQuadrature(int *Q, CeedScalar *qref1d, CeedScalar *qweight1d,
                            int *err) {
  *err = CeedLobattoQuadrature(*Q, qref1d, qweight1d);
}

static CeedQFunction *CeedQFunction_dict = NULL;
static int CeedQFunction_count = 0;
static int CeedQFunction_n = 0;
static int CeedQFunction_count_max = 0;

struct fContext {
  void (*f)(void *ctx, void *qdata, int *nq,
            const CeedScalar *const u,const CeedScalar *const u1,const CeedScalar *const u2,
            CeedScalar *const v1,CeedScalar *const v2, int *err);
  void *innerctx;
};

static int CeedQFunctionFortranStub(void *ctx, void *qdata, int nq,
                                    const CeedScalar *const *u, CeedScalar *const *v) {
  struct fContext *fctx = ctx;
  int ierr;

  CeedScalar ctx_=1.0;
  fctx->f((void*)&ctx_, qdata, &nq, u[0], u[1], u[4], v[0], v[1], &ierr);
  return ierr;
}

#define fCeedQFunctionCreateInterior \
    FORTRAN_NAME(ceedqfunctioncreateinterior, CEEDQFUNCTIONCREATEINTERIOR)
void fCeedQFunctionCreateInterior(int* ceed, int* vlength,
                                  int* nfields, int* qdatasize, int* inmode, int* outmode,
                                  void (*f)(void *ctx, void *qdata, int *nq,
                                      const CeedScalar *u,const CeedScalar *u1,const CeedScalar *u2,
                                      CeedScalar *v1,CeedScalar *v2, int *err),
                                  const char *focca, int *qf, int *err) {
  if (CeedQFunction_count == CeedQFunction_count_max) {
    CeedQFunction_count_max += CeedQFunction_count_max/2 + 1;
    CeedRealloc(CeedQFunction_count_max, &CeedQFunction_dict);
  }

  CeedQFunction *qf_ = &CeedQFunction_dict[CeedQFunction_count];
  *err = CeedQFunctionCreateInterior(Ceed_dict[*ceed], *vlength, *nfields,
                                     *qdatasize, (CeedEvalMode)(*inmode), (CeedEvalMode)(*outmode),
                                     CeedQFunctionFortranStub,focca, qf_);

  if (*err == 0) {
    *qf = CeedQFunction_count++;
    CeedQFunction_n++;
  }

  struct fContext *fctx; CeedMalloc(1, &fctx);
  fctx->f = f; fctx->innerctx = NULL;

  CeedQFunctionSetContext(*qf_, fctx, sizeof(struct fContext));

}

#define fCeedQFunctionSetContext \
    FORTRAN_NAME(ceedqfunctionsetcontext, CEEDQFUNCTIONSETCONTEXT)
void fCeedQFunctionSetContext(int *qf, void *ctx, size_t* ctxsize,
                              int *err) {
  CeedQFunction qf_ = CeedQFunction_dict[*qf];

  struct fContext *newFContext; CeedMalloc(1, &newFContext);
  newFContext->f = ((struct fContext *)(qf_->ctx))->f;
  newFContext->innerctx = ctx;

  *err = CeedQFunctionSetContext(qf_, newFContext, sizeof(struct fContext));
}

#define fCeedQFunctionApply \
    FORTRAN_NAME(ceedqfunctionapply,CEEDQFUNCTIONAPPLY)
//TODO Need Fixing, double pointer
void fCeedQFunctionApply(int *qf, void *qdata, int *Q,
                         const CeedScalar *u, CeedScalar *v, int *err) {
  CeedQFunction qf_ = CeedQFunction_dict[*qf];

  *err = CeedQFunctionApply(qf_, qdata, *Q, &u, &v);
}

#define fCeedQFunctionDestroy \
    FORTRAN_NAME(ceedqfunctiondestroy,CEEDQFUNCTIONDESTROY)
void fCeedQFunctionDestroy(int *qf, int *err) {
  CeedFree(&CeedQFunction_dict[*qf]->ctx);
  *err = CeedQFunctionDestroy(&CeedQFunction_dict[*qf]);

  if (*err == 0) {
    CeedQFunction_n--;
    if (CeedQFunction_n == 0) {
      CeedFree(&CeedQFunction_dict);
      CeedQFunction_count = 0;
      CeedQFunction_count_max = 0;
    }
  }
}

static CeedOperator *CeedOperator_dict = NULL;
static int CeedOperator_count = 0;
static int CeedOperator_n = 0;
static int CeedOperator_count_max = 0;

#define fCeedOperatorCreate \
    FORTRAN_NAME(ceedoperatorcreate, CEEDOPERATORCREATE)
void fCeedOperatorCreate(int* ceed, int* erstrn, int* basis,
                         int* qf, int* dqf, int* dqfT, int *op, int *err) {
  if (CeedOperator_count == CeedOperator_count_max)
    CeedOperator_count_max += CeedOperator_count_max/2 + 1,
                              CeedOperator_dict =
                                realloc(CeedOperator_dict, sizeof(CeedOperator)*CeedOperator_count_max);

  CeedOperator *op_ = &CeedOperator_dict[CeedOperator_count];

  CeedQFunction dqf_  = NULL, dqfT_ = NULL;
  if (*dqf  != FORTRAN_NULL) dqf_  = CeedQFunction_dict[*dqf ];
  if (*dqfT != FORTRAN_NULL) dqfT_ = CeedQFunction_dict[*dqfT];

  *err = CeedOperatorCreate(Ceed_dict[*ceed], CeedElemRestriction_dict[*erstrn],
                            CeedBasis_dict[*basis], CeedQFunction_dict[*qf], dqf_, dqfT_, op_);

  if (*err == 0) {
    *op = CeedOperator_count++;
    CeedOperator_n++;
  }
}

#define fCeedOperatorGetQData \
    FORTRAN_NAME(ceedoperatorgetqdata, CEEDOPERATORGETQDATA)
void fCeedOperatorGetQData(int *op, int *vec, int *err) {
  if (CeedVector_count == CeedVector_count_max) {
    CeedVector_count_max += CeedVector_count_max/2 + 1;
    CeedRealloc(CeedVector_count_max, &CeedVector_dict);
  }

  *err = CeedOperatorGetQData(CeedOperator_dict[*op],
                              &CeedVector_dict[CeedVector_count]);

  if (*err == 0) {
    *vec = CeedVector_count++;
    CeedVector_n++;
  }
}

#define fCeedOperatorApply FORTRAN_NAME(ceedoperatorapply, CEEDOPERATORAPPLY)
void fCeedOperatorApply(int *op, int *qdatavec, int *ustatevec,
                        int *resvec, int *rqst, int *err) {
  // TODO What vector arguments can be NULL?
  CeedVector resvec_;
  if (*resvec == FORTRAN_NULL) resvec_ = NULL;
  else resvec_ = CeedVector_dict[*resvec];

  int createRequest = 1;
  // Check if input is CEED_REQUEST_ORDERED(-2) or CEED_REQUEST_IMMEDIATE(-1)
  if (*rqst == -1 || *rqst == -2) {
    createRequest = 0;
  }

  if (createRequest && CeedRequest_count == CeedRequest_count_max) {
    CeedRequest_count_max += CeedRequest_count_max/2 + 1;
    CeedRealloc(CeedRequest_count_max, &CeedRequest_dict);
  }

  CeedRequest *rqst_;
  if (*rqst == -1) rqst_ = CEED_REQUEST_IMMEDIATE;
  else if (*rqst == -2) rqst_ = CEED_REQUEST_ORDERED;
  else rqst_ = &CeedRequest_dict[CeedRequest_count];

  *err = CeedOperatorApply(CeedOperator_dict[*op], CeedVector_dict[*qdatavec],
                           CeedVector_dict[*ustatevec], resvec_, rqst_);

  if (*err == 0 && createRequest) {
    *rqst = CeedRequest_count++;
    CeedRequest_n++;
  }
}

#define fCeedOperatorApplyJacobian \
    FORTRAN_NAME(ceedoperatorapplyjacobian, CEEDOPERATORAPPLYJACOBIAN)
void fCeedOperatorApplyJacobian(int *op, int *qdatavec, int *ustatevec,
                                int *dustatevec, int *dresvec, int *rqst, int *err) {
// TODO Uncomment this when CeedOperatorApplyJacobian is implemented
//  *err = CeedOperatorApplyJacobian(CeedOperator_dict[*op], CeedVector_dict[*qdatavec],
//             CeedVector_dict[*ustatevec], CeedVector_dict[*dustatevec],
//             CeedVector_dict[*dresvec], &CeedRequest_dict[*rqst]);
}

#define fCeedOperatorDestroy \
    FORTRAN_NAME(ceedoperatordestroy, CEEDOPERATORDESTROY)
void fCeedOperatorDestroy(int *op, int *err) {
  *err = CeedOperatorDestroy(&CeedOperator_dict[*op]);

  if (*err == 0) {
    CeedOperator_n--;
    if (CeedOperator_n == 0) {
      CeedFree(&CeedOperator_dict);
      CeedOperator_count = 0;
      CeedOperator_count_max = 0;
    }
  }
}
