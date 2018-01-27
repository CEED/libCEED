// Fortran interface
#include <ceed.h>
#include <ceed-impl.h>
#include <ceed-fortran-name.h>

#include <stdlib.h>

#define fCeedInit FORTRAN_NAME(ceedinit,CEEDINIT)

#define fCeedDestroy FORTRAN_NAME(ceeddestroy,CEEDDESTROY)

#define fCeedVectorCreate FORTRAN_NAME(ceedvectorcreate,CEEDVECTORCREATE)

#define fCeedVectorDestroy FORTRAN_NAME(ceedvectordestroy,CEEDVECTORDESTROY)

#define fCeedElemRestrictionCreate \
    FORTRAN_NAME(ceedelemrestrictioncreate, CEEDELEMRESTRICTIONCREATE)

#define fCeedElemRestrictionDestroy \
    FORTRAN_NAME(ceedelemrestrictiondestroy,CEEDELEMRESTRICTIONDESTROY)

#define fCeedBasisCreateTensorH1Lagrange \
    FORTRAN_NAME(ceedbasiscreatetensorh1lagrange, CEEDBASISCREATETENSORH1LAGRANGE)

#define fCeedBasisDestroy FORTRAN_NAME(ceedbasisdestroy,CEEDBASISDESTROY)

#define fCeedQFunctionCreateInterior \
    FORTRAN_NAME(ceedqfunctioncreateinterior, CEEDQFUNCTIONCREATEINTERIOR)

#define fCeedQFunctionDestroy \
    FORTRAN_NAME(ceedqfunctiondestroy,ceedqfunctiondestroy)

#define fCeedQFunctionSetContext \
    FORTRAN_NAME(ceedqfunctionsetcontext, CEEDQFUNCTIONSETCONTEXT)

#define fCeedOperatorCreate \
    FORTRAN_NAME(ceedoperatorcreate, CEEDOPERATORCREATE)

#define fCeedOperatorDestroy \
    FORTRAN_NAME(ceedoperatordestroy, CEEDOPERATORDESTROY)

/// @defgroup FortranAPI Ceed: Fortran Interface
/// @{
/// Fortran interface
///
static Ceed *Ceed_dict = NULL;
static int Ceed_count = 0;
static int Ceed_n = 0;
static int Ceed_count_max = 0;

void fCeedInit(const char* resource, CeedInt *ceed, CeedInt *err) {
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

void fCeedDestroy(CeedInt *ceed, CeedInt *err) {
  *err = CeedDestroy(&Ceed_dict[*ceed]);

  if (*err == 0) {
    Ceed_n--;
    if (Ceed_n == 0) CeedFree(&Ceed_dict);
  }
}

static CeedVector *CeedVector_dict = NULL;
static int CeedVector_count = 0;
static int CeedVector_n = 0;
static int CeedVector_count_max = 0;

void fCeedVectorCreate(CeedInt *ceed, CeedInt *length, CeedInt *vec, CeedInt *err) {
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

void fCeedVectorDestroy(CeedInt *vec, CeedInt *err) {
  *err = CeedVectorDestroy(&CeedVector_dict[*vec]);

  if (*err == 0) {
    CeedVector_n--;
    if (CeedVector_n == 0) CeedFree(&CeedVector_dict);
  }
}

static CeedElemRestriction *CeedElemRestriction_dict = NULL;
static int CeedElemRestriction_count = 0;
static int CeedElemRestriction_n = 0;
static int CeedElemRestriction_count_max = 0;

void fCeedElemRestrictionCreate(CeedInt *ceed, CeedInt *nelements,
    CeedInt *esize, CeedInt *ndof, CeedInt *memtype, CeedInt *copymode,
    const CeedInt *indices, CeedInt *elemrestriction, CeedInt *err) {
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

void fCeedElemRestrictionDestroy(CeedInt *elem, CeedInt *err) {
  *err = CeedElemRestrictionDestroy(&CeedElemRestriction_dict[*elem]);

  if (*err == 0) {
    CeedElemRestriction_n--;
    if (CeedElemRestriction_n == 0) CeedFree(&CeedElemRestriction_dict);
  }
}

static CeedBasis *CeedBasis_dict = NULL;
static int CeedBasis_count = 0;
static int CeedBasis_n = 0;
static int CeedBasis_count_max = 0;

void fCeedBasisCreateTensorH1Lagrange(CeedInt *ceed, CeedInt *dim,
    CeedInt *ndof, CeedInt *P, CeedInt *Q, CeedInt *quadmode, CeedInt *basis,
    CeedInt *err) {
  if (CeedBasis_count == CeedBasis_count_max) {
    CeedBasis_count_max += CeedBasis_count_max/2 + 1;
    CeedRealloc(CeedBasis_count_max, &CeedBasis_dict);
  }

  CeedBasis *basis_ = &CeedBasis_dict[CeedBasis_count];
  *err = CeedBasisCreateTensorH1Lagrange(Ceed_dict[*ceed], *dim, *ndof, *P, *Q,
             *quadmode, basis_);

  if (*err == 0) {
    *basis = CeedBasis_count++;
    CeedBasis_n++;
  }
}

void fCeedBasisDestroy(CeedInt *basis, CeedInt *err) {
  *err = CeedBasisDestroy(&CeedBasis_dict[*basis]);

  if (*err == 0) {
    CeedBasis_n--;
    if (CeedBasis_n == 0) CeedFree(&CeedBasis_dict);
  }
}

static CeedQFunction *CeedQFunction_dict = NULL;
static int CeedQFunction_count = 0;
static int CeedQFunction_n = 0;
static int CeedQFunction_count_max = 0;

struct fContext {
  void (*f)(void *ctx, void *qdata, CeedInt *nq, const CeedScalar *const *u,
           CeedScalar *const *v, int *ierr);
  void *innerctx;
};

static int CeedQFunctionFortranStub(void *ctx, void *qdata, CeedInt nq,
    const CeedScalar *const *u, CeedScalar *const *v) {
  struct fContext *fctx = ctx;
  int ierr;
  fctx->f(fctx->innerctx, qdata, &nq, u, v, &ierr);
  return ierr;
}

void fCeedQFunctionCreateInterior(CeedInt* ceed, CeedInt* vlength,
    CeedInt* nfields, size_t* qdatasize, CeedInt* inmode, CeedInt* outmode,
    void (*f)(void *ctx, void *qdata, CeedInt *nq, const CeedScalar *const *u,
             CeedScalar *const *v, int *err), const char *focca, CeedInt *qf,
    CeedInt *err) {
  if (CeedQFunction_count == CeedQFunction_count_max) {
    CeedQFunction_count_max += CeedQFunction_count_max/2 + 1;
    CeedRealloc(CeedQFunction_count_max, &CeedQFunction_dict);
  }

  CeedQFunction *qf_ = &CeedQFunction_dict[CeedQFunction_count];
  *err = CeedQFunctionCreateInterior(Ceed_dict[*ceed], *vlength, *nfields,
             *qdatasize, *inmode, *outmode, CeedQFunctionFortranStub,focca, qf_);

  if (*err == 0) {
    *qf = CeedQFunction_count++;
    CeedQFunction_n++;
  }

  struct fContext *fctx; CeedMalloc(1, &fctx);
  fctx->f = f; fctx->innerctx = NULL;

  CeedQFunctionSetContext(*qf_, fctx, sizeof(struct fContext));

}

void fCeedQFunctionDestroy(CeedInt *qf, CeedInt *err) {
  CeedFree(&CeedQFunction_dict[*qf]->ctx);
  *err = CeedQFunctionDestroy(&CeedQFunction_dict[*qf]);

  if (*err == 0) {
    CeedQFunction_n--;
    if (CeedQFunction_n == 0) CeedFree(&CeedQFunction_dict);
  }
}

void fCeedQFunctionSetContext(CeedInt *qf, void *ctx, size_t* ctxsize,
    CeedInt *err) {
  CeedQFunction qf_ = CeedQFunction_dict[*qf];

  struct fContext *newFContext; CeedMalloc(1, &newFContext);
  newFContext->f = ((struct fContext *)(qf_->ctx))->f;
  newFContext->innerctx = ctx;

  *err = CeedQFunctionSetContext(qf_, newFContext, sizeof(struct fContext));
}

static CeedOperator *CeedOperator_dict = NULL;
static int CeedOperator_count = 0;
static int CeedOperator_n = 0;
static int CeedOperator_count_max = 0;

void fCeedOperatorCreate(CeedInt* ceed, CeedInt* erstrn, CeedInt* basis,
    CeedInt* qf, CeedInt* dqf, CeedInt* dqfT, CeedInt *op, CeedInt *err) {
  if (CeedOperator_count == CeedOperator_count_max)
    CeedOperator_count_max += CeedOperator_count_max/2 + 1,
    CeedOperator_dict =
        realloc(CeedOperator_dict, sizeof(CeedOperator)*CeedOperator_count_max);

  CeedOperator *op_ = &CeedOperator_dict[CeedOperator_count];

  CeedQFunction dqf_  = NULL, dqfT_ = NULL;
  if (dqf  != NULL) dqf_  = CeedQFunction_dict[*dqf ];
  if (dqfT != NULL) dqfT_ = CeedQFunction_dict[*dqfT];

  *err = CeedOperatorCreate(Ceed_dict[*ceed], CeedElemRestriction_dict[*erstrn],
             CeedBasis_dict[*basis], CeedQFunction_dict[*qf], dqf_, dqfT_, op_);

  if (*err == 0) {
    *op = CeedOperator_count++;
    CeedOperator_n++;
  }
}

void fCeedOperatorDestroy(CeedInt *op, CeedInt *err) {
  *err = CeedOperatorDestroy(&CeedOperator_dict[*op]);

  if (*err == 0) {
    CeedOperator_n--;
    if (CeedOperator_n == 0) CeedFree(&CeedOperator_dict);
  }
}
/// @}
