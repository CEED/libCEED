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

/// @defgroup FortranAPI Ceed: Fortran Interface
/// @{
/// Fortran interface
///
static Ceed *Ceed_dict = NULL;
static int Ceed_count = 0;
static int Ceed_count_max = 0;

void fCeedInit(const char* resource, CeedInt *ceed, CeedInt *err) {
  if (Ceed_count == Ceed_count_max)
    Ceed_count_max += Ceed_count_max/2 + 1,
    Ceed_dict = realloc(Ceed_dict, sizeof(Ceed)*Ceed_count_max);

  Ceed *ceed_ = &Ceed_dict[Ceed_count];
  *err = CeedInit(resource, ceed_);

  *ceed = Ceed_count++;
}

void fCeedDestroy(CeedInt *ceed, CeedInt *err) {
  *err = CeedDestroy(&Ceed_dict[*ceed]);
}

static CeedVector *CeedVector_dict = NULL;
static int CeedVector_count = 0;
static int CeedVector_count_max = 0;

void fCeedVectorCreate(CeedInt *ceed, CeedInt *length, CeedInt *vec, CeedInt *err) {
  if (CeedVector_count == CeedVector_count_max)
    CeedVector_count_max += CeedVector_count_max/2 + 1,
    CeedVector_dict =
        realloc(CeedVector_dict, sizeof(CeedVector)*CeedVector_count_max);

  CeedVector* vec_ = &CeedVector_dict[CeedVector_count];
  *err = CeedVectorCreate(Ceed_dict[*ceed], *length, vec_);

  *vec = CeedVector_count++;
}

void fCeedVectorDestroy(CeedInt *vec, CeedInt *err) {
  *err = CeedVectorDestroy(&CeedVector_dict[*vec]);
}

static CeedElemRestriction *CeedElemRestriction_dict = NULL;
static int CeedElemRestriction_count = 0;
static int CeedElemRestriction_count_max = 0;

void fCeedElemRestrictionCreate(CeedInt *ceed, CeedInt *nelements,
    CeedInt *esize, CeedInt *ndof, CeedInt *memtype, CeedInt *copymode,
    const CeedInt *indices, CeedInt *elemrestriction, CeedInt *err) {
  if (CeedElemRestriction_count == CeedElemRestriction_count_max)
    CeedElemRestriction_count_max += CeedElemRestriction_count_max/2 + 1,
    CeedElemRestriction_dict =
        realloc(CeedElemRestriction_dict, \
        sizeof(CeedElemRestriction)*CeedElemRestriction_count_max);

  CeedElemRestriction *elemrestriction_ =
      &CeedElemRestriction_dict[CeedElemRestriction_count];
  *err = CeedElemRestrictionCreate(Ceed_dict[*ceed], *nelements, *esize, *ndof,
             *memtype, *copymode, indices, elemrestriction_);

  *elemrestriction = CeedElemRestriction_count++;
}

void fCeedElemRestrictionDestroy(CeedInt *elem, CeedInt *err) {
  *err = CeedElemRestrictionDestroy(&CeedElemRestriction_dict[*elem]);
}

static CeedBasis *CeedBasis_dict = NULL;
static int CeedBasis_count = 0;
static int CeedBasis_count_max = 0;

void fCeedBasisCreateTensorH1Lagrange(CeedInt *ceed, CeedInt *dim,
    CeedInt *ndof, CeedInt *P, CeedInt *Q, CeedInt *quadmode, CeedInt *basis,
    CeedInt *err) {
  if (CeedBasis_count == CeedBasis_count_max)
    CeedBasis_count_max += CeedBasis_count_max/2 + 1,
    CeedBasis_dict = realloc(CeedBasis_dict, sizeof(CeedBasis)*CeedBasis_count_max);

  CeedBasis *basis_ = &CeedBasis_dict[CeedBasis_count];
  *err = CeedBasisCreateTensorH1Lagrange(Ceed_dict[*ceed], *dim, *ndof, *P, *Q,
             *quadmode, basis_);

  *basis = CeedBasis_count++;
}

void fCeedBasisDestroy(CeedInt *basis, CeedInt *err) {
  *err = CeedBasisDestroy(&CeedBasis_dict[*basis]);
}

static CeedQFunction *CeedQFunction_dict = NULL;
static int CeedQFunction_count = 0;
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
  if (CeedQFunction_count == CeedQFunction_count_max)
    CeedQFunction_count_max += CeedQFunction_count_max/2 + 1,
    CeedQFunction_dict =
        realloc(CeedQFunction_dict, sizeof(CeedQFunction)*CeedQFunction_count_max);

  CeedQFunction *qf_ = &CeedQFunction_dict[CeedQFunction_count];
  *err = CeedQFunctionCreateInterior(Ceed_dict[*ceed], *vlength, *nfields,
             *qdatasize, *inmode, *outmode, CeedQFunctionFortranStub,focca, qf_);

  struct fContext *fctx = malloc(sizeof(struct fContext));
  fctx->f = f; fctx->innerctx = NULL;

  CeedQFunctionSetContext(*qf_, fctx, sizeof(struct fContext));

  *qf = CeedQFunction_count++;
}

void fCeedQFunctionDestroy(CeedInt *qf, CeedInt *err) {
  free(CeedQFunction_dict[*qf]->ctx);
  *err = CeedQFunctionDestroy(&CeedQFunction_dict[*qf]);
}

void fCeedQFunctionSetContext(CeedInt *qf, void *ctx, size_t* ctxsize,
    CeedInt *err) {
  CeedQFunction qf_ = CeedQFunction_dict[*qf];

  struct fContext *newFContext = malloc(sizeof(struct fContext));
  newFContext->f = ((struct fContext *)(qf_->ctx))->f;
  newFContext->innerctx = ctx;

  *err = CeedQFunctionSetContext(qf_, newFContext, sizeof(struct fContext));
}

static CeedOperator *CeedOperator_dict = NULL;
static int CeedOperator_count = 0;
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

  *op = CeedOperator_count++;
}
/// @}
