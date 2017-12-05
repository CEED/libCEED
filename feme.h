#ifndef _feme_h
#define _feme_h

#ifdef __cplusplus
#  define FEME_EXTERN extern "C"
#else
#  define FEME_EXTERN extern
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdarg.h>
#include <stdio.h>

// We can discuss ways to avoid forcing these to be compile-time decisions, but let's leave that for later.
typedef int32_t FemeInt;
typedef double FemeScalar;

typedef struct Feme_private *Feme;
typedef struct FemeRequest_private *FemeRequest;
typedef struct FemeVector_private *FemeVector;
typedef struct FemeElemRestriction_private *FemeElemRestriction;
typedef struct FemeBasis_private *FemeBasis;
typedef struct FemeQFunction_private *FemeQFunction;
typedef struct FemeOperator_private *FemeOperator;

FEME_EXTERN int FemeRegister(const char *prefix, int (*init)(const char *, Feme));

FEME_EXTERN int FemeInit(const char *resource, Feme *feme);
FEME_EXTERN int FemeErrorReturn(Feme, const char *, int, const char *, int, const char *, va_list);
FEME_EXTERN int FemeErrorAbort(Feme, const char *, int, const char *, int, const char *, va_list);
FEME_EXTERN int FemeSetErrorHandler(Feme, int (*)(Feme, int, const char *, va_list));
FEME_EXTERN int FemeErrorImpl(Feme, const char *, int, const char *, int, const char *, ...);
#define FemeError(feme, ecode, ...) FemeErrorImpl((feme), __FILE__, __LINE__, __func__, (ecode), __VA_ARGS__)
FEME_EXTERN int FemeDestroy(Feme *feme);
FEME_EXTERN int FemeCompose(int n, const Feme *femes, Feme *composed);

/* FIXME: rename MEM_CUDA --> MEM_DEVICE? */
typedef enum {FEME_MEM_HOST, FEME_MEM_CUDA} FemeMemType;
typedef enum {FEME_COPY_VALUES, FEME_USE_POINTER, FEME_OWN_POINTER} FemeCopyMode;

/* The FemeVectorGet* and FemeVectorRestore* functions provide access to array
   pointers in the desired memory space. Pairing get/restore allows the Vector
   to track access, thus knowing if norms or other operations may need to be
   recomputed. */
FEME_EXTERN int FemeVectorCreate(Feme feme, FemeInt len, FemeVector *vec);
FEME_EXTERN int FemeVectorSetArray(FemeVector vec, FemeMemType mtype, FemeCopyMode cmode, FemeScalar *array);
FEME_EXTERN int FemeVectorGetArray(FemeVector vec, FemeMemType mtype, FemeScalar **array);
FEME_EXTERN int FemeVectorGetArrayRead(FemeVector vec, FemeMemType mtype, const FemeScalar **array);
FEME_EXTERN int FemeVectorRestoreArray(FemeVector vec, FemeScalar **array);
FEME_EXTERN int FemeVectorRestoreArrayRead(FemeVector vec, const FemeScalar **array);
FEME_EXTERN int FemeVectorDestroy(FemeVector *vec);

/* When FEME_REQUEST_IMMEDIATE is passed as the FemeRequest pointer to a call,
   the called function must ensure that all output is immediately available
   after it returns. In other words, the operation does not need to be executed
   asynchronously, and if it is, the called function will wait for the
   asynchronous execution to complete before returning. */
FEME_EXTERN FemeRequest *FEME_REQUEST_IMMEDIATE;
/* When FEME_REQUEST_NULL (or simply NULL) is given as the FemeRequest pointer
   to a function call, the caller is indicating that he/she will not need to
   call FemeRequestWait to wait for the completion of the operation. In general,
   the operation is expected to be executed asyncronously and its result to be
   available before the execution of next asynchronous operation using the same
   Feme. */
#define FEME_REQUEST_NULL ((FemeRequest *)NULL)
FEME_EXTERN int FemeRequestWait(FemeRequest *req);

typedef enum {FEME_NOTRANSPOSE, FEME_TRANSPOSE} FemeTransposeMode;

FEME_EXTERN int FemeElemRestrictionCreate(Feme feme, FemeInt nelements, FemeInt esize, FemeInt ndof, FemeMemType mtype, FemeCopyMode cmode, const FemeInt *indices, FemeElemRestriction *r);
FEME_EXTERN int FemeElemRestrictionCreateBlocked(Feme feme, FemeInt nelements, FemeInt esize, FemeInt blocksize, FemeMemType mtype, FemeCopyMode cmode, FemeInt *blkindices, FemeElemRestriction *r);
FEME_EXTERN int FemeElemRestrictionApply(FemeElemRestriction r, FemeTransposeMode tmode, FemeVector u, FemeVector ru, FemeRequest *request);
FEME_EXTERN int FemeElemRestrictionDestroy(FemeElemRestriction *r);

// The formalism here is that we have the structure
//   \int_\Omega v^T f_0(u, \nabla u, qdata) + (\nabla v)^T f_1(u, \nabla u, qdata)
// where gradients are with respect to the reference element.

typedef enum {FEME_EVAL_NONE = 0, FEME_EVAL_INTERP = 1, FEME_EVAL_GRAD = 2, FEME_EVAL_DIV = 4, FEME_EVAL_CURL = 8} FemeEvalMode;
typedef enum {FEME_GAUSS = 0, FEME_GAUSS_LOBATTO = 1} FemeQuadMode;

FEME_EXTERN int FemeBasisCreateTensorH1Lagrange(Feme feme, FemeInt dim, FemeInt ndof, FemeInt degree, FemeInt Q, FemeQuadMode qmode, FemeBasis *basis);
FEME_EXTERN int FemeBasisCreateTensorH1(Feme feme, FemeInt dim, FemeInt ndof, FemeInt P1d, FemeInt Q1d, const FemeScalar *interp1d, const FemeScalar *grad1d, const FemeScalar *qref1d, const FemeScalar *qweight1d, FemeBasis *basis);
FEME_EXTERN int FemeBasisView(FemeBasis basis, FILE *stream);
FEME_EXTERN int FemeBasisApply(FemeBasis basis, FemeTransposeMode tmode, FemeEvalMode emode, const FemeScalar *u, FemeScalar *v);
FEME_EXTERN int FemeBasisDestroy(FemeBasis *basis);

FEME_EXTERN int FemeGaussQuadrature(FemeInt Q, FemeScalar *qref1d, FemeScalar *qweight1d);
FEME_EXTERN int FemeLobattoQuadrature(FemeInt Q, FemeScalar *qref1d, FemeScalar *qweight1d);

FEME_EXTERN int FemeQFunctionCreateInterior(Feme feme, FemeInt vlength, FemeInt nfields, size_t qdatasize, FemeEvalMode inmode, FemeEvalMode outmode,
                                            int (*f)(void *ctx, void *qdata, FemeInt nq, const FemeScalar *const *u, FemeScalar *const *v),
                                            const char *focca, FemeQFunction *qf);
FEME_EXTERN int FemeQFunctionSetContext(FemeQFunction qf, void *ctx, size_t ctxsize);
FEME_EXTERN int FemeQFunctionDestroy(FemeQFunction *qf);

FEME_EXTERN int FemeOperatorCreate(Feme feme, FemeElemRestriction r, FemeBasis b, FemeQFunction qf, FemeQFunction dqf, FemeQFunction dqfT, FemeOperator *op);
FEME_EXTERN int FemeOperatorGetQData(FemeOperator op, FemeVector *qdata);
FEME_EXTERN int FemeOperatorApply(FemeOperator op, FemeVector qdata, FemeVector ustate, FemeVector residual, FemeRequest *request);
FEME_EXTERN int FemeOperatorApplyJacobian(FemeOperator op, FemeVector qdata, FemeVector ustate, FemeVector dustate, FemeVector dresidual, FemeRequest *request);
FEME_EXTERN int FemeOperatorDestroy(FemeOperator *op);

static inline FemeInt FemePowInt(FemeInt base, FemeInt power) {
  FemeInt result = 1;
  while (power) {
    if (power & 1) result *= base;
    power >>= 1;
    base *= base;
  }
  return result;
}

#endif
