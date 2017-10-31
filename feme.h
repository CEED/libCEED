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

// We can discuss ways to avoid forcing these to be compile-time decisions, but let's leave that for later.
typedef int32_t FemeInt;
typedef double FemeScalar;

typedef struct Feme_private *Feme;
typedef struct FemeRequest_private *FemeRequest;
typedef struct FemeVec_private *FemeVec;
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

typedef enum {FEME_MEM_HOST, FEME_MEM_CUDA} FemeMemType;
typedef enum {FEME_COPY_VALUES, FEME_USE_POINTER, FEME_OWN_POINTER} FemeCopyMode;

FEME_EXTERN int FemeVecCreate(Feme feme, FemeInt len, FemeVec *vec);
FEME_EXTERN int FemeVecSetArray(FemeVec vec, FemeMemType mtype, FemeCopyMode cmode, FemeScalar *array);
FEME_EXTERN int FemeVecGetArray(FemeVec vec, FemeMemType mtype, FemeScalar **array);
FEME_EXTERN int FemeVecGetArrayRead(FemeVec vec, FemeMemType mtype, const FemeScalar **array);
FEME_EXTERN int FemeVecRestoreArray(FemeVec vec, FemeScalar **array);
FEME_EXTERN int FemeVecRestoreArrayRead(FemeVec vec, const FemeScalar **array);
FEME_EXTERN int FemeVecDestroy(FemeVec *vec);

FEME_EXTERN FemeRequest *FEME_REQUEST_IMMEDIATE; // Use when you don't want to wait
FEME_EXTERN int FemeRequestWait(FemeRequest *req);

typedef enum {FEME_TRANSPOSE, FEME_NOTRANSPOSE} FemeTransposeMode;

FEME_EXTERN int FemeElemRestrictionCreate(Feme feme, FemeInt nelements, FemeInt esize, FemeMemType mtype, FemeCopyMode cmode, FemeInt *indices, FemeElemRestriction *r);
FEME_EXTERN int FemeElemRestrictionCreateBlocked(Feme feme, FemeInt nelements, FemeInt esize, FemeInt blocksize, FemeMemType mtype, FemeCopyMode cmode, FemeInt *blkindices, FemeElemRestriction *r);
FEME_EXTERN int FemeElemRestrictionApply(FemeElemRestriction r, FemeTransposeMode tmode, FemeVec u, FemeVec ru, FemeRequest *request);
FEME_EXTERN int FemeElemRestrictionDestroy(FemeElemRestriction *r);

// The formalism here is that we have the structure
//   \int_\Omega v^T f_0(u, \nabla u, qdata) + (\nabla v)^T f_1(u, \nabla u, qdata)
// where gradients are with respect to the reference element.

typedef enum {FEME_EVAL_NONE = 0, FEME_EVAL_INTERP = 1, FEME_EVAL_GRAD = 2, FEME_EVAL_DIV = 4, FEME_EVAL_CURL = 8} FemeEvalMode;

FEME_EXTERN int FemeBasisCreateTensorH1Lagrange(Feme feme, FemeInt dim, FemeInt degree, FemeInt Q, FemeBasis *basis);
FEME_EXTERN int FemeBasisCreateTensorH1(Feme feme, FemeInt dim, FemeInt P1d, FemeInt Q1d, const FemeScalar *interp1d, const FemeScalar *grad1d, const FemeScalar *qref1d, const FemeScalar *qweight1d, FemeBasis *basis);
FEME_EXTERN int FemeBasisApply(FemeBasis basis, FemeTransposeMode tmode, FemeEvalMode emode, const FemeScalar *const *u, FemeScalar *const *v);
FEME_EXTERN int FemeBasisDestroy(FemeBasis *basis);

FEME_EXTERN int FemeQFunctionCreateInterior(Feme feme, FemeInt vlength, FemeInt nfields, size_t qdatasize, FemeEvalMode inmode, FemeEvalMode outmode,
                                            int (*f)(void *ctx, void *qdata, FemeInt nq, const FemeScalar *const *u, FemeScalar *const *v),
                                            const char *focca, FemeQFunction *qf);
FEME_EXTERN int FemeQFunctionSetContext(FemeQFunction qf, void *ctx, size_t ctxsize);
FEME_EXTERN int FemeQFunctionDestroy(FemeQFunction *qf);

FEME_EXTERN int FemeOperatorCreate(Feme feme, FemeElemRestriction r, FemeBasis b, FemeQFunction qf, FemeQFunction dqf, FemeQFunction dqfT, FemeOperator *op);
FEME_EXTERN int FemeOperatorGetQData(FemeOperator op, FemeVec *qdata);
FEME_EXTERN int FemeOperatorApply(FemeOperator op, FemeVec qdata, FemeVec ustate, FemeVec residual, FemeRequest *request);
FEME_EXTERN int FemeOperatorApplyJacobian(FemeOperator op, FemeVec qdata, FemeVec ustate, FemeVec dustate, FemeVec dresidual, FemeRequest *request);
FEME_EXTERN int FemeOperatorDestroy(FemeOperator *op);

#endif
