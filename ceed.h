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

#ifndef _ceed_h
#define _ceed_h

#ifdef __cplusplus
#  define CEED_EXTERN extern "C"
#else
#  define CEED_EXTERN extern
#endif

#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdarg.h>
#include <stdio.h>

// We can discuss ways to avoid forcing these to be compile-time decisions, but let's leave that for later.
typedef int32_t CeedInt;
typedef double CeedScalar;

typedef struct Ceed_private *Ceed;
typedef struct CeedRequest_private *CeedRequest;
typedef struct CeedVector_private *CeedVector;
typedef struct CeedElemRestriction_private *CeedElemRestriction;
typedef struct CeedBasis_private *CeedBasis;
typedef struct CeedQFunction_private *CeedQFunction;
typedef struct CeedOperator_private *CeedOperator;

CEED_EXTERN int CeedRegister(const char *prefix, int (*init)(const char *,
                             Ceed));

CEED_EXTERN int CeedInit(const char *resource, Ceed *ceed);
CEED_EXTERN int CeedErrorReturn(Ceed, const char *, int, const char *, int,
                                const char *, va_list);
CEED_EXTERN int CeedErrorAbort(Ceed, const char *, int, const char *, int,
                               const char *, va_list);
CEED_EXTERN int CeedSetErrorHandler(Ceed,
                                    int (*)(Ceed, int, const char *, va_list));
CEED_EXTERN int CeedErrorImpl(Ceed, const char *, int, const char *, int,
                              const char *, ...);
#define CeedError(ceed, ecode, ...)                                     \
  CeedErrorImpl((ceed), __FILE__, __LINE__, __func__, (ecode), __VA_ARGS__)
CEED_EXTERN int CeedDestroy(Ceed *ceed);
CEED_EXTERN int CeedCompose(int n, const Ceed *ceeds, Ceed *composed);

typedef enum {CEED_MEM_HOST, CEED_MEM_DEVICE} CeedMemType;
/* When ownership of dynamically alocated CEED_MEM_HOST pointers is transferred
   to the library (CEED_OWN_POINTER mode), they will be deallocated by calling
   the standard C library function, free(). In particular, pointers allocated
   with the C++ operator new should not be used with CEED_OWN_POINTER mode. */
typedef enum {CEED_COPY_VALUES, CEED_USE_POINTER, CEED_OWN_POINTER} CeedCopyMode;

/* The CeedVectorGet* and CeedVectorRestore* functions provide access to array
   pointers in the desired memory space. Pairing get/restore allows the Vector
   to track access, thus knowing if norms or other operations may need to be
   recomputed. */
CEED_EXTERN int CeedVectorCreate(Ceed ceed, CeedInt len, CeedVector *vec);
CEED_EXTERN int CeedVectorSetArray(CeedVector vec, CeedMemType mtype,
                                   CeedCopyMode cmode, CeedScalar *array);
CEED_EXTERN int CeedVectorGetArray(CeedVector vec, CeedMemType mtype,
                                   CeedScalar **array);
CEED_EXTERN int CeedVectorGetArrayRead(CeedVector vec, CeedMemType mtype,
                                       const CeedScalar **array);
CEED_EXTERN int CeedVectorRestoreArray(CeedVector vec, CeedScalar **array);
CEED_EXTERN int CeedVectorRestoreArrayRead(CeedVector vec,
    const CeedScalar **array);
CEED_EXTERN int CeedVectorDestroy(CeedVector *vec);

/* When CEED_REQUEST_IMMEDIATE is passed as the CeedRequest pointer to a call,
   the called function must ensure that all output is immediately available
   after it returns. In other words, the operation does not need to be executed
   asynchronously, and if it is, the called function will wait for the
   asynchronous execution to complete before returning. */
CEED_EXTERN CeedRequest *CEED_REQUEST_IMMEDIATE;
/* When CEED_REQUEST_NULL (or simply NULL) is given as the CeedRequest pointer
   to a function call, the caller is indicating that he/she will not need to
   call CeedRequestWait to wait for the completion of the operation. In general,
   the operation is expected to be executed asyncronously and its result to be
   available before the execution of next asynchronous operation using the same
   Ceed. */
#define CEED_REQUEST_NULL ((CeedRequest *)NULL)
CEED_EXTERN int CeedRequestWait(CeedRequest *req);

typedef enum {CEED_NOTRANSPOSE, CEED_TRANSPOSE} CeedTransposeMode;

/**
  @brief Create a CeedElemRestriction

  @param ceed       A Ceed object where the CeedElemRestriction will be created.
  @param nelements  Number of elements described in the @a indices array.
  @param esize      Size (number of unknowns) per element.
  @param ndof       The total size of the input CeedVector to which the
                    restriction will be applied. This size may include data
                    used by other CeedElemRestriction objects describing
                    different types of elements.
  @param mtype      Memory type of the @a indices array, see CeedMemType.
  @param cmode      Copy mode for the @a indices array, see CeedCopyMode.
  @param indices    A 2D array of dimensions (@a esize x @a nelements) using
                    column-major storage layout. Column i holds the ordered list
                    of the indices (into the input CeedVector) for the unknowns
                    corresponding to element i, where 0 <= i < @a nelements.
                    All indices must be in the range [0, @a ndof).
  @param r          The address of the variable where the newly created
                    CeedElemRestriction will be stored.

  @return An error code: 0 - success, otherwise - failure.
 */
CEED_EXTERN int CeedElemRestrictionCreate(Ceed ceed, CeedInt nelements,
    CeedInt esize, CeedInt ndof, CeedMemType mtype, CeedCopyMode cmode,
    const CeedInt *indices, CeedElemRestriction *r);

/**
  @brief Create a blocked CeedElemRestriction

  @param ceed        A Ceed object where the CeedElemRestriction will be created.
  @param nelements   Number of elements described ...
  @param esize       Size (number of unknowns) per element.
  @param blocksize   ...
  @param mtype       Memory type of the @a blkindices array, see CeedMemType.
  @param cmode       Copy mode for the @a blkindices array, see CeedCopyMode.
  @param blkindices  ...
  @param r           The address of the variable where the newly created
                     CeedElemRestriction will be stored.

  @return An error code: 0 - success, otherwise - failure.
 */
CEED_EXTERN int CeedElemRestrictionCreateBlocked(Ceed ceed, CeedInt nelements,
    CeedInt esize, CeedInt blocksize, CeedMemType mtype, CeedCopyMode cmode,
    CeedInt *blkindices, CeedElemRestriction *r);
CEED_EXTERN int CeedElemRestrictionApply(CeedElemRestriction r,
    CeedTransposeMode tmode, CeedInt ncomp, CeedTransposeMode lmode, CeedVector u,
    CeedVector ru, CeedRequest *request);
CEED_EXTERN int CeedElemRestrictionDestroy(CeedElemRestriction *r);

// The formalism here is that we have the structure
//   \int_\Omega v^T f_0(u, \nabla u, qdata) + (\nabla v)^T f_1(u, \nabla u, qdata)
// where gradients are with respect to the reference element.

typedef enum {CEED_EVAL_NONE   = 0,
              CEED_EVAL_INTERP = 1, // values at quadrature points
              CEED_EVAL_GRAD   = 2, // gradients
              CEED_EVAL_DIV    = 4, // divergence
              CEED_EVAL_CURL   = 8, // curl
              CEED_EVAL_WEIGHT = 16, // quadrature weights for reference element
             } CeedEvalMode;
typedef enum {CEED_GAUSS = 0, CEED_GAUSS_LOBATTO = 1} CeedQuadMode;

CEED_EXTERN int CeedBasisCreateTensorH1Lagrange(Ceed ceed, CeedInt dim,
    CeedInt ndof, CeedInt P, CeedInt Q, CeedQuadMode qmode, CeedBasis *basis);
CEED_EXTERN int CeedBasisCreateTensorH1(Ceed ceed, CeedInt dim, CeedInt ndof,
                                        CeedInt P1d, CeedInt Q1d, const CeedScalar *interp1d, const CeedScalar *grad1d,
                                        const CeedScalar *qref1d, const CeedScalar *qweight1d, CeedBasis *basis);
CEED_EXTERN int CeedBasisView(CeedBasis basis, FILE *stream);
CEED_EXTERN int CeedBasisApply(CeedBasis basis, CeedTransposeMode tmode,
                               CeedEvalMode emode, const CeedScalar *u, CeedScalar *v);
CEED_EXTERN int CeedBasisGetNumNodes(CeedBasis basis, CeedInt *P);
CEED_EXTERN int CeedBasisGetNumQuadraturePoints(CeedBasis basis, CeedInt *Q);
CEED_EXTERN int CeedBasisDestroy(CeedBasis *basis);

CEED_EXTERN int CeedGaussQuadrature(CeedInt Q, CeedScalar *qref1d,
                                    CeedScalar *qweight1d);
CEED_EXTERN int CeedLobattoQuadrature(CeedInt Q, CeedScalar *qref1d,
                                      CeedScalar *qweight1d);

CEED_EXTERN int CeedQFunctionCreateInterior(Ceed ceed, CeedInt vlength,
    CeedInt nfields, size_t qdatasize, CeedEvalMode inmode, CeedEvalMode outmode,
    int (*f)(void *ctx, void *qdata, CeedInt nq, const CeedScalar *const *u,
             CeedScalar *const *v), const char *focca, CeedQFunction *qf);
CEED_EXTERN int CeedQFunctionSetContext(CeedQFunction qf, void *ctx,
                                        size_t ctxsize);
CEED_EXTERN int CeedQFunctionApply(CeedQFunction qf, void *qdata, CeedInt Q,
                                   const CeedScalar *const *u,
                                   CeedScalar *const *v);
CEED_EXTERN int CeedQFunctionDestroy(CeedQFunction *qf);

CEED_EXTERN int CeedOperatorCreate(Ceed ceed, CeedElemRestriction r,
                                   CeedBasis b, CeedQFunction qf, CeedQFunction dqf, CeedQFunction dqfT,
                                   CeedOperator *op);
CEED_EXTERN int CeedOperatorGetQData(CeedOperator op, CeedVector *qdata);
CEED_EXTERN int CeedOperatorApply(CeedOperator op, CeedVector qdata,
                                  CeedVector ustate, CeedVector residual, CeedRequest *request);
CEED_EXTERN int CeedOperatorApplyJacobian(CeedOperator op, CeedVector qdata,
    CeedVector ustate, CeedVector dustate, CeedVector dresidual,
    CeedRequest *request);
CEED_EXTERN int CeedOperatorDestroy(CeedOperator *op);

static inline CeedInt CeedPowInt(CeedInt base, CeedInt power) {
  CeedInt result = 1;
  while (power) {
    if (power & 1) result *= base;
    power >>= 1;
    base *= base;
  }
  return result;
}

#endif
