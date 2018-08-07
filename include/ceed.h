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

/// @file
/// Public header for libCEED
#ifndef _ceed_h
#define _ceed_h

/**
  CEED_EXTERN is used in this header to denote all publicly visible symbols.

  No other file should declare publicly visible symbols, thus it should never be
  used outside ceed.h.
 */
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
/// Integer type, used for indexing
/// @ingroup Ceed
typedef int32_t CeedInt;
/// Scalar (floating point) type
/// @ingroup Ceed
typedef double CeedScalar;

/// Library context created by CeedInit()
/// @ingroup Ceed
typedef struct Ceed_private *Ceed;
/// Non-blocking Ceed interfaces return a CeedRequest.
/// To perform an operation immediately, pass \ref CEED_REQUEST_IMMEDIATE instead.
/// @ingroup Ceed
typedef struct CeedRequest_private *CeedRequest;
/// Handle for vectors over the field \ref CeedScalar
/// @ingroup CeedVector
typedef struct CeedVector_private *CeedVector;
/// Handle for object describing restriction to elements
/// @ingroup CeedElemRestriction
typedef struct CeedElemRestriction_private *CeedElemRestriction;
/// Handle for object describing discrete finite element evaluations
/// @ingroup CeedBasis
typedef struct CeedBasis_private *CeedBasis;
/// Handle for object describing functions evaluated independently at quadrature points
/// @ingroup CeedQFunction
typedef struct CeedQFunction_private *CeedQFunction;
/// Handle for object describing FE-type operators acting on vectors
///
/// Given an element restriction \f$E\f$, basis evaluator \f$B\f$, and quadrature function
/// \f$f\f$, a CeedOperator expresses operations of the form
///   $$ E^T B^T f(B E u) $$
/// acting on the vector \f$u\f$.
typedef struct CeedOperator_private *CeedOperator;

CEED_EXTERN int CeedRegister(const char *prefix,
                             int (*init)(const char *, Ceed), unsigned int priority);

CEED_EXTERN int CeedInit(const char *resource, Ceed *ceed);
CEED_EXTERN int CeedErrorReturn(Ceed, const char *, int, const char *, int,
                                const char *, va_list);
CEED_EXTERN int CeedErrorAbort(Ceed, const char *, int, const char *, int,
                               const char *, va_list);
CEED_EXTERN int CeedSetErrorHandler(Ceed ceed,
                                    int (eh)(Ceed, const char *, int, const char *,
                                        int, const char *, va_list));
CEED_EXTERN int CeedErrorImpl(Ceed, const char *, int, const char *, int,
                              const char *, ...);
/// Raise an error on ceed object
///
/// @param ceed Ceed library context or NULL
/// @param ecode Error code (int)
/// @param ... printf-style format string followed by arguments as needed
///
/// @ingroup Ceed
/// @sa CeedSetErrorHandler()
#define CeedError(ceed, ecode, ...)                                     \
  CeedErrorImpl((ceed), __FILE__, __LINE__, __func__, (ecode), __VA_ARGS__)
CEED_EXTERN int CeedDestroy(Ceed *ceed);

/// Specify memory type
///
/// Many Ceed interfaces take or return pointers to memory.  This enum is used to
/// specify where the memory being provided or requested must reside.
/// @ingroup Ceed
typedef enum {
  /// Memory resides on the host
  CEED_MEM_HOST,
  /// Memory resides on a device (corresponding to \ref Ceed resource)
  CEED_MEM_DEVICE,
} CeedMemType;

/// Conveys ownership status of arrays passed to Ceed interfaces.
/// @ingroup Ceed
typedef enum {
  /// Implementation will copy the values and not store the passed pointer.
  CEED_COPY_VALUES,
  /// Implementation can use and modify the data provided by the user, but does
  /// not take ownership.
  CEED_USE_POINTER,
  /// Implementation takes ownership of the pointer and will free using
  /// CeedFree() when done using it.  The user should not assume that the
  /// pointer remains valid after ownership has been transferred.  Note that
  /// arrays allocated using C++ operator new or other allocators cannot
  /// generally be freed using CeedFree().  CeedFree() is capable of freeing any
  /// memory that can be freed using free(3).
  CEED_OWN_POINTER,
} CeedCopyMode;

CEED_EXTERN int CeedVectorCreate(Ceed ceed, CeedInt len, CeedVector *vec);
CEED_EXTERN int CeedVectorSetArray(CeedVector vec, CeedMemType mtype,
                                   CeedCopyMode cmode, CeedScalar *array);
CEED_EXTERN int CeedVectorSetValue(CeedVector vec, CeedScalar value);
CEED_EXTERN int CeedVectorGetArray(CeedVector vec, CeedMemType mtype,
                                   CeedScalar **array);
CEED_EXTERN int CeedVectorGetArrayRead(CeedVector vec, CeedMemType mtype,
                                       const CeedScalar **array);
CEED_EXTERN int CeedVectorRestoreArray(CeedVector vec, CeedScalar **array);
CEED_EXTERN int CeedVectorRestoreArrayRead(CeedVector vec,
    const CeedScalar **array);
CEED_EXTERN int CeedVectorView(CeedVector vec, const char *fpfmt, FILE *stream);
CEED_EXTERN int CeedVectorGetLength(CeedVector vec, CeedInt *length);
CEED_EXTERN int CeedVectorDestroy(CeedVector *vec);

CEED_EXTERN CeedRequest *const CEED_REQUEST_IMMEDIATE;
CEED_EXTERN CeedRequest *const CEED_REQUEST_ORDERED;
CEED_EXTERN int CeedRequestWait(CeedRequest *req);

/// Argument for CeedOperatorSetField that vector is colocated with
/// quadrature points, used with qfunction eval mode CEED_EVAL_NONE
/// or CEED_EVAL_INTERP only, not with CEED_EVAL_GRAD, CEED_EVAL_DIV,
/// or CEED_EVAL_CURL
/// @ingroup CeedBasis
CEED_EXTERN CeedBasis CEED_BASIS_COLOCATED;

/// Argument for CeedOperatorSetField to use active input or output
/// @ingroup CeedVector
CEED_EXTERN CeedVector CEED_VECTOR_ACTIVE;

/// Argument for CeedOperatorSetField to use no vector, used with
/// qfunction input with eval mode CEED_EVAL_WEIGHTS
/// @ingroup CeedVector
CEED_EXTERN CeedVector CEED_VECTOR_NONE;

/// Denotes whether a linear transformation or its transpose should be applied
/// @ingroup CeedBasis
typedef enum {
  /// Apply the linear transformation
  CEED_NOTRANSPOSE,
  /// Apply the transpose
  CEED_TRANSPOSE
} CeedTransposeMode;

CEED_EXTERN int CeedElemRestrictionCreate(Ceed ceed, CeedInt nelem,
    CeedInt elemsize, CeedInt ndof, CeedInt ncomp, CeedMemType mtype, CeedCopyMode cmode,
    const CeedInt *indices, CeedElemRestriction *r);
CEED_EXTERN int CeedElemRestrictionCreateIdentity(Ceed ceed, CeedInt nelem,
    CeedInt elemsize, CeedInt ndof, CeedInt ncomp, CeedElemRestriction *r);
CEED_EXTERN int CeedElemRestrictionCreateBlocked(Ceed ceed, CeedInt nelem,
    CeedInt elemsize, CeedInt blksize, CeedInt ndof, CeedInt ncomp, CeedMemType mtype,
    CeedCopyMode cmode, const CeedInt *indices, CeedElemRestriction *r);
CEED_EXTERN int CeedElemRestrictionCreateVector(CeedElemRestriction r,
                                                CeedVector *lvec,
                                                CeedVector *evec);
CEED_EXTERN int CeedElemRestrictionGetNumElements(CeedElemRestriction r,
                                                  CeedInt *numelements);
CEED_EXTERN int CeedElemRestrictionApply(CeedElemRestriction r,
    CeedTransposeMode tmode, CeedTransposeMode lmode, CeedVector u,
    CeedVector ru, CeedRequest *request);
CEED_EXTERN int CeedElemRestrictionDestroy(CeedElemRestriction *r);

// The formalism here is that we have the structure
//   \int_\Omega v^T f_0(u, \nabla u, qdata) + (\nabla v)^T f_1(u, \nabla u, qdata)
// where gradients are with respect to the reference element.

/// Basis evaluation mode
///
/// Modes can be bitwise ORed when passing to most functions.
/// @ingroup CeedBasis
typedef enum {
  /// Perform no evaluation (either because there is no data or it is already at
  /// quadrature points)
  CEED_EVAL_NONE   = 0,
  /// Interpolate from nodes to quadrature points
  CEED_EVAL_INTERP = 1,
  /// Evaluate gradients at quadrature points from input in a nodal basis
  CEED_EVAL_GRAD   = 2,
  /// Evaluate divergence at quadrature points from input in a nodal basis
  CEED_EVAL_DIV    = 4,
  /// Evaluate curl at quadrature points from input in a nodal basis
  CEED_EVAL_CURL   = 8,
  /// Using no input, evaluate quadrature weights on the reference element
  CEED_EVAL_WEIGHT = 16,
} CeedEvalMode;

/// Type of quadrature; also used for location of nodes
/// @ingroup CeedBasis
typedef enum {
  /// Gauss-Legendre quadrature
  CEED_GAUSS = 0,
  /// Gauss-Legendre-Lobatto quadrature
  CEED_GAUSS_LOBATTO = 1,
} CeedQuadMode;

CEED_EXTERN int CeedBasisCreateTensorH1Lagrange(Ceed ceed, CeedInt dim,
    CeedInt ndof, CeedInt P, CeedInt Q, CeedQuadMode qmode, CeedBasis *basis);
CEED_EXTERN int CeedBasisCreateTensorH1(Ceed ceed, CeedInt dim, CeedInt ndof,
                                        CeedInt P1d, CeedInt Q1d, const CeedScalar *interp1d, const CeedScalar *grad1d,
                                        const CeedScalar *qref1d, const CeedScalar *qweight1d, CeedBasis *basis);
CEED_EXTERN int CeedBasisView(CeedBasis basis, FILE *stream);
CEED_EXTERN int CeedQRFactorization(CeedScalar *mat, CeedScalar *tau, CeedInt m, CeedInt n);
CEED_EXTERN int CeedBasisGetColocatedGrad(CeedBasis basis, CeedScalar *colograd1d);
CEED_EXTERN int CeedBasisApply(CeedBasis basis, CeedInt nelem, CeedTransposeMode tmode,
                               CeedEvalMode emode, const CeedScalar *u, CeedScalar *v);
CEED_EXTERN int CeedBasisGetNumNodes(CeedBasis basis, CeedInt *P);
CEED_EXTERN int CeedBasisGetNumQuadraturePoints(CeedBasis basis, CeedInt *Q);
CEED_EXTERN int CeedBasisDestroy(CeedBasis *basis);

CEED_EXTERN int CeedGaussQuadrature(CeedInt Q, CeedScalar *qref1d,
                                    CeedScalar *qweight1d);
CEED_EXTERN int CeedLobattoQuadrature(CeedInt Q, CeedScalar *qref1d,
                                      CeedScalar *qweight1d);

CEED_EXTERN int CeedQFunctionCreateInterior(Ceed ceed, CeedInt vlength,
    int (*f)(void *ctx, CeedInt nq, const CeedScalar *const *u,
             CeedScalar *const *v), const char *focca, CeedQFunction *qf);
CEED_EXTERN int CeedQFunctionAddInput(CeedQFunction qf, const char *fieldname,
                                      CeedInt ncomp, CeedEvalMode emode);
CEED_EXTERN int CeedQFunctionAddOutput(CeedQFunction qf, const char *fieldname,
                                       CeedInt ncomp, CeedEvalMode emode);
CEED_EXTERN int CeedQFunctionGetNumArgs(CeedQFunction qf, CeedInt *numinput,
                                        CeedInt *numoutput);
CEED_EXTERN int CeedQFunctionSetContext(CeedQFunction qf, void *ctx,
                                        size_t ctxsize);
CEED_EXTERN int CeedQFunctionApply(CeedQFunction qf, CeedInt Q,
                                   const CeedScalar *const *u,
                                   CeedScalar *const *v);
CEED_EXTERN int CeedQFunctionDestroy(CeedQFunction *qf);

CEED_EXTERN int CeedOperatorCreate(Ceed ceed, CeedQFunction qf,
                                   CeedQFunction dqf, CeedQFunction dqfT,
                                   CeedOperator *op);
CEED_EXTERN int CeedOperatorSetField(CeedOperator op, const char *fieldname,
                                     CeedElemRestriction r, CeedBasis b,
                                     CeedVector v);
CEED_EXTERN int CeedOperatorApply(CeedOperator op, CeedVector in,
                                  CeedVector out, CeedRequest *request);
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
