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
/// Public header for user and utility components of libCEED
#ifndef _ceed_h
#define _ceed_h

/// @defgroup Ceed Ceed: core components
/// @defgroup CeedVector CeedVector: storing and manipulating vectors
/// @defgroup CeedElemRestriction CeedElemRestriction: restriction from local vectors to elements
/// @defgroup CeedBasis CeedBasis: fully discrete finite element-like objects
/// @defgroup CeedQFunction CeedQFunction: independent operations at quadrature points
/// @defgroup CeedOperator CeedOperator: composed FE-type operations on vectors
///
/// @page FunctionCategories libCEED: Types of Functions
///   libCEED provides three different header files depending upon the type of
///   functions a user requires.
/// @section Utility Utility Functions
///   These functions are intended general utilities that may be useful to
///   libCEED developers and users. These functions can generally be found in "ceed.h".
/// @section Basic User Functions
///   These functions are intended to be used by general users of the libCEED
///   interface. These functions can generally be found in "ceed.h".
/// @section Advanced Backend Developer Functions
///   These functions are intended to be used by backend developers of the
///   libCEED interface. These functions can generally be found in "ceed-backend.h".
/// @section Developer Frontend Developer Functions
///   These functions are intended to be used by frontend developers of the
///   libCEED interface. These functions can generally be found in "ceed-impl.h".

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

#ifndef CEED_QFUNCTION
#define CEED_QFUNCTION(name) \
  static const char name ## _loc[] = __FILE__ ":" #name;        \
  static int name
#endif

#ifndef CeedPragmaSIMD
#  if defined(__GNUC__) && __GNUC__ >= 5
#    define CeedPragmaSIMD _Pragma("GCC ivdep")
#  elif defined(_OPENMP) && _OPENMP >= 201307 // OpenMP-4.0 (July, 2013)
#    define CeedPragmaSIMD _Pragma("omp simd")
#  else
#    define CeedPragmaSIMD
#  endif
#endif

#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdbool.h>

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
/// @ingroup CeedOperator
typedef struct CeedOperator_private *CeedOperator;

CEED_EXTERN int CeedInit(const char *resource, Ceed *ceed);
CEED_EXTERN int CeedGetResource(Ceed ceed, const char **resource);
CEED_EXTERN int CeedDestroy(Ceed *ceed);

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
#if defined(__clang__)
// Use nonstandard ternary to convince the compiler/clang-tidy that this
// function never returns zero.
#  define CeedError(ceed, ecode, ...)                                     \
  (CeedErrorImpl((ceed), __FILE__, __LINE__, __func__, (ecode), __VA_ARGS__) ?: (ecode))
#else
#  define CeedError(ceed, ecode, ...)                                     \
  CeedErrorImpl((ceed), __FILE__, __LINE__, __func__, (ecode), __VA_ARGS__) ?: (ecode)
#endif
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

CEED_EXTERN const char *const CeedMemTypes[];

CEED_EXTERN int CeedGetPreferredMemType(Ceed ceed, CeedMemType *type);

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

CEED_EXTERN const char *const CeedCopyModes[];

CEED_EXTERN int CeedVectorCreate(Ceed ceed, CeedInt len, CeedVector *vec);
CEED_EXTERN int CeedVectorSetArray(CeedVector vec, CeedMemType mtype,
                                   CeedCopyMode cmode, CeedScalar *array);
CEED_EXTERN int CeedVectorSetValue(CeedVector vec, CeedScalar value);
CEED_EXTERN int CeedVectorSyncArray(CeedVector vec, CeedMemType mtype);
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

/// Argument for CeedOperatorSetField that vector is collocated with
/// quadrature points, used with QFunction eval mode CEED_EVAL_NONE
/// or CEED_EVAL_INTERP only, not with CEED_EVAL_GRAD, CEED_EVAL_DIV,
/// or CEED_EVAL_CURL
/// @ingroup CeedBasis
CEED_EXTERN CeedBasis CEED_BASIS_COLLOCATED;

/// Argument for CeedOperatorSetField to use active input or output
/// @ingroup CeedVector
CEED_EXTERN CeedVector CEED_VECTOR_ACTIVE;

/// Argument for CeedOperatorSetField to use no vector, used with
/// qfunction input with eval mode CEED_EVAL_WEIGHTS
/// @ingroup CeedVector
CEED_EXTERN CeedVector CEED_VECTOR_NONE;

/// Argument for CeedOperatorCreate that QFunction is not created by user.
/// Only used for QFunctions dqf and dqfT. If implemented, a backend may
/// attempt to provide the action of these QFunctions.
/// @ingroup CeedQFunction
CEED_EXTERN CeedQFunction CEED_QFUNCTION_NONE;

/// Denotes whether a linear transformation or its transpose should be applied
/// @ingroup CeedBasis
typedef enum {
  /// Apply the linear transformation
  CEED_NOTRANSPOSE,
  /// Apply the transpose
  CEED_TRANSPOSE
} CeedTransposeMode;

CEED_EXTERN const char *const CeedTransposeModes[];

CEED_EXTERN int CeedElemRestrictionCreate(Ceed ceed, CeedInt nelem,
    CeedInt elemsize, CeedInt nnodes, CeedInt ncomp, CeedMemType mtype,
    CeedCopyMode cmode,
    const CeedInt *indices, CeedElemRestriction *rstr);
CEED_EXTERN int CeedElemRestrictionCreateIdentity(Ceed ceed, CeedInt nelem,
    CeedInt elemsize, CeedInt nnodes, CeedInt ncomp, CeedElemRestriction *rstr);
CEED_EXTERN int CeedElemRestrictionCreateBlocked(Ceed ceed, CeedInt nelem,
    CeedInt elemsize, CeedInt blksize, CeedInt nnodes, CeedInt ncomp,
    CeedMemType mtype,
    CeedCopyMode cmode, const CeedInt *indices, CeedElemRestriction *rstr);
CEED_EXTERN int CeedElemRestrictionCreateVector(CeedElemRestriction rstr,
    CeedVector *lvec, CeedVector *evec);
CEED_EXTERN int CeedElemRestrictionApply(CeedElemRestriction rstr,
    CeedTransposeMode tmode, CeedTransposeMode lmode, CeedVector u,
    CeedVector ru, CeedRequest *request);
CEED_EXTERN int CeedElemRestrictionApplyBlock(CeedElemRestriction rstr,
    CeedInt block, CeedTransposeMode tmode, CeedTransposeMode lmode,
    CeedVector u, CeedVector ru, CeedRequest *request);
CEED_EXTERN int CeedElemRestrictionGetMultiplicity(CeedElemRestriction rstr,
    CeedVector mult);
CEED_EXTERN int CeedElemRestrictionView(CeedElemRestriction rstr, FILE *stream);
CEED_EXTERN int CeedElemRestrictionDestroy(CeedElemRestriction *rstr);

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

CEED_EXTERN const char *const CeedEvalModes[];

/// Type of quadrature; also used for location of nodes
/// @ingroup CeedBasis
typedef enum {
  /// Gauss-Legendre quadrature
  CEED_GAUSS = 0,
  /// Gauss-Legendre-Lobatto quadrature
  CEED_GAUSS_LOBATTO = 1,
} CeedQuadMode;

CEED_EXTERN const char *const CeedQuadModes[];

/// Type of basis shape to create non-tensor H1 element basis
///
/// Dimension can be extracted with bitwise AND
/// (CeedElemTopology & 2**(dim + 2)) == TRUE
/// @ingroup CeedBasis
typedef enum {
  /// Line
  CEED_LINE = 1 << 16 | 0,
  /// Triangle - 2D shape
  CEED_TRIANGLE = 2 << 16 | 1,
  /// Quadralateral - 2D shape
  CEED_QUAD = 2 << 16 | 2,
  /// Tetrahedron - 3D shape
  CEED_TET = 3 << 16 | 3,
  /// Pyramid - 3D shape
  CEED_PYRAMID = 3 << 16 | 4,
  /// Prism - 3D shape
  CEED_PRISM = 3 << 16 | 5,
  /// Hexehedron - 3D shape
  CEED_HEX = 3 << 16 | 6,
} CeedElemTopology;

CEED_EXTERN const char *const CeedElemTopologies[];

CEED_EXTERN int CeedBasisCreateTensorH1Lagrange(Ceed ceed, CeedInt dim,
    CeedInt ncomp, CeedInt P, CeedInt Q, CeedQuadMode qmode, CeedBasis *basis);
CEED_EXTERN int CeedBasisCreateTensorH1(Ceed ceed, CeedInt dim, CeedInt ncomp,
                                        CeedInt P1d, CeedInt Q1d,
                                        const CeedScalar *interp1d,
                                        const CeedScalar *grad1d,
                                        const CeedScalar *qref1d,
                                        const CeedScalar *qweight1d,
                                        CeedBasis *basis);
CEED_EXTERN int CeedBasisCreateH1(Ceed ceed, CeedElemTopology topo,
                                  CeedInt ncomp,
                                  CeedInt nnodes, CeedInt nqpts,
                                  const CeedScalar *interp,
                                  const CeedScalar *grad,
                                  const CeedScalar *qref,
                                  const CeedScalar *qweight, CeedBasis *basis);
CEED_EXTERN int CeedBasisView(CeedBasis basis, FILE *stream);
CEED_EXTERN int CeedBasisGetNumNodes(CeedBasis basis, CeedInt *P);
CEED_EXTERN int CeedBasisGetNumQuadraturePoints(CeedBasis basis, CeedInt *Q);
CEED_EXTERN int CeedBasisApply(CeedBasis basis, CeedInt nelem,
                               CeedTransposeMode tmode,
                               CeedEvalMode emode, CeedVector u, CeedVector v);
CEED_EXTERN int CeedBasisDestroy(CeedBasis *basis);

CEED_EXTERN int CeedGaussQuadrature(CeedInt Q, CeedScalar *qref1d,
                                    CeedScalar *qweight1d);
CEED_EXTERN int CeedLobattoQuadrature(CeedInt Q, CeedScalar *qref1d,
                                      CeedScalar *qweight1d);
CEED_EXTERN int CeedQRFactorization(Ceed ceed, CeedScalar *mat, CeedScalar *tau,
                                    CeedInt m, CeedInt n);
CEED_EXTERN int CeedSymmetricSchurDecomposition(Ceed ceed, CeedScalar *mat,
    CeedScalar *lambda, CeedInt n);
CEED_EXTERN int CeedSimultaneousDiagonalization(Ceed ceed, CeedScalar *matA,
    CeedScalar *matB, CeedScalar *x, CeedScalar *lambda, CeedInt n);

/// Handle for the object describing the user CeedQFunction
///
/// @param ctx - user-defined context set using CeedQFunctionSetContext() or NULL
///
/// @param Q - number of quadrature points at which to evaluate
///
/// @param in - array of pointers to each input argument in the order provided
///             by the user in CeedQFunctionAddInput().  Each array has shape
///             `[dim, ncomp, Q]` where `dim` is the geometric dimension for
///             \ref CEED_EVAL_GRAD (`dim=1` for \ref CEED_EVAL_INTERP) and
///             `ncomp` is the number of field components (`ncomp=1` for
///             scalar fields).  This results in indexing the `i`th input at
///             quadrature point `j` as `in[i][(d*ncomp + c)*Q + j]`.
///
/// @param out - array of pointers to each output array in the order provided
///              using CeedQFunctionAddOutput().  The shapes are as above for
///              \a in.
///
/// @return 0 on success, nonzero for failure.
///
/// @ingroup CeedQFunction
typedef int (*CeedQFunctionUser)(void *ctx, const CeedInt Q,
                                 const CeedScalar *const *in,
                                 CeedScalar *const *out);

CEED_EXTERN int CeedQFunctionCreateInterior(Ceed ceed, CeedInt vlength,
    CeedQFunctionUser f, const char *source, CeedQFunction *qf);
CEED_EXTERN int CeedQFunctionCreateInteriorByName(Ceed ceed, const char *name,
    CeedQFunction *qf);
CEED_EXTERN int CeedQFunctionCreateIdentity(Ceed ceed, CeedInt size,
    CeedEvalMode inmode, CeedEvalMode outmode, CeedQFunction *qf);
CEED_EXTERN int CeedQFunctionAddInput(CeedQFunction qf, const char *fieldname,
                                      CeedInt size, CeedEvalMode emode);
CEED_EXTERN int CeedQFunctionAddOutput(CeedQFunction qf, const char *fieldname,
                                       CeedInt size, CeedEvalMode emode);
CEED_EXTERN int CeedQFunctionSetContext(CeedQFunction qf, void *ctx,
                                        size_t ctxsize);
CEED_EXTERN int CeedQFunctionView(CeedQFunction qf, FILE *stream);
CEED_EXTERN int CeedQFunctionApply(CeedQFunction qf, CeedInt Q,
                                   CeedVector *u, CeedVector *v);
CEED_EXTERN int CeedQFunctionDestroy(CeedQFunction *qf);

CEED_EXTERN int CeedOperatorCreate(Ceed ceed, CeedQFunction qf,
                                   CeedQFunction dqf, CeedQFunction dqfT,
                                   CeedOperator *op);
CEED_EXTERN int CeedCompositeOperatorCreate(Ceed ceed, CeedOperator *op);
CEED_EXTERN int CeedOperatorSetField(CeedOperator op, const char *fieldname,
                                     CeedElemRestriction r,
                                     CeedTransposeMode lmode, CeedBasis b,
                                     CeedVector v);
CEED_EXTERN int CeedCompositeOperatorAddSub(CeedOperator compositeop,
    CeedOperator subop);
CEED_EXTERN int CeedOperatorAssembleLinearQFunction(CeedOperator op,
    CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request);
CEED_EXTERN int CeedOperatorAssembleLinearDiagonal(CeedOperator op,
    CeedVector *assembled, CeedRequest *request);
CEED_EXTERN int CeedOperatorView(CeedOperator op, FILE *stream);
CEED_EXTERN int CeedOperatorApply(CeedOperator op, CeedVector in,
                                  CeedVector out, CeedRequest *request);
CEED_EXTERN int CeedOperatorDestroy(CeedOperator *op);

/**
  @brief Return integer power

  @param[in] base   The base to exponentiate
  @param[in] power  The power to raise the base to

  @return base^power

  @ref Utility
**/
static inline CeedInt CeedIntPow(CeedInt base, CeedInt power) {
  CeedInt result = 1;
  while (power) {
    if (power & 1) result *= base;
    power >>= 1;
    base *= base;
  }
  return result;
}

/**
  @brief Return minimum of two integers

  @param[in] a  The first integer to compare
  @param[in] b  The second integer to compare

  @return The minimum of the two integers

  @ref Utility
**/
static inline CeedInt CeedIntMin(CeedInt a, CeedInt b) { return a < b ? a : b; }

#endif
