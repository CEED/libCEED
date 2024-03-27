/// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
/// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
///
/// SPDX-License-Identifier: BSD-2-Clause
///
/// This file is part of CEED:  http://github.com/ceed

/// @file
/// Public header for user and utility components of libCEED
#pragma once

/// @defgroup Ceed Ceed: core components
/// @defgroup CeedVector CeedVector: storing and manipulating vectors
/// @defgroup CeedElemRestriction CeedElemRestriction: restriction from local vectors to elements
/// @defgroup CeedBasis CeedBasis: fully discrete finite element-like objects
/// @defgroup CeedQFunction CeedQFunction: independent operations at quadrature points
/// @defgroup CeedOperator CeedOperator: composed FE-type operations on vectors
///
/// @page FunctionCategories libCEED: Types of Functions libCEED provides three different header files depending upon the type of functions a user
/// requires.
/// @section Utility Utility Functions
///    These functions are intended general utilities that may be useful to libCEED developers and users.
///    These functions can generally be found in `"ceed.h"`.
/// @section User User Functions
///    These functions are intended to be used by general users of libCEED and can generally be found in `"ceed.h"`.
/// @section Advanced Advanced Functions
///    These functions are intended to be used by advanced users of libCEED and can generally be found in `"ceed.h"`.
/// @section Backend Backend Developer Functions
///    These functions are intended to be used by backend developers of libCEED and can generally be found in `"ceed-backend.h"`.
/// @section Developer Library Developer Functions
///    These functions are intended to be used by library developers of libCEED and can generally be found in `"ceed-impl.h"`.

#if !defined(CEED_SKIP_VISIBILITY)
#define CEED_VISIBILITY(mode) __attribute__((visibility(#mode)))
#else
#define CEED_VISIBILITY(mode)
#endif

/*
  CEED_EXTERN is used in this header to denote all publicly visible symbols.

  No other file should declare publicly visible symbols, thus it should never be used outside `"ceed.h"`.
*/
#if defined(__clang_analyzer__)
#define CEED_EXTERN extern
#elif defined(__cplusplus)
#define CEED_EXTERN extern "C" CEED_VISIBILITY(default)
#else
#define CEED_EXTERN extern CEED_VISIBILITY(default)
#endif

#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>

/*
  Typedefs and macros used in public interfaces and user `CeedQFunction` source
*/
#include "types.h"  // IWYU pragma: export

/*
  This line prevents IWYU from suggesting "ceed.h"
*/
// IWYU pragma: private, include <ceed.h>

/// Library context created by CeedInit()
/// @ingroup CeedUser
typedef struct Ceed_private *Ceed;
/// Non-blocking Ceed interfaces return a CeedRequest.
/// To perform an operation immediately, pass @ref CEED_REQUEST_IMMEDIATE instead.
/// @ingroup CeedUser
typedef struct CeedRequest_private *CeedRequest;
/// Handle for vectors over the field @ref CeedScalar
/// @ingroup CeedVectorUser
typedef struct CeedVector_private *CeedVector;
/// Handle for object describing restriction to elements
/// @ingroup CeedElemRestrictionUser
typedef struct CeedElemRestriction_private *CeedElemRestriction;
/// Handle for object describing discrete finite element evaluations
/// @ingroup CeedBasisUser
typedef struct CeedBasis_private *CeedBasis;
/// Handle for object describing CeedQFunction fields
/// @ingroup CeedQFunctionBackend
typedef struct CeedQFunctionField_private *CeedQFunctionField;
/// Handle for object describing functions evaluated independently at quadrature points
/// @ingroup CeedQFunctionUser
typedef struct CeedQFunction_private *CeedQFunction;
/// Handle for object describing CeedOperator fields
/// @ingroup CeedOperatorBackend
typedef struct CeedOperatorField_private *CeedOperatorField;
/// Handle for object describing context data for CeedQFunctions
/// @ingroup CeedQFunctionUser
typedef struct CeedQFunctionContext_private *CeedQFunctionContext;
/// Handle for object describing registered fields for CeedQFunctionContext
/// @ingroup CeedQFunctionUser
typedef struct CeedContextFieldLabel_private *CeedContextFieldLabel;
/// Handle for object describing FE-type operators acting on vectors
///
/// Given an element restriction \f$E\f$, basis evaluator \f$B\f$, and quadrature function\f$f\f$, a `CeedOperator` expresses operations of the form \f$E^T B^T f(B E u)\f$ acting on the vector \f$u\f$.
/// @ingroup CeedOperatorUser
typedef struct CeedOperator_private *CeedOperator;

CEED_EXTERN int CeedRegistryGetList(size_t *n, char ***const resources, CeedInt **array);
CEED_EXTERN int CeedInit(const char *resource, Ceed *ceed);
CEED_EXTERN int CeedSetStream(Ceed ceed, void *handle);
CEED_EXTERN int CeedReferenceCopy(Ceed ceed, Ceed *ceed_copy);
CEED_EXTERN int CeedGetResource(Ceed ceed, const char **resource);
CEED_EXTERN int CeedIsDeterministic(Ceed ceed, bool *is_deterministic);
CEED_EXTERN int CeedAddJitSourceRoot(Ceed ceed, const char *jit_source_root);
CEED_EXTERN int CeedView(Ceed ceed, FILE *stream);
CEED_EXTERN int CeedDestroy(Ceed *ceed);
CEED_EXTERN int CeedErrorImpl(Ceed ceed, const char *filename, int lineno, const char *func, int ecode, const char *format, ...);

/// Raise an error on `Ceed` object
///
/// @param ceed `Ceed` library context or `NULL`
/// @param ecode Error code (int)
/// @param ... printf-style format string followed by arguments as needed
///
/// @ingroup Ceed
/// @sa CeedSetErrorHandler()
#define CeedError(ceed, ecode, ...) (CeedErrorImpl((ceed), __FILE__, __LINE__, __func__, (ecode), __VA_ARGS__), (ecode))

/// Ceed error handlers
typedef int (*CeedErrorHandler)(Ceed, const char *, int, const char *, int, const char *, va_list *);
CEED_EXTERN int CeedSetErrorHandler(Ceed ceed, CeedErrorHandler handler);
CEED_EXTERN int CeedGetErrorMessage(Ceed ceed, const char **err_msg);
CEED_EXTERN int CeedResetErrorMessage(Ceed ceed, const char **err_msg);
CEED_EXTERN int CeedErrorReturn(Ceed ceed, const char *filename, int line_no, const char *func, int err_code, const char *format, va_list *args);
CEED_EXTERN int CeedErrorStore(Ceed ceed, const char *filename, int line_no, const char *func, int err_code, const char *format, va_list *args);
CEED_EXTERN int CeedErrorAbort(Ceed ceed, const char *filename, int line_no, const char *func, int err_code, const char *format, va_list *args);
CEED_EXTERN int CeedErrorExit(Ceed ceed, const char *filename, int line_no, const char *func, int err_code, const char *format, va_list *args);

/// libCEED library version numbering
/// @ingroup Ceed
#define CEED_VERSION_MAJOR 0
#define CEED_VERSION_MINOR 12
#define CEED_VERSION_PATCH 0
#define CEED_VERSION_RELEASE false

/// Compile-time check that the the current library version is at least as recent as the specified version.
/// This macro is typically used in
/// @code
/// #if CEED_VERSION_GE(0, 8, 0)
///   code path that needs at least 0.8.0
/// #else
///   fallback code for older versions
/// #endif
/// @endcode
///
/// A non-release version always compares as positive infinity.
///
/// @param major   Major version
/// @param minor   Minor version
/// @param patch   Patch (subminor) version
///
/// @ingroup Ceed
/// @sa CeedGetVersion()
#define CEED_VERSION_GE(major, minor, patch) \
  (!CEED_VERSION_RELEASE ||                  \
   (CEED_VERSION_MAJOR > major ||            \
    (CEED_VERSION_MAJOR == major && (CEED_VERSION_MINOR > minor || (CEED_VERSION_MINOR == minor && CEED_VERSION_PATCH >= patch)))))

CEED_EXTERN int CeedGetVersion(int *major, int *minor, int *patch, bool *release);

CEED_EXTERN int CeedGetScalarType(CeedScalarType *scalar_type);

/// String names for enum pretty printing
CEED_EXTERN const char *const *CeedErrorTypes;
CEED_EXTERN const char *const  CeedMemTypes[];
CEED_EXTERN const char *const  CeedCopyModes[];
CEED_EXTERN const char *const  CeedTransposeModes[];
CEED_EXTERN const char *const  CeedEvalModes[];
CEED_EXTERN const char *const  CeedQuadModes[];
CEED_EXTERN const char *const  CeedElemTopologies[];
CEED_EXTERN const char *const  CeedContextFieldTypes[];

CEED_EXTERN int CeedGetPreferredMemType(Ceed ceed, CeedMemType *type);

CEED_EXTERN int  CeedVectorCreate(Ceed ceed, CeedSize len, CeedVector *vec);
CEED_EXTERN int  CeedVectorReferenceCopy(CeedVector vec, CeedVector *vec_copy);
CEED_EXTERN int  CeedVectorCopy(CeedVector vec, CeedVector vec_copy);
CEED_EXTERN int  CeedVectorSetArray(CeedVector vec, CeedMemType mem_type, CeedCopyMode copy_mode, CeedScalar *array);
CEED_EXTERN int  CeedVectorSetValue(CeedVector vec, CeedScalar value);
CEED_EXTERN int  CeedVectorSyncArray(CeedVector vec, CeedMemType mem_type);
CEED_EXTERN int  CeedVectorTakeArray(CeedVector vec, CeedMemType mem_type, CeedScalar **array);
CEED_EXTERN int  CeedVectorGetArray(CeedVector vec, CeedMemType mem_type, CeedScalar **array);
CEED_EXTERN int  CeedVectorGetArrayRead(CeedVector vec, CeedMemType mem_type, const CeedScalar **array);
CEED_EXTERN int  CeedVectorGetArrayWrite(CeedVector vec, CeedMemType mem_type, CeedScalar **array);
CEED_EXTERN int  CeedVectorRestoreArray(CeedVector vec, CeedScalar **array);
CEED_EXTERN int  CeedVectorRestoreArrayRead(CeedVector vec, const CeedScalar **array);
CEED_EXTERN int  CeedVectorNorm(CeedVector vec, CeedNormType type, CeedScalar *norm);
CEED_EXTERN int  CeedVectorScale(CeedVector x, CeedScalar alpha);
CEED_EXTERN int  CeedVectorAXPY(CeedVector y, CeedScalar alpha, CeedVector x);
CEED_EXTERN int  CeedVectorAXPBY(CeedVector y, CeedScalar alpha, CeedScalar beta, CeedVector x);
CEED_EXTERN int  CeedVectorPointwiseMult(CeedVector w, CeedVector x, CeedVector y);
CEED_EXTERN int  CeedVectorReciprocal(CeedVector vec);
CEED_EXTERN int  CeedVectorViewRange(CeedVector vec, CeedSize start, CeedSize stop, CeedInt step, const char *fp_fmt, FILE *stream);
CEED_EXTERN int  CeedVectorView(CeedVector vec, const char *fp_fmt, FILE *stream);
CEED_EXTERN int  CeedVectorGetCeed(CeedVector vec, Ceed *ceed);
CEED_EXTERN Ceed CeedVectorReturnCeed(CeedVector vec);
CEED_EXTERN int  CeedVectorGetLength(CeedVector vec, CeedSize *length);
CEED_EXTERN int  CeedVectorDestroy(CeedVector *vec);

CEED_EXTERN CeedRequest *const CEED_REQUEST_IMMEDIATE;
CEED_EXTERN CeedRequest *const CEED_REQUEST_ORDERED;
CEED_EXTERN int                CeedRequestWait(CeedRequest *req);

/// Argument for @ref CeedOperatorSetField() to use active input or output.
/// @ingroup CeedVector
CEED_EXTERN const CeedVector CEED_VECTOR_ACTIVE;

/// Argument for @ref CeedOperatorSetField() to use no `CeedVector`.
/// Only use this option with @ref CeedEvalMode @ref CEED_EVAL_WEIGHT.
/// @ingroup CeedVector
CEED_EXTERN const CeedVector CEED_VECTOR_NONE;

/// Argument for @ref CeedOperatorSetField() that no basis operation is needed to translate between the E-vector and the Q-vector.
/// Only use this option with @ref CeedEvalMode @ref CEED_EVAL_NONE.
/// @ingroup CeedBasis
CEED_EXTERN const CeedBasis CEED_BASIS_NONE;

/// Argument for @ref CeedOperatorSetField() to use no `CeedElemRestriction`.
/// Only use this option with @ref CeedEvalMode @ref CEED_EVAL_WEIGHT.
/// @ingroup CeedElemRestriction
CEED_EXTERN const CeedElemRestriction CEED_ELEMRESTRICTION_NONE;

/// Argument for @ref CeedOperatorCreate() that `CeedQFunction` is not created by user.
/// Only used for `CeedQFunction` `dqf` and `dqfT`.
/// If implemented, a backend may attempt to provide the action of these `CeedQFunction`.
/// @ingroup CeedQFunction
CEED_EXTERN const CeedQFunction CEED_QFUNCTION_NONE;

/// Argument for @ref CeedElemRestrictionCreateStrided() that L-vector is in the Ceed backend's preferred layout.
/// This argument should only be used with vectors created by a Ceed backend.
/// @ingroup CeedElemRestriction
CEED_EXTERN const CeedInt CEED_STRIDES_BACKEND[3];

CEED_EXTERN int  CeedElemRestrictionCreate(Ceed ceed, CeedInt num_elem, CeedInt elem_size, CeedInt num_comp, CeedInt comp_stride, CeedSize l_size,
                                           CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *offsets, CeedElemRestriction *rstr);
CEED_EXTERN int  CeedElemRestrictionCreateOriented(Ceed ceed, CeedInt num_elem, CeedInt elem_size, CeedInt num_comp, CeedInt comp_stride,
                                                   CeedSize l_size, CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *offsets,
                                                   const bool *orients, CeedElemRestriction *rstr);
CEED_EXTERN int  CeedElemRestrictionCreateCurlOriented(Ceed ceed, CeedInt num_elem, CeedInt elem_size, CeedInt num_comp, CeedInt comp_stride,
                                                       CeedSize l_size, CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *offsets,
                                                       const CeedInt8 *curl_orients, CeedElemRestriction *rstr);
CEED_EXTERN int  CeedElemRestrictionCreateStrided(Ceed ceed, CeedInt num_elem, CeedInt elem_size, CeedInt num_comp, CeedSize l_size,
                                                  const CeedInt strides[3], CeedElemRestriction *rstr);
CEED_EXTERN int  CeedElemRestrictionCreateAtPoints(Ceed ceed, CeedInt num_elem, CeedInt num_points, CeedInt num_comp, CeedSize l_size,
                                                   CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *offsets, CeedElemRestriction *rstr);
CEED_EXTERN int  CeedElemRestrictionCreateBlocked(Ceed ceed, CeedInt num_elem, CeedInt elem_size, CeedInt block_size, CeedInt num_comp,
                                                  CeedInt comp_stride, CeedSize l_size, CeedMemType mem_type, CeedCopyMode copy_mode,
                                                  const CeedInt *offsets, CeedElemRestriction *rstr);
CEED_EXTERN int  CeedElemRestrictionCreateBlockedOriented(Ceed ceed, CeedInt num_elem, CeedInt elem_size, CeedInt block_size, CeedInt num_comp,
                                                          CeedInt comp_stride, CeedSize l_size, CeedMemType mem_type, CeedCopyMode copy_mode,
                                                          const CeedInt *offsets, const bool *orients, CeedElemRestriction *rstr);
CEED_EXTERN int  CeedElemRestrictionCreateBlockedCurlOriented(Ceed ceed, CeedInt num_elem, CeedInt elem_size, CeedInt block_size, CeedInt num_comp,
                                                              CeedInt comp_stride, CeedSize l_size, CeedMemType mem_type, CeedCopyMode copy_mode,
                                                              const CeedInt *offsets, const CeedInt8 *curl_orients, CeedElemRestriction *rstr);
CEED_EXTERN int  CeedElemRestrictionCreateBlockedStrided(Ceed ceed, CeedInt num_elem, CeedInt elem_size, CeedInt block_size, CeedInt num_comp,
                                                         CeedSize l_size, const CeedInt strides[3], CeedElemRestriction *rstr);
CEED_EXTERN int  CeedElemRestrictionCreateUnsignedCopy(CeedElemRestriction rstr, CeedElemRestriction *rstr_unsigned);
CEED_EXTERN int  CeedElemRestrictionCreateUnorientedCopy(CeedElemRestriction rstr, CeedElemRestriction *rstr_unoriented);
CEED_EXTERN int  CeedElemRestrictionReferenceCopy(CeedElemRestriction rstr, CeedElemRestriction *rstr_copy);
CEED_EXTERN int  CeedElemRestrictionCreateVector(CeedElemRestriction rstr, CeedVector *lvec, CeedVector *evec);
CEED_EXTERN int  CeedElemRestrictionApply(CeedElemRestriction rstr, CeedTransposeMode t_mode, CeedVector u, CeedVector ru, CeedRequest *request);
CEED_EXTERN int  CeedElemRestrictionApplyAtPointsInElement(CeedElemRestriction rstr, CeedInt elem, CeedTransposeMode t_mode, CeedVector u,
                                                           CeedVector ru, CeedRequest *request);
CEED_EXTERN int  CeedElemRestrictionApplyBlock(CeedElemRestriction rstr, CeedInt block, CeedTransposeMode t_mode, CeedVector u, CeedVector ru,
                                               CeedRequest *request);
CEED_EXTERN int  CeedElemRestrictionGetCeed(CeedElemRestriction rstr, Ceed *ceed);
CEED_EXTERN Ceed CeedElemRestrictionReturnCeed(CeedElemRestriction rstr);
CEED_EXTERN int  CeedElemRestrictionGetCompStride(CeedElemRestriction rstr, CeedInt *comp_stride);
CEED_EXTERN int  CeedElemRestrictionGetNumElements(CeedElemRestriction rstr, CeedInt *num_elem);
CEED_EXTERN int  CeedElemRestrictionGetElementSize(CeedElemRestriction rstr, CeedInt *elem_size);
CEED_EXTERN int  CeedElemRestrictionGetNumPoints(CeedElemRestriction rstr, CeedInt *num_points);
CEED_EXTERN int  CeedElemRestrictionGetNumPointsInElement(CeedElemRestriction rstr, CeedInt elem, CeedInt *num_points);
CEED_EXTERN int  CeedElemRestrictionGetMaxPointsInElement(CeedElemRestriction rstr, CeedInt *max_points);
CEED_EXTERN int  CeedElemRestrictionGetLVectorSize(CeedElemRestriction rstr, CeedSize *l_size);
CEED_EXTERN int  CeedElemRestrictionGetEVectorSize(CeedElemRestriction rstr, CeedSize *e_size);
CEED_EXTERN int  CeedElemRestrictionGetNumComponents(CeedElemRestriction rstr, CeedInt *num_comp);
CEED_EXTERN int  CeedElemRestrictionGetNumBlocks(CeedElemRestriction rstr, CeedInt *num_block);
CEED_EXTERN int  CeedElemRestrictionGetBlockSize(CeedElemRestriction rstr, CeedInt *block_size);
CEED_EXTERN int  CeedElemRestrictionGetMultiplicity(CeedElemRestriction rstr, CeedVector mult);
CEED_EXTERN int  CeedElemRestrictionView(CeedElemRestriction rstr, FILE *stream);
CEED_EXTERN int  CeedElemRestrictionDestroy(CeedElemRestriction *rstr);

// The formalism here is that we have the structure
//  \int_\Omega v^T f_0(u, \nabla u, qdata) + (\nabla v)^T f_1(u, \nabla u, qdata)
// where gradients are with respect to the reference element.

CEED_EXTERN int  CeedBasisCreateTensorH1Lagrange(Ceed ceed, CeedInt dim, CeedInt num_comp, CeedInt P, CeedInt Q, CeedQuadMode quad_mode,
                                                 CeedBasis *basis);
CEED_EXTERN int  CeedBasisCreateTensorH1(Ceed ceed, CeedInt dim, CeedInt num_comp, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d,
                                         const CeedScalar *grad_1d, const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis *basis);
CEED_EXTERN int  CeedBasisCreateH1(Ceed ceed, CeedElemTopology topo, CeedInt num_comp, CeedInt num_nodes, CeedInt nqpts, const CeedScalar *interp,
                                   const CeedScalar *grad, const CeedScalar *q_ref, const CeedScalar *q_weights, CeedBasis *basis);
CEED_EXTERN int  CeedBasisCreateHdiv(Ceed ceed, CeedElemTopology topo, CeedInt num_comp, CeedInt num_nodes, CeedInt nqpts, const CeedScalar *interp,
                                     const CeedScalar *div, const CeedScalar *q_ref, const CeedScalar *q_weights, CeedBasis *basis);
CEED_EXTERN int  CeedBasisCreateHcurl(Ceed ceed, CeedElemTopology topo, CeedInt num_comp, CeedInt num_nodes, CeedInt nqpts, const CeedScalar *interp,
                                      const CeedScalar *curl, const CeedScalar *q_ref, const CeedScalar *q_weights, CeedBasis *basis);
CEED_EXTERN int  CeedBasisCreateProjection(CeedBasis basis_from, CeedBasis basis_to, CeedBasis *basis_project);
CEED_EXTERN int  CeedBasisReferenceCopy(CeedBasis basis, CeedBasis *basis_copy);
CEED_EXTERN int  CeedBasisView(CeedBasis basis, FILE *stream);
CEED_EXTERN int  CeedBasisApply(CeedBasis basis, CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u, CeedVector v);
CEED_EXTERN int  CeedBasisApplyAtPoints(CeedBasis basis, CeedInt num_points, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector x_ref,
                                        CeedVector u, CeedVector v);
CEED_EXTERN int  CeedBasisGetCeed(CeedBasis basis, Ceed *ceed);
CEED_EXTERN Ceed CeedBasisReturnCeed(CeedBasis basis);
CEED_EXTERN int  CeedBasisGetDimension(CeedBasis basis, CeedInt *dim);
CEED_EXTERN int  CeedBasisGetTopology(CeedBasis basis, CeedElemTopology *topo);
CEED_EXTERN int  CeedBasisGetNumComponents(CeedBasis basis, CeedInt *num_comp);
CEED_EXTERN int  CeedBasisGetNumNodes(CeedBasis basis, CeedInt *P);
CEED_EXTERN int  CeedBasisGetNumNodes1D(CeedBasis basis, CeedInt *P_1d);
CEED_EXTERN int  CeedBasisGetNumQuadraturePoints(CeedBasis basis, CeedInt *Q);
CEED_EXTERN int  CeedBasisGetNumQuadraturePoints1D(CeedBasis basis, CeedInt *Q_1d);
CEED_EXTERN int  CeedBasisGetQRef(CeedBasis basis, const CeedScalar **q_ref);
CEED_EXTERN int  CeedBasisGetQWeights(CeedBasis basis, const CeedScalar **q_weights);
CEED_EXTERN int  CeedBasisGetInterp(CeedBasis basis, const CeedScalar **interp);
CEED_EXTERN int  CeedBasisGetInterp1D(CeedBasis basis, const CeedScalar **interp_1d);
CEED_EXTERN int  CeedBasisGetGrad(CeedBasis basis, const CeedScalar **grad);
CEED_EXTERN int  CeedBasisGetGrad1D(CeedBasis basis, const CeedScalar **grad_1d);
CEED_EXTERN int  CeedBasisGetDiv(CeedBasis basis, const CeedScalar **div);
CEED_EXTERN int  CeedBasisGetCurl(CeedBasis basis, const CeedScalar **curl);
CEED_EXTERN int  CeedBasisDestroy(CeedBasis *basis);

CEED_EXTERN int CeedGaussQuadrature(CeedInt Q, CeedScalar *q_ref_1d, CeedScalar *q_weight_1d);
CEED_EXTERN int CeedLobattoQuadrature(CeedInt Q, CeedScalar *q_ref_1d, CeedScalar *q_weight_1d);

/** Handle for the user provided `CeedQFunction` callback function

 @param[in,out] ctx User-defined context set using @ref CeedQFunctionSetContext() or `NULL`
 @param[in] Q       Number of quadrature points at which to evaluate
 @param[in] in      Array of pointers to each input argument in the order provided by the user in @ref CeedQFunctionAddInput().
                      Each array has shape `[dim, num_comp, Q]` where `dim` is the geometric dimension for @ref CEED_EVAL_GRAD (`dim=1` for @ref CEED_EVAL_INTERP) and `num_comp` is the number of field components (`num_comp=1` for scalar fields).
                      This results in indexing the `i`th input at quadrature point `j` as `in[i][(d*num_comp + c)*Q + j]`.
 @param[out]   out  Array of pointers to each output array in the order provided using @ref CeedQFunctionAddOutput().
                      The shapes are as above for `in`.

 @return An error code: 0 - success, otherwise - failure

 @ingroup CeedQFunction
**/
typedef int (*CeedQFunctionUser)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out);

CEED_EXTERN int  CeedQFunctionCreateInterior(Ceed ceed, CeedInt vec_length, CeedQFunctionUser f, const char *source, CeedQFunction *qf);
CEED_EXTERN int  CeedQFunctionCreateInteriorByName(Ceed ceed, const char *name, CeedQFunction *qf);
CEED_EXTERN int  CeedQFunctionCreateIdentity(Ceed ceed, CeedInt size, CeedEvalMode in_mode, CeedEvalMode out_mode, CeedQFunction *qf);
CEED_EXTERN int  CeedQFunctionReferenceCopy(CeedQFunction qf, CeedQFunction *qf_copy);
CEED_EXTERN int  CeedQFunctionAddInput(CeedQFunction qf, const char *field_name, CeedInt size, CeedEvalMode eval_mode);
CEED_EXTERN int  CeedQFunctionAddOutput(CeedQFunction qf, const char *field_name, CeedInt size, CeedEvalMode eval_mode);
CEED_EXTERN int  CeedQFunctionGetFields(CeedQFunction qf, CeedInt *num_input_fields, CeedQFunctionField **input_fields, CeedInt *num_output_fields,
                                        CeedQFunctionField **output_fields);
CEED_EXTERN int  CeedQFunctionSetContext(CeedQFunction qf, CeedQFunctionContext ctx);
CEED_EXTERN int  CeedQFunctionSetContextWritable(CeedQFunction qf, bool is_writable);
CEED_EXTERN int  CeedQFunctionSetUserFlopsEstimate(CeedQFunction qf, CeedSize flops);
CEED_EXTERN int  CeedQFunctionView(CeedQFunction qf, FILE *stream);
CEED_EXTERN int  CeedQFunctionGetCeed(CeedQFunction qf, Ceed *ceed);
CEED_EXTERN Ceed CeedQFunctionReturnCeed(CeedQFunction qf);
CEED_EXTERN int  CeedQFunctionApply(CeedQFunction qf, CeedInt Q, CeedVector *u, CeedVector *v);
CEED_EXTERN int  CeedQFunctionDestroy(CeedQFunction *qf);

CEED_EXTERN int CeedQFunctionFieldGetName(CeedQFunctionField qf_field, const char **field_name);
CEED_EXTERN int CeedQFunctionFieldGetSize(CeedQFunctionField qf_field, CeedInt *size);
CEED_EXTERN int CeedQFunctionFieldGetEvalMode(CeedQFunctionField qf_field, CeedEvalMode *eval_mode);
CEED_EXTERN int CeedQFunctionFieldGetData(CeedQFunctionField qf_field, const char **field_name, CeedInt *size, CeedEvalMode *eval_mode);

/** Handle for the user provided @ref CeedQFunctionContextDestroy() callback function

 @param[in,out] data  User `CeedQFunctionContext` data

 @return An error code: 0 - success, otherwise - failure

 @ingroup CeedQFunction
**/
typedef int (*CeedQFunctionContextDataDestroyUser)(void *data);

CEED_EXTERN int CeedQFunctionContextCreate(Ceed ceed, CeedQFunctionContext *ctx);
CEED_EXTERN int CeedQFunctionContextReferenceCopy(CeedQFunctionContext ctx, CeedQFunctionContext *ctx_copy);
CEED_EXTERN int CeedQFunctionContextSetData(CeedQFunctionContext ctx, CeedMemType mem_type, CeedCopyMode copy_mode, size_t size, void *data);
CEED_EXTERN int CeedQFunctionContextTakeData(CeedQFunctionContext ctx, CeedMemType mem_type, void *data);
CEED_EXTERN int CeedQFunctionContextGetData(CeedQFunctionContext ctx, CeedMemType mem_type, void *data);
CEED_EXTERN int CeedQFunctionContextGetDataRead(CeedQFunctionContext ctx, CeedMemType mem_type, void *data);
CEED_EXTERN int CeedQFunctionContextRestoreData(CeedQFunctionContext ctx, void *data);
CEED_EXTERN int CeedQFunctionContextRestoreDataRead(CeedQFunctionContext ctx, void *data);
CEED_EXTERN int CeedQFunctionContextRegisterDouble(CeedQFunctionContext ctx, const char *field_name, size_t field_offset, size_t num_values,
                                                   const char *field_description);
CEED_EXTERN int CeedQFunctionContextRegisterInt32(CeedQFunctionContext ctx, const char *field_name, size_t field_offset, size_t num_values,
                                                  const char *field_description);
CEED_EXTERN int CeedQFunctionContextRegisterBoolean(CeedQFunctionContext ctx, const char *field_name, size_t field_offset, size_t num_values,
                                                    const char *field_description);
CEED_EXTERN int CeedQFunctionContextGetAllFieldLabels(CeedQFunctionContext ctx, const CeedContextFieldLabel **field_labels, CeedInt *num_fields);
CEED_EXTERN int CeedContextFieldLabelGetDescription(CeedContextFieldLabel label, const char **field_name, size_t *field_offset, size_t *num_values,
                                                    const char **field_description, CeedContextFieldType *field_type);
CEED_EXTERN int CeedQFunctionContextGetContextSize(CeedQFunctionContext ctx, size_t *ctx_size);
CEED_EXTERN int CeedQFunctionContextView(CeedQFunctionContext ctx, FILE *stream);
CEED_EXTERN int CeedQFunctionContextSetDataDestroy(CeedQFunctionContext ctx, CeedMemType f_mem_type, CeedQFunctionContextDataDestroyUser f);
CEED_EXTERN int CeedQFunctionContextDestroy(CeedQFunctionContext *ctx);

CEED_EXTERN int CeedOperatorCreate(Ceed ceed, CeedQFunction qf, CeedQFunction dqf, CeedQFunction dqfT, CeedOperator *op);
CEED_EXTERN int CeedOperatorCreateAtPoints(Ceed ceed, CeedQFunction qf, CeedQFunction dqf, CeedQFunction dqfT, CeedOperator *op);
CEED_EXTERN int CeedCompositeOperatorCreate(Ceed ceed, CeedOperator *op);
CEED_EXTERN int CeedOperatorReferenceCopy(CeedOperator op, CeedOperator *op_copy);
CEED_EXTERN int CeedOperatorSetField(CeedOperator op, const char *field_name, CeedElemRestriction rstr, CeedBasis basis, CeedVector vec);
CEED_EXTERN int CeedOperatorGetFields(CeedOperator op, CeedInt *num_input_fields, CeedOperatorField **input_fields, CeedInt *num_output_fields,
                                      CeedOperatorField **output_fields);

CEED_EXTERN int  CeedOperatorAtPointsSetPoints(CeedOperator op, CeedElemRestriction rstr_points, CeedVector point_coords);
CEED_EXTERN int  CeedOperatorAtPointsGetPoints(CeedOperator op, CeedElemRestriction *rstr_points, CeedVector *point_coords);
CEED_EXTERN int  CeedOperatorIsAtPoints(CeedOperator op, bool *is_at_points);
CEED_EXTERN int  CeedCompositeOperatorAddSub(CeedOperator composite_op, CeedOperator sub_op);
CEED_EXTERN int  CeedCompositeOperatorGetNumSub(CeedOperator op, CeedInt *num_suboperators);
CEED_EXTERN int  CeedCompositeOperatorGetSubList(CeedOperator op, CeedOperator **sub_operators);
CEED_EXTERN int  CeedOperatorCheckReady(CeedOperator op);
CEED_EXTERN int  CeedOperatorGetActiveVectorLengths(CeedOperator op, CeedSize *input_size, CeedSize *output_size);
CEED_EXTERN int  CeedOperatorSetQFunctionAssemblyReuse(CeedOperator op, bool reuse_assembly_data);
CEED_EXTERN int  CeedOperatorSetQFunctionAssemblyDataUpdateNeeded(CeedOperator op, bool needs_data_update);
CEED_EXTERN int  CeedOperatorLinearAssembleQFunction(CeedOperator op, CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request);
CEED_EXTERN int  CeedOperatorLinearAssembleQFunctionBuildOrUpdate(CeedOperator op, CeedVector *assembled, CeedElemRestriction *rstr,
                                                                  CeedRequest *request);
CEED_EXTERN int  CeedOperatorLinearAssembleDiagonal(CeedOperator op, CeedVector assembled, CeedRequest *request);
CEED_EXTERN int  CeedOperatorLinearAssembleAddDiagonal(CeedOperator op, CeedVector assembled, CeedRequest *request);
CEED_EXTERN int  CeedOperatorLinearAssemblePointBlockDiagonal(CeedOperator op, CeedVector assembled, CeedRequest *request);
CEED_EXTERN int  CeedOperatorLinearAssembleAddPointBlockDiagonal(CeedOperator op, CeedVector assembled, CeedRequest *request);
CEED_EXTERN int  CeedOperatorLinearAssemblePointBlockDiagonalSymbolic(CeedOperator op, CeedSize *num_entries, CeedInt **rows, CeedInt **cols);
CEED_EXTERN int  CeedOperatorLinearAssembleSymbolic(CeedOperator op, CeedSize *num_entries, CeedInt **rows, CeedInt **cols);
CEED_EXTERN int  CeedOperatorLinearAssemble(CeedOperator op, CeedVector values);
CEED_EXTERN int  CeedCompositeOperatorGetMultiplicity(CeedOperator op, CeedInt num_skip_indices, CeedInt *skip_indices, CeedVector mult);
CEED_EXTERN int  CeedOperatorMultigridLevelCreate(CeedOperator op_fine, CeedVector p_mult_fine, CeedElemRestriction rstr_coarse,
                                                  CeedBasis basis_coarse, CeedOperator *op_coarse, CeedOperator *op_prolong,
                                                  CeedOperator *op_restrict);
CEED_EXTERN int  CeedOperatorMultigridLevelCreateTensorH1(CeedOperator op_fine, CeedVector p_mult_fine, CeedElemRestriction rstr_coarse,
                                                          CeedBasis basis_coarse, const CeedScalar *interp_c_to_f, CeedOperator *op_coarse,
                                                          CeedOperator *op_prolong, CeedOperator *op_restrict);
CEED_EXTERN int  CeedOperatorMultigridLevelCreateH1(CeedOperator op_fine, CeedVector p_mult_fine, CeedElemRestriction rstr_coarse,
                                                    CeedBasis basis_coarse, const CeedScalar *interp_c_to_f, CeedOperator *op_coarse,
                                                    CeedOperator *op_prolong, CeedOperator *op_restrict);
CEED_EXTERN int  CeedOperatorCreateFDMElementInverse(CeedOperator op, CeedOperator *fdm_inv, CeedRequest *request);
CEED_EXTERN int  CeedOperatorSetName(CeedOperator op, const char *name);
CEED_EXTERN int  CeedOperatorView(CeedOperator op, FILE *stream);
CEED_EXTERN int  CeedOperatorGetCeed(CeedOperator op, Ceed *ceed);
CEED_EXTERN Ceed CeedOperatorReturnCeed(CeedOperator op);
CEED_EXTERN int  CeedOperatorGetNumElements(CeedOperator op, CeedInt *num_elem);
CEED_EXTERN int  CeedOperatorGetNumQuadraturePoints(CeedOperator op, CeedInt *num_qpts);
CEED_EXTERN int  CeedOperatorGetFlopsEstimate(CeedOperator op, CeedSize *flops);
CEED_EXTERN int  CeedOperatorGetContext(CeedOperator op, CeedQFunctionContext *ctx);
CEED_EXTERN int  CeedOperatorGetContextFieldLabel(CeedOperator op, const char *field_name, CeedContextFieldLabel *field_label);
CEED_EXTERN int  CeedOperatorSetContextDouble(CeedOperator op, CeedContextFieldLabel field_label, double *values);
CEED_EXTERN int  CeedOperatorGetContextDoubleRead(CeedOperator op, CeedContextFieldLabel field_label, size_t *num_values, const double **values);
CEED_EXTERN int  CeedOperatorRestoreContextDoubleRead(CeedOperator op, CeedContextFieldLabel field_label, const double **values);
CEED_EXTERN int  CeedOperatorSetContextInt32(CeedOperator op, CeedContextFieldLabel field_label, int32_t *values);
CEED_EXTERN int  CeedOperatorGetContextInt32Read(CeedOperator op, CeedContextFieldLabel field_label, size_t *num_values, const int32_t **values);
CEED_EXTERN int  CeedOperatorRestoreContextInt32Read(CeedOperator op, CeedContextFieldLabel field_label, const int32_t **values);
CEED_EXTERN int  CeedOperatorSetContextBoolean(CeedOperator op, CeedContextFieldLabel field_label, bool *values);
CEED_EXTERN int  CeedOperatorGetContextBooleanRead(CeedOperator op, CeedContextFieldLabel field_label, size_t *num_values, const bool **values);
CEED_EXTERN int  CeedOperatorRestoreContextBooleanRead(CeedOperator op, CeedContextFieldLabel field_label, const bool **values);
CEED_EXTERN int  CeedOperatorApply(CeedOperator op, CeedVector in, CeedVector out, CeedRequest *request);
CEED_EXTERN int  CeedOperatorApplyAdd(CeedOperator op, CeedVector in, CeedVector out, CeedRequest *request);
CEED_EXTERN int  CeedOperatorDestroy(CeedOperator *op);

CEED_EXTERN int CeedOperatorGetFieldByName(CeedOperator op, const char *field_name, CeedOperatorField *op_field);
CEED_EXTERN int CeedOperatorFieldGetName(CeedOperatorField op_field, const char **field_name);
CEED_EXTERN int CeedOperatorFieldGetElemRestriction(CeedOperatorField op_field, CeedElemRestriction *rstr);
CEED_EXTERN int CeedOperatorFieldGetBasis(CeedOperatorField op_field, CeedBasis *basis);
CEED_EXTERN int CeedOperatorFieldGetVector(CeedOperatorField op_field, CeedVector *vec);
CEED_EXTERN int CeedOperatorFieldGetData(CeedOperatorField op_field, const char **field_name, CeedElemRestriction *rstr, CeedBasis *basis,
                                         CeedVector *vec);

/**
  @brief Return integer power

  @param[in] base  The base to exponentiate
  @param[in] power The power to raise the base to

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

  @param[in] a The first integer to compare
  @param[in] b The second integer to compare

  @return The minimum of the two integers

  @ref Utility
**/
static inline CeedInt CeedIntMin(CeedInt a, CeedInt b) { return a < b ? a : b; }

/**
  @brief Return maximum of two integers

  @param[in] a The first integer to compare
  @param[in] b The second integer to compare

  @return The maximum of the two integers

  @ref Utility
**/
static inline CeedInt CeedIntMax(CeedInt a, CeedInt b) { return a > b ? a : b; }

// Used to ensure initialization before CeedInit()
CEED_EXTERN int CeedRegisterAll(void);
// Used to ensure initialization before CeedQFunctionCreate*()
CEED_EXTERN int CeedQFunctionRegisterAll(void);
