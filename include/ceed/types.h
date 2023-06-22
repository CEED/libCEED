/// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
/// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
///
/// SPDX-License-Identifier: BSD-2-Clause
///
/// This file is part of CEED:  http://github.com/ceed

/// @file
/// Public header for types and macros used in user QFunction source code
#ifndef _ceed_qfunction_defs_h
#define _ceed_qfunction_defs_h

#include <stddef.h>
#include <stdint.h>

/**
  @ingroup CeedQFunction
  This macro defines compiler attributes to the CEED_QFUNCTION to force inlining for called functions.
    The `inline` declaration does not necessarily enforce a compiler to inline a function.
    This can be detrimental to performance, so here we force inlining to occur unless inlining has been forced off (like during debugging).
**/
#ifndef CEED_QFUNCTION_ATTR
#ifndef __NO_INLINE__
#if defined(__GNUC__) || defined(__clang__)
#define CEED_QFUNCTION_ATTR __attribute__((flatten))
#elif defined(__INTEL_COMPILER)
#define CEED_QFUNCTION_ATTR _Pragma("forceinline")
#else
#define CEED_QFUNCTION_ATTR
#endif
#else
#define CEED_QFUNCTION_ATTR
#endif
#if defined(__GNUC__) || defined(__clang__)
#define CEED_QFUNCTION_HELPER_ATTR CEED_QFUNCTION_ATTR __attribute__((always_inline))
#else
#define CEED_QFUNCTION_HELPER_ATTR CEED_QFUNCTION_ATTR
#endif
#endif

/**
  @ingroup CeedQFunction
  This macro populates the correct function annotations for User QFunction source for code generation backends or populates default values for CPU
backends. It also creates a variable `name_loc` populated with the correct source path for creating the respective User QFunction.
**/
#ifndef CEED_QFUNCTION
#define CEED_QFUNCTION(name)                                        \
  static const char              name##_loc[] = __FILE__ ":" #name; \
  CEED_QFUNCTION_ATTR static int name
#endif

/**
  @ingroup CeedQFunction
  This macro populates the correct function annotations for User QFunction helper function source for code generation backends or populates default
values for CPU backends.
**/
#ifndef CEED_QFUNCTION_HELPER
#define CEED_QFUNCTION_HELPER CEED_QFUNCTION_HELPER_ATTR static inline
#endif

/**
  @ingroup CeedQFunction
  Using VLA syntax to reshape User QFunction inputs and outputs can make user code more readable.
    VLA is a C99 feature that is not supported by the C++ dialect used by CUDA.
    This macro allows users to use the VLA syntax with the CUDA backends.
**/
#ifndef CEED_Q_VLA
#define CEED_Q_VLA Q
#endif

/**
  @ingroup Ceed
  This macro provides the appropriate SIMD Pragma for the compilation environment.
    Code generation backends may redefine this macro, as needed.
**/
#ifndef CeedPragmaSIMD
#if defined(__INTEL_COMPILER)
#define CeedPragmaSIMD _Pragma("vector")
/// Cannot use Intel pragma ivdep because it miscompiles unpacking symmetric tensors, as in Poisson2DApply, where the SIMD loop body contains
/// temporaries such as the following.
///
///     const CeedScalar dXdxdXdxT[2][2] = {{qd[i+0*Q], qd[i+2*Q]},
///                                         {qd[i+2*Q], qd[i+1*Q]}};
///     for (int j=0; j<2; j++)
///        vg[i+j*Q] = (du[0] * dXdxdXdxT[0][j] + du[1] * dXdxdXdxT[1][j]);
///
/// Miscompilation with pragma ivdep observed with icc (ICC) 19.0.5.281 20190815 at -O2 and above.
#elif defined(__GNUC__) && __GNUC__ >= 5
#define CeedPragmaSIMD _Pragma("GCC ivdep")
#elif defined(_OPENMP) && _OPENMP >= 201307  // OpenMP-4.0 (July, 2013)
#define CeedPragmaSIMD _Pragma("omp simd")
#else
#define CeedPragmaSIMD
#endif
#endif

/// Integer type, used for indexing
/// @ingroup Ceed
typedef int32_t CeedInt;
#define CeedInt_FMT "d"

/// Integer type, used array sizes
/// @ingroup Ceed
typedef ptrdiff_t CeedSize;
#define CeedSize_FMT "td"

/// Integer type, for small integers
/// @ingroup Ceed
typedef int8_t CeedInt8;
#define CeedInt8_FMT "d"

/// Scalar (floating point) types
///
/// @ingroup Ceed
typedef enum {
  /// Single precision
  CEED_SCALAR_FP32,
  /// Double precision
  CEED_SCALAR_FP64
} CeedScalarType;
/// Base scalar type for the library to use: change which header is included to change the precision.
#include "ceed-f64.h"  // IWYU pragma: export

/// Ceed error code.
///
/// This enum is used to specify the type of error returned by a function.
/// A zero error code is success, negative error codes indicate terminal errors and positive error codes indicate nonterminal errors.
/// With nonterminal errors the object state has not been modified, but with terminal errors the object data is likely modified or corrupted.
/// @ingroup Ceed
typedef enum {
  /// Success error code
  CEED_ERROR_SUCCESS = 0,
  /// Minor error, generic
  CEED_ERROR_MINOR = 1,
  /// Minor error, dimension mismatch in inputs
  CEED_ERROR_DIMENSION = 2,
  /// Minor error, incomplete object setup
  CEED_ERROR_INCOMPLETE = 3,
  /// Minor error, incompatible arguments/configuration
  CEED_ERROR_INCOMPATIBLE = 4,
  /// Minor error, access lock problem
  CEED_ERROR_ACCESS = 5,
  /// Major error, generic
  CEED_ERROR_MAJOR = -1,
  /// Major error, internal backend error
  CEED_ERROR_BACKEND = -2,
  /// Major error, operation unsupported by current backend
  CEED_ERROR_UNSUPPORTED = -3,
} CeedErrorType;

/// Specify memory type.
/// Many Ceed interfaces take or return pointers to memory.
/// This enum is used to specify where the memory being provided or requested must reside.
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
  /// Implementation can use and modify the data provided by the user, but does not take ownership.
  CEED_USE_POINTER,
  /// Implementation takes ownership of the pointer and will free using CeedFree() when done using it.
  /// The user should not assume that the pointer remains valid after ownership has been transferred.
  /// Note that arrays allocated using C++ operator new or other allocators cannot generally be freed using CeedFree().
  /// CeedFree() is capable of freeing any memory that can be freed using free().
  CEED_OWN_POINTER,
} CeedCopyMode;

/// Denotes type of vector norm to be computed
/// @ingroup CeedVector
typedef enum {
  /// \f$\Vert \bm{x}\Vert_1 = \sum_i \vert x_i\vert\f$
  CEED_NORM_1,
  /// \f$\Vert \bm{x} \Vert_2 = \sqrt{\sum_i x_i^2}\f$
  CEED_NORM_2,
  /// \f$\Vert \bm{x} \Vert_\infty = \max_i \vert x_i \vert\f$
  CEED_NORM_MAX,
} CeedNormType;

/// Denotes whether a linear transformation or its transpose should be applied
/// @ingroup CeedBasis
typedef enum {
  /// Apply the linear transformation
  CEED_NOTRANSPOSE,
  /// Apply the transpose
  CEED_TRANSPOSE
} CeedTransposeMode;

/// Basis evaluation mode
/// @ingroup CeedBasis
typedef enum {
  /// Perform no evaluation (either because there is no data or it is already at quadrature points)
  CEED_EVAL_NONE = 0,
  /// Interpolate from nodes to quadrature points
  CEED_EVAL_INTERP = 1,
  /// Evaluate gradients at quadrature points from input in the basis
  CEED_EVAL_GRAD = 2,
  /// Evaluate divergence at quadrature points from input in the basis
  CEED_EVAL_DIV = 4,
  /// Evaluate curl at quadrature points from input in the basis
  CEED_EVAL_CURL = 8,
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

/// Type of basis shape to create non-tensor element basis.
/// Dimension can be extracted with bitwise AND (CeedElemTopology & 2**(dim + 2)) == TRUE
/// @ingroup CeedBasis
typedef enum {
  /// Line
  CEED_TOPOLOGY_LINE = 1 << 16 | 0,
  /// Triangle - 2D shape
  CEED_TOPOLOGY_TRIANGLE = 2 << 16 | 1,
  /// Quadralateral - 2D shape
  CEED_TOPOLOGY_QUAD = 2 << 16 | 2,
  /// Tetrahedron - 3D shape
  CEED_TOPOLOGY_TET = 3 << 16 | 3,
  /// Pyramid - 3D shape
  CEED_TOPOLOGY_PYRAMID = 3 << 16 | 4,
  /// Prism - 3D shape
  CEED_TOPOLOGY_PRISM = 3 << 16 | 5,
  /// Hexehedron - 3D shape
  CEED_TOPOLOGY_HEX = 3 << 16 | 6,
} CeedElemTopology;

/// Denotes type of data stored in a CeedQFunctionContext field
/// @ingroup CeedQFunction
typedef enum {
  /// Double precision value
  CEED_CONTEXT_FIELD_DOUBLE = 1,
  /// 32 bit integer value
  CEED_CONTEXT_FIELD_INT32 = 2,
} CeedContextFieldType;

#endif
