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
  This macro defines compiler attributes to the CEED_QFUNCTION to force inlining
    for called functions. The `inline` declaration does not necessarily enforce a
    compiler to inline a function. This can be deterimental to performance, so
    here we force inlining to occur unless inlining has been forced off (like
    during debugging).
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
#endif

/**
  @ingroup CeedQFunction
  This macro populates the correct function annotations for User QFunction
    source for code generation backends or populates default values for CPU
    backends. It also creates a variable `name_loc` populated with the correct
    source path for creating the respective User QFunction.
**/
#ifndef CEED_QFUNCTION
#define CEED_QFUNCTION(name)                                        \
  static const char              name##_loc[] = __FILE__ ":" #name; \
  CEED_QFUNCTION_ATTR static int name
#endif

/**
  @ingroup CeedQFunction
  This macro populates the correct function annotations for User QFunction
    helper function source for code generation backends or populates default
    values for CPU backends.
**/
#ifndef CEED_QFUNCTION_HELPER
#define CEED_QFUNCTION_HELPER CEED_QFUNCTION_ATTR static inline
#endif

/**
  @ingroup CeedQFunction
  Using VLA syntax to reshape User QFunction inputs and outputs can make
    user code more readable. VLA is a C99 feature that is not supported by
    the C++ dialect used by CUDA. This macro allows users to use the VLA
    syntax with the CUDA backends.
**/
#ifndef CEED_Q_VLA
#define CEED_Q_VLA Q
#endif

/**
  @ingroup Ceed
  This macro provides the appropriate SIMD Pragma for the compilation
    environment. Code generation backends may redefine this macro, as needed.
**/
#ifndef CeedPragmaSIMD
#if defined(__INTEL_COMPILER)
#define CeedPragmaSIMD _Pragma("vector")
/// Cannot use Intel pragma ivdep because it miscompiles unpacking symmetric tensors, as in
/// Poisson2DApply, where the SIMD loop body contains temporaries such as the following.
///
///     const CeedScalar dXdxdXdxT[2][2] = {{qd[i+0*Q], qd[i+2*Q]},
///                                         {qd[i+2*Q], qd[i+1*Q]}};
///     for (int j=0; j<2; j++)
///        vg[i+j*Q] = (du[0] * dXdxdXdxT[0][j] + du[1] * dXdxdXdxT[1][j]);
///
/// Miscompilation with pragma ivdep observed with icc (ICC) 19.0.5.281 20190815
/// at -O2 and above.
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

/// Scalar (floating point) types
///
/// @ingroup Ceed
typedef enum {
  /// Single precision
  CEED_SCALAR_FP32,
  /// Double precision
  CEED_SCALAR_FP64
} CeedScalarType;
/// Base scalar type for the library to use: change which header is
/// included to change the precision.
#include "ceed-f64.h"

/// Ceed Errors
///
/// This enum is used to specify the type of error returned by a function.
/// A zero error code is success, negative error codes indicate terminal errors
/// and positive error codes indicate nonterminal errors. With nonterminal errors
/// the object state has not been modifiend, but with terminal errors the object
/// data is likely modified or corrupted.
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

#endif
