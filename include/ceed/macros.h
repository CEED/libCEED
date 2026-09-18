/// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
/// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
///
/// SPDX-License-Identifier: BSD-2-Clause
///
/// This file is part of CEED:  http://github.com/ceed

/// @file
/// Public header for additional macros used in CPU libCEED library code

#ifndef CEED_LIBRARY_DEFS_H
#define CEED_LIBRARY_DEFS_H



/// This macro provides the appropriate OpenMP Pragmas for the compilation environment.
/// @ingroup Ceed
#ifndef CeedPragmaOMP
#ifdef _OPENMP
#define CeedPragmaOMPHelper(x) _Pragma(#x)
#define CeedPragmaOMP(x) CeedPragmaOMPHelper(omp x)
#else
#define CeedPragmaOMP(x)
#endif
#endif
#ifndef CeedPragmaAtomic
#define CeedPragmaAtomic CeedPragmaOMP(atomic update)
#endif
#ifndef CeedPragmaCritical
#define CeedPragmaCritical(x) CeedPragmaOMP(critical(x))
#endif

/**
  @brief Calls a libCEED function and then checks the resulting error code.
  If the error code is non-zero, then the error handler is called and the call from the current function with the error code.

  @ref Developer
**/
#define CeedCall(...)        \
  do {                       \
    int ierr_ = __VA_ARGS__; \
    if (ierr_) return ierr_; \
  } while (0)

/**
  @brief Calls a libCEED function and then checks the resulting error code.
  If the error code is non-zero, then the error handler is called and the call from the current function with the error code.
  All interface level error codes are upgraded to `CEED_ERROR_BACKEND`.

  @ref Developer
**/
#define CeedCallBackend(...)                                                     \
  do {                                                                           \
    int ierr_ = __VA_ARGS__;                                                     \
    if (ierr_) return (ierr_ > CEED_ERROR_SUCCESS) ? CEED_ERROR_BACKEND : ierr_; \
  } while (0)

/**
  @brief Check that a particular condition is true and returns a `CeedError` if not.

  @ref Developer
**/
#define CeedCheck(cond, ceed, ecode, ...)                    \
  do {                                                       \
    if (!(cond)) return CeedError(ceed, ecode, __VA_ARGS__); \
  } while (0)

#endif  // CEED_LIBRARY_DEFS_H
