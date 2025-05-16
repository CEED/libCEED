// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-backend-register.h"
#include <ceed.h>
#include "ceed-backend-init.h"

#define CEED_PRIORITY_UNCOMPILED 1024  // Revisit if we ever use 4 digit priority values

//------------------------------------------------------------------------------
// Backend register functions
//------------------------------------------------------------------------------
// LCOV_EXCL_START
// This function provides error messages on initialization attempt for uncompiled backends
static int CeedInit_Uncompiled(const char *resource, Ceed ceed) {
  return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Backend not currently compiled: %s\nConsult the installation instructions to compile this backend",
                   resource);
}
// LCOV_EXCL_STOP

// This function registers uncompiled backends
static int CeedRegister_Uncompiled(int (*f)(const char *, struct Ceed_private *), int num_prefixes, ...) {
  va_list prefixes;
  int     ierr;

  va_start(prefixes, num_prefixes);
  for (int i = 0; i < num_prefixes; i++) {
    const char *prefix   = va_arg(prefixes, const char *);
    const int   priority = va_arg(prefixes, int);

    CeedDebugEnv("Weak Register : %s, %d", prefix, priority);
    ierr = CeedRegisterImpl(prefix, CeedInit_Uncompiled, CEED_PRIORITY_UNCOMPILED);
    if (ierr) {
      va_end(prefixes);  // Prevent leak on error
      return ierr;
    }
  }
  va_end(prefixes);
  return CEED_ERROR_SUCCESS;
}

// This function registers compiled backends
static int CeedRegister_Compiled(int (*f)(const char *, struct Ceed_private *), int num_prefixes, ...) {
  va_list prefixes;
  int     ierr;

  va_start(prefixes, num_prefixes);
  for (int i = 0; i < num_prefixes; i++) {
    const char *prefix   = va_arg(prefixes, const char *);
    const int   priority = va_arg(prefixes, int);

    ierr = CeedRegister(prefix, f, priority);
    if (ierr) {
      va_end(prefixes);  // Prevent leak on error
      return ierr;
    }
  }
  va_end(prefixes);
  return CEED_ERROR_SUCCESS;
}

// This macro creates a wrapper registration function that calls the compiled/uncompiled registration function
#define CEED_BACKEND(name, suffix, is_enabled, num_prefixes, ...)               \
  CEED_INTERN int CeedRegister_##name##suffix(void) {                           \
    if (is_enabled) {                                                           \
      return CeedRegister_Compiled(CeedInit_##name, num_prefixes, __VA_ARGS__); \
    } else {                                                                    \
      return CeedRegister_Uncompiled(NULL, num_prefixes, __VA_ARGS__);          \
    }                                                                           \
  }
#include "../backends/ceed-backend-list.h"
#undef CEED_BACKEND

//------------------------------------------------------------------------------
