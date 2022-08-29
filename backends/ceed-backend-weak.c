// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>

// LCOV_EXCL_START
// This function provides improved error messages for uncompiled backends
static int CeedInit_Weak(const char *resource, Ceed ceed) {
  return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                   "Backend not currently compiled: %s\n"
                   "Consult the installation instructions to compile this backend",
                   resource);
}

// This function provides a debug target for weak symbols
static int CeedRegister_Weak(const char *name, int num_prefixes, ...) {
  va_list prefixes;
  va_start(prefixes, num_prefixes);
  int ierr;
  for (int i = 0; i < num_prefixes; i++) {
    const char *prefix = va_arg(prefixes, const char *);
    CeedDebugEnv("** Weak Register: %s", prefix);
    ierr = CeedRegisterImpl(prefix, CeedInit_Weak, CEED_MAX_BACKEND_PRIORITY);
    if (ierr) va_end(prefixes);  // Prevent leak on error
    CeedChk(ierr);
  }
  va_end(prefixes);
  return CEED_ERROR_SUCCESS;
}
// LCOV_EXCL_STOP

#define MACRO(name, num_prefixes, ...)              \
  CEED_INTERN int name(void) __attribute__((weak)); \
  int             name(void) { return CeedRegister_Weak(__func__, num_prefixes, __VA_ARGS__); }
#include "ceed-backend-list.h"
#undef MACRO
