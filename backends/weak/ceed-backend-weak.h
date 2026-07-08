// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <stdarg.h>

CEED_INTERN int CeedRegister_Weak(const char *name, int num_prefixes, ...);

#define CEED_BACKEND(name, num_prefixes, ...) \
  CEED_INTERN int __attribute__((weak)) name(void) { return CeedRegister_Weak(__func__, num_prefixes, __VA_ARGS__); }
