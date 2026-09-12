// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed.h>
#include <ceed/backend.h>

#include <dlfcn.h>

#define CeedRunFunction_Cpu(ceed, handle, name, function, ...)                 \
  do {                                                                         \
    *(void **)(&op_function) = (void *)dlsym(handle, name);                    \
    if (function == NULL) {                                                    \
      return CeedError((ceed), CEED_ERROR_BACKEND, "Failed to load function"); \
    }                                                                          \
    CeedCallBackend((*function)(__VA_ARGS__));                                 \
  } while (0)

CEED_INTERN int CeedCompile_Cpu(Ceed ceed, const char *source, const char *name, void **handle, const CeedInt num_defines, ...);
CEED_INTERN int CeedTryCompile_Cpu(Ceed ceed, const char *source, const char *name, bool *is_compile_good, void **handle, const CeedInt num_defines,
                                   ...);
