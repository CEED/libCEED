/// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
/// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
///
/// SPDX-License-Identifier: BSD-2-Clause
///
/// This file is part of CEED:  http://github.com/ceed

/// @file
/// Header for declaring environment variables used by libCEED.
#pragma once

#include "macros.h"

#include <stdbool.h>

// This is to make clang-tidy happy
#ifndef CEED_EXTERN
#if defined(__clang_analyzer__)
#define CEED_EXTERN extern
#elif defined(__cplusplus)
#define CEED_EXTERN extern "C" CEED_VISIBILITY(default)
#else
#define CEED_EXTERN extern CEED_VISIBILITY(default)
#endif
typedef struct Ceed_private *Ceed;
#endif

#define CEED_ENV_ONLY_FLAG(suffix, default, ...) CEED_EXTERN int CeedGetEnv##suffix(bool *value, bool *is_set);
#define CEED_ENV_FLAG(suffix, default, ...)                      \
  CEED_ENV_ONLY_FLAG(suffix, default, ##__VA_ARGS__)             \
  CEED_EXTERN int CeedGet##suffix(Ceed ceed, bool *value);       \
  CEED_EXTERN int CeedGetIsSet##suffix(Ceed ceed, bool *is_set); \
  CEED_EXTERN int CeedSet##suffix(Ceed ceed, bool value);
#define CEED_ENV_ONLY_STRING(suffix, default, ...) CEED_EXTERN int CeedGetEnv##suffix(const char **value, bool *is_set);
#define CEED_ENV_STRING(suffix, default, ...)                     \
  CEED_ENV_ONLY_STRING(suffix, default, ##__VA_ARGS__)            \
  CEED_EXTERN int CeedGet##suffix(Ceed ceed, const char **value); \
  CEED_EXTERN int CeedGetIsSet##suffix(Ceed ceed, bool *is_set);  \
  CEED_EXTERN int CeedSet##suffix(Ceed ceed, const char *value);

#include "ceed-env-list.h"

#undef CEED_ENV_STRING
#undef CEED_ENV_ONLY_STRING
#undef CEED_ENV_FLAG
#undef CEED_ENV_ONLY_FLAG
