// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed-impl.h>
#include <ceed/ceed-env.h>

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

static inline int CeedGetEnvVariadic(char **output, ...) {
  va_list     args;
  const char *alias;

  va_start(args, output);
  while ((alias = va_arg(args, char *))) {
    *output = getenv(alias);
    if (*output) break;
  }
  va_end(args);
  return CEED_ERROR_SUCCESS;
}

static inline int CeedStringToBool(const char *str, bool *value) {
  if (str && (strstr("false", str) || strstr("FALSE", str) || strstr("0", str))) {
    *value = false;
  } else if (str && (strstr("true", str) || strstr("TRUE", str) || strstr("1", str))) {
    *value = true;
  }
  return CEED_ERROR_SUCCESS;
}

#define CeedGetEnvVariadicM(output, ...) CeedGetEnvVariadic(output, ##__VA_ARGS__, NULL)

#define CEED_ENV_ONLY_FLAG(suffix, default, ...)              \
  /**
    @brief Get the value of flag for the @a ceed context from the environment.

    @param[out] value Pointer to store read value

    @return An error code: 0 - success, otherwise - failure

    @ref User
  **/                                                         \
  int CeedGetEnv##suffix(bool *value) {                       \
    char *env_value;                                          \
                                                              \
    *value = default;                                         \
    CeedCall(CeedGetEnvVariadicM(&env_value, ##__VA_ARGS__)); \
    CeedCall(CeedStringToBool(env_value, value));             \
    return CEED_ERROR_SUCCESS;                                \
  }

#define CEED_ENV_FLAG(suffix, default, ...)                                 \
  CEED_ENV_ONLY_FLAG(suffix, default, ##__VA_ARGS__)                        \
  /**
    @brief Get the value of flag for the @a ceed context, either from the user value or environment if unset.

    @param[in,out] ceed  `CeedQFunction` to set device pointer
    @param[out]    value Pointer to store read value

    @return An error code: 0 - success, otherwise - failure

    @ref User
  **/                                                                       \
  int CeedGet##suffix(Ceed ceed, bool *value) {                             \
    while (ceed->parent) ceed = ceed->parent;                               \
    if (!ceed->ceed_env_set_##suffix && !ceed->ceed_env_checked_##suffix) { \
      CeedCall(CeedGetEnv##suffix(&ceed->ceed_env_##suffix));               \
      ceed->ceed_env_checked_##suffix = true;                               \
    }                                                                       \
    *value = ceed->ceed_env_##suffix;                                       \
    return CEED_ERROR_SUCCESS;                                              \
  }                                                                         \
                                                                            \
  /**
    @brief Set the value of flag for the @a ceed context.

    @param[in,out] ceed  `CeedQFunction` to set device pointer
    @param[in]     value Value to set for the flag

    @return An error code: 0 - success, otherwise - failure

    @ref User
  **/                                                                       \
  int CeedSet##suffix(Ceed ceed, bool value) {                              \
    while (ceed->parent) ceed = ceed->parent;                               \
    ceed->ceed_env_##suffix     = value;                                    \
    ceed->ceed_env_set_##suffix = true;                                     \
    return CEED_ERROR_SUCCESS;                                              \
  }

#define CEED_ENV_ONLY_STRING(suffix, default, ...)            \
  /**
    @brief Get the value of string for the @a ceed context from the environment.

    @param[out] value Pointer to store read value, do not free

    @return An error code: 0 - success, otherwise - failure

    @ref User
  **/                                                         \
  int CeedGetEnv##suffix(const char **value) {                \
    char *env_value;                                          \
                                                              \
    CeedCall(CeedGetEnvVariadicM(&env_value, ##__VA_ARGS__)); \
    *value = env_value ? env_value : default;                 \
    return CEED_ERROR_SUCCESS;                                \
  }

#define CEED_ENV_STRING(suffix, default, ...)                                                                 \
  CEED_ENV_ONLY_STRING(suffix, default, ##__VA_ARGS__)                                                        \
  /**
    @brief Get the value of string for the @a ceed context, either from the user value or environment if unset.

    @param[in,out] ceed  `CeedQFunction` to set device pointer
    @param[out]    value Pointer to store read value

    @return An error code: 0 - success, otherwise - failure

    @ref User
  **/                                                                                                         \
  int CeedGet##suffix(Ceed ceed, const char **value) {                                                        \
    while (ceed->parent) ceed = ceed->parent;                                                                 \
    if (!ceed->ceed_env_set_##suffix && !ceed->ceed_env_checked_##suffix) {                                   \
      CeedCall(CeedGetEnv##suffix((const char **)&ceed->ceed_env_##suffix));                                  \
      ceed->ceed_env_checked_##suffix = true;                                                                 \
    }                                                                                                         \
    *value = ceed->ceed_env_##suffix;                                                                         \
    return CEED_ERROR_SUCCESS;                                                                                \
  }                                                                                                           \
                                                                                                              \
  /**
    @brief Set the value of for the @a ceed context.

    @param[in,out] ceed  `CeedQFunction` to set device pointer
    @param[in]     value Value to set for the string

    @return An error code: 0 - success, otherwise - failure

    @ref User
  **/                                                                                                         \
  int CeedSet##suffix(Ceed ceed, const char *value) {                                                         \
    while (ceed->parent) ceed = ceed->parent;                                                                 \
    if (ceed->ceed_env_##suffix && ceed->ceed_env_set_##suffix) CeedCall(CeedFree(&ceed->ceed_env_##suffix)); \
    if (value) {                                                                                              \
      CeedCall(CeedStringAllocCopy(value, &ceed->ceed_env_##suffix));                                         \
    } else {                                                                                                  \
      ceed->ceed_env_##suffix = NULL;                                                                         \
    }                                                                                                         \
    ceed->ceed_env_set_##suffix = true;                                                                       \
    return CEED_ERROR_SUCCESS;                                                                                \
  }

// Each environment variable must appear here and in include/ceed/ceed-env.h
#include <ceed/ceed-env-list.h>

#undef CeedGetEnvVariadicM
#undef CEED_ENV_STRING
#undef CEED_ENV_ONLY_STRING
#undef CEED_ENV_FLAG
#undef CEED_ENV_ONLY_FLAG
