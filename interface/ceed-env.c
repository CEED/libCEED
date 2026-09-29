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

static inline int CeedGetEnvVariadic(const char **output, bool *is_set, ...) {
  va_list     args;
  const char *alias;

  if (is_set) *is_set = false;
  va_start(args, is_set);
  while ((alias = va_arg(args, char *))) {
    *output = getenv(alias);
    if (*output) {
      if (is_set) *is_set = true;
      break;
    }
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

#define CEED_ENV_ONLY_FLAG(suffix, default, ...)                                  \
  /**
    @brief Get the value of flag for the @a ceed context from the environment.

    @param[out] value  Pointer to store read value
    @param[out] is_set Pointer to store whether value was provided by the environment

    @return An error code: 0 - success, otherwise - failure

    @ref User
  **/                                                                             \
  int CeedGetEnv##suffix(bool *value, bool *is_set) {                             \
    const char *env_value;                                                        \
    bool        is_set_local;                                                     \
                                                                                  \
    *value = default;                                                             \
    CeedCall(CeedGetEnvVariadic(&env_value, &is_set_local, ##__VA_ARGS__, NULL)); \
    if (is_set) *is_set = is_set_local;                                           \
    CeedCall(CeedStringToBool(env_value, value));                                 \
    return CEED_ERROR_SUCCESS;                                                    \
  }

#define CEED_ENV_FLAG(suffix, default, ...)                                   \
  CEED_ENV_ONLY_FLAG(suffix, default, ##__VA_ARGS__)                          \
  /**
    @brief Get the value of flag for the @a ceed context, either from the user value or environment if unset.

    @param[in,out] ceed  `Ceed` context
    @param[out]    value Pointer to store read value

    @return An error code: 0 - success, otherwise - failure

    @ref User
  **/                                                                         \
  int CeedGet##suffix(Ceed ceed, bool *value) {                               \
    while (ceed->parent) ceed = ceed->parent;                                 \
                                                                              \
    CeedEnvFlag *flag = &ceed->ceed_env_##suffix;                             \
                                                                              \
    if (!flag->have_checked_environment) {                                    \
      bool env_value;                                                         \
                                                                              \
      CeedCall(CeedGetEnv##suffix(&env_value, &flag->is_set_in_environment)); \
      flag->have_checked_environment = true;                                  \
      if (!flag->is_set_by_user) flag->value = env_value;                     \
    }                                                                         \
    *value = flag->value;                                                     \
    return CEED_ERROR_SUCCESS;                                                \
  }                                                                           \
                                                                              \
  /**
    @brief Get whether the value of flag for the @a ceed context was set by the user or through the environment.

    @param[in,out] ceed   `Ceed` context
    @param[out]    is_set Pointer to store whether the value was set

    @return An error code: 0 - success, otherwise - failure

    @ref User
  **/                                                                         \
  int CeedGetIsSet##suffix(Ceed ceed, bool *is_set) {                         \
    while (ceed->parent) ceed = ceed->parent;                                 \
                                                                              \
    CeedEnvFlag *flag = &ceed->ceed_env_##suffix;                             \
                                                                              \
    if (!flag->have_checked_environment) {                                    \
      bool env_value;                                                         \
                                                                              \
      CeedCall(CeedGetEnv##suffix(&env_value, &flag->is_set_in_environment)); \
      flag->have_checked_environment = true;                                  \
      if (!flag->is_set_by_user) flag->value = env_value;                     \
    }                                                                         \
    *is_set = flag->is_set_in_environment || flag->is_set_by_user;            \
    return CEED_ERROR_SUCCESS;                                                \
  }                                                                           \
                                                                              \
  /**
    @brief Set the value of flag for the @a ceed context.

    @param[in,out] ceed  `Ceed` context
    @param[in]     value Value to set for the flag

    @return An error code: 0 - success, otherwise - failure

    @ref User
  **/                                                                         \
  int CeedSet##suffix(Ceed ceed, bool value) {                                \
    while (ceed->parent) ceed = ceed->parent;                                 \
                                                                              \
    CeedEnvFlag *flag = &ceed->ceed_env_##suffix;                             \
                                                                              \
    flag->value          = value;                                             \
    flag->is_set_by_user = true;                                              \
    return CEED_ERROR_SUCCESS;                                                \
  }

#define CEED_ENV_ONLY_STRING(suffix, default, ...)                                \
  /**
    @brief Get the value of string for the @a ceed context from the environment.

    @param[out] value  Pointer to store read value, do not free, invalidated by calls to `CeedSet##suffix()`
    @param[out] is_set Pointer to store read value, do not free

    @return An error code: 0 - success, otherwise - failure

    @ref User
  **/                                                                             \
  int CeedGetEnv##suffix(const char **value, bool *is_set) {                      \
    const char *env_value;                                                        \
    bool        is_set_local;                                                     \
                                                                                  \
    CeedCall(CeedGetEnvVariadic(&env_value, &is_set_local, ##__VA_ARGS__, NULL)); \
    if (is_set) *is_set = is_set_local;                                           \
    *value = env_value ? env_value : default;                                     \
    return CEED_ERROR_SUCCESS;                                                    \
  }

#define CEED_ENV_STRING(suffix, default, ...)                                \
  CEED_ENV_ONLY_STRING(suffix, default, ##__VA_ARGS__)                       \
  /**
    @brief Get the value of string for the @a ceed context, either from the user value or environment if unset.

    @param[in,out] ceed  `Ceed` context
    @param[out]    value Pointer to store read value

    @return An error code: 0 - success, otherwise - failure

    @ref User
  **/                                                                        \
  int CeedGet##suffix(Ceed ceed, const char **value) {                       \
    while (ceed->parent) ceed = ceed->parent;                                \
                                                                             \
    CeedEnvString *str = &ceed->ceed_env_##suffix;                           \
                                                                             \
    if (!str->have_checked_environment) {                                    \
      const char *env_value;                                                 \
                                                                             \
      CeedCall(CeedGetEnv##suffix(&env_value, &str->is_set_in_environment)); \
      str->have_checked_environment = true;                                  \
      if (!str->is_set_by_user) str->value = env_value;                      \
    }                                                                        \
    *value = str->value;                                                     \
    return CEED_ERROR_SUCCESS;                                               \
  }                                                                          \
                                                                             \
  /**
    @brief Get whether the value of string for the @a ceed context was set by the user or through the environment.

    @param[in,out] ceed   `Ceed` context
    @param[out]    is_set Pointer to store whether the value was set

    @return An error code: 0 - success, otherwise - failure

    @ref User
  **/                                                                        \
  int CeedGetIsSet##suffix(Ceed ceed, bool *is_set) {                        \
    while (ceed->parent) ceed = ceed->parent;                                \
                                                                             \
    CeedEnvString *str = &ceed->ceed_env_##suffix;                           \
                                                                             \
    if (!str->have_checked_environment) {                                    \
      const char *env_value;                                                 \
                                                                             \
      CeedCall(CeedGetEnv##suffix(&env_value, &str->is_set_in_environment)); \
      str->have_checked_environment = true;                                  \
      if (!str->is_set_by_user) str->value = env_value;                      \
    }                                                                        \
    *is_set = str->is_set_in_environment || str->is_set_by_user;             \
    return CEED_ERROR_SUCCESS;                                               \
  }                                                                          \
                                                                             \
  /**
    @brief Set the value of for the @a ceed context.

    @param[in,out] ceed  `Ceed` context
    @param[in]     value Value to set for the string

    @return An error code: 0 - success, otherwise - failure

    @ref User
  **/                                                                        \
  int CeedSet##suffix(Ceed ceed, const char *value) {                        \
    while (ceed->parent) ceed = ceed->parent;                                \
                                                                             \
    CeedEnvString *str = &ceed->ceed_env_##suffix;                           \
                                                                             \
    if (str->value && str->is_set_by_user) CeedCall(CeedFree(&str->value));  \
    if (value) {                                                             \
      CeedCall(CeedStringAllocCopy(value, (char **)&str->value));            \
    } else {                                                                 \
      str->value = NULL;                                                     \
    }                                                                        \
    str->is_set_by_user = true;                                              \
    return CEED_ERROR_SUCCESS;                                               \
  }

// Each environment variable must appear here and in include/ceed/ceed-env.h
#include <ceed/ceed-env-list.h>

#undef CeedGetEnvVariadicM
#undef CEED_ENV_STRING
#undef CEED_ENV_ONLY_STRING
#undef CEED_ENV_FLAG
#undef CEED_ENV_ONLY_FLAG
