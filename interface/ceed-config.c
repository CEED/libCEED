// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed-impl.h>

const char *CeedGitVersion         = CEED_GIT_VERSION;
const char *CeedBuildConfiguration = CEED_BUILD_CONFIGURATION;

/// @addtogroup CeedUser
/// @{

/**
  @brief Get output of `git describe --dirty` from build time.

  While @ref CeedGetVersion() uniquely identifies the source code for release
  builds, it does not identify builds from other commits.

  @param[out] git_version A static string containing the Git commit description.

  If `git describe --always --dirty` fails, the string `"unknown"` will be provided.
  This could occur if Git is not installed or if libCEED is not being built from a repository, for example.`

  @ref Developer

  @sa CeedGetVersion() CeedGetBuildConfiguration()

  @return An error code: 0 - success, otherwise - failure
*/
int CeedGetGitVersion(const char **git_version) {
  *git_version = CeedGitVersion;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set whether or not to use clang when compiling for GPU (instead of nvrtc)

  @param[in] is_clang Whether or not to use clang on GPU

  @ref Developer

  @sa CeedGetIsClang()

  @return An error code: 0 - success, otherwise - failure
 */
int CeedSetIsClang(Ceed ceed, bool is_clang) {
  ceed->cuda_compile_with_clang = is_clang;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Determine if the current ceed is set to compile with clang when on GPU

  @param[out] is_clang The location to write the current GPU clang status to

  @ref Developer

  @sa CeedSetIsClang()

  @return An error code: 0 - success, otherwise - failure
 */
int CeedGetIsClang(Ceed ceed, bool *is_clang) {
  *is_clang = ceed->cuda_compile_with_clang;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get build variables as a multi-line string.

  Each line of the string has the format `VARNAME = value`.

  @param[out] build_config A static string containing build variables

  @ref Developer

  @sa CeedGetVersion() CeedGetGitVersion()

  @return An error code: 0 - success, otherwise - failure
*/
int CeedGetBuildConfiguration(const char **build_config) {
  *build_config = CeedBuildConfiguration;
  return CEED_ERROR_SUCCESS;
}

/// @}
