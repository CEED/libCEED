// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
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
  @brief Get output of `git describe --dirty` from build time

  While @ref CeedGetVersion() uniqely identifies the source code for release
  builds, it does not identify builds from other commits.

  @param[out] git_version A static string containing the Git commit description.

  If `git describe --always --dirty` fails, the string `"unknown"` will be provided.
  This could occur if Git is not installed or if libCEED is not being built from a repository, for example.`

  @return An error code: 0 - success, otherwise - failure

  @ref Developer

  @sa CeedGetVersion() CeedGetBuildConfiguration()
*/
int CeedGetGitVersion(const char **git_version) {
  *git_version = CeedGitVersion;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get build variables as a multi-line string

  Each line of the string has the format `VARNAME = value`.

  @param[out] build_config A static string containing build variables

  @return An error code: 0 - success, otherwise - failure

  @ref Developer

  @sa CeedGetVersion() CeedGetGitVersion()
*/
int CeedGetBuildConfiguration(const char **build_config) {
  *build_config = CeedBuildConfiguration;
  return CEED_ERROR_SUCCESS;
}

/// @}
