// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed-impl.h>
#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

/**
  @brief Check if valid file exists at path given

  @param[in]  ceed             `Ceed` object for error handling
  @param[in]  source_file_path Absolute path to source file
  @param[out] is_valid         Boolean flag indicating if file can be opened

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedCheckFilePath(Ceed ceed, const char *source_file_path, bool *is_valid) {
  // Sometimes we have path/to/file.h:function_name
  // Create temporary file path without name, if needed
  char *source_file_path_only;
  char *last_colon = strrchr(source_file_path, ':');

  if (last_colon) {
    size_t source_file_path_length = (last_colon - source_file_path + 1);

    CeedCall(CeedCalloc(source_file_path_length, &source_file_path_only));
    memcpy(source_file_path_only, source_file_path, source_file_path_length - 1);
  } else {
    source_file_path_only = (char *)source_file_path;
  }

  // Debug
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "Checking for source file: ");
  CeedDebug(ceed, "%s\n", source_file_path_only);

  // Check for valid file path
  FILE *source_file;
  source_file = fopen(source_file_path_only, "rb");
  *is_valid   = source_file;

  if (*is_valid) {
    // Debug
    CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "Found JiT source file: ");
    CeedDebug(ceed, "%s\n", source_file_path_only);
    fclose(source_file);
  }

  // Free temp file path, if used
  if (last_colon) CeedCall(CeedFree(&source_file_path_only));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Build an absolute filepath from a base filepath and an absolute filepath.

  This helps construct source file paths for @ref CeedLoadSourceToBuffer().

  Note: Caller is responsible for freeing the string buffer with @ref CeedFree().

  @param[in]  ceed               `Ceed` object for error handling
  @param[in]  base_file_path     Absolute path to current file
  @param[in]  relative_file_path Relative path to target file
  @param[out] new_file_path      String buffer for absolute path to target file

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
static int CeedPathConcatenate(Ceed ceed, const char *base_file_path, const char *relative_file_path, char **new_file_path) {
  char  *last_slash  = strrchr(base_file_path, '/');
  size_t base_length = (last_slash - base_file_path + 1), relative_length = strlen(relative_file_path),
         new_file_path_length = base_length + relative_length + 1;

  CeedCall(CeedCalloc(new_file_path_length, new_file_path));
  memcpy(*new_file_path, base_file_path, base_length);
  memcpy(&((*new_file_path)[base_length]), relative_file_path, relative_length);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Find the relative filepath to an installed JiT file

  @param[in]  absolute_file_path Absolute path to installed JiT file
  @param[out] relative_file_path Relative path to installed JiT file, a substring of the absolute path

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedGetJitRelativePath(const char *absolute_file_path, const char **relative_file_path) {
  *(relative_file_path) = strstr(absolute_file_path, "ceed/jit-source");
  CeedCheck(*relative_file_path, NULL, CEED_ERROR_MAJOR, "Couldn't find relative path including 'ceed/jit-source' for: %s", absolute_file_path);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Build an absolute filepath to a JiT file

  @param[in]  ceed               `Ceed` object for error handling
  @param[in]  relative_file_path Relative path to installed JiT file
  @param[out] absolute_file_path String buffer for absolute path to target file, to be freed by caller

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedGetJitAbsolutePath(Ceed ceed, const char *relative_file_path, const char **absolute_file_path) {
  const char **jit_source_dirs;
  CeedInt      num_source_dirs;

  // Debug
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "---------- Ceed JiT ----------\n");
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "Relative JiT source file: ");
  CeedDebug(ceed, "%s\n", relative_file_path);

  CeedCallBackend(CeedGetJitSourceRoots(ceed, &num_source_dirs, &jit_source_dirs));
  for (CeedInt i = 0; i < num_source_dirs; i++) {
    bool is_valid;

    // Debug
    CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "Checking JiT root: ");
    CeedDebug(ceed, "%s\n", jit_source_dirs[i]);

    // Build and check absolute path with current root
    CeedCall(CeedPathConcatenate(ceed, jit_source_dirs[i], relative_file_path, (char **)absolute_file_path));
    CeedCall(CeedCheckFilePath(ceed, *absolute_file_path, &is_valid));

    if (is_valid) {
      CeedCallBackend(CeedRestoreJitSourceRoots(ceed, &jit_source_dirs));
      return CEED_ERROR_SUCCESS;
    }
    // LCOV_EXCL_START
    else {
      CeedCall(CeedFree(absolute_file_path));
    }
    // LCOV_EXCL_STOP
  }
  // LCOV_EXCL_START
  return CeedError(ceed, CEED_ERROR_MAJOR, "Couldn't find matching JiT source file: %s", relative_file_path);
  // LCOV_EXCL_STOP
}
