// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
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
  @brief Normalize a filepath

  @param[in]   ceed                        `Ceed` object for error handling
  @param[in]   source_file_path            Absolute path to source file
  @param[out]  normalized_source_file_path Normalized filepath

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
static int CeedNormalizePath(Ceed ceed, const char *source_file_path, char **normalized_source_file_path) {
  CeedCall(CeedStringAllocCopy(source_file_path, normalized_source_file_path));

  char *first_dot = strchr(*normalized_source_file_path, '.');

  while (first_dot) {
    char *search_from = first_dot + 1;
    char  keyword[5]  = "";

    // -- Check for /./ and covert to /
    if (first_dot != *normalized_source_file_path && strlen(first_dot) > 2) memcpy(keyword, &first_dot[-1], 3);
    bool is_here = !strcmp(keyword, "/./");

    if (is_here) {
      for (CeedInt i = 0; first_dot[i - 1]; i++) first_dot[i] = first_dot[i + 2];
      search_from = first_dot;
    } else {
      // -- Check for /foo/../ and convert to /
      if (first_dot != *normalized_source_file_path && strlen(first_dot) > 3) memcpy(keyword, &first_dot[-1], 4);
      bool is_up_one = !strcmp(keyword, "/../");

      if (is_up_one) {
        char *last_slash = &first_dot[-2];

        while (last_slash[0] != '/' && last_slash != *normalized_source_file_path) last_slash--;
        CeedCheck(last_slash != *normalized_source_file_path, ceed, CEED_ERROR_MAJOR, "Malformed source path %s", source_file_path);
        for (CeedInt i = 0; first_dot[i - 1]; i++) last_slash[i] = first_dot[i + 2];
        search_from = last_slash;
      }
    }
    first_dot = strchr(search_from, '.');
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Load source file into initialized string buffer, including full text of local files in place of `#include "local.h"`.
    This also updates the `num_file_paths` and `source_file_paths`.
    Callers are responsible freeing all filepath strings and the string buffer with @ref CeedFree().

  @param[in]     ceed             `Ceed` object for error handling
  @param[in]     source_file_path Absolute path to source file
  @param[in,out] num_file_paths   Number of files already included
  @param[in,out] file_paths       Paths of files already included
  @param[out]    buffer           String buffer for source file contents

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedLoadSourceToInitializedBuffer(Ceed ceed, const char *source_file_path, CeedInt *num_file_paths, char ***file_paths, char **buffer) {
  FILE *source_file;
  long  file_size, file_offset = 0;
  char *temp_buffer;

  // Debug
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "---------- Ceed JiT ----------\n");
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "Current source file: ");
  CeedDebug(ceed, "%s\n", source_file_path);
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "Current buffer:\n");
  CeedDebug(ceed, "%s\n", *buffer);

  // Read file to temporary buffer
  source_file = fopen(source_file_path, "rb");
  CeedCheck(source_file, ceed, CEED_ERROR_MAJOR, "Couldn't open source file: %s", source_file_path);
  // -- Compute size of source
  fseek(source_file, 0L, SEEK_END);
  file_size = ftell(source_file);
  rewind(source_file);
  //  -- Allocate memory for entire source file
  CeedCall(CeedCalloc(file_size + 1, &temp_buffer));
  // -- Copy the file into the buffer
  if (1 != fread(temp_buffer, file_size, 1, source_file)) {
    // LCOV_EXCL_START
    fclose(source_file);
    CeedCall(CeedFree(&temp_buffer));
    return CeedError(ceed, CEED_ERROR_MAJOR, "Couldn't read source file: %s", source_file_path);
    // LCOV_EXCL_STOP
  }
  fclose(source_file);

  // Search for headers to include
  const char *first_hash = strchr(temp_buffer, '#');

  while (first_hash) {
    // -- Check for 'pragma' keyword
    const char *next_m     = strchr(first_hash, 'm');
    char        keyword[8] = "";

    if (next_m && next_m - first_hash >= 5) memcpy(keyword, &next_m[-4], 6);
    bool is_hash_pragma = !strcmp(keyword, "pragma");

    // ---- Spaces allowed in '#  pragma'
    if (next_m) {
      for (CeedInt i = 1; first_hash - next_m + i < -5; i++) {
        is_hash_pragma &= first_hash[i] == ' ';
      }
    }
    if (is_hash_pragma) {
      // -- Check if '#pragma once'
      char *next_o         = strchr(first_hash, 'o');
      char *next_new_line  = strchr(first_hash, '\n');
      bool  is_pragma_once = next_o && (next_new_line - next_o > 0) && !strncmp(next_o, "once", 4);

      // -- Copy into buffer, omitting last line if #pragma once
      long current_size = strlen(*buffer);
      long copy_size    = first_hash - &temp_buffer[file_offset] + (is_pragma_once ? 0 : (next_new_line - first_hash + 1));

      CeedCall(CeedRealloc(current_size + copy_size + 2, buffer));
      memcpy(&(*buffer)[current_size], "\n", 2);
      memcpy(&(*buffer)[current_size + 1], &temp_buffer[file_offset], copy_size);
      memcpy(&(*buffer)[current_size + copy_size], "", 1);

      file_offset = strchr(first_hash, '\n') - temp_buffer + 1;
    }

    // -- Check for 'include' keyword
    const char *next_e = strchr(first_hash, 'e');

    if (next_e && next_e - first_hash >= 7) memcpy(keyword, &next_e[-6], 7);
    bool is_hash_include = !strcmp(keyword, "include");

    // ---- Spaces allowed in '#  include <header.h>'
    if (next_e) {
      for (CeedInt i = 1; first_hash - next_e + i < -6; i++) {
        is_hash_include &= first_hash[i] == ' ';
      }
    }
    if (is_hash_include) {
      // -- Copy into buffer all preceding #
      long current_size = strlen(*buffer);
      long copy_size    = first_hash - &temp_buffer[file_offset];

      CeedCall(CeedRealloc(current_size + copy_size + 2, buffer));
      memcpy(&(*buffer)[current_size], "\n", 2);
      memcpy(&(*buffer)[current_size + 1], &temp_buffer[file_offset], copy_size);
      memcpy(&(*buffer)[current_size + copy_size], "", 1);
      // -- Load local "header.h"
      char *next_quote        = strchr(first_hash, '"');
      char *next_new_line     = strchr(first_hash, '\n');
      bool  is_local_header   = is_hash_include && next_quote && (next_new_line - next_quote > 0);
      char *next_left_chevron = strchr(first_hash, '<');
      bool  is_ceed_header    = next_left_chevron && (next_new_line - next_left_chevron > 0) &&
                            (!strncmp(next_left_chevron, "<ceed/jit-source/", 17) || !strncmp(next_left_chevron, "<ceed/types.h>", 14) ||
                             !strncmp(next_left_chevron, "<ceed/ceed-f32.h>", 17) || !strncmp(next_left_chevron, "<ceed/ceed-f64.h>", 17));

      if (is_local_header || is_ceed_header) {
        // ---- Build source path
        bool  is_included = false;
        char *include_source_path;

        if (is_local_header) {
          long root_length           = strrchr(source_file_path, '/') - source_file_path;
          long include_file_name_len = strchr(&next_quote[1], '"') - next_quote - 1;

          CeedCall(CeedCalloc(root_length + include_file_name_len + 2, &include_source_path));
          memcpy(include_source_path, source_file_path, root_length + 1);
          memcpy(&include_source_path[root_length + 1], &next_quote[1], include_file_name_len);
          memcpy(&include_source_path[root_length + include_file_name_len + 1], "", 1);
        } else {
          char *next_right_chevron = strchr(first_hash, '>');
          char *ceed_relative_path;
          long  ceed_relative_path_length = next_right_chevron - next_left_chevron - 1;

          CeedCall(CeedCalloc(ceed_relative_path_length + 1, &ceed_relative_path));
          memcpy(ceed_relative_path, &next_left_chevron[1], ceed_relative_path_length);
          CeedCall(CeedGetJitAbsolutePath(ceed, ceed_relative_path, (const char **)&include_source_path));
          CeedCall(CeedFree(&ceed_relative_path));
        }
        // ---- Recursive call to load source to buffer
        char *normalized_include_source_path;

        CeedCall(CeedNormalizePath(ceed, include_source_path, &normalized_include_source_path));
        for (CeedInt i = 0; i < *num_file_paths; i++) is_included |= !strcmp(normalized_include_source_path, (*file_paths)[i]);
        if (!is_included) {
          CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "JiT Including: %s\n", normalized_include_source_path);
          CeedCall(CeedLoadSourceToInitializedBuffer(ceed, normalized_include_source_path, num_file_paths, file_paths, buffer));
          CeedCall(CeedRealloc(*num_file_paths + 1, file_paths));
          CeedCall(CeedStringAllocCopy(normalized_include_source_path, &(*file_paths)[*num_file_paths]));
          (*num_file_paths)++;
        }
        CeedCall(CeedFree(&include_source_path));
        CeedCall(CeedFree(&normalized_include_source_path));
      }
      file_offset = strchr(first_hash, '\n') - temp_buffer + 1;
    }
    // -- Next hash
    first_hash = strchr(&first_hash[1], '#');
  }
  // Copy rest of source file into buffer
  long current_size = strlen(*buffer);
  long copy_size    = strlen(&temp_buffer[file_offset]);

  CeedCall(CeedRealloc(current_size + copy_size + 2, buffer));
  memcpy(&(*buffer)[current_size], "\n", 2);
  memcpy(&(*buffer)[current_size + 1], &temp_buffer[file_offset], copy_size);
  memcpy(&(*buffer)[current_size + copy_size + 1], "", 1);

  // Cleanup
  CeedCall(CeedFree(&temp_buffer));

  // Debug
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "---------- Ceed JiT ----------\n");
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "Current source file: ");
  CeedDebug(ceed, "%s\n", source_file_path);
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "Final buffer:\n");
  CeedDebug(ceed, "%s\n", *buffer);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Load source file into initialized string buffer, including full text of local files in place of `#include "local.h"`.
    This also initializes and populates the `num_file_paths` and `source_file_paths`.
    Callers are responsible freeing all filepath strings and the string buffer with @ref CeedFree().

  @param[in]     ceed             `Ceed` object for error handling
  @param[in]     source_file_path Absolute path to source file
  @param[in,out] num_file_paths   Number of files already included
  @param[in,out] file_paths       Paths of files already included
  @param[out]    buffer           String buffer for source file contents

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedLoadSourceAndInitializeBuffer(Ceed ceed, const char *source_file_path, CeedInt *num_file_paths, char ***file_paths, char **buffer) {
  // Ensure defaults were set
  *num_file_paths = 0;
  *file_paths     = NULL;

  // Initialize
  CeedCall(CeedCalloc(1, buffer));

  // And load source
  CeedCall(CeedLoadSourceToInitializedBuffer(ceed, source_file_path, num_file_paths, file_paths, buffer));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Initialize and load source file into string buffer, including full text of local files in place of `#include "local.h"`.
    User @ref CeedLoadSourceAndInitializeBuffer() and @ref CeedLoadSourceToInitializedBuffer() if loading multiple source files into the same buffer. 
    Caller is responsible for freeing the string buffer with @ref CeedFree().

  @param[in]  ceed             `Ceed` object for error handling
  @param[in]  source_file_path Absolute path to source file
  @param[out] buffer           String buffer for source file contents

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedLoadSourceToBuffer(Ceed ceed, const char *source_file_path, char **buffer) {
  char  **file_paths     = NULL;
  CeedInt num_file_paths = 0;

  // Load
  CeedCall(CeedLoadSourceAndInitializeBuffer(ceed, source_file_path, &num_file_paths, &file_paths, buffer));

  // Cleanup
  for (CeedInt i = 0; i < num_file_paths; i++) CeedCall(CeedFree(&file_paths[i]));
  CeedCall(CeedFree(&file_paths));
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
int CeedPathConcatenate(Ceed ceed, const char *base_file_path, const char *relative_file_path, char **new_file_path) {
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
  Ceed ceed_parent;

  // Debug
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "---------- Ceed JiT ----------\n");
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "Relative JiT source file: ");
  CeedDebug(ceed, "%s\n", relative_file_path);

  CeedCall(CeedGetParent(ceed, &ceed_parent));
  for (CeedInt i = 0; i < ceed_parent->num_jit_source_roots; i++) {
    bool is_valid;

    // Debug
    CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "Checking JiT root: ");
    CeedDebug(ceed, "%s\n", ceed_parent->jit_source_roots[i]);

    // Build and check absolute path with current root
    CeedCall(CeedPathConcatenate(ceed, ceed_parent->jit_source_roots[i], relative_file_path, (char **)absolute_file_path));
    CeedCall(CeedCheckFilePath(ceed, *absolute_file_path, &is_valid));

    if (is_valid) return CEED_ERROR_SUCCESS;
    // LCOV_EXCL_START
    else CeedCall(CeedFree(absolute_file_path));
    // LCOV_EXCL_STOP
  }
  // LCOV_EXCL_START
  return CeedError(ceed, CEED_ERROR_MAJOR, "Couldn't find matching JiT source file: %s", relative_file_path);
  // LCOV_EXCL_STOP
}
