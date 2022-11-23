// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed-impl.h>
#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-tools.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

/**
  @brief Check if valid file exists at path given

  @param ceed                  A Ceed object for error handling
  @param[in]  source_file_path Absolute path to source file
  @param[out] is_valid         Boolean flag indicating if file can be opend

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedCheckFilePath(Ceed ceed, const char *source_file_path, bool *is_valid) {
  // Sometimes we have path/to/file.h:function_name
  // Create tempory file path without name, if needed
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
  CeedDebug256(ceed, 1, "Checking for source file: ");
  CeedDebug(ceed, "%s\n", source_file_path_only);

  // Check for valid file path
  FILE *source_file;
  source_file = fopen(source_file_path_only, "rb");
  *is_valid   = !!source_file;

  if (*is_valid) {
    // Debug
    CeedDebug256(ceed, 1, "Found JiT source file: ");
    CeedDebug(ceed, "%s\n", source_file_path_only);

    fclose(source_file);
  }

  // Free temp file path, if used
  if (last_colon) CeedCall(CeedFree(&source_file_path_only));

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Load source file into initialized string buffer, including full text
           of local files in place of `#include "local.h"`

  @param ceed                  A Ceed object for error handling
  @param[in]  source_file_path Absolute path to source file
  @param[out] buffer           String buffer for source file contents

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedLoadSourceToInitializedBuffer(Ceed ceed, const char *source_file_path, char **buffer) {
  FILE *source_file;
  long  file_size, file_offset = 0;
  char *temp_buffer;

  // Debug
  CeedDebug256(ceed, 1, "---------- Ceed JiT ----------\n");
  CeedDebug256(ceed, 1, "Current source file: ");
  CeedDebug(ceed, "%s\n", source_file_path);
  CeedDebug256(ceed, 1, "Current buffer:\n");
  CeedDebug(ceed, "%s\n", *buffer);

  // Read file to temporary buffer
  source_file = fopen(source_file_path, "rb");
  if (!source_file) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_MAJOR, "Couldn't open source file: %s", source_file_path);
    // LCOV_EXCL_STOP
  }
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
    // -- Check for 'include' keyword
    const char *next_e     = strchr(first_hash, 'e');
    char        keyword[8] = "";
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
      bool  is_ceed_header    = is_hash_include && next_left_chevron && (next_new_line - next_left_chevron > 0) &&
                            (!strncmp(next_left_chevron, "<ceed/jit-source/", 17) || !strncmp(next_left_chevron, "<ceed/types.h>", 14) ||
                             !strncmp(next_left_chevron, "<ceed/ceed-f32.h>", 17) || !strncmp(next_left_chevron, "<ceed/ceed-f64.h>", 17));
      if (is_local_header || is_ceed_header) {
        // ---- Build source path
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
          CeedCall(CeedGetJitAbsolutePath(ceed, ceed_relative_path, &include_source_path));
          CeedCall(CeedFree(&ceed_relative_path));
        }
        // ---- Recursive call to load source to buffer
        CeedDebug256(ceed, 2, "JiT Including: %s\n", include_source_path);
        CeedCall(CeedLoadSourceToInitializedBuffer(ceed, include_source_path, buffer));
        CeedCall(CeedFree(&include_source_path));
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
  CeedDebug256(ceed, 1, "---------- Ceed JiT ----------\n");
  CeedDebug256(ceed, 1, "Current source file: ");
  CeedDebug(ceed, "%s\n", source_file_path);
  CeedDebug256(ceed, 1, "Final buffer:\n");
  CeedDebug(ceed, "%s\n", *buffer);

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Initalize and load source file into string buffer, including full text
           of local files in place of `#include "local.h"`.
         Note: Caller is responsible for freeing the string buffer with `CeedFree()`.

  @param ceed                  A Ceed object for error handling
  @param[in]  source_file_path Absolute path to source file
  @param[out] buffer           String buffer for source file contents

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedLoadSourceToBuffer(Ceed ceed, const char *source_file_path, char **buffer) {
  // Initalize buffer
  CeedCall(CeedCalloc(1, buffer));

  // Load to initalized buffer
  CeedCall(CeedLoadSourceToInitializedBuffer(ceed, source_file_path, buffer));

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Build an absolute filepath from a base filepath and an absolute filepath.
           This helps construct source file paths for `CeedLoadSourceToBuffer()`.
         Note: Caller is responsible for freeing the string buffer with `CeedFree()`.

  @param ceed                     A Ceed object for error handling
  @param[in]  base_file_path      Absolute path to current file
  @param[in]  relative_file_path  Relative path to target file
  @param[out] new_file_path       String buffer for absolute path to target file

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
  @param[out] relative_file_path Relative path to installed JiT file

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedGetJitRelativePath(const char *absolute_file_path, const char **relative_file_path) {
  *(relative_file_path) = strstr(absolute_file_path, "ceed/jit-source");

  if (!*relative_file_path) {
    // LCOV_EXCL_START
    return CeedError(NULL, CEED_ERROR_MAJOR, "Couldn't find relative path including 'ceed/jit-source' for: %s", absolute_file_path);
    // LCOV_EXCL_STOP
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Build an absolute filepath to a JiT file

  @param ceed                    A Ceed object for error handling
  @param[in]  relative_file_path Relative path to installed JiT file
  @param[out] absolute_file_path String buffer for absolute path to target file

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedGetJitAbsolutePath(Ceed ceed, const char *relative_file_path, char **absolute_file_path) {
  Ceed ceed_parent;

  // Debug
  CeedDebug256(ceed, 1, "---------- Ceed JiT ----------\n");
  CeedDebug256(ceed, 1, "Relative JiT source file: ");
  CeedDebug(ceed, "%s\n", relative_file_path);

  CeedCall(CeedGetParent(ceed, &ceed_parent));
  for (CeedInt i = 0; i < ceed_parent->num_jit_source_roots; i++) {
    bool is_valid;

    // Debug
    CeedDebug256(ceed, 1, "Checking JiT root: ");
    CeedDebug(ceed, "%s\n", ceed_parent->jit_source_roots[i]);

    // Build and check absolute path with current root
    CeedCall(CeedPathConcatenate(ceed, ceed_parent->jit_source_roots[i], relative_file_path, absolute_file_path));
    CeedCall(CeedCheckFilePath(ceed, *absolute_file_path, &is_valid));

    if (is_valid) return CEED_ERROR_SUCCESS;
    else CeedCall(CeedFree(absolute_file_path));
  }

  // LCOV_EXCL_START
  return CeedError(ceed, CEED_ERROR_MAJOR, "Couldn't find matching JiT source file: %s", relative_file_path);
  // LCOV_EXCL_STOP
}
