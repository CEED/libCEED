// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>
#include <ceed-impl.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

/**
  @brief Load source file into initalized string buffer, including full text
           of local files in place of `#include "local.h"`

  @param ceed                  A Ceed object for error handling
  @param[in]  source_file_path Absolute path to source file
  @param[out] buffer           String buffer for source file contents

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
static inline int CeedLoadSourceToInitalizedBuffer(Ceed ceed,
    const char *source_file_path, char **buffer) {
  int ierr;
  FILE *source_file;
  long file_size, file_offset = 0;
  char *temp_buffer;

  // Debug
  CeedDebug256(ceed, 1, "---------- Ceed JiT ----------\n");
  CeedDebug256(ceed, 1, "Current source file: ");
  CeedDebug256(ceed, 255, "%s\n", source_file_path);
  CeedDebug256(ceed, 1, "Current buffer:\n");
  CeedDebug256(ceed, 255, "%s\n", *buffer);

  // Read file to temporary buffer
  source_file = fopen(source_file_path, "rb");
  if (!source_file)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_MAJOR, "Couldn't open source file: %s",
                     source_file_path);
  // LCOV_EXCL_STOP
  // -- Compute size of source
  fseek(source_file, 0L, SEEK_END);
  file_size = ftell(source_file);
  rewind(source_file);
  //  -- Allocate memory for entire source file
  ierr = CeedCalloc(file_size + 1, &temp_buffer); CeedChk(ierr);
  // -- Copy the file into the buffer
  if (1 != fread(temp_buffer, file_size, 1, source_file)) {
    // LCOV_EXCL_START
    fclose(source_file);
    ierr = CeedFree(&temp_buffer); CeedChk(ierr);
    return CeedError(ceed, CEED_ERROR_MAJOR, "Couldn't read source file: %s",
                     source_file_path);
    // LCOV_EXCL_STOP
  }
  fclose(source_file);

  // Search for headers to include
  const char *first_hash = strchr(temp_buffer, '#');
  while (first_hash) {
    // -- Check for 'include' keyword
    const char *next_e = strchr(first_hash, 'e');
    char keyword[8] = "";
    if (next_e)
      strncpy(keyword, &next_e[-6], 7);
    bool is_hash_include = !strcmp(keyword, "include");
    // ---- Spaces allowed in '#  include <header.h>'
    if (next_e)
      for (CeedInt i = 1; first_hash - next_e + i < -6; i++)
        is_hash_include &= first_hash[i] == ' ';
    if (is_hash_include) {
      // -- Copy into buffer all preceding #
      long current_size = strlen(*buffer);
      long copy_size = first_hash - &temp_buffer[file_offset];
      ierr = CeedRealloc(current_size + copy_size + 2, buffer); CeedChk(ierr);
      strncpy(&(*buffer)[current_size], "\n", 2);
      strncpy(&(*buffer)[current_size + 1], &temp_buffer[file_offset], copy_size);
      strncpy(&(*buffer)[current_size + copy_size], "", 1);
      // -- Load local "header.h"
      char *next_quote = strchr(first_hash, '"');
      char *next_new_line = strchr(first_hash, '\n');
      bool is_local_header = is_hash_include && next_quote
                             && (next_new_line - next_quote > 0);
      if (is_local_header) {
        // ---- Build source path
        char *include_source_path;
        long root_length = strrchr(source_file_path, '/') - source_file_path;
        long include_file_name_len = strchr(&next_quote[1], '"') - next_quote - 1;
        ierr = CeedCalloc(root_length + include_file_name_len + 2,
                          &include_source_path); CeedChk(ierr);
        strncpy(include_source_path, source_file_path, root_length + 1);
        strncpy(&include_source_path[root_length + 1], &next_quote[1],
                include_file_name_len);
        strncpy(&include_source_path[root_length + include_file_name_len + 1], "", 1);
        // ---- Recursive call to load source to buffer
        ierr = CeedLoadSourceToInitalizedBuffer(ceed, include_source_path, buffer);
        CeedChk(ierr);
        ierr = CeedFree(&include_source_path); CeedChk(ierr);
      }
      file_offset = strchr(first_hash, '\n') - temp_buffer + 1;
    }
    // -- Next hash
    first_hash = strchr(&first_hash[1], '#');
  }
  // Copy rest of source file into buffer
  long current_size = strlen(*buffer);
  long copy_size = strlen(&temp_buffer[file_offset]);
  ierr = CeedRealloc(current_size + copy_size + 2, buffer); CeedChk(ierr);
  strncpy(&(*buffer)[current_size], "\n", 2);
  strncpy(&(*buffer)[current_size + 1], &temp_buffer[file_offset], copy_size);
  strncpy(&(*buffer)[current_size + copy_size + 1], "", 1);

  // Cleanup
  ierr = CeedFree(&temp_buffer); CeedChk(ierr);

  // Debug
  CeedDebug256(ceed, 1, "---------- Ceed JiT ----------\n");
  CeedDebug256(ceed, 1, "Current source file: ");
  CeedDebug256(ceed, 255, "%s\n", source_file_path);
  CeedDebug256(ceed, 1, "Final buffer:\n");
  CeedDebug256(ceed, 255, "%s\n", *buffer);

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
int CeedLoadSourceToBuffer(Ceed ceed, const char *source_file_path,
                           char **buffer) {
  int ierr;

  // Initalize buffer
  ierr = CeedCalloc(1, buffer); CeedChk(ierr);

  // Load to initalized buffer
  ierr = CeedLoadSourceToInitalizedBuffer(ceed, source_file_path, buffer);
  CeedChk(ierr);

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get root of search path for installed files for JiT

  @param ceed                 A Ceed object for error handling
  @param[out] jit_source_root String for search path root

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedGetJitSourceRoot(Ceed ceed, const char **jit_source_root) {
  CeedDebug256(ceed, 1, "JiT Source Root: ");
  CeedDebug256(ceed, 255, "%s\n", ceed->jit_source_root);
  *jit_source_root = ceed->jit_source_root;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Find the relative filepath to an installed JiT file

  @param[in]  absolute_file_path Absolute path to installed JiT file
  @param[out] relative_file_path Relative path to installed JiT file

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedGetJitRelativePath(const char *absolute_file_path,
                           const char **relative_file_path) {
  *(relative_file_path) = strstr(absolute_file_path, "ceed-jit-source");

  if (!*relative_file_path)
    // LCOV_EXCL_START
    return CeedError(NULL, CEED_ERROR_MAJOR,
                     "Couldn't find relative path including "
                     "'ceed-jit-source' for: %s", absolute_file_path);
  // LCOV_EXCL_STOP

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
int CeedPathConcatenate(Ceed ceed, const char *base_file_path,
                        const char *relative_file_path, char **new_file_path) {
  int ierr;
  char *last_slash = strrchr(base_file_path, '/');
  size_t base_length = (last_slash - base_file_path + 1),
         relative_length = strlen(relative_file_path),
         new_file_path_length = base_length + relative_length + 1;

  ierr = CeedCalloc(new_file_path_length, new_file_path); CeedChk(ierr);
  memcpy(*new_file_path, base_file_path, base_length);
  memcpy(&((*new_file_path)[base_length]), relative_file_path, relative_length);

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Build an absolute filepath to an installed JiT file

  @param ceed                     A Ceed object for error handling
  @param[in]  relative_file_path  Relative path to installed JiT file
  @param[out] new_file_path       String buffer for absolute path to target file

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedGetInstalledJitPath(Ceed ceed, const char *relative_file_path,
                            char **jit_file_path) {
  int ierr;
  const char *jit_source_root;

  ierr = CeedGetJitSourceRoot(ceed, &jit_source_root); CeedChk(ierr);

  char *last_slash = strrchr(jit_source_root, '/');
  size_t base_length = (last_slash - jit_source_root + 1),
         relative_length = strlen(relative_file_path),
         new_file_path_length = base_length + relative_length + 1;

  ierr = CeedCalloc(new_file_path_length, jit_file_path); CeedChk(ierr);
  memcpy(*jit_file_path, jit_source_root, base_length);
  memcpy(&((*jit_file_path)[base_length]), relative_file_path, relative_length);

  return CEED_ERROR_SUCCESS;
}
