// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <ceed/jit.h>
#include <stdio.h>
#include <string.h>

static inline int CeedLoadSourceToInitalizedBuffer(Ceed ceed, char **buffer,
    const char *source_file_path) {
  int ierr;
  FILE *source_file;
  long file_size, file_offset = 0;
  char *temp_buffer;

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
      bool is_local_header = is_hash_include && next_e[2] == '"';
      if (is_local_header) {
        // ---- Build source path
        char *include_source_path;
        long root_length = strrchr(source_file_path, '/') - source_file_path;
        long include_file_name_len = strchr(&next_e[3], '"') - next_e - 3;
        ierr = CeedCalloc(root_length + include_file_name_len + 1,
                          &include_source_path); CeedChk(ierr);
        strncpy(include_source_path, source_file_path, root_length + 1);
        strncpy(&include_source_path[root_length + 1], &next_e[3],
                include_file_name_len);
        strncpy(&include_source_path[root_length + include_file_name_len + 1], "", 1);
        // ---- Recursive call to load source to buffer
        ierr = CeedLoadSourceToInitalizedBuffer(ceed, buffer, include_source_path);
        CeedChk(ierr);
      }
      file_offset = strchr(first_hash, '\n') - temp_buffer + 1;
    }
    // -- Next hash
    first_hash = strchr(&first_hash[1], '#');
  }
  // Copy rest of source file into buffer
  long current_size = strlen(*buffer);
  long copy_size = strlen(&temp_buffer[file_offset]);
  ierr = CeedRealloc(current_size + copy_size + 1, buffer); CeedChk(ierr);
  strncpy(&(*buffer)[current_size], "\n", 2);
  strncpy(&(*buffer)[current_size + 1], &temp_buffer[file_offset], copy_size);
  strncpy(&(*buffer)[current_size + copy_size + 1], "", 1);

  // Cleanup
  fclose(source_file);
  ierr = CeedFree(&temp_buffer); CeedChk(ierr);

  return CEED_ERROR_SUCCESS;
}

int CeedLoadSourceToBuffer(Ceed ceed, char **buffer,
                           const char *source_file_path) {
  int ierr;

  // Initalize buffer
  ierr = CeedCalloc(1, buffer); CeedChk(ierr);

  // Load to initalized buffer
  ierr = CeedLoadSourceToInitalizedBuffer(ceed, buffer, source_file_path);
  CeedChk(ierr);

  return CEED_ERROR_SUCCESS;
}
