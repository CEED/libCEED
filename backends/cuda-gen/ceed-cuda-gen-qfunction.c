// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
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
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include "ceed-cuda-gen.h"
#include "../cuda/ceed-cuda.h"

//------------------------------------------------------------------------------
// Apply QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionApply_Cuda_gen(CeedQFunction qf, CeedInt Q,
                                       CeedVector *U, CeedVector *V) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChkBackend(ierr);
  return CeedError(ceed, CEED_ERROR_BACKEND,
                   "Backend does not implement QFunctionApply");
}

//------------------------------------------------------------------------------
// Destroy QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionDestroy_Cuda_gen(CeedQFunction qf) {
  int ierr;
  CeedQFunction_Cuda_gen *data;
  ierr = CeedQFunctionGetData(qf, &data); CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChkBackend(ierr);
  ierr = cudaFree(data->d_c); CeedChk_Cu(ceed, ierr);
  ierr = CeedFree(&data->qFunctionSource); CeedChkBackend(ierr);
  ierr = CeedFree(&data); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Load QFunction
//------------------------------------------------------------------------------
static int loadCudaFunction(CeedQFunction qf, char *c_src_file) {
  int ierr;
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);
  CeedQFunction_Cuda_gen *data;
  ierr = CeedQFunctionGetData(qf, &data); CeedChkBackend(ierr);

  // Count number of source file(s)
  CeedInt num_files = 1;
  CeedInt file_offset = 0;
  for (CeedInt i = 0; i < (CeedInt) strlen(c_src_file); i++) {
    if (c_src_file[i] == ';')
      num_files++;
  }
  file_offset = 0;

  // Loop over all source file(s)
  char *buffer;
  CeedInt buffer_offset = 0;
  ierr = CeedCalloc(buffer_offset + 1, &buffer); CeedChkBackend(ierr);
  for (CeedInt i = 0; i < num_files; i++) {
    // Find current source file path
    char *cuda_file;
    ierr = CeedCalloc(CUDA_MAX_PATH, &cuda_file); CeedChkBackend(ierr);
    CeedInt file_path_len = strlen(c_src_file) - file_offset;
    const char *semicolon_loc = semicolon_loc = strchr(&c_src_file[file_offset],
                                ';');
    if (semicolon_loc)
      file_path_len = semicolon_loc - &c_src_file[file_offset];
    memcpy(cuda_file, &c_src_file[file_offset], file_path_len);
    const char *last_dot = strrchr(cuda_file, '.');
    if (!last_dot)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_BACKEND, "Cannot find file's extension!");
    // LCOV_EXCL_STOP
    const size_t cuda_path_len = last_dot - cuda_file;
    strncpy(&cuda_file[cuda_path_len], ".h", 3);

    // Open source file
    FILE *fp;
    long lSize;
    fp = fopen (cuda_file, "rb");
    if (!fp)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "Couldn't open the Cuda file for the QFunction.");
    // LCOV_EXCL_STOP

    // Compute size of source
    fseek(fp, 0L, SEEK_END);
    lSize = ftell(fp);
    rewind(fp);

    // Allocate memory for entire content
    ierr = CeedRealloc(buffer_offset+lSize+1, &buffer); CeedChkBackend(ierr);

    // Copy the file into the buffer
    if (1 != fread(&buffer[buffer_offset], lSize, 1, fp)) {
      // LCOV_EXCL_START
      fclose(fp);
      ierr = CeedFree(&buffer); CeedChkBackend(ierr);
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "Couldn't read the Cuda file for the QFunction.");
      // LCOV_EXCL_STOP
    }
    strncpy(&buffer[buffer_offset + lSize], "\n", 2);

    // Cleanup
    fclose(fp);
    ierr = CeedFree(&cuda_file); CeedChkBackend(ierr);

    // Update offsets
    file_offset += file_path_len + 1;
    buffer_offset += lSize + 1;
  }

  // Append typedef and save source string
  // FIXME: the magic number 16 should be defined somewhere...
  char *fields_string =
    "typedef struct { const CeedScalar* inputs[16]; CeedScalar* outputs[16]; } Fields_Cuda_gen;";
  ierr = CeedMalloc(1 + strlen(fields_string) + strlen(buffer),
                    &data->qFunctionSource); CeedChkBackend(ierr);
  memcpy(data->qFunctionSource, fields_string, strlen(fields_string));
  memcpy(data->qFunctionSource + strlen(fields_string), buffer,
         strlen(buffer) + 1);

  // Cleanup
  ierr = CeedFree(&buffer); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create QFunction
//------------------------------------------------------------------------------
int CeedQFunctionCreate_Cuda_gen(CeedQFunction qf) {
  int ierr;
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);
  CeedQFunction_Cuda_gen *data;
  ierr = CeedCalloc(1, &data); CeedChkBackend(ierr);
  ierr = CeedQFunctionSetData(qf, data); CeedChkBackend(ierr);

  char *source;
  ierr = CeedQFunctionGetSourcePath(qf, &source); CeedChkBackend(ierr);
  const char *funname = strrchr(source, ':') + 1;
  data->qFunctionName = (char *)funname;
  const int filenamelen = funname - source;
  char filename[filenamelen];
  memcpy(filename, source, filenamelen - 1);
  filename[filenamelen - 1] = '\0';
  ierr = loadCudaFunction(qf, filename); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Apply",
                                CeedQFunctionApply_Cuda_gen); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy",
                                CeedQFunctionDestroy_Cuda_gen); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
