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

#include <string.h>
#include <stdio.h>
#include "ceed-cuda-gen.h"

//------------------------------------------------------------------------------
// Apply QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionApply_Cuda_gen(CeedQFunction qf, CeedInt Q,
                                       CeedVector *U, CeedVector *V) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
  return CeedError(ceed, 1, "Backend does not implement QFunctionApply");
}

//------------------------------------------------------------------------------
// Destroy QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionDestroy_Cuda_gen(CeedQFunction qf) {
  int ierr;
  CeedQFunction_Cuda_gen *data;
  ierr = CeedQFunctionGetData(qf, (void *)&data); CeedChk(ierr);
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
  ierr = cudaFree(data->d_c); CeedChk_Cu(ceed, ierr);
  ierr = CeedFree(&data->qFunctionSource); CeedChk(ierr);
  ierr = CeedFree(&data); CeedChk(ierr);
  return 0;
}

//------------------------------------------------------------------------------
// Load QFunction
//------------------------------------------------------------------------------
static int loadCudaFunction(CeedQFunction qf, char *c_src_file) {
  int ierr;
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);
  CeedQFunction_Cuda_gen *data;
  ierr = CeedQFunctionGetData(qf, (void *)&data); CeedChk(ierr);

  // Find source file
  char *cuda_file;
  ierr = CeedCalloc(CUDA_MAX_PATH, &cuda_file); CeedChk(ierr);
  memcpy(cuda_file, c_src_file, strlen(c_src_file));
  const char *last_dot = strrchr(cuda_file, '.');
  if (!last_dot)
    return CeedError(ceed, 1, "Cannot find file's extension!");
  const size_t cuda_path_len = last_dot - cuda_file;
  strncpy(&cuda_file[cuda_path_len], ".h", 3);

  // Open source file
  FILE *fp;
  long lSize;
  char *buffer;
  fp = fopen ( cuda_file, "rb" );
  if (!fp)
    CeedError(ceed, 1, "Couldn't open the Cuda file for the QFunction.");

  // Compute size of source file
  fseek(fp, 0L, SEEK_END);
  lSize = ftell(fp);
  rewind(fp);

  // Allocate memory for entire content
  ierr = CeedCalloc(lSize+1, &buffer); CeedChk(ierr);

  // Copy the file into the buffer
  if (1 != fread(buffer, lSize, 1, fp)) {
    fclose(fp);
    ierr = CeedFree(&buffer); CeedChk(ierr);
    CeedError(ceed, 1, "Couldn't read the Cuda file for the QFunction.");
  }

  // Append typedef and save source string
  // FIXME: the magic number 16 should be defined somewhere...
  char *fields_string =
    "typedef struct { const CeedScalar* inputs[16]; CeedScalar* outputs[16]; } Fields_Cuda_gen;";
  ierr = CeedMalloc(1 + strlen(fields_string) + strlen(buffer),
                    &data->qFunctionSource); CeedChk(ierr);
  memcpy(data->qFunctionSource, fields_string, strlen(fields_string));
  memcpy(data->qFunctionSource + strlen(fields_string), buffer,
         strlen(buffer) + 1);

  // Cleanup
  ierr = CeedFree(&buffer); CeedChk(ierr);
  fclose(fp);
  return 0;
}

//------------------------------------------------------------------------------
// Create QFunction
//------------------------------------------------------------------------------
int CeedQFunctionCreate_Cuda_gen(CeedQFunction qf) {
  int ierr;
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);
  CeedQFunction_Cuda_gen *data;
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  ierr = CeedQFunctionSetData(qf, (void *)&data); CeedChk(ierr);
  size_t ctxsize;
  ierr = CeedQFunctionGetContextSize(qf, &ctxsize); CeedChk(ierr);
  ierr = cudaMalloc(&data->d_c, ctxsize); CeedChk_Cu(ceed, ierr);

  char *source;
  ierr = CeedQFunctionGetSourcePath(qf, &source); CeedChk(ierr);
  const char *funname = strrchr(source, ':') + 1;
  data->qFunctionName = (char *)funname;
  const int filenamelen = funname - source;
  char filename[filenamelen];
  memcpy(filename, source, filenamelen - 1);
  filename[filenamelen - 1] = '\0';
  ierr = loadCudaFunction(qf, filename); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Apply",
                                CeedQFunctionApply_Cuda_gen); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy",
                                CeedQFunctionDestroy_Cuda_gen); CeedChk(ierr);
  return 0;
}
//------------------------------------------------------------------------------
