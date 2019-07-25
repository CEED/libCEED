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

#include <ceed-backend.h>
#include <string.h>
#include <stdio.h>
#include "../cuda/ceed-cuda.h"
#include "ceed-cuda-gen.h"

static int CeedQFunctionApply_Cuda_gen(CeedQFunction qf, CeedInt Q,
                                       CeedVector *U, CeedVector *V) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
  return CeedError(ceed, 1, "Backend does not implement QFunctionApply");
}

static int CeedQFunctionDestroy_Cuda_gen(CeedQFunction qf) {
  int ierr;
  CeedQFunction_Cuda_gen *data;
  ierr = CeedQFunctionGetData(qf, (void *)&data); CeedChk(ierr);
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);

  ierr = cudaFree(data->d_c); CeedChk_Cu(ceed, ierr);

  ierr = CeedFree(&data); CeedChk(ierr);

  return 0;
}

static int loadCudaFunction(CeedQFunction qf, char *c_src_file) {
  int ierr;
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);
  char *cuda_file;
  ierr = CeedCalloc(CUDA_MAX_PATH, &cuda_file); CeedChk(ierr);
  memcpy(cuda_file, c_src_file, strlen(c_src_file));
  const char *last_dot = strrchr(cuda_file, '.');
  if (!last_dot)
    return CeedError(ceed, 1, "Cannot find file's extension!");
  const size_t cuda_path_len = last_dot - cuda_file;
  strcpy(&cuda_file[cuda_path_len], ".qf");
  //*******************
  FILE *fp;
  long lSize;
  char *buffer;

  fp = fopen ( cuda_file, "rb" );
  if( !fp ) CeedError(ceed, 1, "Couldn't open the Cuda file for the QFunction.");

  fseek( fp, 0L, SEEK_END);
  lSize = ftell( fp );
  rewind( fp );

  /* allocate memory for entire content */
  ierr = CeedCalloc( lSize+1, &buffer ); CeedChk(ierr);

  /* copy the file into the buffer */
  if( 1!=fread( buffer, lSize, 1, fp) ) {
    fclose(fp);
    CeedFree(&buffer);
    CeedError(ceed, 1, "Couldn't read the Cuda file for the QFunction.");
  }

  //FIXME: the magic number 16 should be defined somewhere...
  char *fields_string =
    "typedef struct { const CeedScalar* inputs[16]; CeedScalar* outputs[16]; } Fields_Cuda_gen;";
  char *source = (char *) malloc(1 + strlen(fields_string)+ strlen(buffer) );
  strcpy(source, fields_string);
  strcat(source, buffer);

  //********************
  CeedQFunction_Cuda_gen *data;
  ierr = CeedQFunctionGetData(qf, (void *)&data); CeedChk(ierr);
  data->qFunctionSource = buffer;

  //********************
  fclose(fp);

  return 0;
}

int CeedQFunctionCreate_Cuda_gen(CeedQFunction qf) {
  int ierr;
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);
  CeedQFunction_Cuda_gen *data;
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  ierr = CeedQFunctionSetData(qf, (void *)&data); CeedChk(ierr);
  // CeedInt numinputfields, numoutputfields;
  // ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  size_t ctxsize;
  ierr = CeedQFunctionGetContextSize(qf, &ctxsize); CeedChk(ierr);
  ierr = cudaMalloc(&data->d_c, ctxsize); CeedChk_Cu(ceed, ierr);

  char *focca;
  ierr = CeedQFunctionGetFOCCA(qf, &focca); CeedChk(ierr);
  const char *funname = strrchr(focca, ':') + 1;
  data->qFunctionName = (char *)funname;
  const int filenamelen = funname - focca;
  char filename[filenamelen];
  memcpy(filename, focca, filenamelen - 1);
  filename[filenamelen - 1] = '\0';
  ierr = loadCudaFunction(qf, filename); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Apply",
                                CeedQFunctionApply_Cuda_gen); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy",
                                CeedQFunctionDestroy_Cuda_gen); CeedChk(ierr);
  return 0;
}
