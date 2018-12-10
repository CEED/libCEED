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

#include <ceed-impl.h>
#include <string.h>
#include <stdio.h>
#include "ceed-cuda.h"

// int CeedQFunctionApplyElems_Cuda(CeedQFunction qf, const CeedInt Q,
//     const CeedVector *const u, const CeedVector* v) {
//   int ierr;
//   Ceed ceed;
//   ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
//   Ceed_Cuda* ceed_Cuda;
//   ierr = CeedGetData(ceed, (void*)&ceed_Cuda); CeedChk(ierr);
//   CeedQFunction_Cuda *data;
//   ierr = CeedQFunctionGetData(qf, (void*)&data); CeedChk(ierr);
//   const int blocksize = ceed_Cuda->optblocksize;

//   if (qf->ctxsize > 0) {
//     ierr = cudaMemcpy(data->d_c, qf->ctx, qf->ctxsize, cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);
//   }

//   const CeedScalar *h_u[qf->numinputfields];
//   for (CeedInt i = 0; i < qf->numinputfields; i++) {
//     CeedVectorGetArrayRead(u[i], CEED_MEM_DEVICE, h_u + i);
//   }
//   ierr = cudaMemcpy((void**)data->d_u, h_u, qf->numinputfields * sizeof(CeedScalar*), cudaMemcpyHostToDevice);
//   CeedChk_Cu(ceed, ierr);

//   CeedScalar *h_v[qf->numoutputfields];
//   for (CeedInt i = 0; i < qf->numoutputfields; i++) {
//     CeedVectorGetArray(v[i], CEED_MEM_DEVICE, h_v + i);
//   }
//   ierr = cudaMemcpy((void*)data->d_v, h_v, qf->numoutputfields * sizeof(CeedScalar*), cudaMemcpyDeviceToHost);
//   CeedChk_Cu(ceed, ierr);

//   void *args[] = {&data->d_c, (void*)&Q, &data->d_u, &data->d_v};
//   ierr = run_kernel(qf->ceed, data->callback, CeedDivUpInt(Q, blocksize), blocksize, args);
//   CeedChk(ierr);


//   if (qf->ctxsize > 0) {
//     ierr = cudaMemcpy(qf->ctx, data->d_c, qf->ctxsize, cudaMemcpyDeviceToHost); CeedChk_Cu(ceed, ierr);
//   }

//   return 0;
// }

static int CeedQFunctionApply_Cuda(CeedQFunction qf, CeedInt Q,
                                   CeedVector *U, CeedVector *V) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
  CeedQFunction_Cuda *data;
  ierr = CeedQFunctionGetData(qf, (void*)&data); CeedChk(ierr);
  Ceed_Cuda *ceed_Cuda;
  ierr = CeedGetData(ceed, (void*)&ceed_Cuda);
  CeedInt numinputfields, numoutputfields;
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChk(ierr);
  const int blocksize = ceed_Cuda->optblocksize;

  for (CeedInt i = 0; i < numinputfields; i++) {
    ierr = CeedVectorGetArrayRead(U[i], CEED_MEM_DEVICE, data->d_u + i);
    CeedChk(ierr);
  }

  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedVectorGetArray(V[i], CEED_MEM_DEVICE, data->d_v + i);
    CeedChk(ierr);
  }

  // void *args[] = {&data->d_c, (void*)&Q, &data->d_u, &data->d_v};
  void* ctx;
  ierr = CeedQFunctionGetContext(qf, &ctx); CeedChk(ierr);
  void *args[] = {&ctx, (void*)&Q, &data->d_u, &data->d_v};
  ierr = run_kernel(ceed, data->qFunction, CeedDivUpInt(Q, blocksize), blocksize, args);
  CeedChk(ierr);

  for (CeedInt i = 0; i < numinputfields; i++) {
    ierr = CeedVectorRestoreArrayRead(U[i], data->d_u + i);
    CeedChk(ierr);
  }

  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedVectorRestoreArray(V[i], data->d_v + i);
    CeedChk(ierr);
  }

  return 0;
}

static int CeedQFunctionDestroy_Cuda(CeedQFunction qf) {
  int ierr;
  CeedQFunction_Cuda *data;
  ierr = CeedQFunctionGetData(qf, (void*)&data); CeedChk(ierr);
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);

  CeedChk_Cu(ceed, cuModuleUnload(data->module)); 
  ierr = cudaFree((void*)data->d_u); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree((void*)data->d_v); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree(data->d_c); CeedChk_Cu(ceed, ierr);

  ierr = CeedFree(&data); CeedChk(ierr);

  return 0;
}

static int loadCudaFunction(CeedQFunction qf, char* c_src_file) {
  int ierr;
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);
  char* cuda_file;
  ierr = CeedCalloc(CUDA_MAX_PATH, &cuda_file); CeedChk(ierr);
  memcpy(cuda_file, c_src_file, strlen(c_src_file));
  const char *last_dot = strrchr(cuda_file, '.');
  if (!last_dot)
    return CeedError(ceed, 1, "Cannot find file's extension!");
  const size_t cuda_path_len = last_dot - cuda_file;
  strcpy(&cuda_file[cuda_path_len], ".cu");
  printf("cudafile: %s\n", cuda_file);
  //*******************
  FILE *fp;
  long lSize;
  char *buffer;

  fp = fopen ( cuda_file , "rb" );
  if( !fp ) perror(cuda_file),exit(1);

  fseek( fp , 0L , SEEK_END);
  lSize = ftell( fp );
  rewind( fp );

  /* allocate memory for entire content */
  ierr = CeedMalloc( lSize+1, &buffer ); CeedChk(ierr);

  /* copy the file into the buffer */
  if( 1!=fread( buffer , lSize, 1 , fp) )
    fclose(fp),free(buffer),fputs("entire read fails",stderr),exit(1);

  //********************
  CeedQFunction_Cuda *data;
  ierr = CeedQFunctionGetData(qf, (void*)&data); CeedChk(ierr);
  printf("buffer: %s\n", buffer);
  ierr = compile(ceed, buffer, &data->module, 0); CeedChk(ierr);
  ierr = get_kernel(ceed, data->module, data->qFunctionName, &data->qFunction); CeedChk(ierr);

  //********************
  fclose(fp);
  ierr = CeedFree(&buffer); CeedChk(ierr);

  return 0;
}

int CeedQFunctionCreate_Cuda(CeedQFunction qf) {
  int ierr;
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed); 
  CeedQFunction_Cuda *data;
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  CeedInt numinputfields, numoutputfields;
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  ierr = cudaMalloc((void**)&data->d_u, numinputfields * sizeof(CeedScalar*)); CeedChk_Cu(ceed, ierr);
  ierr = cudaMalloc((void**)&data->d_v, numoutputfields * sizeof(CeedScalar*)); CeedChk_Cu(ceed, ierr);
  size_t ctxsize;
  ierr = CeedQFunctionGetContextSize(qf, &ctxsize); CeedChk(ierr);
  ierr = cudaMalloc(&data->d_c, ctxsize); CeedChk_Cu(ceed, ierr);
  
  const char *funname = strrchr(qf->focca, ':') + 1;
  data->qFunctionName = (char*)funname;
  printf("funname: %s\n", funname);
  // Including final NUL char
  const int filenamelen = funname - qf->focca;
  char filename[filenamelen];
  memcpy(filename, qf->focca, filenamelen - 1);
  filename[filenamelen - 1] = '\0';
  printf("filename: %s\n", filename);
  ierr = loadCudaFunction(qf, filename); CeedChk(ierr);
  // FILE *file = fopen(filename, "r");
  // if (!file) {
  //   return CeedError(qf->ceed, 1, "The file %s cannot be read", filename);
  // }

  // fseek(file, 0, SEEK_END);
  // const int contentslen = ftell(file);
  // fseek (file, 0, SEEK_SET);
  // char *contents;
  // ierr = CeedCalloc(contentslen + 1, &contents); CeedChk(ierr);
  // fread(contents, 1, contentslen, file);

  // printf("file: %s\n", file);

  // ierr = compile(ceed, contents, &data->module, 0); CeedChk(ierr);
  // ierr = get_kernel(ceed, data->module, funname, &data->qFunction); CeedChk(ierr);
  // ierr = CeedFree(&contents); CeedChk(ierr);

  // ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
  // ierr = CeedQFunctionSetData(qf, (void*)&data); CeedChk(ierr);
  // // File reading
  // char *fcuda;
  // ierr = CeedQFunctionGetFOCCA(qf, &fcuda); CeedChk(ierr);
  // const char *last_colon = strrchr(fcuda,':');
  // const char *last_dot = strrchr(fcuda,'.');
  // if (!last_colon)
  //   return CeedError(ceed, 1, "Can not find ':' in fcuda field!");
  // if (!last_dot)
  //   return CeedError(ceed, 1, "Can not find '.' in fcuda field!");
  // // get the function name
  // data->qFunctionName = last_colon+1;
  // // extract file base name
  // const char *last_slash_pos = strrchr(fcuda,'/');
  // // if no slash has been found, revert to fcuda field
  // const char *last_slash = last_slash_pos?last_slash_pos+1:fcuda;
  // // extract c_src_file & cuda_base_name
  // char *c_src_file, *cuda_base_name;
  // ierr = CeedCalloc(CUDA_MAX_PATH,&cuda_base_name); CeedChk(ierr);
  // ierr = CeedCalloc(CUDA_MAX_PATH,&c_src_file); CeedChk(ierr);
  // memcpy(cuda_base_name,last_slash,last_dot-last_slash);
  // memcpy(c_src_file,fcuda,last_colon-fcuda);
  // Now fetch Cuda filename ****************************************************
  // TODO do something with NVRTC Johann's style
  // ierr = loadCudaFunction(qf, c_src_file, cuda_base_name);
  // CeedChk(ierr);
  // free **********************************************************************
  // ierr = CeedFree(&cuda_base_name); CeedChk(ierr);
  // ierr = CeedFree(&c_src_file); CeedChk(ierr);

  //

  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Apply",
                                CeedQFunctionApply_Cuda); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy",
                                CeedQFunctionDestroy_Cuda); CeedChk(ierr);
  return 0;
}
