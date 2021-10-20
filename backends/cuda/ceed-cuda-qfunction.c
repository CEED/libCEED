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
#include <ceed/jit.h>
#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include "ceed-cuda.h"
#include "ceed-cuda-qfunction-load.h"

//------------------------------------------------------------------------------
// Apply QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionApply_Cuda(CeedQFunction qf, CeedInt Q,
                                   CeedVector *U, CeedVector *V) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChkBackend(ierr);

  // Build and compile kernel, if not done
  ierr = CeedCudaBuildQFunction(qf); CeedChkBackend(ierr);

  CeedQFunction_Cuda *data;
  ierr = CeedQFunctionGetData(qf, &data); CeedChkBackend(ierr);
  Ceed_Cuda *ceed_Cuda;
  ierr = CeedGetData(ceed, &ceed_Cuda); CeedChkBackend(ierr);
  CeedInt numinputfields, numoutputfields;
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChkBackend(ierr);

  // Read vectors
  for (CeedInt i = 0; i < numinputfields; i++) {
    ierr = CeedVectorGetArrayRead(U[i], CEED_MEM_DEVICE, &data->fields.inputs[i]);
    CeedChkBackend(ierr);
  }
  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedVectorGetArray(V[i], CEED_MEM_DEVICE, &data->fields.outputs[i]);
    CeedChkBackend(ierr);
  }

  // Get context data
  CeedQFunctionContext ctx;
  ierr = CeedQFunctionGetInnerContext(qf, &ctx); CeedChkBackend(ierr);
  if (ctx) {
    ierr = CeedQFunctionContextGetData(ctx, CEED_MEM_DEVICE, &data->d_c);
    CeedChkBackend(ierr);
  }

  // Run kernel
  void *args[] = {&data->d_c, (void *) &Q, &data->fields};
  ierr = CeedRunKernelAutoblockCuda(ceed, data->qFunction, Q, args);
  CeedChkBackend(ierr);

  // Restore vectors
  for (CeedInt i = 0; i < numinputfields; i++) {
    ierr = CeedVectorRestoreArrayRead(U[i], &data->fields.inputs[i]);
    CeedChkBackend(ierr);
  }
  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedVectorRestoreArray(V[i], &data->fields.outputs[i]);
    CeedChkBackend(ierr);
  }

  // Restore context
  if (ctx) {
    ierr = CeedQFunctionContextRestoreData(ctx, &data->d_c);
    CeedChkBackend(ierr);
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionDestroy_Cuda(CeedQFunction qf) {
  int ierr;
  CeedQFunction_Cuda *data;
  ierr = CeedQFunctionGetData(qf, &data); CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChkBackend(ierr);
  if  (data->module)
    CeedChk_Cu(ceed, cuModuleUnload(data->module));
  ierr = CeedFree(&data); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set User QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionSetCUDAUserFunction_Cuda(CeedQFunction qf,
    CUfunction f) {
  int ierr;
  CeedQFunction_Cuda *data;
  ierr = CeedQFunctionGetData(qf, &data); CeedChkBackend(ierr);
  data->qFunction = f;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Load QFunction source file
//------------------------------------------------------------------------------
static int CeedCudaLoadQFunction(CeedQFunction qf, char *c_src_file) {
  int ierr;
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);

  // Find source file
  char *cuda_file;
  ierr = CeedCalloc(CUDA_MAX_PATH, &cuda_file); CeedChkBackend(ierr);
  memcpy(cuda_file, c_src_file, strlen(c_src_file));
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
  char *buffer;
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
  ierr = CeedCalloc(lSize+1, &buffer); CeedChkBackend(ierr);

  // Copy the file into the buffer
  if(1 != fread(buffer, lSize, 1, fp)) {
    // LCOV_EXCL_START
    fclose(fp);
    ierr = CeedFree(&buffer); CeedChkBackend(ierr);
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Couldn't read the Cuda file for the QFunction.");
    // LCOV_EXCL_STOP
  }

  // Cleanup
  fclose(fp);
  ierr = CeedFree(&cuda_file); CeedChkBackend(ierr);

  // Save QFunction source
  CeedQFunction_Cuda *data;
  ierr = CeedQFunctionGetData(qf, &data); CeedChkBackend(ierr);
  data->qFunctionSource = buffer;
  data->qFunction = NULL;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create QFunction
//------------------------------------------------------------------------------
int CeedQFunctionCreate_Cuda(CeedQFunction qf) {
  int ierr;
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);
  CeedQFunction_Cuda *data;
  ierr = CeedCalloc(1, &data); CeedChkBackend(ierr);
  ierr = CeedQFunctionSetData(qf, data); CeedChkBackend(ierr);

  // Read source
  char *source;
  ierr = CeedQFunctionGetSourcePath(qf, &source); CeedChkBackend(ierr);
  // Empty source path indicates user must supply Q-Function
  if (source[0] != '\0') {
    const char *funname = strrchr(source, ':') + 1;
    data->qFunctionName = (char *)funname;
    const int filenamelen = funname - source;
    char filename[filenamelen];
    memcpy(filename, source, filenamelen - 1);
    filename[filenamelen - 1] = '\0';
    ierr = CeedLoadSourceToBuffer(ceed, &data->qFunctionSource, filename);
    CeedChkBackend(ierr);
  } else {
    data->module = NULL;
    data->qFunctionName = "";
    data->qFunctionSource = "";
    data->qFunction = NULL;
  }

  // Register backend functions
  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Apply",
                                CeedQFunctionApply_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy",
                                CeedQFunctionDestroy_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "SetCUDAUserFunction",
                                CeedQFunctionSetCUDAUserFunction_Cuda);
  CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
