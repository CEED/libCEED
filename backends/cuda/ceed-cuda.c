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
#include <stdarg.h>
#include "ceed-cuda.h"


int CeedCompileCuda(Ceed ceed, const char *source, CUmodule *module,
                    const CeedInt numopts, ...) {
  int ierr;
  cudaFree(0);//Make sure a Context exists for nvrtc
  nvrtcProgram prog;
  CeedChk_Nvrtc(ceed, nvrtcCreateProgram(&prog, source, NULL, 0, NULL, NULL));

  const int optslen = 32;
  const int optsextra = 3;
  const char *opts[numopts + optsextra];
  char buf[numopts][optslen];
  if (numopts>0) {
    va_list args;
    va_start(args, numopts);
    char *name;
    int val;
    for (int i = 0; i < numopts; i++) {
      name = va_arg(args, char *);
      val = va_arg(args, int);
      snprintf(&buf[i][0], optslen,"-D%s=%d", name, val);
      opts[i] = &buf[i][0];
    }
  }
  opts[numopts]     = "-DCeedScalar=double";
  opts[numopts + 1] = "-DCeedInt=int";
  struct cudaDeviceProp prop;
  Ceed_Cuda *ceed_data;
  Ceed delegate;
  CeedGetDelegate(ceed, &delegate);
  //We assume that the delegate is always the Cuda one
  if (delegate) {
    ierr = CeedGetData(delegate, (void *)&ceed_data); CeedChk(ierr);
  } else {
    ierr = CeedGetData(ceed, (void *)&ceed_data); CeedChk(ierr);
  }
  ierr = cudaGetDeviceProperties(&prop, ceed_data->deviceId);
  CeedChk_Cu(ceed, ierr);
  char buff[optslen];
  snprintf(buff, optslen,"-arch=compute_%d%d", prop.major, prop.minor);
  opts[numopts + 2] = buff;

  nvrtcResult result = nvrtcCompileProgram(prog, numopts + optsextra, opts);
  if (result != NVRTC_SUCCESS) {
    size_t logsize;
    CeedChk_Nvrtc(ceed, nvrtcGetProgramLogSize(prog, &logsize));
    char *log;
    ierr = CeedMalloc(logsize, &log); CeedChk(ierr);
    CeedChk_Nvrtc(ceed, nvrtcGetProgramLog(prog, log));
    return CeedError(ceed, (int)result, "%s\n%s", nvrtcGetErrorString(result), log);
  }

  size_t ptxsize;
  CeedChk_Nvrtc(ceed, nvrtcGetPTXSize(prog, &ptxsize));
  char *ptx;
  ierr = CeedMalloc(ptxsize, &ptx); CeedChk(ierr);
  CeedChk_Nvrtc(ceed, nvrtcGetPTX(prog, ptx));
  CeedChk_Nvrtc(ceed, nvrtcDestroyProgram(&prog));

  CeedChk_Cu(ceed, cuModuleLoadData(module, ptx));
  ierr = CeedFree(&ptx); CeedChk(ierr);

  return 0;
}

int CeedGetKernelCuda(Ceed ceed, CUmodule module, const char *name,
                      CUfunction *kernel) {
  CeedChk_Cu(ceed, cuModuleGetFunction(kernel, module, name));
  return 0;
}

int CeedRunKernelCuda(Ceed ceed, CUfunction kernel, const int gridSize,
                      const int blockSize, void **args) {
  CeedChk_Cu(ceed, cuLaunchKernel(kernel,
                                  gridSize, 1, 1,
                                  blockSize, 1, 1,
                                  0, NULL,
                                  args, NULL));
  return 0;
}

int CeedRunKernelDimCuda(Ceed ceed, CUfunction kernel, const int gridSize,
                         const int blockSizeX, const int blockSizeY,
                         const int blockSizeZ, void **args) {
  CeedChk_Cu(ceed, cuLaunchKernel(kernel,
                                  gridSize, 1, 1,
                                  blockSizeX, blockSizeY, blockSizeZ,
                                  0, NULL,
                                  args, NULL));
  return 0;
}

int CeedRunKernelDimSharedCuda(Ceed ceed, CUfunction kernel, const int gridSize,
                               const int blockSizeX, const int blockSizeY,
                               const int blockSizeZ, const int sharedMemSize,
                               void **args) {
  CeedChk_Cu(ceed, cuLaunchKernel(kernel,
                                  gridSize, 1, 1,
                                  blockSizeX, blockSizeY, blockSizeZ,
                                  sharedMemSize, NULL,
                                  args, NULL));
  return 0;
}

static int CeedGetPreferredMemType_Cuda(CeedMemType *type) {
  *type = CEED_MEM_DEVICE;
  return 0;
}

static int CeedInit_Cuda(const char *resource, Ceed ceed) {
  int ierr;
  const int nrc = 9; // number of characters in resource
  if (strncmp(resource, "/gpu/cuda/ref", nrc))
    return CeedError(ceed, 1, "Cuda backend cannot use resource: %s", resource);

  const int rlen = strlen(resource);
  const bool slash = (rlen>nrc) ? (resource[nrc] == '/') : false;
  const int deviceID = (slash && rlen > nrc + 1) ? atoi(&resource[nrc + 1]) : 0;

  int currentDeviceID;
  ierr = cudaGetDevice(&currentDeviceID); CeedChk_Cu(ceed,ierr);
  if (currentDeviceID!=deviceID) {
    ierr = cudaSetDevice(deviceID); CeedChk_Cu(ceed,ierr);
  }

  Ceed_Cuda *data;
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  data->deviceId = deviceID;

  struct cudaDeviceProp deviceProp;
  ierr = cudaGetDeviceProperties(&deviceProp, deviceID); CeedChk_Cu(ceed,ierr);

  data->optblocksize = deviceProp.maxThreadsPerBlock;

  ierr = CeedSetData(ceed,(void *)&data); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "GetPreferredMemType",
                                CeedGetPreferredMemType_Cuda); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "VectorCreate",
                                CeedVectorCreate_Cuda); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1",
                                CeedBasisCreateTensorH1_Cuda); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateH1",
                                CeedBasisCreateH1_Cuda); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "ElemRestrictionCreate",
                                CeedElemRestrictionCreate_Cuda); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed,
                                "ElemRestrictionCreateBlocked",
                                CeedElemRestrictionCreateBlocked_Cuda);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionCreate",
                                CeedQFunctionCreate_Cuda); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate",
                                CeedOperatorCreate_Cuda); CeedChk(ierr);
  return 0;
}

__attribute__((constructor))
static void Register(void) {
  CeedRegister("/gpu/cuda/ref", CeedInit_Cuda, 20);
}
