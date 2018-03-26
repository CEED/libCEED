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
#include "ceed-occa.h"

// *****************************************************************************
// * OCCA modes, should be dynamic like OCCA_DEVICE_ID/PLATFORM_ID
// *****************************************************************************
static const char *occaCPU = "mode: 'Serial'";
static const char *occaOMP = "mode: 'OpenMP'";
static const char *occaGPU = "mode: 'CUDA', deviceID: 0";
static const char *occaOCL = "mode: 'OpenCL', platformID: 0, deviceID: 0";
extern void occaSetVerboseCompilation(const int value);

// *****************************************************************************
// * CeedError_Occa
// *****************************************************************************
static int CeedError_Occa(Ceed ceed,
                          const char *file, int line,
                          const char *func, int code,
                          const char *format, va_list args) {
  fprintf(stderr,"\033[31;1m");
  fprintf(stderr, "CEED-OCCA error @ %s:%d %s\n", file, line, func);
  vfprintf(stderr, format, args);
  fprintf(stderr,"\033[m\n");
  fflush(stderr);
  abort();
  return code;
}

// *****************************************************************************
// * CeedDestroy_Occa
// *****************************************************************************
static int CeedDestroy_Occa(Ceed ceed) {
  int ierr;
  Ceed_Occa *data=ceed->data;
  CeedDebug("\033[1m[CeedDestroy]");
  occaFree(data->device);
  ierr = CeedFree(&data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * INIT
// *****************************************************************************
static int CeedInit_Occa(const char *resource, Ceed ceed) {
  int ierr;
  Ceed_Occa *data;
  const bool cpu = !strcmp(resource, "/cpu/occa");
  const bool omp = !strcmp(resource, "/omp/occa");
  const bool ocl = !strcmp(resource, "/ocl/occa");
  const bool gpu = !strcmp(resource, "/gpu/occa");
  CeedDebug("\033[1m[CeedInit] resource='%s'", resource);
  if (!cpu && !omp && !ocl && !gpu)
    return CeedError(ceed, 1, "OCCA backend cannot use resource: %s", resource);
  ceed->Error = CeedError_Occa;
  ceed->Destroy = CeedDestroy_Occa;
  ceed->VecCreate = CeedVectorCreate_Occa;
  ceed->ElemRestrictionCreate = CeedElemRestrictionCreate_Occa;
  ceed->BasisCreateTensorH1 = CeedBasisCreateTensorH1_Occa;
  ceed->QFunctionCreate = CeedQFunctionCreate_Occa;
  ceed->OperatorCreate = CeedOperatorCreate_Occa;
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  ceed->data = data;
#ifdef CDEBUG
  occaPropertiesSet(occaSettings(),
                    "verbose-compilation",
                    occaBool(1));
#endif
  const char *mode = gpu?occaGPU : omp?occaOMP : ocl ? occaOCL : occaCPU;
  // Now creating OCCA device
  data->device = occaCreateDevice(occaString(mode));
  const char *deviceMode = occaDeviceMode(data->device);
  CeedDebug("\033[1m[CeedInit] deviceMode='%s'", deviceMode);
  if (cpu && strcmp(occaDeviceMode(data->device), "Serial"))
    return CeedError(ceed,1, "OCCA backend failed to use Serial resource");
  if (omp && strcmp(occaDeviceMode(data->device), "OpenMP"))
    return CeedError(ceed,1, "OCCA backend failed to use OpenMP resource");
  if (gpu && strcmp(occaDeviceMode(data->device), "CUDA"))
    return CeedError(ceed,1, "OCCA backend failed to use CUDA resource");
  if (ocl && strcmp(occaDeviceMode(data->device), "OpenCL"))
    return CeedError(ceed,1, "OCCA backend failed to use OpenCL resource");
  return 0;
}

// *****************************************************************************
// * REGISTER
// *****************************************************************************
__attribute__((constructor))
static void Register(void) {
  CeedDebug("\033[1m[Register] occa: cpu, gpu, omp, ocl");
  CeedRegister("/cpu/occa", CeedInit_Occa);
  CeedRegister("/gpu/occa", CeedInit_Occa);
  CeedRegister("/omp/occa", CeedInit_Occa);
  CeedRegister("/ocl/occa", CeedInit_Occa);
}
