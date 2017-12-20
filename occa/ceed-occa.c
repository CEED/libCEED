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
#include "ceed-occa.h"

// *****************************************************************************
// * OCCA stuff
// *****************************************************************************
occaDevice device;
//const int occaTileSize = 32;
const char *occaDeviceInfo = "mode: 'Serial'";
const bool occaVerboseCompilationMode = true;
extern void occaSetVerboseCompilation(const int value);

// *****************************************************************************
// * CeedErrorOcca
// *****************************************************************************
static int CeedErrorOcca(Ceed ceed,
                         const char *file, int line,
                         const char *func, int code,
                         const char* format, va_list args){
  fprintf(stderr,"\033[31;1m");
  vfprintf(stderr, format, args);
  fprintf(stderr,"\033[m\n");
  fflush(stderr);
  return 0;
}

// *****************************************************************************
// * CeedDestroyOcca
// *****************************************************************************
static int CeedDestroyOcca(Ceed ceed){
  dbg("\033[1m[CeedDestroy][Occa]");
  occaDeviceFree(device);
  return 0;
}

// *****************************************************************************
// * INIT
// *****************************************************************************
static int CeedInitOcca(const char* resource, Ceed ceed) {
  dbg("\033[1m[CeedInit][Occa] resource='%s'", resource);
  if (strcmp(resource, "/cpu/occa"))
    return CeedError(ceed, 1, "Ref backend cannot use resource: %s", resource);
  ceed->Error = CeedErrorOcca;
  ceed->Destroy = CeedDestroyOcca;
  ceed->VecCreate = CeedVectorCreateOcca;
  ceed->ElemRestrictionCreate = CeedElemRestrictionCreateOcca;
  ceed->BasisCreateTensorH1 = CeedBasisCreateTensorH1Occa;
  ceed->QFunctionCreate = CeedQFunctionCreateOcca;
  ceed->OperatorCreate = CeedOperatorCreateOcca;
  // Now creating OCCA device
  occaPrintModeInfo();
  occaSetVerboseCompilation(occaVerboseCompilationMode);
  device = occaCreateDevice(occaString(occaDeviceInfo));
  return 0;
} 


// *****************************************************************************
// * REGISTER
// *****************************************************************************
__attribute__((constructor))
static void Register(void) {
  dbg("\033[1m[Register] /cpu/occa");
  CeedRegister("/cpu/occa", CeedInitOcca);
}
