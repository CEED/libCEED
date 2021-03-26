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

#ifndef _ceed_hip_compile_h
#define _ceed_hip_compile_h

#include <ceed/ceed.h>
#include <hip/hip_runtime.h>

CEED_INTERN int CeedCompileHip(Ceed ceed, const char *source,
                               hipModule_t *module,
                               const CeedInt numopts, ...);

CEED_INTERN int CeedGetKernelHip(Ceed ceed, hipModule_t module,
                                 const char *name,
                                 hipFunction_t *kernel);

CEED_INTERN int CeedRunKernelHip(Ceed ceed, hipFunction_t kernel,
                                 const int gridSize,
                                 const int blockSize, void **args);

CEED_INTERN int CeedRunKernelDimHip(Ceed ceed, hipFunction_t kernel,
                                    const int gridSize,
                                    const int blockSizeX, const int blockSizeY,
                                    const int blockSizeZ, void **args);

CEED_INTERN int CeedRunKernelDimSharedHip(Ceed ceed, hipFunction_t kernel,
    const int gridSize, const int blockSizeX,
    const int blockSizeY, const int blockSizeZ,
    const int sharedMemSize, void **args);

#endif // _ceed_hip_compile_h
