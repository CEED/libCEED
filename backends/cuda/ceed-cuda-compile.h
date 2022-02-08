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

#ifndef _ceed_cuda_compile_h
#define _ceed_cuda_compile_h

#include <ceed/ceed.h>
#include <cuda.h>
#include <nvrtc.h>

static inline CeedInt CeedDivUpInt(CeedInt numerator, CeedInt denominator) {
  return (numerator + denominator - 1) / denominator;
}

CEED_INTERN int CeedCompileCuda(Ceed ceed, const char *source, CUmodule *module,
                                const CeedInt num_opts, ...);

CEED_INTERN int CeedGetKernelCuda(Ceed ceed, CUmodule module, const char *name,
                                  CUfunction *kernel);

CEED_INTERN int CeedRunKernelCuda(Ceed ceed, CUfunction kernel,
                                  const int grid_size,
                                  const int block_size, void **args);

CEED_INTERN int CeedRunKernelAutoblockCuda(Ceed ceed, CUfunction kernel,
    size_t size, void **args);

CEED_INTERN int CeedRunKernelDimCuda(Ceed ceed, CUfunction kernel,
                                     const int grid_size,
                                     const int block_size_x, const int block_size_y,
                                     const int block_size_z, void **args);

CEED_INTERN int CeedRunKernelDimSharedCuda(Ceed ceed, CUfunction kernel,
    const int grid_size, const int block_size_x, const int block_size_y,
    const int block_size_z, const int shared_mem_size, void **args);

#endif // _ceed_cuda_compile_h
