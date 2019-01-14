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
#include "../include/ceed.h"
#include <ceed-impl.h>
#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_MAX_PATH 256

#define CeedChk_Nvrtc(ceed, x) \
do { \
  nvrtcResult result = x; \
  if (result != NVRTC_SUCCESS) \
    return CeedError((ceed), result, nvrtcGetErrorString(result)); \
} while (0)

#define CeedChk_Cu(ceed, x) \
do { \
  CUresult result = x; \
  if (result != CUDA_SUCCESS) { \
    const char *msg; \
    cuGetErrorName(result, &msg); \
    return CeedError((ceed), result, msg); \
  } \
} while (0)

#define QUOTE(...) #__VA_ARGS__

typedef struct {
  CUmodule module;
  CUfunction interp;
  CUfunction grad;
  CUfunction weight;
  CeedScalar *d_interp1d;
  CeedScalar *d_grad1d;
  CeedScalar *d_qweight1d;
  CeedScalar *c_B;
  CeedScalar *c_G;
} CeedBasis_Cuda_reg;

typedef struct {
} Ceed_Cuda_reg;

CEED_INTERN int CeedBasisCreateTensorH1_Cuda_reg(CeedInt dim, CeedInt P1d,
    CeedInt Q1d,
    const CeedScalar *interp1d,
    const CeedScalar *grad1d,
    const CeedScalar *qref1d,
    const CeedScalar *qweight1d,
    CeedBasis basis);

CEED_INTERN int CeedBasisCreateH1_Cuda_reg(CeedElemTopology, CeedInt, CeedInt,
                                       CeedInt, const CeedScalar *,
                                       const CeedScalar *, const CeedScalar *, const CeedScalar *, CeedBasis);
