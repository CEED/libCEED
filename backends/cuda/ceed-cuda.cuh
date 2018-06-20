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

#include <algorithm>
#include <ceed-impl.h>
#include <stdbool.h>

typedef struct {
  CeedScalar *h_array;
  CeedScalar *used_pointer;
  CeedScalar *d_array;
} CeedVector_Cuda;

typedef struct {
  CeedInt *d_indices;
} CeedElemRestriction_Cuda;

typedef struct {
  CeedVector etmp;
  CeedVector qdata;
  CeedVector BEu, BEv;
} CeedOperator_Cuda;

typedef struct {
  bool ready;
  int nc, dim, nelem, elemsize;
  void *d_q;
  CeedScalar **d_u;
  CeedScalar **d_v;
  void *d_c;
} CeedQFunction_Cuda;

typedef struct {
  CeedElemRestriction er;
  CeedScalar *d_qweight1d;
  CeedScalar *d_interp1d;
  CeedScalar *d_grad1d;
} CeedBasis_Cuda;

typedef struct {
  int deviceID;
  int x_thread_limit, y_thread_limit, z_thread_limit;
} Ceed_Cuda;

static int divup(int a, int b) {
  return (a + b - 1) / b;
}


template <typename CudaFunc, typename... Args>
void run1d(const Ceed_Cuda* ceed, CudaFunc f, const CeedInt x, Args... args) {
  int blocks, threads;
  cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, f);

  int actThreads = std::min({x, threads, ceed->x_thread_limit});
  int actBlocks = std::min(divup(x, actThreads), blocks);
  f<<<actBlocks, actThreads>>>(x, args...);
}

template <typename CudaFunc, typename... Args>
void run2d(const Ceed_Cuda* ceed, CudaFunc f, const CeedInt x, const CeedInt y, Args... args) {
  int blocks, threads;
  cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, f);

  int actThreadsX = std::min({x, threads, ceed->x_thread_limit});
  int actThreadsY = std::min({y, std::max(threads / actThreadsX, 1), ceed->y_thread_limit});

  int actBlocksX = std::min(divup(x, actThreadsX), blocks);
  int actBlocksY = std::min(divup(y, actThreadsY), divup(blocks, actBlocksX));
  f<<<dim3(actBlocksX, actBlocksY), dim3(actThreadsX, actThreadsY)>>>(x, y, args...);
}

template <typename CudaFunc, typename... Args>
void run3d(const Ceed_Cuda* ceed, CudaFunc f, const CeedInt x, const CeedInt y, const CeedInt z, Args... args) {
  int blocks, threads;
  cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, f);

  int actThreadsX = std::min({x, threads, ceed->x_thread_limit});
  int actThreadsY = std::min({y, std::max(threads / actThreadsX, 1), ceed->y_thread_limit});
  int actThreadsZ = std::min({z, std::max(threads / (actThreadsX * actThreadsY), 1), ceed->z_thread_limit});

  int actBlocksX = std::min(divup(x, actThreadsX), blocks);
  int actBlocksY = std::min(divup(y, actThreadsY), divup(blocks, actBlocksX));
  int actBlocksZ = std::min(divup(z, actThreadsZ), divup(blocks, actBlocksX * actBlocksY));
  f<<<dim3(actBlocksX, actBlocksY, actBlocksZ), dim3(actThreadsX, actThreadsY, actThreadsZ)>>>(x, y, z, args...);
}

CEED_INTERN int CeedVectorCreate_Cuda(Ceed ceed, CeedInt n, CeedVector vec);

CEED_INTERN int CeedElemRestrictionCreate_Cuda(CeedElemRestriction r,
    CeedMemType mtype,
    CeedCopyMode cmode, const CeedInt *indices);

CEED_INTERN int CeedBasisApplyElems_Cuda(CeedBasis basis, 
    CeedTransposeMode tmode, CeedEvalMode emode, const CeedVector u, CeedVector v);

CEED_INTERN int CeedBasisCreateTensorH1_Cuda(Ceed ceed, CeedInt dim, CeedInt P1d,
    CeedInt Q1d, const CeedScalar *interp1d,
    const CeedScalar *grad1d,
    const CeedScalar *qref1d,
    const CeedScalar *qweight1d,
    CeedBasis basis);

CEED_INTERN int CeedQFunctionCreate_Cuda(CeedQFunction qf);

CEED_INTERN int CeedOperatorCreate_Cuda(CeedOperator op);
