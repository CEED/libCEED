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
#include <ceed.h>
#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_MAX_PATH 256

#define CeedChk_Nvrtc(ceed, x) \
do { \
  nvrtcResult result = x; \
  if (result != NVRTC_SUCCESS) \
    return CeedError((ceed), (int)result, nvrtcGetErrorString(result)); \
} while (0)

#define CeedChk_Cu(ceed, x) \
do { \
  CUresult result = x; \
  if (result != CUDA_SUCCESS) { \
    const char *msg; \
    cuGetErrorName(result, &msg); \
    return CeedError((ceed), (int)result, msg); \
  } \
} while (0)

#define QUOTE(...) #__VA_ARGS__

typedef enum {HOST_SYNC, DEVICE_SYNC, BOTH_SYNC, NONE_SYNC} SyncState;

typedef struct {
  CeedScalar *h_array;
  CeedScalar *h_array_allocated;
  CeedScalar *d_array;
  CeedScalar *d_array_allocated;
  SyncState memState;
} CeedVector_Cuda;

typedef struct {
  CUmodule module;
  CUfunction noTrNoTr;
  CUfunction noTrTr;
  CUfunction trNoTr;
  CUfunction trTr;
  CeedInt *h_ind;
  CeedInt *h_ind_allocated;
  CeedInt *d_ind;
  CeedInt *d_ind_allocated;
} CeedElemRestriction_Cuda;

// We use a struct to avoid having to memCpy the array of pointers
// __global__ copies by value the struct.
typedef struct {
  const CeedScalar *inputs[16];
  CeedScalar *outputs[16];
} Fields_Cuda;

typedef struct {
  CUmodule module;
  char *qFunctionName;
  char *qFunctionSource;
  CUfunction qFunction;
  Fields_Cuda fields;
  void *d_c;
} CeedQFunction_Cuda;

typedef struct {
  CUmodule module;
  CUfunction interp;
  CUfunction grad;
  CUfunction weight;
  CeedScalar *d_interp1d;
  CeedScalar *d_grad1d;
  CeedScalar *d_qweight1d;
} CeedBasis_Cuda;

typedef struct {
  CUmodule module;
  CUfunction interp;
  CUfunction grad;
  CUfunction weight;
  CeedScalar *d_interp;
  CeedScalar *d_grad;
  CeedScalar *d_qweight;
} CeedBasisNonTensor_Cuda;

typedef struct {
  CeedVector
  *evecs;   /// E-vectors needed to apply operator (input followed by outputs)
  CeedScalar **edata;
  CeedVector *qvecsin;   /// Input Q-vectors needed to apply operator
  CeedVector *qvecsout;   /// Output Q-vectors needed to apply operator
  CeedInt    numein;
  CeedInt    numeout;
} CeedOperator_Cuda;

typedef struct {
  int optblocksize;
  int deviceId;
} Ceed_Cuda;

static inline CeedInt CeedDivUpInt(CeedInt numer, CeedInt denom) {
  return (numer + denom - 1) / denom;
}

CEED_INTERN int CeedCompileCuda(Ceed ceed, const char *source, CUmodule *module,
                                const CeedInt numopts, ...);

CEED_INTERN int CeedGetKernelCuda(Ceed ceed, CUmodule module, const char *name,
                                  CUfunction *kernel);

CEED_INTERN int CeedRunKernelCuda(Ceed ceed, CUfunction kernel,
                                  const int gridSize,
                                  const int blockSize, void **args);

CEED_INTERN int CeedRunKernelDimCuda(Ceed ceed, CUfunction kernel,
                                     const int gridSize,
                                     const int blockSizeX, const int blockSizeY,
                                     const int blockSizeZ, void **args);

CEED_INTERN int CeedRunKernelDimSharedCuda(Ceed ceed, CUfunction kernel,
                                           const int gridSize,
                                           const int blockSizeX,
                                           const int blockSizeY,
                                           const int blockSizeZ,
                                           const int sharedMemSize,
                                           void **args);

CEED_INTERN int run_kernel_dim_shared(Ceed ceed, CUfunction kernel,
                                      const int gridSize,
                                      const int blockSizeX,
                                      const int blockSizeY,
                                      const int blockSizeZ,
                                      const int sharedMemSize, void **args);

CEED_INTERN int run_kernel_dim_shared(Ceed ceed, CUfunction kernel,
                                      const int gridSize,
                                      const int blockSizeX, const int blockSizeY,
                                      const int blockSizeZ,
                                      const int sharedMemSize,
                                      void **args);

CEED_INTERN int CeedVectorCreate_Cuda(CeedInt n, CeedVector vec);

CEED_INTERN int CeedElemRestrictionCreate_Cuda(CeedMemType mtype,
    CeedCopyMode cmode, const CeedInt *indices, CeedElemRestriction r);

CEED_INTERN int CeedElemRestrictionCreateBlocked_Cuda(const CeedMemType mtype,
    const CeedCopyMode cmode, const CeedInt *indices,
    const CeedElemRestriction res);

CEED_INTERN int CeedBasisApplyElems_Cuda(CeedBasis basis, const CeedInt nelem,
    CeedTransposeMode tmode, CeedEvalMode emode, const CeedVector u, CeedVector v);

CEED_INTERN int CeedQFunctionApplyElems_Cuda(CeedQFunction qf, const CeedInt Q,
    const CeedVector *const u, const CeedVector *v);

CEED_INTERN int CeedBasisCreateTensorH1_Cuda(CeedInt dim, CeedInt P1d,
    CeedInt Q1d,
    const CeedScalar *interp1d,
    const CeedScalar *grad1d,
    const CeedScalar *qref1d,
    const CeedScalar *qweight1d,
    CeedBasis basis);

CEED_INTERN int CeedBasisCreateH1_Cuda(CeedElemTopology, CeedInt, CeedInt,
                                       CeedInt, const CeedScalar *,
                                       const CeedScalar *, const CeedScalar *,
                                       const CeedScalar *, CeedBasis);

CEED_INTERN int CeedQFunctionCreate_Cuda(CeedQFunction qf);

CEED_INTERN int CeedOperatorCreate_Cuda(CeedOperator op);

CEED_INTERN int CeedCompositeOperatorCreate_Cuda(CeedOperator op);
