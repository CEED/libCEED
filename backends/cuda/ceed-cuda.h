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

#ifndef _ceed_cuda_h
#define _ceed_cuda_h

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <nvrtc.h>

#define CUDA_MAX_PATH 256

#define CeedChk_Nvrtc(ceed, x) \
do { \
  nvrtcResult result = x; \
  if (result != NVRTC_SUCCESS) \
    return CeedError((ceed), CEED_ERROR_BACKEND, nvrtcGetErrorString(result)); \
} while (0)

#define CeedChk_Cu(ceed, x) \
do { \
  CUresult result = x; \
  if (result != CUDA_SUCCESS) { \
    const char *msg; \
    cuGetErrorName(result, &msg); \
    return CeedError((ceed), CEED_ERROR_BACKEND, msg); \
  } \
} while (0)

#define CeedChk_Cublas(ceed, x) \
do { \
  cublasStatus_t result = x; \
  if (result != CUBLAS_STATUS_SUCCESS) { \
    const char *msg = cublasGetErrorName(result); \
    return CeedError((ceed), CEED_ERROR_BACKEND, msg); \
   } \
} while (0)

#define QUOTE(...) #__VA_ARGS__

#define CASE(name) case name: return #name
// LCOV_EXCL_START
static const char *cublasGetErrorName(cublasStatus_t error) {
  switch (error) {
    CASE(CUBLAS_STATUS_SUCCESS);
    CASE(CUBLAS_STATUS_NOT_INITIALIZED);
    CASE(CUBLAS_STATUS_ALLOC_FAILED);
    CASE(CUBLAS_STATUS_INVALID_VALUE);
    CASE(CUBLAS_STATUS_ARCH_MISMATCH);
    CASE(CUBLAS_STATUS_MAPPING_ERROR);
    CASE(CUBLAS_STATUS_EXECUTION_FAILED);
    CASE(CUBLAS_STATUS_INTERNAL_ERROR);
  default: return "CUBLAS_STATUS_UNKNOWN_ERROR";
  }
}
// LCOV_EXCL_STOP

typedef enum {
  CEED_CUDA_HOST_SYNC,
  CEED_CUDA_DEVICE_SYNC,
  CEED_CUDA_BOTH_SYNC,
  CEED_CUDA_NONE_SYNC
} CeedCudaSyncState;

typedef struct {
  CeedScalar *h_array;
  CeedScalar *h_array_allocated;
  CeedScalar *d_array;
  CeedScalar *d_array_allocated;
  CeedCudaSyncState memState;
} CeedVector_Cuda;

typedef struct {
  CUmodule module;
  CUfunction noTrStrided;
  CUfunction noTrOffset;
  CUfunction trStrided;
  CUfunction trOffset;
  CeedInt nnodes;
  CeedInt *h_ind;
  CeedInt *h_ind_allocated;
  CeedInt *d_ind;
  CeedInt *d_ind_allocated;
  CeedInt *d_toffsets;
  CeedInt *d_tindices;
  CeedInt *d_lvec_indices;
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
  CeedScalar *h_data;
  CeedScalar *h_data_allocated;
  CeedScalar *d_data;
  CeedScalar *d_data_allocated;
  CeedCudaSyncState memState;
} CeedQFunctionContext_Cuda;

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
  CUmodule module;
  CUfunction linearDiagonal;
  CUfunction linearPointBlock;
  CeedBasis basisin, basisout;
  CeedElemRestriction diagrstr, pbdiagrstr;
  CeedInt numemodein, numemodeout, nnodes;
  CeedEvalMode *h_emodein, *h_emodeout;
  CeedEvalMode *d_emodein, *d_emodeout;
  CeedScalar *d_identity, *d_interpin, *d_interpout, *d_gradin, *d_gradout;
} CeedOperatorDiag_Cuda;

typedef struct {
  CeedVector
  *evecs;   // E-vectors needed to apply operator (input followed by outputs)
  CeedScalar **edata;
  CeedVector *qvecsin;    // Input Q-vectors needed to apply operator
  CeedVector *qvecsout;   // Output Q-vectors needed to apply operator
  CeedInt    numein;
  CeedInt    numeout;
  CeedOperatorDiag_Cuda *diag;
} CeedOperator_Cuda;

typedef struct {
  int deviceId;
  cublasHandle_t cublasHandle;
  struct cudaDeviceProp deviceProp;
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

CEED_INTERN int CeedRunKernelAutoblockCuda(Ceed ceed, CUfunction kernel,
    size_t size, void **args);

CEED_INTERN int CeedRunKernelDimCuda(Ceed ceed, CUfunction kernel,
                                     const int gridSize,
                                     const int blockSizeX, const int blockSizeY,
                                     const int blockSizeZ, void **args);

CEED_INTERN int CeedRunKernelDimSharedCuda(Ceed ceed, CUfunction kernel,
    const int gridSize, const int blockSizeX, const int blockSizeY,
    const int blockSizeZ, const int sharedMemSize, void **args);

CEED_INTERN int CeedCudaInit(Ceed ceed, const char *resource, int nrc);

CEED_INTERN int CeedCudaGetCublasHandle(Ceed ceed, cublasHandle_t *handle);

CEED_INTERN int CeedDestroy_Cuda(Ceed ceed);

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

CEED_INTERN int CeedQFunctionContextCreate_Cuda(CeedQFunctionContext ctx);

CEED_INTERN int CeedOperatorCreate_Cuda(CeedOperator op);

CEED_INTERN int CeedCompositeOperatorCreate_Cuda(CeedOperator op);
#endif
