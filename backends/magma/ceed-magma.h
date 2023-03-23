// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

// magma functions specific to ceed
#ifndef _ceed_magma_h
#define _ceed_magma_h

#include <ceed.h>
#include <ceed/backend.h>
#include <magma_v2.h>

#define MAGMA_MAXTHREADS_1D 128
#define MAGMA_MAXTHREADS_2D 128
#define MAGMA_MAXTHREADS_3D 64
#define MAGMA_NONTENSOR_MAXTHREADS (128)

// Define macro for determining number of threads in y-direction
// for basis kernels
#define MAGMA_BASIS_NTCOL(x, maxt) (((maxt) < (x)) ? 1 : ((maxt) / (x)))
#define MAGMA_NONTENSOR_BASIS_NTCOL(N) (CeedIntMax(1, (MAGMA_NONTENSOR_MAXTHREADS / (N))))
#define MAGMA_CEILDIV(A, B) (((A) + (B)-1) / (B))

#define MAGMA_NONTENSOR_CUSTOM_KERNEL_MAX_P (40)
#define MAGMA_NONTENSOR_CUSTOM_KERNEL_MAX_Q (40)

// Define macro for computing the total threads in a block
// for use with __launch_bounds__()
#define MAGMA_BASIS_BOUNDS(x, maxt) (x * MAGMA_BASIS_NTCOL(x, maxt))

// Define macro for non-tensor kernel instances
#define MAGMA_NONTENSOR_KERNEL_INSTANCES (5)
#define MAGMA_NONTENSOR_N_VALUES 10240, 51200, 102400, 512000, 1024000

#ifdef CEED_MAGMA_USE_HIP
typedef hipModule_t   CeedMagmaModule;
typedef hipFunction_t CeedMagmaFunction;
#define CeedCompileMagma CeedCompileHip
#define CeedGetKernelMagma CeedGetKernelHip
#define CeedRunKernelMagma CeedRunKernelHip
#define CeedRunKernelDimMagma CeedRunKernelDimHip
#define CeedRunKernelDimSharedMagma CeedRunKernelDimSharedHip
#else
typedef CUmodule   CeedMagmaModule;
typedef CUfunction CeedMagmaFunction;
#define CeedCompileMagma CeedCompileCuda
#define CeedGetKernelMagma CeedGetKernelCuda
#define CeedRunKernelMagma CeedRunKernelCuda
#define CeedRunKernelDimMagma CeedRunKernelDimCuda
#define CeedRunKernelDimSharedMagma CeedRunKernelDimSharedCuda
#endif

typedef enum { MAGMA_KERNEL_DIM_GENERIC = 101, MAGMA_KERNEL_DIM_SPECIFIC = 102 } magma_kernel_mode_t;

typedef struct {
  magma_kernel_mode_t basis_kernel_mode;
  magma_device_t      device;
  magma_queue_t       queue;
} Ceed_Magma;

typedef struct {
  CeedMagmaModule   module;
  CeedMagmaFunction magma_interp;
  CeedMagmaFunction magma_interp_tr;
  CeedMagmaFunction magma_grad;
  CeedMagmaFunction magma_grad_tr;
  CeedMagmaFunction magma_weight;
  CeedScalar       *dqref1d;
  CeedScalar       *dinterp1d;
  CeedScalar       *dgrad1d;
  CeedScalar       *dqweight1d;
} CeedBasis_Magma;

typedef struct {
  CeedMagmaModule   module[MAGMA_NONTENSOR_KERNEL_INSTANCES];
  CeedMagmaFunction magma_interp_nontensor[MAGMA_NONTENSOR_KERNEL_INSTANCES];
  CeedMagmaFunction magma_interp_tr_nontensor[MAGMA_NONTENSOR_KERNEL_INSTANCES];
  CeedMagmaFunction magma_grad_nontensor[MAGMA_NONTENSOR_KERNEL_INSTANCES];
  CeedMagmaFunction magma_grad_tr_nontensor[MAGMA_NONTENSOR_KERNEL_INSTANCES];
  CeedScalar       *dqref;
  CeedScalar       *dinterp;
  CeedScalar       *dgrad;
  CeedScalar       *dqweight;
} CeedBasisNonTensor_Magma;

typedef enum {
  OWNED_NONE = 0,
  OWNED_UNPINNED,
  OWNED_PINNED,
} OwnershipMode;

typedef struct {
  CeedMagmaModule   module;
  CeedMagmaFunction StridedTranspose;
  CeedMagmaFunction StridedNoTranspose;
  CeedMagmaFunction OffsetTranspose;
  CeedMagmaFunction OffsetNoTranspose;
  CeedInt          *offsets;
  CeedInt          *doffsets;
  OwnershipMode     own_;
  int               down_;  // cover a case where we own Device memory
} CeedElemRestriction_Magma;

typedef struct {
  const CeedScalar **inputs;
  CeedScalar       **outputs;
  bool               setupdone;
} CeedQFunction_Magma;

#define USE_MAGMA_BATCH
#define USE_MAGMA_BATCH2
#define USE_MAGMA_BATCH3
#define USE_MAGMA_BATCH4

CEED_INTERN void magma_weight_nontensor(magma_int_t grid, magma_int_t threads, magma_int_t nelem, magma_int_t Q, CeedScalar *dqweight, CeedScalar *dv,
                                        magma_queue_t queue);

CEED_INTERN int magma_gemm_nontensor(magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k, CeedScalar alpha,
                                     const CeedScalar *dA, magma_int_t ldda, const CeedScalar *dB, magma_int_t lddb, CeedScalar beta, CeedScalar *dC,
                                     magma_int_t lddc, magma_queue_t queue);

CEED_INTERN void gemm_selector(int gpu_arch, char precision, char transA, int m, int n, int k, int *nbatch, int *use_magma);

CEED_INTERN CeedInt nontensor_rtc_get_nb(int gpu_arch, char precision, CeedEvalMode emode, CeedTransposeMode tmode, int P_, int N, int Q_);

CEED_INTERN magma_int_t magma_isdevptr(const void *A);

CEED_INTERN int CeedBasisCreateTensorH1_Magma(CeedInt dim, CeedInt P1d, CeedInt Q1d, const CeedScalar *interp1d, const CeedScalar *grad1d,
                                              const CeedScalar *qref1d, const CeedScalar *qweight1d, CeedBasis basis);

CEED_INTERN int CeedBasisCreateH1_Magma(CeedElemTopology topo, CeedInt dim, CeedInt ndof, CeedInt nqpts, const CeedScalar *interp,
                                        const CeedScalar *grad, const CeedScalar *qref, const CeedScalar *qweight, CeedBasis basis);

CEED_INTERN int CeedElemRestrictionCreate_Magma(CeedMemType mtype, CeedCopyMode cmode, const CeedInt *offsets, CeedElemRestriction r);

CEED_INTERN int CeedElemRestrictionCreateBlocked_Magma(const CeedMemType mtype, const CeedCopyMode cmode, const CeedInt *offsets,
                                                       const CeedElemRestriction res);

CEED_INTERN int CeedOperatorCreate_Magma(CeedOperator op);

// comment the line below to use the default magma_is_devptr function
#define magma_is_devptr magma_isdevptr

// if magma and cuda/ref are using the null stream, then ceed_magma_queue_sync
// should do nothing
#define ceed_magma_queue_sync(...)

// batch stride, override using -DMAGMA_BATCH_STRIDE=<desired-value>
#ifndef MAGMA_BATCH_STRIDE
#define MAGMA_BATCH_STRIDE (1000)
#endif

#endif  // _ceed_magma_h
