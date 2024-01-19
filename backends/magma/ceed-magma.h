// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

// magma functions specific to ceed
#ifndef CEED_MAGMA_H
#define CEED_MAGMA_H

#include <ceed.h>
#include <ceed/backend.h>
#include <magma_v2.h>

#define MAGMA_MAXTHREADS_1D 128
#define MAGMA_MAXTHREADS_2D 128
#define MAGMA_MAXTHREADS_3D 64

// Define macro for determining number of threads in y-direction for basis kernels
#define MAGMA_BASIS_NTCOL(x, maxt) (((maxt) < (x)) ? 1 : ((maxt) / (x)))

// Define macros for non-tensor kernel instances
#define MAGMA_NONTENSOR_CUSTOM_KERNEL_MAX_P 40
#define MAGMA_NONTENSOR_CUSTOM_KERNEL_MAX_Q 40
#define MAGMA_NONTENSOR_KERNEL_INSTANCES 7
#define MAGMA_NONTENSOR_KERNEL_N_VALUES 1024, 5120, 10240, 51200, 102400, 512000, 1024000

#ifdef CEED_MAGMA_USE_HIP
typedef hipModule_t   CeedMagmaModule;
typedef hipFunction_t CeedMagmaFunction;
#define CeedCompileMagma CeedCompile_Hip
#define CeedGetKernelMagma CeedGetKernel_Hip
#define CeedRunKernelMagma CeedRunKernel_Hip
#define CeedRunKernelDimMagma CeedRunKernelDim_Hip
#define CeedRunKernelDimSharedMagma CeedRunKernelDimShared_Hip
#else
typedef CUmodule   CeedMagmaModule;
typedef CUfunction CeedMagmaFunction;
#define CeedCompileMagma CeedCompile_Cuda
#define CeedGetKernelMagma CeedGetKernel_Cuda
#define CeedRunKernelMagma CeedRunKernel_Cuda
#define CeedRunKernelDimMagma CeedRunKernelDim_Cuda
#define CeedRunKernelDimSharedMagma CeedRunKernelDimShared_Cuda
#endif

typedef struct {
  CeedMagmaModule   module;
  CeedMagmaFunction Interp;
  CeedMagmaFunction InterpTranspose;
  CeedMagmaFunction Grad;
  CeedMagmaFunction GradTranspose;
  CeedMagmaFunction Weight;
  CeedScalar       *d_interp_1d;
  CeedScalar       *d_grad_1d;
  CeedScalar       *d_q_weight_1d;
} CeedBasis_Magma;

typedef struct {
  CeedMagmaModule   module[MAGMA_NONTENSOR_KERNEL_INSTANCES];
  CeedMagmaFunction Interp[MAGMA_NONTENSOR_KERNEL_INSTANCES];
  CeedMagmaFunction InterpTranspose[MAGMA_NONTENSOR_KERNEL_INSTANCES];
  CeedMagmaFunction Deriv[MAGMA_NONTENSOR_KERNEL_INSTANCES];
  CeedMagmaFunction DerivTranspose[MAGMA_NONTENSOR_KERNEL_INSTANCES];
  CeedMagmaFunction Weight;
  CeedInt           NB_interp[MAGMA_NONTENSOR_KERNEL_INSTANCES], NB_interp_t[MAGMA_NONTENSOR_KERNEL_INSTANCES];
  CeedInt           NB_deriv[MAGMA_NONTENSOR_KERNEL_INSTANCES], NB_deriv_t[MAGMA_NONTENSOR_KERNEL_INSTANCES];
  CeedScalar       *d_interp;
  CeedScalar       *d_grad;
  CeedScalar       *d_div;
  CeedScalar       *d_curl;
  CeedScalar       *d_q_weight;
} CeedBasisNonTensor_Magma;

CEED_INTERN int CeedBasisCreateTensorH1_Magma(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                              const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis);
CEED_INTERN int CeedBasisCreateH1_Magma(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                        const CeedScalar *grad, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);
CEED_INTERN int CeedBasisCreateHdiv_Magma(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                          const CeedScalar *div, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);
CEED_INTERN int CeedBasisCreateHcurl_Magma(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                           const CeedScalar *curl, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);

CEED_INTERN magma_int_t magma_isdevptr(const void *);

// Comment the line below to use the default magma_is_devptr function
#define magma_is_devptr magma_isdevptr

// If magma and cuda/ref are using the null stream, then ceed_magma_queue_sync should do nothing
#define ceed_magma_queue_sync(...)

#endif  // CEED_MAGMA_H
