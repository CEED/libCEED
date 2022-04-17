// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>
#include <hip/hip_runtime.h>
#include "ceed-hip-ref.h"
#include "../hip/ceed-hip-compile.h"

//------------------------------------------------------------------------------
// Basis apply - tensor
//------------------------------------------------------------------------------
int CeedBasisApply_Hip(CeedBasis basis, const CeedInt num_elem,
                       CeedTransposeMode t_mode,
                       CeedEvalMode eval_mode, CeedVector u, CeedVector v) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  Ceed_Hip *ceed_Hip;
  ierr = CeedGetData(ceed, &ceed_Hip); CeedChkBackend(ierr);
  CeedBasis_Hip *data;
  ierr = CeedBasisGetData(basis, &data); CeedChkBackend(ierr);
  const CeedInt transpose = t_mode == CEED_TRANSPOSE;
  const int max_block_size = 64;

  // Read vectors
  const CeedScalar *d_u;
  CeedScalar *d_v;
  if (eval_mode != CEED_EVAL_WEIGHT) {
    ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u); CeedChkBackend(ierr);
  }
  ierr = CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v); CeedChkBackend(ierr);

  // Clear v for transpose operation
  if (t_mode == CEED_TRANSPOSE) {
    CeedSize length;
    ierr = CeedVectorGetLength(v, &length); CeedChkBackend(ierr);
    ierr = hipMemset(d_v, 0, length * sizeof(CeedScalar));
    CeedChk_Hip(ceed, ierr);
  }

  // Basis action
  switch (eval_mode) {
  case CEED_EVAL_INTERP: {
    void *interp_args[] = {(void *) &num_elem, (void *) &transpose,
                           &data->d_interp_1d, &d_u, &d_v
                          };
    CeedInt Q_1d, dim;
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d); CeedChkBackend(ierr);
    ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
    CeedInt block_size = CeedIntMin(CeedIntPow(Q_1d, dim), max_block_size);

    ierr = CeedRunKernelHip(ceed, data->Interp, num_elem, block_size, interp_args);
    CeedChkBackend(ierr);
  } break;
  case CEED_EVAL_GRAD: {
    void *grad_args[] = {(void *) &num_elem, (void *) &transpose, &data->d_interp_1d,
                         &data->d_grad_1d, &d_u, &d_v
                        };
    CeedInt block_size = max_block_size;

    ierr = CeedRunKernelHip(ceed, data->Grad, num_elem, block_size, grad_args);
    CeedChkBackend(ierr);
  } break;
  case CEED_EVAL_WEIGHT: {
    void *weight_args[] = {(void *) &num_elem, (void *) &data->d_q_weight_1d, &d_v};
    const int block_size = 64;
    int grid_size = num_elem / block_size;
    if (block_size * grid_size < num_elem)
      grid_size += 1;

    ierr = CeedRunKernelHip(ceed, data->Weight, grid_size, block_size,
                            weight_args); CeedChkBackend(ierr);
  } break;
  // LCOV_EXCL_START
  // Evaluate the divergence to/from the quadrature points
  case CEED_EVAL_DIV:
    return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_DIV not supported");
  // Evaluate the curl to/from the quadrature points
  case CEED_EVAL_CURL:
    return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_CURL not supported");
  // Take no action, BasisApply should not have been called
  case CEED_EVAL_NONE:
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "CEED_EVAL_NONE does not make sense in this context");
    // LCOV_EXCL_STOP
  }

  // Restore vectors
  if (eval_mode != CEED_EVAL_WEIGHT) {
    ierr = CeedVectorRestoreArrayRead(u, &d_u); CeedChkBackend(ierr);
  }
  ierr = CeedVectorRestoreArray(v, &d_v); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis apply - non-tensor
//------------------------------------------------------------------------------
int CeedBasisApplyNonTensor_Hip(CeedBasis basis, const CeedInt num_elem,
                                CeedTransposeMode t_mode, CeedEvalMode eval_mode,
                                CeedVector u, CeedVector v) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  Ceed_Hip *ceed_Hip;
  ierr = CeedGetData(ceed, &ceed_Hip); CeedChkBackend(ierr);
  CeedBasisNonTensor_Hip *data;
  ierr = CeedBasisGetData(basis, &data); CeedChkBackend(ierr);
  CeedInt num_nodes, num_qpts;
  ierr = CeedBasisGetNumQuadraturePoints(basis, &num_qpts); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumNodes(basis, &num_nodes); CeedChkBackend(ierr);
  const CeedInt transpose = t_mode == CEED_TRANSPOSE;
  int elemsPerBlock = 1;
  int grid = num_elem/elemsPerBlock+((
                                       num_elem/elemsPerBlock*elemsPerBlock<num_elem)?1:0);

  // Read vectors
  const CeedScalar *d_u;
  CeedScalar *d_v;
  if (eval_mode != CEED_EVAL_WEIGHT) {
    ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u); CeedChkBackend(ierr);
  }
  ierr = CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v); CeedChkBackend(ierr);

  // Clear v for transpose operation
  if (t_mode == CEED_TRANSPOSE) {
    CeedSize length;
    ierr = CeedVectorGetLength(v, &length); CeedChkBackend(ierr);
    ierr = hipMemset(d_v, 0, length * sizeof(CeedScalar));
    CeedChk_Hip(ceed, ierr);
  }

  // Apply basis operation
  switch (eval_mode) {
  case CEED_EVAL_INTERP: {
    void *interp_args[] = {(void *) &num_elem, (void *) &transpose,
                           &data->d_interp, &d_u, &d_v
                          };
    if (transpose) {
      ierr = CeedRunKernelDimHip(ceed, data->Interp, grid, num_nodes, 1,
                                 elemsPerBlock, interp_args); CeedChkBackend(ierr);
    } else {
      ierr = CeedRunKernelDimHip(ceed, data->Interp, grid, num_qpts, 1,
                                 elemsPerBlock, interp_args); CeedChkBackend(ierr);
    }
  } break;
  case CEED_EVAL_GRAD: {
    void *grad_args[] = {(void *) &num_elem, (void *) &transpose, &data->d_grad,
                         &d_u, &d_v
                        };
    if (transpose) {
      ierr = CeedRunKernelDimHip(ceed, data->Grad, grid, num_nodes, 1,
                                 elemsPerBlock, grad_args); CeedChkBackend(ierr);
    } else {
      ierr = CeedRunKernelDimHip(ceed, data->Grad, grid, num_qpts, 1,
                                 elemsPerBlock, grad_args); CeedChkBackend(ierr);
    }
  } break;
  case CEED_EVAL_WEIGHT: {
    void *weight_args[] = {(void *) &num_elem, (void *) &data->d_q_weight, &d_v};
    ierr = CeedRunKernelDimHip(ceed, data->Weight, grid, num_qpts, 1,
                               elemsPerBlock, weight_args); CeedChkBackend(ierr);
  } break;
  // LCOV_EXCL_START
  // Evaluate the divergence to/from the quadrature points
  case CEED_EVAL_DIV:
    return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_DIV not supported");
  // Evaluate the curl to/from the quadrature points
  case CEED_EVAL_CURL:
    return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_CURL not supported");
  // Take no action, BasisApply should not have been called
  case CEED_EVAL_NONE:
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "CEED_EVAL_NONE does not make sense in this context");
    // LCOV_EXCL_STOP
  }

  // Restore vectors
  if (eval_mode != CEED_EVAL_WEIGHT) {
    ierr = CeedVectorRestoreArrayRead(u, &d_u); CeedChkBackend(ierr);
  }
  ierr = CeedVectorRestoreArray(v, &d_v); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy tensor basis
//------------------------------------------------------------------------------
static int CeedBasisDestroy_Hip(CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);

  CeedBasis_Hip *data;
  ierr = CeedBasisGetData(basis, &data); CeedChkBackend(ierr);

  CeedChk_Hip(ceed, hipModuleUnload(data->module));

  ierr = hipFree(data->d_q_weight_1d); CeedChk_Hip(ceed, ierr);
  ierr = hipFree(data->d_interp_1d); CeedChk_Hip(ceed, ierr);
  ierr = hipFree(data->d_grad_1d); CeedChk_Hip(ceed, ierr);
  ierr = CeedFree(&data); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy non-tensor basis
//------------------------------------------------------------------------------
static int CeedBasisDestroyNonTensor_Hip(CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);

  CeedBasisNonTensor_Hip *data;
  ierr = CeedBasisGetData(basis, &data); CeedChkBackend(ierr);

  CeedChk_Hip(ceed, hipModuleUnload(data->module));

  ierr = hipFree(data->d_q_weight); CeedChk_Hip(ceed, ierr);
  ierr = hipFree(data->d_interp); CeedChk_Hip(ceed, ierr);
  ierr = hipFree(data->d_grad); CeedChk_Hip(ceed, ierr);
  ierr = CeedFree(&data); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create tensor
//------------------------------------------------------------------------------
int CeedBasisCreateTensorH1_Hip(CeedInt dim, CeedInt P_1d, CeedInt Q_1d,
                                const CeedScalar *interp_1d,
                                const CeedScalar *grad_1d,
                                const CeedScalar *qref1d,
                                const CeedScalar *q_weight_1d,
                                CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  CeedBasis_Hip *data;
  ierr = CeedCalloc(1, &data); CeedChkBackend(ierr);

  // Copy data to GPU
  const CeedInt q_bytes = Q_1d * sizeof(CeedScalar);
  ierr = hipMalloc((void **)&data->d_q_weight_1d, q_bytes);
  CeedChk_Hip(ceed, ierr);
  ierr = hipMemcpy(data->d_q_weight_1d, q_weight_1d, q_bytes,
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);

  const CeedInt interp_bytes = q_bytes * P_1d;
  ierr = hipMalloc((void **)&data->d_interp_1d, interp_bytes);
  CeedChk_Hip(ceed, ierr);
  ierr = hipMemcpy(data->d_interp_1d, interp_1d, interp_bytes,
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);

  ierr = hipMalloc((void **)&data->d_grad_1d, interp_bytes);
  CeedChk_Hip(ceed, ierr);
  ierr = hipMemcpy(data->d_grad_1d, grad_1d, interp_bytes,
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);

  // Complie basis kernels
  CeedInt ncomp;
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChkBackend(ierr);
  char *basis_kernel_path, *basis_kernel_source;
  ierr = CeedGetJitAbsolutePath(ceed,
                                "ceed/jit-source/hip/hip-ref-basis-tensor.h",
                                &basis_kernel_path); CeedChkBackend(ierr);
  CeedDebug256(ceed, 2, "----- Loading Basis Kernel Source -----\n");
  ierr = CeedLoadSourceToBuffer(ceed, basis_kernel_path, &basis_kernel_source);
  CeedChkBackend(ierr);
  CeedDebug256(ceed, 2, "----- Loading Basis Kernel Source Complete! -----\n");
  ierr = CeedCompileHip(ceed, basis_kernel_source, &data->module, 7,
                        "BASIS_Q_1D", Q_1d,
                        "BASIS_P_1D", P_1d,
                        "BASIS_BUF_LEN", ncomp * CeedIntPow(Q_1d > P_1d ?
                            Q_1d : P_1d, dim),
                        "BASIS_DIM", dim,
                        "BASIS_NUM_COMP", ncomp,
                        "BASIS_NUM_NODES", CeedIntPow(P_1d, dim),
                        "BASIS_NUM_QPTS", CeedIntPow(Q_1d, dim)
                       ); CeedChkBackend(ierr);
  ierr = CeedGetKernelHip(ceed, data->module, "Interp", &data->Interp);
  CeedChkBackend(ierr);
  ierr = CeedGetKernelHip(ceed, data->module, "Grad", &data->Grad);
  CeedChkBackend(ierr);
  ierr = CeedGetKernelHip(ceed, data->module, "Weight", &data->Weight);
  CeedChkBackend(ierr);
  ierr = CeedFree(&basis_kernel_path); CeedChkBackend(ierr);
  ierr = CeedFree(&basis_kernel_source); CeedChkBackend(ierr);

  ierr = CeedBasisSetData(basis, data); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Apply",
                                CeedBasisApply_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Destroy",
                                CeedBasisDestroy_Hip); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create non-tensor
//------------------------------------------------------------------------------
int CeedBasisCreateH1_Hip(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes,
                          CeedInt num_qpts, const CeedScalar *interp,
                          const CeedScalar *grad, const CeedScalar *qref,
                          const CeedScalar *q_weight, CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  CeedBasisNonTensor_Hip *data;
  ierr = CeedCalloc(1, &data); CeedChkBackend(ierr);

  // Copy basis data to GPU
  const CeedInt q_bytes = num_qpts * sizeof(CeedScalar);
  ierr = hipMalloc((void **)&data->d_q_weight, q_bytes); CeedChk_Hip(ceed, ierr);
  ierr = hipMemcpy(data->d_q_weight, q_weight, q_bytes,
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);

  const CeedInt interp_bytes = q_bytes * num_nodes;
  ierr = hipMalloc((void **)&data->d_interp, interp_bytes);
  CeedChk_Hip(ceed, ierr);
  ierr = hipMemcpy(data->d_interp, interp, interp_bytes,
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);

  const CeedInt grad_bytes = q_bytes * num_nodes * dim;
  ierr = hipMalloc((void **)&data->d_grad, grad_bytes); CeedChk_Hip(ceed, ierr);
  ierr = hipMemcpy(data->d_grad, grad, grad_bytes,
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);

  // Compile basis kernels
  CeedInt ncomp;
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChkBackend(ierr);
  char *basis_kernel_path, *basis_kernel_source;
  ierr = CeedGetJitAbsolutePath(ceed,
                                "ceed/jit-source/hip/hip-ref-basis-nontensor.h",
                                &basis_kernel_path); CeedChkBackend(ierr);
  CeedDebug256(ceed, 2, "----- Loading Basis Kernel Source -----\n");
  ierr = CeedLoadSourceToBuffer(ceed, basis_kernel_path, &basis_kernel_source);
  CeedChkBackend(ierr);
  CeedDebug256(ceed, 2, "----- Loading Basis Kernel Source Complete! -----\n");
  ierr = CeedCompileHip(ceed, basis_kernel_source, &data->module, 4,
                        "BASIS_Q", num_qpts,
                        "BASIS_P", num_nodes,
                        "BASIS_DIM", dim,
                        "BASIS_NUM_COMP", ncomp
                       ); CeedChk_Hip(ceed, ierr);
  ierr = CeedGetKernelHip(ceed, data->module, "Interp", &data->Interp);
  CeedChk_Hip(ceed, ierr);
  ierr = CeedGetKernelHip(ceed, data->module, "Grad", &data->Grad);
  CeedChk_Hip(ceed, ierr);
  ierr = CeedGetKernelHip(ceed, data->module, "Weight", &data->Weight);
  CeedChk_Hip(ceed, ierr);
  ierr = CeedFree(&basis_kernel_path); CeedChkBackend(ierr);
  ierr = CeedFree(&basis_kernel_source); CeedChkBackend(ierr);

  ierr = CeedBasisSetData(basis, data); CeedChkBackend(ierr);

  // Register backend functions
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Apply",
                                CeedBasisApplyNonTensor_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Destroy",
                                CeedBasisDestroyNonTensor_Hip); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
