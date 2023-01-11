// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-tools.h>
#include <hip/hip_runtime.h>
#include <stddef.h>

#include "../hip/ceed-hip-common.h"
#include "../hip/ceed-hip-compile.h"
#include "ceed-hip-shared.h"

//------------------------------------------------------------------------------
// Compute a block size based on required minimum threads
//------------------------------------------------------------------------------
static CeedInt ComputeBlockSizeFromRequirement(const CeedInt required) {
  CeedInt maxSize     = 1024;  // Max total threads per block
  CeedInt currentSize = 64;    // Start with one group

  while (currentSize < maxSize) {
    if (currentSize > required) break;
    else currentSize = currentSize * 2;
  }
  return currentSize;
}

//------------------------------------------------------------------------------
// Compute required thread block sizes for basis kernels given P, Q, dim, and
// num_comp (num_comp not currently used, but may be again in other basis
// parallelization options)
//------------------------------------------------------------------------------
static int ComputeBasisThreadBlockSizes(const CeedInt dim, const CeedInt P_1d, const CeedInt Q_1d, const CeedInt num_comp, CeedInt *block_sizes) {
  // Note that this will use the same block sizes for all dimensions when compiling,
  // but as each basis object is defined for a particular dimension, we will never
  // call any kernels except the ones for the dimension for which we have computed the
  // block sizes.
  const CeedInt thread_1d = CeedIntMax(P_1d, Q_1d);
  switch (dim) {
    case 1: {
      // Interp kernels:
      block_sizes[0] = 256;

      // Grad kernels:
      block_sizes[1] = 256;

      // Weight kernels:
      block_sizes[2] = 256;
    } break;
    case 2: {
      // Interp kernels:
      CeedInt required = thread_1d * thread_1d;
      block_sizes[0]   = CeedIntMax(256, ComputeBlockSizeFromRequirement(required));

      // Grad kernels: currently use same required minimum threads
      block_sizes[1] = CeedIntMax(256, ComputeBlockSizeFromRequirement(required));

      // Weight kernels:
      required       = CeedIntMax(64, Q_1d * Q_1d);
      block_sizes[2] = CeedIntMax(256, ComputeBlockSizeFromRequirement(required));

    } break;
    case 3: {
      // Interp kernels:
      CeedInt required = thread_1d * thread_1d;
      block_sizes[0]   = CeedIntMax(256, ComputeBlockSizeFromRequirement(required));

      // Grad kernels: currently use same required minimum threads
      block_sizes[1] = CeedIntMax(256, ComputeBlockSizeFromRequirement(required));

      // Weight kernels:
      required       = Q_1d * Q_1d * Q_1d;
      block_sizes[2] = CeedIntMax(256, ComputeBlockSizeFromRequirement(required));
    }
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply basis
//------------------------------------------------------------------------------
int CeedBasisApplyTensor_Hip_shared(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
                                    CeedVector v) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  Ceed_Hip *ceed_Hip;
  CeedCallBackend(CeedGetData(ceed, &ceed_Hip));
  CeedBasis_Hip_shared *data;
  CeedCallBackend(CeedBasisGetData(basis, &data));
  CeedInt dim, num_comp;
  CeedCallBackend(CeedBasisGetDimension(basis, &dim));
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));

  // Read vectors
  const CeedScalar *d_u;
  CeedScalar       *d_v;
  if (eval_mode != CEED_EVAL_WEIGHT) {
    CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u));
  }
  CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v));

  // Apply basis operation
  switch (eval_mode) {
    case CEED_EVAL_INTERP: {
      CeedInt P_1d, Q_1d;
      CeedInt block_size = data->block_sizes[0];
      CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
      CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
      CeedInt thread_1d     = CeedIntMax(Q_1d, P_1d);
      void   *interp_args[] = {(void *)&num_elem, &data->d_interp_1d, &d_u, &d_v};
      if (dim == 1) {
        CeedInt elems_per_block = 64 * thread_1d > 256 ? 256 / thread_1d : 64;
        elems_per_block         = elems_per_block > 0 ? elems_per_block : 1;
        CeedInt grid            = num_elem / elems_per_block + ((num_elem / elems_per_block * elems_per_block < num_elem) ? 1 : 0);
        CeedInt shared_mem      = elems_per_block * thread_1d * sizeof(CeedScalar);
        if (t_mode == CEED_TRANSPOSE) {
          CeedCallBackend(CeedRunKernelDimSharedHip(ceed, data->InterpTranspose, grid, thread_1d, 1, elems_per_block, shared_mem, interp_args));
        } else {
          CeedCallBackend(CeedRunKernelDimSharedHip(ceed, data->Interp, grid, thread_1d, 1, elems_per_block, shared_mem, interp_args));
        }
      } else if (dim == 2) {
        // Check if required threads is small enough to do multiple elems
        const CeedInt elems_per_block = CeedIntMax(block_size / (thread_1d * thread_1d), 1);
        CeedInt       grid            = num_elem / elems_per_block + ((num_elem / elems_per_block * elems_per_block < num_elem) ? 1 : 0);
        CeedInt       shared_mem      = elems_per_block * thread_1d * thread_1d * sizeof(CeedScalar);
        if (t_mode == CEED_TRANSPOSE) {
          CeedCallBackend(
              CeedRunKernelDimSharedHip(ceed, data->InterpTranspose, grid, thread_1d, thread_1d, elems_per_block, shared_mem, interp_args));
        } else {
          CeedCallBackend(CeedRunKernelDimSharedHip(ceed, data->Interp, grid, thread_1d, thread_1d, elems_per_block, shared_mem, interp_args));
        }
      } else if (dim == 3) {
        const CeedInt elems_per_block = CeedIntMax(block_size / (thread_1d * thread_1d), 1);
        CeedInt       grid            = num_elem / elems_per_block + ((num_elem / elems_per_block * elems_per_block < num_elem) ? 1 : 0);
        CeedInt       shared_mem      = elems_per_block * thread_1d * thread_1d * sizeof(CeedScalar);
        if (t_mode == CEED_TRANSPOSE) {
          CeedCallBackend(
              CeedRunKernelDimSharedHip(ceed, data->InterpTranspose, grid, thread_1d, thread_1d, elems_per_block, shared_mem, interp_args));
        } else {
          CeedCallBackend(CeedRunKernelDimSharedHip(ceed, data->Interp, grid, thread_1d, thread_1d, elems_per_block, shared_mem, interp_args));
        }
      }
    } break;
    case CEED_EVAL_GRAD: {
      CeedInt P_1d, Q_1d;
      CeedInt block_size = data->block_sizes[1];
      CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
      CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
      CeedInt     thread_1d = CeedIntMax(Q_1d, P_1d);
      CeedScalar *d_grad_1d = data->d_grad_1d;
      if (data->d_collo_grad_1d) {
        d_grad_1d = data->d_collo_grad_1d;
      }
      void *grad_args[] = {(void *)&num_elem, &data->d_interp_1d, &d_grad_1d, &d_u, &d_v};
      if (dim == 1) {
        CeedInt elems_per_block = 64 * thread_1d > 256 ? 256 / thread_1d : 64;
        elems_per_block         = elems_per_block > 0 ? elems_per_block : 1;
        CeedInt grid            = num_elem / elems_per_block + ((num_elem / elems_per_block * elems_per_block < num_elem) ? 1 : 0);
        CeedInt shared_mem      = elems_per_block * thread_1d * sizeof(CeedScalar);
        if (t_mode == CEED_TRANSPOSE) {
          CeedCallBackend(CeedRunKernelDimSharedHip(ceed, data->GradTranspose, grid, thread_1d, 1, elems_per_block, shared_mem, grad_args));
        } else {
          CeedCallBackend(CeedRunKernelDimSharedHip(ceed, data->Grad, grid, thread_1d, 1, elems_per_block, shared_mem, grad_args));
        }
      } else if (dim == 2) {
        // Check if required threads is small enough to do multiple elems
        const CeedInt elems_per_block = CeedIntMax(block_size / (thread_1d * thread_1d), 1);
        CeedInt       grid            = num_elem / elems_per_block + ((num_elem / elems_per_block * elems_per_block < num_elem) ? 1 : 0);
        CeedInt       shared_mem      = elems_per_block * thread_1d * thread_1d * sizeof(CeedScalar);
        if (t_mode == CEED_TRANSPOSE) {
          CeedCallBackend(CeedRunKernelDimSharedHip(ceed, data->GradTranspose, grid, thread_1d, thread_1d, elems_per_block, shared_mem, grad_args));
        } else {
          CeedCallBackend(CeedRunKernelDimSharedHip(ceed, data->Grad, grid, thread_1d, thread_1d, elems_per_block, shared_mem, grad_args));
        }
      } else if (dim == 3) {
        const CeedInt elems_per_block = CeedIntMax(block_size / (thread_1d * thread_1d), 1);
        CeedInt       grid            = num_elem / elems_per_block + ((num_elem / elems_per_block * elems_per_block < num_elem) ? 1 : 0);
        CeedInt       shared_mem      = elems_per_block * thread_1d * thread_1d * sizeof(CeedScalar);
        if (t_mode == CEED_TRANSPOSE) {
          CeedCallBackend(CeedRunKernelDimSharedHip(ceed, data->GradTranspose, grid, thread_1d, thread_1d, elems_per_block, shared_mem, grad_args));
        } else {
          CeedCallBackend(CeedRunKernelDimSharedHip(ceed, data->Grad, grid, thread_1d, thread_1d, elems_per_block, shared_mem, grad_args));
        }
      }
    } break;
    case CEED_EVAL_WEIGHT: {
      CeedInt Q_1d;
      CeedInt block_size = data->block_sizes[2];
      CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
      void *weight_args[] = {(void *)&num_elem, (void *)&data->d_q_weight_1d, &d_v};
      if (dim == 1) {
        const CeedInt opt_elems       = block_size / Q_1d;
        const CeedInt elems_per_block = opt_elems > 0 ? opt_elems : 1;
        const CeedInt grid_size       = num_elem / elems_per_block + ((num_elem / elems_per_block * elems_per_block < num_elem) ? 1 : 0);
        CeedCallBackend(CeedRunKernelDimHip(ceed, data->Weight, grid_size, Q_1d, elems_per_block, 1, weight_args));
      } else if (dim == 2) {
        const CeedInt opt_elems       = block_size / (Q_1d * Q_1d);
        const CeedInt elems_per_block = opt_elems > 0 ? opt_elems : 1;
        const CeedInt grid_size       = num_elem / elems_per_block + ((num_elem / elems_per_block * elems_per_block < num_elem) ? 1 : 0);
        CeedCallBackend(CeedRunKernelDimHip(ceed, data->Weight, grid_size, Q_1d, Q_1d, elems_per_block, weight_args));
      } else if (dim == 3) {
        const CeedInt opt_elems       = block_size / (Q_1d * Q_1d);
        const CeedInt elems_per_block = opt_elems > 0 ? opt_elems : 1;
        const CeedInt grid_size       = num_elem / elems_per_block + ((num_elem / elems_per_block * elems_per_block < num_elem) ? 1 : 0);
        CeedCallBackend(CeedRunKernelDimHip(ceed, data->Weight, grid_size, Q_1d, Q_1d, elems_per_block, weight_args));
      }
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
      return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_NONE does not make sense in this context");
      // LCOV_EXCL_STOP
  }

  // Restore vectors
  if (eval_mode != CEED_EVAL_WEIGHT) {
    CeedCallBackend(CeedVectorRestoreArrayRead(u, &d_u));
  }
  CeedCallBackend(CeedVectorRestoreArray(v, &d_v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy basis
//------------------------------------------------------------------------------
static int CeedBasisDestroy_Hip_shared(CeedBasis basis) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));

  CeedBasis_Hip_shared *data;
  CeedCallBackend(CeedBasisGetData(basis, &data));

  CeedCallHip(ceed, hipModuleUnload(data->module));

  CeedCallHip(ceed, hipFree(data->d_q_weight_1d));
  CeedCallHip(ceed, hipFree(data->d_interp_1d));
  CeedCallHip(ceed, hipFree(data->d_grad_1d));
  CeedCallHip(ceed, hipFree(data->d_collo_grad_1d));
  CeedCallBackend(CeedFree(&data));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create tensor basis
//------------------------------------------------------------------------------
int CeedBasisCreateTensorH1_Hip_shared(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                       const CeedScalar *q_ref1d, const CeedScalar *q_weight_1d, CeedBasis basis) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedBasis_Hip_shared *data;
  CeedCallBackend(CeedCalloc(1, &data));

  // Copy basis data to GPU
  const CeedInt qBytes = Q_1d * sizeof(CeedScalar);
  CeedCallHip(ceed, hipMalloc((void **)&data->d_q_weight_1d, qBytes));
  CeedCallHip(ceed, hipMemcpy(data->d_q_weight_1d, q_weight_1d, qBytes, hipMemcpyHostToDevice));

  const CeedInt iBytes = qBytes * P_1d;
  CeedCallHip(ceed, hipMalloc((void **)&data->d_interp_1d, iBytes));
  CeedCallHip(ceed, hipMemcpy(data->d_interp_1d, interp_1d, iBytes, hipMemcpyHostToDevice));

  CeedCallHip(ceed, hipMalloc((void **)&data->d_grad_1d, iBytes));
  CeedCallHip(ceed, hipMemcpy(data->d_grad_1d, grad_1d, iBytes, hipMemcpyHostToDevice));

  // Compute collocated gradient and copy to GPU
  data->d_collo_grad_1d    = NULL;
  bool has_collocated_grad = dim == 3 && Q_1d >= P_1d;
  if (has_collocated_grad) {
    CeedScalar *collo_grad_1d;
    CeedCallBackend(CeedMalloc(Q_1d * Q_1d, &collo_grad_1d));
    CeedCallBackend(CeedBasisGetCollocatedGrad(basis, collo_grad_1d));
    CeedCallHip(ceed, hipMalloc((void **)&data->d_collo_grad_1d, qBytes * Q_1d));
    CeedCallHip(ceed, hipMemcpy(data->d_collo_grad_1d, collo_grad_1d, qBytes * Q_1d, hipMemcpyHostToDevice));
    CeedCallBackend(CeedFree(&collo_grad_1d));
  }

  // Set number of threads per block for basis kernels
  CeedInt num_comp;
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(ComputeBasisThreadBlockSizes(dim, P_1d, Q_1d, num_comp, data->block_sizes));

  // Compile basis kernels
  char *basis_kernel_path, *basis_kernel_source;
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/hip/hip-shared-basis-tensor.h", &basis_kernel_path));
  CeedDebug256(ceed, 2, "----- Loading Basis Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, basis_kernel_path, &basis_kernel_source));
  CeedDebug256(ceed, 2, "----- Loading Basis Kernel Source Complete! -----\n");
  CeedCallBackend(CeedCompileHip(ceed, basis_kernel_source, &data->module, 11, "BASIS_Q_1D", Q_1d, "BASIS_P_1D", P_1d, "T_1D", CeedIntMax(Q_1d, P_1d),
                                 "BASIS_DIM", dim, "BASIS_NUM_COMP", num_comp, "BASIS_NUM_NODES", CeedIntPow(P_1d, dim), "BASIS_NUM_QPTS",
                                 CeedIntPow(Q_1d, dim), "BASIS_INTERP_BLOCK_SIZE", data->block_sizes[0], "BASIS_GRAD_BLOCK_SIZE",
                                 data->block_sizes[1], "BASIS_WEIGHT_BLOCK_SIZE", data->block_sizes[2], "BASIS_HAS_COLLOCATED_GRAD",
                                 has_collocated_grad));
  CeedCallBackend(CeedGetKernelHip(ceed, data->module, "Interp", &data->Interp));
  CeedCallBackend(CeedGetKernelHip(ceed, data->module, "InterpTranspose", &data->InterpTranspose));
  CeedCallBackend(CeedGetKernelHip(ceed, data->module, "Grad", &data->Grad));
  CeedCallBackend(CeedGetKernelHip(ceed, data->module, "GradTranspose", &data->GradTranspose));
  CeedCallBackend(CeedGetKernelHip(ceed, data->module, "Weight", &data->Weight));
  CeedCallBackend(CeedFree(&basis_kernel_path));
  CeedCallBackend(CeedFree(&basis_kernel_source));

  CeedCallBackend(CeedBasisSetData(basis, data));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyTensor_Hip_shared));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroy_Hip_shared));
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
