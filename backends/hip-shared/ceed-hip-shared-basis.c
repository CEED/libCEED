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

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>
#include <hip/hip_runtime.h>
#include <stddef.h>
#include "ceed-hip-shared.h"
#include "../hip/ceed-hip-common.h"
#include "../hip/ceed-hip-compile.h"

//------------------------------------------------------------------------------
// Compute a block size based on required minimum threads
//------------------------------------------------------------------------------
static CeedInt ComputeBlockSizeFromRequirement(const CeedInt required) {
  CeedInt maxSize = 1024;    // Max total threads per block
  CeedInt currentSize = 64;  // Start with one group

  while(currentSize < maxSize) {
    if (currentSize > required)
      break;
    else
      currentSize = currentSize * 2;
  }
  return currentSize;
}

//------------------------------------------------------------------------------
// Compute required thread block sizes for basis kernels given P, Q, dim, and
// num_comp
//------------------------------------------------------------------------------
static int ComputeBasisThreadBlockSizes(const CeedInt dim, const CeedInt P_1d,
                                        const CeedInt Q_1d,
                                        const CeedInt num_comp, CeedInt *block_sizes) {

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
    CeedInt required = thread_1d * thread_1d * num_comp;
    block_sizes[0]  = ComputeBlockSizeFromRequirement(required);

    // Grad kernels: currently use same required minimum threads
    block_sizes[1]  = ComputeBlockSizeFromRequirement(required);

    // Weight kernels:
    required = CeedIntMax(64, Q_1d * Q_1d);
    block_sizes[2]  = ComputeBlockSizeFromRequirement(required);

  } break;
  case 3: {
    // Interp kernels:
    CeedInt required = thread_1d * thread_1d * num_comp;
    block_sizes[0]  = ComputeBlockSizeFromRequirement(required);

    // Grad kernels: currently use same required minimum threads
    block_sizes[1]  = ComputeBlockSizeFromRequirement(required);

    // Weight kernels:
    required = Q_1d * Q_1d * Q_1d;
    block_sizes[2]  = ComputeBlockSizeFromRequirement(required);
  }
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply basis
//------------------------------------------------------------------------------
int CeedBasisApplyTensor_Hip_shared(CeedBasis basis, const CeedInt num_elem,
                                    CeedTransposeMode t_mode,
                                    CeedEvalMode eval_mode, CeedVector u,
                                    CeedVector v) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  Ceed_Hip *ceed_Hip;
  CeedGetData(ceed, &ceed_Hip); CeedChkBackend(ierr);
  CeedBasis_Hip_shared *data;
  CeedBasisGetData(basis, &data); CeedChkBackend(ierr);
  const CeedInt transpose = t_mode == CEED_TRANSPOSE;
  CeedInt dim, num_comp;
  ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumComponents(basis, &num_comp); CeedChkBackend(ierr);

  // Read vectors
  const CeedScalar *d_u;
  CeedScalar *d_v;
  if (eval_mode != CEED_EVAL_WEIGHT) {
    ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u); CeedChkBackend(ierr);
  }
  ierr = CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v); CeedChkBackend(ierr);

  // Clear v for transpose mode
  if (t_mode == CEED_TRANSPOSE) {
    CeedInt length;
    ierr = CeedVectorGetLength(v, &length); CeedChkBackend(ierr);
    ierr = hipMemset(d_v, 0, length * sizeof(CeedScalar)); CeedChkBackend(ierr);
  }

  // Apply basis operation
  switch (eval_mode) {
  case CEED_EVAL_INTERP: {
    CeedInt P_1d, Q_1d;
    CeedInt block_size = data->block_sizes[0];
    ierr = CeedBasisGetNumNodes1D(basis, &P_1d); CeedChkBackend(ierr);
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d); CeedChkBackend(ierr);
    CeedInt thread_1d = CeedIntMax(Q_1d, P_1d);
    void *interp_args[] = {(void *) &num_elem, (void *) &transpose, &data->d_interp_1d,
                           &d_u, &d_v
                          };
    if (dim == 1) {
      CeedInt elems_per_block = 64 * thread_1d > 256 ? 256 / thread_1d : 64;
      elems_per_block = elems_per_block > 0 ? elems_per_block : 1;
      CeedInt grid = num_elem / elems_per_block +
                     ((num_elem / elems_per_block*elems_per_block < num_elem) ? 1 : 0 );
      CeedInt shared_mem = elems_per_block*thread_1d*sizeof(CeedScalar);
      ierr = CeedRunKernelDimSharedHip(ceed, data->Interp, grid, thread_1d, 1,
                                       elems_per_block, shared_mem,
                                       interp_args); CeedChkBackend(ierr);
    } else if (dim == 2) {
      // Check if required threads is small enough to do multiple elems
      const CeedInt elems_per_block = CeedIntMax(block_size /
                                      (thread_1d*thread_1d*num_comp), 1);
      CeedInt grid = num_elem / elems_per_block +
                     ((num_elem / elems_per_block*elems_per_block < num_elem) ? 1 : 0 );
      CeedInt shared_mem = num_comp*elems_per_block*thread_1d*thread_1d*sizeof(
                             CeedScalar);
      ierr = CeedRunKernelDimSharedHip(ceed, data->Interp, grid, thread_1d, thread_1d,
                                       num_comp*elems_per_block, shared_mem,
                                       interp_args); CeedChkBackend(ierr);
    } else if (dim == 3) {
      CeedInt elems_per_block = 1;
      CeedInt grid = num_elem / elems_per_block +
                     ((num_elem / elems_per_block*elems_per_block < num_elem) ? 1 : 0 );
      CeedInt shared_mem = num_comp*elems_per_block*thread_1d*thread_1d*sizeof(
                             CeedScalar);
      ierr = CeedRunKernelDimSharedHip(ceed, data->Interp, grid, thread_1d, thread_1d,
                                       num_comp*elems_per_block, shared_mem,
                                       interp_args); CeedChkBackend(ierr);
    }
  } break;
  case CEED_EVAL_GRAD: {
    CeedInt P_1d, Q_1d;
    CeedInt block_size = data->block_sizes[1];
    ierr = CeedBasisGetNumNodes1D(basis, &P_1d); CeedChkBackend(ierr);
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d); CeedChkBackend(ierr);
    CeedInt thread_1d = CeedIntMax(Q_1d, P_1d);
    void *grad_args[] = {(void *) &num_elem, (void *) &transpose, &data->d_interp_1d,
                         &data->d_grad_1d, &d_u, &d_v
                        };
    if (dim == 1) {
      CeedInt elems_per_block = 64 * thread_1d > 256 ? 256 / thread_1d : 64;
      elems_per_block = elems_per_block > 0 ? elems_per_block : 1;
      CeedInt grid = num_elem / elems_per_block +
                     ((num_elem / elems_per_block*elems_per_block < num_elem) ? 1 : 0 );
      CeedInt shared_mem = elems_per_block*thread_1d*sizeof(CeedScalar);
      ierr = CeedRunKernelDimSharedHip(ceed, data->Grad, grid, thread_1d, 1,
                                       elems_per_block, shared_mem, grad_args);
      CeedChkBackend(ierr);
    } else if (dim == 2) {
      // Check if required threads is small enough to do multiple elems
      const CeedInt elems_per_block = CeedIntMax(block_size/
                                      (thread_1d*thread_1d*num_comp), 1);
      CeedInt grid = num_elem / elems_per_block +
                     ((num_elem / elems_per_block*elems_per_block < num_elem) ? 1 : 0 );
      CeedInt shared_mem = num_comp*elems_per_block*thread_1d*thread_1d*sizeof(
                             CeedScalar);
      ierr = CeedRunKernelDimSharedHip(ceed, data->Grad, grid, thread_1d, thread_1d,
                                       num_comp*elems_per_block, shared_mem,
                                       grad_args); CeedChkBackend(ierr);
    } else if (dim == 3) {
      CeedInt elems_per_block = 1;
      CeedInt grid = num_elem / elems_per_block +
                     ((num_elem / elems_per_block*elems_per_block < num_elem) ? 1 : 0 );
      CeedInt shared_mem = num_comp*elems_per_block*thread_1d*thread_1d*sizeof(
                             CeedScalar);
      ierr = CeedRunKernelDimSharedHip(ceed, data->Grad, grid, thread_1d, thread_1d,
                                       num_comp*elems_per_block, shared_mem,
                                       grad_args); CeedChkBackend(ierr);
    }
  } break;
  case CEED_EVAL_WEIGHT: {
    CeedInt Q_1d;
    CeedInt block_size = data->block_sizes[2];
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d); CeedChkBackend(ierr);
    void *weight_args[] = {(void *) &num_elem, (void *) &data->d_q_weight_1d, &d_v};
    if (dim == 1) {
      const CeedInt opt_elems = block_size / Q_1d;
      const CeedInt elems_per_block = opt_elems > 0 ? opt_elems : 1;
      const CeedInt grid_size = num_elem / elems_per_block +
                                ((num_elem / elems_per_block*elems_per_block < num_elem) ? 1 : 0 );
      ierr = CeedRunKernelDimHip(ceed, data->Weight, grid_size, Q_1d,
                                 elems_per_block, 1, weight_args);
      CeedChkBackend(ierr);
    } else if (dim == 2) {
      const CeedInt opt_elems = block_size / (Q_1d * Q_1d);
      const CeedInt elems_per_block = opt_elems > 0 ? opt_elems : 1;
      const CeedInt grid_size = num_elem / elems_per_block +
                                ((num_elem / elems_per_block*elems_per_block < num_elem) ? 1 : 0 );
      ierr = CeedRunKernelDimHip(ceed, data->Weight, grid_size, Q_1d, Q_1d,
                                 elems_per_block, weight_args);
      CeedChkBackend(ierr);
    } else if (dim == 3) {
      const CeedInt grid_size = num_elem;
      ierr = CeedRunKernelDimHip(ceed, data->Weight, grid_size, Q_1d, Q_1d, Q_1d,
                                 weight_args);
      CeedChkBackend(ierr);
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
// Destroy basis
//------------------------------------------------------------------------------
static int CeedBasisDestroy_Hip_shared(CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);

  CeedBasis_Hip_shared *data;
  ierr = CeedBasisGetData(basis, &data); CeedChkBackend(ierr);

  CeedChk_Hip(ceed, hipModuleUnload(data->module));

  ierr = hipFree(data->d_q_weight_1d); CeedChk_Hip(ceed, ierr);
  ierr = hipFree(data->d_interp_1d); CeedChk_Hip(ceed, ierr);
  ierr = hipFree(data->d_grad_1d); CeedChk_Hip(ceed, ierr);
  ierr = hipFree(data->d_collo_grad_1d); CeedChk_Hip(ceed, ierr);
  ierr = CeedFree(&data); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create tensor basis
//------------------------------------------------------------------------------
int CeedBasisCreateTensorH1_Hip_shared(CeedInt dim, CeedInt P_1d, CeedInt Q_1d,
                                       const CeedScalar *interp_1d,
                                       const CeedScalar *grad_1d,
                                       const CeedScalar *q_ref1d,
                                       const CeedScalar *q_weight_1d,
                                       CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  CeedBasis_Hip_shared *data;
  ierr = CeedCalloc(1, &data); CeedChkBackend(ierr);

  // Copy basis data to GPU
  const CeedInt qBytes = Q_1d * sizeof(CeedScalar);
  ierr = hipMalloc((void **)&data->d_q_weight_1d, qBytes);
  CeedChk_Hip(ceed, ierr);
  ierr = hipMemcpy(data->d_q_weight_1d, q_weight_1d, qBytes,
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);

  const CeedInt iBytes = qBytes * P_1d;
  ierr = hipMalloc((void **)&data->d_interp_1d, iBytes); CeedChk_Hip(ceed, ierr);
  ierr = hipMemcpy(data->d_interp_1d, interp_1d, iBytes,
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);

  ierr = hipMalloc((void **)&data->d_grad_1d, iBytes); CeedChk_Hip(ceed, ierr);
  ierr = hipMemcpy(data->d_grad_1d, grad_1d, iBytes,
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);

  // Compute collocated gradient and copy to GPU
  data->d_collo_grad_1d = NULL;
  if (dim == 3 && Q_1d >= P_1d) {
    CeedScalar *collo_grad_1d;
    ierr = CeedMalloc(Q_1d*Q_1d, &collo_grad_1d); CeedChkBackend(ierr);
    ierr = CeedBasisGetCollocatedGrad(basis, collo_grad_1d); CeedChkBackend(ierr);
    ierr = hipMalloc((void **)&data->d_collo_grad_1d, qBytes * Q_1d);
    CeedChk_Hip(ceed, ierr);
    ierr = hipMemcpy(data->d_collo_grad_1d, collo_grad_1d, qBytes * Q_1d,
                     hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);
    ierr = CeedFree(&collo_grad_1d); CeedChkBackend(ierr);
  }

  // Set number of threads per block for basis kernels
  CeedInt num_comp;
  ierr = CeedBasisGetNumComponents(basis, &num_comp); CeedChkBackend(ierr);
  ierr = ComputeBasisThreadBlockSizes(dim, P_1d, Q_1d, num_comp,
                                      data->block_sizes);
  CeedChkBackend(ierr);

  // Compile basis kernels
  char *basis_kernel_path, *basis_kernel_source;
  ierr = CeedPathConcatenate(ceed, __FILE__, "kernels/hip-shared-basis.h",
                             &basis_kernel_path); CeedChkBackend(ierr);
  CeedDebug256(ceed, 2, "----- Loading Basis Kernel Source -----\n");
  ierr = CeedLoadSourceToBuffer(ceed, basis_kernel_path, &basis_kernel_source);
  CeedChkBackend(ierr);
  CeedDebug256(ceed, 2, "----- Loading Basis Kernel Source Complete! -----\n");
  ierr = CeedCompileHip(ceed, basis_kernel_source, &data->module, 11,
                        "Q1D", Q_1d,
                        "P1D", P_1d,
                        "T1D", CeedIntMax(Q_1d, P_1d),
                        "BASIS_BUF_LEN", num_comp * CeedIntPow(Q_1d > P_1d ?
                            Q_1d : P_1d, dim),
                        "BASIS_DIM", dim,
                        "BASIS_NCOMP", num_comp,
                        "BASIS_ELEMSIZE", CeedIntPow(P_1d, dim),
                        "BASIS_NQPT", CeedIntPow(Q_1d, dim),
                        "INTERP_BLKSIZE", data->block_sizes[0],
                        "GRAD_BLKSIZE", data->block_sizes[1],
                        "WEIGHT_BLKSIZE", data->block_sizes[2]
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

  // Register backend functions
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Apply",
                                CeedBasisApplyTensor_Hip_shared);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Destroy",
                                CeedBasisDestroy_Hip_shared); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
