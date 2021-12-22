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
#include <cuda.h>
#include <cuda_runtime.h>
#include "ceed-cuda-ref.h"
#include "../cuda/ceed-cuda-compile.h"

//------------------------------------------------------------------------------
// Basis apply - tensor
//------------------------------------------------------------------------------
int CeedBasisApply_Cuda(CeedBasis basis, const CeedInt num_elem,
                        CeedTransposeMode t_mode, CeedEvalMode eval_mode,
                        CeedVector u, CeedVector v) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  Ceed_Cuda *ceed_Cuda;
  ierr = CeedGetData(ceed, &ceed_Cuda); CeedChkBackend(ierr);
  CeedBasis_Cuda *data;
  ierr = CeedBasisGetData(basis, &data); CeedChkBackend(ierr);
  const CeedInt transpose = t_mode == CEED_TRANSPOSE;
  const int max_block_size = 32;

  // Read vectors
  const CeedScalar *d_u;
  CeedScalar *d_v;
  if (eval_mode != CEED_EVAL_WEIGHT) {
    ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u); CeedChkBackend(ierr);
  }
  ierr = CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v); CeedChkBackend(ierr);

  // Clear v for transpose operation
  if (t_mode == CEED_TRANSPOSE) {
    CeedInt length;
    ierr = CeedVectorGetLength(v, &length); CeedChkBackend(ierr);
    ierr = cudaMemset(d_v, 0, length * sizeof(CeedScalar));
    CeedChk_Cu(ceed, ierr);
  }
  CeedInt Q_1d, dim;
  ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d); CeedChkBackend(ierr);
  ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);

  // Basis action
  switch (eval_mode) {
  case CEED_EVAL_INTERP: {
    void *interp_args[] = {(void *) &num_elem, (void *) &transpose,
                           &data->d_interp_1d, &d_u, &d_v
                          };
    CeedInt block_size = CeedIntMin(CeedIntPow(Q_1d, dim), max_block_size);

    ierr = CeedRunKernelCuda(ceed, data->Interp, num_elem, block_size, interp_args);
    CeedChkBackend(ierr);
  } break;
  case CEED_EVAL_GRAD: {
    void *grad_args[] = {(void *) &num_elem, (void *) &transpose, &data->d_interp_1d,
                         &data->d_grad_1d, &d_u, &d_v
                        };
    CeedInt block_size = max_block_size;

    ierr = CeedRunKernelCuda(ceed, data->Grad, num_elem, block_size, grad_args);
    CeedChkBackend(ierr);
  } break;
  case CEED_EVAL_WEIGHT: {
    void *weight_args[] = {(void *) &num_elem, (void *) &data->d_q_weight_1d, &d_v};
    const int grid_size = num_elem;
    ierr = CeedRunKernelDimCuda(ceed, data->Weight, grid_size,
                                Q_1d, dim >= 2 ? Q_1d : 1, 1,
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
int CeedBasisApplyNonTensor_Cuda(CeedBasis basis, const CeedInt num_elem,
                                 CeedTransposeMode t_mode, CeedEvalMode eval_mode,
                                 CeedVector u, CeedVector v) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  Ceed_Cuda *ceed_Cuda;
  ierr = CeedGetData(ceed, &ceed_Cuda); CeedChkBackend(ierr);
  CeedBasisNonTensor_Cuda *data;
  ierr = CeedBasisGetData(basis, &data); CeedChkBackend(ierr);
  CeedInt num_nodes, num_qpts;
  ierr = CeedBasisGetNumQuadraturePoints(basis, &num_qpts); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumNodes(basis, &num_nodes); CeedChkBackend(ierr);
  const CeedInt transpose = t_mode == CEED_TRANSPOSE;
  int elems_per_block = 1;
  int grid = num_elem / elems_per_block +
             ((num_elem / elems_per_block * elems_per_block < num_elem) ? 1 : 0);

  // Read vectors
  const CeedScalar *d_u;
  CeedScalar *d_v;
  if (eval_mode != CEED_EVAL_WEIGHT) {
    ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u); CeedChkBackend(ierr);
  }
  ierr = CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v); CeedChkBackend(ierr);

  // Clear v for transpose operation
  if (t_mode == CEED_TRANSPOSE) {
    CeedInt length;
    ierr = CeedVectorGetLength(v, &length); CeedChkBackend(ierr);
    ierr = cudaMemset(d_v, 0, length * sizeof(CeedScalar));
    CeedChk_Cu(ceed, ierr);
  }

  // Apply basis operation
  switch (eval_mode) {
  case CEED_EVAL_INTERP: {
    void *interp_args[] = {(void *) &num_elem, (void *) &transpose,
                           &data->d_interp, &d_u, &d_v
                          };
    if (transpose) {
      ierr = CeedRunKernelDimCuda(ceed, data->Interp, grid, num_nodes, 1,
                                  elems_per_block, interp_args); CeedChkBackend(ierr);
    } else {
      ierr = CeedRunKernelDimCuda(ceed, data->Interp, grid, num_qpts, 1,
                                  elems_per_block, interp_args); CeedChkBackend(ierr);
    }
  } break;
  case CEED_EVAL_GRAD: {
    void *grad_args[] = {(void *) &num_elem, (void *) &transpose, &data->d_grad,
                         &d_u, &d_v
                        };
    if (transpose) {
      ierr = CeedRunKernelDimCuda(ceed, data->Grad, grid, num_nodes, 1,
                                  elems_per_block, grad_args); CeedChkBackend(ierr);
    } else {
      ierr = CeedRunKernelDimCuda(ceed, data->Grad, grid, num_qpts, 1,
                                  elems_per_block, grad_args); CeedChkBackend(ierr);
    }
  } break;
  case CEED_EVAL_WEIGHT: {
    void *weight_args[] = {(void *) &num_elem, (void *) &data->d_q_weight, &d_v};
    ierr = CeedRunKernelDimCuda(ceed, data->Weight, grid, num_qpts, 1,
                                elems_per_block, weight_args); CeedChkBackend(ierr);
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
static int CeedBasisDestroy_Cuda(CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);

  CeedBasis_Cuda *data;
  ierr = CeedBasisGetData(basis, &data); CeedChkBackend(ierr);

  CeedChk_Cu(ceed, cuModuleUnload(data->module));

  ierr = cudaFree(data->d_q_weight_1d); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree(data->d_interp_1d); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree(data->d_grad_1d); CeedChk_Cu(ceed, ierr);
  ierr = CeedFree(&data); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy non-tensor basis
//------------------------------------------------------------------------------
static int CeedBasisDestroyNonTensor_Cuda(CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);

  CeedBasisNonTensor_Cuda *data;
  ierr = CeedBasisGetData(basis, &data); CeedChkBackend(ierr);

  CeedChk_Cu(ceed, cuModuleUnload(data->module));

  ierr = cudaFree(data->d_q_weight); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree(data->d_interp); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree(data->d_grad); CeedChk_Cu(ceed, ierr);
  ierr = CeedFree(&data); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create tensor
//------------------------------------------------------------------------------
int CeedBasisCreateTensorH1_Cuda(CeedInt dim, CeedInt P_1d, CeedInt Q_1d,
                                 const CeedScalar *interp_1d,
                                 const CeedScalar *grad_1d,
                                 const CeedScalar *q_ref_1d,
                                 const CeedScalar *q_weight_1d,
                                 CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  CeedBasis_Cuda *data;
  ierr = CeedCalloc(1, &data); CeedChkBackend(ierr);

  // Copy data to GPU
  const CeedInt q_bytes = Q_1d * sizeof(CeedScalar);
  ierr = cudaMalloc((void **)&data->d_q_weight_1d, q_bytes);
  CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(data->d_q_weight_1d, q_weight_1d, q_bytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);

  const CeedInt interp_bytes = q_bytes * P_1d;
  ierr = cudaMalloc((void **)&data->d_interp_1d, interp_bytes);
  CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(data->d_interp_1d, interp_1d, interp_bytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);

  ierr = cudaMalloc((void **)&data->d_grad_1d, interp_bytes);
  CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(data->d_grad_1d, grad_1d, interp_bytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);

  // Complie basis kernels
  CeedInt num_comp;
  ierr = CeedBasisGetNumComponents(basis, &num_comp); CeedChkBackend(ierr);
  char *basis_kernel_path, *basis_kernel_source;
  ierr = CeedPathConcatenate(ceed, __FILE__, "kernels/cuda-ref-basis-tensor.h",
                             &basis_kernel_path); CeedChkBackend(ierr);
  ierr = CeedLoadSourceToBuffer(ceed, basis_kernel_path, &basis_kernel_source);
  CeedChkBackend(ierr);
  ierr = CeedCompileCuda(ceed, basis_kernel_source, &data->module, 7,
                         "BASIS_Q1D", Q_1d,
                         "BASIS_P1D", P_1d,
                         "BASIS_BUF_LEN", num_comp * CeedIntPow(Q_1d > P_1d ?
                             Q_1d : P_1d, dim),
                         "BASIS_DIM", dim,
                         "BASIS_NCOMP", num_comp,
                         "BASIS_ELEMSIZE", CeedIntPow(P_1d, dim),
                         "BASIS_NQPT", CeedIntPow(Q_1d, dim)
                        ); CeedChkBackend(ierr);
  ierr = CeedGetKernelCuda(ceed, data->module, "Interp", &data->Interp);
  CeedChkBackend(ierr);
  ierr = CeedGetKernelCuda(ceed, data->module, "Grad", &data->Grad);
  CeedChkBackend(ierr);
  ierr = CeedGetKernelCuda(ceed, data->module, "Weight", &data->Weight);
  CeedChkBackend(ierr);
  ierr = CeedFree(&basis_kernel_path); CeedChkBackend(ierr);
  ierr = CeedFree(&basis_kernel_source); CeedChkBackend(ierr);

  ierr = CeedBasisSetData(basis, data); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Apply",
                                CeedBasisApply_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Destroy",
                                CeedBasisDestroy_Cuda); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create non-tensor
//------------------------------------------------------------------------------
int CeedBasisCreateH1_Cuda(CeedElemTopology topo, CeedInt dim,
                           CeedInt num_nodes,
                           CeedInt num_qpts, const CeedScalar *interp,
                           const CeedScalar *grad, const CeedScalar *qref,
                           const CeedScalar *q_weight, CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  CeedBasisNonTensor_Cuda *data;
  ierr = CeedCalloc(1, &data); CeedChkBackend(ierr);

  // Copy basis data to GPU
  const CeedInt q_bytes = num_qpts * sizeof(CeedScalar);
  ierr = cudaMalloc((void **)&data->d_q_weight, q_bytes); CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(data->d_q_weight, q_weight, q_bytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);

  const CeedInt interp_bytes = q_bytes * num_nodes;
  ierr = cudaMalloc((void **)&data->d_interp, interp_bytes);
  CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(data->d_interp, interp, interp_bytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);

  const CeedInt grad_bytes = q_bytes * num_nodes * dim;
  ierr = cudaMalloc((void **)&data->d_grad, grad_bytes); CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(data->d_grad, grad, grad_bytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);

  // Compile basis kernels
  CeedInt num_comp;
  ierr = CeedBasisGetNumComponents(basis, &num_comp); CeedChkBackend(ierr);
  char *basis_kernel_path, *basis_kernel_source;
  ierr = CeedPathConcatenate(ceed, __FILE__, "kernels/cuda-ref-basis-nontensor.h",
                             &basis_kernel_path); CeedChkBackend(ierr);
  ierr = CeedLoadSourceToBuffer(ceed, basis_kernel_path, &basis_kernel_source);
  CeedChkBackend(ierr);
  ierr = CeedCompileCuda(ceed, basis_kernel_source, &data->module, 4,
                         "Q", num_qpts,
                         "P", num_nodes,
                         "BASIS_DIM", dim,
                         "BASIS_NCOMP", num_comp
                        ); CeedChk_Cu(ceed, ierr);
  ierr = CeedGetKernelCuda(ceed, data->module, "Interp", &data->Interp);
  CeedChk_Cu(ceed, ierr);
  ierr = CeedGetKernelCuda(ceed, data->module, "Grad", &data->Grad);
  CeedChk_Cu(ceed, ierr);
  ierr = CeedGetKernelCuda(ceed, data->module, "Weight", &data->Weight);
  CeedChk_Cu(ceed, ierr);
  ierr = CeedFree(&basis_kernel_path); CeedChkBackend(ierr);
  ierr = CeedFree(&basis_kernel_source); CeedChkBackend(ierr);

  ierr = CeedBasisSetData(basis, data); CeedChkBackend(ierr);

  // Register backend functions
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Apply",
                                CeedBasisApplyNonTensor_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Destroy",
                                CeedBasisDestroyNonTensor_Cuda); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
