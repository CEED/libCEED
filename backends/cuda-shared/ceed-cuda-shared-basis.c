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
#include <stddef.h>
#include "ceed-cuda-shared.h"
#include "../cuda/ceed-cuda-compile.h"


//------------------------------------------------------------------------------
// Device initalization
//------------------------------------------------------------------------------
int CeedCudaInitInterp(float *d_B, CeedInt P_1d, CeedInt Q_1d,
                       float **c_B);
int CeedCudaInitInterpGrad(float *d_B, float *d_G, CeedInt P_1d,
                           CeedInt Q_1d, float **c_B_ptr,
                           float **c_G_ptr);

//------------------------------------------------------------------------------
// Apply basis
//------------------------------------------------------------------------------
int CeedBasisApplyTensor_Cuda_shared(CeedBasis basis, const CeedInt num_elem,
                                     CeedTransposeMode t_mode,
                                     CeedEvalMode eval_mode, CeedVector u,
                                     CeedVector v) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  Ceed_Cuda *ceed_Cuda;
  CeedGetData(ceed, &ceed_Cuda); CeedChkBackend(ierr);
  CeedBasis_Cuda_shared *data;
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
    ierr = cudaMemset(d_v, 0, length * sizeof(CeedScalar)); CeedChkBackend(ierr);
  }

  // Apply basis operation
  switch (eval_mode) {
  case CEED_EVAL_INTERP: {
    CeedInt P_1d, Q_1d;
    ierr = CeedBasisGetNumNodes1D(basis, &P_1d); CeedChkBackend(ierr);
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d); CeedChkBackend(ierr);
    CeedInt thread_1d = CeedIntMax(Q_1d, P_1d);
    ierr = CeedCudaInitInterp(data->d_interp_1d, P_1d, Q_1d, &data->c_B);
    CeedChkBackend(ierr);
    void *interp_args[] = {(void *) &num_elem, (void *) &transpose, &data->c_B,
                           &d_u, &d_v
                          };
    if (dim == 1) {
      CeedInt elems_per_block = 32;
      CeedInt grid = num_elem/elems_per_block +
                     ((num_elem / elems_per_block*elems_per_block < num_elem) ? 1 : 0 );
      CeedInt shared_mem = elems_per_block*thread_1d*sizeof(CeedScalar);
      ierr = CeedRunKernelDimSharedCuda(ceed, data->Interp, grid, thread_1d, 1,
                                        elems_per_block, shared_mem,
                                        interp_args); CeedChkBackend(ierr);
    } else if (dim == 2) {
      const CeedInt opt_elems[7] = {0, 32, 8, 6, 4, 2, 8};
      // elems_per_block must be at least 1
      CeedInt elems_per_block = CeedIntMax(thread_1d < 7 ? opt_elems[thread_1d] /
                                           num_comp : 1, 1);
      CeedInt grid = num_elem / elems_per_block +
                     ((num_elem / elems_per_block*elems_per_block < num_elem) ? 1 : 0 );
      CeedInt shared_mem = num_comp*elems_per_block*thread_1d*thread_1d*sizeof(
                             CeedScalar);
      ierr = CeedRunKernelDimSharedCuda(ceed, data->Interp, grid, thread_1d,
                                        thread_1d,
                                        num_comp*elems_per_block, shared_mem,
                                        interp_args); CeedChkBackend(ierr);
    } else if (dim == 3) {
      CeedInt elems_per_block = 1;
      CeedInt grid = num_elem / elems_per_block +
                     ((num_elem / elems_per_block*elems_per_block < num_elem) ? 1 : 0 );
      CeedInt shared_mem = num_comp*elems_per_block*thread_1d*thread_1d*sizeof(
                             CeedScalar);
      ierr = CeedRunKernelDimSharedCuda(ceed, data->Interp, grid, thread_1d,
                                        thread_1d,
                                        num_comp*elems_per_block, shared_mem,
                                        interp_args); CeedChkBackend(ierr);
    }
  } break;
  case CEED_EVAL_GRAD: {
    CeedInt P_1d, Q_1d;
    ierr = CeedBasisGetNumNodes1D(basis, &P_1d); CeedChkBackend(ierr);
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d); CeedChkBackend(ierr);
    CeedInt thread_1d = CeedIntMax(Q_1d, P_1d);
    ierr = CeedCudaInitInterpGrad(data->d_interp_1d, data->d_grad_1d, P_1d,
                                  Q_1d, &data->c_B, &data->c_G);
    CeedChkBackend(ierr);
    void *grad_args[] = {(void *) &num_elem, (void *) &transpose, &data->c_B,
                         &data->c_G, &d_u, &d_v
                        };
    if (dim == 1) {
      CeedInt elems_per_block = 32;
      CeedInt grid = num_elem / elems_per_block +
                     ((num_elem / elems_per_block*elems_per_block<num_elem) ? 1 : 0 );
      CeedInt shared_mem = elems_per_block*thread_1d*sizeof(CeedScalar);
      ierr = CeedRunKernelDimSharedCuda(ceed, data->Grad, grid, thread_1d, 1,
                                        elems_per_block, shared_mem, grad_args);
      CeedChkBackend(ierr);
    } else if (dim == 2) {
      const CeedInt opt_elems[7] = {0, 32, 8, 6, 4, 2, 8};
      // elems_per_block must be at least 1
      CeedInt elems_per_block = CeedIntMax(thread_1d < 7 ? opt_elems[thread_1d] /
                                           num_comp : 1, 1);
      CeedInt grid = num_elem / elems_per_block +
                     ((num_elem / elems_per_block*elems_per_block < num_elem) ? 1 : 0 );
      CeedInt shared_mem = num_comp*elems_per_block*thread_1d*thread_1d*sizeof(
                             CeedScalar);
      ierr = CeedRunKernelDimSharedCuda(ceed, data->Grad, grid, thread_1d, thread_1d,
                                        num_comp*elems_per_block, shared_mem,
                                        grad_args); CeedChkBackend(ierr);
    } else if (dim == 3) {
      CeedInt elems_per_block = 1;
      CeedInt grid = num_elem / elems_per_block +
                     ((num_elem / elems_per_block*elems_per_block < num_elem) ? 1 : 0 );
      CeedInt shared_mem = num_comp*elems_per_block*thread_1d*thread_1d*sizeof(
                             CeedScalar);
      ierr = CeedRunKernelDimSharedCuda(ceed, data->Grad, grid, thread_1d, thread_1d,
                                        num_comp*elems_per_block, shared_mem,
                                        grad_args); CeedChkBackend(ierr);
    }
  } break;
  case CEED_EVAL_WEIGHT: {
    CeedInt Q_1d;
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d); CeedChkBackend(ierr);
    void *weight_args[] = {(void *) &num_elem, (void *) &data->d_q_weight_1d, &d_v};
    if (dim == 1) {
      const CeedInt elems_per_block = 32 / Q_1d;
      const CeedInt gridsize = num_elem / elems_per_block +
                               ((num_elem / elems_per_block*elems_per_block < num_elem) ? 1 : 0 );
      ierr = CeedRunKernelDimCuda(ceed, data->Weight, gridsize, Q_1d,
                                  elems_per_block, 1, weight_args);
      CeedChkBackend(ierr);
    } else if (dim == 2) {
      const CeedInt opt_elems = 32 / (Q_1d * Q_1d);
      const CeedInt elems_per_block = opt_elems > 0 ? opt_elems : 1;
      const CeedInt gridsize = num_elem / elems_per_block +
                               ((num_elem / elems_per_block*elems_per_block < num_elem) ? 1 : 0 );
      ierr = CeedRunKernelDimCuda(ceed, data->Weight, gridsize, Q_1d, Q_1d,
                                  elems_per_block, weight_args);
      CeedChkBackend(ierr);
    } else if (dim == 3) {
      const CeedInt gridsize = num_elem;
      ierr = CeedRunKernelDimCuda(ceed, data->Weight, gridsize, Q_1d, Q_1d, Q_1d,
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
static int CeedBasisDestroy_Cuda_shared(CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);

  CeedBasis_Cuda_shared *data;
  ierr = CeedBasisGetData(basis, &data); CeedChkBackend(ierr);

  CeedChk_Cu(ceed, cuModuleUnload(data->module));

  ierr = cudaFree(data->d_q_weight_1d); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree(data->d_interp_1d); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree(data->d_grad_1d); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree(data->d_collo_grad_1d); CeedChk_Cu(ceed, ierr);

  ierr = CeedFree(&data); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create tensor basis
//------------------------------------------------------------------------------
int CeedBasisCreateTensorH1_Cuda_shared(CeedInt dim, CeedInt P_1d, CeedInt Q_1d,
                                        const CeedScalar *interp_1d,
                                        const CeedScalar *grad_1d,
                                        const CeedScalar *q_ref_1d,
                                        const CeedScalar *q_weight_1d,
                                        CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  CeedBasis_Cuda_shared *data;
  ierr = CeedCalloc(1, &data); CeedChkBackend(ierr);

  // Create float versions of basis data
  float *interp_1d_flt = (float*)(malloc(Q_1d*P_1d*sizeof(float)));
  for (CeedInt i = 0; i < Q_1d * P_1d; i++) {
    interp_1d_flt[i] = (float) interp_1d[i];
  }
  float *grad_1d_flt = (float*)(malloc(Q_1d*P_1d*sizeof(float)));
  for (CeedInt i = 0; i < Q_1d * P_1d; i++) {
    grad_1d_flt[i] = (float) grad_1d[i];
  }
  float *q_weight_1d_flt = (float*)(malloc(Q_1d*sizeof(float)));
  for (CeedInt i = 0; i < Q_1d; i++) {
    q_weight_1d_flt[i] = (float) q_weight_1d[i];
  }

  // Copy basis data to GPU
  const CeedInt q_bytes = Q_1d * sizeof(float);
  ierr = cudaMalloc((void **)&data->d_q_weight_1d, q_bytes); CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(data->d_q_weight_1d, q_weight_1d_flt, q_bytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);

  const CeedInt interp_bytes = q_bytes * P_1d;
  ierr = cudaMalloc((void **)&data->d_interp_1d, interp_bytes); CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(data->d_interp_1d, interp_1d_flt, interp_bytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);

  ierr = cudaMalloc((void **)&data->d_grad_1d, interp_bytes); CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(data->d_grad_1d, grad_1d_flt, interp_bytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);
 
  free(interp_1d_flt);
  free(grad_1d_flt);
  free(q_weight_1d_flt);

  // Compute collocated gradient and copy to GPU
  data->d_collo_grad_1d = NULL;
  if (dim == 3 && Q_1d >= P_1d) {
    CeedScalar *collo_grad_1d;
    ierr = CeedMalloc(Q_1d*Q_1d, &collo_grad_1d); CeedChkBackend(ierr);
    ierr = CeedBasisGetCollocatedGrad(basis, collo_grad_1d); CeedChkBackend(ierr);
    // Again, create float version to copy to GPU
    float *collo_grad_1d_flt = (float*)(malloc(Q_1d*Q_1d*sizeof(float)));
    for (CeedInt i = 0; i < Q_1d * Q_1d; i++) {
      collo_grad_1d_flt[i] = (float) collo_grad_1d[i];
    }
    ierr = cudaMalloc((void **)&data->d_collo_grad_1d, q_bytes * Q_1d);
    CeedChk_Cu(ceed, ierr);
    ierr = cudaMemcpy(data->d_collo_grad_1d, collo_grad_1d_flt, q_bytes * Q_1d,
                      cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);
    ierr = CeedFree(&collo_grad_1d); CeedChkBackend(ierr);
    free(collo_grad_1d_flt);
  }

  // Compile basis kernels
  CeedInt num_comp;
  ierr = CeedBasisGetNumComponents(basis, &num_comp); CeedChkBackend(ierr);
  char *basis_kernel_path, *basis_kernel_source;
  ierr = CeedPathConcatenate(ceed, __FILE__, "kernels/cuda-shared-basis.h",
                             &basis_kernel_path); CeedChkBackend(ierr);
  CeedDebug256(ceed, 2, "----- Loading Basis Kernel Source -----\n");
  ierr = CeedLoadSourceToBuffer(ceed, basis_kernel_path, &basis_kernel_source);
  CeedChkBackend(ierr);
  CeedDebug256(ceed, 2, "----- Loading Basis Kernel Source Complete -----\n");
  ierr = CeedCompileCuda(ceed, basis_kernel_source, &data->module, 8,
                         "BASIS_Q_1D", Q_1d,
                         "BASIS_P_1D", P_1d,
                         "BASIS_T_1D", CeedIntMax(Q_1d, P_1d),
                         "BASIS_BUF_LEN", num_comp * CeedIntPow(Q_1d > P_1d ?
                             Q_1d : P_1d, dim),
                         "BASIS_DIM", dim,
                         "BASIS_NUM_COMP", num_comp,
                         "BASIS_NUM_NODES", CeedIntPow(P_1d, dim),
                         "BASIS_NUM_QPTS", CeedIntPow(Q_1d, dim)
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

  // Register backend functions
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Apply",
                                CeedBasisApplyTensor_Cuda_shared);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Destroy",
                                CeedBasisDestroy_Cuda_shared); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
