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
#include <cuda.h>
#include <cuda_runtime.h>
#include "ceed-cuda.h"
#include "ceed-cuda-jit.h"
#include "kernel-strings/cuda-tensor-basis.h"
#include "kernel-strings/cuda-non-tensor-basis.h"

//------------------------------------------------------------------------------
// Basis apply - tensor
//------------------------------------------------------------------------------
int CeedBasisApply_Cuda(CeedBasis basis, const CeedInt nelem,
                        CeedTransposeMode tmode,
                        CeedEvalMode emode, CeedVector u, CeedVector v) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  Ceed_Cuda *ceed_Cuda;
  ierr = CeedGetData(ceed, &ceed_Cuda); CeedChkBackend(ierr);
  CeedBasis_Cuda *data;
  ierr = CeedBasisGetData(basis, &data); CeedChkBackend(ierr);
  const CeedInt transpose = tmode == CEED_TRANSPOSE;
  const int maxblocksize = 32;

  // Read vectors
  const CeedScalar *d_u;
  CeedScalar *d_v;
  if (emode != CEED_EVAL_WEIGHT) {
    ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u); CeedChkBackend(ierr);
  }
  ierr = CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v); CeedChkBackend(ierr);

  // Clear v for transpose operation
  if (tmode == CEED_TRANSPOSE) {
    CeedInt length;
    ierr = CeedVectorGetLength(v, &length); CeedChkBackend(ierr);
    ierr = cudaMemset(d_v, 0, length * sizeof(CeedScalar));
    CeedChk_Cu(ceed,ierr);
  }
  CeedInt Q1d, dim;
  ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChkBackend(ierr);
  ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);

  // Basis action
  switch (emode) {
  case CEED_EVAL_INTERP: {
    void *interpargs[] = {(void *) &nelem, (void *) &transpose,
                          &data->d_interp1d, &d_u, &d_v
                         };
    CeedInt blocksize = CeedIntPow(Q1d, dim);
    blocksize = blocksize > maxblocksize ? maxblocksize : blocksize;

    ierr = CeedRunKernelCuda(ceed, data->interp, nelem, blocksize, interpargs);
    CeedChkBackend(ierr);
  } break;
  case CEED_EVAL_GRAD: {
    void *gradargs[] = {(void *) &nelem, (void *) &transpose, &data->d_interp1d,
                        &data->d_grad1d, &d_u, &d_v
                       };
    CeedInt blocksize = maxblocksize;

    ierr = CeedRunKernelCuda(ceed, data->grad, nelem, blocksize, gradargs);
    CeedChkBackend(ierr);
  } break;
  case CEED_EVAL_WEIGHT: {
    void *weightargs[] = {(void *) &nelem, (void *) &data->d_qweight1d, &d_v};
    const int gridsize = nelem;
    ierr = CeedRunKernelDimCuda(ceed, data->weight, gridsize,
                                Q1d, dim >= 2 ? Q1d : 1, 1,
                                weightargs); CeedChkBackend(ierr);
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
  if (emode != CEED_EVAL_WEIGHT) {
    ierr = CeedVectorRestoreArrayRead(u, &d_u); CeedChkBackend(ierr);
  }
  ierr = CeedVectorRestoreArray(v, &d_v); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis apply - non-tensor
//------------------------------------------------------------------------------
int CeedBasisApplyNonTensor_Cuda(CeedBasis basis, const CeedInt nelem,
                                 CeedTransposeMode tmode, CeedEvalMode emode,
                                 CeedVector u, CeedVector v) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  Ceed_Cuda *ceed_Cuda;
  ierr = CeedGetData(ceed, &ceed_Cuda); CeedChkBackend(ierr);
  CeedBasisNonTensor_Cuda *data;
  ierr = CeedBasisGetData(basis, &data); CeedChkBackend(ierr);
  CeedInt nnodes, nqpt;
  ierr = CeedBasisGetNumQuadraturePoints(basis, &nqpt); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumNodes(basis, &nnodes); CeedChkBackend(ierr);
  const CeedInt transpose = tmode == CEED_TRANSPOSE;
  int elemsPerBlock = 1;
  int grid = nelem/elemsPerBlock+((nelem/elemsPerBlock*elemsPerBlock<nelem)?1:0);

  // Read vectors
  const CeedScalar *d_u;
  CeedScalar *d_v;
  if (emode != CEED_EVAL_WEIGHT) {
    ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u); CeedChkBackend(ierr);
  }
  ierr = CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v); CeedChkBackend(ierr);

  // Clear v for transpose operation
  if (tmode == CEED_TRANSPOSE) {
    CeedInt length;
    ierr = CeedVectorGetLength(v, &length); CeedChkBackend(ierr);
    ierr = cudaMemset(d_v, 0, length * sizeof(CeedScalar));
    CeedChk_Cu(ceed, ierr);
  }

  // Apply basis operation
  switch (emode) {
  case CEED_EVAL_INTERP: {
    void *interpargs[] = {(void *) &nelem, (void *) &transpose,
                          &data->d_interp, &d_u, &d_v
                         };
    if (!transpose) {
      ierr = CeedRunKernelDimCuda(ceed, data->interp, grid, nqpt, 1,
                                  elemsPerBlock, interpargs); CeedChkBackend(ierr);
    } else {
      ierr = CeedRunKernelDimCuda(ceed, data->interp, grid, nnodes, 1,
                                  elemsPerBlock, interpargs); CeedChkBackend(ierr);
    }
  } break;
  case CEED_EVAL_GRAD: {
    void *gradargs[] = {(void *) &nelem, (void *) &transpose, &data->d_grad,
                        &d_u, &d_v
                       };
    if (!transpose) {
      ierr = CeedRunKernelDimCuda(ceed, data->grad, grid, nqpt, 1,
                                  elemsPerBlock, gradargs); CeedChkBackend(ierr);
    } else {
      ierr = CeedRunKernelDimCuda(ceed, data->grad, grid, nnodes, 1,
                                  elemsPerBlock, gradargs); CeedChkBackend(ierr);
    }
  } break;
  case CEED_EVAL_WEIGHT: {
    void *weightargs[] = {(void *) &nelem, (void *) &data->d_qweight, &d_v};
    ierr = CeedRunKernelDimCuda(ceed, data->weight, grid, nqpt, 1,
                                elemsPerBlock, weightargs); CeedChkBackend(ierr);
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
  if (emode != CEED_EVAL_WEIGHT) {
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

  ierr = cudaFree(data->d_qweight1d); CeedChk_Cu(ceed,ierr);
  ierr = cudaFree(data->d_interp1d); CeedChk_Cu(ceed,ierr);
  ierr = cudaFree(data->d_grad1d); CeedChk_Cu(ceed,ierr);

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

  ierr = cudaFree(data->d_qweight); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree(data->d_interp); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree(data->d_grad); CeedChk_Cu(ceed, ierr);

  ierr = CeedFree(&data); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create tensor
//------------------------------------------------------------------------------
int CeedBasisCreateTensorH1_Cuda(CeedInt dim, CeedInt P1d, CeedInt Q1d,
                                 const CeedScalar *interp1d,
                                 const CeedScalar *grad1d,
                                 const CeedScalar *qref1d,
                                 const CeedScalar *qweight1d,
                                 CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  CeedBasis_Cuda *data;
  ierr = CeedCalloc(1, &data); CeedChkBackend(ierr);

  // Copy data to GPU
  const CeedInt qBytes = Q1d * sizeof(CeedScalar);
  ierr = cudaMalloc((void **)&data->d_qweight1d, qBytes); CeedChk_Cu(ceed,ierr);
  ierr = cudaMemcpy(data->d_qweight1d, qweight1d, qBytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed,ierr);

  const CeedInt iBytes = qBytes * P1d;
  ierr = cudaMalloc((void **)&data->d_interp1d, iBytes); CeedChk_Cu(ceed,ierr);
  ierr = cudaMemcpy(data->d_interp1d, interp1d, iBytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed,ierr);

  ierr = cudaMalloc((void **)&data->d_grad1d, iBytes); CeedChk_Cu(ceed,ierr);
  ierr = cudaMemcpy(data->d_grad1d, grad1d, iBytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed,ierr);

  // Complie basis kernels
  CeedInt ncomp;
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChkBackend(ierr);
  ierr = CeedCompileCuda(ceed, basiskernels, &data->module, 7,
                         "BASIS_Q1D", Q1d,
                         "BASIS_P1D", P1d,
                         "BASIS_BUF_LEN", ncomp * CeedIntPow(Q1d > P1d ?
                             Q1d : P1d, dim),
                         "BASIS_DIM", dim,
                         "BASIS_NCOMP", ncomp,
                         "BASIS_ELEMSIZE", CeedIntPow(P1d, dim),
                         "BASIS_NQPT", CeedIntPow(Q1d, dim)
                        ); CeedChkBackend(ierr);
  ierr = CeedGetKernelCuda(ceed, data->module, "interp", &data->interp);
  CeedChkBackend(ierr);
  ierr = CeedGetKernelCuda(ceed, data->module, "grad", &data->grad);
  CeedChkBackend(ierr);
  ierr = CeedGetKernelCuda(ceed, data->module, "weight", &data->weight);
  CeedChkBackend(ierr);
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
int CeedBasisCreateH1_Cuda(CeedElemTopology topo, CeedInt dim, CeedInt nnodes,
                           CeedInt nqpts, const CeedScalar *interp,
                           const CeedScalar *grad, const CeedScalar *qref,
                           const CeedScalar *qweight, CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  CeedBasisNonTensor_Cuda *data;
  ierr = CeedCalloc(1, &data); CeedChkBackend(ierr);

  // Copy basis data to GPU
  const CeedInt qBytes = nqpts * sizeof(CeedScalar);
  ierr = cudaMalloc((void **)&data->d_qweight, qBytes); CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(data->d_qweight, qweight, qBytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);

  const CeedInt iBytes = qBytes * nnodes;
  ierr = cudaMalloc((void **)&data->d_interp, iBytes); CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(data->d_interp, interp, iBytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);

  const CeedInt gBytes = qBytes * nnodes * dim;
  ierr = cudaMalloc((void **)&data->d_grad, gBytes); CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(data->d_grad, grad, gBytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);

  // Compile basis kernels
  CeedInt ncomp;
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChkBackend(ierr);
  ierr = CeedCompileCuda(ceed, kernelsNonTensorRef, &data->module, 4,
                         "Q", nqpts,
                         "P", nnodes,
                         "BASIS_DIM", dim,
                         "BASIS_NCOMP", ncomp
                        ); CeedChk_Cu(ceed, ierr);
  ierr = CeedGetKernelCuda(ceed, data->module, "interp", &data->interp);
  CeedChk_Cu(ceed, ierr);
  ierr = CeedGetKernelCuda(ceed, data->module, "grad", &data->grad);
  CeedChk_Cu(ceed, ierr);
  ierr = CeedGetKernelCuda(ceed, data->module, "weight", &data->weight);
  CeedChk_Cu(ceed, ierr);

  ierr = CeedBasisSetData(basis, data); CeedChkBackend(ierr);

  // Register backend functions
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Apply",
                                CeedBasisApplyNonTensor_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Destroy",
                                CeedBasisDestroyNonTensor_Cuda); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
