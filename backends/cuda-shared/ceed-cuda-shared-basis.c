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
#include <string.h>
#include "ceed-cuda-shared.h"
#include "../cuda/ceed-cuda-jit.h"

//------------------------------------------------------------------------------
// Device initalization
//------------------------------------------------------------------------------
int CeedCudaInitInterp(CeedScalar *d_B, CeedInt P1d, CeedInt Q1d,
                       CeedScalar **c_B);
int CeedCudaInitInterpGrad(CeedScalar *d_B, CeedScalar *d_G, CeedInt P1d,
                           CeedInt Q1d, CeedScalar **c_B_ptr,
                           CeedScalar **c_G_ptr);

//------------------------------------------------------------------------------
// Apply basis
//------------------------------------------------------------------------------
int CeedBasisApplyTensor_Cuda_shared(CeedBasis basis, const CeedInt nelem,
                                     CeedTransposeMode tmode,
                                     CeedEvalMode emode, CeedVector u,
                                     CeedVector v) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  Ceed_Cuda *ceed_Cuda;
  CeedGetData(ceed, &ceed_Cuda); CeedChkBackend(ierr);
  CeedBasis_Cuda_shared *data;
  CeedBasisGetData(basis, &data); CeedChkBackend(ierr);
  const CeedInt transpose = tmode == CEED_TRANSPOSE;
  CeedInt dim, ncomp;
  ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChkBackend(ierr);

  // Read vectors
  const CeedScalar *d_u;
  CeedScalar *d_v;
  if (emode != CEED_EVAL_WEIGHT) {
    ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u); CeedChkBackend(ierr);
  }
  ierr = CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v); CeedChkBackend(ierr);

  // Clear v for transpose mode
  if (tmode == CEED_TRANSPOSE) {
    CeedInt length;
    ierr = CeedVectorGetLength(v, &length); CeedChkBackend(ierr);
    ierr = cudaMemset(d_v, 0, length * sizeof(CeedScalar)); CeedChkBackend(ierr);
  }

  // Apply basis operation
  switch (emode) {
  case CEED_EVAL_INTERP: {
    CeedInt P1d, Q1d;
    ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChkBackend(ierr);
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChkBackend(ierr);
    CeedInt thread1d = CeedIntMax(Q1d, P1d);
    ierr = CeedCudaInitInterp(data->d_interp1d, P1d, Q1d, &data->c_B);
    CeedChkBackend(ierr);
    void *interpargs[] = {(void *) &nelem, (void *) &transpose, &data->c_B,
                          &d_u, &d_v
                         };
    if (dim == 1) {
      CeedInt elemsPerBlock = 32;
      CeedInt grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)
                                             ? 1 : 0 );
      CeedInt sharedMem = elemsPerBlock*thread1d*sizeof(CeedScalar);
      ierr = CeedRunKernelDimSharedCuda(ceed, data->interp, grid, thread1d, 1,
                                        elemsPerBlock, sharedMem,
                                        interpargs); CeedChkBackend(ierr);
    } else if (dim == 2) {
      const CeedInt optElems[7] = {0,32,8,6,4,2,8};
      // elemsPerBlock must be at least 1
      CeedInt elemsPerBlock = CeedIntMax(thread1d<7?optElems[thread1d]/ncomp:1, 1);
      CeedInt grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)
                                             ? 1 : 0 );
      CeedInt sharedMem = ncomp*elemsPerBlock*thread1d*thread1d*sizeof(CeedScalar);
      ierr = CeedRunKernelDimSharedCuda(ceed, data->interp, grid, thread1d, thread1d,
                                        ncomp*elemsPerBlock, sharedMem,
                                        interpargs); CeedChkBackend(ierr);
    } else if (dim == 3) {
      CeedInt elemsPerBlock = 1;
      CeedInt grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)
                                             ? 1 : 0 );
      CeedInt sharedMem = ncomp*elemsPerBlock*thread1d*thread1d*sizeof(CeedScalar);
      ierr = CeedRunKernelDimSharedCuda(ceed, data->interp, grid, thread1d, thread1d,
                                        ncomp*elemsPerBlock, sharedMem,
                                        interpargs); CeedChkBackend(ierr);
    }
  } break;
  case CEED_EVAL_GRAD: {
    CeedInt P1d, Q1d;
    ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChkBackend(ierr);
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChkBackend(ierr);
    CeedInt thread1d = CeedIntMax(Q1d, P1d);
    ierr = CeedCudaInitInterpGrad(data->d_interp1d, data->d_grad1d, P1d,
                                  Q1d, &data->c_B, &data->c_G);
    CeedChkBackend(ierr);
    void *gradargs[] = {(void *) &nelem, (void *) &transpose, &data->c_B,
                        &data->c_G, &d_u, &d_v
                       };
    if (dim == 1) {
      CeedInt elemsPerBlock = 32;
      CeedInt grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)
                                             ? 1 : 0 );
      CeedInt sharedMem = elemsPerBlock*thread1d*sizeof(CeedScalar);
      ierr = CeedRunKernelDimSharedCuda(ceed, data->grad, grid, thread1d, 1,
                                        elemsPerBlock, sharedMem, gradargs);
      CeedChkBackend(ierr);
    } else if (dim == 2) {
      const CeedInt optElems[7] = {0,32,8,6,4,2,8};
      // elemsPerBlock must be at least 1
      CeedInt elemsPerBlock = CeedIntMax(thread1d<7?optElems[thread1d]/ncomp:1, 1);
      CeedInt grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)
                                             ? 1 : 0 );
      CeedInt sharedMem = ncomp*elemsPerBlock*thread1d*thread1d*sizeof(CeedScalar);
      ierr = CeedRunKernelDimSharedCuda(ceed, data->grad, grid, thread1d, thread1d,
                                        ncomp*elemsPerBlock, sharedMem,
                                        gradargs); CeedChkBackend(ierr);
    } else if (dim == 3) {
      CeedInt elemsPerBlock = 1;
      CeedInt grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)
                                             ? 1 : 0 );
      CeedInt sharedMem = ncomp*elemsPerBlock*thread1d*thread1d*sizeof(CeedScalar);
      ierr = CeedRunKernelDimSharedCuda(ceed, data->grad, grid, thread1d, thread1d,
                                        ncomp*elemsPerBlock, sharedMem,
                                        gradargs); CeedChkBackend(ierr);
    }
  } break;
  case CEED_EVAL_WEIGHT: {
    CeedInt Q1d;
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChkBackend(ierr);
    void *weightargs[] = {(void *) &nelem, (void *) &data->d_qweight1d, &d_v};
    if (dim == 1) {
      const CeedInt elemsPerBlock = 32/Q1d;
      const CeedInt gridsize = nelem/elemsPerBlock + ( (
                                 nelem/elemsPerBlock*elemsPerBlock<nelem)? 1 : 0 );
      ierr = CeedRunKernelDimCuda(ceed, data->weight, gridsize, Q1d,
                                  elemsPerBlock, 1, weightargs);
      CeedChkBackend(ierr);
    } else if (dim == 2) {
      const CeedInt optElems = 32/(Q1d*Q1d);
      const CeedInt elemsPerBlock = optElems>0?optElems:1;
      const CeedInt gridsize = nelem/elemsPerBlock + ( (
                                 nelem/elemsPerBlock*elemsPerBlock<nelem)? 1 : 0 );
      ierr = CeedRunKernelDimCuda(ceed, data->weight, gridsize, Q1d, Q1d,
                                  elemsPerBlock, weightargs);
      CeedChkBackend(ierr);
    } else if (dim == 3) {
      const CeedInt gridsize = nelem;
      ierr = CeedRunKernelDimCuda(ceed, data->weight, gridsize, Q1d, Q1d, Q1d,
                                  weightargs);
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
  if (emode != CEED_EVAL_WEIGHT) {
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

  ierr = cudaFree(data->d_qweight1d); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree(data->d_interp1d); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree(data->d_grad1d); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree(data->d_collograd1d); CeedChk_Cu(ceed, ierr);

  ierr = CeedFree(&data); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create tensor basis
//------------------------------------------------------------------------------
int CeedBasisCreateTensorH1_Cuda_shared(CeedInt dim, CeedInt P1d, CeedInt Q1d,
                                        const CeedScalar *interp1d,
                                        const CeedScalar *grad1d,
                                        const CeedScalar *qref1d,
                                        const CeedScalar *qweight1d,
                                        CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  CeedBasis_Cuda_shared *data;
  ierr = CeedCalloc(1, &data); CeedChkBackend(ierr);

  // Copy basis data to GPU
  const CeedInt qBytes = Q1d * sizeof(CeedScalar);
  ierr = cudaMalloc((void **)&data->d_qweight1d, qBytes); CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(data->d_qweight1d, qweight1d, qBytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);

  const CeedInt iBytes = qBytes * P1d;
  ierr = cudaMalloc((void **)&data->d_interp1d, iBytes); CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(data->d_interp1d, interp1d, iBytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);

  ierr = cudaMalloc((void **)&data->d_grad1d, iBytes); CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(data->d_grad1d, grad1d, iBytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);

  // Compute collocated gradient and copy to GPU
  data->d_collograd1d = NULL;
  if (dim == 3 && Q1d >= P1d) {
    CeedScalar *collograd1d;
    ierr = CeedMalloc(Q1d*Q1d, &collograd1d); CeedChkBackend(ierr);
    ierr = CeedBasisGetCollocatedGrad(basis, collograd1d); CeedChkBackend(ierr);
    ierr = cudaMalloc((void **)&data->d_collograd1d, qBytes * Q1d);
    CeedChk_Cu(ceed, ierr);
    ierr = cudaMemcpy(data->d_collograd1d, collograd1d, qBytes * Q1d,
                      cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);
    ierr = CeedFree(&collograd1d); CeedChkBackend(ierr);
  }

  // Compile basis kernels
  CeedInt ncomp;
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChkBackend(ierr);
  char source_path[CEED_MAX_PATH_LEN] = __FILE__;
  CeedInt end = strrchr(source_path, '/') - source_path;
  strncpy(&source_path[end], "/kernels/cuda-shared-basis.h", 29);
  char *basisKernels;
  ierr = CeedLoadSourceToBuffer(ceed, source_path, &basisKernels);
  CeedChkBackend(ierr);
  ierr = CeedCompileCuda(ceed, basisKernels, &data->module, 8,
                         "Q1D", Q1d,
                         "P1D", P1d,
                         "T1D", CeedIntMax(Q1d, P1d),
                         "BASIS_BUF_LEN", ncomp * CeedIntPow(Q1d > P1d ?
                             Q1d : P1d, dim),
                         "BASIS_DIM", dim,
                         "BASIS_NCOMP", ncomp,
                         "BASIS_ELEMSIZE", CeedIntPow(P1d, dim),
                         "BASIS_NQPT", CeedIntPow(Q1d, dim)
                        ); CeedChkBackend(ierr);
  ierr = CeedFree(&basisKernels); CeedChkBackend(ierr);
  ierr = CeedGetKernelCuda(ceed, data->module, "interp", &data->interp);
  CeedChkBackend(ierr);
  ierr = CeedGetKernelCuda(ceed, data->module, "grad", &data->grad);
  CeedChkBackend(ierr);
  ierr = CeedGetKernelCuda(ceed, data->module, "weight", &data->weight);
  CeedChkBackend(ierr);

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
