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
#include <stddef.h>
#include "ceed-hip-shared.h"
#include "../hip/ceed-hip.h"
#include "../hip/ceed-hip-jit.h"
#include "kernel-strings/hip-shared-basis.h"

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
// ncomp
//------------------------------------------------------------------------------
static int ComputeBasisThreadBlockSizes(const CeedInt dim, const CeedInt P1d,
                                        const CeedInt Q1d,
                                        const CeedInt ncomp, CeedInt *blksizes) {

  // Note that this will use the same block sizes for all dimensions when compiling,
  // but as each basis object is defined for a particular dimension, we will never
  // call any kernels except the ones for the dimension for which we have computed the
  // block sizes.
  const CeedInt thread1d = CeedIntMax(P1d, Q1d);
  switch (dim) {
  case 1: {
    // Interp kernels:
    blksizes[0] = 256;

    // Grad kernels:
    blksizes[1] = 256;

    // Weight kernels:
    blksizes[2] = 256;

  } break;
  case 2: {
    // Interp kernels:
    CeedInt required = thread1d * thread1d * ncomp;
    blksizes[0]  = ComputeBlockSizeFromRequirement(required);

    // Grad kernels: currently use same required minimum threads
    blksizes[1]  = ComputeBlockSizeFromRequirement(required);

    // Weight kernels:
    required = CeedIntMax(64, Q1d * Q1d);
    blksizes[2]  = ComputeBlockSizeFromRequirement(required);

  } break;
  case 3: {
    // Interp kernels:
    CeedInt required = thread1d * thread1d * ncomp;
    blksizes[0]  = ComputeBlockSizeFromRequirement(required);

    // Grad kernels: currently use same required minimum threads
    blksizes[1]  = ComputeBlockSizeFromRequirement(required);

    // Weight kernels:
    required = Q1d * Q1d * Q1d;
    blksizes[2]  = ComputeBlockSizeFromRequirement(required);
  }
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply basis
//------------------------------------------------------------------------------
int CeedBasisApplyTensor_Hip_shared(CeedBasis basis, const CeedInt nelem,
                                    CeedTransposeMode tmode,
                                    CeedEvalMode emode, CeedVector u,
                                    CeedVector v) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  Ceed_Hip *ceed_Hip;
  CeedGetData(ceed, &ceed_Hip); CeedChkBackend(ierr);
  CeedBasis_Hip_shared *data;
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
    ierr = hipMemset(d_v, 0, length * sizeof(CeedScalar)); CeedChkBackend(ierr);
  }

  // Apply basis operation
  switch (emode) {
  case CEED_EVAL_INTERP: {
    CeedInt P1d, Q1d;
    CeedInt blksize = data->blksizes[0];
    ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChkBackend(ierr);
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChkBackend(ierr);
    CeedInt thread1d = CeedIntMax(Q1d, P1d);
    void *interpargs[] = {(void *) &nelem, (void *) &transpose, &data->d_interp1d,
                          &d_u, &d_v
                         };
    if (dim == 1) {
      CeedInt elemsPerBlock = 64*thread1d > 256? 256/thread1d : 64;
      elemsPerBlock = elemsPerBlock>0?elemsPerBlock:1;
      CeedInt grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)
                                             ? 1 : 0 );
      CeedInt sharedMem = elemsPerBlock*thread1d*sizeof(CeedScalar);
      ierr = CeedRunKernelDimSharedHip(ceed, data->interp, grid, thread1d, 1,
                                       elemsPerBlock, sharedMem,
                                       interpargs); CeedChkBackend(ierr);
    } else if (dim == 2) {
      // Check if required threads is small enough to do multiple elems
      const CeedInt elemsPerBlock = CeedIntMax(blksize/(thread1d*thread1d*ncomp), 1);
      CeedInt grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)
                                             ? 1 : 0 );
      CeedInt sharedMem = ncomp*elemsPerBlock*thread1d*thread1d*sizeof(CeedScalar);
      ierr = CeedRunKernelDimSharedHip(ceed, data->interp, grid, thread1d, thread1d,
                                       ncomp*elemsPerBlock, sharedMem,
                                       interpargs); CeedChkBackend(ierr);
    } else if (dim == 3) {
      CeedInt elemsPerBlock = 1;
      CeedInt grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)
                                             ? 1 : 0 );
      CeedInt sharedMem = ncomp*elemsPerBlock*thread1d*thread1d*sizeof(CeedScalar);
      ierr = CeedRunKernelDimSharedHip(ceed, data->interp, grid, thread1d, thread1d,
                                       ncomp*elemsPerBlock, sharedMem,
                                       interpargs); CeedChkBackend(ierr);
    }
  } break;
  case CEED_EVAL_GRAD: {
    CeedInt P1d, Q1d;
    CeedInt blksize = data->blksizes[1];
    ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChkBackend(ierr);
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChkBackend(ierr);
    CeedInt thread1d = CeedIntMax(Q1d, P1d);
    void *gradargs[] = {(void *) &nelem, (void *) &transpose, &data->d_interp1d,
                        &data->d_grad1d, &d_u, &d_v
                       };
    if (dim == 1) {
      CeedInt elemsPerBlock = 64*thread1d > 256? 256/thread1d : 64;
      elemsPerBlock = elemsPerBlock>0?elemsPerBlock:1;
      CeedInt grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)
                                             ? 1 : 0 );
      CeedInt sharedMem = elemsPerBlock*thread1d*sizeof(CeedScalar);
      ierr = CeedRunKernelDimSharedHip(ceed, data->grad, grid, thread1d, 1,
                                       elemsPerBlock, sharedMem, gradargs);
      CeedChkBackend(ierr);
    } else if (dim == 2) {
      // Check if required threads is small enough to do multiple elems
      const CeedInt elemsPerBlock = CeedIntMax(blksize/(thread1d*thread1d*ncomp), 1);
      CeedInt grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)
                                             ? 1 : 0 );
      CeedInt sharedMem = ncomp*elemsPerBlock*thread1d*thread1d*sizeof(CeedScalar);
      ierr = CeedRunKernelDimSharedHip(ceed, data->grad, grid, thread1d, thread1d,
                                       ncomp*elemsPerBlock, sharedMem,
                                       gradargs); CeedChkBackend(ierr);
    } else if (dim == 3) {
      CeedInt elemsPerBlock = 1;
      CeedInt grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)
                                             ? 1 : 0 );
      CeedInt sharedMem = ncomp*elemsPerBlock*thread1d*thread1d*sizeof(CeedScalar);
      ierr = CeedRunKernelDimSharedHip(ceed, data->grad, grid, thread1d, thread1d,
                                       ncomp*elemsPerBlock, sharedMem,
                                       gradargs); CeedChkBackend(ierr);
    }
  } break;
  case CEED_EVAL_WEIGHT: {
    CeedInt Q1d;
    CeedInt blksize = data->blksizes[2];
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChkBackend(ierr);
    void *weightargs[] = {(void *) &nelem, (void *) &data->d_qweight1d, &d_v};
    if (dim == 1) {
      const CeedInt optElems = blksize/Q1d;
      const CeedInt elemsPerBlock = optElems>0?optElems:1;
      const CeedInt gridsize = nelem/elemsPerBlock + ( (
                                 nelem/elemsPerBlock*elemsPerBlock<nelem)? 1 : 0 );
      ierr = CeedRunKernelDimHip(ceed, data->weight, gridsize, Q1d,
                                 elemsPerBlock, 1, weightargs);
      CeedChkBackend(ierr);
    } else if (dim == 2) {
      const CeedInt optElems = blksize/(Q1d*Q1d);
      const CeedInt elemsPerBlock = optElems>0?optElems:1;
      const CeedInt gridsize = nelem/elemsPerBlock + ( (
                                 nelem/elemsPerBlock*elemsPerBlock<nelem)? 1 : 0 );
      ierr = CeedRunKernelDimHip(ceed, data->weight, gridsize, Q1d, Q1d,
                                 elemsPerBlock, weightargs);
      CeedChkBackend(ierr);
    } else if (dim == 3) {
      const CeedInt gridsize = nelem;
      ierr = CeedRunKernelDimHip(ceed, data->weight, gridsize, Q1d, Q1d, Q1d,
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
static int CeedBasisDestroy_Hip_shared(CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);

  CeedBasis_Hip_shared *data;
  ierr = CeedBasisGetData(basis, &data); CeedChkBackend(ierr);

  CeedChk_Hip(ceed, hipModuleUnload(data->module));

  ierr = hipFree(data->d_qweight1d); CeedChk_Hip(ceed, ierr);
  ierr = hipFree(data->d_interp1d); CeedChk_Hip(ceed, ierr);
  ierr = hipFree(data->d_grad1d); CeedChk_Hip(ceed, ierr);
  ierr = hipFree(data->d_collograd1d); CeedChk_Hip(ceed, ierr);

  ierr = CeedFree(&data); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create tensor basis
//------------------------------------------------------------------------------
int CeedBasisCreateTensorH1_Hip_shared(CeedInt dim, CeedInt P1d, CeedInt Q1d,
                                       const CeedScalar *interp1d,
                                       const CeedScalar *grad1d,
                                       const CeedScalar *qref1d,
                                       const CeedScalar *qweight1d,
                                       CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  CeedBasis_Hip_shared *data;
  ierr = CeedCalloc(1, &data); CeedChkBackend(ierr);

  // Copy basis data to GPU
  const CeedInt qBytes = Q1d * sizeof(CeedScalar);
  ierr = hipMalloc((void **)&data->d_qweight1d, qBytes); CeedChk_Hip(ceed, ierr);
  ierr = hipMemcpy(data->d_qweight1d, qweight1d, qBytes,
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);

  const CeedInt iBytes = qBytes * P1d;
  ierr = hipMalloc((void **)&data->d_interp1d, iBytes); CeedChk_Hip(ceed, ierr);
  ierr = hipMemcpy(data->d_interp1d, interp1d, iBytes,
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);

  ierr = hipMalloc((void **)&data->d_grad1d, iBytes); CeedChk_Hip(ceed, ierr);
  ierr = hipMemcpy(data->d_grad1d, grad1d, iBytes,
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);

  // Compute collocated gradient and copy to GPU
  data->d_collograd1d = NULL;
  if (dim == 3 && Q1d >= P1d) {
    CeedScalar *collograd1d;
    ierr = CeedMalloc(Q1d*Q1d, &collograd1d); CeedChkBackend(ierr);
    ierr = CeedBasisGetCollocatedGrad(basis, collograd1d); CeedChkBackend(ierr);
    ierr = hipMalloc((void **)&data->d_collograd1d, qBytes * Q1d);
    CeedChk_Hip(ceed, ierr);
    ierr = hipMemcpy(data->d_collograd1d, collograd1d, qBytes * Q1d,
                     hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);
    ierr = CeedFree(&collograd1d); CeedChkBackend(ierr);
  }

  // Set number of threads per block for basis kernels
  CeedInt ncomp;
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChkBackend(ierr);
  ierr = ComputeBasisThreadBlockSizes(dim, P1d, Q1d, ncomp, data->blksizes);
  CeedChkBackend(ierr);

  // Compile basis kernels
  ierr = CeedCompileHip(ceed, kernelsShared, &data->module, 11,
                        "Q1D", Q1d,
                        "P1D", P1d,
                        "T1D", CeedIntMax(Q1d, P1d),
                        "BASIS_BUF_LEN", ncomp * CeedIntPow(Q1d > P1d ?
                            Q1d : P1d, dim),
                        "BASIS_DIM", dim,
                        "BASIS_NCOMP", ncomp,
                        "BASIS_ELEMSIZE", CeedIntPow(P1d, dim),
                        "BASIS_NQPT", CeedIntPow(Q1d, dim),
                        "INTERP_BLKSIZE", data->blksizes[0],
                        "GRAD_BLKSIZE", data->blksizes[1],
                        "WEIGHT_BLKSIZE", data->blksizes[2]
                       ); CeedChkBackend(ierr);
  ierr = CeedGetKernelHip(ceed, data->module, "interp", &data->interp);
  CeedChkBackend(ierr);
  ierr = CeedGetKernelHip(ceed, data->module, "grad", &data->grad);
  CeedChkBackend(ierr);
  ierr = CeedGetKernelHip(ceed, data->module, "weight", &data->weight);
  CeedChkBackend(ierr);

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
