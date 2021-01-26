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

#include <ceed.h>
#include <ceed-backend.h>
#include <stddef.h>
#include "ceed-cuda-gen.h"
#include "ceed-cuda-gen-operator-build.h"
#include "../cuda/ceed-cuda.h"

//------------------------------------------------------------------------------
// Destroy operator
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Cuda_gen(CeedOperator op) {
  int ierr;
  CeedOperator_Cuda_gen *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChkBackend(ierr);
  ierr = CeedFree(&impl); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply and add to output
//------------------------------------------------------------------------------
static int CeedOperatorApplyAdd_Cuda_gen(CeedOperator op, CeedVector invec,
    CeedVector outvec, CeedRequest *request) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  CeedOperator_Cuda_gen *data;
  ierr = CeedOperatorGetData(op, &data); CeedChkBackend(ierr);
  CeedQFunction qf;
  CeedQFunction_Cuda_gen *qf_data;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChkBackend(ierr);
  ierr = CeedQFunctionGetData(qf, &qf_data); CeedChkBackend(ierr);
  CeedInt nelem, numinputfields, numoutputfields;
  ierr = CeedOperatorGetNumElements(op, &nelem); CeedChkBackend(ierr);
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChkBackend(ierr);
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &opinputfields, &opoutputfields);
  CeedChkBackend(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, &qfinputfields, &qfoutputfields);
  CeedChkBackend(ierr);
  CeedEvalMode emode;
  CeedVector vec, outvecs[16] = {};

  //Creation of the operator
  ierr = CeedCudaGenOperatorBuild(op); CeedChkBackend(ierr);

  // Input vectors
  for (CeedInt i = 0; i < numinputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChkBackend(ierr);
    if (emode == CEED_EVAL_WEIGHT) { // Skip
      data->fields.in[i] = NULL;
    } else {
      // Get input vector
      ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChkBackend(ierr);
      if (vec == CEED_VECTOR_ACTIVE) vec = invec;
      ierr = CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE, &data->fields.in[i]);
      CeedChkBackend(ierr);
    }
  }

  // Output vectors
  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChkBackend(ierr);
    if (emode == CEED_EVAL_WEIGHT) { // Skip
      data->fields.out[i] = NULL;
    } else {
      // Get output vector
      ierr = CeedOperatorFieldGetVector(opoutputfields[i], &vec);
      CeedChkBackend(ierr);
      if (vec == CEED_VECTOR_ACTIVE) vec = outvec;
      outvecs[i] = vec;
      // Check for multiple output modes
      CeedInt index = -1;
      for (CeedInt j = 0; j < i; j++) {
        if (vec == outvecs[j]) {
          index = j;
          break;
        }
      }
      if (index == -1) {
        ierr = CeedVectorGetArray(vec, CEED_MEM_DEVICE, &data->fields.out[i]);
        CeedChkBackend(ierr);
      } else {
        data->fields.out[i] = data->fields.out[index];
      }
    }
  }

  // Get context data
  CeedQFunctionContext ctx;
  ierr = CeedQFunctionGetInnerContext(qf, &ctx); CeedChkBackend(ierr);
  if (ctx) {
    ierr = CeedQFunctionContextGetData(ctx, CEED_MEM_DEVICE, &qf_data->d_c);
    CeedChkBackend(ierr);
  }

  // Apply operator
  void *opargs[] = {(void *) &nelem, &qf_data->d_c, &data->indices,
                    &data->fields, &data->B, &data->G, &data->W
                   };
  const CeedInt dim = data->dim;
  const CeedInt Q1d = data->Q1d;
  const CeedInt P1d = data->maxP1d;
  const CeedInt thread1d = CeedIntMax(Q1d, P1d);
  if (dim==1) {
    const CeedInt elemsPerBlock = 32;
    CeedInt grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)
                                           ? 1 : 0 );
    CeedInt sharedMem = elemsPerBlock*thread1d*sizeof(CeedScalar);
    ierr = CeedRunKernelDimSharedCuda(ceed, data->op, grid, thread1d, 1,
                                      elemsPerBlock, sharedMem, opargs);
  } else if (dim==2) {
    const CeedInt elemsPerBlock = thread1d<4? 16 : 2;
    CeedInt grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)
                                           ? 1 : 0 );
    CeedInt sharedMem = elemsPerBlock*thread1d*thread1d*sizeof(CeedScalar);
    ierr = CeedRunKernelDimSharedCuda(ceed, data->op, grid, thread1d, thread1d,
                                      elemsPerBlock, sharedMem, opargs);
  } else if (dim==3) {
    const CeedInt elemsPerBlock = thread1d<6? 4 : (thread1d<8? 2 : 1);
    CeedInt grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)
                                           ? 1 : 0 );
    CeedInt sharedMem = elemsPerBlock*thread1d*thread1d*sizeof(CeedScalar);
    ierr = CeedRunKernelDimSharedCuda(ceed, data->op, grid, thread1d, thread1d,
                                      elemsPerBlock, sharedMem, opargs);
  }
  CeedChkBackend(ierr);

  // Restore input arrays
  for (CeedInt i = 0; i < numinputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChkBackend(ierr);
    if (emode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChkBackend(ierr);
      if (vec == CEED_VECTOR_ACTIVE) vec = invec;
      ierr = CeedVectorRestoreArrayRead(vec, &data->fields.in[i]);
      CeedChkBackend(ierr);
    }
  }

  // Restore output arrays
  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChkBackend(ierr);
    if (emode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      ierr = CeedOperatorFieldGetVector(opoutputfields[i], &vec);
      CeedChkBackend(ierr);
      if (vec == CEED_VECTOR_ACTIVE) vec = outvec;
      // Check for multiple output modes
      CeedInt index = -1;
      for (CeedInt j = 0; j < i; j++) {
        if (vec == outvecs[j]) {
          index = j;
          break;
        }
      }
      if (index == -1) {
        ierr = CeedVectorRestoreArray(vec, &data->fields.out[i]);
        CeedChkBackend(ierr);
      }
    }
  }

  // Restore context data
  if (ctx) {
    ierr = CeedQFunctionContextRestoreData(ctx, &qf_data->d_c);
    CeedChkBackend(ierr);
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create FDM element inverse not supported
//------------------------------------------------------------------------------
static int CeedOperatorCreateFDMElementInverse_Cuda(CeedOperator op) {
  // LCOV_EXCL_START
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  return CeedError(ceed, CEED_ERROR_BACKEND,
                   "Backend does not implement FDM inverse creation");
  // LCOV_EXCL_STOP
}

//------------------------------------------------------------------------------
// Create operator
//------------------------------------------------------------------------------
int CeedOperatorCreate_Cuda_gen(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  CeedOperator_Cuda_gen *impl;

  ierr = CeedCalloc(1, &impl); CeedChkBackend(ierr);
  ierr = CeedOperatorSetData(op, impl); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "Operator", op, "CreateFDMElementInverse",
                                CeedOperatorCreateFDMElementInverse_Cuda);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd",
                                CeedOperatorApplyAdd_Cuda_gen); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "Destroy",
                                CeedOperatorDestroy_Cuda_gen); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
