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

#include <ceed-backend.h>
#include "ceed-cuda-gen.h"
#include "ceed-cuda-gen-operator-build.h"
#include "../cuda/ceed-cuda.h"

static int CeedOperatorDestroy_Cuda_gen(CeedOperator op) {
  int ierr;
  CeedOperator_Cuda_gen *impl;
  ierr = CeedOperatorGetData(op, (void *)&impl); CeedChk(ierr);

  // for (CeedInt i = 0; i < impl->numein + impl->numeout; i++) {
  //   ierr = CeedVectorDestroy(&impl->evecs[i]); CeedChk(ierr);
  // }
  // ierr = CeedFree(&impl->evecs); CeedChk(ierr);
  // ierr = CeedFree(&impl->edata); CeedChk(ierr);

  // for (CeedInt i = 0; i < impl->numein; i++) {
  //   ierr = CeedVectorDestroy(&impl->qvecsin[i]); CeedChk(ierr);
  // }
  // ierr = CeedFree(&impl->qvecsin); CeedChk(ierr);

  // for (CeedInt i = 0; i < impl->numeout; i++) {
  //   ierr = CeedVectorDestroy(&impl->qvecsout[i]); CeedChk(ierr);
  // }
  // ierr = CeedFree(&impl->qvecsout); CeedChk(ierr);

  // ierr = CeedFree(&impl->W); CeedChk(ierr);

  ierr = CeedFree(&impl); CeedChk(ierr);
  return 0;
}

static int CeedOperatorApply_Cuda_gen(CeedOperator op, CeedVector invec,
                                  CeedVector outvec, CeedRequest *request) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  CeedOperator_Cuda_gen *data;
  ierr = CeedOperatorGetData(op, (void *)&data); CeedChk(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
  CeedInt nelem, numinputfields, numoutputfields;
  ierr = CeedOperatorGetNumElements(op, &nelem); CeedChk(ierr);
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChk(ierr);
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &opinputfields, &opoutputfields);
  CeedChk(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, &qfinputfields, &qfoutputfields);
  CeedChk(ierr);
  CeedEvalMode emode;
  CeedVector vec;

  //Creation of the operator
  ierr = CeedCudaGenOperatorBuild(op); CeedChk(ierr);

  // Zero lvecs
  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedOperatorFieldGetVector(opoutputfields[i], &vec); CeedChk(ierr);
    if (vec == CEED_VECTOR_ACTIVE)
      vec = outvec;
    ierr = CeedVectorSetValue(vec, 0.0); CeedChk(ierr);
  }

  // Input vectors
  for (CeedInt i = 0; i < numinputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChk(ierr);
    if (emode == CEED_EVAL_WEIGHT) { // Skip
      data->fields.in[i] = NULL;
    } else {
      // Get input vector
      ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE) vec = invec;
      ierr = CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE, &data->fields.in[i]);
      CeedChk(ierr);
    }
  }

  // Output vectors
  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChk(ierr);
    if (emode == CEED_EVAL_WEIGHT) { // Skip
      data->fields.out[i] = NULL;
    } else {
      // Get output vector
      ierr = CeedOperatorFieldGetVector(opoutputfields[i], &vec); CeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE) vec = outvec;
      ierr = CeedVectorGetArray(vec, CEED_MEM_DEVICE, &data->fields.out[i]);
      CeedChk(ierr);
    }
  }

  // Apply operator
  void *opargs[] = {(void *) &nelem, &data->indices, &data->fields, &data->B, &data->G, &data->W};
  const CeedInt dim = data->dim;
  const CeedInt Q1d = data->Q1d;
  if (dim==1) {
    const CeedInt elemsPerBlock = 1;
    CeedInt grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)? 1 : 0 );
    CeedInt sharedMem = elemsPerBlock*Q1d*sizeof(CeedScalar);
    ierr = run_kernel_dim_shared(ceed, data->op, grid, Q1d, 1, elemsPerBlock, sharedMem, opargs);
  } else if (dim==2) {
    const CeedInt elemsPerBlock = 1;
    CeedInt grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)? 1 : 0 );
    CeedInt sharedMem = elemsPerBlock*Q1d*Q1d*sizeof(CeedScalar);
    ierr = run_kernel_dim_shared(ceed, data->op, grid, Q1d, Q1d, elemsPerBlock, sharedMem, opargs);
  } else if (dim==3) {
    const CeedInt elemsPerBlock = 1;
    CeedInt grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)? 1 : 0 );
    CeedInt sharedMem = elemsPerBlock*Q1d*Q1d*sizeof(CeedScalar);
    ierr = run_kernel_dim_shared(ceed, data->op, grid, Q1d, Q1d, elemsPerBlock, sharedMem, opargs);
  }
  CeedChk(ierr);

  // Restore input arrays
  for (CeedInt i = 0; i < numinputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChk(ierr);
    if (emode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE) vec = invec;
      ierr = CeedVectorRestoreArrayRead(vec, &data->fields.in[i]);
      CeedChk(ierr);
    }
  }

  // Restore output arrays
  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChk(ierr);
    if (emode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      ierr = CeedOperatorFieldGetVector(opoutputfields[i], &vec); CeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE) vec = outvec;
      ierr = CeedVectorRestoreArray(vec, &data->fields.out[i]);
      CeedChk(ierr);
    }
  }

  return 0;
}

int CeedOperatorCreate_Cuda_gen(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  CeedOperator_Cuda_gen *impl;

  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  ierr = CeedOperatorSetData(op, (void *)&impl);

  ierr = CeedSetBackendFunction(ceed, "Operator", op, "Apply",
                                CeedOperatorApply_Cuda_gen); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "Destroy",
                                CeedOperatorDestroy_Cuda_gen); CeedChk(ierr);
  return 0;
}

int CeedCompositeOperatorCreate_Cuda_gen(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  return CeedError(ceed, 1, "Backend does not implement composite operators");
}
