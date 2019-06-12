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

/*
  Setup infields or outfields
 */
// static int CeedOperatorSetupFields_Cuda_gen(CeedQFunction qf, CeedOperator op,
//                                         bool inOrOut, CeedVector *evecs,
//                                         CeedVector *qvecs, CeedInt starte,
//                                         CeedInt numfields, CeedInt Q, CeedInt numelements) {
//   CeedInt dim, ierr, ncomp;
//   Ceed ceed;
//   ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
//   CeedBasis basis;
//   CeedElemRestriction Erestrict;
//   CeedOperatorField *opfields;
//   CeedQFunctionField *qffields;
//   if (inOrOut) {
//     ierr = CeedOperatorGetFields(op, NULL, &opfields);
//     CeedChk(ierr);
//     ierr = CeedQFunctionGetFields(qf, NULL, &qffields);
//     CeedChk(ierr);
//   } else {
//     ierr = CeedOperatorGetFields(op, &opfields, NULL);
//     CeedChk(ierr);
//     ierr = CeedQFunctionGetFields(qf, &qffields, NULL);
//     CeedChk(ierr);
//   }

//   // Loop over fields
//   for (CeedInt i = 0; i < numfields; i++) {
//     CeedEvalMode emode;
//     ierr = CeedQFunctionFieldGetEvalMode(qffields[i], &emode); CeedChk(ierr);

//     if (emode != CEED_EVAL_WEIGHT) {
//       ierr = CeedOperatorFieldGetElemRestriction(opfields[i], &Erestrict);
//       CeedChk(ierr);
//       ierr = CeedElemRestrictionCreateVector(Erestrict, NULL,
//                                              &evecs[i + starte]);
//       CeedChk(ierr);
//     }

//     switch (emode) {
//     case CEED_EVAL_NONE:
//       ierr = CeedQFunctionFieldGetNumComponents(qffields[i], &ncomp);
//       CeedChk(ierr);
//       ierr = CeedVectorCreate(ceed, numelements * Q * ncomp, &qvecs[i]);
//       CeedChk(ierr);
//       break;
//     case CEED_EVAL_INTERP:
//       ierr = CeedQFunctionFieldGetNumComponents(qffields[i], &ncomp);
//       CeedChk(ierr);
//       ierr = CeedVectorCreate(ceed, numelements * Q * ncomp, &qvecs[i]);
//       CeedChk(ierr);
//       break;
//     case CEED_EVAL_GRAD:
//       ierr = CeedOperatorFieldGetBasis(opfields[i], &basis); CeedChk(ierr);
//       ierr = CeedQFunctionFieldGetNumComponents(qffields[i], &ncomp);
//       ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
//       ierr = CeedVectorCreate(ceed, numelements * Q * ncomp * dim, &qvecs[i]);
//       CeedChk(ierr);
//       break;
//     case CEED_EVAL_WEIGHT: // Only on input fields
//       ierr = CeedOperatorFieldGetBasis(opfields[i], &basis); CeedChk(ierr);
//       ierr = CeedVectorCreate(ceed, numelements * Q, &qvecs[i]); CeedChk(ierr);
//       ierr = CeedBasisApply(basis, numelements, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT,
//                             NULL, qvecs[i]); CeedChk(ierr);
//       break;
//     case CEED_EVAL_DIV:
//       break; // TODO: Not implemented
//     case CEED_EVAL_CURL:
//       break; // TODO: Not implemented
//     }
//   }
//   return 0;
// }

/*
  CeedOperator needs to connect all the named fields (be they active or passive)
  to the named inputs and outputs of its CeedQFunction.
 */
// static int CeedOperatorSetup_Cuda_gen(CeedOperator op) {
//   int ierr;
//   bool setupdone;
//   ierr = CeedOperatorGetSetupStatus(op, &setupdone); CeedChk(ierr);
//   if (setupdone) return 0;
//   Ceed ceed;
//   ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
//   CeedOperator_Cuda_gen *impl;
//   ierr = CeedOperatorGetData(op, (void *)&impl); CeedChk(ierr);
//   CeedQFunction qf;
//   ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
//   CeedInt Q, numelements, numinputfields, numoutputfields;
//   ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChk(ierr);
//   ierr = CeedOperatorGetNumElements(op, &numelements); CeedChk(ierr);
//   ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
//   CeedChk(ierr);
//   CeedOperatorField *opinputfields, *opoutputfields;
//   ierr = CeedOperatorGetFields(op, &opinputfields, &opoutputfields);
//   CeedChk(ierr);
//   CeedQFunctionField *qfinputfields, *qfoutputfields;
//   ierr = CeedQFunctionGetFields(qf, &qfinputfields, &qfoutputfields);
//   CeedChk(ierr);

//   // Allocate
//   ierr = CeedCalloc(numinputfields + numoutputfields, &impl->evecs);
//   CeedChk(ierr);
//   ierr = CeedCalloc(numinputfields + numoutputfields, &impl->edata);
//   CeedChk(ierr);

//   ierr = CeedCalloc(16, &impl->qvecsin); CeedChk(ierr);
//   ierr = CeedCalloc(16, &impl->qvecsout); CeedChk(ierr);

//   impl->numein = numinputfields; impl->numeout = numoutputfields;

//   // Set up infield and outfield evecs and qvecs
//   // Infields
//   ierr = CeedOperatorSetupFields_Cuda_gen(qf, op, 0,
//                                       impl->evecs, impl->qvecsin, 0,
//                                       numinputfields, Q, numelements);
//   CeedChk(ierr);

//   // Outfields
//   ierr = CeedOperatorSetupFields_Cuda_gen(qf, op, 1,
//                                       impl->evecs, impl->qvecsout,
//                                       numinputfields, numoutputfields, Q, numelements);
//   CeedChk(ierr);

//   ierr = CeedOperatorSetSetupDone(op); CeedChk(ierr);

//   return 0;
// }

static int CeedOperatorApply_Cuda_gen(CeedOperator op, CeedVector invec,
                                  CeedVector outvec, CeedRequest *request) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  CeedOperator_Cuda_gen *data;
  ierr = CeedOperatorGetData(op, (void *)&data); CeedChk(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
  // CeedInt Q, P1d, Q1d = -1, numelements, elemsize, numinputfields, numoutputfields, ncomp, dim;
  CeedInt nelem, numinputfields, numoutputfields;
  // ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChk(ierr);
  ierr = CeedOperatorGetNumElements(op, &nelem); CeedChk(ierr);
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChk(ierr);
  // CeedTransposeMode lmode;
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &opinputfields, &opoutputfields);
  CeedChk(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, &qfinputfields, &qfoutputfields);
  CeedChk(ierr);
  CeedEvalMode emode;
  CeedVector vec;
  // CeedBasis basis;
  // CeedElemRestriction Erestrict;

  //Creation of the operator
  ierr = CeedCudaGenOperatorBuild(op); CeedChk(ierr);

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
      ierr = CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE, (const CeedScalar **) &data->fields.in[i]);
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
      ierr = CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE, (const CeedScalar **) &data->fields.out[i]);
      CeedChk(ierr);
    }
  }

  // printf("const CeedInt Dim = %d;\n", dim);
  // printf("const CeedInt Q1d = %d;\n", Q1d);
  // printf("const CeedInt Q   = %d;\n", Q);
  // printf("for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < nelem; elem += gridDim.x*blockDim.z) {\n");
  // // Input basis apply if needed
  // for (CeedInt i = 0; i < numinputfields; i++) {
  //   printf("// Input field %d\n", i);
  //   // Get elemsize, emode, ncomp
  //   ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict);
  //   CeedChk(ierr);
  //   ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
  //   CeedChk(ierr);
  //   ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
  //   CeedChk(ierr);
  //   ierr = CeedQFunctionFieldGetNumComponents(qfinputfields[i], &ncomp);
  //   CeedChk(ierr);
  //   // Basis action
  //   switch (emode) {
  //   case CEED_EVAL_NONE:
  //     printf("  const CeedInt ncomp_in_%d = %d;\n", i, ncomp);
  //     printf("  CeedScalar r_t%d[ncomp_in_%d*Q1d];\n", i, i);
  //     printf("  readQuads%dd<ncomp_in_%d,Q1d>(elem, d_u%d, r_t%d);\n", dim, i, i, i);
  //     // ierr = CeedVectorSetArray(impl->qvecsin[i], CEED_MEM_DEVICE,
  //     //                           CEED_USE_POINTER,
  //     //                           impl->edata[i]); CeedChk(ierr);
  //     break;
  //   case CEED_EVAL_INTERP:
  //     ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChk(ierr);
  //     ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
  //     printf("  const CeedInt P_in_%d = %d;\n", i, P1d);
  //     printf("  const CeedInt ncomp_in_%d = %d;\n", i, ncomp);
  //     printf("  CeedScalar r_u%d[ncomp_in_%d*P_in_%d];\n", i, i, i);
  //     printf("  readDofs%dd<ncomp_in_%d,P_in_%d>(elem, d_u%d, r_u%d);\n", dim, i, i, i, i);
  //     printf("  CeedScalar r_t%d[ncomp_in_%d*Q1d];\n", i, i);
  //     printf("  interp%dd<ncomp_in_%d,P_in_%d,Q1d>(r_u%d, B%d, r_t%d);\n", dim, i, i, i, i, i);
  //     // ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChk(ierr);
  //     // ierr = CeedBasisApply(basis, numelements, CEED_NOTRANSPOSE,
  //     //                       CEED_EVAL_INTERP, impl->evecs[i],
  //     //                       impl->qvecsin[i]); CeedChk(ierr);
  //     break;
  //   case CEED_EVAL_GRAD:
  //     ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChk(ierr);
  //     printf("  const CeedInt ncomp_in_%d = %d;\n", i, ncomp);
  //     ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
  //     printf("  const CeedInt P_in_%d = %d;\n", i, P1d);
  //     printf("  CeedScalar r_u%d[ncomp_in_%d*P_in_%d];\n", i, i, i);
  //     printf("  readDofs%dd<ncomp_in_%d,P_in_%d>(elem, d_u%d, r_u%d);\n", dim, i, i, i, i);
  //     printf("  CeedScalar r_t%d[ncomp_in_%d*Dim*Q1d];\n", i, i);
  //     printf("  grad%dd<ncomp_in_%d,P_in_%d,Q1d>(r_u%d, B%d, G%d, r_t%d);\n", dim, i, i, i, i, i, i);
  //     // ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChk(ierr);
  //     // ierr = CeedBasisApply(basis, numelements, CEED_NOTRANSPOSE,
  //     //                       CEED_EVAL_GRAD, impl->evecs[i],
  //     //                       impl->qvecsin[i]); CeedChk(ierr);
  //     break;
  //   case CEED_EVAL_WEIGHT:
  //     printf("  CeedScalar r_t%d[Q1d];\n", i);
  //     printf("  weight%dd<Q1d>(W, r_t%d);\n", dim, i);
  //     break; // No action
  //   case CEED_EVAL_DIV:
  //     break; // TODO: Not implemented
  //   case CEED_EVAL_CURL:
  //     break; // TODO: Not implemented
  //   }
  // }
  // // Output pointers
  // for (CeedInt i = 0; i < numoutputfields; i++) {
  //   // ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
  //   // CeedChk(ierr);
  //   // if (emode == CEED_EVAL_NONE) {
  //     // ierr = CeedVectorGetArray(impl->evecs[i + impl->numein], CEED_MEM_DEVICE,
  //     //                           &impl->edata[i + numinputfields]); CeedChk(ierr);
  //     // ierr = CeedQFunctionFieldGetNumComponents(qfoutputfields[i], &ncomp);
  //     // CeedChk(ierr);
  //     // ierr = CeedVectorSetArray(impl->qvecsout[i], CEED_MEM_DEVICE,
  //     //                           CEED_USE_POINTER,
  //     //                           impl->edata[i + numinputfields]);
  //     // CeedChk(ierr);
  //   // }
  // }
  // // Q function
  // printf("// QFunction\n");
  // for (CeedInt i = 0; i < numoutputfields; i++) {
  //   ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
  //   CeedChk(ierr);
  //   ierr = CeedQFunctionFieldGetNumComponents(qfoutputfields[i], &ncomp);
  //   CeedChk(ierr);
  //   if (emode==CEED_EVAL_GRAD)
  //   {
  //     printf("  const CeedInt ncomp_out_%d = %d;\n", i, ncomp);
  //     ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis); CeedChk(ierr);
  //     printf("  CeedScalar r_tt%d[ncomp_out_%d*Dim*Q1d];\n", i, i);
  //   }
  //   if (emode==CEED_EVAL_NONE || emode==CEED_EVAL_INTERP)
  //   {
  //     printf("  const CeedInt ncomp_out_%d = %d;\n", i, ncomp);
  //     printf("  CeedScalar r_tt%d[ncomp_out_%d*Q1d];\n", i, i);
  //   }
  // }
  // //TODO write qfunction load for this backend
  // printf("  qfunction(");
  // for (CeedInt i = 0; i < numinputfields; i++) {
  //   printf("r_t%d, ", i);
  // }
  // for (CeedInt i = 0; i < numoutputfields; i++) {
  //   printf("r_tt%d", i);
  //   if (i<numoutputfields-1)
  //   {
  //     printf(", ");
  //   }
  // }
  // printf(");\n");
  // // ierr = CeedQFunctionApply(qf, numelements * Q, impl->qvecsin, impl->qvecsout);
  // // CeedChk(ierr);

  // // Output basis apply if needed
  // for (CeedInt i = 0; i < numoutputfields; i++) {
  //   printf("// Output field %d\n", i);
  //   // Get elemsize, emode, ncomp
  //   ierr = CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict);
  //   CeedChk(ierr);
  //   ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
  //   CeedChk(ierr);
  //   ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
  //   CeedChk(ierr);
  //   ierr = CeedQFunctionFieldGetNumComponents(qfoutputfields[i], &ncomp);
  //   CeedChk(ierr);
  //   // Basis action
  //   switch (emode) {
  //   case CEED_EVAL_NONE:
  //     printf("  writeQuads%dd<ncomp_out_%d,Q1d>(elem, r_tt%d, d_v%d);\n", dim, i, i, i);
  //     break; // No action
  //   case CEED_EVAL_INTERP:
  //     ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis); CeedChk(ierr);
  //     ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
  //     printf("  const CeedInt P_out_%d = %d;\n", i, P1d);
  //     printf("  CeedScalar r_v%d[ncomp_out_%d*P_out_%d];\n", i, i, i);
  //     printf("  interpTranspose%dd<ncomp_out_%d,P_out_%d,Q1d>(r_tt%d, B%d, r_v%d);\n", dim, i, i, i, i, i);
  //     printf("  writeDofs%dd<ncomp_out_%d,P_out_%d>(elem, r_v%d, d_v%d);\n", dim, i, i, i, i);
  //     // ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis);
  //     // CeedChk(ierr);
  //     // ierr = CeedBasisApply(basis, numelements, CEED_TRANSPOSE,
  //     //                       CEED_EVAL_INTERP, impl->qvecsout[i],
  //     //                       impl->evecs[i + impl->numein]); CeedChk(ierr);
  //     break;
  //   case CEED_EVAL_GRAD:
  //     ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis); CeedChk(ierr);
  //     ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
  //     printf("  const CeedInt P_out_%d = %d;\n", i, P1d);
  //     printf("  CeedScalar r_v%d[ncomp_out_%d*P_out_%d];\n", i, i, i);
  //     printf("  gradTranspose%dd<ncomp_out_%d,P_out_%d,Q1d>(r_tt%d, B%d, G%d, r_v%d);\n", dim, i, i, i ,i, i, i);
  //     printf("  writeDofs%dd<ncomp_out_%d,P_out_%d>(elem, r_v%d, d_v%d);\n", dim, i, i, i, i);
  //     // ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis);
  //     // CeedChk(ierr);
  //     // ierr = CeedBasisApply(basis, numelements, CEED_TRANSPOSE,
  //     //                       CEED_EVAL_GRAD, impl->qvecsout[i],
  //     //                       impl->evecs[i + impl->numein]); CeedChk(ierr);
  //     break;
  //   case CEED_EVAL_WEIGHT: {
  //     // Ceed ceed;
  //     // ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  //     // return CeedError(ceed, 1,
  //     //                  "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
  //     break; // Should not occur
  //   }
  //   case CEED_EVAL_DIV:
  //     break; // TODO: Not implemented
  //   case CEED_EVAL_CURL:
  //     break; // TODO: Not implemented
  //   }
  // }


  // // Output restriction
  // for (CeedInt i = 0; i < numoutputfields; i++) {
  //   // Restore evec
  //   // ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
  //   // CeedChk(ierr);
  //   // if (emode == CEED_EVAL_NONE) {
  //   //   ierr = CeedVectorRestoreArray(impl->evecs[i+impl->numein],
  //   //                                 &impl->edata[i + numinputfields]);
  //   //   CeedChk(ierr);
  //   // }
  //   // // Get output vector
  //   // ierr = CeedOperatorFieldGetVector(opoutputfields[i], &vec); CeedChk(ierr);
  //   // // Active
  //   // if (vec == CEED_VECTOR_ACTIVE)
  //   //   vec = outvec;
  //   // // Restrict
  //   // ierr = CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict);
  //   // CeedChk(ierr);
  //   // ierr = CeedOperatorFieldGetLMode(opoutputfields[i], &lmode); CeedChk(ierr);
  //   // ierr = CeedElemRestrictionApply(Erestrict, CEED_TRANSPOSE,
  //   //                                 lmode, impl->evecs[i + impl->numein], vec,
  //   //                                 request); CeedChk(ierr);
  // }

  // Zero lvecs
  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedOperatorFieldGetVector(opoutputfields[i], &vec); CeedChk(ierr);
    if (vec == CEED_VECTOR_ACTIVE)
      vec = outvec;
    ierr = CeedVectorSetValue(vec, 0.0); CeedChk(ierr);
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
      ierr = CeedVectorRestoreArrayRead(vec, (const CeedScalar **) &data->fields.in[i]);
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
      ierr = CeedVectorRestoreArrayRead(vec, (const CeedScalar **) &data->fields.out[i]);
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
