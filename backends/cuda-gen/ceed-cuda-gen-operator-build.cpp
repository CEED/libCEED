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
#include <iostream>
#include <sstream>
#include "../cuda/ceed-cuda.h"

static const char *deviceFunctions = QUOTE(
// typedef struct { const CeedScalar* inputs[16]; CeedScalar* outputs[16]; } CudaFields;
typedef struct { CeedScalar* fields[16]; } CudaFields;

//Read non interleaved dofs
template <int NCOMP, int P1d>
inline __device__ void readDofs1d(const CeedInt elem, const CeedScalar* d_u, CeedScalar* r_u) {
}

template <int NCOMP, int P1d>
inline __device__ void readDofs2d(const CeedInt elem, const CeedScalar* d_u, CeedScalar* r_u) {
}

template <int NCOMP, int P1d>
inline __device__ void readDofs3d(const CeedInt elem, const CeedScalar* d_u, CeedScalar* r_u) {
}

//read interleaved quads
template <int NCOMP, int Q1d>
inline __device__ void readQuads1d(const CeedInt elem, const CeedScalar* d_u, CeedScalar* r_u) {
}

template <int NCOMP, int Q1d>
inline __device__ void readQuads2d(const CeedInt elem, const CeedScalar* d_u, CeedScalar* r_u) {
}

template <int NCOMP, int Q1d>
inline __device__ void readQuads3d(const CeedInt elem, const CeedScalar* d_u, CeedScalar* r_u) {
}

//Write non interleaved dofs
template <int NCOMP, int P1d>
inline __device__ void writeDofs1d(const CeedInt elem, CeedScalar* r_v, const CeedScalar* d_v) {
}

template <int NCOMP, int P1d>
inline __device__ void writeDofs2d(const CeedInt elem, CeedScalar* r_v, const CeedScalar* d_v) {
}

template <int NCOMP, int P1d>
inline __device__ void writeDofs3d(const CeedInt elem, CeedScalar* r_v, const CeedScalar* d_v) {
}

//Write interleaved quads
template <int NCOMP, int Q1d>
inline __device__ void writeQuads1d(const CeedInt elem, CeedScalar* r_v, const CeedScalar* d_v) {
}

template <int NCOMP, int Q1d>
inline __device__ void writeQuads2d(const CeedInt elem, CeedScalar* r_v, const CeedScalar* d_v) {
}

template <int NCOMP, int Q1d>
inline __device__ void writeQuads3d(const CeedInt elem, CeedScalar* r_v, const CeedScalar* d_v) {
}

//****
// 1D
template <int NCOMP, int P1d, int Q1d>
inline __device__ void interp1d(const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                CeedScalar *__restrict__ r_V) {
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void interpTranspose1d(const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                CeedScalar *__restrict__ r_V) {
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void grad1d(const CeedScalar *__restrict__ r_U,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              CeedScalar *__restrict__ r_V) {
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void gradTranspose1d(const CeedScalar *__restrict__ r_U,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              CeedScalar *__restrict__ r_V) {
}

//****
// 2D
template <int NCOMP, int P1d, int Q1d>
inline __device__ void interp2d(const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                CeedScalar *__restrict__ r_V) {
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void interpTranspose2d(const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                CeedScalar *__restrict__ r_V) {
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void grad2d(const CeedScalar *__restrict__ r_U,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              CeedScalar *__restrict__ r_V) {
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void gradTranspose2d(const CeedScalar *__restrict__ r_U,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              CeedScalar *__restrict__ r_V) {
}

//****
// 3D
template <int NCOMP, int P1d, int Q1d>
inline __device__ void interp3d(const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                CeedScalar *__restrict__ r_V) {
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void interpTranspose3d(const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                CeedScalar *__restrict__ r_V) {
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void grad3d(const CeedScalar *__restrict__ r_U,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              CeedScalar *__restrict__ r_V) {
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void gradTranspose3d(const CeedScalar *__restrict__ r_U,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              CeedScalar *__restrict__ r_V) {
}

template <int Q1d>
inline __device__ void weight1d(const CeedScalar *qweight1d, CeedScalar *w) {
}

template <int Q1d>
inline __device__ void weight2d(const CeedScalar *qweight1d, CeedScalar *w) {
}

template <int Q1d>
inline __device__ void weight3d(const CeedScalar *qweight1d, CeedScalar *w) {
}

);

extern "C" int CeedCudaGenOperatorBuild(CeedOperator op, CeedVector invec,
                                  CeedVector outvec, CeedRequest *request) {

	using std::ostringstream;
  using std::string;
  using std::cout;
  int ierr;
  // CeedOperator_Cuda *impl;
  // ierr = CeedOperatorGetData(op, (void *)&impl); CeedChk(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
  CeedInt Q, P1d, Q1d = -1, numelements, elemsize, numinputfields, numoutputfields, ncomp, dim;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChk(ierr);
  ierr = CeedOperatorGetNumElements(op, &numelements); CeedChk(ierr);
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
  CeedBasis basis;
  CeedElemRestriction Erestrict;

  ostringstream code;
  string devFunctions(deviceFunctions);

  code << devFunctions;

  //TODO concatenate device function code readDofs, readQuads, writeDofs, writeQuads, interp, grad

  // Setup
  // ierr = CeedOperatorSetup_Cuda_gen(op); CeedChk(ierr);
  code << "\nextern \"C\" __global__ void oper(CeedInt nelem, CudaFields in, CudaFields B, CudaFields G, CeedScalar* W, CudaFields out) {\n";
  // Input Evecs and Restriction
  for (CeedInt i = 0; i < numinputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChk(ierr);
    if (emode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      // Get input vector
      // ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChk(ierr);
      // if (vec == CEED_VECTOR_ACTIVE)
      //   printf("%s", "invec,");
        // vec = invec;
    //   // Restrict
    //   ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict);
      code << "CeedScalar* d_u" <<i<<" = in.fields["<<i<<"];\n";
      if (emode != CEED_EVAL_NONE)
      {
        ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChk(ierr);
        bool isTensor;
        ierr = CeedBasisGetTensorStatus(basis, &isTensor); CeedChk(ierr);
        //TODO check that all are the same
        ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
        if (isTensor)
        {
          //TODO check that all are the same
          ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChk(ierr);
        }
      }
    //   CeedChk(ierr);
    //   ierr = CeedOperatorFieldGetLMode(opinputfields[i], &lmode); CeedChk(ierr);
    //   ierr = CeedElemRestrictionApply(Erestrict, CEED_NOTRANSPOSE,
    //                                   lmode, vec, impl->evecs[i],
    //                                   request); CeedChk(ierr);
    //   // Get evec
    //   ierr = CeedVectorGetArrayRead(impl->evecs[i], CEED_MEM_DEVICE,
    //                                 (const CeedScalar **) &impl->edata[i]);
    //   CeedChk(ierr);
    }
  }

  for (CeedInt i = 0; i < numoutputfields; i++) {
    // ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChk(ierr);
    // if (vec == CEED_VECTOR_ACTIVE)
    //   printf("%s", "outvec");
    code << "CeedScalar* d_v"<<i<<" = out.fields["<<i<<"];\n";
    // if (i<numoutputfields-1)
    // {
    //   printf(",");
    // }
  }

  code << "const CeedInt Dim = "<<dim<<";\n";
  code << "const CeedInt Q1d = "<<Q1d<<";\n";
  code << "const CeedInt Q   = "<<Q<<";\n";
  code << "for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < nelem; elem += gridDim.x*blockDim.z) {\n";
  // Input basis apply if needed
  for (CeedInt i = 0; i < numinputfields; i++) {
    code << "// Input field "<<i<<"\n";
    // Get elemsize, emode, ncomp
    ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict);
    CeedChk(ierr);
    ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
    CeedChk(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChk(ierr);
    ierr = CeedQFunctionFieldGetNumComponents(qfinputfields[i], &ncomp);
    CeedChk(ierr);
    // Basis action
    switch (emode) {
    case CEED_EVAL_NONE:
      code << "  const CeedInt ncomp_in_"<<i<<" = "<<ncomp<<";\n";
      code << "  CeedScalar r_t"<<i<<"[ncomp_in_"<<i<<"*Q1d];\n";
      code << "  readQuads"<<dim<<"d<ncomp_in_"<<i<<",Q1d>(elem, d_u"<<i<<", r_t"<<i<<");\n";
      // ierr = CeedVectorSetArray(impl->qvecsin[i], CEED_MEM_DEVICE,
      //                           CEED_USE_POINTER,
      //                           impl->edata[i]); CeedChk(ierr);
      break;
    case CEED_EVAL_INTERP:
      ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChk(ierr);
      ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
      code << "  const CeedInt P_in_"<<i<<" = "<<P1d<<";\n";
      code << "  const CeedInt ncomp_in_"<<i<<" = "<<ncomp<<";\n";
      code << "  CeedScalar r_u"<<i<<"[ncomp_in_"<<i<<"*P_in_"<<i<<"];\n";
      code << "  readDofs"<<dim<<"d<ncomp_in_"<<i<<",P_in_"<<i<<">(elem, d_u"<<i<<", r_u"<<i<<");\n";
      code << "  CeedScalar r_t"<<i<<"[ncomp_in_"<<i<<"*Q1d];\n";
      code << "  CeedScalar* B_in_"<<i<<" = B.fields["<<i<<"];";
      code << "  interp"<<dim<<"d<ncomp_in_"<<i<<",P_in_"<<i<<",Q1d>(r_u"<<i<<", B_in_"<<i<<", r_t"<<i<<");\n";
      // ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChk(ierr);
      // ierr = CeedBasisApply(basis, numelements, CEED_NOTRANSPOSE,
      //                       CEED_EVAL_INTERP, impl->evecs[i],
      //                       impl->qvecsin[i]); CeedChk(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChk(ierr);
      code << "  const CeedInt ncomp_in_"<<i<<" = "<<ncomp<<";\n";
      ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
      code << "  const CeedInt P_in_"<<i<<" = "<<P1d<<";\n";
      code << "  CeedScalar r_u"<<i<<"[ncomp_in_"<<i<<"*P_in_"<<i<<"];\n";
      code << "  readDofs"<<dim<<"d<ncomp_in_"<<i<<",P_in_"<<i<<">(elem, d_u"<<i<<", r_u"<<i<<");\n";
      code << "  CeedScalar r_t"<<i<<"[ncomp_in_"<<i<<"*Dim*Q1d];\n";
      code << "  CeedScalar* B_in_"<<i<<" = B.fields["<<i<<"];";
      code << "  CeedScalar* G_in_"<<i<<" = G.fields["<<i<<"];";
      code << "  grad"<<dim<<"d<ncomp_in_"<<i<<",P_in_"<<i<<",Q1d>(r_u"<<i<<", B_in_"<<i<<", G_in_"<<i<<", r_t"<<i<<");\n";
      // ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChk(ierr);
      // ierr = CeedBasisApply(basis, numelements, CEED_NOTRANSPOSE,
      //                       CEED_EVAL_GRAD, impl->evecs[i],
      //                       impl->qvecsin[i]); CeedChk(ierr);
      break;
    case CEED_EVAL_WEIGHT:
      code << "  CeedScalar r_t"<<i<<"[Q1d];\n";
      code << "  weight"<<dim<<"d<Q1d>(W, r_t"<<i<<");\n";
      break; // No action
    case CEED_EVAL_DIV:
      break; // TODO: Not implemented
    case CEED_EVAL_CURL:
      break; // TODO: Not implemented
    }
  }
  // Output pointers
  for (CeedInt i = 0; i < numoutputfields; i++) {
    // ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    // CeedChk(ierr);
    // if (emode == CEED_EVAL_NONE) {
      // ierr = CeedVectorGetArray(impl->evecs[i + impl->numein], CEED_MEM_DEVICE,
      //                           &impl->edata[i + numinputfields]); CeedChk(ierr);
      // ierr = CeedQFunctionFieldGetNumComponents(qfoutputfields[i], &ncomp);
      // CeedChk(ierr);
      // ierr = CeedVectorSetArray(impl->qvecsout[i], CEED_MEM_DEVICE,
      //                           CEED_USE_POINTER,
      //                           impl->edata[i + numinputfields]);
      // CeedChk(ierr);
    // }
  }
  // Q function
  code << "// QFunction\n";
  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChk(ierr);
    ierr = CeedQFunctionFieldGetNumComponents(qfoutputfields[i], &ncomp);
    CeedChk(ierr);
    if (emode==CEED_EVAL_GRAD)
    {
      code << "  const CeedInt ncomp_out_"<<i<<" = "<<ncomp<<";\n";
      ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis); CeedChk(ierr);
      code << "  CeedScalar r_tt"<<i<<"[ncomp_out_"<<i<<"*Dim*Q1d];\n";
    }
    if (emode==CEED_EVAL_NONE || emode==CEED_EVAL_INTERP)
    {
      code << "  const CeedInt ncomp_out_"<<i<<" = "<<ncomp<<";\n";
      code << "  CeedScalar r_tt"<<i<<"[ncomp_out_"<<i<<"*Q1d];\n";
    }
  }
  //TODO write qfunction load for this backend
  code << "  qfunction(";
  for (CeedInt i = 0; i < numinputfields; i++) {
    code << "r_t"<<i<<", ";
  }
  for (CeedInt i = 0; i < numoutputfields; i++) {
    code << "r_tt"<<i;
    if (i<numoutputfields-1)
    {
      code << ", ";
    }
  }
  code << ");\n";
  // ierr = CeedQFunctionApply(qf, numelements * Q, impl->qvecsin, impl->qvecsout);
  // CeedChk(ierr);

  // Output basis apply if needed
  for (CeedInt i = 0; i < numoutputfields; i++) {
    code << "// Output field %d\n";
    // Get elemsize, emode, ncomp
    ierr = CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict);
    CeedChk(ierr);
    ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
    CeedChk(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChk(ierr);
    ierr = CeedQFunctionFieldGetNumComponents(qfoutputfields[i], &ncomp);
    CeedChk(ierr);
    // Basis action
    switch (emode) {
    case CEED_EVAL_NONE:
      code << "  writeQuads"<<dim<<"d<ncomp_out_"<<i<<",Q1d>(elem, r_tt"<<i<<", d_v"<<i<<");\n";
      break; // No action
    case CEED_EVAL_INTERP:
      ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis); CeedChk(ierr);
      ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
      code << "  const CeedInt P_out_"<<i<<" = "<<P1d<<";\n";
      code << "  CeedScalar r_v"<<i<<"[ncomp_out_"<<i<<"*P_out_"<<i<<"];\n";
      code << "  CeedScalar* B_out_"<<i<<" = B.fields["<<i<<"];";
      code << "  interpTranspose"<<dim<<"d<ncomp_out_"<<i<<",P_out_"<<i<<",Q1d>(r_tt"<<i<<", B_out_"<<i<<", r_v"<<i<<");\n";
      code << "  writeDofs"<<dim<<"d<ncomp_out_"<<i<<",P_out_"<<i<<">(elem, r_v"<<i<<", d_v"<<i<<");\n";
      // ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis);
      // CeedChk(ierr);
      // ierr = CeedBasisApply(basis, numelements, CEED_TRANSPOSE,
      //                       CEED_EVAL_INTERP, impl->qvecsout[i],
      //                       impl->evecs[i + impl->numein]); CeedChk(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis); CeedChk(ierr);
      ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
      code << "  const CeedInt P_out_"<<i<<" = "<<P1d<<";\n";
      code << "  CeedScalar r_v"<<i<<"[ncomp_out_"<<i<<"*P_out_"<<i<<"];\n";
      code << "  CeedScalar* B_out_"<<i<<" = B.fields["<<i<<"];";
      code << "  CeedScalar* G_out_"<<i<<" = G.fields["<<i<<"];";
      code << "  gradTranspose"<<dim<<"d<ncomp_out_"<<i<<",P_out_"<<i<<",Q1d>(r_tt"<<i<<", B_out_"<<i<<", G_out_"<<i<<", r_v"<<i<<");\n";
      code << "  writeDofs"<<dim<<"d<ncomp_out_"<<i<<",P_out_"<<i<<">(elem, r_v"<<i<<", d_v"<<i<<");\n";
      // ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis);
      // CeedChk(ierr);
      // ierr = CeedBasisApply(basis, numelements, CEED_TRANSPOSE,
      //                       CEED_EVAL_GRAD, impl->qvecsout[i],
      //                       impl->evecs[i + impl->numein]); CeedChk(ierr);
      break;
    case CEED_EVAL_WEIGHT: {
      // Ceed ceed;
      // ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
      // return CeedError(ceed, 1,
      //                  "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
      break; // Should not occur
    }
    case CEED_EVAL_DIV:
      break; // TODO: Not implemented
    case CEED_EVAL_CURL:
      break; // TODO: Not implemented
    }
  }

  // Zero lvecs
  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedOperatorFieldGetVector(opoutputfields[i], &vec); CeedChk(ierr);
    if (vec == CEED_VECTOR_ACTIVE)
      vec = outvec;
    ierr = CeedVectorSetValue(vec, 0.0); CeedChk(ierr);
  }

  // Output restriction
  for (CeedInt i = 0; i < numoutputfields; i++) {
    // Restore evec
    // ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    // CeedChk(ierr);
    // if (emode == CEED_EVAL_NONE) {
    //   ierr = CeedVectorRestoreArray(impl->evecs[i+impl->numein],
    //                                 &impl->edata[i + numinputfields]);
    //   CeedChk(ierr);
    // }
    // // Get output vector
    // ierr = CeedOperatorFieldGetVector(opoutputfields[i], &vec); CeedChk(ierr);
    // // Active
    // if (vec == CEED_VECTOR_ACTIVE)
    //   vec = outvec;
    // // Restrict
    // ierr = CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict);
    // CeedChk(ierr);
    // ierr = CeedOperatorFieldGetLMode(opoutputfields[i], &lmode); CeedChk(ierr);
    // ierr = CeedElemRestrictionApply(Erestrict, CEED_TRANSPOSE,
    //                                 lmode, impl->evecs[i + impl->numein], vec,
    //                                 request); CeedChk(ierr);
  }

  // Restore input arrays
  for (CeedInt i = 0; i < numinputfields; i++) {
    // ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    // CeedChk(ierr);
    // if (emode == CEED_EVAL_WEIGHT) { // Skip
    // } else {
    //   ierr = CeedVectorRestoreArrayRead(impl->evecs[i],
    //                                     (const CeedScalar **) &impl->edata[i]);
    //   CeedChk(ierr);
    // }
  }
  code << "  }\n";
  code << "}\n\n";

  cout << code.str();

  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  CeedOperator_Cuda_gen *data;
  ierr = CeedOperatorGetData(op, (void **)&data); CeedChk(ierr);
  ierr = compile(ceed, code.str().c_str(), &data->module, 0); CeedChk(ierr);
  ierr = get_kernel(ceed, data->module, "oper", &data->op);
  CeedChk(ierr);

  return 0;
}