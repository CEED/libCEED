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
typedef struct { CeedScalar* in[16]; CeedScalar* out[16]; } CudaFields;
typedef struct { CeedInt* in[16]; CeedInt* out[16]; } CudaFieldsInt;

typedef struct {
  CeedInt tidx;
  CeedInt tidy;
  CeedInt tidz;
  CeedScalar* slice;
} BackendData;

#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old =
      atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val +
                                     __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN
    // (since NaN != NaN)
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif // __CUDA_ARCH__ < 600

//****
// 1D
template <int NCOMP, int P1d>
inline __device__ void readDofs1d(BackendData& data, const CeedInt ndofs, const CeedInt elem, const CeedInt* indices, const CeedScalar* d_u, CeedScalar* r_u) {
  const CeedInt dof = data.tidx;
  const CeedInt ind = indices[dof + elem * P1d];
  for(CeedInt comp = 0; comp < NCOMP; ++comp) {
    r_u[comp] = d_u[ind + ndofs * comp];
  }
}

template <int NCOMP, int P1d>
inline __device__ void readDofsTranspose1d(BackendData& data, const CeedInt ndofs, const CeedInt elem, const CeedInt* indices, const CeedScalar* d_u, CeedScalar* r_u) {
  const CeedInt dof = data.tidx;
  const CeedInt ind = indices[dof + elem * P1d];
  for(CeedInt comp = 0; comp < NCOMP; ++comp) {
    r_u[comp] = d_u[ind * NCOMP + comp];
  }
}

template <int NCOMP, int Q1d>
inline __device__ void readQuads1d(BackendData& data, const CeedInt nquads, const CeedInt elem, const CeedScalar* d_u, CeedScalar* r_u) {
  const CeedInt dof = data.tidx;
  const CeedInt ind = dof + elem * Q1d;
  for(CeedInt comp = 0; comp < NCOMP; ++comp) {
    r_u[comp] = d_u[ind + nquads * comp];
  }
}

template <int NCOMP, int Q1d>
inline __device__ void readQuadsTranspose1d(BackendData& data, const CeedInt nquads, const CeedInt elem, const CeedScalar* d_u, CeedScalar* r_u) {
  const CeedInt dof = data.tidx;
  const CeedInt ind = dof + elem * Q1d;
  for(CeedInt comp = 0; comp < NCOMP; ++comp) {
    r_u[comp] = d_u[ind * NCOMP + comp];
  }
}

template <int NCOMP, int P1d>
inline __device__ void writeDofs1d(BackendData& data, const CeedInt ndofs, const CeedInt elem, const CeedInt* indices, const CeedScalar* r_v, CeedScalar* d_v) {
  const CeedInt dof = data.tidx;
  const CeedInt ind = indices[dof + elem * P1d];
  for(CeedInt comp = 0; comp < NCOMP; ++comp) {
    atomicAdd(&d_v[ind + ndofs * comp], r_v[comp]);
  }
}

template <int NCOMP, int P1d>
inline __device__ void writeDofsTranspose1d(BackendData& data, const CeedInt ndofs, const CeedInt elem, const CeedInt* indices, const CeedScalar* r_v, CeedScalar* d_v) {
  const CeedInt dof = data.tidx;
  const CeedInt ind = indices[dof + elem * P1d];
  for(CeedInt comp = 0; comp < NCOMP; ++comp) {
    atomicAdd(&d_v[ind * NCOMP + comp], r_v[comp]);
  }
}

template <int NCOMP, int Q1d>
inline __device__ void writeQuads1d(BackendData& data, const CeedInt nquads, const CeedInt elem, const CeedScalar* r_v, CeedScalar* d_v) {
  const CeedInt dof = data.tidx;
  const CeedInt ind = dof + elem * Q1d;
  for(CeedInt comp = 0; comp < NCOMP; ++comp) {
    d_v[ind + nquads * comp] = r_v[comp];
  }
}

template <int NCOMP, int Q1d>
inline __device__ void writeQuadsTranspose1d(BackendData& data, const CeedInt nquads, const CeedInt elem, const CeedScalar* r_v, CeedScalar* d_v) {
  const CeedInt dof = data.tidx;
  const CeedInt ind = dof + elem * Q1d;
  for(CeedInt comp = 0; comp < NCOMP; ++comp) {
    d_v[ind * NCOMP + comp] = r_v[comp];
  }
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractX1d(BackendData& data,
                                   const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.tidx] = *U;
  __syncthreads();
  *V = 0.0;
  for (int i = 0; i < P1d; ++i) {
    *V += B[i + data.tidx*P1d] * data.slice[i];//contract x direction
  }
  __syncthreads();
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractTransposeX1d(BackendData& data,
                                            const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.tidx] = *U;
  __syncthreads();
  *V = 0.0;
  for (int i = 0; i < Q1d; ++i) {
    *V += B[data.tidx + i*P1d] * data.slice[i];//contract x direction
  }
  __syncthreads();
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void interp1d(BackendData& data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                CeedScalar *__restrict__ r_V) {
  for(int comp=0; comp<NCOMP; comp++) {
    ContractX1d<NCOMP,P1d,Q1d>(data, r_U, c_B, r_V);
  }
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void interpTranspose1d(BackendData& data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                CeedScalar *__restrict__ r_V) {
  for(int comp=0; comp<NCOMP; comp++) {
    ContractTransposeX1d<NCOMP,P1d,Q1d>(data, r_U, c_B, r_V);
  }
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void grad1d(BackendData& data, const CeedScalar *__restrict__ r_U,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              CeedScalar *__restrict__ r_V) {
  for(int comp=0; comp<NCOMP; comp++) {
    ContractX1d<NCOMP,P1d,Q1d>(data, r_U, c_G, r_V);
  }
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void gradTranspose1d(BackendData& data, const CeedScalar *__restrict__ r_U,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              CeedScalar *__restrict__ r_V) {
  for(int comp=0; comp<NCOMP; comp++) {
    ContractTransposeX1d<NCOMP,P1d,Q1d>(data, r_U, c_G, r_V);
  }
}

//****
// 2D
template <int NCOMP, int P1d>
inline __device__ void readDofs2d(BackendData& data, const CeedInt elem, const CeedScalar* d_u, CeedScalar* r_u) {
}

template <int NCOMP, int Q1d>
inline __device__ void readQuads2d(BackendData& data, const CeedInt elem, const CeedScalar* d_u, CeedScalar* r_u) {
}

template <int NCOMP, int P1d>
inline __device__ void writeDofs2d(BackendData& data, const CeedInt elem, CeedScalar* r_v, const CeedScalar* d_v) {
}

template <int NCOMP, int Q1d>
inline __device__ void writeQuads2d(BackendData& data, const CeedInt elem, CeedScalar* r_v, const CeedScalar* d_v) {
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void interp2d(BackendData& data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                CeedScalar *__restrict__ r_V) {
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void interpTranspose2d(BackendData& data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                CeedScalar *__restrict__ r_V) {
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void grad2d(BackendData& data, const CeedScalar *__restrict__ r_U,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              CeedScalar *__restrict__ r_V) {
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void gradTranspose2d(BackendData& data, const CeedScalar *__restrict__ r_U,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              CeedScalar *__restrict__ r_V) {
}

//****
// 3D
template <int NCOMP, int P1d>
inline __device__ void readDofs3d(BackendData& data, const CeedInt elem, const CeedScalar* d_u, CeedScalar* r_u) {
}

template <int NCOMP, int Q1d>
inline __device__ void readQuads3d(BackendData& data, const CeedInt elem, const CeedScalar* d_u, CeedScalar* r_u) {
}

template <int NCOMP, int P1d>
inline __device__ void writeDofs3d(BackendData& data, const CeedInt elem, CeedScalar* r_v, const CeedScalar* d_v) {
}

template <int NCOMP, int Q1d>
inline __device__ void writeQuads3d(BackendData& data, const CeedInt elem, CeedScalar* r_v, const CeedScalar* d_v) {
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void interp3d(BackendData& data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                CeedScalar *__restrict__ r_V) {
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void interpTranspose3d(BackendData& data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                CeedScalar *__restrict__ r_V) {
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void grad3d(BackendData& data, const CeedScalar *__restrict__ r_U,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              CeedScalar *__restrict__ r_V) {
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void gradTranspose3d(BackendData& data, const CeedScalar *__restrict__ r_U,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              CeedScalar *__restrict__ r_V) {
}

template <int Q1d>
inline __device__ void weight1d(BackendData& data, const CeedScalar *qweight1d, CeedScalar *w) {
}

template <int Q1d>
inline __device__ void weight2d(BackendData& data, const CeedScalar *qweight1d, CeedScalar *w) {
}

template <int Q1d>
inline __device__ void weight3d(BackendData& data, const CeedScalar *qweight1d, CeedScalar *w) {
}

inline __device__ void qfunction(...) {
}

);

extern "C" int CeedCudaGenOperatorBuild(CeedOperator op, CeedVector invec,
                                  CeedVector outvec, CeedRequest *request) {

	using std::ostringstream;
  using std::string;
  int ierr;
  // CeedOperator_Cuda *impl;
  // ierr = CeedOperatorGetData(op, (void *)&impl); CeedChk(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
  CeedInt Q, P1d, Q1d = -1, numelements, elemsize, numinputfields, numoutputfields, ncomp, dim, ndof;
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
  CeedTransposeMode lmode;
  CeedVector vec;
  CeedBasis basis;
  CeedElemRestriction Erestrict;

  ostringstream code;
  string devFunctions(deviceFunctions);

  code << devFunctions;

  //TODO concatenate device function code readDofs, readQuads, writeDofs, writeQuads, interp, grad

  // Setup
  // ierr = CeedOperatorSetup_Cuda_gen(op); CeedChk(ierr);
  code << "\nextern \"C\" __global__ void oper(CeedInt nelem, CudaFieldsInt indices, CudaFields fields, CudaFields B, CudaFields G, CeedScalar* W) {\n";
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
      code << "CeedScalar* d_u" <<i<<" = fields.in["<<i<<"];\n";
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
    code << "CeedScalar* d_v"<<i<<" = fields.out["<<i<<"];\n";
    // if (i<numoutputfields-1)
    // {
    //   printf(",");
    // }
  }

  code << "BackendData data;\n";
  code << "const CeedInt Dim = "<<dim<<";\n";
  code << "const CeedInt Q1d = "<<Q1d<<";\n";
  // code << "const CeedInt Q   = "<<Q<<";\n";
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
      ierr = CeedElemRestrictionGetNumDoF(Erestrict, &ndof); CeedChk(ierr);
      code << "  const CeedInt ncomp_in_"<<i<<" = "<<ncomp<<";\n";
      code << "  const CeedInt nquads_in_"<<i<<" = "<<ndof<<";\n";
      code << "  CeedScalar r_t"<<i<<"[ncomp_in_"<<i<<"*Q1d];\n";
      ierr = CeedOperatorFieldGetLMode(opinputfields[i], &lmode); CeedChk(ierr);
      code << "  readQuads"<<(lmode==CEED_NOTRANSPOSE?"":"Transpose")<<dim<<"d<ncomp_in_"<<i<<",Q1d>(data, nquads_in_"<<i<<", elem, d_u"<<i<<", r_t"<<i<<");\n";
      // ierr = CeedVectorSetArray(impl->qvecsin[i], CEED_MEM_DEVICE,
      //                           CEED_USE_POINTER,
      //                           impl->edata[i]); CeedChk(ierr);
      break;
    case CEED_EVAL_INTERP:
      ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChk(ierr);
      ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
      CeedChk(ierr);
      ierr = CeedElemRestrictionGetNumDoF(Erestrict, &ndof); CeedChk(ierr);
      code << "  const CeedInt P_in_"<<i<<" = "<<P1d<<";\n";
      code << "  const CeedInt ncomp_in_"<<i<<" = "<<ncomp<<";\n";
      code << "  const CeedInt ndofs_in_"<<i<<" = "<<ndof<<";\n";
      code << "  CeedScalar r_u"<<i<<"[ncomp_in_"<<i<<"*P_in_"<<i<<"];\n";
      ierr = CeedOperatorFieldGetLMode(opinputfields[i], &lmode); CeedChk(ierr);
      code << "  readDofs"<<(lmode==CEED_NOTRANSPOSE?"":"Transpose")<<dim<<"d<ncomp_in_"<<i<<",P_in_"<<i<<">(data, ndofs_in_"<<i<<", elem, indices.in["<<i<<"], d_u"<<i<<", r_u"<<i<<");\n";
      code << "  CeedScalar r_t"<<i<<"[ncomp_in_"<<i<<"*Q1d];\n";
      code << "  interp"<<dim<<"d<ncomp_in_"<<i<<",P_in_"<<i<<",Q1d>(data, r_u"<<i<<", B.in["<<i<<"], r_t"<<i<<");\n";
      // ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChk(ierr);
      // ierr = CeedBasisApply(basis, numelements, CEED_NOTRANSPOSE,
      //                       CEED_EVAL_INTERP, impl->evecs[i],
      //                       impl->qvecsin[i]); CeedChk(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChk(ierr);
      CeedChk(ierr);
      ierr = CeedElemRestrictionGetNumDoF(Erestrict, &ndof); CeedChk(ierr);
      code << "  const CeedInt ncomp_in_"<<i<<" = "<<ncomp<<";\n";
      code << "  const CeedInt ndofs_in_"<<i<<" = "<<ndof<<";\n";
      ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
      code << "  const CeedInt P_in_"<<i<<" = "<<P1d<<";\n";
      code << "  CeedScalar r_u"<<i<<"[ncomp_in_"<<i<<"*P_in_"<<i<<"];\n";
      ierr = CeedOperatorFieldGetLMode(opinputfields[i], &lmode); CeedChk(ierr);
      code << "  readDofs"<<(lmode==CEED_NOTRANSPOSE?"":"Transpose")<<dim<<"d<ncomp_in_"<<i<<",P_in_"<<i<<">(data, ndofs_in_"<<i<<", elem, indices.in["<<i<<"], d_u"<<i<<", r_u"<<i<<");\n";
      code << "  CeedScalar r_t"<<i<<"[ncomp_in_"<<i<<"*Dim*Q1d];\n";
      code << "  grad"<<dim<<"d<ncomp_in_"<<i<<",P_in_"<<i<<",Q1d>(data, r_u"<<i<<", B.in["<<i<<"], G.in["<<i<<"], r_t"<<i<<");\n";
      // ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChk(ierr);
      // ierr = CeedBasisApply(basis, numelements, CEED_NOTRANSPOSE,
      //                       CEED_EVAL_GRAD, impl->evecs[i],
      //                       impl->qvecsin[i]); CeedChk(ierr);
      break;
    case CEED_EVAL_WEIGHT:
      code << "  CeedScalar r_t"<<i<<"[Q1d];\n";
      code << "  weight"<<dim<<"d<Q1d>(data, W, r_t"<<i<<");\n";
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
      ierr = CeedElemRestrictionGetNumDoF(Erestrict, &ndof); CeedChk(ierr);
      ierr = CeedOperatorFieldGetLMode(opoutputfields[i], &lmode); CeedChk(ierr);
      code << "  const CeedInt nquads_out_"<<i<<" = "<<ndof<<";\n";
      code << "  writeQuads"<<(lmode==CEED_NOTRANSPOSE?"":"Transpose")<<dim<<"d<ncomp_out_"<<i<<",Q1d>(data, nquads_out_"<<i<<", elem, r_tt"<<i<<", d_v"<<i<<");\n";
      break; // No action
    case CEED_EVAL_INTERP:
      ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis); CeedChk(ierr);
      ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
      code << "  const CeedInt P_out_"<<i<<" = "<<P1d<<";\n";
      code << "  CeedScalar r_v"<<i<<"[ncomp_out_"<<i<<"*P_out_"<<i<<"];\n";
      code << "  interpTranspose"<<dim<<"d<ncomp_out_"<<i<<",P_out_"<<i<<",Q1d>(data, r_tt"<<i<<", B.out["<<i<<"], r_v"<<i<<");\n";
      ierr = CeedElemRestrictionGetNumDoF(Erestrict, &ndof); CeedChk(ierr);
      code << "  const CeedInt ndofs_out_"<<i<<" = "<<ndof<<";\n";
      ierr = CeedOperatorFieldGetLMode(opoutputfields[i], &lmode); CeedChk(ierr);
      code << "  writeDofs"<<(lmode==CEED_NOTRANSPOSE?"":"Transpose")<<dim<<"d<ncomp_out_"<<i<<",P_out_"<<i<<">(data, ndofs_out_"<<i<<", elem, indices.out["<<i<<"], r_v"<<i<<", d_v"<<i<<");\n";
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
      code << "  gradTranspose"<<dim<<"d<ncomp_out_"<<i<<",P_out_"<<i<<",Q1d>(data, r_tt"<<i<<", B.out["<<i<<"], G.out["<<i<<"], r_v"<<i<<");\n";
      ierr = CeedElemRestrictionGetNumDoF(Erestrict, &ndof); CeedChk(ierr);
      code << "  const CeedInt ndofs_out_"<<i<<" = "<<ndof<<";\n";
      ierr = CeedOperatorFieldGetLMode(opoutputfields[i], &lmode); CeedChk(ierr);
      code << "  writeDofs"<<(lmode==CEED_NOTRANSPOSE?"":"Transpose")<<dim<<"d<ncomp_out_"<<i<<",P_out_"<<i<<">(data, ndofs_out_"<<i<<", elem, indices.out["<<i<<"], r_v"<<i<<", d_v"<<i<<");\n";
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

  std::cout << code.str();

  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  CeedOperator_Cuda_gen *data;
  ierr = CeedOperatorGetData(op, (void **)&data); CeedChk(ierr);
  ierr = compile(ceed, code.str().c_str(), &data->module, 0); CeedChk(ierr);
  ierr = get_kernel(ceed, data->module, "oper", &data->op);
  CeedChk(ierr);

  return 0;
}