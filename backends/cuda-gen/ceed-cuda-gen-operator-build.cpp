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
  if (data.tidx<P1d)
  {
    const CeedInt dof = data.tidx;
    const CeedInt ind = indices[dof + elem * P1d];//TODO handle indices==NULL
    for(CeedInt comp = 0; comp < NCOMP; ++comp) {
      r_u[comp] = d_u[ind + ndofs * comp];
    }
  }
}

template <int NCOMP, int P1d>
inline __device__ void readDofsTranspose1d(BackendData& data, const CeedInt ndofs, const CeedInt elem, const CeedInt* indices, const CeedScalar* d_u, CeedScalar* r_u) {
  if (data.tidx<P1d)
  {
    const CeedInt dof = data.tidx;
    const CeedInt ind = indices[dof + elem * P1d];
    for(CeedInt comp = 0; comp < NCOMP; ++comp) {
      r_u[comp] = d_u[ind * NCOMP + comp];
    }
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
  if (data.tidx<P1d)
  {
    const CeedInt dof = data.tidx;
    const CeedInt ind = indices[dof + elem * P1d];
    for(CeedInt comp = 0; comp < NCOMP; ++comp) {
      atomicAdd(&d_v[ind + ndofs * comp], r_v[comp]);
    }
  }
}

template <int NCOMP, int P1d>
inline __device__ void writeDofsTranspose1d(BackendData& data, const CeedInt ndofs, const CeedInt elem, const CeedInt* indices, const CeedScalar* r_v, CeedScalar* d_v) {
  if (data.tidx<P1d)
  {
    const CeedInt dof = data.tidx;
    const CeedInt ind = indices[dof + elem * P1d];
    for(CeedInt comp = 0; comp < NCOMP; ++comp) {
      atomicAdd(&d_v[ind * NCOMP + comp], r_v[comp]);
    }
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
    ContractX1d<NCOMP,P1d,Q1d>(data, r_U+comp, c_B, r_V+comp);
  }
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void interpTranspose1d(BackendData& data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                CeedScalar *__restrict__ r_V) {
  for(int comp=0; comp<NCOMP; comp++) {
    ContractTransposeX1d<NCOMP,P1d,Q1d>(data, r_U+comp, c_B, r_V+comp);
  }
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void grad1d(BackendData& data, const CeedScalar *__restrict__ r_U,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              CeedScalar *__restrict__ r_V) {
  for(int comp=0; comp<NCOMP; comp++) {
    ContractX1d<NCOMP,P1d,Q1d>(data, r_U+comp, c_G, r_V+comp);
  }
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void gradTranspose1d(BackendData& data, const CeedScalar *__restrict__ r_U,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              CeedScalar *__restrict__ r_V) {
  for(int comp=0; comp<NCOMP; comp++) {
    ContractTransposeX1d<NCOMP,P1d,Q1d>(data, r_U+comp, c_G, r_V+comp);
  }
}

//****
// 2D
template <int NCOMP, int P1d>
inline __device__ void readDofs2d(BackendData& data, const CeedInt ndofs, const CeedInt elem, const CeedInt* indices, const CeedScalar* d_u, CeedScalar* r_u) {
  if (data.tidx<P1d && data.tidy<P1d)
  {
    const CeedInt dof = data.tidx + data.tidy*P1d;
    const CeedInt ind = indices[dof + elem * P1d*P1d];//TODO handle indices==NULL
    for(CeedInt comp = 0; comp < NCOMP; ++comp) {
      r_u[comp] = d_u[ind + ndofs * comp];
    }
  }
}

template <int NCOMP, int P1d>
inline __device__ void readDofsTranspose2d(BackendData& data, const CeedInt ndofs, const CeedInt elem, const CeedInt* indices, const CeedScalar* d_u, CeedScalar* r_u) {
  if (data.tidx<P1d && data.tidy<P1d)
  {
    const CeedInt dof = data.tidx + data.tidy*P1d;
    const CeedInt ind = indices[dof + elem * P1d*P1d];//TODO handle indices==NULL
    for(CeedInt comp = 0; comp < NCOMP; ++comp) {
      r_u[comp] = d_u[ind * NCOMP + comp];
    }
  }
}

template <int NCOMP, int Q1d>
inline __device__ void readQuads2d(BackendData& data, const CeedInt nquads, const CeedInt elem, const CeedScalar* d_u, CeedScalar* r_u) {
  const CeedInt dof = data.tidx + data.tidy*Q1d;
  const CeedInt ind = dof + elem * Q1d*Q1d;
  for(CeedInt comp = 0; comp < NCOMP; ++comp) {
    r_u[comp] = d_u[ind + nquads * comp];
  }
}

template <int NCOMP, int Q1d>
inline __device__ void readQuadsTranspose2d(BackendData& data, const CeedInt nquads, const CeedInt elem, const CeedScalar* d_u, CeedScalar* r_u) {
  const CeedInt dof = data.tidx + data.tidy*Q1d;
  const CeedInt ind = dof + elem * Q1d*Q1d;
  for(CeedInt comp = 0; comp < NCOMP; ++comp) {
    r_u[comp] = d_u[ind * NCOMP + comp];
  }
}

template <int NCOMP, int P1d>
inline __device__ void writeDofs2d(BackendData& data, const CeedInt ndofs, const CeedInt elem, const CeedInt* indices, const CeedScalar* r_v, CeedScalar* d_v) {
  if (data.tidx<P1d && data.tidy<P1d)
  {
    const CeedInt dof = data.tidx + data.tidy*P1d;
    const CeedInt ind = indices[dof + elem * P1d*P1d];
    for(CeedInt comp = 0; comp < NCOMP; ++comp) {
      atomicAdd(&d_v[ind + ndofs * comp], r_v[comp]);
    }
  }
}

template <int NCOMP, int P1d>
inline __device__ void writeDofsTranspose2d(BackendData& data, const CeedInt ndofs, const CeedInt elem, const CeedInt* indices, const CeedScalar* r_v, CeedScalar* d_v) {
  if (data.tidx<P1d && data.tidy<P1d)
  {
    const CeedInt dof = data.tidx + data.tidy*P1d;
    const CeedInt ind = indices[dof + elem * P1d*P1d];
    for(CeedInt comp = 0; comp < NCOMP; ++comp) {
      atomicAdd(&d_v[ind * NCOMP + comp], r_v[comp]);
    }
  }
}

template <int NCOMP, int Q1d>
inline __device__ void writeQuads2d(BackendData& data, const CeedInt nquads, const CeedInt elem, const CeedScalar* r_v, CeedScalar* d_v) {
  const CeedInt dof = data.tidx + data.tidy*Q1d;
  const CeedInt ind = dof + elem * Q1d*Q1d;
  for(CeedInt comp = 0; comp < NCOMP; ++comp) {
    d_v[ind + nquads * comp] = r_v[comp];
  }
}

template <int NCOMP, int Q1d>
inline __device__ void writeQuadsTranspose2d(BackendData& data, const CeedInt nquads, const CeedInt elem, const CeedScalar* r_v, CeedScalar* d_v) {
  const CeedInt dof = data.tidx + data.tidy*Q1d;
  const CeedInt ind = dof + elem * Q1d*Q1d;
  for(CeedInt comp = 0; comp < NCOMP; ++comp) {
    d_v[ind * NCOMP + comp] = r_v[comp];
  }
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
inline __device__ void readDofs3d(BackendData& data, const CeedInt ndofs, const CeedInt elem, const CeedInt* indices, const CeedScalar* d_u, CeedScalar* r_u) {
  if (data.tidx<P1d && data.tidy<P1d)
  {
    for(CeedInt dof = data.tidx + data.tidy*P1d; dof < P1d*P1d*P1d; dof += P1d*P1d){
      const CeedInt ind = indices[dof + elem * P1d*P1d*P1d];//TODO handle indices==NULL
      for(CeedInt comp = 0; comp < NCOMP; ++comp) {
        r_u[comp] = d_u[ind + ndofs * comp];
      }
    }
  }
}

template <int NCOMP, int P1d>
inline __device__ void readDofsTranspose3d(BackendData& data, const CeedInt ndofs, const CeedInt elem, const CeedInt* indices, const CeedScalar* d_u, CeedScalar* r_u) {
  if (data.tidx<P1d && data.tidy<P1d)
  {
    for(CeedInt dof = data.tidx + data.tidy*P1d; dof < P1d*P1d*P1d; dof += P1d*P1d){
      const CeedInt ind = indices[dof + elem * P1d*P1d*P1d];//TODO handle indices==NULL
      for(CeedInt comp = 0; comp < NCOMP; ++comp) {
        r_u[comp] = d_u[ind * NCOMP + comp];
      }
    }
  }
}

template <int NCOMP, int Q1d>
inline __device__ void readQuads3d(BackendData& data, const CeedInt nquads, const CeedInt elem, const CeedScalar* d_u, CeedScalar* r_u) {
  for(CeedInt dof = data.tidx + data.tidy*Q1d; dof < Q1d*Q1d*Q1d; dof+= Q1d*Q1d){
    const CeedInt ind = dof + elem * Q1d*Q1d*Q1d;
    for(CeedInt comp = 0; comp < NCOMP; ++comp) {
      r_u[comp] = d_u[ind + nquads * comp];
    }
  }
}

template <int NCOMP, int Q1d>
inline __device__ void readQuadsTranspose3d(BackendData& data, const CeedInt nquads, const CeedInt elem, const CeedScalar* d_u, CeedScalar* r_u) {
  for(CeedInt dof = data.tidx + data.tidy*Q1d; dof < Q1d*Q1d*Q1d; dof+= Q1d*Q1d){
    const CeedInt ind = dof + elem * Q1d*Q1d*Q1d;
    for(CeedInt comp = 0; comp < NCOMP; ++comp) {
      r_u[comp] = d_u[ind * NCOMP + comp];
    }
  }
}

template <int NCOMP, int P1d>
inline __device__ void writeDofs3d(BackendData& data, const CeedInt ndofs, const CeedInt elem, const CeedInt* indices, const CeedScalar* r_v, CeedScalar* d_v) {
  if (data.tidx<P1d && data.tidy<P1d)
  {
    for(CeedInt dof = data.tidx + data.tidy*P1d; dof < P1d*P1d*P1d; dof += P1d*P1d){
      const CeedInt ind = indices[dof + elem * P1d*P1d*P1d];//TODO handle indices==NULL
      for(CeedInt comp = 0; comp < NCOMP; ++comp) {
        atomicAdd(&d_v[ind + ndofs * comp], r_v[comp]);
      }
    }
  }
}

template <int NCOMP, int P1d>
inline __device__ void writeDofsTranspose3d(BackendData& data, const CeedInt ndofs, const CeedInt elem, const CeedInt* indices, const CeedScalar* r_v, CeedScalar* d_v) {
  if (data.tidx<P1d && data.tidy<P1d)
  {
    for(CeedInt dof = data.tidx + data.tidy*P1d; dof < P1d*P1d*P1d; dof += P1d*P1d){
      const CeedInt ind = indices[dof + elem * P1d*P1d*P1d];//TODO handle indices==NULL
      for(CeedInt comp = 0; comp < NCOMP; ++comp) {
        atomicAdd(&d_v[ind * NCOMP + comp], r_v[comp]);
      }
    }
  }
}

template <int NCOMP, int Q1d>
inline __device__ void writeQuads3d(BackendData& data, const CeedInt nquads, const CeedInt elem, const CeedScalar* r_v, CeedScalar* d_v) {
  for(CeedInt dof = data.tidx + data.tidy*Q1d; dof < Q1d*Q1d*Q1d; dof+= Q1d*Q1d){
    const CeedInt ind = dof + elem * Q1d*Q1d*Q1d;
    for(CeedInt comp = 0; comp < NCOMP; ++comp) {
      d_v[ind + nquads * comp] = r_v[comp];
    }
  }
}

template <int NCOMP, int Q1d>
inline __device__ void writeQuadsTranspose3d(BackendData& data, const CeedInt nquads, const CeedInt elem, const CeedScalar* r_v, CeedScalar* d_v) {
  for(CeedInt dof = data.tidx + data.tidy*Q1d; dof < Q1d*Q1d*Q1d; dof+= Q1d*Q1d){
    const CeedInt ind = dof + elem * Q1d*Q1d*Q1d;
    for(CeedInt comp = 0; comp < NCOMP; ++comp) {
      d_v[ind * NCOMP + comp] = r_v[comp];
    }
  }
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
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &opinputfields, &opoutputfields);
  CeedChk(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, &qfinputfields, &qfoutputfields);
  CeedChk(ierr);
  CeedEvalMode emode;
  CeedTransposeMode lmode;
  // CeedVector vec;
  CeedBasis basis;
  CeedElemRestriction Erestrict;

  ostringstream code;
  string devFunctions(deviceFunctions);

  code << devFunctions;

  // Setup
  code << "\nextern \"C\" __global__ void oper(CeedInt nelem, CudaFieldsInt indices, CudaFields fields, CudaFields B, CudaFields G, CeedScalar* W) {\n";
  // Input Evecs and Restriction
  for (CeedInt i = 0; i < numinputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChk(ierr);
    if (emode == CEED_EVAL_WEIGHT) { // Skip
    } else {
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
    }
  }

  for (CeedInt i = 0; i < numoutputfields; i++) {
    code << "CeedScalar* d_v"<<i<<" = fields.out["<<i<<"];\n";
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

  // Output basis apply if needed
  for (CeedInt i = 0; i < numoutputfields; i++) {
    code << "// Output field "<<i<<"\n";
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
      break;
    case CEED_EVAL_WEIGHT: {
      Ceed ceed;
      ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
      return CeedError(ceed, 1,
                       "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
      break; // Should not occur
    }
    case CEED_EVAL_DIV:
      break; // TODO: Not implemented
    case CEED_EVAL_CURL:
      break; // TODO: Not implemented
    }
  }

  //TODO do this first in the apply?
  // Zero lvecs
  for (CeedInt i = 0; i < numoutputfields; i++) {
    // ierr = CeedOperatorFieldGetVector(opoutputfields[i], &vec); CeedChk(ierr);
    // if (vec == CEED_VECTOR_ACTIVE)
    //   vec = outvec;
    // ierr = CeedVectorSetValue(vec, 0.0); CeedChk(ierr);
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