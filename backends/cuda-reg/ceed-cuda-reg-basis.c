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
#include <ceed.h>
#include "ceed-cuda-reg.h"
#include "../cuda/ceed-cuda.h"

//*********************
// reg kernels
static const char *kernels3dreg = QUOTE(

typedef CeedScalar real;

//TODO remove the magic number 32

//Read non interleaved dofs
inline __device__ void readDofs(const int bid, const int tid, const int comp,
                                const int size, const int nelem,
                                const CeedScalar *d_U, real *r_U) {
  for (int i = 0; i < size; i++)
    //r_U[i] = d_U[tid + i*32 + bid*32*size + comp*size*nelem];
    //r_U[i] = d_U[i + tid*size + bid*32*size + comp*size*nelem ];
    r_U[i] = d_U[i + comp*size + tid*BASIS_NCOMP*size + bid*32*BASIS_NCOMP*size ];
}

//read interleaved quads
inline __device__ void readQuads(const int bid, const int tid, const int comp,
                                 const int dim, const int size, const int nelem,
                                 const CeedScalar *d_U, real *r_U) {
  for (int i = 0; i < size; i++)
    r_U[i] = d_U[i + tid*size + bid*32*size + comp*size*nelem +
                 dim*BASIS_NCOMP*nelem*size];
  //r_U[i] = d_U[tid + i*32 + bid*32*size + comp*nelem*size + dim*BASIS_NCOMP*nelem*size];
}

//Write non interleaved dofs
inline __device__ void writeDofs(const int bid, const int tid, const int comp,
                                 const int size, const int nelem,
                                 const CeedScalar *r_V, real *d_V) {
  for (int i = 0; i < size; i++)
    //d_V[i + tid*size + bid*32*size + comp*size*nelem ] = r_V[i];
    d_V[i + comp*size + tid*BASIS_NCOMP*size + bid*32*BASIS_NCOMP*size ] = r_V[i];
}

//Write interleaved quads
inline __device__ void writeQuads(const int bid, const int tid, const int comp,
                                  const int dim, const int size, const int nelem,
                                  const CeedScalar *r_V, real *d_V) {
  for (int i = 0; i < size; i++)
    d_V[i + tid*size + bid*32*size + comp*size*nelem + dim*BASIS_NCOMP*nelem*size ]
      = r_V[i];
  //d_V[tid + i*32 + bid*32*size + comp*nelem*size + dim*BASIS_NCOMP*nelem*size] = r_V[i];
}

inline __device__ void add(const int size, CeedScalar *r_V,
                           const CeedScalar *r_U) {
  for (int i = 0; i < size; i++)
    r_V[i] += r_U[i];
}

//****
// 1D
inline __device__ void Contract1d(const real *A, const real *B,
                                  int nA1,
                                  int nB1, int nB2, real *T) {
//_Pragma("unroll")
  for (int l = 0; l < nB2; l++) T[l] = 0.0;
//_Pragma("unroll")
  for (int b2 = 0; b2 < nB2; b2++)
//_Pragma("unroll")
    for (int t = 0; t < nB1; t++) {
      T[b2] += B[b2*nB1 + t] * A[t];
    }
}

inline __device__ void ContractTranspose1d(const real *A, const real *B,
    int nA1,
    int nB1, int nB2, real *T) {
//_Pragma("unroll")
  for (int l = 0; l < nB1; l++) T[l] = 0.0;
//_Pragma("unroll")
  for (int b1 = 0; b1 < nB1; b1++)
//_Pragma("unroll")
    for (int t = 0; t < nB2; t++) {
      T[b1] += B[t*nB1 + b1] * A[t];
    }
}

inline __device__ void interp1d(const CeedInt nelem, const int transpose,
                                const CeedScalar *c_B,
                                const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V) {
  real r_V[Q1D];
  real r_t[Q1D];

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  if(bid*32+tid<nelem) {
    for(int comp=0; comp<BASIS_NCOMP; comp++) {
      if(!transpose) {
        const int sizeU = P1D;
        readDofs(bid, tid, comp, sizeU, nelem, d_U, r_V);
        Contract1d(r_V, c_B, P1D, P1D, Q1D, r_t);
        const int sizeV = Q1D;
        writeQuads(bid, tid, comp, 0, sizeV, nelem, r_t, d_V);
      } else {
        const int sizeU = Q1D;
        readQuads(bid, tid, comp, 0, sizeU, nelem, d_U, r_V);
        ContractTranspose1d(r_V, c_B, Q1D, P1D, Q1D, r_t);
        const int sizeV = P1D;
        writeDofs(bid, tid, comp, sizeV, nelem, r_t, d_V);
      }
    }
  }
}

inline __device__ void grad1d(const CeedInt nelem, const int transpose,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              const CeedScalar *__restrict__ d_U,
                              CeedScalar *__restrict__ d_V) {
  //use P1D for one of these
  real r_U[Q1D];
  real r_V[Q1D];

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  int dim;

  if(bid*32+tid<nelem) {
    for(int comp=0; comp<BASIS_NCOMP; comp++) {
      if(!transpose) {
        const int sizeU = P1D;
        const int sizeV = Q1D;
        readDofs(bid, tid, comp, sizeU, nelem, d_U, r_U);
        Contract1d(r_U, c_G, P1D, P1D, Q1D, r_V);
        dim = 0;
        writeQuads(bid, tid, comp, dim, sizeV, nelem, r_V, d_V);
      } else {
        const int sizeU = Q1D;
        const int sizeV = P1D;
        dim = 0;
        readQuads(bid, tid, comp, dim, sizeU, nelem, d_U, r_U);
        ContractTranspose1d(r_U, c_G, Q1D, P1D, Q1D, r_V);
        writeDofs(bid, tid, comp, sizeV, nelem, r_V, d_V);
      }
    }
  }
}

//****
// 2D
inline __device__ void Contract2d(const real *A, const real *B,
                                  int nA1, int nA2,
                                  int nB1, int nB2, real *T) {
//_Pragma("unroll")
  for (int l = 0; l < nA2*nB2; l++) T[l] = 0.0;
//_Pragma("unroll")
  for (int a2 = 0; a2 < nA2; a2++)
//_Pragma("unroll")
    for (int b2 = 0; b2 < nB2; b2++)
//_Pragma("unroll")
      for (int t = 0; t < nB1; t++) {
        T[a2 + b2*nA2] += B[b2*nB1 + t] * A[a2*nA1 + t];
      }
}

inline __device__ void ContractTranspose2d(const real *A, const real *B,
    int nA1, int nA2,
    int nB1, int nB2, real *T) {
//_Pragma("unroll")
  for (int l = 0; l < nA2*nB1; l++) T[l] = 0.0;
//_Pragma("unroll")
  for (int a2 = 0; a2 < nA2; a2++)
//_Pragma("unroll")
    for (int b1 = 0; b1 < nB1; b1++)
//_Pragma("unroll")
      for (int t = 0; t < nB2; t++) {
        T[a2 + b1*nA2] += B[t*nB1 + b1] * A[a2*nA1 + t];
      }
}

inline __device__ void interp2d(const CeedInt nelem, const int transpose,
                                const CeedScalar *c_B,
                                const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V) {
  real r_V[Q1D*Q1D];
  real r_t[Q1D*Q1D];

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  if(bid*32+tid<nelem) {
    for(int comp=0; comp<BASIS_NCOMP; comp++) {
      if(!transpose) {
        const int sizeU = P1D*P1D;
        readDofs(bid, tid, comp, sizeU, nelem, d_U, r_V);
        Contract2d(r_V, c_B, P1D, P1D, P1D, Q1D, r_t);
        Contract2d(r_t, c_B, P1D, Q1D, P1D, Q1D, r_V);
        const int sizeV = Q1D*Q1D;
        writeQuads(bid, tid, comp, 0, sizeV, nelem, r_V, d_V);
      } else {
        const int sizeU = Q1D*Q1D;
        readQuads(bid, tid, comp, 0, sizeU, nelem, d_U, r_V);
        ContractTranspose2d(r_V, c_B, Q1D, Q1D, P1D, Q1D, r_t);
        ContractTranspose2d(r_t, c_B, Q1D, P1D, P1D, Q1D, r_V);
        const int sizeV = P1D*P1D;
        writeDofs(bid, tid, comp, sizeV, nelem, r_V, d_V);
      }
    }
  }
}

inline __device__ void grad2d(const CeedInt nelem, const int transpose,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              const CeedScalar *__restrict__ d_U,
                              CeedScalar *__restrict__ d_V) {
  //use P1D for one of these
  real r_U[Q1D*Q1D];
  real r_V[Q1D*Q1D];
  real r_t[Q1D*Q1D];

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  int dim;

  if(bid*32+tid<nelem) {
    for(int comp=0; comp<BASIS_NCOMP; comp++) {
      if(!transpose) {
        const int sizeU = P1D*P1D;
        const int sizeV = Q1D*Q1D;
        readDofs(bid, tid, comp, sizeU, nelem, d_U, r_U);
        Contract2d(r_U, c_G, P1D, P1D, P1D, Q1D, r_t);
        Contract2d(r_t, c_B, P1D, Q1D, P1D, Q1D, r_V);
        dim = 0;
        writeQuads(bid, tid, comp, dim, sizeV, nelem, r_V, d_V);
        Contract2d(r_U, c_B, P1D, P1D, P1D, Q1D, r_t);
        Contract2d(r_t, c_G, P1D, Q1D, P1D, Q1D, r_V);
        dim = 1;
        writeQuads(bid, tid, comp, dim, sizeV, nelem, r_V, d_V);
      } else {
        const int sizeU = Q1D*Q1D;
        const int sizeV = P1D*P1D;
        dim = 0;
        readQuads(bid, tid, comp, dim, sizeU, nelem, d_U, r_U);
        ContractTranspose2d(r_U, c_G, Q1D, Q1D, P1D, Q1D, r_t);
        ContractTranspose2d(r_t, c_B, Q1D, P1D, P1D, Q1D, r_V);
        dim = 1;
        readQuads(bid, tid, comp, dim, sizeU, nelem, d_U, r_U);
        ContractTranspose2d(r_U, c_B, Q1D, Q1D, P1D, Q1D, r_t);
        ContractTranspose2d(r_t, c_G, Q1D, P1D, P1D, Q1D, r_U);
        add(sizeV, r_V, r_U);
        writeDofs(bid, tid, comp, sizeV, nelem, r_V, d_V);
      }
    }
  }
}

//****
// 3D
inline __device__ void Contract3d(const real *A, const real *B,
                                  int nA1, int nA2, int nA3,
                                  int nB1, int nB2, real *T) {
//_Pragma("unroll")
  for (int l = 0; l < nA2*nA3*nB2; l++) T[l] = 0.0;
//_Pragma("unroll")
  for (int a2 = 0; a2 < nA2; a2++)
//_Pragma("unroll")
    for (int a3 = 0; a3 < nA3; a3++)
//_Pragma("unroll")
      for (int b2 = 0; b2 < nB2; b2++)
//_Pragma("unroll")
        for (int t = 0; t < nB1; t++) {
          T[a2 + a3*nA2 + b2*nA2*nA3] += B[b2*nB1 + t] * A[a3*nA2*nA1 + a2*nA1 + t];
        }
}

inline __device__ void ContractTranspose3d(const real *A, const real *B,
    int nA1, int nA2, int nA3,
    int nB1, int nB2, real *T) {
//_Pragma("unroll")
  for (int l = 0; l < nA2*nA3*nB1; l++) T[l] = 0.0;
//_Pragma("unroll")
  for (int a2 = 0; a2 < nA2; a2++)
//_Pragma("unroll")
    for (int a3 = 0; a3 < nA3; a3++)
//_Pragma("unroll")
      for (int b1 = 0; b1 < nB1; b1++)
//_Pragma("unroll")
        for (int t = 0; t < nB2; t++) {
          T[a2 + a3*nA2 + b1*nA2*nA3] += B[t*nB1 + b1] * A[a3*nA2*nA1 + a2*nA1 + t];
        }
}

inline __device__ void interp3d(const CeedInt nelem, const int transpose,
                                const CeedScalar *c_B,
                                const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V) {
  real r_V[Q1D*Q1D*Q1D];
  real r_t[Q1D*Q1D*Q1D];

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  if(bid*32+tid<nelem) {
    for(int comp=0; comp<BASIS_NCOMP; comp++) {
      if(!transpose) {
        const int sizeU = P1D*P1D*P1D;
        const int sizeV = Q1D*Q1D*Q1D;
        readDofs(bid, tid, comp, sizeU, nelem, d_U, r_V);
        Contract3d(r_V, c_B, P1D, P1D, P1D, P1D, Q1D, r_t);
        Contract3d(r_t, c_B, P1D, P1D, Q1D, P1D, Q1D, r_V);
        Contract3d(r_V, c_B, P1D, Q1D, Q1D, P1D, Q1D, r_t);
        writeQuads(bid, tid, comp, 0, sizeV, nelem, r_t, d_V);
      } else {
        const int sizeU = Q1D*Q1D*Q1D;
        const int sizeV = P1D*P1D*P1D;
        readQuads(bid, tid, comp, 0, sizeU, nelem, d_U, r_V);
        ContractTranspose3d(r_V, c_B, Q1D, Q1D, Q1D, P1D, Q1D, r_t);
        ContractTranspose3d(r_t, c_B, Q1D, Q1D, P1D, P1D, Q1D, r_V);
        ContractTranspose3d(r_V, c_B, Q1D, P1D, P1D, P1D, Q1D, r_t);
        writeDofs(bid, tid, comp, sizeV, nelem, r_t, d_V);
      }
    }
  }
}

inline __device__ void grad3d(const CeedInt nelem, const int transpose,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              const CeedScalar *__restrict__ d_U,
                              CeedScalar *__restrict__ d_V) {
  //use P1D for one of these
  real r_U[Q1D*Q1D*Q1D];
  real r_V[Q1D*Q1D*Q1D];
  real r_t[Q1D*Q1D*Q1D];

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  int dim;

  if(bid*32+tid<nelem) {
    for(int comp=0; comp<BASIS_NCOMP; comp++) {
      if(!transpose) {
        const int sizeU = P1D*P1D*P1D;
        const int sizeV = Q1D*Q1D*Q1D;
        readDofs(bid, tid, comp, sizeU, nelem, d_U, r_U);
        Contract3d(r_U, c_G, P1D, P1D, P1D, P1D, Q1D, r_V);
        Contract3d(r_V, c_B, P1D, P1D, Q1D, P1D, Q1D, r_t);
        Contract3d(r_t, c_B, P1D, Q1D, Q1D, P1D, Q1D, r_V);
        dim = 0;
        writeQuads(bid, tid, comp, dim, sizeV, nelem, r_V, d_V);
        Contract3d(r_U, c_B, P1D, P1D, P1D, P1D, Q1D, r_V);
        Contract3d(r_V, c_G, P1D, P1D, Q1D, P1D, Q1D, r_t);
        Contract3d(r_t, c_B, P1D, Q1D, Q1D, P1D, Q1D, r_V);
        dim = 1;
        writeQuads(bid, tid, comp, dim, sizeV, nelem, r_V, d_V);
        Contract3d(r_U, c_B, P1D, P1D, P1D, P1D, Q1D, r_V);
        Contract3d(r_V, c_B, P1D, P1D, Q1D, P1D, Q1D, r_t);
        Contract3d(r_t, c_G, P1D, Q1D, Q1D, P1D, Q1D, r_V);
        dim = 2;
        writeQuads(bid, tid, comp, dim, sizeV, nelem, r_V, d_V);
      } else {
        const int sizeU = Q1D*Q1D*Q1D;
        const int sizeV = P1D*P1D*P1D;
        dim = 0;
        readQuads(bid, tid, comp, dim, sizeU, nelem, d_U, r_U);
        ContractTranspose3d(r_U, c_G, Q1D, Q1D, Q1D, P1D, Q1D, r_t);
        ContractTranspose3d(r_t, c_B, Q1D, Q1D, P1D, P1D, Q1D, r_U);
        ContractTranspose3d(r_U, c_B, Q1D, P1D, P1D, P1D, Q1D, r_V);
        dim = 1;
        readQuads(bid, tid, comp, dim, sizeU, nelem, d_U, r_U);
        ContractTranspose3d(r_U, c_B, Q1D, Q1D, Q1D, P1D, Q1D, r_t);
        ContractTranspose3d(r_t, c_G, Q1D, Q1D, P1D, P1D, Q1D, r_U);
        ContractTranspose3d(r_U, c_B, Q1D, P1D, P1D, P1D, Q1D, r_t);
        add(sizeV, r_V, r_t);
        dim = 2;
        readQuads(bid, tid, comp, dim, sizeU, nelem, d_U, r_U);
        ContractTranspose3d(r_U, c_B, Q1D, Q1D, Q1D, P1D, Q1D, r_t);
        ContractTranspose3d(r_t, c_B, Q1D, Q1D, P1D, P1D, Q1D, r_U);
        ContractTranspose3d(r_U, c_G, Q1D, P1D, P1D, P1D, Q1D, r_t);
        add(sizeV, r_V, r_t);
        writeDofs(bid, tid, comp, sizeV, nelem, r_V, d_V);
      }
    }
  }
}

extern "C" __global__ void interp(const CeedInt nelem, const int transpose,
                                  const CeedScalar *c_B,
                                  const CeedScalar *__restrict__ d_U,
                                  CeedScalar *__restrict__ d_V) {
  if (BASIS_DIM==1) {
    interp1d(nelem, transpose, c_B, d_U, d_V);
  } else if (BASIS_DIM==2) {
    interp2d(nelem, transpose, c_B, d_U, d_V);
  } else if (BASIS_DIM==3) {
    interp3d(nelem, transpose, c_B, d_U, d_V);
  }
}

extern "C" __global__ void grad(const CeedInt nelem, const int transpose,
                                const CeedScalar *c_B, const CeedScalar *c_G,
                                const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V) {
  if (BASIS_DIM==1) {
    grad1d(nelem, transpose, c_B, c_G, d_U, d_V);
  } else if (BASIS_DIM==2) {
    grad2d(nelem, transpose, c_B, c_G, d_U, d_V);
  } else if (BASIS_DIM==3) {
    grad3d(nelem, transpose, c_B, c_G, d_U, d_V);
  }
}

__device__ void weight1d(const CeedInt nelem, const CeedScalar *qweight1d,
                         CeedScalar *w) {
  CeedScalar w1d[Q1D];
  for (int i = 0; i < Q1D; ++i) {
    w1d[i] = qweight1d[i];
  }
  for (int e = blockIdx.x * blockDim.x + threadIdx.x;
       e < nelem;
       e += blockDim.x * gridDim.x) {
    for (int i = 0; i < Q1D; ++i) {
      //const int ind = e + i*nelem;//interleaved
      const int ind = e*Q1D + i;//sequential
      w[ind] = w1d[i];
    }
  }
}

__device__ void weight2d(const CeedInt nelem, const CeedScalar *qweight1d,
                         CeedScalar *w) {
  CeedScalar w1d[Q1D];
  for (int i = 0; i < Q1D; ++i) {
    w1d[i] = qweight1d[i];
  }
  for (int e = blockIdx.x * blockDim.x + threadIdx.x;
       e < nelem;
       e += blockDim.x * gridDim.x) {
    for (int i = 0; i < Q1D; ++i) {
      for (int j = 0; j < Q1D; ++j) {
        //const int ind = e + i*nelem + j*Q1D*nelem;//interleaved
        const int ind = e*Q1D*Q1D + i + j*Q1D;//sequential
        w[ind] = w1d[i]*w1d[j];
      }
    }
  }
}

__device__ void weight3d(const CeedInt nelem, const CeedScalar *qweight1d,
                         CeedScalar *w) {
  CeedScalar w1d[Q1D];
  for (int i = 0; i < Q1D; ++i) {
    w1d[i] = qweight1d[i];
  }
  for (int e = blockIdx.x * blockDim.x + threadIdx.x;
       e < nelem;
       e += blockDim.x * gridDim.x) {
    for (int i = 0; i < Q1D; ++i) {
      for (int j = 0; j < Q1D; ++j) {
        for (int k = 0; k < Q1D; ++k) {
          //const int ind = e + i*nelem + j*Q1D*nelem + k*Q1D*Q1D*nelem;//interleaved
          const int ind = e*Q1D*Q1D*Q1D + i + j*Q1D + k*Q1D*Q1D;//sequential
          w[ind] = w1d[i]*w1d[j]*w1d[k];
        }
      }
    }
  }
}

extern "C" __global__ void weight(const CeedInt nelem,
                                  const CeedScalar *__restrict__ qweight1d,
                                  CeedScalar *__restrict__ v) {
  if (BASIS_DIM==1) {
    weight1d(nelem, qweight1d, v);
  } else if (BASIS_DIM==2) {
    weight2d(nelem, qweight1d, v);
  } else if (BASIS_DIM==3) {
    weight3d(nelem, qweight1d, v);
  }
}

);

int CeedCudaInitInterp(CeedScalar *d_B, CeedInt P1d, CeedInt Q1d,
                       CeedScalar **c_B);
int CeedCudaInitInterpGrad(CeedScalar *d_B, CeedScalar *d_G, CeedInt P1d,
                           CeedInt Q1d, CeedScalar **c_B_ptr,
                           CeedScalar **c_G_ptr);

int CeedBasisApply_Cuda_reg(CeedBasis basis, const CeedInt nelem,
                            CeedTransposeMode tmode,
                            CeedEvalMode emode, CeedVector u, CeedVector v) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
  Ceed_Cuda_reg *ceed_Cuda;
  CeedGetData(ceed, (void *) &ceed_Cuda); CeedChk(ierr);
  CeedBasis_Cuda_reg *data;
  CeedBasisGetData(basis, (void *)&data); CeedChk(ierr);
  const CeedInt transpose = tmode == CEED_TRANSPOSE;
  const int warpsize  = 32;
  const int blocksize = warpsize;
  const int gridsize  = nelem/warpsize + ( (nelem/warpsize*warpsize<nelem)? 1 :
                        0 );

  const CeedScalar *d_u;
  CeedScalar *d_v;
  if(emode!=CEED_EVAL_WEIGHT) {
    ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u); CeedChk(ierr);
  }
  ierr = CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v); CeedChk(ierr);

  if (tmode == CEED_TRANSPOSE) {
    CeedInt length;
    ierr = CeedVectorGetLength(v, &length); CeedChk(ierr);
    ierr = cudaMemset(d_v, 0, length * sizeof(CeedScalar)); CeedChk_Cu(ceed,ierr);
  }
  if (emode == CEED_EVAL_INTERP) {
    //TODO: check performance difference between c_B and d_B
    CeedInt P1d, Q1d;
    ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChk(ierr);
    ierr = CeedCudaInitInterp(data->d_interp1d, P1d, Q1d, &data->c_B);
    CeedChk(ierr);
    void *interpargs[] = {(void *) &nelem, (void *) &transpose, &data->c_B,
                          &d_u, &d_v};
    ierr = CeedRunKernelCuda(ceed, data->interp, gridsize, blocksize, interpargs);
    CeedChk(ierr);
  } else if (emode == CEED_EVAL_GRAD) {
    CeedInt P1d, Q1d;
    ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChk(ierr);
    ierr = CeedCudaInitInterpGrad(data->d_interp1d, data->d_grad1d, P1d, Q1d,
                                  &data->c_B, &data->c_G);
    CeedChk(ierr);
    void *gradargs[] = {(void *) &nelem, (void *) &transpose, &data->c_B,
                        &data->c_G, &d_u, &d_v};
    ierr = CeedRunKernelCuda(ceed, data->grad, gridsize, blocksize, gradargs);
    CeedChk(ierr);
  } else if (emode == CEED_EVAL_WEIGHT) {
    void *weightargs[] = {(void *) &nelem, (void *) &data->d_qweight1d, &d_v};
    const int blocksize = 32;
    int gridsize = nelem/32;
    if (blocksize * gridsize < nelem)
      gridsize += 1;
    ierr = CeedRunKernelCuda(ceed, data->weight, gridsize, blocksize, weightargs);
  }

  if(emode!=CEED_EVAL_WEIGHT) {
    ierr = CeedVectorRestoreArrayRead(u, &d_u); CeedChk(ierr);
  }
  ierr = CeedVectorRestoreArray(v, &d_v); CeedChk(ierr);

  return 0;
}

static int CeedBasisDestroy_Cuda_reg(CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);

  CeedBasis_Cuda_reg *data;
  ierr = CeedBasisGetData(basis, (void *) &data); CeedChk(ierr);

  CeedChk_Cu(ceed, cuModuleUnload(data->module));

  ierr = cudaFree(data->d_qweight1d); CeedChk(ierr);
  ierr = cudaFree(data->d_interp1d); CeedChk(ierr);
  ierr = cudaFree(data->d_grad1d); CeedChk(ierr);

  ierr = CeedFree(&data); CeedChk(ierr);

  return 0;
}

int CeedBasisCreateTensorH1_Cuda_reg(CeedInt dim, CeedInt P1d, CeedInt Q1d,
                                     const CeedScalar *interp1d,
                                     const CeedScalar *grad1d,
                                     const CeedScalar *qref1d,
                                     const CeedScalar *qweight1d,
                                     CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
  if (Q1d<P1d) {
    return CeedError(ceed, 1, "Backend does not implement underintegrated basis.");
  }
  CeedBasis_Cuda_reg *data;
  ierr = CeedCalloc(1, &data); CeedChk(ierr);

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

  CeedInt ncomp;
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChk(ierr);
  ierr = CeedCompileCuda(ceed, kernels3dreg, &data->module, 7,
                         "Q1D", Q1d,
                         "P1D", P1d,
                         "BASIS_BUF_LEN", ncomp * CeedIntPow(Q1d > P1d ?
                             Q1d : P1d, dim),
                         "BASIS_DIM", dim,
                         "BASIS_NCOMP", ncomp,
                         "BASIS_ELEMSIZE", CeedIntPow(P1d, dim),
                         "BASIS_NQPT", CeedIntPow(Q1d, dim)
                        ); CeedChk(ierr);
  ierr = CeedGetKernelCuda(ceed, data->module, "interp", &data->interp);
  CeedChk(ierr);
  ierr = CeedGetKernelCuda(ceed, data->module, "grad", &data->grad);
  CeedChk(ierr);
  ierr = CeedGetKernelCuda(ceed, data->module, "weight", &data->weight);
  CeedChk(ierr);

  ierr = CeedBasisSetData(basis, (void *)&data);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Apply",
                                CeedBasisApply_Cuda_reg);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Destroy",
                                CeedBasisDestroy_Cuda_reg);
  CeedChk(ierr);
  return 0;
}
