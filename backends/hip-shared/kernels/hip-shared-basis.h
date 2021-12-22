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

//------------------------------------------------------------------------------
// Shared mem kernels
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Sum input into output
//------------------------------------------------------------------------------
inline __device__ void add(CeedScalar *r_V, const CeedScalar *r_U) {
  for (int i = 0; i < P1D; i++)
    r_V[i] += r_U[i];
}

//------------------------------------------------------------------------------
// Load matrices for basis actions
//------------------------------------------------------------------------------
inline __device__ void loadMatrix(const CeedScalar* d_B, CeedScalar* B) {
  CeedInt tid = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y*blockDim.x;
  for (CeedInt i = tid; i < P1D*Q1D; i += blockDim.x*blockDim.y*blockDim.z)
    B[i] = d_B[i];
}

//------------------------------------------------------------------------------
// 1D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Read DoFs
//------------------------------------------------------------------------------
inline __device__ void readDofs1d(const int elem, const int t_idx,
                                  const int t_idy, const int t_idz,const int comp,
                                  const int num_elem, const CeedScalar *d_U,
                                  CeedScalar *slice) {
  for (int i = 0; i < P1D; i++)
    slice[i + t_idz*T1D] = d_U[i + elem*P1D + comp*P1D*num_elem];
  for (int i = P1D; i < Q1D; i++)
    slice[i + t_idz*T1D] = 0.0;
}

//------------------------------------------------------------------------------
// Write DoFs
//------------------------------------------------------------------------------
inline __device__ void writeDofs1d(const int elem, const int t_idx,
                                   const int t_idy, const int comp,
                                   const int num_elem, const CeedScalar &r_V,
                                   CeedScalar *d_V) {
  if (t_idx<P1D)
    d_V[t_idx + elem*P1D + comp*P1D*num_elem] = r_V;
}

//------------------------------------------------------------------------------
// Read quadrature point data
//------------------------------------------------------------------------------
inline __device__ void readQuads1d(const int elem, const int t_idx,
                                   const int t_idy, const int t_idz, const int comp,
                                   const int dim, const int num_elem,
                                   const CeedScalar *d_U, CeedScalar *slice) {
  for (int i = 0; i < Q1D; i++)
    slice[i + t_idz*T1D] = d_U[i + elem*Q1D + comp*Q1D*num_elem +
                            dim*BASIS_NCOMP*num_elem*Q1D];
  for (int i = Q1D; i < P1D; i++)
    slice[i + t_idz*T1D] = 0.0;
}

//------------------------------------------------------------------------------
// Write quadrature point data
//------------------------------------------------------------------------------
inline __device__ void writeQuads1d(const int elem, const int t_idx,
                                    const int t_idy, const int comp,
                                    const int dim, const int num_elem,
                                    const CeedScalar &r_V, CeedScalar *d_V) {
  if (t_idx<Q1D)
    d_V[t_idx + elem*Q1D + comp*Q1D*num_elem + dim*BASIS_NCOMP*num_elem*Q1D] = r_V;
}

//------------------------------------------------------------------------------
// 1D tensor contraction
//------------------------------------------------------------------------------
inline __device__ void ContractX1d(CeedScalar *slice, const int t_idx,
                                   const int t_idy, const int t_idz,
                                   const CeedScalar &U, const CeedScalar *B,
                                   CeedScalar &V) {
  V = 0.0;
  for (int i = 0; i < P1D; ++i)
    V += B[i + t_idx*P1D] * slice[i + t_idz*T1D]; // Contract x direction
}

//------------------------------------------------------------------------------
// 1D transpose tensor contraction
//------------------------------------------------------------------------------
inline __device__ void ContractTransposeX1d(CeedScalar *slice, const int t_idx,
    const int t_idy, const int t_idz,
    const CeedScalar &U, const CeedScalar *B, CeedScalar &V) {
  V = 0.0;
  for (int i = 0; i < Q1D; ++i)
    V += B[t_idx + i*P1D] * slice[i + t_idz*T1D]; // Contract x direction
}

//------------------------------------------------------------------------------
// 1D interpolate to quadrature points
//------------------------------------------------------------------------------
inline __device__ void interp_1d(const CeedInt num_elem, const int transpose,
                                const CeedScalar *s_B,
                                const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V,
                                CeedScalar *slice) {
  CeedScalar r_V;
  CeedScalar r_t;

  const int t_idx = threadIdx.x;
  const int t_idy = threadIdx.y;
  const int t_idz = threadIdx.z;

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < num_elem;
       elem += gridDim.x*blockDim.z) {
    for (int comp = 0; comp < BASIS_NCOMP; comp++) {
      if (!transpose) {
        readDofs1d(elem, t_idx, t_idy, t_idz, comp, num_elem, d_U, slice);
        ContractX1d(slice, t_idx, t_idy, t_idz, r_t, s_B, r_V);
        writeQuads1d(elem, t_idx, t_idy, comp, 0, num_elem, r_V, d_V);
      } else {
        readQuads1d(elem, t_idx, t_idy, t_idz, comp, 0, num_elem, d_U, slice);
        ContractTransposeX1d(slice, t_idx, t_idy, t_idz, r_t, s_B, r_V);
        writeDofs1d(elem, t_idx, t_idy, comp, num_elem, r_V, d_V);
      }
    }
  }
}

//------------------------------------------------------------------------------
// 1D derivatives at quadrature points
//------------------------------------------------------------------------------
inline __device__ void grad_1d(const CeedInt num_elem, const int transpose,
                              const CeedScalar *s_B, const CeedScalar *s_G,
                              const CeedScalar *__restrict__ d_U,
                              CeedScalar *__restrict__ d_V,
                              CeedScalar *slice) {
  CeedScalar r_U;
  CeedScalar r_V;

  const int t_idx = threadIdx.x;
  const int t_idy = threadIdx.y;
  const int t_idz = threadIdx.z;
  int dim;

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < num_elem;
       elem += gridDim.x*blockDim.z) {
    for(int comp = 0; comp < BASIS_NCOMP; comp++) {
      if (!transpose) {
        readDofs1d(elem, t_idx, t_idy, t_idz, comp, num_elem, d_U, slice);
        ContractX1d(slice, t_idx, t_idy, t_idz, r_U, s_G, r_V);
        dim = 0;
        writeQuads1d(elem, t_idx, t_idy, comp, dim, num_elem, r_V, d_V);
      } else {
        dim = 0;
        readQuads1d(elem, t_idx, t_idy, t_idz, comp, dim, num_elem, d_U, slice);
        ContractTransposeX1d(slice, t_idx, t_idy, t_idz, r_U, s_G, r_V);
        writeDofs1d(elem, t_idx, t_idy, comp, num_elem, r_V, d_V);
      }
    }
  }
}

//------------------------------------------------------------------------------
// 1D Quadrature weights
//------------------------------------------------------------------------------
__device__ void weight_1d(const CeedInt num_elem, const CeedScalar *q_weight_1d,
                         CeedScalar *w) {
  const int tid = threadIdx.x;
  const CeedScalar weight = q_weight_1d[tid];
  for (CeedInt elem = blockIdx.x*blockDim.y + threadIdx.y; elem < num_elem;
       elem += gridDim.x*blockDim.y) {
    const int ind = elem*Q1D + tid;
    w[ind] = weight;
  }
}

//------------------------------------------------------------------------------
// 2D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Read DoFs
//------------------------------------------------------------------------------
inline __device__ void readDofs2d(const int elem, const int t_idx,
                                  const int t_idy, const int comp,
                                  const int num_elem, const CeedScalar *d_U,
                                  CeedScalar &U) {
  U = (t_idx<P1D && t_idy<P1D) ?
      d_U[t_idx + t_idy*P1D + elem*P1D*P1D + comp*P1D*P1D*num_elem] : 0.0;
}

//------------------------------------------------------------------------------
// Write DoFs
//------------------------------------------------------------------------------
inline __device__ void writeDofs2d(const int elem, const int t_idx,
                                   const int t_idy, const int comp,
                                   const int num_elem, const CeedScalar &r_V,
                                   CeedScalar *d_V) {
  if (t_idx<P1D && t_idy<P1D)
    d_V[t_idx + t_idy*P1D + elem*P1D*P1D + comp*P1D*P1D*num_elem] = r_V;
}

//------------------------------------------------------------------------------
// Read quadrature point data
//------------------------------------------------------------------------------
inline __device__ void readQuads2d(const int elem, const int t_idx,
                                   const int t_idy, const int comp,
                                   const int dim, const int num_elem,
                                   const CeedScalar *d_U, CeedScalar &U ) {
  U = (t_idx<Q1D && t_idy<Q1D) ?
      d_U[t_idx + t_idy*Q1D + elem*Q1D*Q1D + comp*Q1D*Q1D*num_elem +
      dim*BASIS_NCOMP*num_elem*Q1D*Q1D] : 0.0;
}

//------------------------------------------------------------------------------
// Write quadrature point data
//------------------------------------------------------------------------------
inline __device__ void writeQuads2d(const int elem, const int t_idx,
                                    const int t_idy, const int comp,
                                    const int dim, const int num_elem,
                                    const CeedScalar &r_V, CeedScalar *d_V) {
  if (t_idx<Q1D && t_idy<Q1D)
    d_V[t_idx + t_idy*Q1D + elem*Q1D*Q1D + comp*Q1D*Q1D*num_elem +
    dim*BASIS_NCOMP*num_elem*Q1D*Q1D] = r_V;
}

//------------------------------------------------------------------------------
// 2D tensor contraction x
//------------------------------------------------------------------------------
inline __device__ void ContractX2d(CeedScalar *slice, const int t_idx,
                                   const int t_idy, const int t_idz,
                                   const CeedScalar &U, const CeedScalar *B,
                                   CeedScalar &V) {
  slice[t_idx + t_idy*T1D + t_idz*T1D*T1D] = U;
  __syncthreads();
  V = 0.0;
  if (t_idx < Q1D)
    for (int i = 0; i < P1D; ++i)
      V += B[i + t_idx*P1D] * slice[i + t_idy*T1D + t_idz*T1D*T1D]; // Contract x direction
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D tensor contraction y
//------------------------------------------------------------------------------
inline __device__ void ContractY2d(CeedScalar *slice, const int t_idx,
                                   const int t_idy, const int t_idz,
                                   const CeedScalar &U, const CeedScalar *B,
                                   CeedScalar &V) {
  slice[t_idx + t_idy*T1D + t_idz*T1D*T1D] = U;
  __syncthreads();
  V = 0.0;
  if (t_idy < Q1D)
    for (int i = 0; i < P1D; ++i)
      V += B[i + t_idy*P1D] * slice[t_idx + i*T1D + t_idz*T1D*T1D]; // Contract y direction
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D transpose tensor contraction y
//------------------------------------------------------------------------------
inline __device__ void ContractTransposeY2d(CeedScalar *slice, const int t_idx,
    const int t_idy, const int t_idz,
    const CeedScalar &U, const CeedScalar *B, CeedScalar &V) {
  slice[t_idx + t_idy*T1D + t_idz*T1D*T1D] = U;
  __syncthreads();
  V = 0.0;
  if (t_idy < P1D)
    for (int i = 0; i < Q1D; ++i)
      V += B[t_idy + i*P1D] * slice[t_idx + i*T1D + t_idz*T1D*T1D]; // Contract y direction
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D transpose tensor contraction x
//------------------------------------------------------------------------------
inline __device__ void ContractTransposeX2d(CeedScalar *slice, const int t_idx,
    const int t_idy, const int t_idz,
    const CeedScalar &U, const CeedScalar *B, CeedScalar &V) {
  slice[t_idx + t_idy*T1D + t_idz*T1D*T1D] = U;
  __syncthreads();
  V = 0.0;
  if (t_idx < P1D)
    for (int i = 0; i < Q1D; ++i)
      V += B[t_idx + i*P1D] * slice[i + t_idy*T1D + t_idz*T1D*T1D]; // Contract x direction
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D interpolate to quadrature points
//------------------------------------------------------------------------------
inline __device__ void interp2d(const CeedInt num_elem, const int transpose,
                                const CeedScalar *s_B,
                                const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V,
                                CeedScalar *slice) {
  CeedScalar r_V;
  CeedScalar r_t;

  const int t_idx = threadIdx.x;
  const int t_idy = threadIdx.y;
  const int t_idz = threadIdx.z;
  const int blockElem = t_idz/BASIS_NCOMP;
  const int elemsPerBlock = blockDim.z/BASIS_NCOMP;
  const int comp = t_idz%BASIS_NCOMP;

  for (CeedInt elem = blockIdx.x*elemsPerBlock + blockElem; elem < num_elem;
       elem += gridDim.x*elemsPerBlock) {
    const int comp = t_idz%BASIS_NCOMP;
    r_V = 0.0;
    r_t = 0.0;
    if (!transpose) {
      readDofs2d(elem, t_idx, t_idy, comp, num_elem, d_U, r_V);
      ContractX2d(slice, t_idx, t_idy, t_idz, r_V, s_B, r_t);
      ContractY2d(slice, t_idx, t_idy, t_idz, r_t, s_B, r_V);
      writeQuads2d(elem, t_idx, t_idy, comp, 0, num_elem, r_V, d_V);
    } else {
      readQuads2d(elem, t_idx, t_idy, comp, 0, num_elem, d_U, r_V);
      ContractTransposeY2d(slice, t_idx, t_idy, t_idz, r_V, s_B, r_t);
      ContractTransposeX2d(slice, t_idx, t_idy, t_idz, r_t, s_B, r_V);
      writeDofs2d(elem, t_idx, t_idy, comp, num_elem, r_V, d_V);
    }
  }
}

//------------------------------------------------------------------------------
// 2D derivatives at quadrature points
//------------------------------------------------------------------------------
inline __device__ void grad2d(const CeedInt num_elem, const int transpose,
                              const CeedScalar *s_B, const CeedScalar *s_G,
                              const CeedScalar *__restrict__ d_U,
                              CeedScalar *__restrict__ d_V, CeedScalar *slice) {
  CeedScalar r_U;
  CeedScalar r_V;
  CeedScalar r_t;

  const int t_idx = threadIdx.x;
  const int t_idy = threadIdx.y;
  const int t_idz = threadIdx.z;
  const int blockElem = t_idz/BASIS_NCOMP;
  const int elemsPerBlock = blockDim.z/BASIS_NCOMP;
  const int comp = t_idz%BASIS_NCOMP;
  int dim;

  for (CeedInt elem = blockIdx.x*elemsPerBlock + blockElem; elem < num_elem;
       elem += gridDim.x*elemsPerBlock) {
    if (!transpose) {
      readDofs2d(elem, t_idx, t_idy, comp, num_elem, d_U, r_U);
      ContractX2d(slice, t_idx, t_idy, t_idz, r_U, s_G, r_t);
      ContractY2d(slice, t_idx, t_idy, t_idz, r_t, s_B, r_V);
      dim = 0;
      writeQuads2d(elem, t_idx, t_idy, comp, dim, num_elem, r_V, d_V);
      ContractX2d(slice, t_idx, t_idy, t_idz, r_U, s_B, r_t);
      ContractY2d(slice, t_idx, t_idy, t_idz, r_t, s_G, r_V);
      dim = 1;
      writeQuads2d(elem, t_idx, t_idy, comp, dim, num_elem, r_V, d_V);
    } else {
      dim = 0;
      readQuads2d(elem, t_idx, t_idy, comp, dim, num_elem, d_U, r_U);
      ContractTransposeY2d(slice, t_idx, t_idy, t_idz, r_U, s_B, r_t);
      ContractTransposeX2d(slice, t_idx, t_idy, t_idz, r_t, s_G, r_V);
      dim = 1;
      readQuads2d(elem, t_idx, t_idy, comp, dim, num_elem, d_U, r_U);
      ContractTransposeY2d(slice, t_idx, t_idy, t_idz, r_U, s_G, r_t);
      ContractTransposeX2d(slice, t_idx, t_idy, t_idz, r_t, s_B, r_U);
      r_V += r_U;
      writeDofs2d(elem, t_idx, t_idy, comp, num_elem, r_V, d_V);
    }
  }
}

//------------------------------------------------------------------------------
// 2D quadrature weights
//------------------------------------------------------------------------------
__device__ void weight2d(const CeedInt num_elem, const CeedScalar *q_weight_1d,
                         CeedScalar *w) {
  const int i = threadIdx.x;
  const int j = threadIdx.y;
  const CeedScalar weight = q_weight_1d[i]*q_weight_1d[j];
  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < num_elem;
       elem += gridDim.x*blockDim.z) {
    const int ind = elem*Q1D*Q1D + i + j*Q1D;
    w[ind] = weight;
  }
}

//------------------------------------------------------------------------------
// 3D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Read DoFs
//------------------------------------------------------------------------------
inline __device__ void readDofs3d(const int elem, const int t_idx,
                                  const int t_idy, const int comp,
                                  const int num_elem, const CeedScalar *d_U,
                                  CeedScalar *r_U) {
  for (int i = 0; i < P1D; i++)
    r_U[i] = (t_idx < P1D && t_idy < P1D) ?
              d_U[t_idx + t_idy*P1D + i*P1D*P1D + elem*P1D*P1D*P1D +
                  comp*P1D*P1D*P1D*num_elem] : 0.0;
  for (int i = P1D; i < Q1D; i++)
    r_U[i] = 0.0;
}

//------------------------------------------------------------------------------
// Write DoFs
//------------------------------------------------------------------------------
inline __device__ void writeDofs3d(const int elem, const int t_idx,
                                   const int t_idy, const int comp,
                                   const int num_elem, const CeedScalar *r_V,
                                   CeedScalar *d_V) {
  if (t_idx < P1D && t_idy < P1D) {
    for (int i = 0; i < P1D; i++)
      d_V[t_idx + t_idy*P1D + i*P1D*P1D + elem*P1D*P1D*P1D +
          comp*P1D*P1D*P1D*num_elem] = r_V[i];
  }
}

//------------------------------------------------------------------------------
// Read quadrature point data
//------------------------------------------------------------------------------
inline __device__ void readQuads3d(const int elem, const int t_idx,
                                   const int t_idy, const int comp,
                                   const int dim, const int num_elem,
                                   const CeedScalar *d_U, CeedScalar *r_U) {
  for (int i = 0; i < Q1D; i++)
    r_U[i] = (t_idx < Q1D && t_idy < Q1D) ? 
              d_U[t_idx + t_idy*Q1D + i*Q1D*Q1D + elem*Q1D*Q1D*Q1D +
              comp*Q1D*Q1D*Q1D*num_elem + dim*BASIS_NCOMP*num_elem*Q1D*Q1D*Q1D] : 0.0;
  for (int i = Q1D; i < P1D; i++)
    r_U[i] = 0.0;
}

//------------------------------------------------------------------------------
// Write quadrature point data
//------------------------------------------------------------------------------
inline __device__ void writeQuads3d(const int elem, const int t_idx,
                                    const int t_idy, const int comp,
                                    const int dim, const int num_elem,
                                    const CeedScalar *r_V, CeedScalar *d_V) {
  if (t_idx < Q1D && t_idy < Q1D) {
    for (int i = 0; i < Q1D; i++)
      d_V[t_idx + t_idy*Q1D + i*Q1D*Q1D + elem*Q1D*Q1D*Q1D + comp*Q1D*Q1D*Q1D*num_elem +
          dim*BASIS_NCOMP*num_elem*Q1D*Q1D*Q1D] = r_V[i];
  }
}

//------------------------------------------------------------------------------
// 3D tensor contract x
//------------------------------------------------------------------------------
inline __device__ void ContractX3d(CeedScalar *slice, const int t_idx,
                                   const int t_idy, const int t_idz,
                                   const CeedScalar *U,
                                   const CeedScalar *B,
                                   CeedScalar *V) {
  for (int k = 0; k < P1D; ++k) {
    slice[t_idx + t_idy*T1D + t_idz*T1D*T1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (t_idx < Q1D && t_idy < P1D)
      for (int i = 0; i < P1D; ++i)
        V[k] += B[i + t_idx*P1D] * slice[i + t_idy*T1D + t_idz*T1D*T1D]; // Contract x direction
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D tensor contract y
//------------------------------------------------------------------------------
inline __device__ void ContractY3d(CeedScalar *slice, const int t_idx,
                                   const int t_idy, const int t_idz,
                                   const CeedScalar *U,
                                   const CeedScalar *B,
                                   CeedScalar *V) {
  for (int k = 0; k < P1D; ++k) {
    slice[t_idx + t_idy*T1D + t_idz*T1D*T1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (t_idx < Q1D && t_idy < Q1D)
      for (int i = 0; i < P1D; ++i)
        V[k] += B[i + t_idy*P1D] * slice[t_idx + i*T1D + t_idz*T1D*T1D]; // Contract y direction
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D tensor contract z
//------------------------------------------------------------------------------
inline __device__ void ContractZ3d(CeedScalar *slice, const int t_idx,
                                   const int t_idy, const int t_idz,
                                   const CeedScalar *U,
                                   const CeedScalar *B,
                                   CeedScalar *V) {
  for (int k = 0; k < Q1D; ++k) {
    V[k] = 0.0;
    if (t_idx < Q1D && t_idy < Q1D)
      for (int i = 0; i < P1D; ++i)
        V[k] += B[i + k*P1D] * U[i]; // Contract z direction
  }
  for (int k = Q1D; k < P1D; ++k)
    V[k] = 0.0;
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract z
//------------------------------------------------------------------------------
inline __device__ void ContractTransposeZ3d(CeedScalar *slice, const int t_idx,
                                            const int t_idy, const int t_idz,
                                            const CeedScalar *U,
                                            const CeedScalar *B,
                                            CeedScalar *V) {
  for (int k = 0; k < P1D; ++k) {
    V[k] = 0.0;
    if (t_idx < Q1D && t_idy < Q1D)
      for (int i = 0; i < Q1D; ++i)
        V[k] += B[k + i*P1D] * U[i]; // Contract z direction
  }
  for (int k = P1D; k < Q1D; ++k)
    V[k] = 0.0;
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract y
//------------------------------------------------------------------------------
inline __device__ void ContractTransposeY3d(CeedScalar *slice, const int t_idx,
                                            const int t_idy, const int t_idz,
                                            const CeedScalar *U,
                                            const CeedScalar *B,
                                            CeedScalar *V) {
  for (int k = 0; k < P1D; ++k) {
    slice[t_idx + t_idy*T1D + t_idz*T1D*T1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (t_idx < Q1D && t_idy < P1D)
      for (int i = 0; i < Q1D; ++i)
        V[k] += B[t_idy + i*P1D] * slice[t_idx + i*T1D + t_idz*T1D*T1D]; // Contract y direction
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract x
//------------------------------------------------------------------------------
inline __device__ void ContractTransposeX3d(CeedScalar *slice, const int t_idx,
                                            const int t_idy, const int t_idz,
                                            const CeedScalar *U,
                                            const CeedScalar *B,
                                            CeedScalar *V) {
  for (int k = 0; k < P1D; ++k) {
    slice[t_idx + t_idy*T1D + t_idz*T1D*T1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (t_idx < P1D && t_idy < P1D)
      for (int i = 0; i < Q1D; ++i)
        V[k] += B[t_idx + i*P1D] * slice[i + t_idy*T1D + t_idz*T1D*T1D]; // Contract x direction
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D interpolate to quadrature points
//------------------------------------------------------------------------------
inline __device__ void interp3d(const CeedInt num_elem, const int transpose,
                                const CeedScalar *s_B,
                                const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V,
                                CeedScalar *slice) {
  CeedScalar r_V[T1D];
  CeedScalar r_t[T1D];

  const int t_idx = threadIdx.x;
  const int t_idy = threadIdx.y;
  const int t_idz = threadIdx.z;
  const int blockElem = t_idz/BASIS_NCOMP;
  const int elemsPerBlock = blockDim.z/BASIS_NCOMP;
  const int comp = t_idz%BASIS_NCOMP;

  for (CeedInt elem = blockIdx.x*elemsPerBlock + blockElem; elem < num_elem;
       elem += gridDim.x*elemsPerBlock) {
    for (int i = 0; i < T1D; ++i) {
      r_V[i] = 0.0;
      r_t[i] = 0.0;
    }
    if (!transpose) {
      readDofs3d(elem, t_idx, t_idy, comp, num_elem, d_U, r_V);
      ContractX3d(slice, t_idx, t_idy, t_idz, r_V, s_B, r_t);
      ContractY3d(slice, t_idx, t_idy, t_idz, r_t, s_B, r_V);
      ContractZ3d(slice, t_idx, t_idy, t_idz, r_V, s_B, r_t);
      writeQuads3d(elem, t_idx, t_idy, comp, 0, num_elem, r_t, d_V);
    } else {
      readQuads3d(elem, t_idx, t_idy, comp, 0, num_elem, d_U, r_V);
      ContractTransposeZ3d(slice, t_idx, t_idy, t_idz, r_V, s_B, r_t);
      ContractTransposeY3d(slice, t_idx, t_idy, t_idz, r_t, s_B, r_V);
      ContractTransposeX3d(slice, t_idx, t_idy, t_idz, r_V, s_B, r_t);
      writeDofs3d(elem, t_idx, t_idy, comp, num_elem, r_t, d_V);
    }
  }
}

//------------------------------------------------------------------------------
// 3D derivatives at quadrature points
//------------------------------------------------------------------------------
inline __device__ void grad3d(const CeedInt num_elem, const int transpose,
                              const CeedScalar *s_B, const CeedScalar *s_G,
                              const CeedScalar *__restrict__ d_U,
                              CeedScalar *__restrict__ d_V,
                              CeedScalar *slice) {
  // Use P1D for one of these
  CeedScalar r_U[T1D];
  CeedScalar r_V[T1D];
  CeedScalar r_t[T1D];

  const int t_idx = threadIdx.x;
  const int t_idy = threadIdx.y;
  const int t_idz = threadIdx.z;
  const int blockElem = t_idz/BASIS_NCOMP;
  const int elemsPerBlock = blockDim.z/BASIS_NCOMP;
  const int comp = t_idz%BASIS_NCOMP;
  int dim;

  for (CeedInt elem = blockIdx.x*elemsPerBlock + blockElem; elem < num_elem;
       elem += gridDim.x*elemsPerBlock) {
    for (int i = 0; i < T1D; ++i) {
      r_U[i] = 0.0;
      r_V[i] = 0.0;
      r_t[i] = 0.0;
    }
    if (!transpose) {
      readDofs3d(elem, t_idx, t_idy, comp, num_elem, d_U, r_U);
      ContractX3d(slice, t_idx, t_idy, t_idz, r_U, s_G, r_V);
      ContractY3d(slice, t_idx, t_idy, t_idz, r_V, s_B, r_t);
      ContractZ3d(slice, t_idx, t_idy, t_idz, r_t, s_B, r_V);
      dim = 0;
      writeQuads3d(elem, t_idx, t_idy, comp, dim, num_elem, r_V, d_V);
      ContractX3d(slice, t_idx, t_idy, t_idz, r_U, s_B, r_V);
      ContractY3d(slice, t_idx, t_idy, t_idz, r_V, s_G, r_t);
      ContractZ3d(slice, t_idx, t_idy, t_idz, r_t, s_B, r_V);
      dim = 1;
      writeQuads3d(elem, t_idx, t_idy, comp, dim, num_elem, r_V, d_V);
      ContractX3d(slice, t_idx, t_idy, t_idz, r_U, s_B, r_V);
      ContractY3d(slice, t_idx, t_idy, t_idz, r_V, s_B, r_t);
      ContractZ3d(slice, t_idx, t_idy, t_idz, r_t, s_G, r_V);
      dim = 2;
      writeQuads3d(elem, t_idx, t_idy, comp, dim, num_elem, r_V, d_V);
    } else {
      dim = 0;
      readQuads3d(elem, t_idx, t_idy, comp, dim, num_elem, d_U, r_U);
      ContractTransposeZ3d(slice, t_idx, t_idy, t_idz, r_U, s_B, r_t);
      ContractTransposeY3d(slice, t_idx, t_idy, t_idz, r_t, s_B, r_U);
      ContractTransposeX3d(slice, t_idx, t_idy, t_idz, r_U, s_G, r_V);
      dim = 1;
      readQuads3d(elem, t_idx, t_idy, comp, dim, num_elem, d_U, r_U);
      ContractTransposeZ3d(slice, t_idx, t_idy, t_idz, r_U, s_B, r_t);
      ContractTransposeY3d(slice, t_idx, t_idy, t_idz, r_t, s_G, r_U);
      ContractTransposeX3d(slice, t_idx, t_idy, t_idz, r_U, s_B, r_t);
      add(r_V, r_t);
      dim = 2;
      readQuads3d(elem, t_idx, t_idy, comp, dim, num_elem, d_U, r_U);
      ContractTransposeZ3d(slice, t_idx, t_idy, t_idz, r_U, s_G, r_t);
      ContractTransposeY3d(slice, t_idx, t_idy, t_idz, r_t, s_B, r_U);
      ContractTransposeX3d(slice, t_idx, t_idy, t_idz, r_U, s_B, r_t);
      add(r_V, r_t);
      writeDofs3d(elem, t_idx, t_idy, comp, num_elem, r_V, d_V);
    }
  }
}

//------------------------------------------------------------------------------
// 3D quadrature weights
//------------------------------------------------------------------------------
__device__ void weight3d(const CeedInt num_elem, const CeedScalar *q_weight_1d,
                         CeedScalar *w) {
  const int i = threadIdx.x;
  const int j = threadIdx.y;
  const int k = threadIdx.z;
  const CeedScalar weight = q_weight_1d[i]*q_weight_1d[j]*q_weight_1d[k];
  for (int e = blockIdx.x; e < num_elem; e += gridDim.x) {
    const int ind = e*Q1D*Q1D*Q1D + i + j*Q1D + k*Q1D*Q1D;
    w[ind] = weight;
  }
}

//------------------------------------------------------------------------------
// Basis kernels
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Interp kernel by dim
//------------------------------------------------------------------------------
extern "C" __launch_bounds__(INTERP_BLKSIZE) __global__ void Interp(
                                  const CeedInt num_elem, const int transpose,
                                  CeedScalar *d_interp_1d,
                                  const CeedScalar *__restrict__ d_U,
                                  CeedScalar *__restrict__ d_V) {

  HIP_DYNAMIC_SHARED( CeedScalar, slice)
  // load interp_1d into shared memory
  __shared__ CeedScalar s_B[P1D*Q1D];
  loadMatrix(d_interp_1d, s_B); 
  __syncthreads();

  if (BASIS_DIM == 1) {
    interp_1d(num_elem, transpose, s_B, d_U, d_V, slice);
  } else if (BASIS_DIM == 2) {
    interp2d(num_elem, transpose, s_B, d_U, d_V, slice);
  } else if (BASIS_DIM == 3) {
    interp3d(num_elem, transpose, s_B, d_U, d_V, slice);
  }
}

//------------------------------------------------------------------------------
// Grad kernel by dim
//------------------------------------------------------------------------------
extern "C" __launch_bounds__(GRAD_BLKSIZE) __global__ void Grad(const CeedInt num_elem,
                                const int transpose,
                                CeedScalar *d_interp_1d, CeedScalar *d_grad_1d,
                                const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V) {
  HIP_DYNAMIC_SHARED( CeedScalar, slice)
  // load interp_1d and grad_1d into shared memory
  __shared__ CeedScalar s_B[P1D*Q1D];
  loadMatrix(d_interp_1d, s_B); 
  __shared__ CeedScalar s_G[P1D*Q1D];
  loadMatrix(d_grad_1d, s_G); 
  __syncthreads();

  if (BASIS_DIM == 1) {
    grad_1d(num_elem, transpose, s_B, s_G, d_U, d_V, slice);
  } else if (BASIS_DIM == 2) {
    grad2d(num_elem, transpose, s_B, s_G, d_U, d_V, slice);
  } else if (BASIS_DIM == 3) {
    grad3d(num_elem, transpose, s_B, s_G, d_U, d_V, slice);
  }
}

//------------------------------------------------------------------------------
// Weight kernels by dim
//------------------------------------------------------------------------------
extern "C" __launch_bounds__(WEIGHT_BLKSIZE) __global__ void Weight(const CeedInt num_elem,
                                  const CeedScalar *__restrict__ q_weight_1d,
                                  CeedScalar *__restrict__ v) {
  if (BASIS_DIM == 1) {
    weight_1d(num_elem, q_weight_1d, v);
  } else if (BASIS_DIM == 2) {
    weight2d(num_elem, q_weight_1d, v);
  } else if (BASIS_DIM == 3) {
    weight3d(num_elem, q_weight_1d, v);
  }
}

//------------------------------------------------------------------------------
