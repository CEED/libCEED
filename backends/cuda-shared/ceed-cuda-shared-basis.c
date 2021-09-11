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
#include <cuda.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include "ceed-cuda-shared.h"
#include "../cuda/ceed-cuda.h"

//------------------------------------------------------------------------------
// Shared mem kernels
//------------------------------------------------------------------------------
// *INDENT-OFF*
static const char *kernelsShared = QUOTE(

//------------------------------------------------------------------------------
// Sum input into output
//------------------------------------------------------------------------------
inline __device__ void add(CeedScalar *r_V, const CeedScalar *r_U) {
  for (int i = 0; i < P1D; i++)
    r_V[i] += r_U[i];
}

//------------------------------------------------------------------------------
// 1D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Read DoFs
//------------------------------------------------------------------------------
inline __device__ void readDofs1d(const int elem, const int tidx,
                                  const int tidy, const int tidz,const int comp,
                                  const int nelem, const CeedScalar *d_U,
                                  CeedScalar *slice) {
  for (int i = 0; i < P1D; i++)
    slice[i + tidz*T1D] = d_U[i + elem*P1D + comp*P1D*nelem];
  for (int i = P1D; i < Q1D; i++)
    slice[i + tidz*T1D] = 0.0;
}

//------------------------------------------------------------------------------
// Write DoFs
//------------------------------------------------------------------------------
inline __device__ void writeDofs1d(const int elem, const int tidx,
                                   const int tidy, const int comp,
                                   const int nelem, const CeedScalar &r_V,
                                   CeedScalar *d_V) {
  if (tidx<P1D)
    d_V[tidx + elem*P1D + comp*P1D*nelem] = r_V;
}

//------------------------------------------------------------------------------
// Read quadrature point data
//------------------------------------------------------------------------------
inline __device__ void readQuads1d(const int elem, const int tidx,
                                   const int tidy, const int tidz, const int comp,
                                   const int dim, const int nelem,
                                   const CeedScalar *d_U, CeedScalar *slice) {
  for (int i = 0; i < Q1D; i++)
    slice[i + tidz*T1D] = d_U[i + elem*Q1D + comp*Q1D*nelem +
                            dim*BASIS_NCOMP*nelem*Q1D];
  for (int i = Q1D; i < P1D; i++)
    slice[i + tidz*T1D] = 0.0;
}

//------------------------------------------------------------------------------
// Write quadrature point data
//------------------------------------------------------------------------------
inline __device__ void writeQuads1d(const int elem, const int tidx,
                                    const int tidy, const int comp,
                                    const int dim, const int nelem,
                                    const CeedScalar &r_V, CeedScalar *d_V) {
  if (tidx<Q1D)
    d_V[tidx + elem*Q1D + comp*Q1D*nelem + dim*BASIS_NCOMP*nelem*Q1D] = r_V;
}

//------------------------------------------------------------------------------
// 1D tensor contraction
//------------------------------------------------------------------------------
inline __device__ void ContractX1d(CeedScalar *slice, const int tidx,
                                   const int tidy, const int tidz,
                                   const CeedScalar &U, const CeedScalar *B,
                                   CeedScalar &V) {
  V = 0.0;
  for (int i = 0; i < P1D; ++i)
    V += B[i + tidx*P1D] * slice[i + tidz*T1D]; // Contract x direction
}

//------------------------------------------------------------------------------
// 1D transpose tensor contraction
//------------------------------------------------------------------------------
inline __device__ void ContractTransposeX1d(CeedScalar *slice, const int tidx,
    const int tidy, const int tidz,
    const CeedScalar &U, const CeedScalar *B, CeedScalar &V) {
  V = 0.0;
  for (int i = 0; i < Q1D; ++i)
    V += B[tidx + i*P1D] * slice[i + tidz*T1D]; // Contract x direction
}

//------------------------------------------------------------------------------
// 1D interpolate to quadrature points
//------------------------------------------------------------------------------
inline __device__ void interp1d(const CeedInt nelem, const int transpose,
                                const CeedScalar *c_B,
                                const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V,
                                CeedScalar *slice) {
  CeedScalar r_V;
  CeedScalar r_t;

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int tidz = threadIdx.z;

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < nelem;
       elem += gridDim.x*blockDim.z) {
    for (int comp = 0; comp < BASIS_NCOMP; comp++) {
      if (!transpose) {
        readDofs1d(elem, tidx, tidy, tidz, comp, nelem, d_U, slice);
        ContractX1d(slice, tidx, tidy, tidz, r_t, c_B, r_V);
        writeQuads1d(elem, tidx, tidy, comp, 0, nelem, r_V, d_V);
      } else {
        readQuads1d(elem, tidx, tidy, tidz, comp, 0, nelem, d_U, slice);
        ContractTransposeX1d(slice, tidx, tidy, tidz, r_t, c_B, r_V);
        writeDofs1d(elem, tidx, tidy, comp, nelem, r_V, d_V);
      }
    }
  }
}

//------------------------------------------------------------------------------
// 1D derivatives at quadrature points
//------------------------------------------------------------------------------
inline __device__ void grad1d(const CeedInt nelem, const int transpose,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              const CeedScalar *__restrict__ d_U,
                              CeedScalar *__restrict__ d_V,
                              CeedScalar *slice) {
  CeedScalar r_U;
  CeedScalar r_V;

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int tidz = threadIdx.z;
  int dim;

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < nelem;
       elem += gridDim.x*blockDim.z) {
    for(int comp = 0; comp < BASIS_NCOMP; comp++) {
      if (!transpose) {
        readDofs1d(elem, tidx, tidy, tidz, comp, nelem, d_U, slice);
        ContractX1d(slice, tidx, tidy, tidz, r_U, c_G, r_V);
        dim = 0;
        writeQuads1d(elem, tidx, tidy, comp, dim, nelem, r_V, d_V);
      } else {
        dim = 0;
        readQuads1d(elem, tidx, tidy, tidz, comp, dim, nelem, d_U, slice);
        ContractTransposeX1d(slice, tidx, tidy, tidz, r_U, c_G, r_V);
        writeDofs1d(elem, tidx, tidy, comp, nelem, r_V, d_V);
      }
    }
  }
}

//------------------------------------------------------------------------------
// 1D Quadrature weights
//------------------------------------------------------------------------------
__device__ void weight1d(const CeedInt nelem, const CeedScalar *qweight1d,
                         CeedScalar *w) {
  const int tid = threadIdx.x;
  const CeedScalar weight = qweight1d[tid];
  for (CeedInt elem = blockIdx.x*blockDim.y + threadIdx.y; elem < nelem;
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
inline __device__ void readDofs2d(const int elem, const int tidx,
                                  const int tidy, const int comp,
                                  const int nelem, const CeedScalar *d_U,
                                  CeedScalar &U) {
  U = (tidx<P1D && tidy<P1D) ?
      d_U[tidx + tidy*P1D + elem*P1D*P1D + comp*P1D*P1D*nelem] : 0.0;
}

//------------------------------------------------------------------------------
// Write DoFs
//------------------------------------------------------------------------------
inline __device__ void writeDofs2d(const int elem, const int tidx,
                                   const int tidy, const int comp,
                                   const int nelem, const CeedScalar &r_V,
                                   CeedScalar *d_V) {
  if (tidx<P1D && tidy<P1D)
    d_V[tidx + tidy*P1D + elem*P1D*P1D + comp*P1D*P1D*nelem] = r_V;
}

//------------------------------------------------------------------------------
// Read quadrature point data
//------------------------------------------------------------------------------
inline __device__ void readQuads2d(const int elem, const int tidx,
                                   const int tidy, const int comp,
                                   const int dim, const int nelem,
                                   const CeedScalar *d_U, CeedScalar &U ) {
  U = (tidx<Q1D && tidy<Q1D) ?
      d_U[tidx + tidy*Q1D + elem*Q1D*Q1D + comp*Q1D*Q1D*nelem +
      dim*BASIS_NCOMP*nelem*Q1D*Q1D] : 0.0;
}

//------------------------------------------------------------------------------
// Write quadrature point data
//------------------------------------------------------------------------------
inline __device__ void writeQuads2d(const int elem, const int tidx,
                                    const int tidy, const int comp,
                                    const int dim, const int nelem,
                                    const CeedScalar &r_V, CeedScalar *d_V) {
  if (tidx<Q1D && tidy<Q1D)
    d_V[tidx + tidy*Q1D + elem*Q1D*Q1D + comp*Q1D*Q1D*nelem +
    dim*BASIS_NCOMP*nelem*Q1D*Q1D] = r_V;
}

//------------------------------------------------------------------------------
// 2D tensor contraction x
//------------------------------------------------------------------------------
inline __device__ void ContractX2d(CeedScalar *slice, const int tidx,
                                   const int tidy, const int tidz,
                                   const CeedScalar &U, const CeedScalar *B,
                                   CeedScalar &V) {
  slice[tidx + tidy*T1D + tidz*T1D*T1D] = U;
  __syncthreads();
  V = 0.0;
  if (tidx < Q1D)
    for (int i = 0; i < P1D; ++i)
      V += B[i + tidx*P1D] * slice[i + tidy*T1D + tidz*T1D*T1D]; // Contract x direction
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D tensor contraction y
//------------------------------------------------------------------------------
inline __device__ void ContractY2d(CeedScalar *slice, const int tidx,
                                   const int tidy, const int tidz,
                                   const CeedScalar &U, const CeedScalar *B,
                                   CeedScalar &V) {
  slice[tidx + tidy*T1D + tidz*T1D*T1D] = U;
  __syncthreads();
  V = 0.0;
  if (tidy < Q1D)
    for (int i = 0; i < P1D; ++i)
      V += B[i + tidy*P1D] * slice[tidx + i*T1D + tidz*T1D*T1D]; // Contract y direction
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D transpose tensor contraction y
//------------------------------------------------------------------------------
inline __device__ void ContractTransposeY2d(CeedScalar *slice, const int tidx,
    const int tidy, const int tidz,
    const CeedScalar &U, const CeedScalar *B, CeedScalar &V) {
  slice[tidx + tidy*T1D + tidz*T1D*T1D] = U;
  __syncthreads();
  V = 0.0;
  if (tidy < P1D)
    for (int i = 0; i < Q1D; ++i)
      V += B[tidy + i*P1D] * slice[tidx + i*T1D + tidz*T1D*T1D]; // Contract y direction
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D transpose tensor contraction x
//------------------------------------------------------------------------------
inline __device__ void ContractTransposeX2d(CeedScalar *slice, const int tidx,
    const int tidy, const int tidz,
    const CeedScalar &U, const CeedScalar *B, CeedScalar &V) {
  slice[tidx + tidy*T1D + tidz*T1D*T1D] = U;
  __syncthreads();
  V = 0.0;
  if (tidx < P1D)
    for (int i = 0; i < Q1D; ++i)
      V += B[tidx + i*P1D] * slice[i + tidy*T1D + tidz*T1D*T1D]; // Contract x direction
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D interpolate to quadrature points
//------------------------------------------------------------------------------
inline __device__ void interp2d(const CeedInt nelem, const int transpose,
                                const CeedScalar *c_B,
                                const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V,
                                CeedScalar *slice) {
  CeedScalar r_V;
  CeedScalar r_t;

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int tidz = threadIdx.z;
  const int blockElem = tidz/BASIS_NCOMP;
  const int elemsPerBlock = blockDim.z/BASIS_NCOMP;
  const int comp = tidz%BASIS_NCOMP;

  for (CeedInt elem = blockIdx.x*elemsPerBlock + blockElem; elem < nelem;
       elem += gridDim.x*elemsPerBlock) {
    const int comp = tidz%BASIS_NCOMP;
    r_V = 0.0;
    r_t = 0.0;
    if (!transpose) {
      readDofs2d(elem, tidx, tidy, comp, nelem, d_U, r_V);
      ContractX2d(slice, tidx, tidy, tidz, r_V, c_B, r_t);
      ContractY2d(slice, tidx, tidy, tidz, r_t, c_B, r_V);
      writeQuads2d(elem, tidx, tidy, comp, 0, nelem, r_V, d_V);
    } else {
      readQuads2d(elem, tidx, tidy, comp, 0, nelem, d_U, r_V);
      ContractTransposeY2d(slice, tidx, tidy, tidz, r_V, c_B, r_t);
      ContractTransposeX2d(slice, tidx, tidy, tidz, r_t, c_B, r_V);
      writeDofs2d(elem, tidx, tidy, comp, nelem, r_V, d_V);
    }
  }
}

//------------------------------------------------------------------------------
// 2D derivatives at quadrature points
//------------------------------------------------------------------------------
inline __device__ void grad2d(const CeedInt nelem, const int transpose,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              const CeedScalar *__restrict__ d_U,
                              CeedScalar *__restrict__ d_V, CeedScalar *slice) {
  CeedScalar r_U;
  CeedScalar r_V;
  CeedScalar r_t;

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int tidz = threadIdx.z;
  const int blockElem = tidz/BASIS_NCOMP;
  const int elemsPerBlock = blockDim.z/BASIS_NCOMP;
  const int comp = tidz%BASIS_NCOMP;
  int dim;

  for (CeedInt elem = blockIdx.x*elemsPerBlock + blockElem; elem < nelem;
       elem += gridDim.x*elemsPerBlock) {
    if (!transpose) {
      readDofs2d(elem, tidx, tidy, comp, nelem, d_U, r_U);
      ContractX2d(slice, tidx, tidy, tidz, r_U, c_G, r_t);
      ContractY2d(slice, tidx, tidy, tidz, r_t, c_B, r_V);
      dim = 0;
      writeQuads2d(elem, tidx, tidy, comp, dim, nelem, r_V, d_V);
      ContractX2d(slice, tidx, tidy, tidz, r_U, c_B, r_t);
      ContractY2d(slice, tidx, tidy, tidz, r_t, c_G, r_V);
      dim = 1;
      writeQuads2d(elem, tidx, tidy, comp, dim, nelem, r_V, d_V);
    } else {
      dim = 0;
      readQuads2d(elem, tidx, tidy, comp, dim, nelem, d_U, r_U);
      ContractTransposeY2d(slice, tidx, tidy, tidz, r_U, c_B, r_t);
      ContractTransposeX2d(slice, tidx, tidy, tidz, r_t, c_G, r_V);
      dim = 1;
      readQuads2d(elem, tidx, tidy, comp, dim, nelem, d_U, r_U);
      ContractTransposeY2d(slice, tidx, tidy, tidz, r_U, c_G, r_t);
      ContractTransposeX2d(slice, tidx, tidy, tidz, r_t, c_B, r_U);
      r_V += r_U;
      writeDofs2d(elem, tidx, tidy, comp, nelem, r_V, d_V);
    }
  }
}

//------------------------------------------------------------------------------
// 2D quadrature weights
//------------------------------------------------------------------------------
__device__ void weight2d(const CeedInt nelem, const CeedScalar *qweight1d,
                         CeedScalar *w) {
  const int i = threadIdx.x;
  const int j = threadIdx.y;
  const CeedScalar weight = qweight1d[i]*qweight1d[j];
  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < nelem;
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
inline __device__ void readDofs3d(const int elem, const int tidx,
                                  const int tidy, const int comp,
                                  const int nelem, const CeedScalar *d_U,
                                  CeedScalar *r_U) {
  for (int i = 0; i < P1D; i++)
    r_U[i] = (tidx < P1D && tidy < P1D) ?
              d_U[tidx + tidy*P1D + i*P1D*P1D + elem*P1D*P1D*P1D +
                  comp*P1D*P1D*P1D*nelem] : 0.0;
  for (int i = P1D; i < Q1D; i++)
    r_U[i] = 0.0;
}

//------------------------------------------------------------------------------
// Write DoFs
//------------------------------------------------------------------------------
inline __device__ void writeDofs3d(const int elem, const int tidx,
                                   const int tidy, const int comp,
                                   const int nelem, const CeedScalar *r_V,
                                   CeedScalar *d_V) {
  if (tidx < P1D && tidy < P1D) {
    for (int i = 0; i < P1D; i++)
      d_V[tidx + tidy*P1D + i*P1D*P1D + elem*P1D*P1D*P1D +
          comp*P1D*P1D*P1D*nelem] = r_V[i];
  }
}

//------------------------------------------------------------------------------
// Read quadrature point data
//------------------------------------------------------------------------------
inline __device__ void readQuads3d(const int elem, const int tidx,
                                   const int tidy, const int comp,
                                   const int dim, const int nelem,
                                   const CeedScalar *d_U, CeedScalar *r_U) {
  for (int i = 0; i < Q1D; i++)
    r_U[i] = (tidx < Q1D && tidy < Q1D) ? 
              d_U[tidx + tidy*Q1D + i*Q1D*Q1D + elem*Q1D*Q1D*Q1D +
              comp*Q1D*Q1D*Q1D*nelem + dim*BASIS_NCOMP*nelem*Q1D*Q1D*Q1D] : 0.0;
  for (int i = Q1D; i < P1D; i++)
    r_U[i] = 0.0;
}

//------------------------------------------------------------------------------
// Write quadrature point data
//------------------------------------------------------------------------------
inline __device__ void writeQuads3d(const int elem, const int tidx,
                                    const int tidy, const int comp,
                                    const int dim, const int nelem,
                                    const CeedScalar *r_V, CeedScalar *d_V) {
  if (tidx < Q1D && tidy < Q1D) {
    for (int i = 0; i < Q1D; i++)
      d_V[tidx + tidy*Q1D + i*Q1D*Q1D + elem*Q1D*Q1D*Q1D + comp*Q1D*Q1D*Q1D*nelem +
          dim*BASIS_NCOMP*nelem*Q1D*Q1D*Q1D] = r_V[i];
  }
}

//------------------------------------------------------------------------------
// 3D tensor contract x
//------------------------------------------------------------------------------
inline __device__ void ContractX3d(CeedScalar *slice, const int tidx,
                                   const int tidy, const int tidz,
                                   const CeedScalar *U,
                                   const CeedScalar *B,
                                   CeedScalar *V) {
  for (int k = 0; k < P1D; ++k) {
    slice[tidx + tidy*T1D + tidz*T1D*T1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (tidx < Q1D && tidy < P1D)
      for (int i = 0; i < P1D; ++i)
        V[k] += B[i + tidx*P1D] * slice[i + tidy*T1D + tidz*T1D*T1D]; // Contract x direction
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D tensor contract y
//------------------------------------------------------------------------------
inline __device__ void ContractY3d(CeedScalar *slice, const int tidx,
                                   const int tidy, const int tidz,
                                   const CeedScalar *U,
                                   const CeedScalar *B,
                                   CeedScalar *V) {
  for (int k = 0; k < P1D; ++k) {
    slice[tidx + tidy*T1D + tidz*T1D*T1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (tidx < Q1D && tidy < Q1D)
      for (int i = 0; i < P1D; ++i)
        V[k] += B[i + tidy*P1D] * slice[tidx + i*T1D + tidz*T1D*T1D]; // Contract y direction
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D tensor contract z
//------------------------------------------------------------------------------
inline __device__ void ContractZ3d(CeedScalar *slice, const int tidx,
                                   const int tidy, const int tidz,
                                   const CeedScalar *U,
                                   const CeedScalar *B,
                                   CeedScalar *V) {
  for (int k = 0; k < Q1D; ++k) {
    V[k] = 0.0;
    if (tidx < Q1D && tidy < Q1D)
      for (int i = 0; i < P1D; ++i)
        V[k] += B[i + k*P1D] * U[i]; // Contract z direction
  }
  for (int k = Q1D; k < P1D; ++k)
    V[k] = 0.0;
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract z
//------------------------------------------------------------------------------
inline __device__ void ContractTransposeZ3d(CeedScalar *slice, const int tidx,
                                            const int tidy, const int tidz,
                                            const CeedScalar *U,
                                            const CeedScalar *B,
                                            CeedScalar *V) {
  for (int k = 0; k < P1D; ++k) {
    V[k] = 0.0;
    if (tidx < Q1D && tidy < Q1D)
      for (int i = 0; i < Q1D; ++i)
        V[k] += B[k + i*P1D] * U[i]; // Contract z direction
  }
  for (int k = P1D; k < Q1D; ++k)
    V[k] = 0.0;
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract y
//------------------------------------------------------------------------------
inline __device__ void ContractTransposeY3d(CeedScalar *slice, const int tidx,
                                            const int tidy, const int tidz,
                                            const CeedScalar *U,
                                            const CeedScalar *B,
                                            CeedScalar *V) {
  for (int k = 0; k < P1D; ++k) {
    slice[tidx + tidy*T1D + tidz*T1D*T1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (tidx < Q1D && tidy < P1D)
      for (int i = 0; i < Q1D; ++i)
        V[k] += B[tidy + i*P1D] * slice[tidx + i*T1D + tidz*T1D*T1D]; // Contract y direction
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract x
//------------------------------------------------------------------------------
inline __device__ void ContractTransposeX3d(CeedScalar *slice, const int tidx,
                                            const int tidy, const int tidz,
                                            const CeedScalar *U,
                                            const CeedScalar *B,
                                            CeedScalar *V) {
  for (int k = 0; k < P1D; ++k) {
    slice[tidx + tidy*T1D + tidz*T1D*T1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (tidx < P1D && tidy < P1D)
      for (int i = 0; i < Q1D; ++i)
        V[k] += B[tidx + i*P1D] * slice[i + tidy*T1D + tidz*T1D*T1D]; // Contract x direction
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D interpolate to quadrature points
//------------------------------------------------------------------------------
inline __device__ void interp3d(const CeedInt nelem, const int transpose,
                                const CeedScalar *c_B,
                                const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V,
                                CeedScalar *slice) {
  CeedScalar r_V[T1D];
  CeedScalar r_t[T1D];

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int tidz = threadIdx.z;
  const int blockElem = tidz/BASIS_NCOMP;
  const int elemsPerBlock = blockDim.z/BASIS_NCOMP;
  const int comp = tidz%BASIS_NCOMP;

  for (CeedInt elem = blockIdx.x*elemsPerBlock + blockElem; elem < nelem;
       elem += gridDim.x*elemsPerBlock) {
    for (int i = 0; i < T1D; ++i) {
      r_V[i] = 0.0;
      r_t[i] = 0.0;
    }
    if (!transpose) {
      readDofs3d(elem, tidx, tidy, comp, nelem, d_U, r_V);
      ContractX3d(slice, tidx, tidy, tidz, r_V, c_B, r_t);
      ContractY3d(slice, tidx, tidy, tidz, r_t, c_B, r_V);
      ContractZ3d(slice, tidx, tidy, tidz, r_V, c_B, r_t);
      writeQuads3d(elem, tidx, tidy, comp, 0, nelem, r_t, d_V);
    } else {
      readQuads3d(elem, tidx, tidy, comp, 0, nelem, d_U, r_V);
      ContractTransposeZ3d(slice, tidx, tidy, tidz, r_V, c_B, r_t);
      ContractTransposeY3d(slice, tidx, tidy, tidz, r_t, c_B, r_V);
      ContractTransposeX3d(slice, tidx, tidy, tidz, r_V, c_B, r_t);
      writeDofs3d(elem, tidx, tidy, comp, nelem, r_t, d_V);
    }
  }
}

//------------------------------------------------------------------------------
// 3D derivatives at quadrature points
//------------------------------------------------------------------------------
inline __device__ void grad3d(const CeedInt nelem, const int transpose,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              const CeedScalar *__restrict__ d_U,
                              CeedScalar *__restrict__ d_V,
                              CeedScalar *slice) {
  // Use P1D for one of these
  CeedScalar r_U[T1D];
  CeedScalar r_V[T1D];
  CeedScalar r_t[T1D];

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int tidz = threadIdx.z;
  const int blockElem = tidz/BASIS_NCOMP;
  const int elemsPerBlock = blockDim.z/BASIS_NCOMP;
  const int comp = tidz%BASIS_NCOMP;
  int dim;

  for (CeedInt elem = blockIdx.x*elemsPerBlock + blockElem; elem < nelem;
       elem += gridDim.x*elemsPerBlock) {
    for (int i = 0; i < T1D; ++i) {
      r_U[i] = 0.0;
      r_V[i] = 0.0;
      r_t[i] = 0.0;
    }
    if (!transpose) {
      readDofs3d(elem, tidx, tidy, comp, nelem, d_U, r_U);
      ContractX3d(slice, tidx, tidy, tidz, r_U, c_G, r_V);
      ContractY3d(slice, tidx, tidy, tidz, r_V, c_B, r_t);
      ContractZ3d(slice, tidx, tidy, tidz, r_t, c_B, r_V);
      dim = 0;
      writeQuads3d(elem, tidx, tidy, comp, dim, nelem, r_V, d_V);
      ContractX3d(slice, tidx, tidy, tidz, r_U, c_B, r_V);
      ContractY3d(slice, tidx, tidy, tidz, r_V, c_G, r_t);
      ContractZ3d(slice, tidx, tidy, tidz, r_t, c_B, r_V);
      dim = 1;
      writeQuads3d(elem, tidx, tidy, comp, dim, nelem, r_V, d_V);
      ContractX3d(slice, tidx, tidy, tidz, r_U, c_B, r_V);
      ContractY3d(slice, tidx, tidy, tidz, r_V, c_B, r_t);
      ContractZ3d(slice, tidx, tidy, tidz, r_t, c_G, r_V);
      dim = 2;
      writeQuads3d(elem, tidx, tidy, comp, dim, nelem, r_V, d_V);
    } else {
      dim = 0;
      readQuads3d(elem, tidx, tidy, comp, dim, nelem, d_U, r_U);
      ContractTransposeZ3d(slice, tidx, tidy, tidz, r_U, c_B, r_t);
      ContractTransposeY3d(slice, tidx, tidy, tidz, r_t, c_B, r_U);
      ContractTransposeX3d(slice, tidx, tidy, tidz, r_U, c_G, r_V);
      dim = 1;
      readQuads3d(elem, tidx, tidy, comp, dim, nelem, d_U, r_U);
      ContractTransposeZ3d(slice, tidx, tidy, tidz, r_U, c_B, r_t);
      ContractTransposeY3d(slice, tidx, tidy, tidz, r_t, c_G, r_U);
      ContractTransposeX3d(slice, tidx, tidy, tidz, r_U, c_B, r_t);
      add(r_V, r_t);
      dim = 2;
      readQuads3d(elem, tidx, tidy, comp, dim, nelem, d_U, r_U);
      ContractTransposeZ3d(slice, tidx, tidy, tidz, r_U, c_G, r_t);
      ContractTransposeY3d(slice, tidx, tidy, tidz, r_t, c_B, r_U);
      ContractTransposeX3d(slice, tidx, tidy, tidz, r_U, c_B, r_t);
      add(r_V, r_t);
      writeDofs3d(elem, tidx, tidy, comp, nelem, r_V, d_V);
    }
  }
}

//------------------------------------------------------------------------------
// 3D quadrature weights
//------------------------------------------------------------------------------
__device__ void weight3d(const CeedInt nelem, const CeedScalar *qweight1d,
                         CeedScalar *w) {
  const int i = threadIdx.x;
  const int j = threadIdx.y;
  const int k = threadIdx.z;
  const CeedScalar weight = qweight1d[i]*qweight1d[j]*qweight1d[k];
  for (int e = blockIdx.x; e < nelem; e += gridDim.x) {
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
extern "C" __global__ void interp(const CeedInt nelem, const int transpose,
                                  const CeedScalar *c_B,
                                  const CeedScalar *__restrict__ d_U,
                                  CeedScalar *__restrict__ d_V) {
  extern __shared__ CeedScalar slice[];
  if (BASIS_DIM == 1) {
    interp1d(nelem, transpose, c_B, d_U, d_V, slice);
  } else if (BASIS_DIM == 2) {
    interp2d(nelem, transpose, c_B, d_U, d_V, slice);
  } else if (BASIS_DIM == 3) {
    interp3d(nelem, transpose, c_B, d_U, d_V, slice);
  }
}

//------------------------------------------------------------------------------
// Grad kernel by dim
//------------------------------------------------------------------------------
extern "C" __global__ void grad(const CeedInt nelem, const int transpose,
                                const CeedScalar *c_B, const CeedScalar *c_G,
                                const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V) {
  extern __shared__ CeedScalar slice[];
  if (BASIS_DIM == 1) {
    grad1d(nelem, transpose, c_B, c_G, d_U, d_V, slice);
  } else if (BASIS_DIM == 2) {
    grad2d(nelem, transpose, c_B, c_G, d_U, d_V, slice);
  } else if (BASIS_DIM == 3) {
    grad3d(nelem, transpose, c_B, c_G, d_U, d_V, slice);
  }
}

//------------------------------------------------------------------------------
// Weight kernels by dim
//------------------------------------------------------------------------------
extern "C" __global__ void weight(const CeedInt nelem,
                                  const CeedScalar *__restrict__ qweight1d,
                                  CeedScalar *__restrict__ v) {
  if (BASIS_DIM == 1) {
    weight1d(nelem, qweight1d, v);
  } else if (BASIS_DIM == 2) {
    weight2d(nelem, qweight1d, v);
  } else if (BASIS_DIM == 3) {
    weight3d(nelem, qweight1d, v);
  }
}

);
// *INDENT-ON*

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

  // Create float versions of basis data
  float *interp1d_dbl = (float*)(malloc(Q1d*P1d*sizeof(float)));
  for (CeedInt i = 0; i < Q1d * P1d; i++) {
    interp1d_dbl[i] = (float) interp1d[i];
  }
  float *grad1d_dbl = (float*)(malloc(Q1d*P1d*sizeof(float)));
  for (CeedInt i = 0; i < Q1d * P1d; i++) {
    grad1d_dbl[i] = (float) grad1d[i];
  }
  float *qweight1d_dbl = (float*)(malloc(Q1d*sizeof(float)));
  for (CeedInt i = 0; i < Q1d; i++) {
    qweight1d_dbl[i] = (float) qweight1d[i];
  }

  // Copy basis data to GPU
  const CeedInt qBytes = Q1d * sizeof(float);
  ierr = cudaMalloc((void **)&data->d_qweight1d, qBytes); CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(data->d_qweight1d, qweight1d_dbl, qBytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);

  const CeedInt iBytes = qBytes * P1d;
  ierr = cudaMalloc((void **)&data->d_interp1d, iBytes); CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(data->d_interp1d, interp1d_dbl, iBytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);

  ierr = cudaMalloc((void **)&data->d_grad1d, iBytes); CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(data->d_grad1d, grad1d_dbl, iBytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);
 
  free(interp1d_dbl);
  free(grad1d_dbl);
  free(qweight1d_dbl);

  // Compute collocated gradient and copy to GPU
  data->d_collograd1d = NULL;
  if (dim == 3 && Q1d >= P1d) {
    CeedScalar *collograd1d;
    ierr = CeedMalloc(Q1d*Q1d, &collograd1d); CeedChkBackend(ierr);
    ierr = CeedBasisGetCollocatedGrad(basis, collograd1d); CeedChkBackend(ierr);
    // Again, create float version to copy to GPU
    float *collograd1d_dbl = (float*)(malloc(Q1d*Q1d*sizeof(float)));
    for (CeedInt i = 0; i < Q1d * Q1d; i++) {
      collograd1d_dbl[i] = (float) collograd1d[i];
    }
    ierr = cudaMalloc((void **)&data->d_collograd1d, qBytes * Q1d);
    CeedChk_Cu(ceed, ierr);
    ierr = cudaMemcpy(data->d_collograd1d, collograd1d_dbl, qBytes * Q1d,
                      cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);
    ierr = CeedFree(&collograd1d); CeedChkBackend(ierr);
    free(collograd1d_dbl);
  }

  // Compile basis kernels
  CeedInt ncomp;
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChkBackend(ierr);
  ierr = CeedCompileCuda(ceed, kernelsShared, &data->module, 8,
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
