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

#include <ceed-impl.h>
#include "../include/ceed.h"
#include "ceed-cuda-reg.h"
#include "../cuda/ceed-cuda.h"

//*********************
// shared mem kernels
static const char *kernelsShared = QUOTE(

inline __device__ void add(CeedScalar *r_V, const CeedScalar *r_U) {
  for (int i = 0; i < Q1D; i++)
    r_V[i] += r_U[i];
}

  //////////
 //  1D  //
//////////

inline __device__ void readDofs1d(const int elem, const int tidx, const int tidy, const int comp,
                                  const int nelem, const CeedScalar *d_U, CeedScalar *slice) {
  for (int i = 0; i < P1D; i++)
      slice[i] = d_U[i + comp*P1D + elem*BASIS_NCOMP*P1D];
  for (int i = P1D; i < Q1D; i++)
      slice[i] = 0.0;
}

inline __device__ void writeDofs1d(const int elem, const int tidx, const int tidy, const int comp,
                                   const int nelem, const CeedScalar &r_V, CeedScalar *d_V) {
  if (tidx<P1D)
  {
      d_V[tidx + comp*P1D + elem*BASIS_NCOMP*P1D] = r_V;
  }
}

inline __device__ void readQuads1d(const int elem, const int tidx, const int tidy, const int comp,
                                   const int dim, const int nelem, const CeedScalar *d_U, CeedScalar *slice) {
  for (int i = 0; i < Q1D; i++)
      slice[i] = d_U[i + elem*Q1D + comp*Q1D*nelem + dim*BASIS_NCOMP*nelem*Q1D];
}

inline __device__ void writeQuads1d(const int elem, const int tidx, const int tidy, const int comp,
                                  const int dim, const int nelem, const CeedScalar &r_V, CeedScalar *d_V) {
    d_V[tidx + elem*Q1D + comp*Q1D*nelem + dim*BASIS_NCOMP*nelem*Q1D] = r_V;
}

inline __device__ void ContractX1d(CeedScalar *slice, const int tidx, const int tidy,
                                   const CeedScalar &U, const CeedScalar *B, CeedScalar &V) {
  V = 0.0;
  for (int i = 0; i < P1D; ++i)
  {
    V += B[i + tidx*P1D] * slice[i];//contract x direction
  }
}

inline __device__ void ContractTransposeX1d(CeedScalar *slice, const int tidx, const int tidy,
                                   const CeedScalar &U, const CeedScalar *B, CeedScalar &V) {
  V = 0.0;
  for (int i = 0; i < Q1D; ++i)
  {
    V += B[tidx + i*P1D] * slice[i];//contract x direction
  }
}

inline __device__ void interp1d(const CeedInt nelem, const int transpose,
                                const CeedScalar *c_B, const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V,
                                CeedScalar *slice) {
  CeedScalar r_V;
  CeedScalar r_t;

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;


  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < nelem; elem += gridDim.x*blockDim.z) {
    for(int comp=0; comp<BASIS_NCOMP; comp++) {
      if(!transpose) {
        readDofs1d(elem, tidx, tidy, comp, nelem, d_U, slice);
        ContractX1d(slice, tidx, tidy, r_t, c_B, r_V);
        writeQuads1d(elem, tidx, tidy, comp, 0, nelem, r_V, d_V);
      } else {
        readQuads1d(elem, tidx, tidy, comp, 0, nelem, d_U, slice);
        ContractTransposeX1d(slice, tidx, tidy, r_t, c_B, r_V);
        writeDofs1d(elem, tidx, tidy, comp, nelem, r_V, d_V);
      }
    }
  }
}

inline __device__ void grad1d(const CeedInt nelem, const int transpose,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V,
                              CeedScalar *slice) {
  CeedScalar r_U;
  CeedScalar r_V;

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;//=>this is really a nb of elements per block
  int dim;

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < nelem; elem += gridDim.x*blockDim.z) {
    for(int comp=0; comp<BASIS_NCOMP; comp++) {
      if(!transpose) {
        readDofs1d(elem, tidx, tidy, comp, nelem, d_U, slice);
        ContractX1d(slice, tidx, tidy, r_U, c_G, r_V);
        dim = 0;
        writeQuads1d(elem, tidx, tidy, comp, dim, nelem, r_V, d_V);
      } else {
        dim = 0;
        readQuads1d(elem, tidx, tidy, comp, dim, nelem, d_U, slice);
        ContractTransposeX1d(slice, tidx, tidy, r_U, c_G, r_V);
        writeDofs1d(elem, tidx, tidy, comp, nelem, r_V, d_V);
      }
    }
  }
}
  //////////
 //  2D  //
//////////

inline __device__ void readDofs2d(const int elem, const int tidx, const int tidy, const int comp,
                                  const int nelem, const CeedScalar *d_U, CeedScalar &U) {
  U = (tidx<P1D && tidy<P1D) ? d_U[tidx + tidy*P1D + comp*P1D*P1D + elem*BASIS_NCOMP*P1D*P1D ] : 0.0;
}

inline __device__ void writeDofs2d(const int elem, const int tidx, const int tidy, const int comp,
                                   const int nelem, const CeedScalar &r_V, CeedScalar *d_V) {
  if (tidx<P1D && tidy<P1D)
  {
      d_V[tidx + tidy*P1D + comp*P1D*P1D + elem*BASIS_NCOMP*P1D*P1D ] = r_V;
  }
}

inline __device__ void readQuads2d(const int elem, const int tidx, const int tidy, const int comp,
                                   const int dim, const int nelem, const CeedScalar *d_U, CeedScalar &U ) {
  U = d_U[tidx + tidy*Q1D + elem*Q1D*Q1D + comp*Q1D*Q1D*nelem + dim*BASIS_NCOMP*nelem*Q1D*Q1D];
}

inline __device__ void writeQuads2d(const int elem, const int tidx, const int tidy, const int comp,
                                    const int dim, const int nelem, const CeedScalar &r_V, CeedScalar *d_V) {
    d_V[tidx + tidy*Q1D + elem*Q1D*Q1D + comp*Q1D*Q1D*nelem + dim*BASIS_NCOMP*nelem*Q1D*Q1D ] = r_V;
}

inline __device__ void ContractX2d(CeedScalar *slice, const int tidx, const int tidy,
                                   const CeedScalar &U, const CeedScalar *B, CeedScalar &V) {
  slice[tidx+tidy*Q1D] = U;
  __syncthreads();
  V = 0.0;
  for (int i = 0; i < P1D; ++i)
  {
    V += B[i + tidx*P1D] * slice[i + tidy*Q1D];//contract x direction
  }
  __syncthreads();
}

inline __device__ void ContractY2d(CeedScalar *slice, const int tidx, const int tidy,
                                   const CeedScalar &U, const CeedScalar *B, CeedScalar &V) {
  slice[tidx+tidy*Q1D] = U;
  __syncthreads();
  V = 0.0;
  for (int i = 0; i < P1D; ++i)
  {
    V += B[i + tidy*P1D] * slice[tidx + i*Q1D];//contract y direction
  }
  __syncthreads();
}

inline __device__ void ContractTransposeY2d(CeedScalar *slice, const int tidx, const int tidy,
                                   const CeedScalar &U, const CeedScalar *B, CeedScalar &V) {
  slice[tidx+tidy*Q1D] = U;
  __syncthreads();
  V = 0.0;
  if (tidy<P1D)
  {
    for (int i = 0; i < Q1D; ++i)
    {
      V += B[tidy + i*P1D] * slice[tidx + i*Q1D];//contract y direction
    }
  }
  __syncthreads();
}

inline __device__ void ContractTransposeX2d(CeedScalar *slice, const int tidx, const int tidy,
                                   const CeedScalar &U, const CeedScalar *B, CeedScalar &V) {
  slice[tidx+tidy*Q1D] = U;
  __syncthreads();
  V = 0.0;
  if (tidx<P1D)
  {
    for (int i = 0; i < Q1D; ++i)
    {
      V += B[tidx + i*P1D] * slice[i + tidy*Q1D];//contract x direction
    }
  }
  __syncthreads();
}

inline __device__ void interp2d(const CeedInt nelem, const int transpose,
                                const CeedScalar *c_B, const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V,
                                CeedScalar *slice) {
  CeedScalar r_V;
  CeedScalar r_t;

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < nelem; elem += gridDim.x*blockDim.z) {
    for(int comp=0; comp<BASIS_NCOMP; comp++) {
      r_V = 0.0;
      r_t = 0.0;
      if(!transpose) {
        readDofs2d(elem, tidx, tidy, comp, nelem, d_U, r_V);
        ContractX2d(slice, tidx, tidy, r_V, c_B, r_t);
        ContractY2d(slice, tidx, tidy, r_t, c_B, r_V);
        writeQuads2d(elem, tidx, tidy, comp, 0, nelem, r_V, d_V);
      } else {
        readQuads2d(elem, tidx, tidy, comp, 0, nelem, d_U, r_V);
        ContractTransposeY2d(slice, tidx, tidy, r_V, c_B, r_t);
        ContractTransposeX2d(slice, tidx, tidy, r_t, c_B, r_V);
        writeDofs2d(elem, tidx, tidy, comp, nelem, r_V, d_V);
      }
    }
  }
}

inline __device__ void grad2d(const CeedInt nelem, const int transpose,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V,
                              CeedScalar *slice) {
  CeedScalar r_U;
  CeedScalar r_V;
  CeedScalar r_t;

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  int dim;

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < nelem; elem += gridDim.x*blockDim.z) {
    for(int comp=0; comp<BASIS_NCOMP; comp++) {
      if(!transpose) {
        readDofs2d(elem, tidx, tidy, comp, nelem, d_U, r_U);
        ContractX2d(slice, tidx, tidy, r_U, c_G, r_t);
        ContractY2d(slice, tidx, tidy, r_t, c_B, r_V);
        dim = 0;
        writeQuads2d(elem, tidx, tidy, comp, dim, nelem, r_V, d_V);
        ContractX2d(slice, tidx, tidy, r_U, c_B, r_t);
        ContractY2d(slice, tidx, tidy, r_t, c_G, r_V);
        dim = 1;
        writeQuads2d(elem, tidx, tidy, comp, dim, nelem, r_V, d_V);
      } else {
        dim = 0;
        readQuads2d(elem, tidx, tidy, comp, dim, nelem, d_U, r_U);
        ContractTransposeY2d(slice, tidx, tidy, r_U, c_G, r_t);
        ContractTransposeX2d(slice, tidx, tidy, r_t, c_B, r_V);
        dim = 1;
        readQuads2d(elem, tidx, tidy, comp, dim, nelem, d_U, r_U);
        ContractTransposeY2d(slice, tidx, tidy, r_U, c_B, r_t);
        ContractTransposeX2d(slice, tidx, tidy, r_t, c_G, r_U);
        r_V+=r_U;
        writeDofs2d(elem, tidx, tidy, comp, nelem, r_V, d_V);
      }
    }
  }
}
  //////////
 //  3D  //
//////////

inline __device__ void readDofs3d(const int elem, const int tidx, const int tidy, const int comp,
                                  const int nelem, const CeedScalar *d_U, CeedScalar *r_U) {
  for (int i = 0; i < P1D; i++)
    r_U[i] = (tidx<P1D && tidy<P1D) ? d_U[tidx + tidy*P1D + i*P1D*P1D + comp*P1D*P1D*P1D + elem*BASIS_NCOMP*P1D*P1D*P1D ] : 0.0;
  for (int i = P1D; i < Q1D; i++)
    r_U[i] = 0.0;
}

inline __device__ void readQuads3d(const int elem, const int tidx, const int tidy, const int comp,
                                   const int dim, const int nelem, const CeedScalar *d_U, CeedScalar *r_U) {
  for (int i = 0; i < Q1D; i++)
    r_U[i] = d_U[tidx + tidy*Q1D + i*Q1D*Q1D + elem*Q1D*Q1D*Q1D + comp*Q1D*Q1D*Q1D*nelem + dim*BASIS_NCOMP*nelem*Q1D*Q1D*Q1D];
}

inline __device__ void writeDofs3d(const int elem, const int tidx, const int tidy, const int comp,
                                   const int nelem, const CeedScalar *r_V, CeedScalar *d_V) {
  if (tidx<P1D && tidy<P1D)
  {
    for (int i = 0; i < P1D; i++)
      d_V[tidx + tidy*P1D + i*P1D*P1D + comp*P1D*P1D*P1D + elem*BASIS_NCOMP*P1D*P1D*P1D ] = r_V[i];
  }
}

inline __device__ void writeQuads3d(const int elem, const int tidx, const int tidy, const int comp,
                                    const int dim, const int nelem, const CeedScalar *r_V, CeedScalar *d_V) {
  for (int i = 0; i < Q1D; i++)
    d_V[tidx + tidy*Q1D + i*Q1D*Q1D + elem*Q1D*Q1D*Q1D + comp*Q1D*Q1D*Q1D*nelem + dim*BASIS_NCOMP*nelem*Q1D*Q1D*Q1D ] = r_V[i];
}

inline __device__ void ContractX3d(CeedScalar *slice, const int tidx, const int tidy,
                                   const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  for (int k = 0; k < P1D; ++k)
  {
    slice[tidx+tidy*Q1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    for (int i = 0; i < P1D; ++i)
    {
      V[k] += B[i + tidx*P1D] * slice[i + tidy*Q1D];//contract x direction
    }
    __syncthreads();
  }
}

inline __device__ void ContractY3d(CeedScalar *slice, const int tidx, const int tidy,
                                   const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  for (int k = 0; k < P1D; ++k)
  {
    slice[tidx+tidy*Q1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    for (int i = 0; i < P1D; ++i)
    {
      V[k] += B[i + tidy*P1D] * slice[tidx + i*Q1D];//contract y direction
    }
    __syncthreads();
  }
}

inline __device__ void ContractZ3d(CeedScalar *slice, const int tidx, const int tidy,
                                   const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  for (int k = 0; k < Q1D; ++k)
  {
    V[k] = 0.0;
    for (int i = 0; i < P1D; ++i)
    {
      V[k] += B[i + k*P1D] * U[i];//contract z direction
    }
  }
}

inline __device__ void ContractTransposeZ3d(CeedScalar *slice, const int tidx, const int tidy,
                                            const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  for (int k = 0; k < Q1D; ++k)
  {
    V[k] = 0.0;
    if (k<P1D)
    {
      for (int i = 0; i < Q1D; ++i)
      {
        V[k] += B[k + i*P1D] * U[i];//contract z direction
      }
    }
  }
}

inline __device__ void ContractTransposeY3d(CeedScalar *slice, const int tidx, const int tidy,
                                            const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  for (int k = 0; k < P1D; ++k)
  {
    slice[tidx+tidy*Q1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (tidy<P1D)
    {
      for (int i = 0; i < Q1D; ++i)
      {
        V[k] += B[tidy + i*P1D] * slice[tidx + i*Q1D];//contract y direction
      }
    }
    __syncthreads();
  }
}

inline __device__ void ContractTransposeX3d(CeedScalar *slice, const int tidx, const int tidy,
                                            const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  for (int k = 0; k < P1D; ++k)
  {
    slice[tidx+tidy*Q1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (tidx<P1D)
    {
      for (int i = 0; i < Q1D; ++i)
      {
        V[k] += B[tidx + i*P1D] * slice[i + tidy*Q1D];//contract x direction
      }
    }
    __syncthreads();
  }
}

inline __device__ void interp3d(const CeedInt nelem, const int transpose,
                                const CeedScalar *c_B, const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V,
                                CeedScalar *slice) {
  CeedScalar r_V[Q1D];
  CeedScalar r_t[Q1D];

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < nelem; elem += gridDim.x*blockDim.z) {
    for(int comp=0; comp<BASIS_NCOMP; comp++) {
      for (int i = 0; i < Q1D; ++i)
      {
        r_V[i] = 0.0;
        r_t[i] = 0.0;
      }
      if(!transpose) {
        readDofs3d(elem, tidx, tidy, comp, nelem, d_U, r_V);
        ContractX3d(slice, tidx, tidy, r_V, c_B, r_t);
        ContractY3d(slice, tidx, tidy, r_t, c_B, r_V);
        ContractZ3d(slice, tidx, tidy, r_V, c_B, r_t);
        writeQuads3d(elem, tidx, tidy, comp, 0, nelem, r_t, d_V);
      } else {
        readQuads3d(elem, tidx, tidy, comp, 0, nelem, d_U, r_V);
        ContractTransposeZ3d(slice, tidx, tidy, r_V, c_B, r_t);
        ContractTransposeY3d(slice, tidx, tidy, r_t, c_B, r_V);
        ContractTransposeX3d(slice, tidx, tidy, r_V, c_B, r_t);
        writeDofs3d(elem, tidx, tidy, comp, nelem, r_t, d_V);
      }
    }
  }
}

inline __device__ void grad3d(const CeedInt nelem, const int transpose,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V,
                              CeedScalar *slice) {
  //use P1D for one of these
  CeedScalar r_U[Q1D];
  CeedScalar r_V[Q1D];
  CeedScalar r_t[Q1D];

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  int dim;

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < nelem; elem += gridDim.x*blockDim.z) {
    for(int comp=0; comp<BASIS_NCOMP; comp++) {
      if(!transpose) {
        readDofs3d(elem, tidx, tidy, comp, nelem, d_U, r_U);
        ContractX3d(slice, tidx, tidy, r_U, c_G, r_V);
        ContractY3d(slice, tidx, tidy, r_V, c_B, r_t);
        ContractZ3d(slice, tidx, tidy, r_t, c_B, r_V);
        dim = 0;
        writeQuads3d(elem, tidx, tidy, comp, dim, nelem, r_V, d_V);
        ContractX3d(slice, tidx, tidy, r_U, c_B, r_V);
        ContractY3d(slice, tidx, tidy, r_V, c_G, r_t);
        ContractZ3d(slice, tidx, tidy, r_t, c_B, r_V);
        dim = 1;
        writeQuads3d(elem, tidx, tidy, comp, dim, nelem, r_V, d_V);
        ContractX3d(slice, tidx, tidy, r_U, c_B, r_V);
        ContractY3d(slice, tidx, tidy, r_V, c_B, r_t);
        ContractZ3d(slice, tidx, tidy, r_t, c_G, r_V);
        dim = 2;
        writeQuads3d(elem, tidx, tidy, comp, dim, nelem, r_V, d_V);
      } else {
        dim = 0;
        readQuads3d(elem, tidx, tidy, comp, dim, nelem, d_U, r_U);
        ContractTransposeZ3d(slice, tidx, tidy, r_U, c_G, r_t);
        ContractTransposeY3d(slice, tidx, tidy, r_t, c_B, r_U);
        ContractTransposeX3d(slice, tidx, tidy, r_U, c_B, r_V);
        dim = 1;
        readQuads3d(elem, tidx, tidy, comp, dim, nelem, d_U, r_U);
        ContractTransposeZ3d(slice, tidx, tidy, r_U, c_B, r_t);
        ContractTransposeY3d(slice, tidx, tidy, r_t, c_G, r_U);
        ContractTransposeX3d(slice, tidx, tidy, r_U, c_B, r_t);
        add(r_V, r_t);
        dim = 2;
        readQuads3d(elem, tidx, tidy, comp, dim, nelem, d_U, r_U);
        ContractTransposeZ3d(slice, tidx, tidy, r_U, c_B, r_t);
        ContractTransposeY3d(slice, tidx, tidy, r_t, c_B, r_U);
        ContractTransposeX3d(slice, tidx, tidy, r_U, c_G, r_t);
        add(r_V, r_t);
        writeDofs3d(elem, tidx, tidy, comp, nelem, r_V, d_V);
      }
    }
  }
}

  /////////////
 // Kernels //
/////////////
extern "C" __global__ void interp(const CeedInt nelem, const int transpose,
                                  const CeedScalar *c_B, const CeedScalar *__restrict__ d_U,
                                  CeedScalar *__restrict__ d_V) {
  __shared__ double slice[Q1D*Q1D];
  if (BASIS_DIM==1) {
    interp1d(nelem, transpose, c_B, d_U, d_V, slice);
  } else if (BASIS_DIM==2) {
    interp2d(nelem, transpose, c_B, d_U, d_V, slice);
  } else if (BASIS_DIM==3) {
    interp3d(nelem, transpose, c_B, d_U, d_V, slice);
  }
}

extern "C" __global__ void grad(const CeedInt nelem, const int transpose,
                                const CeedScalar *c_B, const CeedScalar *c_G,
                                const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V) {
  __shared__ double slice[Q1D*Q1D];
  if (BASIS_DIM==1) {
    grad1d(nelem, transpose, c_B, c_G, d_U, d_V, slice);
  } else if (BASIS_DIM==2) {
    grad2d(nelem, transpose, c_B, c_G, d_U, d_V, slice);
  } else if (BASIS_DIM==3) {
    grad3d(nelem, transpose, c_B, c_G, d_U, d_V, slice);
  }
}

  /////////////
 // Weights //
/////////////
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
          const int ind = e*Q1D*Q1D*Q1D + i + j*Q1D + k*Q1D*Q1D;//sequential
          w[ind] = w1d[i]*w1d[j]*w1d[k];
        }
      }
    }
  }
}

extern "C" __global__ void weight(const CeedInt nelem,
                                  const CeedScalar *__restrict__ qweight1d, CeedScalar *__restrict__ v) {
  if (BASIS_DIM==1) {
    weight1d(nelem, qweight1d, v);
  } else if (BASIS_DIM==2) {
    weight2d(nelem, qweight1d, v);
  } else if (BASIS_DIM==3) {
    weight3d(nelem, qweight1d, v);
  }
}

);


//*********************
// reg kernels
static const char *kernels3dreg = QUOTE(

typedef CeedScalar real;

//TODO remove the magic number 32

//Read non interleaved dofs
inline __device__ void readDofs(const int bid, const int tid, const int comp,
const int size, const int nelem, const CeedScalar *d_U, real *r_U) {
  for (int i = 0; i < size; i++)
    //r_U[i] = d_U[tid + i*32 + bid*32*size + comp*size*nelem];
    //r_U[i] = d_U[i + tid*size + bid*32*size + comp*size*nelem ];
    r_U[i] = d_U[i + comp*size + tid*BASIS_NCOMP*size + bid*32*BASIS_NCOMP*size ];
}

//read interleaved quads
inline __device__ void readQuads(const int bid, const int tid, const int comp,
                                 const int dim, const int size, const int nelem, const CeedScalar *d_U,
                                 real *r_U) {
  for (int i = 0; i < size; i++)
    r_U[i] = d_U[i + tid*size + bid*32*size + comp*size*nelem +
                 dim*BASIS_NCOMP*nelem*size];
  //r_U[i] = d_U[tid + i*32 + bid*32*size + comp*nelem*size + dim*BASIS_NCOMP*nelem*size];
}

//Write non interleaved dofs
inline __device__ void writeDofs(const int bid, const int tid, const int comp,
                                 const int size, const int nelem, const CeedScalar *r_V, real *d_V) {
  for (int i = 0; i < size; i++)
    //d_V[i + tid*size + bid*32*size + comp*size*nelem ] = r_V[i];
    d_V[i + comp*size + tid*BASIS_NCOMP*size + bid*32*BASIS_NCOMP*size ] = r_V[i];
}

//Write interleaved quads
inline __device__ void writeQuads(const int bid, const int tid, const int comp,
                                  const int dim, const int size, const int nelem, const CeedScalar *r_V,
                                  real *d_V) {
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
                                const CeedScalar *c_B, const CeedScalar *__restrict__ d_U,
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
                              const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V) {
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
                                const CeedScalar *c_B, const CeedScalar *__restrict__ d_U,
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
                              const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V) {
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
                                const CeedScalar *c_B, const CeedScalar *__restrict__ d_U,
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
                              const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V) {
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
                                  const CeedScalar *c_B, const CeedScalar *__restrict__ d_U,
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
                                const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V) {
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
                                  const CeedScalar *__restrict__ qweight1d, CeedScalar *__restrict__ v) {
  if (BASIS_DIM==1) {
    weight1d(nelem, qweight1d, v);
  } else if (BASIS_DIM==2) {
    weight2d(nelem, qweight1d, v);
  } else if (BASIS_DIM==3) {
    weight3d(nelem, qweight1d, v);
  }
}

                                  );

int CeedCudaRegInitInterp(CeedScalar *d_B, CeedInt P1d, CeedInt Q1d,
                          CeedScalar **c_B);
int CeedCudaRegInitInterpGrad(CeedScalar *d_B, CeedScalar *d_G, CeedInt P1d,
                              CeedInt Q1d, CeedScalar **c_B_ptr, CeedScalar **c_G_ptr);

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
  // const int warpsize  = 32;
  // const int blocksize = warpsize;
  // const int blocksize = basis->Q1d*basis->Q1d;
  // const int gridsize  = nelem/warpsize + ( (nelem/warpsize*warpsize<nelem)? 1 :
  //                       0 );
  // const int gridsize  = nelem;
  const int optElems[7] = {0,32,8,3,2,1,8};
  int elemsPerBlock = 1;//basis->Q1d < 7 ? optElems[basis->Q1d] : 1;
  int grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)? 1 : 0 );

  const CeedScalar *d_u;
  CeedScalar *d_v;
  if(emode!=CEED_EVAL_WEIGHT) {
    ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u); CeedChk(ierr);
  }
  ierr = CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v); CeedChk(ierr);

  if (tmode == CEED_TRANSPOSE) {
    ierr = cudaMemset(d_v, 0, v->length * sizeof(CeedScalar)); CeedChk(ierr);
  }
  if (emode == CEED_EVAL_INTERP) {
    //TODO: check performance difference between c_B and d_B
    ierr = CeedCudaRegInitInterp(data->d_interp1d, basis->P1d, basis->Q1d,
                                 &data->c_B);
    CeedChk(ierr);
    void *interpargs[] = {(void *) &nelem, (void *) &transpose, &data->c_B, &d_u, &d_v};
    ierr = run_kernel_dim(ceed, data->interp, grid, basis->Q1d, basis->Q1d, elemsPerBlock, interpargs);
    CeedChk(ierr);
  } else if (emode == CEED_EVAL_GRAD) {
    ierr = CeedCudaRegInitInterpGrad(data->d_interp1d, data->d_grad1d, basis->P1d,
                                     basis->Q1d, &data->c_B, &data->c_G);
    CeedChk(ierr);
    void *gradargs[] = {(void *) &nelem, (void *) &transpose, &data->c_B, &data->c_G, &d_u, &d_v};
    ierr = run_kernel_dim(ceed, data->grad, grid, basis->Q1d, basis->Q1d, elemsPerBlock, gradargs);
    CeedChk(ierr);
  } else if (emode == CEED_EVAL_WEIGHT) {
    void *weightargs[] = {(void *) &nelem, (void *) &data->d_qweight1d, &d_v};
    const int blocksize = 32;
    int gridsize = nelem/32;
    if (blocksize * gridsize < nelem)
      gridsize += 1;
    ierr = run_kernel(ceed, data->weight, gridsize, blocksize, weightargs);
  }

  if(emode!=CEED_EVAL_WEIGHT) {
    ierr = CeedVectorRestoreArrayRead(u, &d_u); CeedChk(ierr);
  }
  ierr = CeedVectorRestoreArray(v, &d_v); CeedChk(ierr);

  return 0;
}

static int CeedBasisDestroy_Cuda_reg(CeedBasis basis) {
  int ierr;

  CeedBasis_Cuda_reg *data;
  ierr = CeedBasisGetData(basis, (void *) &data); CeedChk(ierr);

  CeedChk_Cu(basis->ceed, cuModuleUnload(data->module));

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
  CeedBasis_Cuda_reg *data;
  ierr = CeedCalloc(1, &data); CeedChk(ierr);

  const CeedInt qBytes = basis->Q1d * sizeof(CeedScalar);
  ierr = cudaMalloc((void **)&data->d_qweight1d, qBytes); CeedChk(ierr);
  ierr = cudaMemcpy(data->d_qweight1d, basis->qweight1d, qBytes,
                    cudaMemcpyHostToDevice); CeedChk(ierr);

  const CeedInt iBytes = qBytes * basis->P1d;
  ierr = cudaMalloc((void **)&data->d_interp1d, iBytes); CeedChk(ierr);
  ierr = cudaMemcpy(data->d_interp1d, basis->interp1d, iBytes,
                    cudaMemcpyHostToDevice); CeedChk(ierr);

  ierr = cudaMalloc((void **)&data->d_grad1d, iBytes); CeedChk(ierr);
  ierr = cudaMemcpy(data->d_grad1d, basis->grad1d, iBytes,
                    cudaMemcpyHostToDevice); CeedChk(ierr);

  ierr = compile(basis->ceed, kernelsShared, &data->module, 7,
                 "Q1D", basis->Q1d,
                 "P1D", basis->P1d,
                 "BASIS_BUF_LEN", basis->ncomp * CeedIntPow(basis->Q1d > basis->P1d ?
                     basis->Q1d : basis->P1d, basis->dim),
                 "BASIS_DIM", basis->dim,
                 "BASIS_NCOMP", basis->ncomp,
                 "BASIS_ELEMSIZE", CeedIntPow(basis->P1d, basis->dim),
                 "BASIS_NQPT", CeedIntPow(basis->Q1d, basis->dim)
                ); CeedChk(ierr);
  ierr = get_kernel(basis->ceed, data->module, "interp", &data->interp);
  CeedChk(ierr);
  ierr = get_kernel(basis->ceed, data->module, "grad", &data->grad);
  CeedChk(ierr);
  ierr = get_kernel(basis->ceed, data->module, "weight", &data->weight);
  CeedChk(ierr);

  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
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

int CeedBasisCreateH1_Cuda_reg(CeedElemTopology topo, CeedInt dim,
                               CeedInt ndof, CeedInt nqpts,
                               const CeedScalar *interp,
                               const CeedScalar *grad,
                               const CeedScalar *qref,
                               const CeedScalar *qweight,
                               CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
  return CeedError(ceed, 1, "Backend does not implement generic H1 basis");
}
