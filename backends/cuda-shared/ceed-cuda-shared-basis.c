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
#include "ceed-cuda-shared.h"
#include "../cuda/ceed-cuda.h"

//*********************
// shared mem kernels
// *INDENT-OFF*
static const char *kernelsShared = QUOTE(

inline __device__ void add(CeedScalar *r_V, const CeedScalar *r_U) {
  for (int i = 0; i < Q1D; i++)
    r_V[i] += r_U[i];
}

//////////
//  1D  //
//////////

inline __device__ void readDofs1d(const int elem, const int tidx,
                                  const int tidy, const int tidz,const int comp,
                                  const int nelem, const CeedScalar *d_U,
                                  CeedScalar *slice) {
  for (int i = 0; i < P1D; i++)
    slice[i+tidz*Q1D] = d_U[i + comp*P1D + elem*BASIS_NCOMP*P1D];
  for (int i = P1D; i < Q1D; i++)
    slice[i+tidz*Q1D] = 0.0;
}

inline __device__ void writeDofs1d(const int elem, const int tidx,
                                   const int tidy, const int comp,
                                   const int nelem, const CeedScalar &r_V,
                                   CeedScalar *d_V) {
  if (tidx<P1D) {
    d_V[tidx + comp*P1D + elem*BASIS_NCOMP*P1D] = r_V;
  }
}

inline __device__ void readQuads1d(const int elem, const int tidx,
                                   const int tidy, const int tidz, const int comp,
                                   const int dim, const int nelem,
                                   const CeedScalar *d_U, CeedScalar *slice) {
  for (int i = 0; i < Q1D; i++)
    slice[i+tidz*Q1D] = d_U[i + elem*Q1D + comp*Q1D*nelem +
                            dim*BASIS_NCOMP*nelem*Q1D];
}

inline __device__ void writeQuads1d(const int elem, const int tidx,
                                    const int tidy, const int comp,
                                    const int dim, const int nelem,
                                    const CeedScalar &r_V, CeedScalar *d_V) {
  d_V[tidx + elem*Q1D + comp*Q1D*nelem + dim*BASIS_NCOMP*nelem*Q1D] = r_V;
}

inline __device__ void ContractX1d(CeedScalar *slice, const int tidx,
                                   const int tidy, const int tidz,
                                   const CeedScalar &U, const CeedScalar *B,
                                   CeedScalar &V) {
  V = 0.0;
  for (int i = 0; i < P1D; ++i) {
    V += B[i + tidx*P1D] * slice[i+tidz*Q1D];//contract x direction
  }
}

inline __device__ void ContractTransposeX1d(CeedScalar *slice, const int tidx,
    const int tidy, const int tidz,
    const CeedScalar &U, const CeedScalar *B, CeedScalar &V) {
  V = 0.0;
  for (int i = 0; i < Q1D; ++i) {
    V += B[tidx + i*P1D] * slice[i+tidz*Q1D];//contract x direction
  }
}

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
    for(int comp=0; comp<BASIS_NCOMP; comp++) {
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
    for(int comp=0; comp<BASIS_NCOMP; comp++) {
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
//////////
//  2D  //
//////////

inline __device__ void readDofs2d(const int elem, const int tidx,
                                  const int tidy, const int comp,
                                  const int nelem, const CeedScalar *d_U,
                                  CeedScalar &U) {
  U = (tidx<P1D
       && tidy<P1D) ? d_U[tidx + tidy*P1D + comp*P1D*P1D +
                          elem*BASIS_NCOMP*P1D*P1D ] :
      0.0;
}

inline __device__ void writeDofs2d(const int elem, const int tidx,
                                   const int tidy, const int comp,
                                   const int nelem, const CeedScalar &r_V,
                                   CeedScalar *d_V) {
  if (tidx<P1D && tidy<P1D) {
    d_V[tidx + tidy*P1D + comp*P1D*P1D + elem*BASIS_NCOMP*P1D*P1D ] = r_V;
  }
}

inline __device__ void readQuads2d(const int elem, const int tidx,
                                   const int tidy, const int comp,
                                   const int dim, const int nelem,
                                   const CeedScalar *d_U, CeedScalar &U ) {
  U = d_U[tidx + tidy*Q1D + elem*Q1D*Q1D + comp*Q1D*Q1D*nelem +
               dim*BASIS_NCOMP*nelem*Q1D*Q1D];
}

inline __device__ void writeQuads2d(const int elem, const int tidx,
                                    const int tidy, const int comp,
                                    const int dim, const int nelem,
                                    const CeedScalar &r_V, CeedScalar *d_V) {
  d_V[tidx + tidy*Q1D + elem*Q1D*Q1D + comp*Q1D*Q1D*nelem +
           dim*BASIS_NCOMP*nelem*Q1D*Q1D ] = r_V;
}

inline __device__ void ContractX2d(CeedScalar *slice, const int tidx,
                                   const int tidy, const int tidz,
                                   const CeedScalar &U, const CeedScalar *B,
                                   CeedScalar &V) {
  slice[tidx+tidy*Q1D+tidz*Q1D*Q1D] = U;
  __syncthreads();
  V = 0.0;
  for (int i = 0; i < P1D; ++i) {
    V += B[i + tidx*P1D] * slice[i + tidy*Q1D + tidz*Q1D*Q1D];//contract x direction
  }
  __syncthreads();
}

inline __device__ void ContractY2d(CeedScalar *slice, const int tidx,
                                   const int tidy, const int tidz,
                                   const CeedScalar &U, const CeedScalar *B,
                                   CeedScalar &V) {
  slice[tidx+tidy*Q1D+tidz*Q1D*Q1D] = U;
  __syncthreads();
  V = 0.0;
  for (int i = 0; i < P1D; ++i) {
    V += B[i + tidy*P1D] * slice[tidx + i*Q1D + tidz*Q1D*Q1D];//contract y direction
  }
  __syncthreads();
}

inline __device__ void ContractTransposeY2d(CeedScalar *slice, const int tidx,
    const int tidy, const int tidz,
    const CeedScalar &U, const CeedScalar *B, CeedScalar &V) {
  slice[tidx+tidy*Q1D+tidz*Q1D*Q1D] = U;
  __syncthreads();
  V = 0.0;
  if (tidy<P1D) {
    for (int i = 0; i < Q1D; ++i) {
      V += B[tidy + i*P1D] * slice[tidx + i*Q1D + tidz*Q1D*Q1D];//contract y direction
    }
  }
  __syncthreads();
}

inline __device__ void ContractTransposeX2d(CeedScalar *slice, const int tidx,
    const int tidy, const int tidz,
    const CeedScalar &U, const CeedScalar *B, CeedScalar &V) {
  slice[tidx+tidy*Q1D+tidz*Q1D*Q1D] = U;
  __syncthreads();
  V = 0.0;
  if (tidx<P1D) {
    for (int i = 0; i < Q1D; ++i) {
      V += B[tidx + i*P1D] * slice[i + tidy*Q1D + tidz*Q1D*Q1D];//contract x direction
    }
  }
  __syncthreads();
}

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
      r_V+=r_U;
      writeDofs2d(elem, tidx, tidy, comp, nelem, r_V, d_V);
    }
  }
}
//////////
//  3D  //
//////////

inline __device__ void readDofs3d(const int elem, const int tidx,
                                  const int tidy, const int comp,
                                  const int nelem, const CeedScalar *d_U,
                                  CeedScalar *r_U) {
  for (int i = 0; i < P1D; i++)
    r_U[i] = (tidx<P1D
              && tidy<P1D) ? d_U[tidx + tidy*P1D + i*P1D*P1D + comp*P1D*P1D*P1D +
                                      elem*BASIS_NCOMP*P1D*P1D*P1D ] : 0.0;
  for (int i = P1D; i < Q1D; i++)
    r_U[i] = 0.0;
}

inline __device__ void readQuads3d(const int elem, const int tidx,
                                   const int tidy, const int comp,
                                   const int dim, const int nelem,
                                   const CeedScalar *d_U, CeedScalar *r_U) {
  for (int i = 0; i < Q1D; i++)
    r_U[i] = d_U[tidx + tidy*Q1D + i*Q1D*Q1D + elem*Q1D*Q1D*Q1D +
                 comp*Q1D*Q1D*Q1D*nelem + dim*BASIS_NCOMP*nelem*Q1D*Q1D*Q1D];
}

inline __device__ void writeDofs3d(const int elem, const int tidx,
                                   const int tidy, const int comp,
                                   const int nelem, const CeedScalar *r_V,
                                   CeedScalar *d_V) {
  if (tidx<P1D && tidy<P1D) {
    for (int i = 0; i < P1D; i++)
      d_V[tidx + tidy*P1D + i*P1D*P1D + comp*P1D*P1D*P1D +
          elem*BASIS_NCOMP*P1D*P1D*P1D ] = r_V[i];
  }
}

inline __device__ void writeQuads3d(const int elem, const int tidx,
                                    const int tidy, const int comp,
                                    const int dim, const int nelem,
                                    const CeedScalar *r_V, CeedScalar *d_V) {
  for (int i = 0; i < Q1D; i++)
    d_V[tidx + tidy*Q1D + i*Q1D*Q1D + elem*Q1D*Q1D*Q1D + comp*Q1D*Q1D*Q1D*nelem +
        dim*BASIS_NCOMP*nelem*Q1D*Q1D*Q1D ] = r_V[i];
}

inline __device__ void ContractX3d(CeedScalar *slice, const int tidx,
                                   const int tidy, const int tidz,
                                   const CeedScalar *U, const CeedScalar *B,
                                   CeedScalar *V) {
  for (int k = 0; k < P1D; ++k) {
    slice[tidx+tidy*Q1D+tidz*Q1D*Q1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    for (int i = 0; i < P1D; ++i) {
      V[k] += B[i + tidx*P1D] * slice[i + tidy*Q1D +
                                      tidz*Q1D*Q1D];//contract x direction
    }
    __syncthreads();
  }
}

inline __device__ void ContractY3d(CeedScalar *slice, const int tidx,
                                   const int tidy, const int tidz,
                                   const CeedScalar *U, const CeedScalar *B,
                                   CeedScalar *V) {
  for (int k = 0; k < P1D; ++k) {
    slice[tidx+tidy*Q1D+tidz*Q1D*Q1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    for (int i = 0; i < P1D; ++i) {
      V[k] += B[i + tidy*P1D] * slice[tidx + i*Q1D +
                                      tidz*Q1D*Q1D];//contract y direction
    }
    __syncthreads();
  }
}

inline __device__ void ContractZ3d(CeedScalar *slice, const int tidx,
                                   const int tidy, const int tidz,
                                   const CeedScalar *U, const CeedScalar *B,
                                   CeedScalar *V) {
  for (int k = 0; k < Q1D; ++k) {
    V[k] = 0.0;
    for (int i = 0; i < P1D; ++i) {
      V[k] += B[i + k*P1D] * U[i];//contract z direction
    }
  }
}

inline __device__ void ContractTransposeZ3d(CeedScalar *slice, const int tidx,
    const int tidy, const int tidz,
    const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  for (int k = 0; k < Q1D; ++k) {
    V[k] = 0.0;
    if (k<P1D) {
      for (int i = 0; i < Q1D; ++i) {
        V[k] += B[k + i*P1D] * U[i];//contract z direction
      }
    }
  }
}

inline __device__ void ContractTransposeY3d(CeedScalar *slice, const int tidx,
    const int tidy, const int tidz,
    const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  for (int k = 0; k < P1D; ++k) {
    slice[tidx+tidy*Q1D+tidz*Q1D*Q1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (tidy<P1D) {
      for (int i = 0; i < Q1D; ++i) {
        V[k] += B[tidy + i*P1D] * slice[tidx + i*Q1D +
                                        tidz*Q1D*Q1D];//contract y direction
      }
    }
    __syncthreads();
  }
}

inline __device__ void ContractTransposeX3d(CeedScalar *slice, const int tidx,
    const int tidy, const int tidz,
    const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  for (int k = 0; k < P1D; ++k) {
    slice[tidx+tidy*Q1D+tidz*Q1D*Q1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (tidx<P1D) {
      for (int i = 0; i < Q1D; ++i) {
        V[k] += B[tidx + i*P1D] * slice[i + tidy*Q1D +
                                        tidz*Q1D*Q1D];//contract x direction
      }
    }
    __syncthreads();
  }
}

inline __device__ void interp3d(const CeedInt nelem, const int transpose,
                                const CeedScalar *c_B,
                                const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V,
                                CeedScalar *slice) {
  CeedScalar r_V[Q1D];
  CeedScalar r_t[Q1D];

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int tidz = threadIdx.z;
  const int blockElem = tidz/BASIS_NCOMP;
  const int elemsPerBlock = blockDim.z/BASIS_NCOMP;
  const int comp = tidz%BASIS_NCOMP;

  for (CeedInt elem = blockIdx.x*elemsPerBlock + blockElem; elem < nelem;
       elem += gridDim.x*elemsPerBlock) {
    for (int i = 0; i < Q1D; ++i) {
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

inline __device__ void grad3d(const CeedInt nelem, const int transpose,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              const CeedScalar *__restrict__ d_U,
                              CeedScalar *__restrict__ d_V,
                              CeedScalar *slice) {
  //use P1D for one of these
  CeedScalar r_U[Q1D];
  CeedScalar r_V[Q1D];
  CeedScalar r_t[Q1D];

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

/////////////
// Kernels //
/////////////
extern "C" __global__ void interp(const CeedInt nelem, const int transpose,
                                  const CeedScalar *c_B,
                                  const CeedScalar *__restrict__ d_U,
                                  CeedScalar *__restrict__ d_V) {
  extern __shared__ double slice[];
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
                                const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V) {
  extern __shared__ double slice[];
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
  const int tid = threadIdx.x;
  const CeedScalar weight = qweight1d[tid];
  for (CeedInt elem = blockIdx.x*blockDim.y + threadIdx.y; elem < nelem;
       elem += gridDim.x*blockDim.y) {
    const int ind = elem*Q1D + tid;
    w[ind] = weight;
  }
}

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
// *INDENT-ON*

int CeedCudaInitInterp(CeedScalar *d_B, CeedInt P1d, CeedInt Q1d,
                       CeedScalar **c_B);
int CeedCudaInitInterpGrad(CeedScalar *d_B, CeedScalar *d_G, CeedInt P1d,
                           CeedInt Q1d, CeedScalar **c_B_ptr,
                           CeedScalar **c_G_ptr);

int CeedBasisApplyTensor_Cuda_shared(CeedBasis basis, const CeedInt nelem,
                                     CeedTransposeMode tmode,
                                     CeedEvalMode emode, CeedVector u,
                                     CeedVector v) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
  Ceed_Cuda_shared *ceed_Cuda;
  CeedGetData(ceed, (void *) &ceed_Cuda); CeedChk(ierr);
  CeedBasis_Cuda_shared *data;
  CeedBasisGetData(basis, (void *)&data); CeedChk(ierr);
  const CeedInt transpose = tmode == CEED_TRANSPOSE;
  CeedInt dim, ncomp;
  ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChk(ierr);

  const CeedScalar *d_u;
  CeedScalar *d_v;
  if (emode != CEED_EVAL_WEIGHT) {
    ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u); CeedChk(ierr);
  }
  ierr = CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v); CeedChk(ierr);

  if (tmode == CEED_TRANSPOSE) {
    CeedInt length;
    ierr = CeedVectorGetLength(v, &length); CeedChk(ierr);
    ierr = cudaMemset(d_v, 0, length * sizeof(CeedScalar)); CeedChk(ierr);
  }
  if (emode == CEED_EVAL_INTERP) {
    CeedInt P1d, Q1d;
    ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChk(ierr);
    ierr = CeedCudaInitInterp(data->d_interp1d, P1d, Q1d, &data->c_B);
    CeedChk(ierr);
    void *interpargs[] = {(void *) &nelem, (void *) &transpose, &data->c_B,
                          &d_u, &d_v
                         };
    if (dim==1) {
      CeedInt elemsPerBlock = 32;
      CeedInt grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)
                                             ? 1 : 0 );
      CeedInt sharedMem = elemsPerBlock*Q1d*sizeof(CeedScalar);
      ierr = CeedRunKernelDimSharedCuda(ceed, data->interp, grid, Q1d, 1,
                                        elemsPerBlock, sharedMem,
                                        interpargs);
      CeedChk(ierr);
    } else if (dim==2) {
      const CeedInt optElems[7] = {0,32,8,6,4,2,8};
      CeedInt elemsPerBlock = Q1d < 7 ? optElems[Q1d]/ncomp : 1;
      CeedInt grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)
                                             ? 1 : 0 );
      CeedInt sharedMem = ncomp*elemsPerBlock*Q1d*Q1d*sizeof(CeedScalar);
      ierr = CeedRunKernelDimSharedCuda(ceed, data->interp, grid, Q1d, Q1d,
                                        ncomp*elemsPerBlock, sharedMem,
                                        interpargs);
      CeedChk(ierr);
    } else if (dim==3) {
      CeedInt elemsPerBlock = 1;
      CeedInt grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)
                                             ? 1 : 0 );
      CeedInt sharedMem = ncomp*elemsPerBlock*Q1d*Q1d*sizeof(CeedScalar);
      ierr = CeedRunKernelDimSharedCuda(ceed, data->interp, grid, Q1d, Q1d,
                                        ncomp*elemsPerBlock, sharedMem,
                                        interpargs);
      CeedChk(ierr);
    }
  } else if (emode == CEED_EVAL_GRAD) {
    CeedInt P1d, Q1d;
    ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChk(ierr);
    ierr = CeedCudaInitInterpGrad(data->d_interp1d, data->d_grad1d, P1d,
                                  Q1d, &data->c_B, &data->c_G);
    CeedChk(ierr);
    void *gradargs[] = {(void *) &nelem, (void *) &transpose, &data->c_B,
                        &data->c_G, &d_u, &d_v
                       };
    if (dim==1) {
      CeedInt elemsPerBlock = 32;
      CeedInt grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)
                                             ? 1 : 0 );
      CeedInt sharedMem = elemsPerBlock*Q1d*sizeof(CeedScalar);
      ierr = CeedRunKernelDimSharedCuda(ceed, data->grad, grid, Q1d, 1, elemsPerBlock,
                                        sharedMem,
                                        gradargs);
      CeedChk(ierr);
    } else if (dim==2) {
      const CeedInt optElems[7] = {0,32,8,6,4,2,8};
      CeedInt elemsPerBlock = Q1d < 7 ? optElems[Q1d]/ncomp : 1;
      CeedInt grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)
                                             ? 1 : 0 );
      CeedInt sharedMem = ncomp*elemsPerBlock*Q1d*Q1d*sizeof(CeedScalar);
      ierr = CeedRunKernelDimSharedCuda(ceed, data->grad, grid, Q1d, Q1d,
                                        ncomp*elemsPerBlock, sharedMem,
                                        gradargs);
      CeedChk(ierr);
    } else if (dim==3) {
      CeedInt elemsPerBlock = 1;
      CeedInt grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)
                                             ? 1 : 0 );
      CeedInt sharedMem = ncomp*elemsPerBlock*Q1d*Q1d*sizeof(CeedScalar);
      ierr = CeedRunKernelDimSharedCuda(ceed, data->grad, grid, Q1d, Q1d,
                                        ncomp*elemsPerBlock, sharedMem,
                                        gradargs);
      CeedChk(ierr);
    }
  } else if (emode == CEED_EVAL_WEIGHT) {
    CeedInt Q1d;
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChk(ierr);
    void *weightargs[] = {(void *) &nelem, (void *) &data->d_qweight1d, &d_v};
    if (dim == 1) {
      const CeedInt elemsPerBlock = 32/Q1d;
      const CeedInt gridsize = nelem/elemsPerBlock + ( (
                                 nelem/elemsPerBlock*elemsPerBlock<nelem)? 1 : 0 );
      ierr = CeedRunKernelDimCuda(ceed, data->weight, gridsize, Q1d,
                                  elemsPerBlock, 1, weightargs);
      CeedChk(ierr);
    } else if (dim == 2) {
      const CeedInt optElems = 32/(Q1d*Q1d);
      const CeedInt elemsPerBlock = optElems>0?optElems:1;
      const CeedInt gridsize = nelem/elemsPerBlock + ( (
                                 nelem/elemsPerBlock*elemsPerBlock<nelem)? 1 : 0 );
      ierr = CeedRunKernelDimCuda(ceed, data->weight, gridsize, Q1d, Q1d,
                                  elemsPerBlock, weightargs);
      CeedChk(ierr);
    } else if (dim == 3) {
      const CeedInt gridsize = nelem;
      ierr = CeedRunKernelDimCuda(ceed, data->weight, gridsize, Q1d, Q1d, Q1d,
                                  weightargs);
      CeedChk(ierr);
    }
  }

  if (emode != CEED_EVAL_WEIGHT) {
    ierr = CeedVectorRestoreArrayRead(u, &d_u); CeedChk(ierr);
  }
  ierr = CeedVectorRestoreArray(v, &d_v); CeedChk(ierr);

  return 0;
}

static int CeedBasisDestroy_Cuda_shared(CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);

  CeedBasis_Cuda_shared *data;
  ierr = CeedBasisGetData(basis, (void *) &data); CeedChk(ierr);

  CeedChk_Cu(ceed, cuModuleUnload(data->module));

  ierr = cudaFree(data->d_qweight1d); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree(data->d_interp1d); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree(data->d_grad1d); CeedChk_Cu(ceed, ierr);

  ierr = CeedFree(&data); CeedChk(ierr);

  return 0;
}

int CeedBasisCreateTensorH1_Cuda_shared(CeedInt dim, CeedInt P1d, CeedInt Q1d,
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
  CeedBasis_Cuda_shared *data;
  ierr = CeedCalloc(1, &data); CeedChk(ierr);

  const CeedInt qBytes = Q1d * sizeof(CeedScalar);
  ierr = cudaMalloc((void **)&data->d_qweight1d, qBytes); CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(data->d_qweight1d, qweight1d, qBytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);

  const CeedInt iBytes = qBytes * P1d;
  ierr = cudaMalloc((void **)&data->d_interp1d, iBytes); CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(data->d_interp1d, interp1d, iBytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);

  ierr = cudaMalloc((void **)&data->d_grad1d, iBytes); CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(data->d_grad1d, grad1d, iBytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);

  data->d_collograd1d = NULL;
  if (dim==3 && Q1d >= P1d) {
    CeedScalar *collograd1d;
    ierr = CeedMalloc(Q1d*Q1d, &collograd1d); CeedChk(ierr);
    ierr = CeedBasisGetCollocatedGrad(basis, collograd1d); CeedChk(ierr);
    ierr = cudaMalloc((void **)&data->d_collograd1d, qBytes * Q1d);
    CeedChk_Cu(ceed, ierr);
    ierr = cudaMemcpy(data->d_collograd1d, collograd1d, qBytes * Q1d,
                      cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);
  }

  CeedInt ncomp;
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChk(ierr);
  ierr = CeedCompileCuda(ceed, kernelsShared, &data->module, 7,
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
                                CeedBasisApplyTensor_Cuda_shared);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Destroy",
                                CeedBasisDestroy_Cuda_shared);
  CeedChk(ierr);
  return 0;
}
