// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#define CEED_DEBUG_COLOR 12

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include "ceed-cuda-gen.h"
#include "../cuda/ceed-cuda-compile.h"
#include "../cuda-ref/ceed-cuda-ref.h"
#include "../cuda-shared/ceed-cuda-shared.h"

static const char *atomicAdd = QUOTE(
//------------------------------------------------------------------------------
// Atomic add, for older CUDA
//------------------------------------------------------------------------------
__device__ CeedScalar atomicAdd(CeedScalar *address, CeedScalar val) {
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
);

static const char *deviceFunctions = QUOTE(

//------------------------------------------------------------------------------
// Typedefs
//------------------------------------------------------------------------------
typedef struct { const CeedScalar* in[16]; CeedScalar* out[16]; } CudaFields;
typedef struct { CeedInt* in[16]; CeedInt* out[16]; } CudaFieldsInt;

typedef struct {
  CeedInt tidx;
  CeedInt tidy;
  CeedInt tidz;
  CeedInt tid;
  CeedScalar* slice;
} BackendData;

//------------------------------------------------------------------------------
// Load matrices for basis actions
//------------------------------------------------------------------------------
template <int P, int Q>
inline __device__ void loadMatrix(BackendData &data, const CeedScalar *__restrict__ d_B, CeedScalar *B) {
  for (CeedInt i = data.tid; i < P*Q; i += blockDim.x*blockDim.y*blockDim.z)
    B[i] = d_B[i];
}

//------------------------------------------------------------------------------
// 1D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// L-vector -> E-vector, offsets provided
//------------------------------------------------------------------------------
template <int NCOMP, int COMPSTRIDE, int P1d>
inline __device__ void readDofsOffset1d(BackendData &data, const CeedInt nnodes, const CeedInt elem, const CeedInt *__restrict__ indices, const CeedScalar *__restrict__ d_u, CeedScalar *r_u) {
  if (data.tidx < P1d) {
    const CeedInt node = data.tidx;
    const CeedInt ind = indices[node + elem * P1d];
    for (CeedInt comp = 0; comp < NCOMP; ++comp)
      r_u[comp] = d_u[ind + COMPSTRIDE * comp];
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, strided
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline __device__ void readDofsStrided1d(BackendData &data, const CeedInt elem, const CeedScalar *__restrict__ d_u, CeedScalar *r_u) {
  if (data.tidx < P1d) {
    const CeedInt node = data.tidx;
    const CeedInt ind = node * STRIDES_NODE + elem * STRIDES_ELEM;
    for (CeedInt comp = 0; comp < NCOMP; ++comp)
      r_u[comp] = d_u[ind + comp * STRIDES_COMP];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, offsets provided
//------------------------------------------------------------------------------
template <int NCOMP, int COMPSTRIDE, int P1d>
inline __device__ void writeDofsOffset1d(BackendData &data, const CeedInt nnodes, const CeedInt elem, const CeedInt *__restrict__ indices, const CeedScalar *r_v, CeedScalar *d_v) {
  if (data.tidx < P1d) {
    const CeedInt node = data.tidx;
    const CeedInt ind = indices[node + elem * P1d];
    for (CeedInt comp = 0; comp < NCOMP; ++comp)
      atomicAdd(&d_v[ind + COMPSTRIDE * comp], r_v[comp]);
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, strided
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline __device__ void writeDofsStrided1d(BackendData &data, const CeedInt elem, const CeedScalar *r_v, CeedScalar *d_v) {
  if (data.tidx < P1d) {
    const CeedInt node = data.tidx;
    const CeedInt ind = node * STRIDES_NODE + elem * STRIDES_ELEM;
    for (CeedInt comp = 0; comp < NCOMP; ++comp)
      d_v[ind + comp * STRIDES_COMP] += r_v[comp];
  }
}

//------------------------------------------------------------------------------
// 1D tensor contraction x
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractX1d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.tidx] = *U;
  __syncthreads();
  *V = 0.0;
  if (data.tidx < Q1d)
    for (CeedInt i = 0; i < P1d; ++i)
      *V += B[i + data.tidx*P1d] * data.slice[i]; // Contract x direction
  __syncthreads();
}

//------------------------------------------------------------------------------
// 1D transpose tensor contraction x
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractTransposeX1d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.tidx] = *U;
  __syncthreads();
  *V = 0.0;
  if (data.tidx < P1d)
    for (CeedInt i = 0; i < Q1d; ++i)
      *V += B[data.tidx + i*P1d] * data.slice[i]; // Contract x direction
  __syncthreads();
}

//------------------------------------------------------------------------------
// 1D interpolate to quadrature points
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void interp1d(BackendData &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, CeedScalar *__restrict__ r_V) {
  for (CeedInt comp = 0; comp < NCOMP; comp++)
    ContractX1d<NCOMP, P1d, Q1d>(data, r_U + comp, c_B, r_V + comp);
}

//------------------------------------------------------------------------------
// 1D interpolate transpose
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void interpTranspose1d(BackendData &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, CeedScalar *__restrict__ r_V) {
  for (CeedInt comp=0; comp<NCOMP; comp++)
    ContractTransposeX1d<NCOMP, P1d, Q1d>(data, r_U + comp, c_B, r_V + comp);
}

//------------------------------------------------------------------------------
// 1D derivatives at quadrature points
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void grad1d(BackendData &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G, CeedScalar *__restrict__ r_V) {
  for (CeedInt comp = 0; comp < NCOMP; comp++)
    ContractX1d<NCOMP, P1d, Q1d>(data, r_U + comp, c_G, r_V + comp);
}

//------------------------------------------------------------------------------
// 1D derivatives transpose
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void gradTranspose1d(BackendData &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G, CeedScalar *__restrict__ r_V) {
  for (CeedInt comp = 0; comp < NCOMP; comp++)
    ContractTransposeX1d<NCOMP, P1d, Q1d>(data, r_U + comp, c_G, r_V + comp);
}

//------------------------------------------------------------------------------
// 2D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// L-vector -> E-vector, offsets provided
//------------------------------------------------------------------------------
template <int NCOMP, int COMPSTRIDE, int P1d>
inline __device__ void readDofsOffset2d(BackendData &data, const CeedInt nnodes, const CeedInt elem, const CeedInt *__restrict__ indices, const CeedScalar *__restrict__ d_u, CeedScalar *r_u) {
  if (data.tidx < P1d && data.tidy < P1d) {
    const CeedInt node = data.tidx + data.tidy*P1d;
    const CeedInt ind = indices[node + elem * P1d*P1d];
    for (CeedInt comp = 0; comp < NCOMP; ++comp)
      r_u[comp] = d_u[ind + COMPSTRIDE * comp];
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, strided
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline __device__ void readDofsStrided2d(BackendData &data, const CeedInt elem, const CeedScalar *__restrict__ d_u, CeedScalar *r_u) {
  if (data.tidx < P1d && data.tidy < P1d) {
    const CeedInt node = data.tidx + data.tidy*P1d;
    const CeedInt ind = node * STRIDES_NODE + elem * STRIDES_ELEM;
    for (CeedInt comp = 0; comp < NCOMP; ++comp)
      r_u[comp] = d_u[ind + comp * STRIDES_COMP];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, offsets provided
//------------------------------------------------------------------------------
template <int NCOMP, int COMPSTRIDE, int P1d>
inline __device__ void writeDofsOffset2d(BackendData &data, const CeedInt nnodes, const CeedInt elem, const CeedInt *__restrict__ indices, const CeedScalar *r_v, CeedScalar *d_v) {
  if (data.tidx < P1d && data.tidy < P1d) {
    const CeedInt node = data.tidx + data.tidy*P1d;
    const CeedInt ind = indices[node + elem * P1d*P1d];
    for (CeedInt comp = 0; comp < NCOMP; ++comp)
      atomicAdd(&d_v[ind + COMPSTRIDE * comp], r_v[comp]);
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, strided
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline __device__ void writeDofsStrided2d(BackendData &data, const CeedInt elem, const CeedScalar *r_v, CeedScalar *d_v) {
  if (data.tidx < P1d && data.tidy < P1d) {
    const CeedInt node = data.tidx + data.tidy*P1d;
    const CeedInt ind = node * STRIDES_NODE + elem * STRIDES_ELEM;
    for (CeedInt comp = 0; comp < NCOMP; ++comp)
      d_v[ind + comp * STRIDES_COMP] += r_v[comp];
  }
}

//------------------------------------------------------------------------------
// 2D tensor contraction x
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractX2d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.tidx+data.tidy*T1d] = *U;
  __syncthreads();
  *V = 0.0;
  if (data.tidx < Q1d && data.tidy < P1d)
    for (CeedInt i = 0; i < P1d; ++i)
      *V += B[i + data.tidx*P1d] * data.slice[i + data.tidy*T1d]; // Contract x direction
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D tensor contract y
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractY2d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.tidx+data.tidy*T1d] = *U;
  __syncthreads();
  *V = 0.0;
  if (data.tidx < Q1d && data.tidy < Q1d)
    for (CeedInt i = 0; i < P1d; ++i)
      *V += B[i + data.tidy*P1d] * data.slice[data.tidx + i*T1d]; // Contract y direction
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D transpose tensor contract y
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractYTranspose2d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.tidx+data.tidy*T1d] = *U;
  __syncthreads();
  *V = 0.0;
  if (data.tidx < Q1d && data.tidy < P1d)
    for (CeedInt i = 0; i < Q1d; ++i)
      *V += B[data.tidy + i*P1d] * data.slice[data.tidx + i*T1d]; // Contract y direction
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D transpose tensor contract x
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractXTranspose2d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.tidx+data.tidy*T1d] = *U;
  __syncthreads();
  *V = 0.0;
  if (data.tidx < P1d && data.tidy < P1d)
    for (CeedInt i = 0; i < Q1d; ++i)
      *V += B[data.tidx + i*P1d] * data.slice[i + data.tidy*T1d]; // Contract x direction
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D transpose tensor contract and add x
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractXTransposeAdd2d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.tidx+data.tidy*T1d] = *U;
  __syncthreads();
  if (data.tidx < P1d && data.tidy < P1d)
    for (CeedInt i = 0; i < Q1d; ++i)
      *V += B[data.tidx + i*P1d] * data.slice[i + data.tidy*T1d]; // Contract x direction
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D interpolate to quadrature points
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void interp2d(BackendData &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, CeedScalar *__restrict__ r_V) {
  CeedScalar r_t[1];
  for (CeedInt comp = 0; comp < NCOMP; comp++) {
    ContractX2d<NCOMP, P1d, Q1d>(data, r_U + comp, c_B, r_t);
    ContractY2d<NCOMP, P1d, Q1d>(data, r_t, c_B, r_V + comp);
  }
}

//------------------------------------------------------------------------------
// 2D interpolate transpose
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void interpTranspose2d(BackendData &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, CeedScalar *__restrict__ r_V) {
  CeedScalar r_t[1];
  for (CeedInt comp = 0; comp < NCOMP; comp++) {
    ContractYTranspose2d<NCOMP, P1d, Q1d>(data, r_U + comp, c_B, r_t);
    ContractXTranspose2d<NCOMP, P1d, Q1d>(data, r_t, c_B, r_V + comp);
  }
}

//------------------------------------------------------------------------------
// 2D derivatives at quadrature points
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void grad2d(BackendData &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G, CeedScalar *__restrict__ r_V) {
  CeedScalar r_t[1];
  for (CeedInt comp = 0; comp < NCOMP; comp++) {
    ContractX2d<NCOMP, P1d, Q1d>(data, r_U + comp, c_G, r_t);
    ContractY2d<NCOMP, P1d, Q1d>(data, r_t, c_B, r_V + comp + 0*NCOMP);
    ContractX2d<NCOMP, P1d, Q1d>(data, r_U + comp, c_B, r_t);
    ContractY2d<NCOMP, P1d, Q1d>(data, r_t, c_G, r_V + comp + 1*NCOMP);
  }
}

//------------------------------------------------------------------------------
// 2D derivatives transpose
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void gradTranspose2d(BackendData &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G, CeedScalar *__restrict__ r_V) {
  CeedScalar r_t[1];
  for (CeedInt comp = 0; comp < NCOMP; comp++) {
    ContractYTranspose2d<NCOMP, P1d, Q1d>(data, r_U + comp + 0*NCOMP, c_B, r_t);
    ContractXTranspose2d<NCOMP, P1d, Q1d>(data, r_t, c_G, r_V + comp);
    ContractYTranspose2d<NCOMP, P1d, Q1d>(data, r_U + comp + 1*NCOMP, c_G, r_t);
    ContractXTransposeAdd2d<NCOMP, P1d, Q1d>(data, r_t, c_B, r_V + comp);
  }
}

//------------------------------------------------------------------------------
// 3D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// L-vector -> E-vector, offsets provided
//------------------------------------------------------------------------------
template <int NCOMP, int COMPSTRIDE, int P1d>
inline __device__ void readDofsOffset3d(BackendData &data, const CeedInt nnodes, const CeedInt elem, const CeedInt *__restrict__ indices, const CeedScalar *__restrict__ d_u, CeedScalar *r_u) {
  if (data.tidx < P1d && data.tidy < P1d)
    for (CeedInt z = 0; z < P1d; ++z) {
      const CeedInt node = data.tidx + data.tidy*P1d + z*P1d*P1d;
      const CeedInt ind = indices[node + elem * P1d*P1d*P1d];
      for (CeedInt comp = 0; comp < NCOMP; ++comp)
        r_u[z+comp*P1d] = d_u[ind + COMPSTRIDE * comp];
    }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, strided
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline __device__ void readDofsStrided3d(BackendData &data, const CeedInt elem, const CeedScalar *__restrict__ d_u, CeedScalar *r_u) {
  if (data.tidx < P1d && data.tidy < P1d)
    for (CeedInt z = 0; z < P1d; ++z) {
      const CeedInt node = data.tidx + data.tidy*P1d + z*P1d*P1d;
      const CeedInt ind = node * STRIDES_NODE + elem * STRIDES_ELEM;
      for (CeedInt comp = 0; comp < NCOMP; ++comp)
        r_u[z+comp*P1d] = d_u[ind + comp * STRIDES_COMP];
    }
}

//------------------------------------------------------------------------------
// E-vector -> Q-vector, offests provided
//------------------------------------------------------------------------------
template <int NCOMP, int COMPSTRIDE, int Q1d>
inline __device__ void readSliceQuadsOffset3d(BackendData &data, const CeedInt nquads, const CeedInt elem, const CeedInt q, const CeedInt *__restrict__ indices, const CeedScalar *__restrict__ d_u, CeedScalar *r_u) {
  if (data.tidx < Q1d && data.tidy < Q1d) {
    const CeedInt node = data.tidx + data.tidy*Q1d + q*Q1d*Q1d;
    const CeedInt ind = indices[node + elem * Q1d*Q1d*Q1d];;
    for (CeedInt comp = 0; comp < NCOMP; ++comp)
      r_u[comp] = d_u[ind + COMPSTRIDE * comp];
  }
}

//------------------------------------------------------------------------------
// E-vector -> Q-vector, strided
//------------------------------------------------------------------------------
template <int NCOMP, int Q1d, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline __device__ void readSliceQuadsStrided3d(BackendData &data, const CeedInt elem, const CeedInt q, const CeedScalar *__restrict__ d_u, CeedScalar *r_u) {
  if (data.tidx < Q1d && data.tidy < Q1d) {
    const CeedInt node = data.tidx + data.tidy*Q1d + q*Q1d*Q1d;
    const CeedInt ind = node * STRIDES_NODE + elem * STRIDES_ELEM;
    for (CeedInt comp = 0; comp < NCOMP; ++comp)
      r_u[comp] = d_u[ind + comp * STRIDES_COMP];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, offsets provided
//------------------------------------------------------------------------------
template <int NCOMP, int COMPSTRIDE, int P1d>
inline __device__ void writeDofsOffset3d(BackendData &data, const CeedInt nnodes, const CeedInt elem, const CeedInt *__restrict__ indices, const CeedScalar *r_v, CeedScalar *d_v) {
  if (data.tidx < P1d && data.tidy < P1d)
    for (CeedInt z = 0; z < P1d; ++z) {
      const CeedInt node = data.tidx + data.tidy*P1d + z*P1d*P1d;
      const CeedInt ind = indices[node + elem * P1d*P1d*P1d];
      for (CeedInt comp = 0; comp < NCOMP; ++comp)
        atomicAdd(&d_v[ind + COMPSTRIDE * comp], r_v[z+comp*P1d]);
    }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, strided
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline __device__ void writeDofsStrided3d(BackendData &data, const CeedInt elem, const CeedScalar *r_v, CeedScalar *d_v) {
  if (data.tidx < P1d && data.tidy < P1d)
    for (CeedInt z = 0; z < P1d; ++z) {
      const CeedInt node = data.tidx + data.tidy*P1d + z*P1d*P1d;
      const CeedInt ind = node * STRIDES_NODE + elem * STRIDES_ELEM;
      for (CeedInt comp = 0; comp < NCOMP; ++comp)
        d_v[ind + comp * STRIDES_COMP] += r_v[z+comp*P1d];
    }
}

//------------------------------------------------------------------------------
// 3D tensor contract x
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractX3d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  CeedScalar r_B[P1d];
  for (CeedInt i = 0; i < P1d; ++i)
    r_B[i] = B[i + data.tidx*P1d];

  for (CeedInt k = 0; k < P1d; ++k) {
    data.slice[data.tidx+data.tidy*T1d] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (data.tidx < Q1d && data.tidy < P1d)
      for (CeedInt i = 0; i < P1d; ++i)
        V[k] += r_B[i] * data.slice[i + data.tidy*T1d]; // Contract x direction
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D tensor contract y
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractY3d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  CeedScalar r_B[P1d];
  for (CeedInt i = 0; i < P1d; ++i)
    r_B[i] = B[i + data.tidy*P1d];

  for (CeedInt k = 0; k < P1d; ++k) {
    data.slice[data.tidx+data.tidy*T1d] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (data.tidx < Q1d && data.tidy < Q1d)
      for (CeedInt i = 0; i < P1d; ++i)
        V[k] += r_B[i] * data.slice[data.tidx + i*T1d]; // Contract y direction
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D tensor contract z
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractZ3d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  for (CeedInt k = 0; k < Q1d; ++k) {
    V[k] = 0.0;
    if (data.tidx < Q1d && data.tidy < Q1d)
      for (CeedInt i = 0; i < P1d; ++i)
        V[k] += B[i + k*P1d] * U[i]; // Contract z direction
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract z
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractTransposeZ3d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  for (CeedInt k = 0; k < P1d; ++k) {
    V[k] = 0.0;
    if (data.tidx < Q1d && data.tidy < Q1d)
      for (CeedInt i = 0; i < Q1d; ++i)
        V[k] += B[k + i*P1d] * U[i]; // Contract z direction
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract y
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractTransposeY3d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  CeedScalar r_B[Q1d];
  for (CeedInt i = 0; i < Q1d; ++i)
    r_B[i] = B[data.tidy + i*P1d];

  for (CeedInt k = 0; k < P1d; ++k) {
    data.slice[data.tidx+data.tidy*T1d] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (data.tidx < Q1d && data.tidy < P1d)
      for (CeedInt i = 0; i < Q1d; ++i)
        V[k] += r_B[i] * data.slice[data.tidx + i*T1d]; // Contract y direction
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract add y
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractTransposeAddY3d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  CeedScalar r_B[Q1d];
  for (CeedInt i = 0; i < Q1d; ++i)
    r_B[i] = B[data.tidy + i*P1d];

  for (CeedInt k = 0; k < P1d; ++k) {
    data.slice[data.tidx+data.tidy*T1d] = U[k];
    __syncthreads();
    if (data.tidx < Q1d && data.tidy < P1d)
      for (CeedInt i = 0; i < Q1d; ++i)
        V[k] += r_B[i] * data.slice[data.tidx + i*T1d]; // Contract y direction
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract x
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractTransposeX3d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  CeedScalar r_B[Q1d];
  for (CeedInt i = 0; i < Q1d; ++i)
    r_B[i] = B[data.tidx + i*P1d];

  for (CeedInt k = 0; k < P1d; ++k) {
    data.slice[data.tidx+data.tidy*T1d] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (data.tidx < P1d && data.tidy < P1d)
      for (CeedInt i = 0; i < Q1d; ++i)
        V[k] += r_B[i] * data.slice[i + data.tidy*T1d]; // Contract x direction
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract add x
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractTransposeAddX3d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  CeedScalar r_B[Q1d];
  for (CeedInt i = 0; i < Q1d; ++i)
    r_B[i] = B[data.tidx + i*P1d];

  for (CeedInt k = 0; k < P1d; ++k) {
    data.slice[data.tidx+data.tidy*T1d] = U[k];
    __syncthreads();
    if (data.tidx < P1d && data.tidy < P1d)
      for (CeedInt i = 0; i < Q1d; ++i)
        V[k] += r_B[i] * data.slice[i + data.tidy*T1d]; // Contract x direction
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D interpolate to quadrature points
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void interp3d(BackendData &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, CeedScalar *__restrict__ r_V) {
  CeedScalar r_t1[T1d];
  CeedScalar r_t2[T1d];
  for (CeedInt comp = 0; comp < NCOMP; comp++) {
    ContractX3d<NCOMP, P1d, Q1d>(data, r_U + comp*P1d, c_B, r_t1);
    ContractY3d<NCOMP, P1d, Q1d>(data, r_t1, c_B, r_t2);
    ContractZ3d<NCOMP, P1d, Q1d>(data, r_t2, c_B, r_V + comp*Q1d);
  }
}

//------------------------------------------------------------------------------
// 3D interpolate transpose
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void interpTranspose3d(BackendData &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, CeedScalar *__restrict__ r_V) {
  CeedScalar r_t1[T1d];
  CeedScalar r_t2[T1d];
  for (CeedInt comp = 0; comp < NCOMP; comp++) {
    ContractTransposeZ3d<NCOMP, P1d, Q1d>(data, r_U + comp*Q1d, c_B, r_t1);
    ContractTransposeY3d<NCOMP, P1d, Q1d>(data, r_t1, c_B, r_t2);
    ContractTransposeX3d<NCOMP, P1d, Q1d>(data, r_t2, c_B, r_V + comp*P1d);
  }
}

//------------------------------------------------------------------------------
// 3D derivatives at quadrature points
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void grad3d(BackendData &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G, CeedScalar *__restrict__ r_V) {
  CeedScalar r_t1[T1d];
  CeedScalar r_t2[T1d];
  for (CeedInt comp = 0; comp < NCOMP; comp++) {
    ContractX3d<NCOMP, P1d, Q1d>(data, r_U + comp*P1d, c_G, r_t1);
    ContractY3d<NCOMP, P1d, Q1d>(data, r_t1, c_B, r_t2);
    ContractZ3d<NCOMP, P1d, Q1d>(data, r_t2, c_B, r_V + comp*Q1d + 0*NCOMP*Q1d);
    ContractX3d<NCOMP, P1d, Q1d>(data, r_U + comp*P1d, c_B, r_t1);
    ContractY3d<NCOMP, P1d, Q1d>(data, r_t1, c_G, r_t2);
    ContractZ3d<NCOMP, P1d, Q1d>(data, r_t2, c_B, r_V + comp*Q1d + 1*NCOMP*Q1d);
    ContractX3d<NCOMP, P1d, Q1d>(data, r_U + comp*P1d, c_B, r_t1);
    ContractY3d<NCOMP, P1d, Q1d>(data, r_t1, c_B, r_t2);
    ContractZ3d<NCOMP, P1d, Q1d>(data, r_t2, c_G, r_V + comp*Q1d + 2*NCOMP*Q1d);
  }
}

//------------------------------------------------------------------------------
// 3D derivatives transpose
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline __device__ void gradTranspose3d(BackendData &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G, CeedScalar *__restrict__ r_V) {
  CeedScalar r_t1[T1d];
  CeedScalar r_t2[T1d];
  for (CeedInt comp = 0; comp < NCOMP; comp++) {
    ContractTransposeZ3d<NCOMP, P1d, Q1d>(data, r_U + comp*Q1d + 0*NCOMP*Q1d, c_B, r_t1);
    ContractTransposeY3d<NCOMP, P1d, Q1d>(data, r_t1, c_B, r_t2);
    ContractTransposeX3d<NCOMP, P1d, Q1d>(data, r_t2, c_G, r_V + comp*P1d);
    ContractTransposeZ3d<NCOMP, P1d, Q1d>(data, r_U + comp*Q1d + 1*NCOMP*Q1d, c_B, r_t1);
    ContractTransposeY3d<NCOMP, P1d, Q1d>(data, r_t1, c_G, r_t2);
    ContractTransposeAddX3d<NCOMP,P1d, Q1d>(data, r_t2, c_B, r_V + comp*P1d);
    ContractTransposeZ3d<NCOMP, P1d, Q1d>(data, r_U + comp*Q1d + 2*NCOMP*Q1d, c_G, r_t1);
    ContractTransposeY3d<NCOMP, P1d, Q1d>(data, r_t1, c_B, r_t2);
    ContractTransposeAddX3d<NCOMP, P1d, Q1d>(data, r_t2, c_B, r_V + comp*P1d);
  }
}

//------------------------------------------------------------------------------
// 3D collocated derivatives computation
//------------------------------------------------------------------------------
template <int NCOMP, int Q1d>
inline __device__ void gradCollo3d(BackendData &data, const CeedInt q, const CeedScalar *__restrict__ r_U, const CeedScalar *c_G, CeedScalar *__restrict__ r_V) {
  if (data.tidx < Q1d && data.tidy < Q1d) {
    for (CeedInt comp = 0; comp < NCOMP; ++comp) {
      data.slice[data.tidx + data.tidy*T1d] = r_U[q + comp*Q1d];
      __syncthreads();
      // X derivative
      r_V[comp+0*NCOMP] = 0.0;
      for (CeedInt i = 0; i < Q1d; ++i)
        r_V[comp+0*NCOMP] += c_G[i + data.tidx*Q1d] * data.slice[i + data.tidy*T1d]; // Contract x direction (X derivative)
      // Y derivative
      r_V[comp+1*NCOMP] = 0.0;
      for (CeedInt i = 0; i < Q1d; ++i)
        r_V[comp+1*NCOMP] += c_G[i + data.tidy*Q1d] * data.slice[data.tidx + i*T1d]; // Contract y direction (Y derivative)
      // Z derivative
      r_V[comp+2*NCOMP] = 0.0;
      for (CeedInt i = 0; i < Q1d; ++i)
        r_V[comp+2*NCOMP] += c_G[i + q*Q1d] * r_U[i + comp*Q1d]; // Contract z direction (Z derivative)
      __syncthreads();
    }
  }
}

//------------------------------------------------------------------------------
// 3D collocated derivatives transpose
//------------------------------------------------------------------------------
template <int NCOMP, int Q1d>
inline __device__ void gradColloTranspose3d(BackendData &data, const CeedInt q, const CeedScalar *__restrict__ r_U, const CeedScalar *c_G, CeedScalar *__restrict__ r_V) {
  if (data.tidx < Q1d && data.tidy < Q1d) {
    for (CeedInt comp = 0; comp < NCOMP; ++comp) {
      // X derivative
      data.slice[data.tidx + data.tidy*T1d] = r_U[comp + 0*NCOMP];
      __syncthreads();
      for (CeedInt i = 0; i < Q1d; ++i)
        r_V[q+comp*Q1d] += c_G[data.tidx + i*Q1d] * data.slice[i + data.tidy*T1d]; // Contract x direction (X derivative)
      __syncthreads();
      // Y derivative
      data.slice[data.tidx + data.tidy*T1d] = r_U[comp + 1*NCOMP];
      __syncthreads();
      for (CeedInt i = 0; i < Q1d; ++i)
        r_V[q+comp*Q1d] += c_G[data.tidy + i*Q1d] * data.slice[data.tidx + i*T1d]; // Contract y direction (Y derivative)
      __syncthreads();
      // Z derivative
      for (CeedInt i = 0; i < Q1d; ++i)
        r_V[i+comp*Q1d] += c_G[i + q*Q1d] * r_U[comp + 2*NCOMP]; // PARTIAL contract z direction (Z derivative)
    }
  }
}

//------------------------------------------------------------------------------
// 1D quadrature weights
//------------------------------------------------------------------------------
template <int Q1d>
inline __device__ void weight1d(BackendData &data, const CeedScalar *__restrict__ qweight1d, CeedScalar *w) {
  *w = (data.tidx < Q1d) ? qweight1d[data.tidx] : 0.0;
}

//------------------------------------------------------------------------------
// 2D quadrature weights
//------------------------------------------------------------------------------
template <int Q1d>
inline __device__ void weight2d(BackendData &data, const CeedScalar *__restrict__ qweight1d, CeedScalar *w) {
  *w = (data.tidx < Q1d && data.tidy < Q1d) ?
        qweight1d[data.tidx]*qweight1d[data.tidy] : 0.0;
}

//------------------------------------------------------------------------------
// 3D quadrature weights
//------------------------------------------------------------------------------
template <int Q1d>
inline __device__ void weight3d(BackendData &data, const CeedScalar *__restrict__ qweight1d, CeedScalar *w) {
  const bool quad = (data.tidx < Q1d && data.tidy < Q1d);
  const CeedScalar pw = quad ? qweight1d[data.tidx]*qweight1d[data.tidy] : 0.0;
  for (CeedInt z = 0; z < Q1d; ++z)
    w[z] = quad ? pw*qweight1d[z] : 0.0;
}

);
//------------------------------------------------------------------------------
// Build singe operator kernel
//------------------------------------------------------------------------------
extern "C" int CeedCudaGenOperatorBuild(CeedOperator op) {

  using std::ostringstream;
  using std::string;
  int ierr;
  bool setupdone;
  ierr = CeedOperatorIsSetupDone(op, &setupdone); CeedChkBackend(ierr);
  if (setupdone) return CEED_ERROR_SUCCESS;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  CeedOperator_Cuda_gen *data;
  ierr = CeedOperatorGetData(op, &data); CeedChkBackend(ierr);
  CeedQFunction qf;
  CeedQFunction_Cuda_gen *qf_data;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChkBackend(ierr);
  ierr = CeedQFunctionGetData(qf, &qf_data); CeedChkBackend(ierr);
  CeedSize lsize;
  CeedInt Q, P1d = 0, Q1d = 0, numelements, elemsize, numinputfields,
          numoutputfields, ncomp, dim = 0;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChkBackend(ierr);
  ierr = CeedOperatorGetNumElements(op, &numelements); CeedChkBackend(ierr);
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &numinputfields, &opinputfields, &numoutputfields, &opoutputfields);
  CeedChkBackend(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, NULL, &qfinputfields, NULL, &qfoutputfields);
  CeedChkBackend(ierr);
  CeedEvalMode emode;
  CeedBasis basis;
  CeedBasis_Cuda_shared *basis_data;
  CeedElemRestriction Erestrict;
  CeedElemRestriction_Cuda *restr_data;

  // Check for restriction only identity operator
  bool is_identity_qf;
  ierr = CeedQFunctionIsIdentity(qf, &is_identity_qf); CeedChkBackend(ierr);
  if (is_identity_qf) {
    CeedEvalMode emodein, emodeout;
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[0], &emodein);  CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[0], &emodeout);  CeedChkBackend(ierr);
    if (emodein == CEED_EVAL_NONE && emodeout == CEED_EVAL_NONE)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "Backend does not implement restriction only identity operators");
    // LCOV_EXCL_STOP
  }

  ostringstream code;
  string devFunctions(deviceFunctions);

  // Add atomicAdd function for old NVidia architectures
  struct cudaDeviceProp prop;
  Ceed_Cuda *ceed_data;
  ierr = CeedGetData(ceed, &ceed_data); CeedChkBackend(ierr); CeedChkBackend(ierr);
  ierr = cudaGetDeviceProperties(&prop, ceed_data->device_id); CeedChkBackend(ierr);
  if ((prop.major < 6) && (CEED_SCALAR_TYPE != CEED_SCALAR_FP32)){
    code << atomicAdd;
  }

  code << devFunctions;

  string qFunction(qf_data->qFunctionSource);
  string qFunctionName(qf_data->qFunctionName);
  string oper;
  oper = "CeedKernel_Cuda_gen_" + qFunctionName;

  code << "\n#define CEED_QFUNCTION(name) inline __device__ int name\n";
  code << "#define CEED_QFUNCTION_HELPER inline __device__\n";
  code << "#define CEED_HOSTDEVICE __host__ __device__\n";
  code << "#define CeedPragmaSIMD\n";
  code << "#define CEED_ERROR_SUCCESS 0\n\n";

  // Find dim and Q1d
  bool useCollograd = true;
  data->maxP1d = 0;
  for (CeedInt i = 0; i < numinputfields; i++) {
    ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChkBackend(ierr);
    if (basis != CEED_BASIS_COLLOCATED) {
      ierr = CeedBasisGetData(basis, &basis_data); CeedChkBackend(ierr);
      ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
      CeedChkBackend(ierr);

      // Check for collocated gradient
      useCollograd = useCollograd && basis_data->d_collo_grad_1d;

      // Collect dim and Q1d
      ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
      bool isTensor;
      ierr = CeedBasisIsTensor(basis, &isTensor); CeedChkBackend(ierr);
      if (isTensor) {
        ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChkBackend(ierr);
        ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChkBackend(ierr);
        if (P1d>data->maxP1d) data->maxP1d = P1d;
      } else {
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_BACKEND, "Backend does not implement operators with non-tensor basis");
        // LCOV_EXCL_STOP
        }
    }
  }
  // Check output bases for Q1d, dim as well
  //   The only imput basis might be CEED_BASIS_COLLOCATED
  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis); CeedChkBackend(ierr);

    if (basis != CEED_BASIS_COLLOCATED) {
      ierr = CeedBasisGetData(basis, &basis_data); CeedChkBackend(ierr);
      ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
      CeedChkBackend(ierr);

      // Collect dim and Q1d
      ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
      bool isTensor;
      ierr = CeedBasisIsTensor(basis, &isTensor); CeedChkBackend(ierr);
      if (isTensor) {
        ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChkBackend(ierr);
      } else {
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_BACKEND, "Backend does not implement operators with non-tensor basis");
        // LCOV_EXCL_STOP
        }

      // Check for collocated gradient
      useCollograd = useCollograd && basis_data->d_collo_grad_1d;
    }
  }
  data->dim = dim;
  data->Q1d = Q1d;

  // Define CEED_Q_VLA
  if (dim != 3 || useCollograd) {
    code << "\n#define CEED_Q_VLA 1\n\n";
  } else {
    code << "\n#define CEED_Q_VLA "<<Q1d<<"\n\n";
  }

  code << qFunction;

  // Setup
  code << "\n// -----------------------------------------------------------------------------\n";
  code << "\nextern \"C\" __global__ void "<<oper<<"(CeedInt nelem, void* ctx, CudaFieldsInt indices, CudaFields fields, CudaFields B, CudaFields G, CeedScalar* W) {\n";
  for (CeedInt i = 0; i < numinputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChkBackend(ierr);
    if (emode != CEED_EVAL_WEIGHT) { // Skip CEED_EVAL_WEIGHT
      code << "  const CeedScalar* d_u" <<i<<" = fields.in["<<i<<"];\n";
    }
  }

  for (CeedInt i = 0; i < numoutputfields; i++) {
    code << "  CeedScalar* d_v"<<i<<" = fields.out["<<i<<"];\n";
  }

  code << "  const CeedInt Dim = "<<dim<<";\n";
  code << "  const CeedInt Q1d = "<<Q1d<<";\n";

  code << "  extern __shared__ CeedScalar slice[];\n";
  code << "  BackendData data;\n";
  code << "  data.tidx = threadIdx.x;\n";
  code << "  data.tidy = threadIdx.y;\n";
  code << "  data.tidz = threadIdx.z;\n";
  code << "  data.tid  = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y*blockDim.x;\n";
  code << "  data.slice = slice+data.tidz*T1d"<<(dim>1?"*T1d":"")<<";\n";

  code << "\n  // -- Input field constants and basis data --\n";
  //Initialize constants, and matrices B and G
  for (CeedInt i = 0; i < numinputfields; i++) {
    code << "  // ---- Input field "<<i<<" ----\n";
    // Get elemsize, emode, ncomp
    ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetNumComponents(Erestrict, &ncomp);
    CeedChkBackend(ierr);

    // Set field constants
    if (emode != CEED_EVAL_WEIGHT) {
      ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChkBackend(ierr);
      if (basis != CEED_BASIS_COLLOCATED) {
        ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChkBackend(ierr);
        code << "  const CeedInt P_in_"<<i<<" = "<<P1d<<";\n";
      } else {
        code << "  const CeedInt P_in_"<<i<<" = "<<Q1d<<";\n";
      }
      code << "  const CeedInt ncomp_in_"<<i<<" = "<<ncomp<<";\n";
    }

    // Load basis data
    code << "  // EvalMode: "<<CeedEvalModes[emode]<<"\n";
    switch (emode) {
    case CEED_EVAL_NONE:
      break;
    case CEED_EVAL_INTERP:
      ierr = CeedBasisGetData(basis, &basis_data); CeedChkBackend(ierr);
      data->B.in[i] = basis_data->d_interp_1d;
      code << "  __shared__ CeedScalar s_B_in_"<<i<<"["<<P1d*Q1d<<"];\n";
      code << "  loadMatrix<P_in_"<<i<<",Q1d>(data, B.in["<<i<<"], s_B_in_"<<i<<");\n";
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedBasisGetData(basis, &basis_data); CeedChkBackend(ierr);
      data->B.in[i] = basis_data->d_interp_1d;
      code << "  __shared__ CeedScalar s_B_in_"<<i<<"["<<P1d*Q1d<<"];\n";
      code << "  loadMatrix<P_in_"<<i<<",Q1d>(data, B.in["<<i<<"], s_B_in_"<<i<<");\n";
      if (useCollograd) {
        data->G.in[i] = basis_data->d_collo_grad_1d;
        code << "  __shared__ CeedScalar s_G_in_"<<i<<"["<<Q1d*Q1d<<"];\n";
        code << "  loadMatrix<Q1d,Q1d>(data, G.in["<<i<<"], s_G_in_"<<i<<");\n";
      } else {
        data->G.in[i] = basis_data->d_grad_1d;
        code << "  __shared__ CeedScalar s_G_in_"<<i<<"["<<P1d*Q1d<<"];\n";
        code << "  loadMatrix<P_in_"<<i<<",Q1d>(data, G.in["<<i<<"], s_G_in_"<<i<<");\n";
      }
      break;
    case CEED_EVAL_WEIGHT:
      break; // No action
    case CEED_EVAL_DIV:
      break; // TODO: Not implemented
    case CEED_EVAL_CURL:
      break; // TODO: Not implemented
    }
  }

  code << "\n  // -- Output field constants and basis data --\n";
  for (CeedInt i = 0; i < numoutputfields; i++) {
    code << "  // ---- Output field "<<i<<" ----\n";
    // Get elemsize, emode, ncomp
    ierr = CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetNumComponents(Erestrict, &ncomp);
    CeedChkBackend(ierr);

    // Set field constants
    ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis); CeedChkBackend(ierr);
    if (basis != CEED_BASIS_COLLOCATED) {
      ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChkBackend(ierr);
      code << "  const CeedInt P_out_"<<i<<" = "<<P1d<<";\n";
    } else {
      code << "  const CeedInt P_out_"<<i<<" = "<<Q1d<<";\n";
    }
    code << "  const CeedInt ncomp_out_"<<i<<" = "<<ncomp<<";\n";

    // Load basis data
    code << "  // EvalMode: "<<CeedEvalModes[emode]<<"\n";
    switch (emode) {
    case CEED_EVAL_NONE:
      break; // No action
    case CEED_EVAL_INTERP:
      ierr = CeedBasisGetData(basis, &basis_data); CeedChkBackend(ierr);
      data->B.out[i] = basis_data->d_interp_1d;
      code << "  __shared__ CeedScalar s_B_out_"<<i<<"["<<P1d*Q1d<<"];\n";
      code << "  loadMatrix<P_out_"<<i<<",Q1d>(data, B.out["<<i<<"], s_B_out_"<<i<<");\n";
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedBasisGetData(basis, &basis_data); CeedChkBackend(ierr);
      data->B.out[i] = basis_data->d_interp_1d;
      code << "  __shared__ CeedScalar s_B_out_"<<i<<"["<<P1d*Q1d<<"];\n";
      code << "  loadMatrix<P_out_"<<i<<",Q1d>(data, B.out["<<i<<"], s_B_out_"<<i<<");\n";
      if (useCollograd) {
        data->G.out[i] = basis_data->d_collo_grad_1d;
        code << "  __shared__ CeedScalar s_G_out_"<<i<<"["<<Q1d*Q1d<<"];\n";
        code << "  loadMatrix<Q1d,Q1d>(data, G.out["<<i<<"], s_G_out_"<<i<<");\n";
      } else {
        data->G.out[i] = basis_data->d_grad_1d;
        code << "  __shared__ CeedScalar s_G_out_"<<i<<"["<<P1d*Q1d<<"];\n";
        code << "  loadMatrix<P_out_"<<i<<",Q1d>(data, G.out["<<i<<"], s_G_out_"<<i<<");\n";
      }
      break;
    // LCOV_EXCL_START
    case CEED_EVAL_WEIGHT: {
      Ceed ceed;
      ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
      break; // Should not occur
    }
    case CEED_EVAL_DIV:
      break; // TODO: Not implemented
    case CEED_EVAL_CURL:
      break; // TODO: Not implemented
      // LCOV_EXCL_STOP
    }
  }
  code << "\n  // -- Element loop --\n";
  code << "  __syncthreads();\n";
  code << "  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < nelem; elem += gridDim.x*blockDim.z) {\n";
  // Input basis apply if needed
  // Generate the correct eval mode code for each input
  code << "    // -- Input field restrictions and basis actions --\n";
  for (CeedInt i = 0; i < numinputfields; i++) {
    code << "    // ---- Input field "<<i<<" ----\n";
    // Get elemsize, emode, ncomp
    ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetNumComponents(Erestrict, &ncomp);
    CeedChkBackend(ierr);

    // Restriction
    if (emode != CEED_EVAL_WEIGHT &&
        !((emode == CEED_EVAL_NONE) && useCollograd)) {
      code << "    CeedScalar r_u"<<i<<"[ncomp_in_"<<i<<"*P_in_"<<i<<"];\n";

      bool isStrided;
      ierr = CeedElemRestrictionIsStrided(Erestrict, &isStrided); CeedChkBackend(ierr);
      if (!isStrided) {
        ierr = CeedElemRestrictionGetLVectorSize(Erestrict, &lsize);
        CeedChkBackend(ierr);
        code << "    const CeedInt lsize_in_"<<i<<" = "<<lsize<<";\n";
        CeedInt compstride;
        ierr = CeedElemRestrictionGetCompStride(Erestrict, &compstride); CeedChkBackend(ierr);
        code << "    // CompStride: "<<compstride<<"\n";
        ierr = CeedElemRestrictionGetData(Erestrict, &restr_data); CeedChkBackend(ierr);
        data->indices.in[i] = restr_data->d_ind;
        code << "    readDofsOffset"<<dim<<"d<ncomp_in_"<<i<<", "<<compstride<<", P_in_"<<i<<">(data, lsize_in_"<<i<<", elem, indices.in["<<i<<"], d_u"<<i<<", r_u"<<i<<");\n";
      } else {
        bool backendstrides;
        ierr = CeedElemRestrictionHasBackendStrides(Erestrict, &backendstrides);
        CeedChkBackend(ierr);
        CeedInt nelem;
        ierr = CeedElemRestrictionGetNumElements(Erestrict, &nelem);
        CeedChkBackend(ierr);
        CeedInt strides[3] = {1, elemsize*nelem, elemsize};
        if (!backendstrides) {
          ierr = CeedElemRestrictionGetStrides(Erestrict, &strides);
          CeedChkBackend(ierr);
        }
        code << "    // Strides: {"<<strides[0]<<", "<<strides[1]<<", "<<strides[2]<<"}\n";
        code << "    readDofsStrided"<<dim<<"d<ncomp_in_"<<i<<",P_in_"<<i<<","<<strides[0]<<","<<strides[1]<<","<<strides[2]<<">(data, elem, d_u"<<i<<", r_u"<<i<<");\n";
      }
    }

    // Basis action
    code << "    // EvalMode: "<<CeedEvalModes[emode]<<"\n";
    switch (emode) {
    case CEED_EVAL_NONE:
      if (!useCollograd) {
        code << "    CeedScalar* r_t"<<i<<" = r_u"<<i<<";\n";
      }
      break;
    case CEED_EVAL_INTERP:
      code << "    CeedScalar r_t"<<i<<"[ncomp_in_"<<i<<"*Q1d];\n";
      code << "    interp"<<dim<<"d<ncomp_in_"<<i<<",P_in_"<<i<<",Q1d>(data, r_u"<<i<<", s_B_in_"<<i<<", r_t"<<i<<");\n";
      break;
    case CEED_EVAL_GRAD:
      if (useCollograd) {
        code << "    CeedScalar r_t"<<i<<"[ncomp_in_"<<i<<"*Q1d];\n";
        code << "    interp"<<dim<<"d<ncomp_in_"<<i<<",P_in_"<<i<<",Q1d>(data, r_u"<<i<<", s_B_in_"<<i<<", r_t"<<i<<");\n";
      } else {
        code << "    CeedScalar r_t"<<i<<"[ncomp_in_"<<i<<"*Dim*Q1d];\n";
        code << "    grad"<<dim<<"d<ncomp_in_"<<i<<",P_in_"<<i<<",Q1d>(data, r_u"<<i<<", s_B_in_"<<i<<", s_G_in_"<<i<<", r_t"<<i<<");\n";
      }
      break;
    case CEED_EVAL_WEIGHT:
      code << "    CeedScalar r_t"<<i<<"[Q1d];\n";
      ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChkBackend(ierr);
      ierr = CeedBasisGetData(basis, &basis_data); CeedChkBackend(ierr);
      data->W = basis_data->d_q_weight_1d;
      code << "    weight"<<dim<<"d<Q1d>(data, W, r_t"<<i<<");\n";
      break; // No action
    case CEED_EVAL_DIV:
      break; // TODO: Not implemented
    case CEED_EVAL_CURL:
      break; // TODO: Not implemented
    }
  }

  // Q function
  code << "\n    // -- Output field setup --\n";
  for (CeedInt i = 0; i < numoutputfields; i++) {
      code << "\n    // ---- Output field "<<i<<" ----\n";
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChkBackend(ierr);
    if (emode==CEED_EVAL_GRAD)
    {
      if (useCollograd) {
        //Accumulator for gradient slices
        code << "    CeedScalar r_tt"<<i<<"[ncomp_out_"<<i<<"*Q1d];\n";
        code << "    for (CeedInt i = 0; i < ncomp_out_"<<i<<"; ++i) {\n";
        code << "      for (CeedInt j = 0; j < Q1d; ++j) {\n";
        code << "        r_tt"<<i<<"[j + i*Q1d] = 0.0;\n";
        code << "      }\n";
        code << "    }\n";
      } else {
        code << "    CeedScalar r_tt"<<i<<"[ncomp_out_"<<i<<"*Dim*Q1d];\n";
      }
    }
    if (emode==CEED_EVAL_NONE || emode==CEED_EVAL_INTERP)
    {
      code << "    CeedScalar r_tt"<<i<<"[ncomp_out_"<<i<<"*Q1d];\n";
    }
  }
  // We treat quadrature points per slice in 3d to save registers
  if (useCollograd) {
    code << "\n    // Note: Collocated Gradient\n";
    code << "#pragma unroll\n";
    code << "    for (CeedInt q=0; q<Q1d; q++) {\n";
    code << "      // -- Input fields --\n";
    for (CeedInt i = 0; i < numinputfields; i++) {
      code << "      // ---- Input field "<<i<<" ----\n";
      // Get elemsize, emode, ncomp
      ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
      CeedChkBackend(ierr);
      // Basis action
      code << "      // EvalMode: "<<CeedEvalModes[emode]<<"\n";
      switch (emode) {
      case CEED_EVAL_NONE:
        code << "      CeedScalar r_q"<<i<<"[ncomp_in_"<<i<<"];\n";

        bool isStrided;
        ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict); CeedChkBackend(ierr);
        ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize); CeedChkBackend(ierr);
        ierr = CeedElemRestrictionIsStrided(Erestrict, &isStrided); CeedChkBackend(ierr);
        if (!isStrided) {
          ierr = CeedElemRestrictionGetLVectorSize(Erestrict, &lsize);
          CeedChkBackend(ierr);
          code << "      const CeedInt lsize_in_"<<i<<" = "<<lsize<<";\n";
          CeedInt compstride;
          ierr = CeedElemRestrictionGetCompStride(Erestrict, &compstride); CeedChkBackend(ierr);
          code << "      // CompStride: "<<compstride<<"\n";
          ierr = CeedElemRestrictionGetData(Erestrict, &restr_data); CeedChkBackend(ierr);
          data->indices.in[i] = restr_data->d_ind;
          code << "      readSliceQuadsOffset"<<"3d<ncomp_in_"<<i<<", "<<compstride<<", Q1d>(data, lsize_in_"<<i<<", elem, q, indices.in["<<i<<"], d_u"<<i<<", r_q"<<i<<");\n";
        } else {
          bool backendstrides;
          ierr = CeedElemRestrictionHasBackendStrides(Erestrict, &backendstrides);
          CeedChkBackend(ierr);
          CeedInt nelem;
          ierr = CeedElemRestrictionGetNumElements(Erestrict, &nelem);
          CeedChkBackend(ierr);
          CeedInt strides[3] = {1, elemsize*nelem, elemsize};
          if (!backendstrides) {
            ierr = CeedElemRestrictionGetStrides(Erestrict, &strides);
            CeedChkBackend(ierr);
          }
          code << "      // Strides: {"<<strides[0]<<", "<<strides[1]<<", "<<strides[2]<<"}\n";
          code << "      readSliceQuadsStrided"<<"3d<ncomp_in_"<<i<<",Q1d"","<<strides[0]<<","<<strides[1]<<","<<strides[2]<<">(data, elem, q, d_u"<<i<<", r_q"<<i<<");\n";
        }
        break;
      case CEED_EVAL_INTERP:
        code << "      CeedScalar r_q"<<i<<"[ncomp_in_"<<i<<"];\n";
        code << "      for (CeedInt j = 0; j < ncomp_in_"<<i<<" ; ++j) {\n";
        code << "        r_q"<<i<<"[j] = r_t"<<i<<"[q + j*Q1d];\n";
        code << "      }\n";
        break;
      case CEED_EVAL_GRAD:
        code << "      CeedScalar r_q"<<i<<"[ncomp_in_"<<i<<"*Dim];\n";
        code << "      gradCollo3d<ncomp_in_"<<i<<",Q1d>(data, q, r_t"<<i<<", s_G_in_"<<i<<", r_q"<<i<<");\n";
        break;
      case CEED_EVAL_WEIGHT:
        code << "      CeedScalar r_q"<<i<<"[1];\n";
        code << "      r_q"<<i<<"[0] = r_t"<<i<<"[q];\n";
        break; // No action
      case CEED_EVAL_DIV:
        break; // TODO: Not implemented
      case CEED_EVAL_CURL:
        break; // TODO: Not implemented
      }
    }
    code << "\n      // -- Output fields --\n";
    for (CeedInt i = 0; i < numoutputfields; i++) {
      code << "      // ---- Output field "<<i<<" ----\n";
      ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
      CeedChkBackend(ierr);
      // Basis action
      switch (emode) {
      case CEED_EVAL_NONE:
        code << "      CeedScalar r_qq"<<i<<"[ncomp_out_"<<i<<"];\n";
        break; // No action
      case CEED_EVAL_INTERP:
        code << "      CeedScalar r_qq"<<i<<"[ncomp_out_"<<i<<"];\n";
        break;
      case CEED_EVAL_GRAD:
        code << "      CeedScalar r_qq"<<i<<"[ncomp_out_"<<i<<"*Dim];\n";
        break;
      case CEED_EVAL_WEIGHT:
        break; // Should not occur
      case CEED_EVAL_DIV:
        break; // TODO: Not implemented
      case CEED_EVAL_CURL:
        break; // TODO: Not implemented
      }
    }
  } else {
    code << "\n      // Note: No Collocated Gradient\n";
    code << "      // -- Input fields --\n";
    for (CeedInt i = 0; i < numinputfields; i++) {
      code << "      // ---- Input field "<<i<<" ----\n";
      code << "      CeedScalar* r_q"<<i<<" = r_t"<<i<<";\n";
    }
    code << "      // -- Output fields --\n";
    for (CeedInt i = 0; i < numoutputfields; i++) {
      code << "      // ---- Output field "<<i<<" ----\n";
      code << "      CeedScalar* r_qq"<<i<<" = r_tt"<<i<<";\n";
    }
  }
  code << "\n      // -- QFunction Inputs and outputs --\n";
  code << "      CeedScalar* in["<<numinputfields<<"];\n";
  for (CeedInt i = 0; i < numinputfields; i++) {
    code << "      // ---- Input field "<<i<<" ----\n";
    code << "      in["<<i<<"] = r_q"<<i<<";\n";
  }
  code << "      CeedScalar* out["<<numoutputfields<<"];\n";
  for (CeedInt i = 0; i < numoutputfields; i++) {
    code << "      // ---- Output field "<<i<<" ----\n";
    code << "      out["<<i<<"] = r_qq"<<i<<";\n";
  }
  code << "\n      // -- Apply QFunction --\n";
  code << "      "<<qFunctionName<<"(ctx, ";
  if (dim != 3 || useCollograd) {
    code << "1";
  } else {
    code << "Q1d";
  }
  code << ", in, out);\n";
  if (useCollograd) {
    code << "\n      // Note: Collocated Gradient\n";
    code << "      // -- Output fields --\n";
    for (CeedInt i = 0; i < numoutputfields; i++) {
      code << "      // ---- Output field "<<i<<" ----\n";
      ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
      CeedChkBackend(ierr);
      // Basis action
      code << "      // EvalMode: "<<CeedEvalModes[emode]<<"\n";
      switch (emode) {
      case CEED_EVAL_NONE:
        code << "      for (CeedInt j = 0; j < ncomp_out_"<<i<<" ; ++j) {\n";
        code << "        r_tt"<<i<<"[q + j*Q1d] = r_qq"<<i<<"[j];\n";
        code << "      }\n";
        break; // No action
      case CEED_EVAL_INTERP:
        code << "      for (CeedInt j = 0; j < ncomp_out_"<<i<<" ; ++j) {\n";
        code << "        r_tt"<<i<<"[q + j*Q1d] = r_qq"<<i<<"[j];\n";
        code << "      }\n";
        break;
      case CEED_EVAL_GRAD:
        code << "      gradColloTranspose3d<ncomp_out_"<<i<<",Q1d>(data, q, r_qq"<<i<<", s_G_out_"<<i<<", r_tt"<<i<<");\n";
        break;
      case CEED_EVAL_WEIGHT:
        break; // Should not occur
      case CEED_EVAL_DIV:
        break; // TODO: Not implemented
      case CEED_EVAL_CURL:
        break; // TODO: Not implemented
      }
    }
    code << "    }\n";
  }

  // Output basis apply if needed
  // Generate the correct eval mode code for each output
  code << "\n    // -- Output field basis action and restrictions --\n";
  for (CeedInt i = 0; i < numoutputfields; i++) {
    code << "    // ---- Output field "<<i<<" ----\n";
    // Get elemsize, emode, ncomp
    ierr = CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetNumComponents(Erestrict, &ncomp);
    CeedChkBackend(ierr);
    // Basis action
    code << "    // EvalMode: "<<CeedEvalModes[emode]<<"\n";
    switch (emode) {
    case CEED_EVAL_NONE:
      code << "    CeedScalar* r_v"<<i<<" = r_tt"<<i<<";\n";
      break; // No action
    case CEED_EVAL_INTERP:
      code << "    CeedScalar r_v"<<i<<"[ncomp_out_"<<i<<"*P_out_"<<i<<"];\n";
      code << "    interpTranspose"<<dim<<"d<ncomp_out_"<<i<<",P_out_"<<i<<",Q1d>(data, r_tt"<<i<<", s_B_out_"<<i<<", r_v"<<i<<");\n";
      break;
    case CEED_EVAL_GRAD:
      code << "    CeedScalar r_v"<<i<<"[ncomp_out_"<<i<<"*P_out_"<<i<<"];\n";
      if (useCollograd) {
        code << "    interpTranspose"<<dim<<"d<ncomp_out_"<<i<<",P_out_"<<i<<",Q1d>(data, r_tt"<<i<<", s_B_out_"<<i<<", r_v"<<i<<");\n";
      } else {
        code << "    gradTranspose"<<dim<<"d<ncomp_out_"<<i<<",P_out_"<<i<<",Q1d>(data, r_tt"<<i<<", s_B_out_"<<i<<", s_G_out_"<<i<<", r_v"<<i<<");\n";
      }
      break;
    // LCOV_EXCL_START
    case CEED_EVAL_WEIGHT: {
      Ceed ceed;
      ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
      break; // Should not occur
    }
    case CEED_EVAL_DIV:
      break; // TODO: Not implemented
    case CEED_EVAL_CURL:
      break; // TODO: Not implemented
      // LCOV_EXCL_STOP
    }
    // Restriction
      bool isStrided;
      ierr = CeedElemRestrictionIsStrided(Erestrict, &isStrided); CeedChkBackend(ierr);
    if (!isStrided) {
      ierr = CeedElemRestrictionGetLVectorSize(Erestrict, &lsize);
      CeedChkBackend(ierr);
      code << "    const CeedInt lsize_out_"<<i<<" = "<<lsize<<";\n";
      CeedInt compstride;
      ierr = CeedElemRestrictionGetCompStride(Erestrict, &compstride); CeedChkBackend(ierr);
      code << "    // CompStride: "<<compstride<<"\n";
      ierr = CeedElemRestrictionGetData(Erestrict, &restr_data); CeedChkBackend(ierr);
      data->indices.out[i] = restr_data->d_ind;
      code << "    writeDofsOffset"<<dim<<"d<ncomp_out_"<<i<<", "<<compstride<<", P_out_"<<i<<">(data, lsize_out_"<<i<<", elem, indices.out["<<i<<"], r_v"<<i<<", d_v"<<i<<");\n";
    } else {
      bool backendstrides;
      ierr = CeedElemRestrictionHasBackendStrides(Erestrict, &backendstrides);
      CeedChkBackend(ierr);
      CeedInt nelem;
      ierr = CeedElemRestrictionGetNumElements(Erestrict, &nelem);
      CeedChkBackend(ierr);
      CeedInt strides[3] = {1, elemsize*nelem, elemsize};
      if (!backendstrides) {
        ierr = CeedElemRestrictionGetStrides(Erestrict, &strides);
        CeedChkBackend(ierr);
      }
      code << "    // Strides: {"<<strides[0]<<", "<<strides[1]<<", "<<strides[2]<<"}\n";
      code << "    writeDofsStrided"<<dim<<"d<ncomp_out_"<<i<<",P_out_"<<i<<","<<strides[0]<<","<<strides[1]<<","<<strides[2]<<">(data, elem, r_v"<<i<<", d_v"<<i<<");\n";
    }
  }

  code << "  }\n";
  code << "}\n";
  code << "// -----------------------------------------------------------------------------\n\n";

  // View kernel for debugging
  CeedDebug256(ceed, 2, "Generated Operator Kernels:\n");
  CeedDebug(ceed, code.str().c_str());

  ierr = CeedCompileCuda(ceed, code.str().c_str(), &data->module, 1,
                         "T1d", CeedIntMax(Q1d, data->maxP1d));
  CeedChkBackend(ierr);
  ierr = CeedGetKernelCuda(ceed, data->module, oper.c_str(), &data->op);
  CeedChkBackend(ierr);

  ierr = CeedOperatorSetSetupDone(op); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
