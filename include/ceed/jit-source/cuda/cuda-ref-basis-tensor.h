// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CUDA tensor product basis
#include <ceed/types.h>

//------------------------------------------------------------------------------
// Tensor Basis Kernels
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Interp
//------------------------------------------------------------------------------
extern "C" __global__ void Interp(const CeedInt num_elem, const CeedInt is_transpose, const CeedScalar *__restrict__ interp_1d,
                                  const CeedScalar *__restrict__ u, CeedScalar *__restrict__ v) {
  const CeedInt i = threadIdx.x;

  __shared__ CeedScalar s_mem[BASIS_Q_1D * BASIS_P_1D + 2 * BASIS_BUF_LEN];
  CeedScalar           *s_interp_1d = s_mem;
  CeedScalar           *s_buffer_1  = s_mem + BASIS_Q_1D * BASIS_P_1D;
  CeedScalar           *s_buffer_2  = s_buffer_1 + BASIS_BUF_LEN;
  for (CeedInt k = i; k < BASIS_Q_1D * BASIS_P_1D; k += blockDim.x) {
    s_interp_1d[k] = interp_1d[k];
  }

  const CeedInt P             = is_transpose ? BASIS_Q_1D : BASIS_P_1D;
  const CeedInt Q             = is_transpose ? BASIS_P_1D : BASIS_Q_1D;
  const CeedInt stride_0      = is_transpose ? 1 : BASIS_P_1D;
  const CeedInt stride_1      = is_transpose ? BASIS_P_1D : 1;
  const CeedInt u_stride      = is_transpose ? BASIS_NUM_QPTS : BASIS_NUM_NODES;
  const CeedInt v_stride      = is_transpose ? BASIS_NUM_NODES : BASIS_NUM_QPTS;
  const CeedInt u_comp_stride = num_elem * (is_transpose ? BASIS_NUM_QPTS : BASIS_NUM_NODES);
  const CeedInt v_comp_stride = num_elem * (is_transpose ? BASIS_NUM_NODES : BASIS_NUM_QPTS);
  const CeedInt u_size        = is_transpose ? BASIS_NUM_QPTS : BASIS_NUM_NODES;

  // Apply basis element by element
  for (CeedInt elem = blockIdx.x; elem < num_elem; elem += gridDim.x) {
    for (CeedInt comp = 0; comp < BASIS_NUM_COMP; comp++) {
      const CeedScalar *cur_u = &u[elem * u_stride + comp * u_comp_stride];
      CeedScalar       *cur_v = &v[elem * v_stride + comp * v_comp_stride];
      CeedInt           pre   = u_size;
      CeedInt           post  = 1;

      for (CeedInt d = 0; d < BASIS_DIM; d++) {
        __syncthreads();
        // Update buffers used
        pre /= P;
        const CeedScalar *in       = d == 0 ? cur_u : (d % 2 ? s_buffer_2 : s_buffer_1);
        CeedScalar       *out      = d == BASIS_DIM - 1 ? cur_v : (d % 2 ? s_buffer_1 : s_buffer_2);
        const CeedInt     writeLen = pre * post * Q;

        // Contract along middle index
        for (CeedInt k = i; k < writeLen; k += blockDim.x) {
          const CeedInt c   = k % post;
          const CeedInt j   = (k / post) % Q;
          const CeedInt a   = k / (post * Q);
          CeedScalar    v_k = 0;

          for (CeedInt b = 0; b < P; b++) v_k += s_interp_1d[j * stride_0 + b * stride_1] * in[(a * P + b) * post + c];
          if (is_transpose && d == BASIS_DIM - 1) out[k] += v_k;
          else out[k] = v_k;
        }
        post *= Q;
      }
    }
  }
}

//------------------------------------------------------------------------------
// Grad
//------------------------------------------------------------------------------
extern "C" __global__ void Grad(const CeedInt num_elem, const CeedInt is_transpose, const CeedScalar *__restrict__ interp_1d,
                                const CeedScalar *__restrict__ grad_1d, const CeedScalar *__restrict__ u, CeedScalar *__restrict__ v) {
  const CeedInt i = threadIdx.x;

  __shared__ CeedScalar s_mem[2 * (BASIS_Q_1D * BASIS_P_1D + BASIS_BUF_LEN)];
  CeedScalar           *s_interp_1d = s_mem;
  CeedScalar           *s_grad_1d   = s_interp_1d + BASIS_Q_1D * BASIS_P_1D;
  CeedScalar           *s_buffer_1  = s_grad_1d + BASIS_Q_1D * BASIS_P_1D;
  CeedScalar           *s_buffer_2  = s_buffer_1 + BASIS_BUF_LEN;
  for (CeedInt k = i; k < BASIS_Q_1D * BASIS_P_1D; k += blockDim.x) {
    s_interp_1d[k] = interp_1d[k];
    s_grad_1d[k]   = grad_1d[k];
  }

  const CeedInt P             = is_transpose ? BASIS_Q_1D : BASIS_P_1D;
  const CeedInt Q             = is_transpose ? BASIS_P_1D : BASIS_Q_1D;
  const CeedInt stride_0      = is_transpose ? 1 : BASIS_P_1D;
  const CeedInt stride_1      = is_transpose ? BASIS_P_1D : 1;
  const CeedInt u_stride      = is_transpose ? BASIS_NUM_QPTS : BASIS_NUM_NODES;
  const CeedInt v_stride      = is_transpose ? BASIS_NUM_NODES : BASIS_NUM_QPTS;
  const CeedInt u_comp_stride = num_elem * (is_transpose ? BASIS_NUM_QPTS : BASIS_NUM_NODES);
  const CeedInt v_comp_stride = num_elem * (is_transpose ? BASIS_NUM_NODES : BASIS_NUM_QPTS);
  const CeedInt u_dim_stride  = is_transpose ? num_elem * BASIS_NUM_QPTS * BASIS_NUM_COMP : 0;
  const CeedInt v_dim_stride  = is_transpose ? 0 : num_elem * BASIS_NUM_QPTS * BASIS_NUM_COMP;

  // Apply basis element by element
  for (CeedInt elem = blockIdx.x; elem < num_elem; elem += gridDim.x) {
    for (CeedInt comp = 0; comp < BASIS_NUM_COMP; comp++) {
      // dim*dim contractions for grad
      for (CeedInt dim_1 = 0; dim_1 < BASIS_DIM; dim_1++) {
        CeedInt           pre   = is_transpose ? BASIS_NUM_QPTS : BASIS_NUM_NODES;
        CeedInt           post  = 1;
        const CeedScalar *cur_u = &u[elem * u_stride + dim_1 * u_dim_stride + comp * u_comp_stride];
        CeedScalar       *cur_v = &v[elem * v_stride + dim_1 * v_dim_stride + comp * v_comp_stride];

        for (CeedInt dim_2 = 0; dim_2 < BASIS_DIM; dim_2++) {
          __syncthreads();
          // Update buffers used
          pre /= P;
          const CeedScalar *op       = dim_1 == dim_2 ? s_grad_1d : s_interp_1d;
          const CeedScalar *in       = dim_2 == 0 ? cur_u : (dim_2 % 2 ? s_buffer_2 : s_buffer_1);
          CeedScalar       *out      = dim_2 == BASIS_DIM - 1 ? cur_v : (dim_2 % 2 ? s_buffer_1 : s_buffer_2);
          const CeedInt     writeLen = pre * post * Q;

          // Contract along middle index
          for (CeedInt k = i; k < writeLen; k += blockDim.x) {
            const CeedInt c   = k % post;
            const CeedInt j   = (k / post) % Q;
            const CeedInt a   = k / (post * Q);
            CeedScalar    v_k = 0;

            for (CeedInt b = 0; b < P; b++) v_k += op[j * stride_0 + b * stride_1] * in[(a * P + b) * post + c];
            if (is_transpose && dim_2 == BASIS_DIM - 1) out[k] += v_k;
            else out[k] = v_k;
          }
          post *= Q;
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
// 1D quadrature weights
//------------------------------------------------------------------------------
__device__ void Weight1d(const CeedInt num_elem, const CeedScalar *q_weight_1d, CeedScalar *w) {
  const CeedInt i = threadIdx.x;

  if (i < BASIS_Q_1D) {
    const size_t elem = blockIdx.x;

    if (elem < num_elem) w[elem * BASIS_Q_1D + i] = q_weight_1d[i];
  }
}

//------------------------------------------------------------------------------
// 2D quadrature weights
//------------------------------------------------------------------------------
__device__ void Weight2d(const CeedInt num_elem, const CeedScalar *q_weight_1d, CeedScalar *w) {
  const CeedInt i = threadIdx.x;
  const CeedInt j = threadIdx.y;

  if (i < BASIS_Q_1D && j < BASIS_Q_1D) {
    const size_t elem = blockIdx.x;

    if (elem < num_elem) {
      const size_t ind = (elem * BASIS_Q_1D + j) * BASIS_Q_1D + i;

      w[ind] = q_weight_1d[i] * q_weight_1d[j];
    }
  }
}

//------------------------------------------------------------------------------
// 3D quadrature weights
//------------------------------------------------------------------------------
__device__ void Weight3d(const CeedInt num_elem, const CeedScalar *q_weight_1d, CeedScalar *w) {
  const CeedInt i = threadIdx.x;
  const CeedInt j = threadIdx.y;

  if (i < BASIS_Q_1D && j < BASIS_Q_1D) {
    const size_t elem = blockIdx.x;

    if (elem < num_elem) {
      for (CeedInt k = 0; k < BASIS_Q_1D; k++) {
        const size_t ind = ((elem * BASIS_Q_1D + k) * BASIS_Q_1D + j) * BASIS_Q_1D + i;

        w[ind] = q_weight_1d[i] * q_weight_1d[j] * q_weight_1d[k];
      }
    }
  }
}

//------------------------------------------------------------------------------
// Quadrature weights
//------------------------------------------------------------------------------
extern "C" __global__ void Weight(const CeedInt num_elem, const CeedScalar *__restrict__ q_weight_1d, CeedScalar *__restrict__ v) {
  if (BASIS_DIM == 1) Weight1d(num_elem, q_weight_1d, v);
  else if (BASIS_DIM == 2) Weight2d(num_elem, q_weight_1d, v);
  else if (BASIS_DIM == 3) Weight3d(num_elem, q_weight_1d, v);
}
