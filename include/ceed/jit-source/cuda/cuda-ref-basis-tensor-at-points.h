// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CUDA tensor product basis with AtPoints evaluation
#include <ceed/types.h>

//------------------------------------------------------------------------------
// Chebyshev values
//------------------------------------------------------------------------------
template <int Q_1D>
inline __device__ void ChebyshevPolynomialsAtPoint(const CeedScalar x, CeedScalar *chebyshev_x) {
  chebyshev_x[0] = 1.0;
  chebyshev_x[1] = 2 * x;
  for (CeedInt i = 2; i < Q_1D; i++) chebyshev_x[i] = 2 * x * chebyshev_x[i - 1] - chebyshev_x[i - 2];
}

template <int Q_1D>
inline __device__ void ChebyshevDerivativeAtPoint(const CeedScalar x, CeedScalar *chebyshev_dx) {
  CeedScalar chebyshev_x[3];

  chebyshev_x[1]  = 1.0;
  chebyshev_x[2]  = 2 * x;
  chebyshev_dx[0] = 0.0;
  chebyshev_dx[1] = 2.0;
  for (CeedInt i = 2; i < Q_1D; i++) {
    chebyshev_x[(i + 1) % 3] = 2 * x * chebyshev_x[(i + 0) % 3] - chebyshev_x[(i + 2) % 3];
    chebyshev_dx[i]          = 2 * x * chebyshev_dx[i - 1] + 2 * chebyshev_x[(i + 0) % 3] - chebyshev_dx[i - 2];
  }
}

//------------------------------------------------------------------------------
// Tensor Basis Kernels AtPoints
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Interp
//------------------------------------------------------------------------------
extern "C" __global__ void InterpAtPoints(const CeedInt num_elem, const CeedInt is_transpose, const CeedScalar *__restrict__ chebyshev_interp_1d,
                                          const CeedInt *__restrict__ points_per_elem, const CeedScalar *__restrict__ coords,
                                          const CeedScalar *__restrict__ u, CeedScalar *__restrict__ v) {
  const CeedInt i = threadIdx.x;

  __shared__ CeedScalar s_mem[BASIS_Q_1D * BASIS_P_1D + 2 * BASIS_BUF_LEN + POINTS_BUFF_LEN * BASIS_Q_1D];
  CeedScalar           *s_chebyshev_interp_1d = s_mem;
  CeedScalar           *s_buffer_1            = s_mem + BASIS_Q_1D * BASIS_P_1D;
  CeedScalar           *s_buffer_2            = s_buffer_1 + BASIS_BUF_LEN;
  CeedScalar           *s_chebyshev_coeffs    = s_buffer_2 + BASIS_BUF_LEN;
  CeedScalar            chebyshev_x[BASIS_Q_1D], buffer_1[POINTS_BUFF_LEN], buffer_2[POINTS_BUFF_LEN];
  for (CeedInt k = i; k < BASIS_Q_1D * BASIS_P_1D; k += blockDim.x) {
    s_chebyshev_interp_1d[k] = chebyshev_interp_1d[k];
  }

  const CeedInt P             = BASIS_P_1D;
  const CeedInt Q             = BASIS_Q_1D;
  const CeedInt u_stride      = is_transpose ? BASIS_NUM_PTS : BASIS_NUM_NODES;
  const CeedInt v_stride      = is_transpose ? BASIS_NUM_NODES : BASIS_NUM_PTS;
  const CeedInt u_comp_stride = num_elem * (is_transpose ? BASIS_NUM_PTS : BASIS_NUM_NODES);
  const CeedInt v_comp_stride = num_elem * (is_transpose ? BASIS_NUM_NODES : BASIS_NUM_PTS);
  const CeedInt u_size        = is_transpose ? BASIS_NUM_PTS : BASIS_NUM_NODES;

  // Apply basis element by element
  if (is_transpose) {
    for (CeedInt elem = blockIdx.x; elem < num_elem; elem += gridDim.x) {
      for (CeedInt comp = 0; comp < BASIS_NUM_COMP; comp++) {
        const CeedScalar *cur_u = &u[elem * u_stride + comp * u_comp_stride];
        CeedScalar       *cur_v = &v[elem * v_stride + comp * v_comp_stride];
        CeedInt           pre   = 1;
        CeedInt           post  = 1;

        // Clear Chebyshev coeffs
        for (CeedInt k = i; k < BASIS_NUM_QPTS; k += blockDim.x) {
          s_chebyshev_coeffs[k] = 0.0;
        }

        // Map from point
        __syncthreads();
        for (CeedInt p = threadIdx.x; p < BASIS_NUM_PTS; p += blockDim.x) {
          if (p >= points_per_elem[elem]) continue;
          pre  = 1;
          post = 1;
          for (CeedInt d = 0; d < BASIS_DIM; d++) {
            // Update buffers used
            pre /= 1;
            const CeedScalar *in  = d == 0 ? (&cur_u[p]) : (d % 2 ? buffer_2 : buffer_1);
            CeedScalar       *out = d == BASIS_DIM - 1 ? s_chebyshev_coeffs : (d % 2 ? buffer_1 : buffer_2);

            // Build Chebyshev polynomial values
            ChebyshevPolynomialsAtPoint<BASIS_Q_1D>(coords[elem * u_stride + d * u_comp_stride + p], chebyshev_x);

            // Contract along middle index
            for (CeedInt a = 0; a < pre; a++) {
              for (CeedInt c = 0; c < post; c++) {
                if (d == BASIS_DIM - 1) {
                  for (CeedInt j = 0; j < Q; j++) atomicAdd(&out[(a * Q + (j + p) % Q) * post + c], chebyshev_x[(j + p) % Q] * in[a * post + c]);
                } else {
                  for (CeedInt j = 0; j < Q; j++) out[(a * Q + j) * post + c] = chebyshev_x[j] * in[a * post + c];
                }
              }
            }
            post *= Q;
          }
        }

        // Map from coefficients
        pre  = BASIS_NUM_QPTS;
        post = 1;
        for (CeedInt d = 0; d < BASIS_DIM; d++) {
          __syncthreads();
          // Update buffers used
          pre /= Q;
          const CeedScalar *in       = d == 0 ? s_chebyshev_coeffs : (d % 2 ? s_buffer_2 : s_buffer_1);
          CeedScalar       *out      = d == BASIS_DIM - 1 ? cur_v : (d % 2 ? s_buffer_1 : s_buffer_2);
          const CeedInt     writeLen = pre * post * P;

          // Contract along middle index
          for (CeedInt k = i; k < writeLen; k += blockDim.x) {
            const CeedInt c   = k % post;
            const CeedInt j   = (k / post) % P;
            const CeedInt a   = k / (post * P);
            CeedScalar    v_k = 0;

            for (CeedInt b = 0; b < Q; b++) v_k += s_chebyshev_interp_1d[j + b * BASIS_P_1D] * in[(a * Q + b) * post + c];
            if (d == BASIS_DIM - 1) out[k] += v_k;
            else out[k] = v_k;
          }
          post *= P;
        }
      }
    }
  } else {
    for (CeedInt elem = blockIdx.x; elem < num_elem; elem += gridDim.x) {
      for (CeedInt comp = 0; comp < BASIS_NUM_COMP; comp++) {
        const CeedScalar *cur_u = &u[elem * u_stride + comp * u_comp_stride];
        CeedScalar       *cur_v = &v[elem * v_stride + comp * v_comp_stride];
        CeedInt           pre   = u_size;
        CeedInt           post  = 1;

        // Map to coefficients
        for (CeedInt d = 0; d < BASIS_DIM; d++) {
          __syncthreads();
          // Update buffers used
          pre /= P;
          const CeedScalar *in       = d == 0 ? cur_u : (d % 2 ? s_buffer_2 : s_buffer_1);
          CeedScalar       *out      = d == BASIS_DIM - 1 ? s_chebyshev_coeffs : (d % 2 ? s_buffer_1 : s_buffer_2);
          const CeedInt     writeLen = pre * post * Q;

          // Contract along middle index
          for (CeedInt k = i; k < writeLen; k += blockDim.x) {
            const CeedInt c   = k % post;
            const CeedInt j   = (k / post) % Q;
            const CeedInt a   = k / (post * Q);
            CeedScalar    v_k = 0;

            for (CeedInt b = 0; b < P; b++) v_k += s_chebyshev_interp_1d[j * BASIS_P_1D + b] * in[(a * P + b) * post + c];
            out[k] = v_k;
          }
          post *= Q;
        }

        // Map to point
        __syncthreads();
        for (CeedInt p = threadIdx.x; p < BASIS_NUM_PTS; p += blockDim.x) {
          pre  = BASIS_NUM_QPTS;
          post = 1;
          for (CeedInt d = 0; d < BASIS_DIM; d++) {
            // Update buffers used
            pre /= Q;
            const CeedScalar *in  = d == 0 ? s_chebyshev_coeffs : (d % 2 ? buffer_2 : buffer_1);
            CeedScalar       *out = d == BASIS_DIM - 1 ? (&cur_v[p]) : (d % 2 ? buffer_1 : buffer_2);

            // Build Chebyshev polynomial values
            ChebyshevPolynomialsAtPoint<BASIS_Q_1D>(coords[elem * v_stride + d * v_comp_stride + p], chebyshev_x);

            // Contract along middle index
            for (CeedInt a = 0; a < pre; a++) {
              for (CeedInt c = 0; c < post; c++) {
                CeedScalar v_k = 0;

                for (CeedInt b = 0; b < Q; b++) v_k += chebyshev_x[b] * in[(a * Q + b) * post + c];
                out[a * post + c] = v_k;
              }
            }
            post *= 1;
          }
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
// Grad
//------------------------------------------------------------------------------
extern "C" __global__ void GradAtPoints(const CeedInt num_elem, const CeedInt is_transpose, const CeedScalar *__restrict__ chebyshev_interp_1d,
                                        const CeedInt *__restrict__ points_per_elem, const CeedScalar *__restrict__ coords,
                                        const CeedScalar *__restrict__ u, CeedScalar *__restrict__ v) {
  const CeedInt i = threadIdx.x;

  __shared__ CeedScalar s_mem[BASIS_Q_1D * BASIS_P_1D + 2 * BASIS_BUF_LEN + POINTS_BUFF_LEN * BASIS_Q_1D];
  CeedScalar           *s_chebyshev_interp_1d = s_mem;
  CeedScalar           *s_buffer_1            = s_mem + BASIS_Q_1D * BASIS_P_1D;
  CeedScalar           *s_buffer_2            = s_buffer_1 + BASIS_BUF_LEN;
  CeedScalar           *s_chebyshev_coeffs    = s_buffer_2 + BASIS_BUF_LEN;
  CeedScalar            chebyshev_x[BASIS_Q_1D], buffer_1[POINTS_BUFF_LEN], buffer_2[POINTS_BUFF_LEN];
  for (CeedInt k = i; k < BASIS_Q_1D * BASIS_P_1D; k += blockDim.x) {
    s_chebyshev_interp_1d[k] = chebyshev_interp_1d[k];
  }

  const CeedInt P             = BASIS_P_1D;
  const CeedInt Q             = BASIS_Q_1D;
  const CeedInt u_stride      = is_transpose ? BASIS_NUM_PTS : BASIS_NUM_NODES;
  const CeedInt v_stride      = is_transpose ? BASIS_NUM_NODES : BASIS_NUM_PTS;
  const CeedInt u_comp_stride = num_elem * (is_transpose ? BASIS_NUM_PTS : BASIS_NUM_NODES);
  const CeedInt v_comp_stride = num_elem * (is_transpose ? BASIS_NUM_NODES : BASIS_NUM_PTS);
  const CeedInt u_size        = is_transpose ? BASIS_NUM_PTS : BASIS_NUM_NODES;
  const CeedInt u_dim_stride  = is_transpose ? num_elem * BASIS_NUM_PTS * BASIS_NUM_COMP : 0;
  const CeedInt v_dim_stride  = is_transpose ? 0 : num_elem * BASIS_NUM_PTS * BASIS_NUM_COMP;

  // Apply basis element by element
  if (is_transpose) {
    for (CeedInt elem = blockIdx.x; elem < num_elem; elem += gridDim.x) {
      for (CeedInt comp = 0; comp < BASIS_NUM_COMP; comp++) {
        CeedScalar *cur_v = &v[elem * v_stride + comp * v_comp_stride];
        CeedInt     pre   = 1;
        CeedInt     post  = 1;

        // Clear Chebyshev coeffs
        for (CeedInt k = i; k < BASIS_NUM_QPTS; k += blockDim.x) {
          s_chebyshev_coeffs[k] = 0.0;
        }

        // Map from point
        __syncthreads();
        for (CeedInt p = threadIdx.x; p < BASIS_NUM_PTS; p += blockDim.x) {
          if (p >= points_per_elem[elem]) continue;
          for (CeedInt dim_1 = 0; dim_1 < BASIS_DIM; dim_1++) {
            const CeedScalar *cur_u = &u[elem * u_stride + dim_1 * u_dim_stride + comp * u_comp_stride];

            pre  = 1;
            post = 1;
            for (CeedInt dim_2 = 0; dim_2 < BASIS_DIM; dim_2++) {
              // Update buffers used
              pre /= 1;
              const CeedScalar *in  = dim_2 == 0 ? (cur_u + p) : (dim_2 % 2 ? buffer_2 : buffer_1);
              CeedScalar       *out = dim_2 == BASIS_DIM - 1 ? s_chebyshev_coeffs : (dim_2 % 2 ? buffer_1 : buffer_2);

              // Build Chebyshev polynomial values
              if (dim_1 == dim_2) ChebyshevDerivativeAtPoint<BASIS_Q_1D>(coords[elem * u_stride + dim_2 * u_comp_stride + p], chebyshev_x);
              else ChebyshevPolynomialsAtPoint<BASIS_Q_1D>(coords[elem * u_stride + dim_2 * u_comp_stride + p], chebyshev_x);

              // Contract along middle index
              for (CeedInt a = 0; a < pre; a++) {
                for (CeedInt c = 0; c < post; c++) {
                  if (dim_2 == BASIS_DIM - 1) {
                    for (CeedInt j = 0; j < Q; j++) atomicAdd(&out[(a * Q + (j + p) % Q) * post + c], chebyshev_x[(j + p) % Q] * in[a * post + c]);
                  } else {
                    for (CeedInt j = 0; j < Q; j++) out[(a * Q + j) * post + c] = chebyshev_x[j] * in[a * post + c];
                  }
                }
              }
              post *= Q;
            }
          }
        }

        // Map from coefficients
        pre  = BASIS_NUM_QPTS;
        post = 1;
        for (CeedInt d = 0; d < BASIS_DIM; d++) {
          __syncthreads();
          // Update buffers used
          pre /= Q;
          const CeedScalar *in       = d == 0 ? s_chebyshev_coeffs : (d % 2 ? s_buffer_2 : s_buffer_1);
          CeedScalar       *out      = d == BASIS_DIM - 1 ? cur_v : (d % 2 ? s_buffer_1 : s_buffer_2);
          const CeedInt     writeLen = pre * post * P;

          // Contract along middle index
          for (CeedInt k = i; k < writeLen; k += blockDim.x) {
            const CeedInt c   = k % post;
            const CeedInt j   = (k / post) % P;
            const CeedInt a   = k / (post * P);
            CeedScalar    v_k = 0;

            for (CeedInt b = 0; b < Q; b++) v_k += s_chebyshev_interp_1d[j + b * BASIS_P_1D] * in[(a * Q + b) * post + c];
            if (d == BASIS_DIM - 1) out[k] += v_k;
            else out[k] = v_k;
          }
          post *= P;
        }
      }
    }
  } else {
    for (CeedInt elem = blockIdx.x; elem < num_elem; elem += gridDim.x) {
      for (CeedInt comp = 0; comp < BASIS_NUM_COMP; comp++) {
        const CeedScalar *cur_u = &u[elem * u_stride + comp * u_comp_stride];
        CeedInt           pre   = u_size;
        CeedInt           post  = 1;

        // Map to coefficients
        for (CeedInt d = 0; d < BASIS_DIM; d++) {
          __syncthreads();
          // Update buffers used
          pre /= P;
          const CeedScalar *in       = d == 0 ? cur_u : (d % 2 ? s_buffer_2 : s_buffer_1);
          CeedScalar       *out      = d == BASIS_DIM - 1 ? s_chebyshev_coeffs : (d % 2 ? s_buffer_1 : s_buffer_2);
          const CeedInt     writeLen = pre * post * Q;

          // Contract along middle index
          for (CeedInt k = i; k < writeLen; k += blockDim.x) {
            const CeedInt c   = k % post;
            const CeedInt j   = (k / post) % Q;
            const CeedInt a   = k / (post * Q);
            CeedScalar    v_k = 0;

            for (CeedInt b = 0; b < P; b++) v_k += s_chebyshev_interp_1d[j * BASIS_P_1D + b] * in[(a * P + b) * post + c];
            out[k] = v_k;
          }
          post *= Q;
        }

        // Map to point
        __syncthreads();
        for (CeedInt p = threadIdx.x; p < BASIS_NUM_PTS; p += blockDim.x) {
          for (CeedInt dim_1 = 0; dim_1 < BASIS_DIM; dim_1++) {
            CeedScalar *cur_v = &v[elem * v_stride + dim_1 * v_dim_stride + comp * v_comp_stride];

            pre  = BASIS_NUM_QPTS;
            post = 1;
            for (CeedInt dim_2 = 0; dim_2 < BASIS_DIM; dim_2++) {
              // Update buffers used
              pre /= Q;
              const CeedScalar *in  = dim_2 == 0 ? s_chebyshev_coeffs : (dim_2 % 2 ? buffer_2 : buffer_1);
              CeedScalar       *out = dim_2 == BASIS_DIM - 1 ? (cur_v + p) : (dim_2 % 2 ? buffer_1 : buffer_2);

              // Build Chebyshev polynomial values
              if (dim_1 == dim_2) ChebyshevDerivativeAtPoint<BASIS_Q_1D>(coords[elem * v_stride + dim_2 * v_comp_stride + p], chebyshev_x);
              else ChebyshevPolynomialsAtPoint<BASIS_Q_1D>(coords[elem * v_stride + dim_2 * v_comp_stride + p], chebyshev_x);

              // Contract along middle index
              for (CeedInt a = 0; a < pre; a++) {
                for (CeedInt c = 0; c < post; c++) {
                  CeedScalar v_k = 0;

                  for (CeedInt b = 0; b < Q; b++) v_k += chebyshev_x[b] * in[(a * Q + b) * post + c];
                  out[a * post + c] = v_k;
                }
              }
              post *= 1;
            }
          }
        }
      }
    }
  }
}
