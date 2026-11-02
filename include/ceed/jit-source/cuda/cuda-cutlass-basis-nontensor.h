// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CUDA CUTLASS non-tensor product basis
#include <ceed/types.h>

// The JiT defines int32_t as a macro, which breaks the cuda::std headers that CuTe includes
#undef int32_t
#include <cute/tensor.hpp>

//------------------------------------------------------------------------------
// GEMM over a block of BN = TN * RN columns: V (M x num_cols) = sum_d A_d (M x K) * U_d (K x num_cols)
//   TM x TN threads, each holding (M / TM) x RN outputs; K is split into tiles of BK that fit in shared memory
//   Forward:   M = Q, K = P, A_d = B_d, one V block per d
//   Transpose: M = P, K = Q, A_d = B_d^T, sum over d stored into V (added if add is set)
//------------------------------------------------------------------------------
template <int M, int K, int Q_COMP, bool TRANSPOSE, int TM, int TN, int RN>
inline __device__ void ContractCutlass(const CeedInt num_cols, const CeedInt add, const CeedScalar *__restrict__ d_B,
                                       const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V) {
  static_assert(M % TM == 0, "M must be a multiple of TM");
  constexpr int BN     = TN * RN;
  constexpr int BK_MAX = 48 * 1024 / ((M + BN) * sizeof(CeedScalar)) - 1;
  constexpr int NUM_K  = (K + BK_MAX - 1) / BK_MAX;
  constexpr int BK     = (K + NUM_K - 1) / NUM_K;
  constexpr int BKS    = BK | 1;  // odd row stride in shared memory avoids bank conflicts
  static_assert(BK_MAX >= 1, "basis too large for shared memory");

  __shared__ CeedScalar s_A[M * BKS];
  __shared__ CeedScalar s_U[BN * BKS];

  const CeedInt  col_0          = blockIdx.x * BN;
  const CeedInt  num_block_cols = min(BN, num_cols - col_0);
  cute::TiledMMA tiled_mma      = cute::make_tiled_mma(cute::UniversalFMA<CeedScalar, CeedScalar, CeedScalar>{},
                                                       cute::Layout<cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::_1>>{});
  cute::ThrMMA   thread_mma     = tiled_mma.get_slice(threadIdx.x);
  // B_d is stored Q x P row major, so A_d is row major (forward) or column major (transpose)
  auto s_A_layout = [] {
    if constexpr (TRANSPOSE) {
      return cute::make_layout(cute::make_shape(cute::Int<M>{}, cute::Int<BK>{}), cute::LayoutLeft{});
    } else {
      return cute::make_layout(cute::make_shape(cute::Int<M>{}, cute::Int<BK>{}), cute::make_stride(cute::Int<BKS>{}, cute::_1{}));
    }
  }();
  cute::Tensor s_A_thread     = thread_mma.partition_A(cute::make_tensor(cute::make_smem_ptr(s_A), s_A_layout));
  cute::Tensor s_U_thread     = thread_mma.partition_B(cute::make_tensor(
      cute::make_smem_ptr(s_U), cute::make_layout(cute::make_shape(cute::Int<BN>{}, cute::Int<BK>{}), cute::make_stride(cute::Int<BKS>{}, cute::_1{}))));
  cute::Tensor coord_V_thread = thread_mma.partition_C(cute::make_identity_tensor(cute::make_shape(cute::Int<M>{}, cute::Int<BN>{})));
  cute::Tensor r_V            = cute::make_tensor<CeedScalar>(cute::shape(coord_V_thread));

  auto get_d_V_thread = [&](const CeedInt d) {
    return thread_mma.partition_C(cute::make_tensor(cute::make_gmem_ptr(d_V + d * num_cols * M + col_0 * M),
                                                    cute::make_layout(cute::make_shape(cute::Int<M>{}, cute::Int<BN>{}))));
  };

  if constexpr (TRANSPOSE) cute::clear(r_V);
  for (CeedInt d = 0; d < Q_COMP; d++) {
    const CeedScalar *d_A       = d_B + d * M * K;
    const CeedScalar *d_U_block = d_U + (TRANSPOSE ? d * num_cols * K : 0) + col_0 * K;

    if constexpr (!TRANSPOSE) cute::clear(r_V);
    for (CeedInt k_0 = 0; k_0 < K; k_0 += BK) {
      // Tail of the last K tile is zero filled
      for (CeedInt i = threadIdx.x; i < M * BK; i += blockDim.x) {
        if constexpr (TRANSPOSE) {
          s_A[i] = k_0 + i / M < K ? d_A[k_0 * M + i] : 0.0;
        } else {
          s_A[(i / BK) * BKS + i % BK] = k_0 + i % BK < K ? d_A[(i / BK) * K + k_0 + i % BK] : 0.0;
        }
      }
      for (CeedInt i = threadIdx.x; i < BN * BK; i += blockDim.x) {
        s_U[(i / BK) * BKS + i % BK] = (i / BK < num_block_cols && k_0 + i % BK < K) ? d_U_block[(i / BK) * K + k_0 + i % BK] : 0.0;
      }
      __syncthreads();
      cute::gemm(tiled_mma, s_A_thread, s_U_thread, r_V);
      __syncthreads();
    }
    if constexpr (!TRANSPOSE) {
      cute::Tensor d_V_thread = get_d_V_thread(d);

      CUTE_UNROLL
      for (CeedInt i = 0; i < cute::size(r_V); i++) {
        if (cute::get<1>(coord_V_thread(i)) < num_block_cols) d_V_thread(i) = r_V(i);
      }
    }
  }
  if constexpr (TRANSPOSE) {
    cute::Tensor d_V_thread = get_d_V_thread(0);

    CUTE_UNROLL
    for (CeedInt i = 0; i < cute::size(r_V); i++) {
      if (cute::get<1>(coord_V_thread(i)) < num_block_cols) d_V_thread(i) = add ? d_V_thread(i) + r_V(i) : r_V(i);
    }
  }
}

//------------------------------------------------------------------------------
// Interp
//------------------------------------------------------------------------------
extern "C" __global__ void Interp(const CeedInt num_elem, const CeedInt add, const CeedScalar *__restrict__ d_B, const CeedScalar *__restrict__ d_U,
                                  CeedScalar *__restrict__ d_V) {
  ContractCutlass<BASIS_Q, BASIS_P, BASIS_Q_COMP_INTERP, false, BASIS_NOTRANSPOSE_TM, BASIS_NOTRANSPOSE_TN, BASIS_NOTRANSPOSE_RN>(num_elem *
                                                                                                                                      BASIS_NUM_COMP,
                                                                                                                                  add, d_B, d_U, d_V);
}

extern "C" __global__ void InterpTranspose(const CeedInt num_elem, const CeedInt add, const CeedScalar *__restrict__ d_B,
                                           const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V) {
  ContractCutlass<BASIS_P, BASIS_Q, BASIS_Q_COMP_INTERP, true, BASIS_TRANSPOSE_TM, BASIS_TRANSPOSE_TN, BASIS_TRANSPOSE_RN>(num_elem * BASIS_NUM_COMP,
                                                                                                                           add, d_B, d_U, d_V);
}

//------------------------------------------------------------------------------
// Deriv
//------------------------------------------------------------------------------
extern "C" __global__ void Deriv(const CeedInt num_elem, const CeedInt add, const CeedScalar *__restrict__ d_B, const CeedScalar *__restrict__ d_U,
                                 CeedScalar *__restrict__ d_V) {
  ContractCutlass<BASIS_Q, BASIS_P, BASIS_Q_COMP_DERIV, false, BASIS_NOTRANSPOSE_TM, BASIS_NOTRANSPOSE_TN, BASIS_NOTRANSPOSE_RN>(num_elem *
                                                                                                                                     BASIS_NUM_COMP,
                                                                                                                                 add, d_B, d_U, d_V);
}

extern "C" __global__ void DerivTranspose(const CeedInt num_elem, const CeedInt add, const CeedScalar *__restrict__ d_B,
                                          const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V) {
  ContractCutlass<BASIS_P, BASIS_Q, BASIS_Q_COMP_DERIV, true, BASIS_TRANSPOSE_TM, BASIS_TRANSPOSE_TN, BASIS_TRANSPOSE_RN>(num_elem * BASIS_NUM_COMP,
                                                                                                                          add, d_B, d_U, d_V);
}

//------------------------------------------------------------------------------
// Weight
//------------------------------------------------------------------------------
extern "C" __global__ void Weight(const CeedInt num_elem, const CeedScalar *__restrict__ q_weight, CeedScalar *__restrict__ d_V) {
  for (CeedInt elem = blockIdx.x; elem < num_elem; elem += gridDim.x) d_V[elem * BASIS_Q + threadIdx.x] = q_weight[threadIdx.x];
}
