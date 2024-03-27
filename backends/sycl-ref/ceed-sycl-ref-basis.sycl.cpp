// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-tools.h>

#include <sycl/sycl.hpp>
#include <vector>

#include "../sycl/ceed-sycl-compile.hpp"
#include "ceed-sycl-ref.hpp"

template <int>
class CeedBasisSyclInterp;
template <int>
class CeedBasisSyclGrad;
class CeedBasisSyclWeight;

class CeedBasisSyclInterpNT;
class CeedBasisSyclGradNT;
class CeedBasisSyclWeightNT;

using SpecID = sycl::specialization_id<CeedInt>;

static constexpr SpecID BASIS_DIM_ID;
static constexpr SpecID BASIS_NUM_COMP_ID;
static constexpr SpecID BASIS_P_1D_ID;
static constexpr SpecID BASIS_Q_1D_ID;

//------------------------------------------------------------------------------
// Interpolation kernel - tensor
//------------------------------------------------------------------------------
template <int is_transpose>
static int CeedBasisApplyInterp_Sycl(sycl::queue &sycl_queue, const SyclModule_t &sycl_module, CeedInt num_elem, const CeedBasis_Sycl *impl,
                                     const CeedScalar *u, CeedScalar *v) {
  const CeedInt     buf_len   = impl->buf_len;
  const CeedInt     op_len    = impl->op_len;
  const CeedScalar *interp_1d = impl->d_interp_1d;

  const sycl::device &sycl_device         = sycl_queue.get_device();
  const CeedInt       max_work_group_size = 32;
  const CeedInt       work_group_size     = CeedIntMin(impl->num_qpts, max_work_group_size);
  sycl::range<1>      local_range(work_group_size);
  sycl::range<1>      global_range(num_elem * work_group_size);
  sycl::nd_range<1>   kernel_range(global_range, local_range);

  // Order queue
  sycl::event e = sycl_queue.ext_oneapi_submit_barrier();
  sycl_queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on({e});
    cgh.use_kernel_bundle(sycl_module);

    sycl::local_accessor<CeedScalar> s_mem(op_len + 2 * buf_len, cgh);

    cgh.parallel_for<CeedBasisSyclInterp<is_transpose>>(kernel_range, [=](sycl::nd_item<1> work_item, sycl::kernel_handler kh) {
      //-------------------------------------------------------------->
      // Retrieve spec constant values
      const CeedInt dim      = kh.get_specialization_constant<BASIS_DIM_ID>();
      const CeedInt num_comp = kh.get_specialization_constant<BASIS_NUM_COMP_ID>();
      const CeedInt P_1d     = kh.get_specialization_constant<BASIS_P_1D_ID>();
      const CeedInt Q_1d     = kh.get_specialization_constant<BASIS_Q_1D_ID>();
      //-------------------------------------------------------------->
      const CeedInt num_nodes     = CeedIntPow(P_1d, dim);
      const CeedInt num_qpts      = CeedIntPow(Q_1d, dim);
      const CeedInt P             = is_transpose ? Q_1d : P_1d;
      const CeedInt Q             = is_transpose ? P_1d : Q_1d;
      const CeedInt stride_0      = is_transpose ? 1 : P_1d;
      const CeedInt stride_1      = is_transpose ? P_1d : 1;
      const CeedInt u_stride      = is_transpose ? num_qpts : num_nodes;
      const CeedInt v_stride      = is_transpose ? num_nodes : num_qpts;
      const CeedInt u_comp_stride = num_elem * u_stride;
      const CeedInt v_comp_stride = num_elem * v_stride;
      const CeedInt u_size        = u_stride;

      sycl::group   work_group = work_item.get_group();
      const CeedInt i          = work_item.get_local_linear_id();
      const CeedInt group_size = work_group.get_local_linear_range();
      const CeedInt elem       = work_group.get_group_linear_id();

      CeedScalar *s_interp_1d = s_mem.get_multi_ptr<sycl::access::decorated::yes>().get();
      CeedScalar *s_buffer_1  = s_interp_1d + Q * P;
      CeedScalar *s_buffer_2  = s_buffer_1 + buf_len;

      for (CeedInt k = i; k < P * Q; k += group_size) {
        s_interp_1d[k] = interp_1d[k];
      }

      // Apply basis element by element
      for (CeedInt comp = 0; comp < num_comp; comp++) {
        const CeedScalar *cur_u = u + elem * u_stride + comp * u_comp_stride;
        CeedScalar       *cur_v = v + elem * v_stride + comp * v_comp_stride;

        for (CeedInt k = i; k < u_size; k += group_size) {
          s_buffer_1[k] = cur_u[k];
        }

        CeedInt pre  = u_size;
        CeedInt post = 1;

        for (CeedInt d = 0; d < dim; d++) {
          // Use older version of sycl workgroup barrier for performance reasons
          // Can be updated in future to align with SYCL2020 spec if performance bottleneck is removed
          // sycl::group_barrier(work_group);
          work_item.barrier(sycl::access::fence_space::local_space);

          pre /= P;
          const CeedScalar *in  = d % 2 ? s_buffer_2 : s_buffer_1;
          CeedScalar       *out = d == dim - 1 ? cur_v : (d % 2 ? s_buffer_1 : s_buffer_2);

          // Contract along middle index
          const CeedInt writeLen = pre * post * Q;
          for (CeedInt k = i; k < writeLen; k += group_size) {
            const CeedInt c = k % post;
            const CeedInt j = (k / post) % Q;
            const CeedInt a = k / (post * Q);

            CeedScalar vk = 0;
            for (CeedInt b = 0; b < P; b++) {
              vk += s_interp_1d[j * stride_0 + b * stride_1] * in[(a * P + b) * post + c];
            }
            out[k] = vk;
          }
          post *= Q;
        }
      }
    });
  });
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Gradient kernel - tensor
//------------------------------------------------------------------------------
template <int is_transpose>
static int CeedBasisApplyGrad_Sycl(sycl::queue &sycl_queue, const SyclModule_t &sycl_module, CeedInt num_elem, const CeedBasis_Sycl *impl,
                                   const CeedScalar *u, CeedScalar *v) {
  const CeedInt     buf_len   = impl->buf_len;
  const CeedInt     op_len    = impl->op_len;
  const CeedScalar *interp_1d = impl->d_interp_1d;
  const CeedScalar *grad_1d   = impl->d_grad_1d;

  const sycl::device &sycl_device     = sycl_queue.get_device();
  const CeedInt       work_group_size = 32;
  sycl::range<1>      local_range(work_group_size);
  sycl::range<1>      global_range(num_elem * work_group_size);
  sycl::nd_range<1>   kernel_range(global_range, local_range);

  // Order queue
  sycl::event e = sycl_queue.ext_oneapi_submit_barrier();
  sycl_queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on({e});
    cgh.use_kernel_bundle(sycl_module);

    sycl::local_accessor<CeedScalar> s_mem(2 * (op_len + buf_len), cgh);

    cgh.parallel_for<CeedBasisSyclGrad<is_transpose>>(kernel_range, [=](sycl::nd_item<1> work_item, sycl::kernel_handler kh) {
      //-------------------------------------------------------------->
      // Retrieve spec constant values
      const CeedInt dim      = kh.get_specialization_constant<BASIS_DIM_ID>();
      const CeedInt num_comp = kh.get_specialization_constant<BASIS_NUM_COMP_ID>();
      const CeedInt P_1d     = kh.get_specialization_constant<BASIS_P_1D_ID>();
      const CeedInt Q_1d     = kh.get_specialization_constant<BASIS_Q_1D_ID>();
      //-------------------------------------------------------------->
      const CeedInt num_nodes     = CeedIntPow(P_1d, dim);
      const CeedInt num_qpts      = CeedIntPow(Q_1d, dim);
      const CeedInt P             = is_transpose ? Q_1d : P_1d;
      const CeedInt Q             = is_transpose ? P_1d : Q_1d;
      const CeedInt stride_0      = is_transpose ? 1 : P_1d;
      const CeedInt stride_1      = is_transpose ? P_1d : 1;
      const CeedInt u_stride      = is_transpose ? num_qpts : num_nodes;
      const CeedInt v_stride      = is_transpose ? num_nodes : num_qpts;
      const CeedInt u_comp_stride = num_elem * u_stride;
      const CeedInt v_comp_stride = num_elem * v_stride;
      const CeedInt u_dim_stride  = is_transpose ? num_elem * num_qpts * num_comp : 0;
      const CeedInt v_dim_stride  = is_transpose ? 0 : num_elem * num_qpts * num_comp;
      sycl::group   work_group    = work_item.get_group();
      const CeedInt i             = work_item.get_local_linear_id();
      const CeedInt group_size    = work_group.get_local_linear_range();
      const CeedInt elem          = work_group.get_group_linear_id();

      CeedScalar *s_interp_1d = s_mem.get_multi_ptr<sycl::access::decorated::yes>().get();
      CeedScalar *s_grad_1d   = s_interp_1d + P * Q;
      CeedScalar *s_buffer_1  = s_grad_1d + P * Q;
      CeedScalar *s_buffer_2  = s_buffer_1 + buf_len;

      for (CeedInt k = i; k < P * Q; k += group_size) {
        s_interp_1d[k] = interp_1d[k];
        s_grad_1d[k]   = grad_1d[k];
      }

      // Apply basis element by element
      for (CeedInt comp = 0; comp < num_comp; comp++) {
        for (CeedInt dim_1 = 0; dim_1 < dim; dim_1++) {
          CeedInt           pre   = is_transpose ? num_qpts : num_nodes;
          CeedInt           post  = 1;
          const CeedScalar *cur_u = u + elem * u_stride + dim_1 * u_dim_stride + comp * u_comp_stride;
          CeedScalar       *cur_v = v + elem * v_stride + dim_1 * v_dim_stride + comp * v_comp_stride;

          for (CeedInt dim_2 = 0; dim_2 < dim; dim_2++) {
            // Use older version of sycl workgroup barrier for performance reasons
            // Can be updated in future to align with SYCL2020 spec if performance bottleneck is removed
            // sycl::group_barrier(work_group);
            work_item.barrier(sycl::access::fence_space::local_space);

            pre /= P;
            const CeedScalar *op  = dim_1 == dim_2 ? s_grad_1d : s_interp_1d;
            const CeedScalar *in  = (dim_2 == 0 ? cur_u : (dim_2 % 2 ? s_buffer_2 : s_buffer_1));
            CeedScalar       *out = dim_2 == dim - 1 ? cur_v : (dim_2 % 2 ? s_buffer_1 : s_buffer_2);

            // Contract along middle index
            const CeedInt writeLen = pre * post * Q;
            for (CeedInt k = i; k < writeLen; k += group_size) {
              const CeedInt c = k % post;
              const CeedInt j = (k / post) % Q;
              const CeedInt a = k / (post * Q);

              CeedScalar v_k = 0;
              for (CeedInt b = 0; b < P; b++) v_k += op[j * stride_0 + b * stride_1] * in[(a * P + b) * post + c];

              if (is_transpose && dim_2 == dim - 1) out[k] += v_k;
              else out[k] = v_k;
            }

            post *= Q;
          }
        }
      }
    });
  });
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Weight kernel - tensor
//------------------------------------------------------------------------------
static int CeedBasisApplyWeight_Sycl(sycl::queue &sycl_queue, CeedInt num_elem, const CeedBasis_Sycl *impl, CeedScalar *w) {
  const CeedInt     dim         = impl->dim;
  const CeedInt     Q_1d        = impl->Q_1d;
  const CeedScalar *q_weight_1d = impl->d_q_weight_1d;

  const CeedInt  num_quad_x = Q_1d;
  const CeedInt  num_quad_y = (dim > 1) ? Q_1d : 1;
  const CeedInt  num_quad_z = (dim > 2) ? Q_1d : 1;
  sycl::range<3> kernel_range(num_elem * num_quad_z, num_quad_y, num_quad_x);

  // Order queue
  sycl::event e = sycl_queue.ext_oneapi_submit_barrier();
  sycl_queue.parallel_for<CeedBasisSyclWeight>(kernel_range, {e}, [=](sycl::item<3> work_item) {
    if (dim == 1) w[work_item.get_linear_id()] = q_weight_1d[work_item[2]];
    if (dim == 2) w[work_item.get_linear_id()] = q_weight_1d[work_item[2]] * q_weight_1d[work_item[1]];
    if (dim == 3) w[work_item.get_linear_id()] = q_weight_1d[work_item[2]] * q_weight_1d[work_item[1]] * q_weight_1d[work_item[0] % Q_1d];
  });
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis apply - tensor
//------------------------------------------------------------------------------
static int CeedBasisApply_Sycl(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
                               CeedVector v) {
  Ceed              ceed;
  const CeedInt     is_transpose = t_mode == CEED_TRANSPOSE;
  const CeedScalar *d_u;
  CeedScalar       *d_v;
  Ceed_Sycl        *data;
  CeedBasis_Sycl   *impl;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedGetData(ceed, &data));
  CeedCallBackend(CeedBasisGetData(basis, &impl));

  // Get read/write access to u, v
  if (u != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u));
  else CeedCheck(eval_mode == CEED_EVAL_WEIGHT, ceed, CEED_ERROR_BACKEND, "An input vector is required for this CeedEvalMode");
  CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v));

  // Clear v for transpose operation
  if (is_transpose) {
    CeedSize length;
    CeedCallBackend(CeedVectorGetLength(v, &length));
    // Order queue
    sycl::event e = data->sycl_queue.ext_oneapi_submit_barrier();
    data->sycl_queue.fill<CeedScalar>(d_v, 0, length, {e});
  }

  // Basis action
  switch (eval_mode) {
    case CEED_EVAL_INTERP: {
      if (is_transpose) {
        CeedCallBackend(CeedBasisApplyInterp_Sycl<CEED_TRANSPOSE>(data->sycl_queue, *impl->sycl_module, num_elem, impl, d_u, d_v));
      } else {
        CeedCallBackend(CeedBasisApplyInterp_Sycl<CEED_NOTRANSPOSE>(data->sycl_queue, *impl->sycl_module, num_elem, impl, d_u, d_v));
      }
    } break;
    case CEED_EVAL_GRAD: {
      if (is_transpose) {
        CeedCallBackend(CeedBasisApplyGrad_Sycl<1>(data->sycl_queue, *impl->sycl_module, num_elem, impl, d_u, d_v));
      } else {
        CeedCallBackend(CeedBasisApplyGrad_Sycl<0>(data->sycl_queue, *impl->sycl_module, num_elem, impl, d_u, d_v));
      }
    } break;
    case CEED_EVAL_WEIGHT: {
      CeedCallBackend(CeedBasisApplyWeight_Sycl(data->sycl_queue, num_elem, impl, d_v));
    } break;
    case CEED_EVAL_NONE: /* handled separately below */
      break;
    // LCOV_EXCL_START
    case CEED_EVAL_DIV:
    case CEED_EVAL_CURL:
      return CeedError(ceed, CEED_ERROR_BACKEND, "%s not supported", CeedEvalModes[eval_mode]);
      // LCOV_EXCL_STOP
  }

  // Restore vectors, cover CEED_EVAL_NONE
  CeedCallBackend(CeedVectorRestoreArray(v, &d_v));
  if (eval_mode == CEED_EVAL_NONE) CeedCallBackend(CeedVectorSetArray(v, CEED_MEM_DEVICE, CEED_COPY_VALUES, (CeedScalar *)d_u));
  if (eval_mode != CEED_EVAL_WEIGHT) CeedCallBackend(CeedVectorRestoreArrayRead(u, &d_u));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Interpolation kernel - non-tensor
//------------------------------------------------------------------------------
static int CeedBasisApplyNonTensorInterp_Sycl(sycl::queue &sycl_queue, CeedInt num_elem, CeedInt is_transpose, const CeedBasisNonTensor_Sycl *impl,
                                              const CeedScalar *d_U, CeedScalar *d_V) {
  const CeedInt     num_comp      = impl->num_comp;
  const CeedInt     P             = is_transpose ? impl->num_qpts : impl->num_nodes;
  const CeedInt     Q             = is_transpose ? impl->num_nodes : impl->num_qpts;
  const CeedInt     stride_0      = is_transpose ? 1 : impl->num_nodes;
  const CeedInt     stride_1      = is_transpose ? impl->num_nodes : 1;
  const CeedInt     u_stride      = P;
  const CeedInt     v_stride      = Q;
  const CeedInt     u_comp_stride = u_stride * num_elem;
  const CeedInt     v_comp_stride = v_stride * num_elem;
  const CeedInt     u_size        = P;
  const CeedInt     v_size        = Q;
  const CeedScalar *d_B           = impl->d_interp;

  sycl::range<2> kernel_range(num_elem, v_size);

  // Order queue
  sycl::event e = sycl_queue.ext_oneapi_submit_barrier();
  sycl_queue.parallel_for<CeedBasisSyclInterpNT>(kernel_range, {e}, [=](sycl::id<2> indx) {
    const CeedInt i    = indx[1];
    const CeedInt elem = indx[0];

    for (CeedInt comp = 0; comp < num_comp; comp++) {
      const CeedScalar *U = d_U + elem * u_stride + comp * u_comp_stride;
      CeedScalar        V = 0.0;

      for (CeedInt j = 0; j < u_size; ++j) {
        V += d_B[i * stride_0 + j * stride_1] * U[j];
      }
      d_V[i + elem * v_stride + comp * v_comp_stride] = V;
    }
  });
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Gradient kernel - non-tensor
//------------------------------------------------------------------------------
static int CeedBasisApplyNonTensorGrad_Sycl(sycl::queue &sycl_queue, CeedInt num_elem, CeedInt is_transpose, const CeedBasisNonTensor_Sycl *impl,
                                            const CeedScalar *d_U, CeedScalar *d_V) {
  const CeedInt     num_comp      = impl->num_comp;
  const CeedInt     P             = is_transpose ? impl->num_qpts : impl->num_nodes;
  const CeedInt     Q             = is_transpose ? impl->num_nodes : impl->num_qpts;
  const CeedInt     stride_0      = is_transpose ? 1 : impl->num_nodes;
  const CeedInt     stride_1      = is_transpose ? impl->num_nodes : 1;
  const CeedInt     g_dim_stride  = P * Q;
  const CeedInt     u_stride      = P;
  const CeedInt     v_stride      = Q;
  const CeedInt     u_comp_stride = u_stride * num_elem;
  const CeedInt     v_comp_stride = v_stride * num_elem;
  const CeedInt     u_dim_stride  = u_comp_stride * num_comp;
  const CeedInt     v_dim_stride  = v_comp_stride * num_comp;
  const CeedInt     u_size        = P;
  const CeedInt     v_size        = Q;
  const CeedInt     in_dim        = is_transpose ? impl->dim : 1;
  const CeedInt     out_dim       = is_transpose ? 1 : impl->dim;
  const CeedScalar *d_G           = impl->d_grad;

  sycl::range<2> kernel_range(num_elem, v_size);

  // Order queue
  sycl::event e = sycl_queue.ext_oneapi_submit_barrier();
  sycl_queue.parallel_for<CeedBasisSyclGradNT>(kernel_range, {e}, [=](sycl::id<2> indx) {
    const CeedInt i    = indx[1];
    const CeedInt elem = indx[0];

    for (CeedInt comp = 0; comp < num_comp; comp++) {
      CeedScalar V[3] = {0.0, 0.0, 0.0};

      for (CeedInt d1 = 0; d1 < in_dim; ++d1) {
        const CeedScalar *U = d_U + elem * u_stride + comp * u_comp_stride + d1 * u_dim_stride;
        const CeedScalar *G = d_G + i * stride_0 + d1 * g_dim_stride;

        for (CeedInt j = 0; j < u_size; ++j) {
          const CeedScalar Uj = U[j];

          for (CeedInt d0 = 0; d0 < out_dim; ++d0) {
            V[d0] += G[j * stride_1 + d0 * g_dim_stride] * Uj;
          }
        }
      }
      for (CeedInt d0 = 0; d0 < out_dim; ++d0) {
        d_V[i + elem * v_stride + comp * v_comp_stride + d0 * v_dim_stride] = V[d0];
      }
    }
  });
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Weight kernel - non-tensor
//------------------------------------------------------------------------------
static int CeedBasisApplyNonTensorWeight_Sycl(sycl::queue &sycl_queue, CeedInt num_elem, const CeedBasisNonTensor_Sycl *impl, CeedScalar *d_V) {
  const CeedInt     num_qpts = impl->num_qpts;
  const CeedScalar *q_weight = impl->d_q_weight;

  sycl::range<2> kernel_range(num_elem, num_qpts);

  // Order queue
  sycl::event e = sycl_queue.ext_oneapi_submit_barrier();
  sycl_queue.parallel_for<CeedBasisSyclWeightNT>(kernel_range, {e}, [=](sycl::id<2> indx) {
    const CeedInt i          = indx[1];
    const CeedInt elem       = indx[0];
    d_V[i + elem * num_qpts] = q_weight[i];
  });
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis apply - non-tensor
//------------------------------------------------------------------------------
static int CeedBasisApplyNonTensor_Sycl(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
                                        CeedVector v) {
  Ceed                     ceed;
  const CeedInt            is_transpose = t_mode == CEED_TRANSPOSE;
  const CeedScalar        *d_u;
  CeedScalar              *d_v;
  CeedBasisNonTensor_Sycl *impl;
  Ceed_Sycl               *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedBasisGetData(basis, &impl));
  CeedCallBackend(CeedGetData(ceed, &data));

  // Get read/write access to u, v
  if (u != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u));
  else CeedCheck(eval_mode == CEED_EVAL_WEIGHT, ceed, CEED_ERROR_BACKEND, "An input vector is required for this CeedEvalMode");
  CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v));

  // Clear v for transpose operation
  if (is_transpose) {
    CeedSize length;
    CeedCallBackend(CeedVectorGetLength(v, &length));
    // Order queue
    sycl::event e = data->sycl_queue.ext_oneapi_submit_barrier();
    data->sycl_queue.fill<CeedScalar>(d_v, 0, length, {e});
  }

  // Apply basis operation
  switch (eval_mode) {
    case CEED_EVAL_INTERP: {
      CeedCallBackend(CeedBasisApplyNonTensorInterp_Sycl(data->sycl_queue, num_elem, is_transpose, impl, d_u, d_v));
    } break;
    case CEED_EVAL_GRAD: {
      CeedCallBackend(CeedBasisApplyNonTensorGrad_Sycl(data->sycl_queue, num_elem, is_transpose, impl, d_u, d_v));
    } break;
    case CEED_EVAL_WEIGHT: {
      CeedCallBackend(CeedBasisApplyNonTensorWeight_Sycl(data->sycl_queue, num_elem, impl, d_v));
    } break;
    case CEED_EVAL_NONE: /* handled separately below */
      break;
    // LCOV_EXCL_START
    case CEED_EVAL_DIV:
    case CEED_EVAL_CURL:
      return CeedError(ceed, CEED_ERROR_BACKEND, "%s not supported", CeedEvalModes[eval_mode]);
      // LCOV_EXCL_STOP
  }

  // Restore vectors, cover CEED_EVAL_NONE
  CeedCallBackend(CeedVectorRestoreArray(v, &d_v));
  if (eval_mode == CEED_EVAL_NONE) CeedCallBackend(CeedVectorSetArray(v, CEED_MEM_DEVICE, CEED_COPY_VALUES, (CeedScalar *)d_u));
  if (eval_mode != CEED_EVAL_WEIGHT) CeedCallBackend(CeedVectorRestoreArrayRead(u, &d_u));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy tensor basis
//------------------------------------------------------------------------------
static int CeedBasisDestroy_Sycl(CeedBasis basis) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedBasis_Sycl *impl;
  CeedCallBackend(CeedBasisGetData(basis, &impl));
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  // Wait for all work to finish before freeing memory
  CeedCallSycl(ceed, data->sycl_queue.wait_and_throw());

  CeedCallSycl(ceed, sycl::free(impl->d_q_weight_1d, data->sycl_context));
  CeedCallSycl(ceed, sycl::free(impl->d_interp_1d, data->sycl_context));
  CeedCallSycl(ceed, sycl::free(impl->d_grad_1d, data->sycl_context));

  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy non-tensor basis
//------------------------------------------------------------------------------
static int CeedBasisDestroyNonTensor_Sycl(CeedBasis basis) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedBasisNonTensor_Sycl *impl;
  CeedCallBackend(CeedBasisGetData(basis, &impl));
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  // Wait for all work to finish before freeing memory
  CeedCallSycl(ceed, data->sycl_queue.wait_and_throw());

  CeedCallSycl(ceed, sycl::free(impl->d_q_weight, data->sycl_context));
  CeedCallSycl(ceed, sycl::free(impl->d_interp, data->sycl_context));
  CeedCallSycl(ceed, sycl::free(impl->d_grad, data->sycl_context));

  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create tensor
//------------------------------------------------------------------------------
int CeedBasisCreateTensorH1_Sycl(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                 const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedBasis_Sycl *impl;
  CeedCallBackend(CeedCalloc(1, &impl));
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  CeedInt num_comp;
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));

  const CeedInt num_nodes = CeedIntPow(P_1d, dim);
  const CeedInt num_qpts  = CeedIntPow(Q_1d, dim);

  impl->dim       = dim;
  impl->P_1d      = P_1d;
  impl->Q_1d      = Q_1d;
  impl->num_comp  = num_comp;
  impl->num_nodes = num_nodes;
  impl->num_qpts  = num_qpts;
  impl->buf_len   = num_comp * CeedIntMax(num_nodes, num_qpts);
  impl->op_len    = Q_1d * P_1d;

  // Order queue
  sycl::event e = data->sycl_queue.ext_oneapi_submit_barrier();

  CeedCallSycl(ceed, impl->d_q_weight_1d = sycl::malloc_device<CeedScalar>(Q_1d, data->sycl_device, data->sycl_context));
  sycl::event copy_weight = data->sycl_queue.copy<CeedScalar>(q_weight_1d, impl->d_q_weight_1d, Q_1d, {e});

  const CeedInt interp_length = Q_1d * P_1d;
  CeedCallSycl(ceed, impl->d_interp_1d = sycl::malloc_device<CeedScalar>(interp_length, data->sycl_device, data->sycl_context));
  sycl::event copy_interp = data->sycl_queue.copy<CeedScalar>(interp_1d, impl->d_interp_1d, interp_length, {e});

  CeedCallSycl(ceed, impl->d_grad_1d = sycl::malloc_device<CeedScalar>(interp_length, data->sycl_device, data->sycl_context));
  sycl::event copy_grad = data->sycl_queue.copy<CeedScalar>(grad_1d, impl->d_grad_1d, interp_length, {e});

  CeedCallSycl(ceed, sycl::event::wait_and_throw({copy_weight, copy_interp, copy_grad}));

  std::vector<sycl::kernel_id> kernel_ids = {sycl::get_kernel_id<CeedBasisSyclInterp<1>>(), sycl::get_kernel_id<CeedBasisSyclInterp<0>>(),
                                             sycl::get_kernel_id<CeedBasisSyclGrad<1>>(), sycl::get_kernel_id<CeedBasisSyclGrad<0>>()};

  sycl::kernel_bundle<sycl::bundle_state::input> input_bundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(data->sycl_context, kernel_ids);
  input_bundle.set_specialization_constant<BASIS_DIM_ID>(dim);
  input_bundle.set_specialization_constant<BASIS_NUM_COMP_ID>(num_comp);
  input_bundle.set_specialization_constant<BASIS_Q_1D_ID>(Q_1d);
  input_bundle.set_specialization_constant<BASIS_P_1D_ID>(P_1d);

  CeedCallSycl(ceed, impl->sycl_module = new SyclModule_t(sycl::build(input_bundle)));

  CeedCallBackend(CeedBasisSetData(basis, impl));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Basis", basis, "Apply", CeedBasisApply_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Basis", basis, "Destroy", CeedBasisDestroy_Sycl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create non-tensor
//------------------------------------------------------------------------------
int CeedBasisCreateH1_Sycl(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp, const CeedScalar *grad,
                           const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedBasisNonTensor_Sycl *impl;
  CeedCallBackend(CeedCalloc(1, &impl));
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  CeedInt num_comp;
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));

  impl->dim       = dim;
  impl->num_comp  = num_comp;
  impl->num_nodes = num_nodes;
  impl->num_qpts  = num_qpts;

  // Order queue
  sycl::event e = data->sycl_queue.ext_oneapi_submit_barrier();

  CeedCallSycl(ceed, impl->d_q_weight = sycl::malloc_device<CeedScalar>(num_qpts, data->sycl_device, data->sycl_context));
  sycl::event copy_weight = data->sycl_queue.copy<CeedScalar>(q_weight, impl->d_q_weight, num_qpts, {e});

  const CeedInt interp_length = num_qpts * num_nodes;
  CeedCallSycl(ceed, impl->d_interp = sycl::malloc_device<CeedScalar>(interp_length, data->sycl_device, data->sycl_context));
  sycl::event copy_interp = data->sycl_queue.copy<CeedScalar>(interp, impl->d_interp, interp_length, {e});

  const CeedInt grad_length = num_qpts * num_nodes * dim;
  CeedCallSycl(ceed, impl->d_grad = sycl::malloc_device<CeedScalar>(grad_length, data->sycl_device, data->sycl_context));
  sycl::event copy_grad = data->sycl_queue.copy<CeedScalar>(grad, impl->d_grad, grad_length, {e});

  CeedCallSycl(ceed, sycl::event::wait_and_throw({copy_weight, copy_interp, copy_grad}));

  CeedCallBackend(CeedBasisSetData(basis, impl));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Basis", basis, "Destroy", CeedBasisDestroyNonTensor_Sycl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
