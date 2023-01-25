// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
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

class CeedBasisSyclInterp;
class CeedBasisSyclGrad;
class CeedBasisSyclWeight;

class CeedBasisSyclInterpNT;
class CeedBasisSyclGradNT;
class CeedBasisSyclWeightNT;

//------------------------------------------------------------------------------
// Interpolation kernel - tensor
//------------------------------------------------------------------------------
static int CeedBasisApplyInterp_Sycl(sycl::queue &sycl_queue, CeedInt num_elem, CeedInt transpose, const CeedBasis_Sycl *impl, const CeedScalar *u,
                                     CeedScalar *v) {
  const CeedInt dim           = impl->dim;
  const CeedInt buf_len       = impl->buf_len;
  const CeedInt num_comp      = impl->num_comp;
  const CeedInt P             = transpose ? impl->Q_1d : impl->P_1d;
  const CeedInt Q             = transpose ? impl->P_1d : impl->Q_1d;
  const CeedInt stride_0      = transpose ? 1 : impl->P_1d;
  const CeedInt stride_1      = transpose ? impl->P_1d : 1;
  const CeedInt u_stride      = transpose ? impl->num_qpts : impl->num_nodes;
  const CeedInt v_stride      = transpose ? impl->num_nodes : impl->num_qpts;
  const CeedInt u_comp_stride = num_elem * u_stride;
  const CeedInt v_comp_stride = num_elem * v_stride;
  const CeedInt u_size        = u_stride;

  const CeedScalar *interp_1d = impl->d_interp_1d;

  const sycl::device &sycl_device         = sycl_queue.get_device();
  const CeedInt       max_work_group_size = sycl_device.get_info<sycl::info::device::max_work_group_size>();
  const CeedInt       work_group_size     = CeedIntMin(impl->num_qpts, max_work_group_size);
  sycl::range<1>      local_range(work_group_size);
  sycl::range<1>      global_range(num_elem * work_group_size);
  sycl::nd_range<1>   kernel_range(global_range, local_range);

  sycl_queue.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<CeedScalar> s_mem(P * Q + 2 * buf_len, cgh);

    cgh.parallel_for<CeedBasisSyclInterp>(kernel_range, [=](sycl::nd_item<1> work_item) {
      sycl::group   work_group = work_item.get_group();
      const CeedInt i          = work_item.get_local_linear_id();
      const CeedInt group_size = work_group.get_local_linear_range();
      const CeedInt elem       = work_group.get_group_linear_id();

      CeedScalar *s_interp_1d = s_mem.get_pointer();
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
          sycl::group_barrier(work_group);

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
static int CeedBasisApplyGrad_Sycl(sycl::queue &sycl_queue, CeedInt num_elem, CeedInt transpose, const CeedBasis_Sycl *impl, const CeedScalar *u,
                                   CeedScalar *v) {
  const CeedInt dim           = impl->dim;
  const CeedInt buf_len       = impl->buf_len;
  const CeedInt num_comp      = impl->num_comp;
  const CeedInt num_qpts      = impl->num_qpts;
  const CeedInt num_nodes     = impl->num_nodes;
  const CeedInt P             = transpose ? impl->Q_1d : impl->P_1d;
  const CeedInt Q             = transpose ? impl->P_1d : impl->Q_1d;
  const CeedInt stride_0      = transpose ? 1 : impl->P_1d;
  const CeedInt stride_1      = transpose ? impl->P_1d : 1;
  const CeedInt u_stride      = transpose ? impl->num_qpts : impl->num_nodes;
  const CeedInt v_stride      = transpose ? impl->num_nodes : impl->num_qpts;
  const CeedInt u_comp_stride = num_elem * u_stride;
  const CeedInt v_comp_stride = num_elem * v_stride;
  const CeedInt u_dim_stride  = transpose ? num_elem * impl->num_qpts * impl->num_comp : 0;
  const CeedInt v_dim_stride  = transpose ? 0 : num_elem * impl->num_qpts * impl->num_comp;

  const CeedScalar *interp_1d = impl->d_interp_1d;
  const CeedScalar *grad_1d   = impl->d_grad_1d;

  const sycl::device &sycl_device         = sycl_queue.get_device();
  const CeedInt       max_work_group_size = sycl_device.get_info<sycl::info::device::max_work_group_size>();
  const CeedInt       work_group_size     = max_work_group_size;
  sycl::range<1>      local_range(work_group_size);
  sycl::range<1>      global_range(num_elem * work_group_size);
  sycl::nd_range<1>   kernel_range(global_range, local_range);

  sycl_queue.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<CeedScalar> s_mem(2 * (P * Q + buf_len), cgh);

    cgh.parallel_for<CeedBasisSyclGrad>(kernel_range, [=](sycl::nd_item<1> work_item) {
      sycl::group   work_group = work_item.get_group();
      const CeedInt i          = work_item.get_local_linear_id();
      const CeedInt group_size = work_group.get_local_linear_range();
      const CeedInt elem       = work_group.get_group_linear_id();

      CeedScalar *s_interp_1d = s_mem.get_pointer();
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
          CeedInt           pre   = transpose ? num_qpts : num_nodes;
          CeedInt           post  = 1;
          const CeedScalar *cur_u = u + elem * u_stride + dim_1 * u_dim_stride + comp * u_comp_stride;
          CeedScalar       *cur_v = v + elem * v_stride + dim_1 * v_dim_stride + comp * v_comp_stride;

          for (CeedInt dim_2 = 0; dim_2 < dim; dim_2++) {
            sycl::group_barrier(work_group);

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

              if (transpose && dim_2 == dim - 1) out[k] += v_k;
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

  CeedInt range_zyx[3];
  range_zyx[2] = Q_1d;
  range_zyx[1] = (dim > 1) ? Q_1d : 1;
  range_zyx[0] = (dim > 2) ? Q_1d : 1;
  sycl::range<3> kernel_range(num_elem * range_zyx[0], range_zyx[1], range_zyx[2]);

  sycl_queue.parallel_for<CeedBasisSyclWeight>(kernel_range, [=](sycl::item<3> indx) {
    CeedScalar q_ijk = q_weight_1d[indx[2]];
    for (CeedInt d = 1; d > 2 - dim; --d) {
      for (int j = range_zyx[d]; j > 1; j -= Q_1d) {
        q_ijk *= q_weight_1d[indx[d]];
      }
    }
    w[indx.get_linear_id()] = q_ijk;
  });

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis apply - tensor
//------------------------------------------------------------------------------
static int CeedBasisApply_Sycl(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
                               CeedVector v) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));
  CeedBasis_Sycl *impl;
  CeedCallBackend(CeedBasisGetData(basis, &impl));

  const CeedInt transpose = t_mode == CEED_TRANSPOSE;

  // Read vectors
  const CeedScalar *d_u;
  CeedScalar       *d_v;
  if (eval_mode != CEED_EVAL_WEIGHT) {
    CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u));
  }
  CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v));

  // Clear v for transpose operation
  if (t_mode == CEED_TRANSPOSE) {
    CeedSize length;
    CeedCallBackend(CeedVectorGetLength(v, &length));
    data->sycl_queue.fill(d_v, 0, length);
  }

  // Basis action
  switch (eval_mode) {
    case CEED_EVAL_INTERP: {
      CeedCallBackend(CeedBasisApplyInterp_Sycl(data->sycl_queue, num_elem, transpose, impl, d_u, d_v));
    } break;
    case CEED_EVAL_GRAD: {
      CeedCallBackend(CeedBasisApplyGrad_Sycl(data->sycl_queue, num_elem, transpose, impl, d_u, d_v));
    } break;
    case CEED_EVAL_WEIGHT: {
      CeedCallBackend(CeedBasisApplyWeight_Sycl(data->sycl_queue, num_elem, impl, d_v));
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
      return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_NONE does not make sense in this context");
      // LCOV_EXCL_STOP
  }

  // Restore vectors
  if (eval_mode != CEED_EVAL_WEIGHT) {
    CeedCallBackend(CeedVectorRestoreArrayRead(u, &d_u));
  }
  CeedCallBackend(CeedVectorRestoreArray(v, &d_v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Interpolation kernel - non-tensor
//------------------------------------------------------------------------------
static int CeedBasisApplyNonTensorInterp_Sycl(sycl::queue &sycl_queue, CeedInt num_elem, CeedInt transpose, const CeedBasisNonTensor_Sycl *impl,
                                              const CeedScalar *d_U, CeedScalar *d_V) {
  const CeedInt     num_comp      = impl->num_comp;
  const CeedInt     P             = transpose ? impl->num_qpts : impl->num_nodes;
  const CeedInt     Q             = transpose ? impl->num_nodes : impl->num_qpts;
  const CeedInt     stride_0      = transpose ? 1 : impl->num_nodes;
  const CeedInt     stride_1      = transpose ? impl->num_nodes : 1;
  const CeedInt     u_stride      = P;
  const CeedInt     v_stride      = Q;
  const CeedInt     u_comp_stride = u_stride * num_elem;
  const CeedInt     v_comp_stride = v_stride * num_elem;
  const CeedInt     u_size        = P;
  const CeedInt     v_size        = Q;
  const CeedScalar *d_B           = impl->d_interp;

  sycl::range<2> kernel_range(num_elem, v_size);

  sycl_queue.parallel_for<CeedBasisSyclInterpNT>(kernel_range, [=](sycl::id<2> indx) {
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
static int CeedBasisApplyNonTensorGrad_Sycl(sycl::queue &sycl_queue, CeedInt num_elem, CeedInt transpose, const CeedBasisNonTensor_Sycl *impl,
                                            const CeedScalar *d_U, CeedScalar *d_V) {
  const CeedInt     num_comp      = impl->num_comp;
  const CeedInt     P             = transpose ? impl->num_qpts : impl->num_nodes;
  const CeedInt     Q             = transpose ? impl->num_nodes : impl->num_qpts;
  const CeedInt     stride_0      = transpose ? 1 : impl->num_nodes;
  const CeedInt     stride_1      = transpose ? impl->num_nodes : 1;
  const CeedInt     g_dim_stride  = P * Q;
  const CeedInt     u_stride      = P;
  const CeedInt     v_stride      = Q;
  const CeedInt     u_comp_stride = u_stride * num_elem;
  const CeedInt     v_comp_stride = v_stride * num_elem;
  const CeedInt     u_dim_stride  = u_comp_stride * num_comp;
  const CeedInt     v_dim_stride  = v_comp_stride * num_comp;
  const CeedInt     u_size        = P;
  const CeedInt     v_size        = Q;
  const CeedInt     in_dim        = transpose ? impl->dim : 1;
  const CeedInt     out_dim       = transpose ? 1 : impl->dim;
  const CeedScalar *d_G           = impl->d_grad;

  sycl::range<2> kernel_range(num_elem, v_size);

  sycl_queue.parallel_for<CeedBasisSyclGradNT>(kernel_range, [=](sycl::id<2> indx) {
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

  sycl_queue.parallel_for<CeedBasisSyclWeightNT>(kernel_range, [=](sycl::id<2> indx) {
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
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedBasisNonTensor_Sycl *impl;
  CeedCallBackend(CeedBasisGetData(basis, &impl));
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  const CeedInt transpose = t_mode == CEED_TRANSPOSE;

  // Read vectors
  const CeedScalar *d_u;
  CeedScalar       *d_v;
  if (eval_mode != CEED_EVAL_WEIGHT) {
    CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u));
  }
  CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v));

  // Clear v for transpose operation
  if (transpose) {
    CeedSize length;
    CeedCallBackend(CeedVectorGetLength(v, &length));
    data->sycl_queue.fill(d_v, 0, length);
  }

  // Apply basis operation
  switch (eval_mode) {
    case CEED_EVAL_INTERP: {
      CeedCallBackend(CeedBasisApplyNonTensorInterp_Sycl(data->sycl_queue, num_elem, transpose, impl, d_u, d_v));
    } break;
    case CEED_EVAL_GRAD: {
      CeedCallBackend(CeedBasisApplyNonTensorGrad_Sycl(data->sycl_queue, num_elem, transpose, impl, d_u, d_v));
    } break;
    case CEED_EVAL_WEIGHT: {
      CeedCallBackend(CeedBasisApplyNonTensorWeight_Sycl(data->sycl_queue, num_elem, impl, d_v));
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
      return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_NONE does not make sense in this context");
      // LCOV_EXCL_STOP
  }

  // Restore vectors
  if (eval_mode != CEED_EVAL_WEIGHT) {
    CeedCallBackend(CeedVectorRestoreArrayRead(u, &d_u));
  }

  CeedCallBackend(CeedVectorRestoreArray(v, &d_v));
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

  CeedCallSycl(ceed, impl->d_q_weight_1d = sycl::malloc_device<CeedScalar>(Q_1d, data->sycl_device, data->sycl_context));
  sycl::event copy_weight = data->sycl_queue.copy<CeedScalar>(q_weight_1d, impl->d_q_weight_1d, Q_1d);

  const CeedInt interp_length = Q_1d * P_1d;
  CeedCallSycl(ceed, impl->d_interp_1d = sycl::malloc_device<CeedScalar>(interp_length, data->sycl_device, data->sycl_context));
  sycl::event copy_interp = data->sycl_queue.copy<CeedScalar>(interp_1d, impl->d_interp_1d, interp_length);

  CeedCallSycl(ceed, impl->d_grad_1d = sycl::malloc_device<CeedScalar>(interp_length, data->sycl_device, data->sycl_context));
  sycl::event copy_grad = data->sycl_queue.copy<CeedScalar>(grad_1d, impl->d_grad_1d, interp_length);

  CeedCallSycl(ceed, sycl::event::wait_and_throw({copy_weight, copy_interp, copy_grad}));

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
                           const CeedScalar *qref, const CeedScalar *q_weight, CeedBasis basis) {
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

  CeedCallSycl(ceed, impl->d_q_weight = sycl::malloc_device<CeedScalar>(num_qpts, data->sycl_device, data->sycl_context));
  sycl::event copy_weight = data->sycl_queue.copy<CeedScalar>(q_weight, impl->d_q_weight, num_qpts);

  const CeedInt interp_length = num_qpts * num_nodes;
  CeedCallSycl(ceed, impl->d_interp = sycl::malloc_device<CeedScalar>(interp_length, data->sycl_device, data->sycl_context));
  sycl::event copy_interp = data->sycl_queue.copy<CeedScalar>(interp, impl->d_interp, interp_length);

  const CeedInt grad_length = num_qpts * num_nodes * dim;
  CeedCallSycl(ceed, impl->d_grad = sycl::malloc_device<CeedScalar>(grad_length, data->sycl_device, data->sycl_context));
  sycl::event copy_grad = data->sycl_queue.copy<CeedScalar>(grad, impl->d_grad, grad_length);

  CeedCallSycl(ceed, sycl::event::wait_and_throw({copy_weight, copy_interp, copy_grad}));

  CeedCallBackend(CeedBasisSetData(basis, impl));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Basis", basis, "Destroy", CeedBasisDestroyNonTensor_Sycl));
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
