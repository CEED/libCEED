// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>
#include <string.h>

#ifdef CEED_MAGMA_USE_HIP
#include "../hip/ceed-hip-common.h"
#include "../hip/ceed-hip-compile.h"
#else
#include "../cuda/ceed-cuda-common.h"
#include "../cuda/ceed-cuda-compile.h"
#endif
#include "ceed-magma-common.h"
#include "ceed-magma.h"

#include "ceed-magma-gemm-nontensor.h"
#include "ceed-magma-gemm-selector.h"

//------------------------------------------------------------------------------
// Basis apply - tensor
//------------------------------------------------------------------------------
static int CeedBasisApply_Magma(CeedBasis basis, CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode e_mode, CeedVector u, CeedVector v) {
  Ceed              ceed;
  Ceed_Magma       *data;
  CeedInt           dim, num_comp, num_nodes, P_1d, Q_1d, P, Q;
  const CeedScalar *d_u;
  CeedScalar       *d_v;
  CeedBasis_Magma  *impl;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedGetData(ceed, &data));
  CeedCallBackend(CeedBasisGetData(basis, &impl));
  CeedCallBackend(CeedBasisGetDimension(basis, &dim));
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedBasisGetNumNodes(basis, &num_nodes));
  CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
  CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
  P = P_1d;
  Q = Q_1d;
  if (t_mode == CEED_TRANSPOSE) {
    P = Q_1d;
    Q = P_1d;
  }

  // Read vectors
  if (u != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u));
  else CeedCheck(e_mode == CEED_EVAL_WEIGHT, ceed, CEED_ERROR_BACKEND, "An input vector is required for this CeedEvalMode");
  CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v));

  // Apply basis operation
  switch (e_mode) {
    case CEED_EVAL_INTERP: {
      // Define element sizes for dofs/quad
      CeedInt elem_qpts_size = CeedIntPow(Q_1d, dim);
      CeedInt elem_dofs_size = CeedIntPow(P_1d, dim);

      // E-vector ordering -------------- Q-vector ordering
      //  component                        component
      //    elem                             elem
      //       node                            node

      // ---  Define strides for NOTRANSPOSE mode: ---
      // Input (d_u) is E-vector, output (d_v) is Q-vector

      // Element strides
      CeedInt u_elem_stride = elem_dofs_size;
      CeedInt v_elem_stride = elem_qpts_size;
      // Component strides
      CeedInt u_comp_stride = num_elem * elem_dofs_size;
      CeedInt v_comp_stride = num_elem * elem_qpts_size;
      if (t_mode == CEED_TRANSPOSE) {
        // Input (d_u) is Q-vector, output (d_v) is E-vector
        // Element strides
        v_elem_stride = elem_dofs_size;
        u_elem_stride = elem_qpts_size;
        // Component strides
        v_comp_stride = num_elem * elem_dofs_size;
        u_comp_stride = num_elem * elem_qpts_size;
      }
      CeedInt num_threads = 1;
      CeedInt num_t_col   = 1;
      CeedInt shared_mem  = 0;
      CeedInt max_P_Q     = CeedIntMax(P, Q);

      switch (dim) {
        case 1:
          num_threads = max_P_Q;
          num_t_col   = MAGMA_BASIS_NTCOL(num_threads, MAGMA_MAXTHREADS_1D);
          shared_mem += sizeof(CeedScalar) * num_t_col * (num_comp * (1 * P + 1 * Q));
          shared_mem += sizeof(CeedScalar) * (P * Q);
          break;
        case 2:
          num_threads = max_P_Q;
          num_t_col   = MAGMA_BASIS_NTCOL(num_threads, MAGMA_MAXTHREADS_2D);
          shared_mem += P * Q * sizeof(CeedScalar);  // for sT
          // for reforming rU we need P x P, and for the intermediate output we need P x Q
          shared_mem += num_t_col * (P * max_P_Q * sizeof(CeedScalar));
          break;
        case 3:
          num_threads = max_P_Q * max_P_Q;
          num_t_col   = MAGMA_BASIS_NTCOL(num_threads, MAGMA_MAXTHREADS_3D);
          shared_mem += sizeof(CeedScalar) * (P * Q);  // for sT
          // rU needs P^2 x P, the intermediate output needs max(P^2 x Q, P x Q^2)
          shared_mem += sizeof(CeedScalar) * num_t_col * (CeedIntMax(P * P * max_P_Q, P * Q * Q));
          break;
      }
      CeedInt grid   = CeedDivUpInt(num_elem, num_t_col);
      void   *args[] = {&impl->d_interp_1d, &d_u, &u_elem_stride, &u_comp_stride, &d_v, &v_elem_stride, &v_comp_stride, &num_elem};

      if (t_mode == CEED_TRANSPOSE) {
        CeedCallBackend(CeedRunKernelDimSharedMagma(ceed, impl->InterpTranspose, grid, num_threads, num_t_col, 1, shared_mem, args));
      } else {
        CeedCallBackend(CeedRunKernelDimSharedMagma(ceed, impl->Interp, grid, num_threads, num_t_col, 1, shared_mem, args));
      }
    } break;
    case CEED_EVAL_GRAD: {
      // Define element sizes for dofs/quad
      CeedInt elem_qpts_size = CeedIntPow(Q_1d, dim);
      CeedInt elem_dofs_size = CeedIntPow(P_1d, dim);

      // In CEED_NOTRANSPOSE mode:
      // d_u is (P^dim x nc), column-major layout (nc = num_comp)
      // d_v is (Q^dim x nc x dim), column-major layout (nc = num_comp)
      // In CEED_TRANSPOSE mode, the sizes of d_u and d_v are switched.

      // E-vector ordering -------------- Q-vector ordering
      //                                  dim
      //  component                        component
      //    elem                              elem
      //       node                            node

      // ---  Define strides for NOTRANSPOSE mode: ---
      // Input (d_u) is E-vector, output (d_v) is Q-vector

      // Element strides
      CeedInt u_elem_stride = elem_dofs_size;
      CeedInt v_elem_stride = elem_qpts_size;
      // Component strides
      CeedInt u_comp_stride = num_elem * elem_dofs_size;
      CeedInt v_comp_stride = num_elem * elem_qpts_size;
      // Dimension strides
      CeedInt u_dim_stride = 0;
      CeedInt v_dim_stride = num_elem * elem_qpts_size * num_comp;
      if (t_mode == CEED_TRANSPOSE) {
        // Input (d_u) is Q-vector, output (d_v) is E-vector
        // Element strides
        v_elem_stride = elem_dofs_size;
        u_elem_stride = elem_qpts_size;
        // Component strides
        v_comp_stride = num_elem * elem_dofs_size;
        u_comp_stride = num_elem * elem_qpts_size;
        // Dimension strides
        v_dim_stride = 0;
        u_dim_stride = num_elem * elem_qpts_size * num_comp;
      }
      CeedInt num_threads = 1;
      CeedInt num_t_col   = 1;
      CeedInt shared_mem  = 0;
      CeedInt max_P_Q     = CeedIntMax(P, Q);

      switch (dim) {
        case 1:
          num_threads = max_P_Q;
          num_t_col   = MAGMA_BASIS_NTCOL(num_threads, MAGMA_MAXTHREADS_1D);
          shared_mem += sizeof(CeedScalar) * num_t_col * (num_comp * (1 * P + 1 * Q));
          shared_mem += sizeof(CeedScalar) * (P * Q);
          break;
        case 2:
          num_threads = max_P_Q;
          num_t_col   = MAGMA_BASIS_NTCOL(num_threads, MAGMA_MAXTHREADS_2D);
          shared_mem += sizeof(CeedScalar) * 2 * P * Q;  // for sTinterp and sTgrad
          // for reforming rU we need P x P, and for the intermediate output we need P x Q
          shared_mem += sizeof(CeedScalar) * num_t_col * (P * max_P_Q);
          break;
        case 3:
          num_threads = max_P_Q * max_P_Q;
          num_t_col   = MAGMA_BASIS_NTCOL(num_threads, MAGMA_MAXTHREADS_3D);
          shared_mem += sizeof(CeedScalar) * 2 * P * Q;  // for sTinterp and sTgrad
          // rU needs P^2 x P, the intermediate outputs need (P^2 x Q + P x Q^2)
          shared_mem += sizeof(CeedScalar) * num_t_col * CeedIntMax(P * P * P, (P * P * Q) + (P * Q * Q));
          break;
      }
      CeedInt grid   = CeedDivUpInt(num_elem, num_t_col);
      void   *args[] = {&impl->d_interp_1d, &impl->d_grad_1d, &d_u,          &u_elem_stride, &u_comp_stride, &u_dim_stride, &d_v,
                        &v_elem_stride,     &v_comp_stride,   &v_dim_stride, &num_elem};

      if (t_mode == CEED_TRANSPOSE) {
        CeedCallBackend(CeedRunKernelDimSharedMagma(ceed, impl->GradTranspose, grid, num_threads, num_t_col, 1, shared_mem, args));
      } else {
        CeedCallBackend(CeedRunKernelDimSharedMagma(ceed, impl->Grad, grid, num_threads, num_t_col, 1, shared_mem, args));
      }
    } break;
    case CEED_EVAL_WEIGHT: {
      CeedCheck(t_mode != CEED_TRANSPOSE, ceed, CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
      CeedInt elem_dofs_size = CeedIntPow(Q, dim);
      CeedInt num_threads    = 1;
      CeedInt num_t_col      = 1;
      CeedInt shared_mem     = 0;

      switch (dim) {
        case 1:
          num_threads = Q;
          num_t_col   = MAGMA_BASIS_NTCOL(num_threads, MAGMA_MAXTHREADS_1D);
          shared_mem += sizeof(CeedScalar) * Q;              // for d_q_weight_1d
          shared_mem += sizeof(CeedScalar) * num_t_col * Q;  // for output
          break;
        case 2:
          num_threads = Q;
          num_t_col   = MAGMA_BASIS_NTCOL(num_threads, MAGMA_MAXTHREADS_2D);
          shared_mem += sizeof(CeedScalar) * Q;  // for d_q_weight_1d
          break;
        case 3:
          num_threads = Q * Q;
          num_t_col   = MAGMA_BASIS_NTCOL(num_threads, MAGMA_MAXTHREADS_3D);
          shared_mem += sizeof(CeedScalar) * Q;  // for d_q_weight_1d
          break;
      }
      CeedInt grid   = CeedDivUpInt(num_elem, num_t_col);
      void   *args[] = {&impl->d_q_weight_1d, &d_v, &elem_dofs_size, &num_elem};

      CeedCallBackend(CeedRunKernelDimSharedMagma(ceed, impl->Weight, grid, num_threads, num_t_col, 1, shared_mem, args));
    } break;
    // LCOV_EXCL_START
    case CEED_EVAL_DIV:
      return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_DIV not supported");
    case CEED_EVAL_CURL:
      return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_CURL not supported");
    case CEED_EVAL_NONE:
      return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_NONE does not make sense in this context");
      // LCOV_EXCL_STOP
  }

  // Must sync to ensure completeness
  ceed_magma_queue_sync(data->queue);

  // Restore vectors
  if (e_mode != CEED_EVAL_WEIGHT) {
    CeedCallBackend(CeedVectorRestoreArrayRead(u, &d_u));
  }
  CeedCallBackend(CeedVectorRestoreArray(v, &d_v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis apply - non-tensor
//------------------------------------------------------------------------------
static int CeedBasisApplyNonTensor_Magma(CeedBasis basis, CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode e_mode, CeedVector u,
                                         CeedVector v) {
  Ceed                      ceed;
  Ceed_Magma               *data;
  CeedInt                   num_comp, q_comp, num_nodes, num_qpts, P, Q, N;
  const CeedScalar         *d_u;
  CeedScalar               *d_v;
  CeedBasisNonTensor_Magma *impl;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedGetData(ceed, &data));
  CeedCallBackend(CeedBasisGetData(basis, &impl));
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, e_mode, &q_comp));
  CeedCallBackend(CeedBasisGetNumNodes(basis, &num_nodes));
  CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis, &num_qpts));
  P = num_nodes;
  Q = num_qpts;
  N = num_elem * num_comp;

  // Read vectors
  if (u != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u));
  else CeedCheck(e_mode == CEED_EVAL_WEIGHT, ceed, CEED_ERROR_BACKEND, "An input vector is required for this CeedEvalMode");
  CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v));

  // Apply basis operation
  if (e_mode != CEED_EVAL_WEIGHT) {
    const CeedScalar *d_b = NULL;
    switch (e_mode) {
      case CEED_EVAL_INTERP:
        d_b = impl->d_interp;
        break;
      case CEED_EVAL_GRAD:
        d_b = impl->d_grad;
        break;
      case CEED_EVAL_DIV:
        d_b = impl->d_div;
        break;
      case CEED_EVAL_CURL:
        d_b = impl->d_curl;
        break;
      // LCOV_EXCL_START
      case CEED_EVAL_WEIGHT:
        return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT does not make sense in this context");
      case CEED_EVAL_NONE:
        return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_NONE does not make sense in this context");
        // LCOV_EXCL_STOP
    }

    // Apply basis operation
    if (P <= MAGMA_NONTENSOR_CUSTOM_KERNEL_MAX_P && Q <= MAGMA_NONTENSOR_CUSTOM_KERNEL_MAX_Q) {
      CeedInt n_array[MAGMA_NONTENSOR_KERNEL_INSTANCES] = {MAGMA_NONTENSOR_KERNEL_N_VALUES};
      CeedInt iN = 0, diff = abs(n_array[iN] - N), idiff;
      CeedInt M = (t_mode == CEED_TRANSPOSE) ? P : Q, K = (t_mode == CEED_TRANSPOSE) ? Q : P;

      for (CeedInt in = iN + 1; in < MAGMA_NONTENSOR_KERNEL_INSTANCES; in++) {
        idiff = abs(n_array[in] - N);
        if (idiff < diff) {
          iN   = in;
          diff = idiff;
        }
      }

      // Compile kernels for N as needed
      if (!impl->NB_interp[iN]) {
        CeedFESpace fe_space;
        CeedInt     q_comp_interp, q_comp_deriv;
        Ceed        ceed_delegate;
        char       *basis_kernel_path, *basis_kernel_source;
        magma_int_t arch = magma_getdevice_arch();

        // Tuning parameters for NB
        CeedCallBackend(CeedBasisGetFESpace(basis, &fe_space));
        CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_INTERP, &q_comp_interp));
        switch (fe_space) {
          case CEED_FE_SPACE_H1:
            CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_GRAD, &q_comp_deriv));
            break;
          case CEED_FE_SPACE_HDIV:
            CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_DIV, &q_comp_deriv));
            break;
          case CEED_FE_SPACE_HCURL:
            CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_CURL, &q_comp_deriv));
            break;
        }
        impl->NB_interp[iN]   = nontensor_rtc_get_nb(arch, 'n', q_comp_interp, P, Q, n_array[iN]);
        impl->NB_interp_t[iN] = nontensor_rtc_get_nb(arch, 't', q_comp_interp, P, Q, n_array[iN]);
        impl->NB_deriv[iN]    = nontensor_rtc_get_nb(arch, 'n', q_comp_deriv, P, Q, n_array[iN]);
        impl->NB_deriv_t[iN]  = nontensor_rtc_get_nb(arch, 't', q_comp_deriv, P, Q, n_array[iN]);

        // The RTC compilation code expects a Ceed with the common Ceed_Cuda or Ceed_Hip data
        CeedCallBackend(CeedGetDelegate(ceed, &ceed_delegate));

        // Compile kernels
        CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/magma/magma-basis-interp-deriv-nontensor.h", &basis_kernel_path));
        CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source -----\n");
        CeedCallBackend(CeedLoadSourceToBuffer(ceed, basis_kernel_path, &basis_kernel_source));
        CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source Complete! -----\n");
        CeedCallBackend(CeedCompileMagma(ceed_delegate, basis_kernel_source, &impl->module_interp[iN], 8, "BASIS_Q_COMP_INTERP", q_comp_interp,
                                         "BASIS_Q_COMP_DERIV", q_comp_deriv, "BASIS_P", P, "BASIS_Q", Q, "BASIS_NB_INTERP_N", impl->NB_interp[iN],
                                         "BASIS_NB_INTERP_T", impl->NB_interp_t[iN], "BASIS_NB_DERIV_N", impl->NB_deriv[iN], "BASIS_NB_DERIV_T",
                                         impl->NB_deriv_t[iN]));
        CeedCallBackend(CeedGetKernelMagma(ceed, impl->module_interp[iN], "magma_interp_nontensor_n", &impl->Interp[iN]));
        CeedCallBackend(CeedGetKernelMagma(ceed, impl->module_interp[iN], "magma_interp_nontensor_t", &impl->InterpTranspose[iN]));
        CeedCallBackend(CeedGetKernelMagma(ceed, impl->module_interp[iN], "magma_deriv_nontensor_n", &impl->Deriv[iN]));
        CeedCallBackend(CeedGetKernelMagma(ceed, impl->module_interp[iN], "magma_deriv_nontensor_t", &impl->DerivTranspose[iN]));
        CeedCallBackend(CeedFree(&basis_kernel_path));
        CeedCallBackend(CeedFree(&basis_kernel_source));
      }
      CeedMagmaFunction Kernel;
      CeedInt           NB;
      if (e_mode == CEED_EVAL_INTERP) {
        if (t_mode == CEED_TRANSPOSE) {
          Kernel = impl->InterpTranspose[iN];
          NB     = impl->NB_interp_t[iN];
        } else {
          Kernel = impl->Interp[iN];
          NB     = impl->NB_interp[iN];
        }
      } else {
        if (t_mode == CEED_TRANSPOSE) {
          Kernel = impl->DerivTranspose[iN];
          NB     = impl->NB_deriv_t[iN];
        } else {
          Kernel = impl->Deriv[iN];
          NB     = impl->NB_deriv[iN];
        }
      }
      CeedInt num_t_col    = MAGMA_BASIS_NTCOL(M, MAGMA_MAXTHREADS_1D);
      CeedInt grid         = CeedDivUpInt(N, num_t_col * NB);
      CeedInt shared_mem_A = P * Q * sizeof(CeedScalar);
      CeedInt shared_mem_B = num_t_col * K * NB * sizeof(CeedScalar);
      CeedInt shared_mem   = (t_mode != CEED_TRANSPOSE && q_comp > 1) ? (shared_mem_A + shared_mem_B) : CeedIntMax(shared_mem_A, shared_mem_B);
      void   *args[]       = {&N, &d_b, &d_u, &d_v};

      CeedCallBackend(CeedRunKernelDimSharedMagma(ceed, Kernel, grid, M, num_t_col, 1, shared_mem, args));
    } else {
      for (CeedInt d = 0; d < q_comp; d++) {
        if (t_mode == CEED_TRANSPOSE) {
          const CeedScalar beta = (d > 0) ? 1.0 : 0.0;
          magma_gemm_nontensor(MagmaNoTrans, MagmaNoTrans, P, N, Q, 1.0, d_b + d * P * Q, P, d_u + d * N * Q, Q, beta, d_v, P, data->queue);
        } else {
          magma_gemm_nontensor(MagmaTrans, MagmaNoTrans, Q, N, P, 1.0, d_b + d * P * Q, P, d_u, P, 0.0, d_v + d * N * Q, Q, data->queue);
        }
      }
    }
  } else {
    CeedCheck(t_mode != CEED_TRANSPOSE, ceed, CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
    CeedInt num_t_col  = MAGMA_BASIS_NTCOL(Q, MAGMA_MAXTHREADS_1D);
    CeedInt grid       = CeedDivUpInt(num_elem, num_t_col);
    CeedInt shared_mem = Q * sizeof(CeedScalar) + num_t_col * Q * sizeof(CeedScalar);
    void   *args[]     = {&num_elem, &impl->d_q_weight, &d_v};

    CeedCallBackend(CeedRunKernelDimSharedMagma(ceed, impl->Weight, grid, Q, num_t_col, 1, shared_mem, args));
  }

  // Must sync to ensure completeness
  ceed_magma_queue_sync(data->queue);

  // Restore vectors
  if (e_mode != CEED_EVAL_WEIGHT) {
    CeedCallBackend(CeedVectorRestoreArrayRead(u, &d_u));
  }
  CeedCallBackend(CeedVectorRestoreArray(v, &d_v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy tensor basis
//------------------------------------------------------------------------------
static int CeedBasisDestroy_Magma(CeedBasis basis) {
  Ceed             ceed;
  CeedBasis_Magma *impl;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedBasisGetData(basis, &impl));
#ifdef CEED_MAGMA_USE_HIP
  CeedCallHip(ceed, hipModuleUnload(impl->module));
#else
  CeedCallCuda(ceed, cuModuleUnload(impl->module));
#endif
  CeedCallBackend(magma_free(impl->d_interp_1d));
  CeedCallBackend(magma_free(impl->d_grad_1d));
  CeedCallBackend(magma_free(impl->d_q_weight_1d));
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy non-tensor basis
//------------------------------------------------------------------------------
static int CeedBasisDestroyNonTensor_Magma(CeedBasis basis) {
  Ceed                      ceed;
  CeedBasisNonTensor_Magma *impl;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedBasisGetData(basis, &impl));
#ifdef CEED_MAGMA_USE_HIP
  CeedCallHip(ceed, hipModuleUnload(impl->module_weight));
#else
  CeedCallCuda(ceed, cuModuleUnload(impl->module_weight));
#endif
  for (CeedInt in = 0; in < MAGMA_NONTENSOR_KERNEL_INSTANCES; in++) {
    if (impl->module_interp[in]) {
#ifdef CEED_MAGMA_USE_HIP
      CeedCallHip(ceed, hipModuleUnload(impl->module_interp[in]));
#else
      CeedCallCuda(ceed, cuModuleUnload(impl->module_interp[in]));
#endif
    }
  }
  CeedCallBackend(magma_free(impl->d_interp));
  CeedCallBackend(magma_free(impl->d_grad));
  CeedCallBackend(magma_free(impl->d_div));
  CeedCallBackend(magma_free(impl->d_curl));
  CeedCallBackend(magma_free(impl->d_q_weight));
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create tensor
//------------------------------------------------------------------------------
int CeedBasisCreateTensorH1_Magma(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                  const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis) {
  Ceed             ceed, ceed_delegate;
  Ceed_Magma      *data;
  char            *interp_kernel_path, *grad_kernel_path, *weight_kernel_path, *basis_kernel_source;
  CeedInt          num_comp;
  CeedBasis_Magma *impl;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedGetData(ceed, &data));
  CeedCallBackend(CeedCalloc(1, &impl));

  // Copy basis data to GPU
  CeedCallBackend(magma_malloc((void **)&impl->d_q_weight_1d, Q_1d * sizeof(q_weight_1d[0])));
  magma_setvector(Q_1d, sizeof(q_weight_1d[0]), q_weight_1d, 1, impl->d_q_weight_1d, 1, data->queue);
  CeedCallBackend(magma_malloc((void **)&impl->d_interp_1d, Q_1d * P_1d * sizeof(interp_1d[0])));
  magma_setvector(Q_1d * P_1d, sizeof(interp_1d[0]), interp_1d, 1, impl->d_interp_1d, 1, data->queue);
  CeedCallBackend(magma_malloc((void **)&impl->d_grad_1d, Q_1d * P_1d * sizeof(grad_1d[0])));
  magma_setvector(Q_1d * P_1d, sizeof(grad_1d[0]), grad_1d, 1, impl->d_grad_1d, 1, data->queue);

  // The RTC compilation code expects a Ceed with the common Ceed_Cuda or Ceed_Hip data
  CeedCallBackend(CeedGetDelegate(ceed, &ceed_delegate));

  // Compile kernels
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  {
    char   *interp_kernel_name_base = "ceed/jit-source/magma/magma-basis-interp";
    CeedInt interp_kernel_name_len  = strlen(interp_kernel_name_base) + 6;
    char    interp_kernel_name[interp_kernel_name_len];

    snprintf(interp_kernel_name, interp_kernel_name_len, "%s-%" CeedInt_FMT "d.h", interp_kernel_name_base, dim);
    CeedCallBackend(CeedGetJitAbsolutePath(ceed, interp_kernel_name, &interp_kernel_path));
  }
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, interp_kernel_path, &basis_kernel_source));
  {
    char   *grad_kernel_name_base = "ceed/jit-source/magma/magma-basis-grad";
    CeedInt grad_kernel_name_len  = strlen(grad_kernel_name_base) + 6;
    char    grad_kernel_name[grad_kernel_name_len];

    snprintf(grad_kernel_name, grad_kernel_name_len, "%s-%" CeedInt_FMT "d.h", grad_kernel_name_base, dim);
    CeedCallBackend(CeedGetJitAbsolutePath(ceed, grad_kernel_name, &grad_kernel_path));
  }
  CeedCallBackend(CeedLoadSourceToInitializedBuffer(ceed, grad_kernel_path, &basis_kernel_source));
  {
    char   *weight_kernel_name_base = "ceed/jit-source/magma/magma-basis-weight";
    CeedInt weight_kernel_name_len  = strlen(weight_kernel_name_base) + 6;
    char    weight_kernel_name[weight_kernel_name_len];

    snprintf(weight_kernel_name, weight_kernel_name_len, "%s-%" CeedInt_FMT "d.h", weight_kernel_name_base, dim);
    CeedCallBackend(CeedGetJitAbsolutePath(ceed, weight_kernel_name, &weight_kernel_path));
  }
  CeedCallBackend(CeedLoadSourceToInitializedBuffer(ceed, weight_kernel_path, &basis_kernel_source));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source Complete! -----\n");
  CeedCallBackend(CeedCompileMagma(ceed_delegate, basis_kernel_source, &impl->module, 5, "BASIS_DIM", dim, "BASIS_NUM_COMP", num_comp, "BASIS_P",
                                   P_1d, "BASIS_Q", Q_1d, "BASIS_MAX_P_Q", CeedIntMax(P_1d, Q_1d)));
  switch (dim) {
    case 1:
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_interpn_1d_kernel", &impl->Interp));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_interpt_1d_kernel", &impl->InterpTranspose));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_gradn_1d_kernel", &impl->Grad));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_gradt_1d_kernel", &impl->GradTranspose));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_weight_1d_kernel", &impl->Weight));
      break;
    case 2:
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_interpn_2d_kernel", &impl->Interp));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_interpt_2d_kernel", &impl->InterpTranspose));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_gradn_2d_kernel", &impl->Grad));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_gradt_2d_kernel", &impl->GradTranspose));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_weight_2d_kernel", &impl->Weight));
      break;
    case 3:
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_interpn_3d_kernel", &impl->Interp));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_interpt_3d_kernel", &impl->InterpTranspose));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_gradn_3d_kernel", &impl->Grad));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_gradt_3d_kernel", &impl->GradTranspose));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_weight_3d_kernel", &impl->Weight));
      break;
  }
  CeedCallBackend(CeedFree(&interp_kernel_path));
  CeedCallBackend(CeedFree(&grad_kernel_path));
  CeedCallBackend(CeedFree(&weight_kernel_path));
  CeedCallBackend(CeedFree(&basis_kernel_source));

  CeedCallBackend(CeedBasisSetData(basis, impl));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApply_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroy_Magma));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create non-tensor H^1
//------------------------------------------------------------------------------
int CeedBasisCreateH1_Magma(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp, const CeedScalar *grad,
                            const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  Ceed                      ceed, ceed_delegate;
  Ceed_Magma               *data;
  char                     *weight_kernel_path, *basis_kernel_source;
  CeedBasisNonTensor_Magma *impl;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedGetData(ceed, &data));
  CeedCallBackend(CeedCalloc(1, &impl));

  // Copy basis data to GPU
  CeedCallBackend(magma_malloc((void **)&impl->d_q_weight, num_qpts * sizeof(q_weight[0])));
  magma_setvector(num_qpts, sizeof(q_weight[0]), q_weight, 1, impl->d_q_weight, 1, data->queue);
  if (interp) {
    CeedInt q_comp_interp;

    CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_INTERP, &q_comp_interp));
    CeedCallBackend(magma_malloc((void **)&impl->d_interp, num_qpts * num_nodes * q_comp_interp * sizeof(interp[0])));
    magma_setvector(num_qpts * num_nodes * q_comp_interp, sizeof(interp[0]), interp, 1, impl->d_interp, 1, data->queue);
  }
  if (grad) {
    CeedInt q_comp_grad;

    CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_GRAD, &q_comp_grad));
    CeedCallBackend(magma_malloc((void **)&impl->d_grad, num_qpts * num_nodes * q_comp_grad * sizeof(grad[0])));
    magma_setvector(num_qpts * num_nodes * q_comp_grad, sizeof(grad[0]), grad, 1, impl->d_grad, 1, data->queue);
  }

  // The RTC compilation code expects a Ceed with the common Ceed_Cuda or Ceed_Hip data
  CeedCallBackend(CeedGetDelegate(ceed, &ceed_delegate));

  // Compile weight kernel (the remainder of kernel compilation happens at first call to CeedBasisApply)
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/magma/magma-basis-weight-nontensor.h", &weight_kernel_path));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, weight_kernel_path, &basis_kernel_source));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source Complete! -----\n");
  CeedCallBackend(CeedCompileMagma(ceed_delegate, basis_kernel_source, &impl->module_weight, 1, "BASIS_Q", num_qpts));
  CeedCallBackend(CeedGetKernelMagma(ceed, impl->module_weight, "magma_weight_nontensor", &impl->Weight));
  CeedCallBackend(CeedFree(&weight_kernel_path));
  CeedCallBackend(CeedFree(&basis_kernel_source));

  CeedCallBackend(CeedBasisSetData(basis, impl));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroyNonTensor_Magma));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create non-tensor H(div)
//------------------------------------------------------------------------------
int CeedBasisCreateHdiv_Magma(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                              const CeedScalar *div, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  Ceed                      ceed, ceed_delegate;
  Ceed_Magma               *data;
  char                     *weight_kernel_path, *basis_kernel_source;
  CeedBasisNonTensor_Magma *impl;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedGetData(ceed, &data));
  CeedCallBackend(CeedCalloc(1, &impl));

  // Copy basis data to GPU
  CeedCallBackend(magma_malloc((void **)&impl->d_q_weight, num_qpts * sizeof(q_weight[0])));
  magma_setvector(num_qpts, sizeof(q_weight[0]), q_weight, 1, impl->d_q_weight, 1, data->queue);
  if (interp) {
    CeedInt q_comp_interp;

    CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_INTERP, &q_comp_interp));
    CeedCallBackend(magma_malloc((void **)&impl->d_interp, num_qpts * num_nodes * q_comp_interp * sizeof(interp[0])));
    magma_setvector(num_qpts * num_nodes * q_comp_interp, sizeof(interp[0]), interp, 1, impl->d_interp, 1, data->queue);
  }
  if (div) {
    CeedInt q_comp_div;

    CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_DIV, &q_comp_div));
    CeedCallBackend(magma_malloc((void **)&impl->d_div, num_qpts * num_nodes * q_comp_div * sizeof(div[0])));
    magma_setvector(num_qpts * num_nodes * q_comp_div, sizeof(div[0]), div, 1, impl->d_div, 1, data->queue);
  }

  // The RTC compilation code expects a Ceed with the common Ceed_Cuda or Ceed_Hip data
  CeedCallBackend(CeedGetDelegate(ceed, &ceed_delegate));

  // Compile weight kernel (the remainder of kernel compilation happens at first call to CeedBasisApply)
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/magma/magma-basis-weight-nontensor.h", &weight_kernel_path));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, weight_kernel_path, &basis_kernel_source));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source Complete! -----\n");
  CeedCallBackend(CeedCompileMagma(ceed_delegate, basis_kernel_source, &impl->module_weight, 1, "BASIS_Q", num_qpts));
  CeedCallBackend(CeedGetKernelMagma(ceed, impl->module_weight, "magma_weight_nontensor", &impl->Weight));
  CeedCallBackend(CeedFree(&weight_kernel_path));
  CeedCallBackend(CeedFree(&basis_kernel_source));

  CeedCallBackend(CeedBasisSetData(basis, impl));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroyNonTensor_Magma));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create non-tensor H(curl)
//------------------------------------------------------------------------------
int CeedBasisCreateHcurl_Magma(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                               const CeedScalar *curl, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  Ceed                      ceed, ceed_delegate;
  Ceed_Magma               *data;
  char                     *weight_kernel_path, *basis_kernel_source;
  CeedBasisNonTensor_Magma *impl;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedGetData(ceed, &data));
  CeedCallBackend(CeedCalloc(1, &impl));

  // Copy basis data to GPU
  CeedCallBackend(magma_malloc((void **)&impl->d_q_weight, num_qpts * sizeof(q_weight[0])));
  magma_setvector(num_qpts, sizeof(q_weight[0]), q_weight, 1, impl->d_q_weight, 1, data->queue);
  if (interp) {
    CeedInt q_comp_interp;

    CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_INTERP, &q_comp_interp));
    CeedCallBackend(magma_malloc((void **)&impl->d_interp, num_qpts * num_nodes * q_comp_interp * sizeof(interp[0])));
    magma_setvector(num_qpts * num_nodes * q_comp_interp, sizeof(interp[0]), interp, 1, impl->d_interp, 1, data->queue);
  }
  if (curl) {
    CeedInt q_comp_curl;

    CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_CURL, &q_comp_curl));
    CeedCallBackend(magma_malloc((void **)&impl->d_curl, num_qpts * num_nodes * q_comp_curl * sizeof(curl[0])));
    magma_setvector(num_qpts * num_nodes * q_comp_curl, sizeof(curl[0]), curl, 1, impl->d_curl, 1, data->queue);
  }

  // The RTC compilation code expects a Ceed with the common Ceed_Cuda or Ceed_Hip data
  CeedCallBackend(CeedGetDelegate(ceed, &ceed_delegate));

  // Compile weight kernel (the remainder of kernel compilation happens at first call to CeedBasisApply)
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/magma/magma-basis-weight-nontensor.h", &weight_kernel_path));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, weight_kernel_path, &basis_kernel_source));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source Complete! -----\n");
  CeedCallBackend(CeedCompileMagma(ceed_delegate, basis_kernel_source, &impl->module_weight, 1, "BASIS_Q", num_qpts));
  CeedCallBackend(CeedGetKernelMagma(ceed, impl->module_weight, "magma_weight_nontensor", &impl->Weight));
  CeedCallBackend(CeedFree(&weight_kernel_path));
  CeedCallBackend(CeedFree(&basis_kernel_source));

  CeedCallBackend(CeedBasisSetData(basis, impl));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroyNonTensor_Magma));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
