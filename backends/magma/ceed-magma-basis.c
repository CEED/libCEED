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

#ifdef __cplusplus
CEED_INTERN "C"
#endif
    int
    CeedBasisApply_Magma(CeedBasis basis, CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode e_mode, CeedVector U, CeedVector V) {
  Ceed              ceed;
  Ceed_Magma       *data;
  CeedInt           dim, num_comp, num_dof, P_1d, Q_1d;
  const CeedScalar *du;
  CeedScalar       *dv;
  CeedBasis_Magma  *impl;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedBasisGetDimension(basis, &dim));
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedBasisGetNumNodes(basis, &num_dof));

  CeedCallBackend(CeedGetData(ceed, &data));

  if (U != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorGetArrayRead(U, CEED_MEM_DEVICE, &du));
  else CeedCheck(e_mode == CEED_EVAL_WEIGHT, ceed, CEED_ERROR_BACKEND, "An input vector is required for this CeedEvalMode");
  CeedCallBackend(CeedVectorGetArrayWrite(V, CEED_MEM_DEVICE, &dv));

  CeedCallBackend(CeedBasisGetData(basis, &impl));

  CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
  CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));

  CeedDebug256(ceed, 4, "[CeedBasisApply_Magma] vsize=%" CeedInt_FMT ", comp = %" CeedInt_FMT, num_comp * CeedIntPow(P_1d, dim), num_comp);

  if (t_mode == CEED_TRANSPOSE) {
    CeedSize length;

    CeedCallBackend(CeedVectorGetLength(V, &length));
    if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
      magmablas_slaset(MagmaFull, length, 1, 0., 0., (float *)dv, length, data->queue);
    } else {
      magmablas_dlaset(MagmaFull, length, 1, 0., 0., (double *)dv, length, data->queue);
    }
    ceed_magma_queue_sync(data->queue);
  }

  switch (e_mode) {
    case CEED_EVAL_INTERP: {
      CeedInt P = P_1d, Q = Q_1d;

      if (t_mode == CEED_TRANSPOSE) {
        P = Q_1d;
        Q = P_1d;
      }

      // Define element sizes for dofs/quad
      CeedInt elem_qpts_size = CeedIntPow(Q_1d, dim);
      CeedInt elem_dofs_size = CeedIntPow(P_1d, dim);

      // E-vector ordering -------------- Q-vector ordering
      //  component                        component
      //    elem                             elem
      //       node                            node

      // ---  Define strides for NOTRANSPOSE mode: ---
      // Input (du) is E-vector, output (dv) is Q-vector

      // Element strides
      CeedInt u_elem_stride = elem_dofs_size;
      CeedInt v_elem_stride = elem_qpts_size;
      // Component strides
      CeedInt u_comp_stride = num_elem * elem_dofs_size;
      CeedInt v_comp_stride = num_elem * elem_qpts_size;

      // ---  Swap strides for TRANSPOSE mode: ---
      if (t_mode == CEED_TRANSPOSE) {
        // Input (du) is Q-vector, output (dv) is E-vector
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
          shared_mem += P * Q * sizeof(CeedScalar);                      // for sT
          shared_mem += num_t_col * (P * max_P_Q * sizeof(CeedScalar));  // for reforming rU we need PxP, and for the intermediate output we need PxQ
          break;
        case 3:
          num_threads = max_P_Q * max_P_Q;
          num_t_col   = MAGMA_BASIS_NTCOL(num_threads, MAGMA_MAXTHREADS_3D);
          shared_mem += sizeof(CeedScalar) * (P * Q);  // for sT
          shared_mem += sizeof(CeedScalar) * num_t_col *
                        (CeedIntMax(P * P * max_P_Q,
                                    P * Q * Q));  // rU needs P^2xP, the intermediate output needs max(P^2xQ,PQ^2)
      }
      CeedInt grid   = (num_elem + num_t_col - 1) / num_t_col;
      void   *args[] = {&impl->d_interp_1d, &du, &u_elem_stride, &u_comp_stride, &dv, &v_elem_stride, &v_comp_stride, &num_elem};

      if (t_mode == CEED_TRANSPOSE) {
        CeedCallBackend(CeedRunKernelDimSharedMagma(ceed, impl->magma_interp_tr, grid, num_threads, num_t_col, 1, shared_mem, args));
      } else {
        CeedCallBackend(CeedRunKernelDimSharedMagma(ceed, impl->magma_interp, grid, num_threads, num_t_col, 1, shared_mem, args));
      }
    } break;
    case CEED_EVAL_GRAD: {
      CeedInt P = P_1d, Q = Q_1d;

      // In CEED_NOTRANSPOSE mode:
      // du is (P^dim x nc), column-major layout (nc = num_comp)
      // dv is (Q^dim x nc x dim), column-major layout (nc = num_comp)
      // In CEED_TRANSPOSE mode, the sizes of du and dv are switched.
      if (t_mode == CEED_TRANSPOSE) {
        P = Q_1d;
        Q = P_1d;
      }

      // Define element sizes for dofs/quad
      CeedInt elem_qpts_size = CeedIntPow(Q_1d, dim);
      CeedInt elem_dofs_size = CeedIntPow(P_1d, dim);

      // E-vector ordering -------------- Q-vector ordering
      //                                  dim
      //  component                        component
      //    elem                              elem
      //       node                            node

      // ---  Define strides for NOTRANSPOSE mode: ---
      // Input (du) is E-vector, output (dv) is Q-vector

      // Element strides
      CeedInt u_elem_stride = elem_dofs_size;
      CeedInt v_elem_stride = elem_qpts_size;
      // Component strides
      CeedInt u_comp_stride = num_elem * elem_dofs_size;
      CeedInt v_comp_stride = num_elem * elem_qpts_size;
      // Dimension strides
      CeedInt u_dim_stride = 0;
      CeedInt v_dim_stride = num_elem * elem_qpts_size * num_comp;

      // ---  Swap strides for TRANSPOSE mode: ---
      if (t_mode == CEED_TRANSPOSE) {
        // Input (du) is Q-vector, output (dv) is E-vector
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
          shared_mem += sizeof(CeedScalar) * 2 * P * Q;                  // for sTinterp and sTgrad
          shared_mem += sizeof(CeedScalar) * num_t_col * (P * max_P_Q);  // for reforming rU we need PxP, and for the intermediate output we need PxQ
          break;
        case 3:
          num_threads = max_P_Q * max_P_Q;
          num_t_col   = MAGMA_BASIS_NTCOL(num_threads, MAGMA_MAXTHREADS_3D);
          shared_mem += sizeof(CeedScalar) * 2 * P * Q;  // for sTinterp and sTgrad
          shared_mem += sizeof(CeedScalar) * num_t_col *
                        CeedIntMax(P * P * P,
                                   (P * P * Q) + (P * Q * Q));  // rU needs P^2xP, the intermediate outputs need (P^2.Q + P.Q^2)
      }
      CeedInt grid   = (num_elem + num_t_col - 1) / num_t_col;
      void   *args[] = {&impl->d_interp_1d, &impl->d_grad_1d, &du,           &u_elem_stride, &u_comp_stride, &u_dim_stride, &dv,
                        &v_elem_stride,     &v_comp_stride,   &v_dim_stride, &num_elem};

      if (t_mode == CEED_TRANSPOSE) {
        CeedCallBackend(CeedRunKernelDimSharedMagma(ceed, impl->magma_grad_tr, grid, num_threads, num_t_col, 1, shared_mem, args));
      } else {
        CeedCallBackend(CeedRunKernelDimSharedMagma(ceed, impl->magma_grad, grid, num_threads, num_t_col, 1, shared_mem, args));
      }
    } break;
    case CEED_EVAL_WEIGHT: {
      CeedCheck(t_mode != CEED_TRANSPOSE, ceed, CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT inum_compatible with CEED_TRANSPOSE");
      CeedInt Q              = Q_1d;
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
      }
      CeedInt grid   = (num_elem + num_t_col - 1) / num_t_col;
      void   *args[] = {&impl->d_q_weight_1d, &dv, &elem_dofs_size, &num_elem};

      CeedCallBackend(CeedRunKernelDimSharedMagma(ceed, impl->magma_weight, grid, num_threads, num_t_col, 1, shared_mem, args));
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

  // must sync to ensure completeness
  ceed_magma_queue_sync(data->queue);

  if (e_mode != CEED_EVAL_WEIGHT) {
    CeedCallBackend(CeedVectorRestoreArrayRead(U, &du));
  }
  CeedCallBackend(CeedVectorRestoreArray(V, &dv));
  return CEED_ERROR_SUCCESS;
}

#ifdef __cplusplus
CEED_INTERN "C"
#endif
    int
    CeedBasisApplyNonTensor_Magma(CeedBasis basis, CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode e_mode, CeedVector U, CeedVector V) {
  Ceed                      ceed;
  Ceed_Magma               *data;
  CeedInt                   dim, num_comp, num_dof, num_qpts, NB = 1;
  const CeedScalar         *du;
  CeedScalar               *dv;
  CeedBasisNonTensor_Magma *impl;
  CeedMagmaFunction        *interp, *grad;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedGetData(ceed, &data));
  magma_int_t arch = magma_getdevice_arch();

  CeedCallBackend(CeedBasisGetDimension(basis, &dim));
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedBasisGetNumNodes(basis, &num_dof));
  CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis, &num_qpts));
  CeedInt P = num_dof, Q = num_qpts, N = num_elem * num_comp;

  if (U != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorGetArrayRead(U, CEED_MEM_DEVICE, &du));
  else CeedCheck(e_mode == CEED_EVAL_WEIGHT, ceed, CEED_ERROR_BACKEND, "An input vector is required for this CeedEvalMode");
  CeedCallBackend(CeedVectorGetArrayWrite(V, CEED_MEM_DEVICE, &dv));

  CeedCallBackend(CeedBasisGetData(basis, &impl));

  CeedDebug256(ceed, 4, "[CeedBasisApplyNonTensor_Magma] vsize=%" CeedInt_FMT ", comp = %" CeedInt_FMT, num_comp * num_dof, num_comp);

  if (t_mode == CEED_TRANSPOSE) {
    CeedSize length;

    CeedCallBackend(CeedVectorGetLength(V, &length));
    if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
      magmablas_slaset(MagmaFull, length, 1, 0., 0., (float *)dv, length, data->queue);
    } else {
      magmablas_dlaset(MagmaFull, length, 1, 0., 0., (double *)dv, length, data->queue);
    }
    ceed_magma_queue_sync(data->queue);
  }

  CeedInt n_array[MAGMA_NONTENSOR_KERNEL_INSTANCES] = {MAGMA_NONTENSOR_N_VALUES};
  CeedInt iN                                        = 0;
  CeedInt diff                                      = abs(n_array[iN] - N);

  for (CeedInt in = iN + 1; in < MAGMA_NONTENSOR_KERNEL_INSTANCES; in++) {
    CeedInt idiff = abs(n_array[in] - N);
    if (idiff < diff) {
      iN   = in;
      diff = idiff;
    }
  }

  NB     = nontensor_rtc_get_nb(arch, 'd', e_mode, t_mode, P, n_array[iN], Q);
  interp = (t_mode == CEED_TRANSPOSE) ? &impl->magma_interp_tr_nontensor[iN] : &impl->magma_interp_nontensor[iN];
  grad   = (t_mode == CEED_TRANSPOSE) ? &impl->magma_grad_tr_nontensor[iN] : &impl->magma_grad_nontensor[iN];

  switch (e_mode) {
    case CEED_EVAL_INTERP: {
      CeedInt P = num_dof, Q = num_qpts;
      if (P < MAGMA_NONTENSOR_CUSTOM_KERNEL_MAX_P && Q < MAGMA_NONTENSOR_CUSTOM_KERNEL_MAX_Q) {
        CeedInt M          = (t_mode == CEED_TRANSPOSE) ? P : Q;
        CeedInt K          = (t_mode == CEED_TRANSPOSE) ? Q : P;
        CeedInt num_t_col  = MAGMA_NONTENSOR_BASIS_NTCOL(M);
        CeedInt shared_mem = 0, shared_mem_A = 0, shared_mem_B = 0;
        shared_mem_B += num_t_col * K * NB * sizeof(CeedScalar);
        shared_mem_A += (t_mode == CEED_TRANSPOSE) ? 0 : K * M * sizeof(CeedScalar);
        shared_mem = (t_mode == CEED_TRANSPOSE) ? (shared_mem_A + shared_mem_B) : CeedIntMax(shared_mem_A, shared_mem_B);

        CeedInt       grid    = MAGMA_CEILDIV(MAGMA_CEILDIV(N, NB), num_t_col);
        magma_trans_t trans_A = (t_mode == CEED_TRANSPOSE) ? MagmaNoTrans : MagmaTrans;
        magma_trans_t trans_B = MagmaNoTrans;
        CeedScalar    alpha = 1.0, beta = 0.0;

        void *args[] = {&trans_A, &trans_B, &N, &alpha, &impl->d_interp, &P, &du, &K, &beta, &dv, &M};
        CeedCallBackend(CeedRunKernelDimSharedMagma(ceed, *interp, grid, M, num_t_col, 1, shared_mem, args));
      } else {
        if (t_mode == CEED_TRANSPOSE) {
          magma_gemm_nontensor(MagmaNoTrans, MagmaNoTrans, P, num_elem * num_comp, Q, 1.0, impl->d_interp, P, du, Q, 0.0, dv, P, data->queue);
        } else {
          magma_gemm_nontensor(MagmaTrans, MagmaNoTrans, Q, num_elem * num_comp, P, 1.0, impl->d_interp, P, du, P, 0.0, dv, Q, data->queue);
        }
      }
    } break;

    case CEED_EVAL_GRAD: {
      CeedInt P = num_dof, Q = num_qpts;
      if (P < MAGMA_NONTENSOR_CUSTOM_KERNEL_MAX_P && Q < MAGMA_NONTENSOR_CUSTOM_KERNEL_MAX_Q) {
        CeedInt M          = (t_mode == CEED_TRANSPOSE) ? P : Q;
        CeedInt K          = (t_mode == CEED_TRANSPOSE) ? Q : P;
        CeedInt num_t_col  = MAGMA_NONTENSOR_BASIS_NTCOL(M);
        CeedInt shared_mem = 0, shared_mem_A = 0, shared_mem_B = 0;
        shared_mem_B += num_t_col * K * NB * sizeof(CeedScalar);
        shared_mem_A += (t_mode == CEED_TRANSPOSE) ? 0 : K * M * sizeof(CeedScalar);
        shared_mem = shared_mem_A + shared_mem_B;

        CeedInt       grid    = MAGMA_CEILDIV(MAGMA_CEILDIV(N, NB), num_t_col);
        magma_trans_t trans_A = (t_mode == CEED_TRANSPOSE) ? MagmaNoTrans : MagmaTrans;
        magma_trans_t trans_B = MagmaNoTrans;

        void *args[] = {&trans_A, &trans_B, &N, &impl->d_grad, &P, &du, &K, &dv, &M};
        CeedCallBackend(CeedRunKernelDimSharedMagma(ceed, *grad, grid, M, num_t_col, 1, shared_mem, args));
      } else {
        if (t_mode == CEED_TRANSPOSE) {
          CeedScalar beta = 0.0;
          for (int d = 0; d < dim; d++) {
            if (d > 0) beta = 1.0;
            magma_gemm_nontensor(MagmaNoTrans, MagmaNoTrans, P, num_elem * num_comp, Q, 1.0, impl->d_grad + d * P * Q, P,
                                 du + d * num_elem * num_comp * Q, Q, beta, dv, P, data->queue);
          }
        } else {
          for (int d = 0; d < dim; d++)
            magma_gemm_nontensor(MagmaTrans, MagmaNoTrans, Q, num_elem * num_comp, P, 1.0, impl->d_grad + d * P * Q, P, du, P, 0.0,
                                 dv + d * num_elem * num_comp * Q, Q, data->queue);
        }
      }
    } break;

    case CEED_EVAL_WEIGHT: {
      CeedCheck(t_mode != CEED_TRANSPOSE, ceed, CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT inum_compatible with CEED_TRANSPOSE");

      int elemsPerBlock = 1;  // basis->Q_1d < 7 ? optElems[basis->Q_1d] : 1;
      int grid          = num_elem / elemsPerBlock + ((num_elem / elemsPerBlock * elemsPerBlock < num_elem) ? 1 : 0);

      magma_weight_nontensor(grid, num_qpts, num_elem, num_qpts, impl->d_q_weight, dv, data->queue);
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

  // must sync to ensure completeness
  ceed_magma_queue_sync(data->queue);

  if (e_mode != CEED_EVAL_WEIGHT) {
    CeedCallBackend(CeedVectorRestoreArrayRead(U, &du));
  }
  CeedCallBackend(CeedVectorRestoreArray(V, &dv));
  return CEED_ERROR_SUCCESS;
}

#ifdef __cplusplus
CEED_INTERN "C"
#endif
    int
    CeedBasisDestroy_Magma(CeedBasis basis) {
  Ceed             ceed;
  CeedBasis_Magma *impl;

  CeedCallBackend(CeedBasisGetData(basis, &impl));
  CeedCallBackend(magma_free(impl->d_q_ref_1d));
  CeedCallBackend(magma_free(impl->d_interp_1d));
  CeedCallBackend(magma_free(impl->d_grad_1d));
  CeedCallBackend(magma_free(impl->d_q_weight_1d));
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
#ifdef CEED_MAGMA_USE_HIP
  CeedCallHip(ceed, hipModuleUnload(impl->module));
#else
  CeedCallCuda(ceed, cuModuleUnload(impl->module));
#endif
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

#ifdef __cplusplus
CEED_INTERN "C"
#endif
    int
    CeedBasisDestroyNonTensor_Magma(CeedBasis basis) {
  Ceed                      ceed;
  CeedBasisNonTensor_Magma *impl;

  CeedCallBackend(CeedBasisGetData(basis, &impl));
  CeedCallBackend(magma_free(impl->d_q_ref));
  CeedCallBackend(magma_free(impl->d_interp));
  CeedCallBackend(magma_free(impl->d_grad));
  CeedCallBackend(magma_free(impl->d_q_weight));
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
#ifdef CEED_MAGMA_USE_HIP
  for (CeedInt in = 0; in < MAGMA_NONTENSOR_KERNEL_INSTANCES; in++) {
    CeedCallHip(ceed, hipModuleUnload(impl->module[in]));
  }
#else
  for (CeedInt in = 0; in < MAGMA_NONTENSOR_KERNEL_INSTANCES; in++) {
    CeedCallCuda(ceed, cuModuleUnload(impl->module[in]));
  }
#endif
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

#ifdef __cplusplus
CEED_INTERN "C"
#endif
    int
    CeedBasisCreateTensorH1_Magma(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                  const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis) {
  Ceed             ceed, ceed_delegate;
  Ceed_Magma      *data;
  char            *magma_common_path, *interp_path, *grad_path, *weight_path, *basis_kernel_source;
  CeedInt          num_comp = 0;
  CeedBasis_Magma *impl;

  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));

  // Check for supported parameters
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedGetData(ceed, &data));

  // Compile kernels
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/magma/magma_common_defs.h", &magma_common_path));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, magma_common_path, &basis_kernel_source));
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/magma/magma_common_tensor.h", &magma_common_path));
  CeedCallBackend(CeedLoadSourceToInitializedBuffer(ceed, magma_common_path, &basis_kernel_source));
  char   *interp_name_base = "ceed/jit-source/magma/interp";
  CeedInt interp_name_len  = strlen(interp_name_base) + 6;
  char    interp_name[interp_name_len];

  snprintf(interp_name, interp_name_len, "%s-%" CeedInt_FMT "d.h", interp_name_base, dim);
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, interp_name, &interp_path));
  CeedCallBackend(CeedLoadSourceToInitializedBuffer(ceed, interp_path, &basis_kernel_source));
  char   *grad_name_base = "ceed/jit-source/magma/grad";
  CeedInt grad_name_len  = strlen(grad_name_base) + 6;
  char    grad_name[grad_name_len];

  snprintf(grad_name, grad_name_len, "%s-%" CeedInt_FMT "d.h", grad_name_base, dim);
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, grad_name, &grad_path));
  CeedCallBackend(CeedLoadSourceToInitializedBuffer(ceed, grad_path, &basis_kernel_source));
  char   *weight_name_base = "ceed/jit-source/magma/weight";
  CeedInt weight_name_len  = strlen(weight_name_base) + 6;
  char    weight_name[weight_name_len];

  snprintf(weight_name, weight_name_len, "%s-%" CeedInt_FMT "d.h", weight_name_base, dim);
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, weight_name, &weight_path));
  CeedCallBackend(CeedLoadSourceToInitializedBuffer(ceed, weight_path, &basis_kernel_source));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source Complete! -----\n");
  // The RTC compilation code expects a Ceed with the common Ceed_Cuda or Ceed_Hip
  // data
  CeedCallBackend(CeedGetDelegate(ceed, &ceed_delegate));
  CeedCallBackend(CeedCompileMagma(ceed_delegate, basis_kernel_source, &impl->module, 5, "DIM", dim, "NCOMP", num_comp, "P", P_1d, "Q", Q_1d, "MAXPQ",
                                   CeedIntMax(P_1d, Q_1d)));

  // Kernel setup
  switch (dim) {
    case 1:
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_interpn_1d_kernel", &impl->magma_interp));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_interpt_1d_kernel", &impl->magma_interp_tr));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_gradn_1d_kernel", &impl->magma_grad));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_gradt_1d_kernel", &impl->magma_grad_tr));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_weight_1d_kernel", &impl->magma_weight));
      break;
    case 2:
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_interpn_2d_kernel", &impl->magma_interp));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_interpt_2d_kernel", &impl->magma_interp_tr));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_gradn_2d_kernel", &impl->magma_grad));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_gradt_2d_kernel", &impl->magma_grad_tr));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_weight_2d_kernel", &impl->magma_weight));
      break;
    case 3:
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_interpn_3d_kernel", &impl->magma_interp));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_interpt_3d_kernel", &impl->magma_interp_tr));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_gradn_3d_kernel", &impl->magma_grad));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_gradt_3d_kernel", &impl->magma_grad_tr));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_weight_3d_kernel", &impl->magma_weight));
  }

  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApply_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroy_Magma));

  // Copy q_ref_1d to the GPU
  CeedCallBackend(magma_malloc((void **)&impl->d_q_ref_1d, Q_1d * sizeof(q_ref_1d[0])));
  magma_setvector(Q_1d, sizeof(q_ref_1d[0]), q_ref_1d, 1, impl->d_q_ref_1d, 1, data->queue);

  // Copy interp_1d to the GPU
  CeedCallBackend(magma_malloc((void **)&impl->d_interp_1d, Q_1d * P_1d * sizeof(interp_1d[0])));
  magma_setvector(Q_1d * P_1d, sizeof(interp_1d[0]), interp_1d, 1, impl->d_interp_1d, 1, data->queue);

  // Copy grad_1d to the GPU
  CeedCallBackend(magma_malloc((void **)&impl->d_grad_1d, Q_1d * P_1d * sizeof(grad_1d[0])));
  magma_setvector(Q_1d * P_1d, sizeof(grad_1d[0]), grad_1d, 1, impl->d_grad_1d, 1, data->queue);

  // Copy q_weight_1d to the GPU
  CeedCallBackend(magma_malloc((void **)&impl->d_q_weight_1d, Q_1d * sizeof(q_weight_1d[0])));
  magma_setvector(Q_1d, sizeof(q_weight_1d[0]), q_weight_1d, 1, impl->d_q_weight_1d, 1, data->queue);

  CeedCallBackend(CeedBasisSetData(basis, impl));
  CeedCallBackend(CeedFree(&magma_common_path));
  CeedCallBackend(CeedFree(&interp_path));
  CeedCallBackend(CeedFree(&grad_path));
  CeedCallBackend(CeedFree(&weight_path));
  CeedCallBackend(CeedFree(&basis_kernel_source));
  return CEED_ERROR_SUCCESS;
}

#ifdef __cplusplus
CEED_INTERN "C"
#endif
    int
    CeedBasisCreateH1_Magma(CeedElemTopology topo, CeedInt dim, CeedInt num_dof, CeedInt num_qpts, const CeedScalar *interp, const CeedScalar *grad,
                            const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  Ceed                      ceed, ceed_delegate;
  Ceed_Magma               *data;
  char                     *magma_common_path, *interp_path, *grad_path, *basis_kernel_source;
  CeedBasisNonTensor_Magma *impl;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedGetData(ceed, &data));
  magma_int_t arch = magma_getdevice_arch();

  CeedCallBackend(CeedCalloc(1, &impl));
  // Compile kernels
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/magma/magma_common_defs.h", &magma_common_path));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, magma_common_path, &basis_kernel_source));
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/magma/magma_common_nontensor.h", &magma_common_path));
  CeedCallBackend(CeedLoadSourceToInitializedBuffer(ceed, magma_common_path, &basis_kernel_source));
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/magma/interp-nontensor.h", &interp_path));
  CeedCallBackend(CeedLoadSourceToInitializedBuffer(ceed, interp_path, &basis_kernel_source));
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/magma/grad-nontensor.h", &grad_path));
  CeedCallBackend(CeedLoadSourceToInitializedBuffer(ceed, grad_path, &basis_kernel_source));

  // tuning parameters for nb
  CeedInt nb_interp_n[MAGMA_NONTENSOR_KERNEL_INSTANCES];
  CeedInt nb_interp_t[MAGMA_NONTENSOR_KERNEL_INSTANCES];
  CeedInt nb_grad_n[MAGMA_NONTENSOR_KERNEL_INSTANCES];
  CeedInt nb_grad_t[MAGMA_NONTENSOR_KERNEL_INSTANCES];
  CeedInt P = num_dof, Q = num_qpts;
  CeedInt n_array[MAGMA_NONTENSOR_KERNEL_INSTANCES] = {MAGMA_NONTENSOR_N_VALUES};

  for (CeedInt in = 0; in < MAGMA_NONTENSOR_KERNEL_INSTANCES; in++) {
    nb_interp_n[in] = nontensor_rtc_get_nb(arch, 'd', CEED_EVAL_INTERP, CEED_NOTRANSPOSE, P, n_array[in], Q);
    nb_interp_t[in] = nontensor_rtc_get_nb(arch, 'd', CEED_EVAL_INTERP, CEED_TRANSPOSE, P, n_array[in], Q);
    nb_grad_n[in]   = nontensor_rtc_get_nb(arch, 'd', CEED_EVAL_GRAD, CEED_NOTRANSPOSE, P, n_array[in], Q);
    nb_grad_t[in]   = nontensor_rtc_get_nb(arch, 'd', CEED_EVAL_GRAD, CEED_TRANSPOSE, P, n_array[in], Q);
  }

  // compile
  CeedCallBackend(CeedGetDelegate(ceed, &ceed_delegate));
  for (CeedInt in = 0; in < MAGMA_NONTENSOR_KERNEL_INSTANCES; in++) {
    CeedCallBackend(CeedCompileMagma(ceed_delegate, basis_kernel_source, &impl->module[in], 7, "DIM", dim, "P", P, "Q", Q, "NB_INTERP_N",
                                     nb_interp_n[in], "NB_INTERP_T", nb_interp_t[in], "NB_GRAD_N", nb_grad_n[in], "NB_GRAD_T", nb_grad_t[in]));
  }

  // get kernels
  for (CeedInt in = 0; in < MAGMA_NONTENSOR_KERNEL_INSTANCES; in++) {
    CeedCallBackend(CeedGetKernelMagma(ceed, impl->module[in], "magma_interp_nontensor_n", &impl->magma_interp_nontensor[in]));
    CeedCallBackend(CeedGetKernelMagma(ceed, impl->module[in], "magma_interp_nontensor_t", &impl->magma_interp_tr_nontensor[in]));
    CeedCallBackend(CeedGetKernelMagma(ceed, impl->module[in], "magma_grad_nontensor_n", &impl->magma_grad_nontensor[in]));
    CeedCallBackend(CeedGetKernelMagma(ceed, impl->module[in], "magma_grad_nontensor_t", &impl->magma_grad_tr_nontensor[in]));
  }

  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroyNonTensor_Magma));

  // Copy q_ref to the GPU
  CeedCallBackend(magma_malloc((void **)&impl->d_q_ref, num_qpts * sizeof(q_ref[0])));
  magma_setvector(num_qpts, sizeof(q_ref[0]), q_ref, 1, impl->d_q_ref, 1, data->queue);

  // Copy interp to the GPU
  CeedCallBackend(magma_malloc((void **)&impl->d_interp, num_qpts * num_dof * sizeof(interp[0])));
  magma_setvector(num_qpts * num_dof, sizeof(interp[0]), interp, 1, impl->d_interp, 1, data->queue);

  // Copy grad to the GPU
  CeedCallBackend(magma_malloc((void **)&impl->d_grad, num_qpts * num_dof * dim * sizeof(grad[0])));
  magma_setvector(num_qpts * num_dof * dim, sizeof(grad[0]), grad, 1, impl->d_grad, 1, data->queue);

  // Copy q_weight to the GPU
  CeedCallBackend(magma_malloc((void **)&impl->d_q_weight, num_qpts * sizeof(q_weight[0])));
  magma_setvector(num_qpts, sizeof(q_weight[0]), q_weight, 1, impl->d_q_weight, 1, data->queue);

  CeedCallBackend(CeedBasisSetData(basis, impl));
  CeedCallBackend(CeedFree(&magma_common_path));
  CeedCallBackend(CeedFree(&interp_path));
  CeedCallBackend(CeedFree(&grad_path));
  CeedCallBackend(CeedFree(&basis_kernel_source));
  return CEED_ERROR_SUCCESS;
}
