// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-tools.h>

#include <map>
#include <string_view>
#include <sycl/sycl.hpp>

#include "../sycl/ceed-sycl-compile.hpp"
#include "ceed-sycl-shared.hpp"

//------------------------------------------------------------------------------
// Compute the local range of for basis kernels
//------------------------------------------------------------------------------
static int ComputeLocalRange(Ceed ceed, CeedInt dim, CeedInt thread_1d, CeedInt *local_range, CeedInt max_group_size = 256) {
  local_range[0] = thread_1d;
  local_range[1] = (dim > 1) ? thread_1d : 1;

  const CeedInt min_group_size = local_range[0] * local_range[1];
  if (min_group_size > max_group_size) {
    return CeedError(ceed, CEED_ERROR_BACKEND, "Requested group size is smaller than the required minimum.");
  }

  local_range[2] = max_group_size / min_group_size;  // elements per group
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply basis
//------------------------------------------------------------------------------
int CeedBasisApplyTensor_Sycl_shared(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
                                     CeedVector v) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  Ceed_Sycl *ceed_Sycl;
  CeedCallBackend(CeedGetData(ceed, &ceed_Sycl));
  CeedBasis_Sycl_shared *impl;
  CeedCallBackend(CeedBasisGetData(basis, &impl));

  // Read vectors
  const CeedScalar *d_u;
  CeedScalar       *d_v;
  if (eval_mode != CEED_EVAL_WEIGHT) {
    CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u));
  }
  CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v));

  // Apply basis operation
  switch (eval_mode) {
    case CEED_EVAL_INTERP: {
      CeedInt       *lrange         = impl->interp_local_range;
      const CeedInt &elem_per_group = lrange[2];
      const CeedInt  group_count    = (num_elem / elem_per_group) + !!(num_elem % elem_per_group);
      //-----------
      sycl::range<3>    local_range(lrange[2], lrange[1], lrange[0]);
      sycl::range<3>    global_range(group_count * lrange[2], lrange[1], lrange[0]);
      sycl::nd_range<3> kernel_range(global_range, local_range);
      //-----------
      sycl::kernel *interp_kernel = (t_mode == CEED_TRANSPOSE) ? impl->interp_transpose_kernel : impl->interp_kernel;

      // Order queue
      sycl::event e = ceed_Sycl->sycl_queue.ext_oneapi_submit_barrier();

      ceed_Sycl->sycl_queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(e);
        cgh.set_args(num_elem, impl->d_interp_1d, d_u, d_v);
        cgh.parallel_for(kernel_range, *interp_kernel);
      });

    } break;
    case CEED_EVAL_GRAD: {
      CeedInt       *lrange         = impl->grad_local_range;
      const CeedInt &elem_per_group = lrange[2];
      const CeedInt  group_count    = (num_elem / elem_per_group) + !!(num_elem % elem_per_group);
      //-----------
      sycl::range<3>    local_range(lrange[2], lrange[1], lrange[0]);
      sycl::range<3>    global_range(group_count * lrange[2], lrange[1], lrange[0]);
      sycl::nd_range<3> kernel_range(global_range, local_range);
      //-----------
      sycl::kernel     *grad_kernel = (t_mode == CEED_TRANSPOSE) ? impl->grad_transpose_kernel : impl->grad_kernel;
      const CeedScalar *d_grad_1d   = (impl->d_collo_grad_1d) ? impl->d_collo_grad_1d : impl->d_grad_1d;
      // Order queue
      sycl::event e = ceed_Sycl->sycl_queue.ext_oneapi_submit_barrier();

      ceed_Sycl->sycl_queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(e);
        cgh.set_args(num_elem, impl->d_interp_1d, d_grad_1d, d_u, d_v);
        cgh.parallel_for(kernel_range, *grad_kernel);
      });
    } break;
    case CEED_EVAL_WEIGHT: {
      CeedInt       *lrange         = impl->weight_local_range;
      const CeedInt &elem_per_group = lrange[2];
      const CeedInt  group_count    = (num_elem / elem_per_group) + !!(num_elem % elem_per_group);
      //-----------
      sycl::range<3>    local_range(lrange[2], lrange[1], lrange[0]);
      sycl::range<3>    global_range(group_count * lrange[2], lrange[1], lrange[0]);
      sycl::nd_range<3> kernel_range(global_range, local_range);
      //-----------
      // Order queue
      sycl::event e = ceed_Sycl->sycl_queue.ext_oneapi_submit_barrier();

      ceed_Sycl->sycl_queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(e);
        cgh.set_args(num_elem, impl->d_q_weight_1d, d_v);
        cgh.parallel_for(kernel_range, *(impl->weight_kernel));
      });
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
// Destroy basis
//------------------------------------------------------------------------------
static int CeedBasisDestroy_Sycl_shared(CeedBasis basis) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedBasis_Sycl_shared *impl;
  CeedCallBackend(CeedBasisGetData(basis, &impl));
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  CeedCallSycl(ceed, data->sycl_queue.wait_and_throw());
  CeedCallSycl(ceed, sycl::free(impl->d_q_weight_1d, data->sycl_context));
  CeedCallSycl(ceed, sycl::free(impl->d_interp_1d, data->sycl_context));
  CeedCallSycl(ceed, sycl::free(impl->d_grad_1d, data->sycl_context));
  CeedCallSycl(ceed, sycl::free(impl->d_collo_grad_1d, data->sycl_context));

  delete impl->interp_kernel;
  delete impl->interp_transpose_kernel;
  delete impl->grad_kernel;
  delete impl->grad_transpose_kernel;
  delete impl->weight_kernel;
  delete impl->sycl_module;

  CeedCallBackend(CeedFree(&impl));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create tensor basis
// TODO: Refactor
//------------------------------------------------------------------------------
int CeedBasisCreateTensorH1_Sycl_shared(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                        const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedBasis_Sycl_shared *impl;
  CeedCallBackend(CeedCalloc(1, &impl));
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  CeedInt num_comp;
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));

  const CeedInt thread_1d = CeedIntMax(Q_1d, P_1d);
  const CeedInt num_nodes = CeedIntPow(P_1d, dim);
  const CeedInt num_qpts  = CeedIntPow(Q_1d, dim);

  CeedInt *interp_lrange = impl->interp_local_range;
  CeedCallBackend(ComputeLocalRange(ceed, dim, thread_1d, interp_lrange));
  const CeedInt interp_group_size = interp_lrange[0] * interp_lrange[1] * interp_lrange[2];

  CeedInt *grad_lrange = impl->grad_local_range;
  CeedCallBackend(ComputeLocalRange(ceed, dim, thread_1d, grad_lrange));
  const CeedInt grad_group_size = grad_lrange[0] * grad_lrange[1] * grad_lrange[2];

  CeedCallBackend(ComputeLocalRange(ceed, dim, Q_1d, impl->weight_local_range));

  // Copy basis data to GPU
  CeedCallSycl(ceed, impl->d_q_weight_1d = sycl::malloc_device<CeedScalar>(Q_1d, data->sycl_device, data->sycl_context));
  sycl::event copy_weight = data->sycl_queue.copy<CeedScalar>(q_weight_1d, impl->d_q_weight_1d, Q_1d);

  const CeedInt interp_length = Q_1d * P_1d;
  CeedCallSycl(ceed, impl->d_interp_1d = sycl::malloc_device<CeedScalar>(interp_length, data->sycl_device, data->sycl_context));
  sycl::event copy_interp = data->sycl_queue.copy<CeedScalar>(interp_1d, impl->d_interp_1d, interp_length);

  CeedCallSycl(ceed, impl->d_grad_1d = sycl::malloc_device<CeedScalar>(interp_length, data->sycl_device, data->sycl_context));
  sycl::event copy_grad = data->sycl_queue.copy<CeedScalar>(grad_1d, impl->d_grad_1d, interp_length);

  CeedCallSycl(ceed, sycl::event::wait_and_throw({copy_weight, copy_interp, copy_grad}));

  // Compute collocated gradient and copy to GPU
  impl->d_collo_grad_1d          = NULL;
  const bool has_collocated_grad = (dim == 3) && (Q_1d >= P_1d);
  if (has_collocated_grad) {
    CeedScalar *collo_grad_1d;
    CeedCallBackend(CeedMalloc(Q_1d * Q_1d, &collo_grad_1d));
    CeedCallBackend(CeedBasisGetCollocatedGrad(basis, collo_grad_1d));
    const CeedInt cgrad_length = Q_1d * Q_1d;
    CeedCallSycl(ceed, impl->d_collo_grad_1d = sycl::malloc_device<CeedScalar>(cgrad_length, data->sycl_device, data->sycl_context));
    CeedCallSycl(ceed, data->sycl_queue.copy<CeedScalar>(collo_grad_1d, impl->d_collo_grad_1d, cgrad_length).wait_and_throw());
    CeedCallBackend(CeedFree(&collo_grad_1d));
  }

  // ---[Refactor into separate function]------>
  // Define compile-time constants
  std::map<std::string, CeedInt> jit_constants;
  jit_constants["BASIS_DIM"]                 = dim;
  jit_constants["BASIS_Q_1D"]                = Q_1d;
  jit_constants["BASIS_P_1D"]                = P_1d;
  jit_constants["T_1D"]                      = thread_1d;
  jit_constants["BASIS_NUM_COMP"]            = num_comp;
  jit_constants["BASIS_NUM_NODES"]           = num_nodes;
  jit_constants["BASIS_NUM_QPTS"]            = num_qpts;
  jit_constants["BASIS_HAS_COLLOCATED_GRAD"] = has_collocated_grad;
  jit_constants["BASIS_INTERP_SCRATCH_SIZE"] = interp_group_size;
  jit_constants["BASIS_GRAD_SCRATCH_SIZE"]   = grad_group_size;

  // Load kernel source
  char *basis_kernel_path, *basis_kernel_source;
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/sycl/sycl-shared-basis-tensor.h", &basis_kernel_path));
  CeedDebug256(ceed, 2, "----- Loading Basis Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, basis_kernel_path, &basis_kernel_source));
  CeedDebug256(ceed, 2, "----- Loading Basis Kernel Source Complete -----\n");

  // Compile kernels into a kernel bundle
  CeedCallBackend(CeedJitBuildModule_Sycl(ceed, basis_kernel_source, &impl->sycl_module, jit_constants));

  // Load kernel functions
  CeedCallBackend(CeedJitGetKernel_Sycl(ceed, impl->sycl_module, "Interp", &impl->interp_kernel));
  CeedCallBackend(CeedJitGetKernel_Sycl(ceed, impl->sycl_module, "InterpTranspose", &impl->interp_transpose_kernel));
  CeedCallBackend(CeedJitGetKernel_Sycl(ceed, impl->sycl_module, "Grad", &impl->grad_kernel));
  CeedCallBackend(CeedJitGetKernel_Sycl(ceed, impl->sycl_module, "GradTranspose", &impl->grad_transpose_kernel));
  CeedCallBackend(CeedJitGetKernel_Sycl(ceed, impl->sycl_module, "Weight", &impl->weight_kernel));

  // Clean-up
  CeedCallBackend(CeedFree(&basis_kernel_path));
  CeedCallBackend(CeedFree(&basis_kernel_source));
  // <---[Refactor into separate function]------

  CeedCallBackend(CeedBasisSetData(basis, impl));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Basis", basis, "Apply", CeedBasisApplyTensor_Sycl_shared));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Basis", basis, "Destroy", CeedBasisDestroy_Sycl_shared));
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
