// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#define CEED_DEBUG_COLOR 12

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-source/sycl/sycl-types.h>
#include <ceed/jit-tools.h>

#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "../sycl-ref/ceed-sycl-ref.hpp"
#include "../sycl-shared/ceed-sycl-shared.hpp"
#include "../sycl/ceed-sycl-compile.hpp"

#include "ceed-sycl-gen.hpp"

//------------------------------------------------------------------------------
// Calculate the block size used for launching the operator kernel
//------------------------------------------------------------------------------
extern "C" int BlockGridCalculate_Sycl_gen(const CeedInt dim, const CeedInt P_1d, const CeedInt Q_1d, CeedInt *block_sizes) {
  const CeedInt thread1d = CeedIntMax(Q_1d, P_1d);

  if (dim == 1) {
    CeedInt elems_per_block = 64 * thread1d > 256 ? 256 / thread1d : 64;

    elems_per_block = elems_per_block > 0 ? elems_per_block : 1;
    block_sizes[0]  = thread1d;
    block_sizes[1]  = 1;
    block_sizes[2]  = elems_per_block;
  } else if (dim == 2) {
    const CeedInt elems_per_block = thread1d < 4 ? 16 : 2;

    block_sizes[0] = thread1d;
    block_sizes[1] = thread1d;
    block_sizes[2] = elems_per_block;
  } else if (dim == 3) {
    const CeedInt elems_per_block = thread1d < 6 ? 4 : (thread1d < 8 ? 2 : 1);

    block_sizes[0] = thread1d;
    block_sizes[1] = thread1d;
    block_sizes[2] = elems_per_block;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Build single operator kernel
// - [ ] Check arguments to device functions reudsed from sycl-shared-basis are correct
// - [ ] Do kernel jitting!
//------------------------------------------------------------------------------
extern "C" int CeedOperatorBuildKernel_Sycl_gen(CeedOperator op) {
  Ceed                      ceed;
  Ceed_Sycl                *sycl_data;
  bool                      is_setup_done, is_identity_qf;
  CeedSize                  l_size;
  CeedInt                   Q, P_1d = 0, Q_1d = 0, elem_size, num_input_fields, num_output_fields, num_comp, dim = 1;
  Fields_Sycl               h_B, h_G;
  FieldsInt_Sycl            h_indices;
  CeedEvalMode              eval_mode;
  CeedElemRestriction       elem_rstr;
  CeedElemRestriction_Sycl *rstr_impl;
  CeedBasis                 basis;
  CeedBasis_Sycl_shared    *basis_impl;
  CeedQFunctionField       *qf_input_fields, *qf_output_fields;
  CeedQFunction_Sycl_gen   *qf_impl;
  CeedQFunction             qf;
  CeedOperatorField        *op_input_fields, *op_output_fields;
  CeedOperator_Sycl_gen    *impl;

  CeedCallBackend(CeedOperatorIsSetupDone(op, &is_setup_done));
  if (is_setup_done) return CEED_ERROR_SUCCESS;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedGetData(ceed, &sycl_data));

  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedQFunctionGetData(qf, &qf_impl));
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  Q_1d = Q;

  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));

  // Check for restriction only identity operator
  CeedCallBackend(CeedQFunctionIsIdentity(qf, &is_identity_qf));
  if (is_identity_qf) {
    CeedEvalMode eval_mode_in, eval_mode_out;

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[0], &eval_mode_in));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[0], &eval_mode_out));
    CeedCheck(eval_mode_in != CEED_EVAL_NONE || eval_mode_out != CEED_EVAL_NONE, ceed, CEED_ERROR_BACKEND,
              "Backend does not implement restriction only identity operators");
  }

  std::ostringstream code;
  // TODO: generalize to accept different device functions?
  {
    char       *tensor_basis_code;
    const char *tensor_basis_kernel_path;

    CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/sycl/sycl-shared-basis-tensor-templates.h", &tensor_basis_kernel_path));
    CeedDebug256(ceed, 2, "----- Loading Tensor Basis Kernel Source -----\n");
    CeedCallBackend(CeedLoadSourceToBuffer(ceed, tensor_basis_kernel_path, &tensor_basis_code));
    code << tensor_basis_code;
    CeedCallBackend(CeedFree(&tensor_basis_kernel_path));
    CeedCallBackend(CeedFree(&tensor_basis_code));
  }
  {
    char       *sycl_gen_template_source;
    const char *sycl_gen_template_path;

    CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/sycl/sycl-gen-templates.h", &sycl_gen_template_path));
    CeedDebug256(ceed, 2, "----- Loading Sycl-Gen Template Source -----\n");
    CeedCallBackend(CeedLoadSourceToBuffer(ceed, sycl_gen_template_path, &sycl_gen_template_source));
    code << sycl_gen_template_source;
    CeedCallBackend(CeedFree(&sycl_gen_template_path));
    CeedCallBackend(CeedFree(&sycl_gen_template_source));
  }

  std::string_view  qfunction_source(qf_impl->qfunction_source);
  std::string_view  qfunction_name(qf_impl->qfunction_name);
  const std::string operator_name = "CeedKernelSyclGenOperator_" + std::string(qfunction_name);

  // Find dim, P_1d, Q_1d
  impl->max_P_1d = 0;
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
    if (basis != CEED_BASIS_NONE) {
      bool is_tensor;

      CeedCallBackend(CeedBasisGetData(basis, &basis_impl));
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));

      // Collect dim, P_1d, and Q_1d
      CeedCallBackend(CeedBasisGetDimension(basis, &dim));
      CeedCallBackend(CeedBasisIsTensor(basis, &is_tensor));
      if (is_tensor) {
        CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
        CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
        if (P_1d > impl->max_P_1d) impl->max_P_1d = P_1d;
      } else {
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_BACKEND, "Backend does not implement operators with non-tensor basis");
        // LCOV_EXCL_STOP
      }
    }
  }
  // Check output bases for Q_1d, dim as well
  //   The only input basis might be CEED_BASIS_NONE
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));

    if (basis != CEED_BASIS_NONE) {
      bool is_tensor;

      CeedCallBackend(CeedBasisGetData(basis, &basis_impl));
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));

      // Collect Q_1d
      CeedCallBackend(CeedBasisGetDimension(basis, &dim));
      CeedCallBackend(CeedBasisIsTensor(basis, &is_tensor));
      if (is_tensor) {
        CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
      } else {
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_BACKEND, "Backend does not implement operators with non-tensor basis");
        // LCOV_EXCL_STOP
      }
    }
  }
  impl->dim  = dim;
  impl->Q_1d = Q_1d;

  // Only use 3D collocated gradient parallelization strategy when gradient is computed
  // TODO: put in a function?
  bool use_collograd_parallelization = false;

  if (dim == 3) {
    bool was_grad_found = false;

    for (CeedInt i = 0; i < num_input_fields; i++) {
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
      if (eval_mode == CEED_EVAL_GRAD) {
        CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
        CeedCallBackend(CeedBasisGetData(basis, &basis_impl));
        use_collograd_parallelization = basis_impl->d_collo_grad_1d && (was_grad_found ? use_collograd_parallelization : true);
        was_grad_found                = true;
      }
    }
    for (CeedInt i = 0; i < num_output_fields; i++) {
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
      if (eval_mode == CEED_EVAL_GRAD) {
        CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
        CeedCallBackend(CeedBasisGetData(basis, &basis_impl));
        use_collograd_parallelization = basis_impl->d_collo_grad_1d && (was_grad_found ? use_collograd_parallelization : true);
        was_grad_found                = true;
      }
    }
  }

  CeedInt block_sizes[3];
  CeedCallBackend(BlockGridCalculate_Sycl_gen(dim, P_1d, Q_1d, block_sizes));

  // Define CEED_Q_VLA
  code << "\n#undef CEED_Q_VLA\n";
  if (dim != 3 || use_collograd_parallelization) {
    code << "#define CEED_Q_VLA 1\n\n";
  } else {
    code << "#define CEED_Q_VLA " << Q_1d << "\n\n";
  }

  // Determine subgroup size based on supported sizes : Default : 16 (if supported)
  std::vector allowed_sg_sizes  = sycl_data->sycl_device.get_info<sycl::info::device::sub_group_sizes>();
  CeedInt     sub_group_size_op = allowed_sg_sizes[allowed_sg_sizes.size() - 1];
  for (const auto &s : allowed_sg_sizes) {
    if (s == 16) {
      sub_group_size_op = s;
      break;
    }
  }

  code << qfunction_source;

  // Kernel function
  code << "\n// -----------------------------------------------------------------------------\n";
  code << "__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, GROUP_SIZE_Z), intel_reqd_sub_group_size(" << sub_group_size_op << ")))\n";
  code << "kernel void " << operator_name << "(";
  code << "const CeedInt num_elem, ";
  code << "global void* ctx, ";
  code << "global const FieldsInt_Sycl* indices, ";
  code << "global Fields_Sycl* fields, ";
  code << "global const Fields_Sycl* B, ";
  code << "global const Fields_Sycl* G, ";
  code << "global const CeedScalar * restrict W";
  code << ") {\n";

  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    if (eval_mode != CEED_EVAL_WEIGHT) {  // Skip CEED_EVAL_WEIGHT
      code << "  global const CeedScalar* d_u_" << i << " = fields->inputs[" << i << "];\n";
    }
  }

  for (CeedInt i = 0; i < num_output_fields; i++) {
    code << "  global CeedScalar* d_v_" << i << " = fields->outputs[" << i << "];\n";
  }

  // TODO: Convert these to defined constants to save on GRF
  code << "  const CeedInt DIM = " << dim << ";\n";
  code << "  const CeedInt Q_1D = " << Q_1d << ";\n";

  const CeedInt scratch_size = block_sizes[0] * block_sizes[1] * block_sizes[2];
  code << "  local CeedScalar scratch[" << scratch_size << "];\n";
  code << "  local CeedScalar * elem_scratch = scratch + get_local_id(2) * T_1D" << (dim > 1 ? "*T_1D" : "") << ";\n";

  code << "\n  // -- Input field constants and basis data --\n";
  // Initialize constants, and matrices B and G
  for (CeedInt i = 0; i < num_input_fields; i++) {
    code << "  // ---- Input field " << i << " ----\n";
    // Get elem_size, eval_mode, num_comp
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &elem_rstr));
    CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    CeedCallBackend(CeedElemRestrictionGetNumComponents(elem_rstr, &num_comp));

    // Set field constants
    if (eval_mode != CEED_EVAL_WEIGHT) {
      CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
      if (basis != CEED_BASIS_NONE) {
        CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
        code << "  const CeedInt P_in_" << i << " = " << P_1d << ";\n";
      } else {
        code << "  const CeedInt P_in_" << i << " = " << Q_1d << ";\n";
      }
      code << "  const CeedInt num_comp_in_" << i << " = " << num_comp << ";\n";
    }

    // Load basis data
    code << "  // EvalMode: " << CeedEvalModes[eval_mode] << "\n";
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        break;
      case CEED_EVAL_INTERP:
        CeedCallBackend(CeedBasisGetData(basis, &basis_impl));
        h_B.inputs[i] = basis_impl->d_interp_1d;
        code << "  local CeedScalar s_B_in_" << i << "[" << P_1d * Q_1d << "];\n";
        code << "  loadMatrix(P_in_" << i << "*Q_1D, B->inputs[" << i << "], s_B_in_" << i << ");\n";
        break;
      case CEED_EVAL_GRAD:
        CeedCallBackend(CeedBasisGetData(basis, &basis_impl));
        h_B.inputs[i] = basis_impl->d_interp_1d;
        code << "  local CeedScalar s_B_in_" << i << "[" << P_1d * Q_1d << "];\n";
        code << "  loadMatrix(P_in_" << i << "*Q_1D, B->inputs[" << i << "], s_B_in_" << i << ");\n";
        if (use_collograd_parallelization) {
          h_G.inputs[i] = basis_impl->d_collo_grad_1d;
          code << "  local CeedScalar s_G_in_" << i << "[" << Q_1d * Q_1d << "];\n";
          code << "  loadMatrix(Q_1D*Q_1D, G->inputs[" << i << "], s_G_in_" << i << ");\n";
        } else {
          bool has_collo_grad = basis_impl->d_collo_grad_1d;
          h_G.inputs[i]       = has_collo_grad ? basis_impl->d_collo_grad_1d : basis_impl->d_grad_1d;
          code << "  local CeedScalar s_G_in_" << i << "[" << Q_1d * (has_collo_grad ? Q_1d : P_1d) << "];\n";
          code << "  loadMatrix(" << (has_collo_grad ? "Q_1D" : ("P_in_" + std::to_string(i))) << "*Q_1D, G->inputs[" << i << "], s_G_in_" << i
               << ");\n";
        }
        break;
      case CEED_EVAL_WEIGHT:
        break;  // No action
      case CEED_EVAL_DIV:
        break;  // TODO: Not implemented
      case CEED_EVAL_CURL:
        break;  // TODO: Not implemented
    }
  }

  code << "\n  // -- Output field constants and basis data --\n";
  for (CeedInt i = 0; i < num_output_fields; i++) {
    code << "  // ---- Output field " << i << " ----\n";
    // Get elem_size, eval_mode, num_comp
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[i], &elem_rstr));
    CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    CeedCallBackend(CeedElemRestrictionGetNumComponents(elem_rstr, &num_comp));

    // Set field constants
    CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
    if (basis != CEED_BASIS_NONE) {
      CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
      code << "  const CeedInt P_out_" << i << " = " << P_1d << ";\n";
    } else {
      code << "  const CeedInt P_out_" << i << " = " << Q_1d << ";\n";
    }
    code << "  const CeedInt num_comp_out_" << i << " = " << num_comp << ";\n";

    // Load basis data
    code << "  // EvalMode: " << CeedEvalModes[eval_mode] << "\n";
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        break;  // No action
      case CEED_EVAL_INTERP:
        CeedCallBackend(CeedBasisGetData(basis, &basis_impl));
        h_B.outputs[i] = basis_impl->d_interp_1d;
        code << "  local CeedScalar s_B_out_" << i << "[" << P_1d * Q_1d << "];\n";
        code << "  loadMatrix(P_out_" << i << "*Q_1D, B->outputs[" << i << "], s_B_out_" << i << ");\n";
        break;
      case CEED_EVAL_GRAD:
        CeedCallBackend(CeedBasisGetData(basis, &basis_impl));
        h_B.outputs[i] = basis_impl->d_interp_1d;
        code << "  local CeedScalar s_B_out_" << i << "[" << P_1d * Q_1d << "];\n";
        code << "  loadMatrix(P_out_" << i << "*Q_1D, B->outputs[" << i << "], s_B_out_" << i << ");\n";
        if (use_collograd_parallelization) {
          h_G.outputs[i] = basis_impl->d_collo_grad_1d;
          code << "  local CeedScalar s_G_out_" << i << "[" << Q_1d * Q_1d << "];\n";
          code << "  loadMatrix(Q_1D*Q_1D, G->outputs[" << i << "], s_G_out_" << i << ");\n";
        } else {
          bool has_collo_grad = basis_impl->d_collo_grad_1d;
          h_G.outputs[i]      = has_collo_grad ? basis_impl->d_collo_grad_1d : basis_impl->d_grad_1d;
          code << "  local CeedScalar s_G_out_" << i << "[" << Q_1d * (has_collo_grad ? Q_1d : P_1d) << "];\n";
          code << "  loadMatrix(" << (has_collo_grad ? "Q_1D" : ("P_out_" + std::to_string(i))) << "*Q_1D, G->outputs[" << i << "], s_G_out_" << i
               << ");\n";
        }
        break;
      // LCOV_EXCL_START
      case CEED_EVAL_WEIGHT: {
        Ceed ceed;

        CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
        return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
        break;  // Should not occur
      }
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL: {
        Ceed ceed;

        CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
        return CeedError(ceed, CEED_ERROR_BACKEND, "%s not supported", CeedEvalModes[eval_mode]);
        break;  // Should not occur
      }
        // LCOV_EXCL_STOP
    }
  }
  code << "\n  // -- Element loop --\n";
  code << "  work_group_barrier(CLK_LOCAL_MEM_FENCE);\n";
  code << "  {\n";
  // Input basis apply if needed
  // Generate the correct eval mode code for each input
  code << "    // -- Input field restrictions and basis actions --\n";
  for (CeedInt i = 0; i < num_input_fields; i++) {
    code << "    // ---- Input field " << i << " ----\n";
    // Get elem_size, eval_mode, num_comp
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &elem_rstr));
    CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    CeedCallBackend(CeedElemRestrictionGetNumComponents(elem_rstr, &num_comp));

    // Restriction
    if (eval_mode != CEED_EVAL_WEIGHT && !((eval_mode == CEED_EVAL_NONE) && use_collograd_parallelization)) {
      bool is_strided;

      code << "    CeedScalar r_u_" << i << "[num_comp_in_" << i << "*P_in_" << i << "];\n";

      CeedCallBackend(CeedElemRestrictionIsStrided(elem_rstr, &is_strided));
      if (!is_strided) {
        CeedInt comp_stride;

        CeedCallBackend(CeedElemRestrictionGetLVectorSize(elem_rstr, &l_size));
        code << "    const CeedInt l_size_in_" << i << " = " << l_size << ";\n";
        CeedCallBackend(CeedElemRestrictionGetCompStride(elem_rstr, &comp_stride));
        code << "    // CompStride: " << comp_stride << "\n";
        CeedCallBackend(CeedElemRestrictionGetData(elem_rstr, &rstr_impl));
        h_indices.inputs[i] = rstr_impl->d_ind;
        code << "    readDofsOffset" << dim << "d(num_comp_in_" << i << ", " << comp_stride << ", P_in_" << i << ", num_elem, indices->inputs[" << i
             << "], d_u_" << i << ", r_u_" << i << ");\n";
      } else {
        bool    has_backend_strides;
        CeedInt num_elem;

        CeedCallBackend(CeedElemRestrictionHasBackendStrides(elem_rstr, &has_backend_strides));
        CeedCallBackend(CeedElemRestrictionGetNumElements(elem_rstr, &num_elem));
        CeedInt strides[3] = {1, elem_size * num_elem, elem_size};

        if (!has_backend_strides) {
          CeedCallBackend(CeedElemRestrictionGetStrides(elem_rstr, strides));
        }
        code << "    // Strides: {" << strides[0] << ", " << strides[1] << ", " << strides[2] << "}\n";
        code << "    readDofsStrided" << dim << "d(num_comp_in_" << i << ",P_in_" << i << "," << strides[0] << "," << strides[1] << "," << strides[2]
             << ", num_elem, d_u_" << i << ", r_u_" << i << ");\n";
      }
    }

    // Basis action
    code << "    // EvalMode: " << CeedEvalModes[eval_mode] << "\n";
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        if (!use_collograd_parallelization) {
          code << "    private CeedScalar* r_t_" << i << " = r_u_" << i << ";\n";
        }
        break;
      case CEED_EVAL_INTERP:
        code << "    CeedScalar r_t_" << i << "[num_comp_in_" << i << "*Q_1D];\n";
        code << "    Interp" << (dim > 1 ? "Tensor" : "") << dim << "d(num_comp_in_" << i << ", P_in_" << i << ", Q_1D, r_u_" << i << ", s_B_in_" << i
             << ", r_t_" << i << ", elem_scratch);\n";
        break;
      case CEED_EVAL_GRAD:
        if (use_collograd_parallelization) {
          code << "    CeedScalar r_t_" << i << "[num_comp_in_" << i << "*Q_1D];\n";
          code << "    Interp" << (dim > 1 ? "Tensor" : "") << dim << "d(num_comp_in_" << i << ", P_in_" << i << ", Q_1D, r_u_" << i << ", s_B_in_"
               << i << ", r_t_" << i << ", elem_scratch);\n";
        } else {
          CeedInt P_1d;
          CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
          CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
          code << "    CeedScalar r_t_" << i << "[num_comp_in_" << i << "*DIM*Q_1D];\n";
          code << "    Grad" << (dim > 1 ? "Tensor" : "") << (dim == 3 && Q_1d >= P_1d ? "Collocated" : "") << dim << "d(num_comp_in_" << i
               << ", P_in_" << i << ", Q_1D, r_u_" << i << (dim > 1 ? ", s_B_in_" : "") << (dim > 1 ? std::to_string(i) : "") << ", s_G_in_" << i
               << ", r_t_" << i << ", elem_scratch);\n";
        }
        break;
      case CEED_EVAL_WEIGHT:
        code << "    CeedScalar r_t_" << i << "[Q_1D];\n";
        CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
        CeedCallBackend(CeedBasisGetData(basis, &basis_impl));
        impl->W = basis_impl->d_q_weight_1d;
        code << "    Weight" << (dim > 1 ? "Tensor" : "") << dim << "d(Q_1D, W, r_t_" << i << ");\n";
        break;  // No action
      case CEED_EVAL_DIV:
        break;  // TODO: Not implemented
      case CEED_EVAL_CURL:
        break;  // TODO: Not implemented
    }
  }

  // Q function
  code << "\n    // -- Output field setup --\n";
  for (CeedInt i = 0; i < num_output_fields; i++) {
    code << "\n    // ---- Output field " << i << " ----\n";
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_GRAD) {
      if (use_collograd_parallelization) {
        // Accumulator for gradient slices
        code << "    CeedScalar r_tt_" << i << "[num_comp_out_" << i << "*Q_1D];\n";
        code << "    for (CeedInt i = 0; i < num_comp_out_" << i << "; i++) {\n";
        code << "      for (CeedInt j = 0; j < Q_1D; ++j) {\n";
        code << "        r_tt_" << i << "[j + i*Q_1D] = 0.0;\n";
        code << "      }\n";
        code << "    }\n";
      } else {
        code << "    CeedScalar r_tt_" << i << "[num_comp_out_" << i << "*DIM*Q_1D];\n";
      }
    }
    if (eval_mode == CEED_EVAL_NONE || eval_mode == CEED_EVAL_INTERP) {
      code << "    CeedScalar r_tt_" << i << "[num_comp_out_" << i << "*Q_1D];\n";
    }
  }
  // We treat quadrature points per slice in 3d to save registers
  if (use_collograd_parallelization) {
    code << "\n    // Note: Using planes of 3D elements\n";
    code << "    for (CeedInt q = 0; q < Q_1D; q++) {\n";
    code << "      // -- Input fields --\n";
    for (CeedInt i = 0; i < num_input_fields; i++) {
      code << "      // ---- Input field " << i << " ----\n";
      // Get elem_size, eval_mode, num_comp
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
      // Basis action
      code << "      // EvalMode: " << CeedEvalModes[eval_mode] << "\n";
      switch (eval_mode) {
        case CEED_EVAL_NONE:
          bool is_strided;

          code << "      CeedScalar r_q_" << i << "[num_comp_in_" << i << "];\n";

          CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &elem_rstr));
          CeedCallBackend(CeedElemRestrictionIsStrided(elem_rstr, &is_strided));
          if (!is_strided) {
            CeedInt comp_stride;

            CeedCallBackend(CeedElemRestrictionGetLVectorSize(elem_rstr, &l_size));
            code << "      const CeedInt l_size_in_" << i << " = " << l_size << ";\n";
            CeedCallBackend(CeedElemRestrictionGetCompStride(elem_rstr, &comp_stride));
            code << "      // CompStride: " << comp_stride << "\n";
            CeedCallBackend(CeedElemRestrictionGetData(elem_rstr, &rstr_impl));
            h_indices.inputs[i] = rstr_impl->d_ind;
            code << "      readSliceQuadsOffset"
                 << "3d(num_comp_in_" << i << ", " << comp_stride << ", Q_1D, l_size_in_" << i << ", num_elem, q, indices->inputs[" << i << "], d_u_"
                 << i << ", r_q_" << i << ");\n";
          } else {
            bool    has_backend_strides;
            CeedInt num_elem;

            CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
            CeedCallBackend(CeedElemRestrictionHasBackendStrides(elem_rstr, &has_backend_strides));
            CeedCallBackend(CeedElemRestrictionGetNumElements(elem_rstr, &num_elem));
            CeedInt strides[3] = {1, elem_size * num_elem, elem_size};

            if (!has_backend_strides) {
              CeedCallBackend(CeedElemRestrictionGetStrides(elem_rstr, strides));
            }
            code << "      // Strides: {" << strides[0] << ", " << strides[1] << ", " << strides[2] << "}\n";
            code << "      readSliceQuadsStrided"
                 << "3d(num_comp_in_" << i << ", Q_1D," << strides[0] << ", " << strides[1] << ", " << strides[2] << ", num_elem, q, d_u_" << i
                 << ", r_q_" << i << ");\n";
          }
          break;
        case CEED_EVAL_INTERP:
          code << "      CeedScalar r_q_" << i << "[num_comp_in_" << i << "];\n";
          code << "      for (CeedInt j = 0; j < num_comp_in_" << i << " ; ++j) {\n";
          code << "        r_q_" << i << "[j] = r_t_" << i << "[q + j*Q_1D];\n";
          code << "      }\n";
          break;
        case CEED_EVAL_GRAD:
          code << "      CeedScalar r_q_" << i << "[num_comp_in_" << i << "*DIM];\n";
          code << "      gradCollo3d(num_comp_in_" << i << ", Q_1D, q, r_t_" << i << ", s_G_in_" << i << ", r_q_" << i << ", elem_scratch);\n";
          break;
        case CEED_EVAL_WEIGHT:
          code << "      CeedScalar r_q_" << i << "[1];\n";
          code << "      r_q_" << i << "[0] = r_t_" << i << "[q];\n";
          break;  // No action
        case CEED_EVAL_DIV:
          break;  // TODO: Not implemented
        case CEED_EVAL_CURL:
          break;  // TODO: Not implemented
      }
    }
    code << "\n      // -- Output fields --\n";
    for (CeedInt i = 0; i < num_output_fields; i++) {
      code << "      // ---- Output field " << i << " ----\n";
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
      // Basis action
      switch (eval_mode) {
        case CEED_EVAL_NONE:
          code << "      CeedScalar r_qq_" << i << "[num_comp_out_" << i << "];\n";
          break;  // No action
        case CEED_EVAL_INTERP:
          code << "      CeedScalar r_qq_" << i << "[num_comp_out_" << i << "];\n";
          break;
        case CEED_EVAL_GRAD:
          code << "      CeedScalar r_qq_" << i << "[num_comp_out_" << i << "*DIM];\n";
          break;
        case CEED_EVAL_WEIGHT:
          break;  // Should not occur
        case CEED_EVAL_DIV:
          break;  // TODO: Not implemented
        case CEED_EVAL_CURL:
          break;  // TODO: Not implemented
      }
    }
  } else {
    code << "\n      // Note: Using full elements\n";
    code << "      // -- Input fields --\n";
    for (CeedInt i = 0; i < num_input_fields; i++) {
      code << "      // ---- Input field " << i << " ----\n";
      code << "      private CeedScalar* r_q_" << i << " = r_t_" << i << ";\n";
    }
    code << "      // -- Output fields --\n";
    for (CeedInt i = 0; i < num_output_fields; i++) {
      code << "      // ---- Output field " << i << " ----\n";
      code << "      private CeedScalar* r_qq_" << i << " = r_tt_" << i << ";\n";
    }
  }
  //--------------------------------------------------
  code << "\n      // -- QFunction Inputs and outputs --\n";
  code << "      const CeedScalar * in[" << num_input_fields << "];\n";
  for (CeedInt i = 0; i < num_input_fields; i++) {
    code << "      // ---- Input field " << i << " ----\n";
    code << "      in[" << i << "] = r_q_" << i << ";\n";
  }
  code << "      CeedScalar * out[" << num_output_fields << "];\n";
  for (CeedInt i = 0; i < num_output_fields; i++) {
    code << "      // ---- Output field " << i << " ----\n";
    code << "      out[" << i << "] = r_qq_" << i << ";\n";
  }

  code << "\n      // -- Apply QFunction --\n";
  code << "      " << qfunction_name << "(ctx, ";
  if (dim != 3 || use_collograd_parallelization) {
    code << "1";
  } else {
    code << "Q_1D";
  }
  code << ", in, out);\n";
  //--------------------------------------------------

  if (use_collograd_parallelization) {
    code << "      // -- Output fields --\n";
    for (CeedInt i = 0; i < num_output_fields; i++) {
      code << "      // ---- Output field " << i << " ----\n";
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
      // Basis action
      code << "      // EvalMode: " << CeedEvalModes[eval_mode] << "\n";
      switch (eval_mode) {
        case CEED_EVAL_NONE:
          code << "      for (CeedInt j = 0; j < num_comp_out_" << i << " ; ++j) {\n";
          code << "        r_tt_" << i << "[q + j*Q_1D] = r_qq_" << i << "[j];\n";
          code << "      }\n";
          break;  // No action
        case CEED_EVAL_INTERP:
          code << "      for (CeedInt j = 0; j < num_comp_out_" << i << " ; ++j) {\n";
          code << "        r_tt_" << i << "[q + j*Q_1D] = r_qq_" << i << "[j];\n";
          code << "      }\n";
          break;
        case CEED_EVAL_GRAD:
          code << "      gradColloTranspose3d(num_comp_out_" << i << ",Q_1D, q, r_qq_" << i << ", s_G_out_" << i << ", r_tt_" << i
               << ", elem_scratch);\n";
          break;
        case CEED_EVAL_WEIGHT:
          break;  // Should not occur
        case CEED_EVAL_DIV:
          break;  // TODO: Not implemented
        case CEED_EVAL_CURL:
          break;  // TODO: Not implemented
      }
    }
    code << "    }\n";
  }

  // Output basis apply if needed
  // Generate the correct eval mode code for each output
  code << "\n    // -- Output field basis action and restrictions --\n";
  for (CeedInt i = 0; i < num_output_fields; i++) {
    code << "    // ---- Output field " << i << " ----\n";
    // Get elem_size, eval_mode, num_comp
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[i], &elem_rstr));
    CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    CeedCallBackend(CeedElemRestrictionGetNumComponents(elem_rstr, &num_comp));
    // Basis action
    code << "    // EvalMode: " << CeedEvalModes[eval_mode] << "\n";
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        code << "    private CeedScalar* r_v_" << i << " = r_tt_" << i << ";\n";
        break;  // No action
      case CEED_EVAL_INTERP:
        code << "    CeedScalar r_v_" << i << "[num_comp_out_" << i << "*P_out_" << i << "];\n";
        code << "    InterpTranspose" << (dim > 1 ? "Tensor" : "") << dim << "d(num_comp_out_" << i << ",P_out_" << i << ", Q_1D, r_tt_" << i
             << ", s_B_out_" << i << ", r_v_" << i << ", elem_scratch);\n";
        break;
      case CEED_EVAL_GRAD:
        code << "    CeedScalar r_v_" << i << "[num_comp_out_" << i << "*P_out_" << i << "];\n";
        if (use_collograd_parallelization) {
          code << "    InterpTranspose" << (dim > 1 ? "Tensor" : "") << dim << "d(num_comp_out_" << i << ",P_out_" << i << ", Q_1D, r_tt_" << i
               << ", s_B_out_" << i << ", r_v_" << i << ", elem_scratch);\n";
        } else {
          CeedInt P_1d;
          CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
          CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
          code << "    GradTranspose" << (dim > 1 ? "Tensor" : "") << (dim == 3 && Q_1d >= P_1d ? "Collocated" : "") << dim << "d(num_comp_out_" << i
               << ", P_out_" << i << ", Q_1D, r_tt_" << i << (dim > 1 ? ", s_B_out_" : "") << (dim > 1 ? std::to_string(i) : "") << ", s_G_out_" << i
               << ", r_v_" << i << ", elem_scratch);\n";
        }
        break;
      // LCOV_EXCL_START
      case CEED_EVAL_WEIGHT: {
        Ceed ceed;

        CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
        return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
        break;  // Should not occur
      }
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL: {
        Ceed ceed;

        CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
        return CeedError(ceed, CEED_ERROR_BACKEND, "%s not supported", CeedEvalModes[eval_mode]);
        break;  // Should not occur
      }
        // LCOV_EXCL_STOP
    }
    // Restriction
    bool is_strided;

    CeedCallBackend(CeedElemRestrictionIsStrided(elem_rstr, &is_strided));
    if (!is_strided) {
      CeedInt comp_stride;

      CeedCallBackend(CeedElemRestrictionGetLVectorSize(elem_rstr, &l_size));
      code << "    const CeedInt l_size_out_" << i << " = " << l_size << ";\n";
      CeedCallBackend(CeedElemRestrictionGetCompStride(elem_rstr, &comp_stride));
      code << "    // CompStride: " << comp_stride << "\n";
      CeedCallBackend(CeedElemRestrictionGetData(elem_rstr, &rstr_impl));
      h_indices.outputs[i] = rstr_impl->d_ind;
      code << "    writeDofsOffset" << dim << "d(num_comp_out_" << i << ", " << comp_stride << ", P_out_" << i << ", num_elem, indices->outputs[" << i
           << "], r_v_" << i << ", d_v_" << i << ");\n";
    } else {
      bool    has_backend_strides;
      CeedInt num_elem;

      CeedCallBackend(CeedElemRestrictionHasBackendStrides(elem_rstr, &has_backend_strides));
      CeedCallBackend(CeedElemRestrictionGetNumElements(elem_rstr, &num_elem));
      CeedInt strides[3] = {1, elem_size * num_elem, elem_size};

      if (!has_backend_strides) {
        CeedCallBackend(CeedElemRestrictionGetStrides(elem_rstr, strides));
      }
      code << "    // Strides: {" << strides[0] << ", " << strides[1] << ", " << strides[2] << "}\n";
      code << "    writeDofsStrided" << dim << "d(num_comp_out_" << i << ",P_out_" << i << "," << strides[0] << "," << strides[1] << "," << strides[2]
           << ", num_elem, r_v_" << i << ", d_v_" << i << ");\n";
    }
  }

  code << "  }\n";
  code << "}\n";
  code << "// -----------------------------------------------------------------------------\n\n";

  // Copy the struct (containing device addresses) from the host to the device
  sycl::event copy_B       = sycl_data->sycl_queue.copy<Fields_Sycl>(&h_B, impl->B, 1);
  sycl::event copy_G       = sycl_data->sycl_queue.copy<Fields_Sycl>(&h_G, impl->G, 1);
  sycl::event copy_indices = sycl_data->sycl_queue.copy<FieldsInt_Sycl>(&h_indices, impl->indices, 1);
  // These copies can happen while the JIT is being done
  CeedCallSycl(ceed, sycl::event::wait_and_throw({copy_B, copy_G, copy_indices}));

  // View kernel for debugging
  CeedDebug256(ceed, 2, "Generated Operator Kernels:\n");
  CeedDebug(ceed, code.str().c_str());

  std::map<std::string, CeedInt> jit_constants;
  jit_constants["T_1D"]         = block_sizes[0];
  jit_constants["GROUP_SIZE_X"] = block_sizes[0];
  jit_constants["GROUP_SIZE_Y"] = block_sizes[1];
  jit_constants["GROUP_SIZE_Z"] = block_sizes[2];

  // Compile kernel into a kernel bundle
  CeedCallBackend(CeedBuildModule_Sycl(ceed, code.str(), &impl->sycl_module, jit_constants));

  // Load kernel function
  CeedCallBackend(CeedGetKernel_Sycl(ceed, impl->sycl_module, operator_name, &impl->op));

  CeedCallBackend(CeedOperatorSetSetupDone(op));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
