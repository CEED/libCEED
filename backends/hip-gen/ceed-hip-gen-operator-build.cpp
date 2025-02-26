// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#define CEED_DEBUG_COLOR 12

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>

#include <iostream>
#include <sstream>
#include <string>

#include "../hip-ref/ceed-hip-ref.h"
#include "../hip-shared/ceed-hip-shared.h"
#include "../hip/ceed-hip-common.h"
#include "../hip/ceed-hip-compile.h"
#include "ceed-hip-gen.h"

struct FieldReuse_Hip {
  CeedInt      index;
  bool         is_input;
  CeedEvalMode eval_mode;
};

//------------------------------------------------------------------------------
// Calculate the block size used for launching the operator kernel
//------------------------------------------------------------------------------
extern "C" int BlockGridCalculate_Hip_gen(const CeedInt dim, const CeedInt num_elem, const CeedInt P_1d, const CeedInt Q_1d, CeedInt *block_sizes) {
  const CeedInt thread_1d = CeedIntMax(Q_1d, P_1d);
  if (dim == 1) {
    CeedInt elems_per_block = 64 * thread_1d > 256 ? 256 / thread_1d : 64;

    elems_per_block = elems_per_block > 0 ? elems_per_block : 1;
    block_sizes[0]  = thread_1d;
    block_sizes[1]  = 1;
    block_sizes[2]  = elems_per_block;
  } else if (dim == 2) {
    const CeedInt elems_per_block = thread_1d < 4 ? 16 : 2;

    block_sizes[0] = thread_1d;
    block_sizes[1] = thread_1d;
    block_sizes[2] = elems_per_block;
  } else if (dim == 3) {
    const CeedInt elems_per_block = thread_1d < 6 ? 4 : (thread_1d < 8 ? 2 : 1);

    block_sizes[0] = thread_1d;
    block_sizes[1] = thread_1d;
    block_sizes[2] = elems_per_block;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Determine type of operator
//------------------------------------------------------------------------------
static int CeedOperatorBuildKernelData_Hip_gen(Ceed ceed, CeedInt num_input_fields, CeedOperatorField *op_input_fields,
                                               CeedQFunctionField *qf_input_fields, CeedInt num_output_fields, CeedOperatorField *op_output_fields,
                                               CeedQFunctionField *qf_output_fields, CeedInt *max_P_1d, CeedInt *Q_1d, CeedInt *dim, bool *is_tensor,
                                               bool *use_3d_slices) {
  // Find dim, P_1d, Q_1d
  *max_P_1d  = 0;
  *Q_1d      = 0;
  *dim       = 0;
  *is_tensor = true;

  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedBasis basis;

    CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
    if (basis != CEED_BASIS_NONE) {
      bool    is_field_tensor;
      CeedInt field_P_1d = 0, field_Q_1d = 0, field_dim = 0;

      // Collect dim, P_1d, and Q_1d
      CeedCallBackend(CeedBasisIsTensor(basis, &is_field_tensor));
      *is_tensor = *is_tensor && is_field_tensor;
      if (is_field_tensor) CeedCallBackend(CeedBasisGetNumNodes1D(basis, &field_P_1d));
      else CeedCallBackend(CeedBasisGetNumNodes(basis, &field_P_1d));
      *max_P_1d = CeedIntMax(*max_P_1d, field_P_1d);
      CeedCallBackend(CeedBasisGetDimension(basis, &field_dim));
      CeedCheck(*dim == 0 || field_dim == *dim, ceed, CEED_ERROR_BACKEND, "Quadrature spaces must be compatible");
      *dim = field_dim;
      if (is_field_tensor) CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &field_Q_1d));
      else CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis, &field_Q_1d));
      CeedCheck(*Q_1d == 0 || field_Q_1d == *Q_1d, ceed, CEED_ERROR_BACKEND, "Quadrature spaces must be compatible");
      *Q_1d = field_Q_1d;
    }
    CeedCallBackend(CeedBasisDestroy(&basis));
  }
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedBasis basis;

    CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
    if (basis != CEED_BASIS_NONE) {
      bool    is_field_tensor;
      CeedInt field_P_1d = 0, field_Q_1d = 0, field_dim = 0;

      // Collect dim, P_1d, and Q_1d
      CeedCallBackend(CeedBasisIsTensor(basis, &is_field_tensor));
      *is_tensor = *is_tensor && is_field_tensor;
      if (is_field_tensor) CeedCallBackend(CeedBasisGetNumNodes1D(basis, &field_P_1d));
      else CeedCallBackend(CeedBasisGetNumNodes(basis, &field_P_1d));
      *max_P_1d = CeedIntMax(*max_P_1d, field_P_1d);
      CeedCallBackend(CeedBasisGetDimension(basis, &field_dim));
      CeedCheck(*dim == 0 || field_dim == *dim, ceed, CEED_ERROR_BACKEND, "Quadrature spaces must be compatible");
      *dim = field_dim;
      if (is_field_tensor) CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &field_Q_1d));
      else CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis, &field_Q_1d));
      CeedCheck(*Q_1d == 0 || field_Q_1d == *Q_1d, ceed, CEED_ERROR_BACKEND, "Quadrature spaces must be compatible");
      *Q_1d = field_Q_1d;
    }
    CeedCallBackend(CeedBasisDestroy(&basis));
  }

  // Only use 3D collocated gradient parallelization strategy when gradient is computed
  *use_3d_slices = false;
  if (*dim == 3) {
    bool was_grad_found = false;

    for (CeedInt i = 0; i < num_input_fields; i++) {
      CeedEvalMode eval_mode;

      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
      if (eval_mode == CEED_EVAL_GRAD) {
        CeedBasis_Hip_shared *basis_data;
        CeedBasis             basis;

        CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
        CeedCallBackend(CeedBasisGetData(basis, &basis_data));
        *use_3d_slices = basis_data->d_collo_grad_1d && (was_grad_found ? *use_3d_slices : true);
        was_grad_found = true;
        CeedCallBackend(CeedBasisDestroy(&basis));
      }
    }
    for (CeedInt i = 0; i < num_output_fields; i++) {
      CeedEvalMode eval_mode;

      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
      if (eval_mode == CEED_EVAL_GRAD) {
        CeedBasis_Hip_shared *basis_data;
        CeedBasis             basis;

        CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
        CeedCallBackend(CeedBasisGetData(basis, &basis_data));
        *use_3d_slices = basis_data->d_collo_grad_1d && (was_grad_found ? *use_3d_slices : true);
        was_grad_found = true;
        CeedCallBackend(CeedBasisDestroy(&basis));
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup fields
//------------------------------------------------------------------------------
static int CeedOperatorBuildKernelFieldData_Hip_gen(std::ostringstream &code, CeedOperator_Hip_gen *data, CeedInt i, CeedOperatorField op_field,
                                                    CeedQFunctionField qf_field, FieldReuse_Hip field_reuse, CeedInt Q_1d, bool is_input,
                                                    bool is_tensor, bool is_at_points, bool use_3d_slices) {
  const char           *field_name;
  std::string           var_suffix = (is_input ? "_in_" : "_out_") + std::to_string(i);
  std::string           P_name = (is_tensor ? "P_1d" : "P") + var_suffix, Q_name = is_tensor ? "Q_1d" : "Q";
  std::string           option_name = (is_input ? "inputs" : "outputs");
  CeedEvalMode          eval_mode   = CEED_EVAL_NONE;
  CeedInt               elem_size = 0, num_comp = 0, P_1d = 0;
  CeedElemRestriction   elem_rstr;
  CeedBasis_Hip_shared *basis_data;
  CeedBasis             basis;

  // Field reuse info
  bool use_previous_field = field_reuse.index != -1;

  CeedCallBackend(CeedOperatorFieldGetName(op_field, &field_name));
  code << "  // -- " << (is_input ? "Input" : "Output") << " field " << i << ": " << field_name << "\n";

  // Get field data
  CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_field, &elem_rstr));
  if (elem_rstr != CEED_ELEMRESTRICTION_NONE) {
    CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
    CeedCallBackend(CeedElemRestrictionGetNumComponents(elem_rstr, &num_comp));
  }
  CeedCallBackend(CeedElemRestrictionDestroy(&elem_rstr));
  CeedCallBackend(CeedOperatorFieldGetBasis(op_field, &basis));
  if (basis != CEED_BASIS_NONE) {
    CeedCallBackend(CeedBasisGetData(basis, &basis_data));
    if (is_tensor) CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
    else CeedCallBackend(CeedBasisGetNumNodes(basis, &P_1d));
  }
  CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_field, &eval_mode));

  // Set field constants
  if (eval_mode != CEED_EVAL_WEIGHT) {
    code << "  const CeedInt " << P_name << " = " << (basis == CEED_BASIS_NONE ? Q_1d : P_1d) << ";\n";
    code << "  const CeedInt num_comp" << var_suffix << " = " << num_comp << ";\n";
  }

  // Load basis data
  code << "  // EvalMode: " << CeedEvalModes[eval_mode] << "\n";
  switch (eval_mode) {
    case CEED_EVAL_NONE:
      break;
    case CEED_EVAL_INTERP:
      if (is_at_points) {
        // AtPoints
        if (!basis_data->d_chebyshev_interp_1d) {
          CeedSize    interp_bytes;
          CeedScalar *chebyshev_interp_1d;

          interp_bytes = P_1d * Q_1d * sizeof(CeedScalar);
          CeedCallBackend(CeedCalloc(P_1d * Q_1d, &chebyshev_interp_1d));
          CeedCallBackend(CeedBasisGetChebyshevInterp1D(basis, chebyshev_interp_1d));
          CeedCallHip(CeedBasisReturnCeed(basis), hipMalloc((void **)&basis_data->d_chebyshev_interp_1d, interp_bytes));
          CeedCallHip(CeedBasisReturnCeed(basis),
                      hipMemcpy(basis_data->d_chebyshev_interp_1d, chebyshev_interp_1d, interp_bytes, hipMemcpyHostToDevice));
          CeedCallBackend(CeedFree(&chebyshev_interp_1d));
        }
        if (is_input) data->B.inputs[i] = basis_data->d_chebyshev_interp_1d;
        else data->B.outputs[i] = basis_data->d_chebyshev_interp_1d;
      } else {
        // Standard quadrature
        if (is_input) data->B.inputs[i] = basis_data->d_interp_1d;
        else data->B.outputs[i] = basis_data->d_interp_1d;
      }
      if (use_previous_field) {
        std::string reuse_var = "s_B" + ((field_reuse.is_input ? "_in_" : "_out_") + std::to_string(field_reuse.index));

        code << "  CeedScalar *s_B" << var_suffix << " = " << reuse_var << ";\n";
      } else {
        code << "  __shared__ CeedScalar s_B" << var_suffix << "[" << P_name << "*" << Q_name << "];\n";
        code << "  LoadMatrix<" << P_name << ", " << Q_name << ">(data, B." << option_name << "[" << i << "], s_B" << var_suffix << ");\n";
      }
      break;
    case CEED_EVAL_GRAD:
      if (is_at_points) {
        // AtPoints
        if (!basis_data->d_chebyshev_interp_1d) {
          CeedSize    interp_bytes;
          CeedScalar *chebyshev_interp_1d;

          interp_bytes = P_1d * Q_1d * sizeof(CeedScalar);
          CeedCallBackend(CeedCalloc(P_1d * Q_1d, &chebyshev_interp_1d));
          CeedCallBackend(CeedBasisGetChebyshevInterp1D(basis, chebyshev_interp_1d));
          CeedCallHip(CeedBasisReturnCeed(basis), hipMalloc((void **)&basis_data->d_chebyshev_interp_1d, interp_bytes));
          CeedCallHip(CeedBasisReturnCeed(basis),
                      hipMemcpy(basis_data->d_chebyshev_interp_1d, chebyshev_interp_1d, interp_bytes, hipMemcpyHostToDevice));
          CeedCallBackend(CeedFree(&chebyshev_interp_1d));
        }
        if (is_input) data->B.inputs[i] = basis_data->d_chebyshev_interp_1d;
        else data->B.outputs[i] = basis_data->d_chebyshev_interp_1d;
      } else {
        // Standard quadrature
        if (is_input) data->B.inputs[i] = basis_data->d_interp_1d;
        else data->B.outputs[i] = basis_data->d_interp_1d;
      }
      if (is_tensor) {
        if (use_previous_field) {
          std::string reuse_var = "s_B" + ((field_reuse.is_input ? "_in_" : "_out_") + std::to_string(field_reuse.index));

          code << "  CeedScalar *s_B" << var_suffix << " = " << reuse_var << ";\n";
        } else {
          code << "  __shared__ CeedScalar s_B" << var_suffix << "[" << P_name << "*" << Q_name << "];\n";
          code << "  LoadMatrix<" << P_name << ", " << Q_name << ">(data, B." << option_name << "[" << i << "], s_B" << var_suffix << ");\n";
        }
      }
      if (is_at_points) break;  // No G mat for AtPoints
      if (use_3d_slices) {
        if (is_input) data->G.inputs[i] = basis_data->d_collo_grad_1d;
        else data->G.outputs[i] = basis_data->d_collo_grad_1d;
        if (use_previous_field && field_reuse.eval_mode == CEED_EVAL_GRAD) {
          std::string reuse_var = "s_G" + ((field_reuse.is_input ? "_in_" : "_out_") + std::to_string(field_reuse.index));

          code << "  CeedScalar *s_G" << var_suffix << " = " << reuse_var << ";\n";
        } else {
          code << "  __shared__ CeedScalar s_G" << var_suffix << "[" << Q_name << "*" << Q_name << "];\n";
          code << "  LoadMatrix<" << Q_name << ", " << Q_name << ">(data, G." << option_name << "[" << i << "], s_G" << var_suffix << ");\n";
        }
      } else {
        bool has_collo_grad = basis_data->d_collo_grad_1d;

        if (is_input) data->G.inputs[i] = has_collo_grad ? basis_data->d_collo_grad_1d : basis_data->d_grad_1d;
        else data->G.outputs[i] = has_collo_grad ? basis_data->d_collo_grad_1d : basis_data->d_grad_1d;
        if (has_collo_grad) {
          if (use_previous_field && field_reuse.eval_mode == CEED_EVAL_GRAD) {
            std::string reuse_var = "s_G" + ((field_reuse.is_input ? "_in_" : "_out_") + std::to_string(field_reuse.index));

            code << "  CeedScalar *s_G" << var_suffix << " = " << reuse_var << ";\n";
          } else {
            code << "  __shared__ CeedScalar s_G" << var_suffix << "[" << Q_name << "*" << Q_name << "];\n";
            code << "  LoadMatrix<" << Q_name << ", " << Q_name << ">(data, G." << option_name << "[" << i << "], s_G" << var_suffix << ");\n";
          }
        } else {
          if (use_previous_field && field_reuse.eval_mode == CEED_EVAL_GRAD) {
            std::string reuse_var = "s_G" + ((field_reuse.is_input ? "_in_" : "_out_") + std::to_string(field_reuse.index));

            code << "  CeedScalar *s_G" << var_suffix << " = " << reuse_var << ";\n";
          } else {
            code << "  __shared__ CeedScalar s_G" << var_suffix << "[" << P_name << "*" << Q_name << (is_tensor ? "" : "*dim") << "];\n";
            code << "  LoadMatrix<" << P_name << ", " << Q_name << (is_tensor ? "" : "*dim") << ">(data, G." << option_name << "[" << i << "], s_G"
                 << var_suffix << ");\n";
          }
        }
      }
      break;
    case CEED_EVAL_WEIGHT:
      break;  // No action
      // LCOV_EXCL_START
    case CEED_EVAL_DIV:
    case CEED_EVAL_CURL:
      break;  // TODO: Not implemented
              // LCOV_EXCL_STOP
  }
  CeedCallBackend(CeedBasisDestroy(&basis));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Restriction
//------------------------------------------------------------------------------
static int CeedOperatorBuildKernelRestriction_Hip_gen(std::ostringstream &code, CeedOperator_Hip_gen *data, CeedInt i, CeedInt dim,
                                                      CeedInt field_input_buffer[], CeedOperatorField op_field, CeedQFunctionField qf_field,
                                                      CeedInt Q_1d, bool is_input, bool is_tensor, bool is_at_points, bool use_3d_slices) {
  std::string              var_suffix = (is_input ? "_in_" : "_out_") + std::to_string(i);
  std::string              P_name     = (is_tensor ? "P_1d" : "P") + var_suffix;
  CeedEvalMode             eval_mode  = CEED_EVAL_NONE;
  CeedInt                  elem_size = 0, num_comp = 0, P_1d = 0;
  CeedSize                 l_size;
  CeedRestrictionType      rstr_type = CEED_RESTRICTION_STANDARD;
  CeedElemRestriction_Hip *rstr_data;
  CeedElemRestriction      elem_rstr;
  CeedBasis                basis;

  // Get field data
  CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_field, &elem_rstr));
  if (elem_rstr != CEED_ELEMRESTRICTION_NONE) {
    CeedCallBackend(CeedElemRestrictionGetType(elem_rstr, &rstr_type));
    CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
    CeedCallBackend(CeedElemRestrictionGetNumComponents(elem_rstr, &num_comp));
    CeedCallBackend(CeedElemRestrictionGetData(elem_rstr, &rstr_data));
  }
  CeedCallBackend(CeedOperatorFieldGetBasis(op_field, &basis));
  if (basis != CEED_BASIS_NONE) {
    if (is_tensor) CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
    else CeedCallBackend(CeedBasisGetNumNodes(basis, &P_1d));
  }
  CeedCallBackend(CeedBasisDestroy(&basis));
  CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_field, &eval_mode));

  // Restriction
  if (is_input) {
    // Input
    if (field_input_buffer[i] != i) {
      std::string buffer_name = "r_e_in_" + std::to_string(field_input_buffer[i]);

      // Restriction was already done for previous input
      code << "    CeedScalar *r_e" << var_suffix << " = " << buffer_name << ";\n";
    } else if (eval_mode != CEED_EVAL_WEIGHT && !((eval_mode == CEED_EVAL_NONE) && use_3d_slices && is_at_points)) {
      if (eval_mode == CEED_EVAL_NONE && rstr_type != CEED_RESTRICTION_POINTS) {
        // No basis action, so r_e_in_* in also r_q_in_* and needs to be allocated
        code << "    CeedScalar r_e" << var_suffix << "[num_comp" << var_suffix << "*" << P_name << "];\n";
      } else if (rstr_type != CEED_RESTRICTION_POINTS) {
        // Otherwise we're using the scratch space
        code << "    CeedScalar *r_e" << var_suffix << " = r_e_scratch;\n";
      }
      switch (rstr_type) {
        case CEED_RESTRICTION_STANDARD: {
          CeedInt comp_stride;

          CeedCallBackend(CeedElemRestrictionGetLVectorSize(elem_rstr, &l_size));
          code << "    const CeedInt l_size" << var_suffix << " = " << l_size << ";\n";
          CeedCallBackend(CeedElemRestrictionGetCompStride(elem_rstr, &comp_stride));
          code << "    // CompStride: " << comp_stride << "\n";
          data->indices.inputs[i] = (CeedInt *)rstr_data->d_offsets;
          code << "    ReadLVecStandard" << (is_tensor ? dim : 1) << "d<num_comp" << var_suffix << ", " << comp_stride << ", " << P_name
               << ">(data, l_size" << var_suffix << ", elem, indices.inputs[" << i << "], d" << var_suffix << ", r_e" << var_suffix << ");\n";
          break;
        }
        case CEED_RESTRICTION_STRIDED: {
          bool    has_backend_strides;
          CeedInt num_elem;

          CeedCallBackend(CeedElemRestrictionHasBackendStrides(elem_rstr, &has_backend_strides));
          CeedCallBackend(CeedElemRestrictionGetNumElements(elem_rstr, &num_elem));
          CeedInt strides[3] = {1, elem_size * num_elem, elem_size};

          if (!has_backend_strides) {
            CeedCallBackend(CeedElemRestrictionGetStrides(elem_rstr, strides));
          }
          code << "    // Strides: {" << strides[0] << ", " << strides[1] << ", " << strides[2] << "}\n";
          code << "    ReadLVecStrided" << (is_tensor ? dim : 1) << "d<num_comp" << var_suffix << ", " << P_name << ", " << strides[0] << ", "
               << strides[1] << ", " << strides[2] << ">(data, elem, d" << var_suffix << ", r_e" << var_suffix << ");\n";
          break;
        }
        case CEED_RESTRICTION_POINTS: {
          CeedInt comp_stride;

          CeedCallBackend(CeedElemRestrictionGetCompStride(elem_rstr, &comp_stride));
          code << "    const CeedInt comp_stride" << var_suffix << " = " << comp_stride << ";\n";
          data->indices.inputs[i] = (CeedInt *)rstr_data->d_offsets;
          break;
        }
        // LCOV_EXCL_START
        case CEED_RESTRICTION_ORIENTED:
        case CEED_RESTRICTION_CURL_ORIENTED:
          break;  // TODO: Not implemented
                  // LCOV_EXCL_STOP
      }
    }
  } else {
    // Output
    switch (rstr_type) {
      case CEED_RESTRICTION_STANDARD: {
        CeedInt comp_stride;

        CeedCallBackend(CeedElemRestrictionGetLVectorSize(elem_rstr, &l_size));
        code << "    const CeedInt l_size" << var_suffix << " = " << l_size << ";\n";
        CeedCallBackend(CeedElemRestrictionGetCompStride(elem_rstr, &comp_stride));
        code << "    // CompStride: " << comp_stride << "\n";
        data->indices.outputs[i] = (CeedInt *)rstr_data->d_offsets;
        code << "    WriteLVecStandard" << (is_tensor ? dim : 1) << "d<num_comp" << var_suffix << ", " << comp_stride << ", " << P_name
             << ">(data, l_size" << var_suffix << ", elem, indices.outputs[" << i << "], r_e" << var_suffix << ", d" << var_suffix << ");\n";
        break;
      }
      case CEED_RESTRICTION_STRIDED: {
        bool    has_backend_strides;
        CeedInt num_elem;

        CeedCallBackend(CeedElemRestrictionHasBackendStrides(elem_rstr, &has_backend_strides));
        CeedCallBackend(CeedElemRestrictionGetNumElements(elem_rstr, &num_elem));
        CeedInt strides[3] = {1, elem_size * num_elem, elem_size};

        if (!has_backend_strides) {
          CeedCallBackend(CeedElemRestrictionGetStrides(elem_rstr, strides));
        }
        code << "    // Strides: {" << strides[0] << ", " << strides[1] << ", " << strides[2] << "}\n";
        code << "    WriteLVecStrided" << (is_tensor ? dim : 1) << "d<num_comp" << var_suffix << ", " << P_name << ", " << strides[0] << ", "
             << strides[1] << ", " << strides[2] << ">(data, elem, r_e" << var_suffix << ", d" << var_suffix << ");\n";
        break;
      }
      case CEED_RESTRICTION_POINTS:
        data->indices.outputs[i] = (CeedInt *)rstr_data->d_offsets;
        break;
      // LCOV_EXCL_START
      case CEED_RESTRICTION_ORIENTED:
      case CEED_RESTRICTION_CURL_ORIENTED:
        break;  // TODO: Not implemented
                // LCOV_EXCL_STOP
    }
  }
  CeedCallBackend(CeedElemRestrictionDestroy(&elem_rstr));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis
//------------------------------------------------------------------------------
static int CeedOperatorBuildKernelBasis_Hip_gen(std::ostringstream &code, CeedOperator_Hip_gen *data, CeedInt i, CeedInt dim,
                                                CeedOperatorField op_field, CeedQFunctionField qf_field, CeedInt Q_1d, bool is_input, bool is_tensor,
                                                bool is_at_points, bool use_3d_slices) {
  std::string         var_suffix = (is_input ? "_in_" : "_out_") + std::to_string(i);
  std::string         P_name = (is_tensor ? "P_1d" : "P") + var_suffix, Q_name = is_tensor ? "Q_1d" : "Q";
  CeedEvalMode        eval_mode = CEED_EVAL_NONE;
  CeedInt             elem_size = 0, num_comp = 0, P_1d = 0;
  CeedElemRestriction elem_rstr;
  CeedBasis           basis;

  // Get field data
  CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_field, &elem_rstr));
  if (elem_rstr != CEED_ELEMRESTRICTION_NONE) {
    CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
    CeedCallBackend(CeedElemRestrictionGetNumComponents(elem_rstr, &num_comp));
  }
  CeedCallBackend(CeedElemRestrictionDestroy(&elem_rstr));
  CeedCallBackend(CeedOperatorFieldGetBasis(op_field, &basis));
  if (basis != CEED_BASIS_NONE) {
    if (is_tensor) CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
    else CeedCallBackend(CeedBasisGetNumNodes(basis, &P_1d));
  }
  CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_field, &eval_mode));

  // Basis
  code << "    // EvalMode: " << CeedEvalModes[eval_mode] << "\n";
  if (is_input) {
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        if (!use_3d_slices && !is_at_points) {
          code << "    CeedScalar *r_q" << var_suffix << " = r_e" << var_suffix << ";\n";
        }
        break;
      case CEED_EVAL_INTERP:
        if (is_at_points) {
          std::string function_name = (dim == 1 ? "Interp" : "InterpTensor") + std::to_string(dim) + "d";

          code << "    CeedScalar r_c" << var_suffix << "[num_comp" << var_suffix << "*" << (dim >= 3 ? Q_name : "1") << "];\n";
          code << "    " << function_name << "<num_comp" << var_suffix << ", " << P_name << ", " << Q_name << ", OP_T_1D>(data, r_e" << var_suffix
               << ", s_B" << var_suffix << ", r_c" << var_suffix << ");\n";
        } else {
          std::string function_name = is_tensor ? ((dim == 1 ? "Interp" : "InterpTensor") + std::to_string(dim) + "d") : "InterpNonTensor";

          code << "    CeedScalar r_q" << var_suffix << "[num_comp" << var_suffix << "*" << (is_tensor && (dim >= 3) ? Q_name : "1") << "];\n";
          code << "    " << function_name << "<num_comp" << var_suffix << ", " << P_name << ", " << Q_name << ", OP_T_1D>(data, r_e" << var_suffix
               << ", s_B" << var_suffix << ", r_q" << var_suffix << ");\n";
        }
        break;
      case CEED_EVAL_GRAD:
        if (is_at_points) {
          std::string function_name = (dim == 1 ? "Interp" : "InterpTensor") + std::to_string(dim) + "d";

          code << "    CeedScalar r_c" << var_suffix << "[num_comp" << var_suffix << "*" << (dim >= 3 ? Q_name : "1") << "];\n";
          code << "    " << function_name << "<num_comp" << var_suffix << ", " << P_name << ", " << Q_name << ", OP_T_1D>(data, r_e" << var_suffix
               << ", s_B" << var_suffix << ", r_c" << var_suffix << ");\n";
        } else if (use_3d_slices) {
          std::string function_name = (dim > 1 ? "InterpTensor" : "Interp") + std::to_string(dim) + "d";

          code << "    CeedScalar r_q" << var_suffix << "[num_comp" << var_suffix << "*" << Q_name << "];\n";
          code << "    " << function_name << "<num_comp" << var_suffix << ", " << P_name << ", " << Q_name << ", OP_T_1D>(data, r_e" << var_suffix
               << ", s_B" << var_suffix << ", r_q" << var_suffix << ");\n";
        } else if (is_tensor) {
          bool        is_collocated = dim == 3 && Q_1d >= P_1d;
          std::string function_name = (dim == 1 ? "Grad" : (is_collocated ? "GradTensorCollocated" : "GradTensor")) + std::to_string(dim) + "d";

          code << "    CeedScalar r_q" << var_suffix << "[num_comp" << var_suffix << "*dim*" << (dim >= 3 ? Q_name : "1") << "];\n";
          code << "    " << function_name << "<num_comp" << var_suffix << ", " << P_name << ", " << Q_name << ", OP_T_1D>(data, r_e" << var_suffix
               << ", s_B" << var_suffix << ", s_G" << var_suffix << ", r_q" << var_suffix << ");\n";
        } else {
          std::string function_name = "GradNonTensor";

          code << "    CeedScalar r_q" << var_suffix << "[num_comp" << var_suffix << "*dim];\n";
          code << "    " << function_name << "<num_comp" << var_suffix << ", dim, " << P_name << ", " << Q_name << ", OP_T_1D>(data, r_e"
               << var_suffix << ", s_G" << var_suffix << ", r_q" << var_suffix << ");\n";
        }
        break;
      case CEED_EVAL_WEIGHT: {
        if (is_at_points) {
          code << "    // Nothing to do AtPoints\n";
        } else {
          CeedBasis_Hip_shared *basis_data;
          std::string           function_name = is_tensor ? ((dim == 1 ? "Weight" : "WeightTensor") + std::to_string(dim) + "d") : "WeightNonTensor";

          code << "    CeedScalar r_q" << var_suffix << "[" << (is_tensor && (dim >= 3) ? Q_name : "1") << "];\n";
          CeedCallBackend(CeedBasisGetData(basis, &basis_data));
          data->W = basis_data->d_q_weight_1d;
          code << "    " << function_name << "<" << Q_name << ">(data, W, r_q" << var_suffix << ");\n";
        }
        break;
      }
      // LCOV_EXCL_START
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        break;  // TODO: Not implemented
                // LCOV_EXCL_STOP
    }
  } else {
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        code << "    CeedScalar *r_e" << var_suffix << " = r_q" << var_suffix << ";\n";
        break;  // No action
      case CEED_EVAL_INTERP:
        code << "    CeedScalar *r_e" << var_suffix << " = r_e_scratch;\n";
        if (is_at_points) {
          std::string function_name = (dim == 1 ? "InterpTranspose" : "InterpTransposeTensor") + std::to_string(dim) + "d";

          code << "    " << function_name << "<num_comp" << var_suffix << ", " << P_name << ", " << Q_name << ", OP_T_1D>(data, r_c" << var_suffix
               << ", s_B" << var_suffix << ", r_e" << var_suffix << ");\n";
        } else {
          std::string function_name =
              is_tensor ? ((dim == 1 ? "InterpTranspose" : "InterpTransposeTensor") + std::to_string(dim) + "d") : "InterpTransposeNonTensor";

          code << "    " << function_name << "<num_comp" << var_suffix << ", " << P_name << ", " << Q_name << ", OP_T_1D>(data, r_q" << var_suffix
               << ", s_B" << var_suffix << ", r_e" << var_suffix << ");\n";
        }
        break;
      case CEED_EVAL_GRAD:
        code << "    CeedScalar *r_e" << var_suffix << " = r_e_scratch;\n";
        if (is_at_points) {
          std::string function_name = (dim == 1 ? "InterpTranspose" : "InterpTransposeTensor") + std::to_string(dim) + "d";

          code << "    " << function_name << "<num_comp" << var_suffix << ", " << P_name << ", " << Q_name << ", OP_T_1D>(data, r_c" << var_suffix
               << ", s_B" << var_suffix << ", r_e" << var_suffix << ");\n";
        } else if (use_3d_slices) {
          std::string function_name = (dim == 1 ? "InterpTranspose" : "InterpTransposeTensor") + std::to_string(dim) + "d";

          code << "    " << function_name << "<num_comp" << var_suffix << ", " << P_name << ", " << Q_name << ", OP_T_1D>(data, r_q" << var_suffix
               << ", s_B" << var_suffix << ", r_e" << var_suffix << ");\n";
        } else if (is_tensor) {
          bool        is_collocated = dim == 3 && Q_1d >= P_1d;
          std::string function_name =
              (dim == 1 ? "GradTranspose" : (is_collocated ? "GradTransposeTensorCollocated" : "GradTransposeTensor")) + std::to_string(dim) + "d";

          code << "    " << function_name << "<num_comp" << var_suffix << ", " << P_name << ", " << Q_name << ", OP_T_1D>(data, r_q" << var_suffix
               << ", s_B" << var_suffix << ", s_G" << var_suffix << ", r_e" << var_suffix << ");\n";
        } else {
          std::string function_name = "GradTransposeNonTensor";

          code << "    " << function_name << "<num_comp" << var_suffix << ", dim, " << P_name << ", " << Q_name << ", OP_T_1D>(data, r_q"
               << var_suffix << ", s_G" << var_suffix << ", r_e" << var_suffix << ");\n";
        }
        break;
      // LCOV_EXCL_START
      case CEED_EVAL_WEIGHT:
        break;  // Should not occur
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        break;  // TODO: Not implemented
                // LCOV_EXCL_STOP
    }
  }
  CeedCallBackend(CeedBasisDestroy(&basis));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunction
//------------------------------------------------------------------------------
static int CeedOperatorBuildKernelQFunction_Hip_gen(std::ostringstream &code, CeedOperator_Hip_gen *data, CeedInt dim, CeedInt max_num_points,
                                                    CeedInt num_input_fields, CeedOperatorField *op_input_fields, CeedQFunctionField *qf_input_fields,
                                                    CeedInt num_output_fields, CeedOperatorField *op_output_fields,
                                                    CeedQFunctionField *qf_output_fields, std::string qfunction_name, CeedInt Q_1d, bool is_tensor,
                                                    bool is_at_points, bool use_3d_slices) {
  std::string         Q_name    = is_tensor ? "Q_1d" : "Q";
  CeedEvalMode        eval_mode = CEED_EVAL_NONE;
  CeedElemRestriction elem_rstr;

  // Setup output arrays
  code << "\n    // -- Output field setup\n";
  for (CeedInt i = 0; i < num_output_fields; i++) {
    const char *field_name;
    std::string var_suffix = "_out_" + std::to_string(i);

    CeedCallBackend(CeedOperatorFieldGetName(op_output_fields[i], &field_name));
    code << "    // ---- Output field " << i << ": " << field_name << "\n";
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        if (is_at_points) {
          code << "    CeedScalar r_q" << var_suffix << "[num_comp" << var_suffix << "];\n";
        } else {
          code << "    CeedScalar r_q" << var_suffix << "[num_comp" << var_suffix << "*" << (is_tensor && (dim >= 3) ? Q_name : "1") << "];\n";
        }
        break;
      case CEED_EVAL_INTERP:
        if (is_at_points) {
          // Accumulator for point data
          code << "    CeedScalar r_c" << var_suffix << "[num_comp" << var_suffix << "*" << (dim >= 3 ? Q_name : "1") << "];\n";
          code << "    for (CeedInt i = 0; i < num_comp" << var_suffix << "*" << (dim >= 3 ? Q_name : "1") << "; i++) {\n";
          code << "      r_c" << var_suffix << "[i] = 0.0;\n";
          code << "    }\n";
        } else {
          code << "    CeedScalar r_q" << var_suffix << "[num_comp" << var_suffix << "*" << (is_tensor && (dim >= 3) ? Q_name : "1") << "];\n";
        }
        break;
      case CEED_EVAL_GRAD:
        if (is_at_points) {
          // Accumulator for point data
          code << "    CeedScalar r_c" << var_suffix << "[num_comp" << var_suffix << "*" << (dim >= 3 ? Q_name : "1") << "*dim];\n";
          code << "    for (CeedInt i = 0; i < num_comp" << var_suffix << "*" << (dim >= 3 ? Q_name : "1") << "; i++) {\n";
          code << "      r_c" << var_suffix << "[i] = 0.0;\n";
          code << "    }\n";
        } else if (use_3d_slices) {
          // Accumulator for gradient slices
          code << "    CeedScalar r_q" << var_suffix << "[num_comp" << var_suffix << "*" << Q_name << "];\n";
          code << "    for (CeedInt i = 0; i < num_comp" << var_suffix << "*" << Q_name << "; i++) {\n";
          code << "      r_q" << var_suffix << "[i] = 0.0;\n";
          code << "    }\n";
        } else {
          code << "    CeedScalar r_q" << var_suffix << "[num_comp" << var_suffix << "*dim*" << (is_tensor && (dim >= 3) ? Q_name : "1") << "];\n";
        }
        break;
      case CEED_EVAL_WEIGHT:
        break;
        // LCOV_EXCL_START
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        break;  // TODO: Not implemented
                // LCOV_EXCL_STOP
    }
  }

  if (is_at_points) {
    // We need to handle batches of points
    code << "\n    // Note: Using batches of points\n";
    code << "    const CeedInt point_loop_bound = (blockDim.x * blockDim.y) * ceil(1.0 * max_num_points / (blockDim.x * blockDim.y));\n\n";
    code << "    #pragma unroll\n";
    code << "    for (CeedInt i = threadIdx.x + threadIdx.y * blockDim.x; i < point_loop_bound; i += blockDim.x * blockDim.y) {\n";
    code << "      const CeedInt p = i % max_num_points;\n\n";

    code << "      // -- Coordinates\n";
    code << "      CeedScalar r_x[dim];\n";
    code << "      ReadPoint<dim, coords_comp_stride, max_num_points>(data, elem, p, max_num_points, points.indices, points.coords, r_x);\n\n";

    code << "      // -- Input fields\n";
    for (CeedInt i = 0; i < num_input_fields; i++) {
      const char *field_name;
      std::string var_suffix = "_in_" + std::to_string(i);
      std::string P_name     = "P_1d" + var_suffix;

      CeedCallBackend(CeedOperatorFieldGetName(op_input_fields[i], &field_name));
      code << "      // ---- Input field " << i << ": " << field_name << "\n";
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
      // Basis action
      code << "      // EvalMode: " << CeedEvalModes[eval_mode] << "\n";
      switch (eval_mode) {
        case CEED_EVAL_NONE:
          code << "      CeedScalar r_s" << var_suffix << "[num_comp" << var_suffix << "];\n";
          code << "      ReadPoint<num_comp" << var_suffix << ", comp_stride" << var_suffix
               << ", max_num_points>(data, elem, p, max_num_points, indices.inputs[" << i << "], d" << var_suffix << ", r_s" << var_suffix << ");\n";
          break;
        case CEED_EVAL_INTERP:
          code << "      CeedScalar r_s" << var_suffix << "[num_comp" << var_suffix << "];\n";
          code << "      InterpAtPoints" << dim << "d<num_comp" << var_suffix << ", max_num_points, " << P_name << ", " << Q_name << ">(data, i, r_c"
               << var_suffix << ", r_x, r_s" << var_suffix << ");\n";
          break;
        case CEED_EVAL_GRAD:
          code << "      CeedScalar r_s" << var_suffix << "[num_comp" << var_suffix << "*dim];\n";
          code << "      GradAtPoints" << dim << "d<num_comp" << var_suffix << ", max_num_points, " << P_name << ", " << Q_name << ">(data, i, r_c"
               << var_suffix << ", r_x, r_s" << var_suffix << ");\n";
          break;
        case CEED_EVAL_WEIGHT:
          code << "      CeedScalar r_s" << var_suffix << "[1];\n";
          code << "      r_s" << var_suffix << "[0] = 1.0;\n";
          break;
          // LCOV_EXCL_START
        case CEED_EVAL_DIV:
        case CEED_EVAL_CURL:
          break;  // TODO: Not implemented
                  // LCOV_EXCL_STOP
      }
    }
    code << "\n      // -- Output fields\n";
    for (CeedInt i = 0; i < num_output_fields; i++) {
      const char *field_name;
      std::string var_suffix = "_out_" + std::to_string(i);

      CeedCallBackend(CeedOperatorFieldGetName(op_output_fields[i], &field_name));
      code << "      // ---- Output field " << i << ": " << field_name << "\n";
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
      // Basis action
      switch (eval_mode) {
        case CEED_EVAL_NONE:
          code << "      CeedScalar r_s" << var_suffix << "[num_comp" << var_suffix << "];\n";
          break;
        case CEED_EVAL_INTERP:
          code << "      CeedScalar r_s" << var_suffix << "[num_comp" << var_suffix << "];\n";
          break;
        case CEED_EVAL_GRAD:
          code << "      CeedScalar r_s" << var_suffix << "[num_comp" << var_suffix << "*dim];\n";
          break;
          // LCOV_EXCL_START
        case CEED_EVAL_WEIGHT:
          break;  // Should not occur
        case CEED_EVAL_DIV:
        case CEED_EVAL_CURL:
          break;  // TODO: Not implemented
                  // LCOV_EXCL_STOP
      }
    }

  } else if (use_3d_slices) {
    // We treat quadrature points per slice in 3d to save registers
    code << "\n    // Note: Using planes of 3D elements\n";
    code << "    #pragma unroll\n";
    code << "    for (CeedInt q = 0; q < " << Q_name << "; q++) {\n";
    code << "      // -- Input fields\n";
    for (CeedInt i = 0; i < num_input_fields; i++) {
      const char *field_name;
      std::string var_suffix = "_in_" + std::to_string(i);

      CeedCallBackend(CeedOperatorFieldGetName(op_input_fields[i], &field_name));
      code << "      // ---- Input field " << i << ": " << field_name << "\n";
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
      // Basis action
      code << "      // EvalMode: " << CeedEvalModes[eval_mode] << "\n";
      switch (eval_mode) {
        case CEED_EVAL_NONE:
          bool is_strided;

          code << "      CeedScalar r_s" << var_suffix << "[num_comp" << var_suffix << "];\n";

          CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &elem_rstr));
          CeedCallBackend(CeedElemRestrictionIsStrided(elem_rstr, &is_strided));
          if (is_strided) {
            bool    has_backend_strides;
            CeedInt num_elem, elem_size;

            CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
            CeedCallBackend(CeedElemRestrictionHasBackendStrides(elem_rstr, &has_backend_strides));
            CeedCallBackend(CeedElemRestrictionGetNumElements(elem_rstr, &num_elem));
            CeedInt strides[3] = {1, elem_size * num_elem, elem_size};

            if (!has_backend_strides) {
              CeedCallBackend(CeedElemRestrictionGetStrides(elem_rstr, strides));
            }
            code << "      // Strides: {" << strides[0] << ", " << strides[1] << ", " << strides[2] << "}\n";
            code << "      ReadEVecSliceStrided3d<num_comp" << var_suffix << ", " << Q_name << ", " << strides[0] << ", " << strides[1] << ", "
                 << strides[2] << ">(data, elem, q, d" << var_suffix << ", r_s" << var_suffix << ");\n";
          } else {
            CeedSize                 l_size = 0;
            CeedInt                  comp_stride;
            CeedElemRestriction_Hip *rstr_data;

            CeedCallBackend(CeedElemRestrictionGetLVectorSize(elem_rstr, &l_size));
            code << "      const CeedInt l_size" << var_suffix << " = " << l_size << ";\n";
            CeedCallBackend(CeedElemRestrictionGetCompStride(elem_rstr, &comp_stride));
            code << "      // CompStride: " << comp_stride << "\n";
            CeedCallBackend(CeedElemRestrictionGetData(elem_rstr, &rstr_data));
            data->indices.inputs[i] = (CeedInt *)rstr_data->d_offsets;
            code << "      ReadEVecSliceStandard3d<num_comp" << var_suffix << ", " << comp_stride << ", " << Q_name << ">(data, l_size" << var_suffix
                 << ", elem, q, indices.inputs[" << i << "], d" << var_suffix << ", r_s" << var_suffix << ");\n";
          }
          CeedCallBackend(CeedElemRestrictionDestroy(&elem_rstr));
          break;
        case CEED_EVAL_INTERP:
          code << "      CeedScalar r_s" << var_suffix << "[num_comp" << var_suffix << "];\n";
          code << "      for (CeedInt j = 0; j < num_comp" << var_suffix << "; j++) {\n";
          code << "        r_s" << var_suffix << "[j] = r_q" << var_suffix << "[q + j*" << Q_name << "];\n";
          code << "      }\n";
          break;
        case CEED_EVAL_GRAD:
          code << "      CeedScalar r_s" << var_suffix << "[num_comp" << var_suffix << "*dim];\n";
          code << "      GradColloSlice3d<num_comp" << var_suffix << ", " << Q_name << ", OP_T_1D>(data, q, r_q" << var_suffix << ", s_G"
               << var_suffix << ", r_s" << var_suffix << ");\n";
          break;
        case CEED_EVAL_WEIGHT:
          code << "      CeedScalar r_s" << var_suffix << "[1];\n";
          code << "      r_s" << var_suffix << "[0] = r_q" << var_suffix << "[q];\n";
          break;
          // LCOV_EXCL_START
        case CEED_EVAL_DIV:
        case CEED_EVAL_CURL:
          break;  // TODO: Not implemented
                  // LCOV_EXCL_STOP
      }
    }
    code << "\n      // -- Output fields\n";
    for (CeedInt i = 0; i < num_output_fields; i++) {
      const char *field_name;
      std::string var_suffix = "_out_" + std::to_string(i);

      CeedCallBackend(CeedOperatorFieldGetName(op_output_fields[i], &field_name));
      code << "      // ---- Output field " << i << ": " << field_name << "\n";
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
      // Basis action
      switch (eval_mode) {
        case CEED_EVAL_NONE:
          code << "      CeedScalar r_s" << var_suffix << "[num_comp" << var_suffix << "];\n";
          break;
        case CEED_EVAL_INTERP:
          code << "      CeedScalar r_s" << var_suffix << "[num_comp" << var_suffix << "];\n";
          break;
        case CEED_EVAL_GRAD:
          code << "      CeedScalar r_s" << var_suffix << "[num_comp" << var_suffix << "*dim];\n";
          break;
          // LCOV_EXCL_START
        case CEED_EVAL_WEIGHT:
          break;  // Should not occur
        case CEED_EVAL_DIV:
        case CEED_EVAL_CURL:
          break;  // TODO: Not implemented
                  // LCOV_EXCL_STOP
      }
    }
  } else {
    code << "\n    // Note: Using full elements\n";
    code << "    {\n";
    code << "      // -- Input fields\n";
    for (CeedInt i = 0; i < num_input_fields; i++) {
      const char *field_name;

      CeedCallBackend(CeedOperatorFieldGetName(op_input_fields[i], &field_name));
      code << "      // ---- Input field " << i << ": " << field_name << "\n";
      code << "      CeedScalar *r_s_in_" << i << " = r_q_in_" << i << ";\n";
    }
    code << "      // -- Output fields\n";
    for (CeedInt i = 0; i < num_output_fields; i++) {
      const char *field_name;

      CeedCallBackend(CeedOperatorFieldGetName(op_output_fields[i], &field_name));
      code << "      // ---- Output field " << i << ": " << field_name << "\n";
      code << "      CeedScalar *r_s_out_" << i << " = r_q_out_" << i << ";\n";
    }
  }

  // Input and output buffers
  code << "\n      // -- QFunction inputs and outputs\n";
  code << "      // ---- Inputs\n";
  code << "      CeedScalar *inputs[" << CeedIntMax(num_input_fields, 1) << "];\n";
  for (CeedInt i = 0; i < num_input_fields; i++) {
    const char *field_name;

    CeedCallBackend(CeedOperatorFieldGetName(op_input_fields[i], &field_name));
    code << "      // ------ Input field " << i << ": " << field_name << "\n";
    code << "      inputs[" << i << "] = r_s_in_" << i << ";\n";
  }
  code << "      // ---- Outputs\n";
  code << "      CeedScalar *outputs[" << CeedIntMax(num_output_fields, 1) << "];\n";
  for (CeedInt i = 0; i < num_output_fields; i++) {
    const char *field_name;

    CeedCallBackend(CeedOperatorFieldGetName(op_output_fields[i], &field_name));
    code << "      // ------ Output field " << i << ": " << field_name << "\n";
    code << "      outputs[" << i << "] = r_s_out_" << i << ";\n";
  }

  // Apply QFunction
  code << "\n      // -- Apply QFunction\n";
  code << "      " << qfunction_name << "(ctx, ";
  if (dim != 3 || is_at_points || use_3d_slices || !is_tensor) {
    code << "1";
  } else {
    code << Q_name;
  }
  code << ", inputs, outputs);\n";

  if (is_at_points) {
    // Map back to coefficients
    code << "\n      // -- Output fields\n";
    for (CeedInt i = 0; i < num_output_fields; i++) {
      const char *field_name;
      std::string var_suffix = "_out_" + std::to_string(i);
      std::string P_name     = "P_1d" + var_suffix;

      CeedCallBackend(CeedOperatorFieldGetName(op_output_fields[i], &field_name));
      code << "      // ---- Output field " << i << ": " << field_name << "\n";
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
      // Basis action
      code << "      // EvalMode: " << CeedEvalModes[eval_mode] << "\n";
      switch (eval_mode) {
        case CEED_EVAL_NONE: {
          CeedInt             comp_stride;
          CeedElemRestriction elem_rstr;

          CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[i], &elem_rstr));
          CeedCallBackend(CeedElemRestrictionGetCompStride(elem_rstr, &comp_stride));
          CeedCallBackend(CeedElemRestrictionDestroy(&elem_rstr));
          code << "      const CeedInt comp_stride" << var_suffix << " = " << comp_stride << ";\n";
          code << "      WritePoint<num_comp" << var_suffix << ", comp_stride" << var_suffix
               << ", max_num_points>(data, elem, i, points.num_per_elem[elem], indices.outputs[" << i << "]"
               << ", r_s" << var_suffix << ", d" << var_suffix << ");\n";
          break;
        }
        case CEED_EVAL_INTERP:
          code << "      if (i >= points.num_per_elem[elem]) {\n";
          code << "        for (CeedInt j = 0; j < num_comp" << var_suffix << "; j++) r_s" << var_suffix << "[j] = 0.0;\n";
          code << "      }\n";
          code << "      InterpTransposeAtPoints" << dim << "d<num_comp" << var_suffix << ", max_num_points, " << P_name << ", " << Q_name
               << ">(data, i, r_s" << var_suffix << ", r_x, r_c" << var_suffix << ");\n";
          break;
        case CEED_EVAL_GRAD:
          code << "      if (i >= points.num_per_elem[elem]) {\n";
          code << "        for (CeedInt j = 0; j < num_comp" << var_suffix << "*dim; j++) r_s" << var_suffix << "[j] = 0.0;\n";
          code << "      }\n";
          code << "      GradTransposeAtPoints" << dim << "d<num_comp" << var_suffix << ", max_num_points, " << P_name << ", " << Q_name
               << ">(data, i, r_s" << var_suffix << ", r_x, r_c" << var_suffix << ");\n";
          break;
          // LCOV_EXCL_START
        case CEED_EVAL_WEIGHT:
          break;  // Should not occur
        case CEED_EVAL_DIV:
        case CEED_EVAL_CURL:
          break;  // TODO: Not implemented
                  // LCOV_EXCL_STOP
      }
    }
  } else if (use_3d_slices) {
    // Copy or apply transpose grad, if needed
    code << "\n      // -- Output fields\n";
    for (CeedInt i = 0; i < num_output_fields; i++) {
      const char *field_name;
      std::string var_suffix = "_out_" + std::to_string(i);
      std::string P_name     = "P_1d" + var_suffix;

      CeedCallBackend(CeedOperatorFieldGetName(op_output_fields[i], &field_name));
      code << "      // ---- Output field " << i << ": " << field_name << "\n";
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
      // Basis action
      code << "      // EvalMode: " << CeedEvalModes[eval_mode] << "\n";
      switch (eval_mode) {
        case CEED_EVAL_NONE:
          code << "      for (CeedInt j = 0; j < num_comp" << var_suffix << " ; j++) {\n";
          code << "        r_q" << var_suffix << "[q + j*" << Q_name << "] = r_s" << var_suffix << "[j];\n";
          code << "      }\n";
          break;
        case CEED_EVAL_INTERP:
          code << "      for (CeedInt j = 0; j < num_comp" << var_suffix << " ; j++) {\n";
          code << "        r_q" << var_suffix << "[q + j*" << Q_name << "] = r_s" << var_suffix << "[j];\n";
          code << "      }\n";
          break;
        case CEED_EVAL_GRAD:
          code << "      GradColloSliceTranspose3d<num_comp" << var_suffix << ", " << Q_name << ", OP_T_1D>(data, q, r_s" << var_suffix << ", s_G"
               << var_suffix << ", r_q" << var_suffix << ");\n";
          break;
          // LCOV_EXCL_START
        case CEED_EVAL_WEIGHT:
          break;  // Should not occur
        case CEED_EVAL_DIV:
        case CEED_EVAL_CURL:
          break;  // TODO: Not implemented
                  // LCOV_EXCL_STOP
      }
    }
  }
  code << "    }\n";
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Build single operator kernel
//------------------------------------------------------------------------------
extern "C" int CeedOperatorBuildKernel_Hip_gen(CeedOperator op, bool *is_good_build) {
  bool                   is_tensor = true, is_at_points = false, use_3d_slices = false;
  Ceed                   ceed;
  CeedInt                Q_1d, num_input_fields, num_output_fields, dim = 1, max_num_points = 0, coords_comp_stride = 0;
  CeedQFunctionField    *qf_input_fields, *qf_output_fields;
  CeedQFunction_Hip_gen *qf_data;
  CeedQFunction          qf;
  CeedOperatorField     *op_input_fields, *op_output_fields;
  CeedOperator_Hip_gen  *data;
  std::ostringstream     code;

  CeedCallBackend(CeedOperatorGetData(op, &data));
  {
    bool is_setup_done;

    CeedCallBackend(CeedOperatorIsSetupDone(op, &is_setup_done));
    if (is_setup_done) {
      *is_good_build = !data->use_fallback;
      return CEED_ERROR_SUCCESS;
    }
  }

  // Check field compatibility
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  {
    bool has_shared_bases = true, is_all_tensor = true, is_all_nontensor = true;

    for (CeedInt i = 0; i < num_input_fields; i++) {
      CeedBasis basis;

      CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
      if (basis != CEED_BASIS_NONE) {
        bool        is_tensor = true;
        const char *resource;
        char       *resource_root;
        Ceed        basis_ceed;

        CeedCallBackend(CeedBasisIsTensor(basis, &is_tensor));
        is_all_tensor    = is_all_tensor && is_tensor;
        is_all_nontensor = is_all_nontensor && !is_tensor;
        CeedCallBackend(CeedBasisGetCeed(basis, &basis_ceed));
        CeedCallBackend(CeedGetResource(basis_ceed, &resource));
        CeedCallBackend(CeedGetResourceRoot(basis_ceed, resource, ":", &resource_root));
        has_shared_bases = has_shared_bases && !strcmp(resource_root, "/gpu/hip/shared");
        CeedCallBackend(CeedFree(&resource_root));
        CeedCallBackend(CeedDestroy(&basis_ceed));
      }
      CeedCallBackend(CeedBasisDestroy(&basis));
    }

    for (CeedInt i = 0; i < num_output_fields; i++) {
      CeedBasis basis;

      CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
      if (basis != CEED_BASIS_NONE) {
        bool        is_tensor = true;
        const char *resource;
        char       *resource_root;
        Ceed        basis_ceed;

        CeedCallBackend(CeedBasisIsTensor(basis, &is_tensor));
        is_all_tensor    = is_all_tensor && is_tensor;
        is_all_nontensor = is_all_nontensor && !is_tensor;

        CeedCallBackend(CeedBasisGetCeed(basis, &basis_ceed));
        CeedCallBackend(CeedGetResource(basis_ceed, &resource));
        CeedCallBackend(CeedGetResourceRoot(basis_ceed, resource, ":", &resource_root));
        has_shared_bases = has_shared_bases && !strcmp(resource_root, "/gpu/hip/shared");
        CeedCallBackend(CeedFree(&resource_root));
        CeedCallBackend(CeedDestroy(&basis_ceed));
      }
      CeedCallBackend(CeedBasisDestroy(&basis));
    }
    // -- Fallback to ref if not all bases are shared
    if (!has_shared_bases || (!is_all_tensor && !is_all_nontensor)) {
      *is_good_build = false;
      return CEED_ERROR_SUCCESS;
    }
  }
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedQFunctionGetData(qf, &qf_data));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));

  // Get operator data
  CeedCallBackend(CeedOperatorIsAtPoints(op, &is_at_points));
  CeedCallBackend(CeedOperatorBuildKernelData_Hip_gen(ceed, num_input_fields, op_input_fields, qf_input_fields, num_output_fields, op_output_fields,
                                                      qf_output_fields, &data->max_P_1d, &Q_1d, &dim, &is_tensor, &use_3d_slices));
  if (dim == 0) dim = 1;
  data->dim = dim;
  if (is_at_points) {
    CeedElemRestriction_Hip *rstr_data;
    CeedElemRestriction      rstr_points = NULL;

    CeedCallBackend(CeedOperatorAtPointsGetPoints(op, &rstr_points, NULL));
    CeedCallBackend(CeedElemRestrictionGetMaxPointsInElement(rstr_points, &max_num_points));
    CeedCallBackend(CeedElemRestrictionGetCompStride(rstr_points, &coords_comp_stride));
    CeedCallBackend(CeedElemRestrictionGetData(rstr_points, &rstr_data));
    data->points.indices = (CeedInt *)rstr_data->d_offsets;
    CeedCallBackend(CeedElemRestrictionDestroy(&rstr_points));
  }
  if (is_at_points) use_3d_slices = false;
  if (Q_1d == 0) {
    if (is_at_points) Q_1d = max_num_points;
    else CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q_1d));
  }
  data->Q_1d = Q_1d;

  // Check for restriction only identity operator
  {
    bool is_identity_qf;

    CeedCallBackend(CeedQFunctionIsIdentity(qf, &is_identity_qf));
    if (is_identity_qf) {
      CeedEvalMode eval_mode_in, eval_mode_out;

      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[0], &eval_mode_in));
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[0], &eval_mode_out));
      CeedCheck(eval_mode_in != CEED_EVAL_NONE || eval_mode_out != CEED_EVAL_NONE, ceed, CEED_ERROR_BACKEND,
                "Backend does not implement restriction only identity operators");
    }
  }

  // Load basis source files
  if (is_tensor) {
    code << "// Tensor basis source\n";
    code << "#include <ceed/jit-source/hip/hip-shared-basis-tensor-templates.h>\n\n";
  } else {
    code << "// Non-tensor basis source\n";
    code << "#include <ceed/jit-source/hip/hip-shared-basis-nontensor-templates.h>\n\n";
  }
  if (is_at_points) {
    code << "// AtPoints basis source\n";
    code << "#include <ceed/jit-source/hip/hip-shared-basis-tensor-at-points-templates.h>\n\n";
  }
  code << "// CodeGen operator source\n";
  code << "#include <ceed/jit-source/hip/hip-gen-templates.h>\n\n";

  // Get QFunction name
  std::string qfunction_name(qf_data->qfunction_name);
  std::string operator_name;

  operator_name = "CeedKernelHipGenOperator_" + qfunction_name;

  // Define CEED_Q_VLA
  code << "\n#undef CEED_Q_VLA\n";
  if (dim != 3 || is_at_points || use_3d_slices || !is_tensor) {
    code << "#define CEED_Q_VLA 1\n\n";
  } else {
    code << "#define CEED_Q_VLA " << Q_1d << "\n\n";
  }

  // Add user QFunction source
  {
    const char *source_path;

    CeedCallBackend(CeedQFunctionGetSourcePath(qf, &source_path));
    CeedCheck(source_path, ceed, CEED_ERROR_UNSUPPORTED, "/gpu/hip/gen backend requires QFunction source code file");

    code << "// User QFunction source\n";
    code << "#include \"" << source_path << "\"\n\n";
  }

  // Setup
  code << "\n// -----------------------------------------------------------------------------\n";
  code << "// Operator Kernel\n";
  code << "// \n";
  code << "// d_[in,out]_i:   CeedVector device array\n";
  code << "// r_[in,out]_e_i: Element vector register\n";
  code << "// r_[in,out]_q_i: Quadrature space vector register\n";
  code << "// r_[in,out]_c_i: AtPoints Chebyshev coefficients register\n";
  code << "// r_[in,out]_s_i: Quadrature space slice vector register\n";
  code << "// \n";
  code << "// s_B_[in,out]_i: Interpolation matrix, shared memory\n";
  code << "// s_G_[in,out]_i: Gradient matrix, shared memory\n";
  code << "// -----------------------------------------------------------------------------\n";
  code << "\nextern \"C\" __launch_bounds__(BLOCK_SIZE)\n";
  code << "__global__ void " << operator_name
       << "(CeedInt num_elem, void* ctx, FieldsInt_Hip indices, Fields_Hip fields, Fields_Hip B, Fields_Hip G, CeedScalar* W, Points_Hip points) {\n";

  // Scratch buffers
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedEvalMode eval_mode;

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    if (eval_mode != CEED_EVAL_WEIGHT) {  // Skip CEED_EVAL_WEIGHT
      code << "  const CeedScalar *__restrict__ d_in_" << i << " = fields.inputs[" << i << "];\n";
    }
  }
  for (CeedInt i = 0; i < num_output_fields; i++) {
    code << "  CeedScalar *__restrict__ d_out_" << i << " = fields.outputs[" << i << "];\n";
  }

  code << "  const CeedInt dim = " << dim << ";\n";
  code << "  const CeedInt " << (is_tensor ? "Q_1d" : "Q") << " = " << Q_1d << ";\n";
  if (is_at_points) {
    code << "  const CeedInt max_num_points = " << max_num_points << ";\n";
    code << "  const CeedInt coords_comp_stride = " << coords_comp_stride << ";\n";
  }

  // Shared data
  code << "  extern __shared__ CeedScalar slice[];\n";
  code << "  SharedData_Hip data;\n";
  code << "  data.t_id_x = threadIdx.x;\n";
  code << "  data.t_id_y = threadIdx.y;\n";
  code << "  data.t_id_z = threadIdx.z;\n";
  code << "  data.t_id  = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y*blockDim.x;\n";
  code << "  data.slice = slice + data.t_id_z*OP_T_1D" << ((!is_tensor || dim == 1) ? "" : "*OP_T_1D") << ";\n";

  // -- Determine input mat reuse
  FieldReuse_Hip input_matrix_reuse[CEED_FIELD_MAX];

  for (CeedInt i = 0; i < num_input_fields; i++) {
    input_matrix_reuse[i].index = -1;
  }
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedEvalMode eval_mode_i;
    CeedBasis    basis_i;

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode_i));
    if (eval_mode_i == CEED_EVAL_WEIGHT) continue;
    CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis_i));
    for (CeedInt j = 0; (input_matrix_reuse[i].index == -1) && (j < i); j++) {
      CeedEvalMode eval_mode_j;
      CeedBasis    basis_j;

      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[j], &eval_mode_j));
      if (eval_mode_j == CEED_EVAL_WEIGHT) continue;
      CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[j], &basis_j));
      if (basis_i == basis_j) {
        if (is_tensor) {
          input_matrix_reuse[i].index     = j;
          input_matrix_reuse[i].is_input  = true;
          input_matrix_reuse[i].eval_mode = eval_mode_j;
        } else {
          // For non-tensor can only re-use with the same eval mode
          if (eval_mode_i == eval_mode_j) {
            input_matrix_reuse[i].index     = j;
            input_matrix_reuse[i].is_input  = true;
            input_matrix_reuse[i].eval_mode = eval_mode_j;
          }
        }
      }
      CeedCallBackend(CeedBasisDestroy(&basis_j));
    }
    CeedCallBackend(CeedBasisDestroy(&basis_i));
  }

  // -- Determine output mat reuse
  FieldReuse_Hip output_matrix_reuse[CEED_FIELD_MAX];

  for (CeedInt i = 0; i < num_output_fields; i++) {
    output_matrix_reuse[i].index = -1;
  }
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedEvalMode eval_mode_i;
    CeedBasis    basis_i;

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode_i));
    CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis_i));
    for (CeedInt j = 0; (output_matrix_reuse[i].index == -1) && (j < num_input_fields); j++) {
      CeedEvalMode eval_mode_j;
      CeedBasis    basis_j;

      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[j], &eval_mode_j));
      if (eval_mode_j == CEED_EVAL_WEIGHT) continue;
      CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[j], &basis_j));
      if (basis_i == basis_j) {
        if (is_tensor) {
          output_matrix_reuse[i].index     = j;
          output_matrix_reuse[i].is_input  = true;
          output_matrix_reuse[i].eval_mode = eval_mode_j;
        } else {
          // For non-tensor can only re-use with the same eval mode
          if (eval_mode_i == eval_mode_j) {
            output_matrix_reuse[i].index     = j;
            output_matrix_reuse[i].is_input  = true;
            output_matrix_reuse[i].eval_mode = eval_mode_j;
          }
        }
      }
      CeedCallBackend(CeedBasisDestroy(&basis_j));
    }
    for (CeedInt j = 0; (output_matrix_reuse[i].index == -1) && (j < i); j++) {
      CeedEvalMode eval_mode_j;
      CeedBasis    basis_j;

      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[j], &eval_mode_j));
      if (eval_mode_j == CEED_EVAL_WEIGHT) continue;
      CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[j], &basis_j));
      if (basis_i == basis_j) {
        if (is_tensor) {
          output_matrix_reuse[i].index     = j;
          output_matrix_reuse[i].is_input  = false;
          output_matrix_reuse[i].eval_mode = eval_mode_j;
        } else {
          // For non-tensor can only re-use with the same eval mode
          if (eval_mode_i == eval_mode_j) {
            output_matrix_reuse[i].index     = j;
            output_matrix_reuse[i].is_input  = false;
            output_matrix_reuse[i].eval_mode = eval_mode_j;
          }
        }
      }
      CeedCallBackend(CeedBasisDestroy(&basis_j));
    }
    CeedCallBackend(CeedBasisDestroy(&basis_i));
  }

  // Initialize constants, and matrices B and G
  code << "\n  // Input field constants and basis data\n";
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedOperatorBuildKernelFieldData_Hip_gen(code, data, i, op_input_fields[i], qf_input_fields[i], input_matrix_reuse[i], Q_1d, true,
                                                             is_tensor, is_at_points, use_3d_slices));
  }
  code << "\n  // Output field constants and basis data\n";
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCallBackend(CeedOperatorBuildKernelFieldData_Hip_gen(code, data, i, op_output_fields[i], qf_output_fields[i], output_matrix_reuse[i], Q_1d,
                                                             false, is_tensor, is_at_points, use_3d_slices));
  }

  // Loop over all elements
  code << "\n  // Element loop\n";
  code << "  __syncthreads();\n";
  code << "  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x*blockDim.z) {\n";

  // -- Compute minimum buffer space needed
  CeedInt max_rstr_buffer_size = 1;

  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedInt             num_comp, elem_size;
    CeedElemRestriction elem_rstr;

    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &elem_rstr));
    CeedCallBackend(CeedElemRestrictionGetNumComponents(elem_rstr, &num_comp));
    CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
    max_rstr_buffer_size = CeedIntMax(max_rstr_buffer_size, num_comp * (is_tensor && (dim >= 3) ? elem_size : 1));
    CeedCallBackend(CeedElemRestrictionDestroy(&elem_rstr));
  }
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedInt             num_comp, elem_size;
    CeedElemRestriction elem_rstr;

    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[i], &elem_rstr));
    CeedCallBackend(CeedElemRestrictionGetNumComponents(elem_rstr, &num_comp));
    CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
    max_rstr_buffer_size = CeedIntMax(max_rstr_buffer_size, num_comp * (is_tensor && (dim >= 3) ? elem_size : 1));
    CeedCallBackend(CeedElemRestrictionDestroy(&elem_rstr));
  }
  code << "    // Scratch restriction buffer space\n";
  code << "    CeedScalar r_e_scratch[" << max_rstr_buffer_size << "];\n";

  // -- Determine best input field processing order
  CeedInt field_rstr_in_buffer[CEED_FIELD_MAX], input_field_order[CEED_FIELD_MAX];

  for (CeedInt i = 0; i < num_input_fields; i++) {
    field_rstr_in_buffer[i] = -1;
    input_field_order[i]    = -1;
  }
  {
    bool    is_ordered[CEED_FIELD_MAX];
    CeedInt curr_index = 0;

    for (CeedInt i = 0; i < num_input_fields; i++) is_ordered[i] = false;
    for (CeedInt i = 0; i < num_input_fields; i++) {
      CeedVector          vec_i;
      CeedElemRestriction rstr_i;

      if (is_ordered[i]) continue;
      field_rstr_in_buffer[i]       = i;
      is_ordered[i]                 = true;
      input_field_order[curr_index] = i;
      curr_index++;
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec_i));
      if (vec_i == CEED_VECTOR_NONE) continue;  // CEED_EVAL_WEIGHT
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &rstr_i));
      for (CeedInt j = i + 1; j < num_input_fields; j++) {
        CeedVector          vec_j;
        CeedElemRestriction rstr_j;

        CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[j], &vec_j));
        CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[j], &rstr_j));
        if (rstr_i == rstr_j && vec_i == vec_j) {
          field_rstr_in_buffer[j]       = i;
          is_ordered[j]                 = true;
          input_field_order[curr_index] = j;
          curr_index++;
        }
        CeedCallBackend(CeedVectorDestroy(&vec_j));
        CeedCallBackend(CeedElemRestrictionDestroy(&rstr_j));
      }
      CeedCallBackend(CeedVectorDestroy(&vec_i));
      CeedCallBackend(CeedElemRestrictionDestroy(&rstr_i));
    }
  }

  // -- Input restriction and basis
  code << "\n    // -- Input field restrictions and basis actions\n";
  for (CeedInt i = 0; i < num_input_fields; i++) {
    const char   *field_name;
    const CeedInt f = input_field_order[i];

    CeedCallBackend(CeedOperatorFieldGetName(op_input_fields[f], &field_name));
    code << "    // ---- Input field " << f << ": " << field_name << "\n";

    // ---- Restriction
    CeedCallBackend(CeedOperatorBuildKernelRestriction_Hip_gen(code, data, f, dim, field_rstr_in_buffer, op_input_fields[f], qf_input_fields[f], Q_1d,
                                                               true, is_tensor, is_at_points, use_3d_slices));

    // ---- Basis action
    CeedCallBackend(CeedOperatorBuildKernelBasis_Hip_gen(code, data, f, dim, op_input_fields[f], qf_input_fields[f], Q_1d, true, is_tensor,
                                                         is_at_points, use_3d_slices));
  }

  // -- Q function
  CeedCallBackend(CeedOperatorBuildKernelQFunction_Hip_gen(code, data, dim, max_num_points, num_input_fields, op_input_fields, qf_input_fields,
                                                           num_output_fields, op_output_fields, qf_output_fields, qfunction_name, Q_1d, is_tensor,
                                                           is_at_points, use_3d_slices));

  // -- Output basis and restriction
  code << "\n    // -- Output field basis action and restrictions\n";
  for (CeedInt i = 0; i < num_output_fields; i++) {
    const char *field_name;

    CeedCallBackend(CeedOperatorFieldGetName(op_output_fields[i], &field_name));
    code << "    // ---- Output field " << i << ": " << field_name << "\n";

    // ---- Basis action
    CeedCallBackend(CeedOperatorBuildKernelBasis_Hip_gen(code, data, i, dim, op_output_fields[i], qf_output_fields[i], Q_1d, false, is_tensor,
                                                         is_at_points, use_3d_slices));

    // ---- Restriction
    CeedCallBackend(CeedOperatorBuildKernelRestriction_Hip_gen(code, data, i, dim, NULL, op_output_fields[i], qf_output_fields[i], Q_1d, false,
                                                               is_tensor, is_at_points, use_3d_slices));
  }

  // Close loop and function
  code << "  }\n";
  code << "}\n";
  code << "// -----------------------------------------------------------------------------\n\n";

  CeedInt block_sizes[3] = {0, 0, 0};
  CeedInt num_elem;

  // Compile
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(BlockGridCalculate_Hip_gen(is_tensor ? dim : 1, num_elem, data->max_P_1d, Q_1d, block_sizes));
  if (is_at_points) block_sizes[2] = 1;
  {
    bool is_compile_good = false;

    CeedCallBackend(CeedTryCompile_Hip(ceed, code.str().c_str(), &is_compile_good, &data->module, 2, "OP_T_1D", block_sizes[0], "BLOCK_SIZE",
                                       block_sizes[0] * block_sizes[1] * block_sizes[2]));
    if (is_compile_good) {
      *is_good_build = true;
      CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, operator_name.c_str(), &data->op));
    } else {
      *is_good_build     = false;
      data->use_fallback = true;
    }
  }
  CeedCallBackend(CeedOperatorSetSetupDone(op));
  CeedCallBackend(CeedDestroy(&ceed));
  CeedCallBackend(CeedQFunctionDestroy(&qf));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
