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
#include <cuda_runtime.h>

#include <iostream>
#include <sstream>
#include <string>

#include "../cuda-ref/ceed-cuda-ref.h"
#include "../cuda-shared/ceed-cuda-shared.h"
#include "../cuda/ceed-cuda-common.h"
#include "../cuda/ceed-cuda-compile.h"
#include "ceed-cuda-gen.h"

//------------------------------------------------------------------------------
// Setup fields
//------------------------------------------------------------------------------
static int CeedOperatorBuildKernelFieldData_Cuda_gen(std::ostringstream &code, CeedOperator_Cuda_gen *data, CeedInt i, CeedOperatorField op_field,
                                                     CeedQFunctionField qf_field, CeedInt Q_1d, bool is_input, bool use_collograd_parallelization) {
  std::string            var_suffix = (is_input ? "_in_" : "_out_") + std::to_string(i);
  std::string            P_name = "P_1d" + var_suffix, Q_name = "Q_1d";
  CeedEvalMode           eval_mode = CEED_EVAL_NONE;
  CeedInt                elem_size = 0, num_comp = 0, P_1d = 0;
  CeedElemRestriction    elem_rstr;
  CeedBasis_Cuda_shared *basis_data;
  CeedBasis              basis;

  code << "  // -- " << (is_input ? "Input" : "Output") << " field " << i << "\n";

  // Get field data
  CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_field, &elem_rstr));
  if (elem_rstr != CEED_ELEMRESTRICTION_NONE) {
    CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
    CeedCallBackend(CeedElemRestrictionGetNumComponents(elem_rstr, &num_comp));
  }
  CeedCallBackend(CeedOperatorFieldGetBasis(op_field, &basis));
  if (basis != CEED_BASIS_NONE) {
    CeedCallBackend(CeedBasisGetData(basis, &basis_data));
    CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
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
      data->B.inputs[i] = basis_data->d_interp_1d;
      code << "  __shared__ CeedScalar s_B" << var_suffix << "[" << P_1d * Q_1d << "];\n";
      code << "  loadMatrix<" << P_name << ", " << Q_name << ">(data, B.inputs[" << i << "], s_B" << var_suffix << ");\n";
      break;
    case CEED_EVAL_GRAD:
      data->B.inputs[i] = basis_data->d_interp_1d;
      code << "  __shared__ CeedScalar s_B" << var_suffix << "[" << P_1d * Q_1d << "];\n";
      code << "  loadMatrix<" << P_name << ", " << Q_name << ">(data, B.inputs[" << i << "], s_B" << var_suffix << ");\n";
      if (use_collograd_parallelization) {
        data->G.inputs[i] = basis_data->d_collo_grad_1d;
        code << "  __shared__ CeedScalar s_G" << var_suffix << "[" << Q_1d * Q_1d << "];\n";
        code << "  loadMatrix<" << Q_name << ", " << Q_name << ">(data, G.inputs[" << i << "], s_G" << var_suffix << ");\n";
      } else {
        bool has_collo_grad = basis_data->d_collo_grad_1d;

        data->G.inputs[i] = has_collo_grad ? basis_data->d_collo_grad_1d : basis_data->d_grad_1d;
        if (has_collo_grad) {
          code << "  __shared__ CeedScalar s_G" << var_suffix << "[" << Q_1d * Q_1d << "];\n";
          code << "  loadMatrix<" << Q_name << ", " << Q_name << ">(data, G.inputs[" << i << "], s_G" << var_suffix << ");\n";
        } else {
          code << "  __shared__ CeedScalar s_G" << var_suffix << "[" << Q_1d * P_1d << "];\n";
          code << "  loadMatrix<" << P_name << ", " << Q_name << ">(data, G.inputs[" << i << "], s_G" << var_suffix << ");\n";
        }
      }
      break;
    case CEED_EVAL_WEIGHT:
      break;  // No action
    case CEED_EVAL_DIV:
      break;  // TODO: Not implemented
    case CEED_EVAL_CURL:
      break;  // TODO: Not implemented
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Restriction
//------------------------------------------------------------------------------
static int CeedOperatorBuildKernelRestriction_Cuda_gen(std::ostringstream &code, CeedOperator_Cuda_gen *data, CeedInt i, CeedInt dim,
                                                       CeedOperatorField op_field, CeedQFunctionField qf_field, CeedInt Q_1d, bool is_input,
                                                       bool use_collograd_parallelization) {
  std::string               var_suffix = (is_input ? "_in_" : "_out_") + std::to_string(i);
  std::string               P_name     = "P_1d" + var_suffix;
  CeedEvalMode              eval_mode  = CEED_EVAL_NONE;
  CeedInt                   elem_size = 0, num_comp = 0, P_1d = 0;
  CeedSize                  l_size;
  CeedElemRestriction_Cuda *rstr_data;
  CeedElemRestriction       elem_rstr;
  CeedBasis                 basis;

  // Get field data
  CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_field, &elem_rstr));
  if (elem_rstr != CEED_ELEMRESTRICTION_NONE) {
    CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
    CeedCallBackend(CeedElemRestrictionGetNumComponents(elem_rstr, &num_comp));
    CeedCallBackend(CeedElemRestrictionGetData(elem_rstr, &rstr_data));
  }
  CeedCallBackend(CeedOperatorFieldGetBasis(op_field, &basis));
  if (basis != CEED_BASIS_NONE) {
    CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
  }
  CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_field, &eval_mode));

  // Restriction
  if (is_input) {
    // Input
    if (eval_mode != CEED_EVAL_WEIGHT && !((eval_mode == CEED_EVAL_NONE) && use_collograd_parallelization)) {
      bool is_strided;

      code << "    CeedScalar r_e" << var_suffix << "[num_comp" << var_suffix << "*" << P_name << "];\n";
      CeedCallBackend(CeedElemRestrictionIsStrided(elem_rstr, &is_strided));
      if (!is_strided) {
        CeedInt comp_stride;

        CeedCallBackend(CeedElemRestrictionGetLVectorSize(elem_rstr, &l_size));
        code << "    const CeedInt l_size" << var_suffix << " = " << l_size << ";\n";
        CeedCallBackend(CeedElemRestrictionGetCompStride(elem_rstr, &comp_stride));
        code << "    // CompStride: " << comp_stride << "\n";
        data->indices.inputs[i] = (CeedInt *)rstr_data->d_offsets;
        code << "    readDofsOffset" << dim << "d<num_comp" << var_suffix << ", " << comp_stride << ", " << P_name << ">(data, l_size" << var_suffix
             << ", elem, indices.inputs[" << i << "], d" << var_suffix << ", r_e" << var_suffix << ");\n";
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
        code << "    readDofsStrided" << dim << "d<num_comp" << var_suffix << ", " << P_name << "," << strides[0] << "," << strides[1] << ","
             << strides[2] << ">(data, elem, d" << var_suffix << ", r_e" << var_suffix << ");\n";
      }
    }
  } else {
    // Output
    bool is_strided;

    CeedCallBackend(CeedElemRestrictionIsStrided(elem_rstr, &is_strided));
    if (!is_strided) {
      CeedInt comp_stride;

      CeedCallBackend(CeedElemRestrictionGetLVectorSize(elem_rstr, &l_size));
      code << "    const CeedInt l_size" << var_suffix << " = " << l_size << ";\n";
      CeedCallBackend(CeedElemRestrictionGetCompStride(elem_rstr, &comp_stride));
      code << "    // CompStride: " << comp_stride << "\n";
      data->indices.outputs[i] = (CeedInt *)rstr_data->d_offsets;
      code << "    writeDofsOffset" << dim << "d<num_comp" << var_suffix << ", " << comp_stride << ", " << P_name << ">(data, l_size" << var_suffix
           << ", elem, indices.outputs[" << i << "], r_e" << var_suffix << ", d" << var_suffix << ");\n";
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
      code << "    writeDofsStrided" << dim << "d<num_comp" << var_suffix << ", " << P_name << "," << strides[0] << "," << strides[1] << ","
           << strides[2] << ">(data, elem, r_e" << var_suffix << ", d" << var_suffix << ");\n";
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis
//------------------------------------------------------------------------------
static int CeedOperatorBuildKernelBasis_Cuda_gen(std::ostringstream &code, CeedOperator_Cuda_gen *data, CeedInt i, CeedInt dim,
                                                 CeedOperatorField op_field, CeedQFunctionField qf_field, CeedInt Q_1d, bool is_input,
                                                 bool use_collograd_parallelization) {
  std::string         var_suffix = (is_input ? "_in_" : "_out_") + std::to_string(i);
  std::string         P_name = "P_1d" + var_suffix, Q_name = "Q_1d";
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
  CeedCallBackend(CeedOperatorFieldGetBasis(op_field, &basis));
  if (basis != CEED_BASIS_NONE) {
    CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
  }
  CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_field, &eval_mode));

  // Basis
  code << "    // EvalMode: " << CeedEvalModes[eval_mode] << "\n";
  if (is_input) {
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        if (!use_collograd_parallelization) {
          code << "    CeedScalar *r_q" << var_suffix << " = r_e" << var_suffix << ";\n";
        }
        break;
      case CEED_EVAL_INTERP:
        code << "    CeedScalar r_q" << var_suffix << "[num_comp" << var_suffix << "*" << Q_name << "];\n";
        code << "    Interp" << (dim > 1 ? "Tensor" : "") << dim << "d<num_comp" << var_suffix << ", P_1d" << var_suffix << ", " << Q_name
             << ">(data, r_e" << var_suffix << ", s_B" << var_suffix << ", r_q" << var_suffix << ");\n";
        break;
      case CEED_EVAL_GRAD:
        if (use_collograd_parallelization) {
          code << "    CeedScalar r_q" << var_suffix << "[num_comp" << var_suffix << "*" << Q_name << "];\n";
          code << "    Interp" << (dim > 1 ? "Tensor" : "") << dim << "d<num_comp" << var_suffix << ", P_1d" << var_suffix << ", " << Q_name
               << ">(data, r_e" << var_suffix << ", s_B" << var_suffix << ", r_q" << var_suffix << ");\n";
        } else {
          code << "    CeedScalar r_q" << var_suffix << "[num_comp" << var_suffix << "*dim*" << Q_name << "];\n";
          code << "    Grad" << (dim > 1 ? "Tensor" : "") << (dim == 3 && Q_1d >= P_1d ? "Collocated" : "") << dim << "d<num_comp" << var_suffix
               << ", P_1d" << var_suffix << ", " << Q_name << ">(data, r_e" << var_suffix << ", s_B" << var_suffix << ", s_G" << var_suffix << ", r_q"
               << var_suffix << ");\n";
        }
        break;
      case CEED_EVAL_WEIGHT: {
        CeedBasis_Cuda_shared *basis_data;

        code << "    CeedScalar r_q" << var_suffix << "[" << Q_name << "];\n";
        CeedCallBackend(CeedBasisGetData(basis, &basis_data));
        data->W = basis_data->d_q_weight_1d;
        code << "    Weight" << (dim > 1 ? "Tensor" : "") << dim << "d<" << Q_name << ">(data, W, r_q" << var_suffix << ");\n";
        break;
      }
      // LCOV_EXCL_START
      case CEED_EVAL_DIV:
        break;  // TODO: Not implemented
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
        code << "    CeedScalar r_e" << var_suffix << "[num_comp" << var_suffix << "*" << P_name << "];\n";
        code << "    InterpTranspose" << (dim > 1 ? "Tensor" : "") << dim << "d<num_comp" << var_suffix << ", " << P_name << ", " << Q_name
             << ">(data, r_q" << var_suffix << ", s_B" << var_suffix << ", r_e" << var_suffix << ");\n";
        break;
      case CEED_EVAL_GRAD:
        code << "    CeedScalar r_e" << var_suffix << "[num_comp" << var_suffix << "*" << P_name << "];\n";
        if (use_collograd_parallelization) {
          code << "    InterpTranspose" << (dim > 1 ? "Tensor" : "") << dim << "d<num_comp" << var_suffix << ", " << P_name << ", " << Q_name
               << ">(data, r_q" << var_suffix << ", s_B" << var_suffix << ", r_e" << var_suffix << ");\n";
        } else {
          code << "    GradTranspose" << (dim > 1 ? "Tensor" : "") << (dim == 3 && Q_1d >= P_1d ? "Collocated" : "") << dim << "d<num_comp"
               << var_suffix << ", " << P_name << "," << Q_name << ">(data, r_q" << var_suffix << ", s_B" << var_suffix << ", s_G" << var_suffix
               << ", r_e" << var_suffix << ");\n";
        }
        break;
      // LCOV_EXCL_START
      case CEED_EVAL_WEIGHT:
        break;  // Should not occur
      case CEED_EVAL_DIV:
        break;  // TODO: Not implemented
      case CEED_EVAL_CURL:
        break;  // TODO: Not implemented
                // LCOV_EXCL_STOP
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunction
//------------------------------------------------------------------------------
static int CeedOperatorBuildKernelQFunction_Cuda_gen(std::ostringstream &code, CeedOperator_Cuda_gen *data, CeedInt dim, CeedInt num_input_fields,
                                                     CeedOperatorField *op_input_fields, CeedQFunctionField *qf_input_fields,
                                                     CeedInt num_output_fields, CeedOperatorField *op_output_fields,
                                                     CeedQFunctionField *qf_output_fields, std::string qfunction_name, CeedInt Q_1d,
                                                     bool use_collograd_parallelization) {
  std::string         Q_name    = "Q_1d";
  CeedEvalMode        eval_mode = CEED_EVAL_NONE;
  CeedElemRestriction elem_rstr;

  // Setup output arays
  code << "\n    // -- Output field setup\n";
  for (CeedInt i = 0; i < num_output_fields; i++) {
    std::string var_suffix = "_out_" + std::to_string(i);

    code << "    // ---- Output field " << i << "\n";
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_NONE || eval_mode == CEED_EVAL_INTERP) {
      code << "    CeedScalar r_q" << var_suffix << "[num_comp" << var_suffix << "*" << Q_name << "];\n";
    }
    if (eval_mode == CEED_EVAL_GRAD) {
      if (use_collograd_parallelization) {
        // Accumulator for gradient slices
        code << "    CeedScalar r_q" << var_suffix << "[num_comp" << var_suffix << "*" << Q_name << "];\n";
        code << "    for (CeedInt i = 0; i < num_comp" << var_suffix << "*" << Q_name << "; i++) {\n";
        code << "      r_q" << var_suffix << "[i] = 0.0;\n";
        code << "    }\n";
      } else {
        code << "    CeedScalar r_q" << var_suffix << "[num_comp" << var_suffix << "*dim*" << Q_name << "];\n";
      }
    }
  }

  // We treat quadrature points per slice in 3d to save registers
  if (use_collograd_parallelization) {
    code << "\n    // Note: Using planes of 3D elements\n";
    code << "#pragma unroll\n";
    code << "    for (CeedInt q = 0; q < " << Q_name << "; q++) {\n";
    code << "      // -- Input fields\n";
    for (CeedInt i = 0; i < num_input_fields; i++) {
      std::string var_suffix = "_in_" + std::to_string(i);

      code << "      // ---- Input field " << i << "\n";
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
            code << "      readSliceQuadsStrided3d<num_comp" << var_suffix << ", " << Q_name << "," << strides[0] << "," << strides[1] << ","
                 << strides[2] << ">(data, elem, q, d" << var_suffix << ", r_s" << var_suffix << ");\n";
          } else {
            CeedSize                  l_size = 0;
            CeedInt                   comp_stride;
            CeedElemRestriction_Cuda *rstr_data;

            CeedCallBackend(CeedElemRestrictionGetLVectorSize(elem_rstr, &l_size));
            code << "      const CeedInt l_size" << var_suffix << " = " << l_size << ";\n";
            CeedCallBackend(CeedElemRestrictionGetCompStride(elem_rstr, &comp_stride));
            code << "      // CompStride: " << comp_stride << "\n";
            CeedCallBackend(CeedElemRestrictionGetData(elem_rstr, &rstr_data));
            data->indices.inputs[i] = (CeedInt *)rstr_data->d_offsets;
            code << "      readSliceQuadsOffset3d<num_comp" << var_suffix << ", " << comp_stride << ", " << Q_name << ">(data, l_size" << var_suffix
                 << ", elem, q, indices.inputs[" << i << "], d" << var_suffix << ", r_s" << var_suffix << ");\n";
          }
          break;
        case CEED_EVAL_INTERP:
          code << "      CeedScalar r_s" << var_suffix << "[num_comp" << var_suffix << "];\n";
          code << "      for (CeedInt j = 0; j < num_comp" << var_suffix << "; j++) {\n";
          code << "        r_s" << var_suffix << "[j] = r_q" << var_suffix << "[q + j*" << Q_name << "];\n";
          code << "      }\n";
          break;
        case CEED_EVAL_GRAD:
          code << "      CeedScalar r_s" << var_suffix << "[num_comp" << var_suffix << "*dim];\n";
          code << "      gradCollo3d<num_comp" << var_suffix << ", " << Q_name << ">(data, q, r_q" << var_suffix << ", s_G" << var_suffix << ", r_s"
               << var_suffix << ");\n";
          break;
        case CEED_EVAL_WEIGHT:
          code << "      CeedScalar r_s" << var_suffix << "[1];\n";
          code << "      r_s" << var_suffix << "[0] = r_q" << var_suffix << "[q];\n";
          break;  // No action
                  // LCOV_EXCL_START
        case CEED_EVAL_DIV:
          break;  // TODO: Not implemented
        case CEED_EVAL_CURL:
          break;  // TODO: Not implemented
                  // LCOV_EXCL_STOP
      }
    }
    code << "\n      // -- Output fields\n";
    for (CeedInt i = 0; i < num_output_fields; i++) {
      std::string var_suffix = "_out_" + std::to_string(i);

      code << "      // ---- Output field " << i << "\n";
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
      // Basis action
      switch (eval_mode) {
        case CEED_EVAL_NONE:
          code << "      CeedScalar r_s" << var_suffix << "[num_comp" << var_suffix << "];\n";
          break;  // No action
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
          break;  // TODO: Not implemented
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
      code << "      // ---- Input field " << i << "\n";
      code << "      CeedScalar *r_s_in_" << i << " = r_q_in_" << i << ";\n";
    }
    code << "      // -- Output fields\n";
    for (CeedInt i = 0; i < num_output_fields; i++) {
      code << "      // ---- Output field " << i << "\n";
      code << "      CeedScalar *r_s_out_" << i << " = r_q_out_" << i << ";\n";
    }
  }

  // Input and output buffers
  code << "\n      // -- QFunction inputs and outputs\n";
  code << "      // ---- Inputs\n";
  code << "      CeedScalar* in[" << num_input_fields << "];\n";
  for (CeedInt i = 0; i < num_input_fields; i++) {
    code << "      // ------ Input field " << i << "\n";
    code << "      in[" << i << "] = r_s_in_" << i << ";\n";
  }
  code << "      // ---- Outputs\n";
  code << "      CeedScalar* out[" << num_output_fields << "];\n";
  for (CeedInt i = 0; i < num_output_fields; i++) {
    code << "      // ------ Output field " << i << "\n";
    code << "      out[" << i << "] = r_s_out_" << i << ";\n";
  }

  // Apply QFunction
  code << "\n      // -- Apply QFunction\n";
  code << "      " << qfunction_name << "(ctx, ";
  if (dim != 3 || use_collograd_parallelization) {
    code << "1";
  } else {
    code << "Q_1d";
  }
  code << ", in, out);\n";

  // Copy or apply transpose grad, if needed
  if (use_collograd_parallelization) {
    code << "      // -- Output fields\n";
    for (CeedInt i = 0; i < num_output_fields; i++) {
      std::string var_suffix = "_out_" + std::to_string(i);
      std::string P_name     = "P_1d" + var_suffix;

      code << "      // ---- Output field " << i << "\n";
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
      // Basis action
      code << "      // EvalMode: " << CeedEvalModes[eval_mode] << "\n";
      switch (eval_mode) {
        case CEED_EVAL_NONE:
          code << "      for (CeedInt j = 0; j < num_comp" << var_suffix << " ; j++) {\n";
          code << "        r_q" << var_suffix << "[q + j*" << Q_name << "] = r_s" << var_suffix << "[j];\n";
          code << "      }\n";
          break;  // No action
        case CEED_EVAL_INTERP:
          code << "      for (CeedInt j = 0; j < num_comp" << var_suffix << " ; j++) {\n";
          code << "        r_q" << var_suffix << "[q + j*" << Q_name << "] = r_s" << var_suffix << "[j];\n";
          code << "      }\n";
          break;
        case CEED_EVAL_GRAD:
          code << "      gradColloTranspose3d<num_comp" << var_suffix << ", " << Q_name << ">(data, q, r_s" << var_suffix << ", s_G" << var_suffix
               << ", r_q" << var_suffix << ");\n";
          break;
          // LCOV_EXCL_START
        case CEED_EVAL_WEIGHT:
          break;  // Should not occur
        case CEED_EVAL_DIV:
          break;  // TODO: Not implemented
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
extern "C" int CeedOperatorBuildKernel_Cuda_gen(CeedOperator op) {
  bool                    is_setup_done, is_identity_qf;
  struct cudaDeviceProp   prop;
  Ceed                    ceed;
  Ceed_Cuda              *ceed_data;
  CeedInt                 Q, P_1d = 0, Q_1d = 0, num_input_fields, num_output_fields, dim = 1;
  CeedEvalMode            eval_mode;
  CeedBasis               basis;
  CeedQFunctionField     *qf_input_fields, *qf_output_fields;
  CeedQFunction_Cuda_gen *qf_data;
  CeedQFunction           qf;
  CeedOperatorField      *op_input_fields, *op_output_fields;
  CeedOperator_Cuda_gen  *data;

  CeedCallBackend(CeedOperatorIsSetupDone(op, &is_setup_done));
  if (is_setup_done) return CEED_ERROR_SUCCESS;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &data));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedQFunctionGetData(qf, &qf_data));
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

  // Add atomicAdd function for old NVidia architectures
  CeedCallBackend(CeedGetData(ceed, &ceed_data));
  CeedCallBackend(cudaGetDeviceProperties(&prop, ceed_data->device_id));
  if ((prop.major < 6) && (CEED_SCALAR_TYPE != CEED_SCALAR_FP32)) {
    char       *atomic_add_source;
    const char *atomic_add_path;

    CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/cuda/cuda-atomic-add-fallback.h", &atomic_add_path));
    CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Atomic Add Source -----\n");
    CeedCallBackend(CeedLoadSourceToBuffer(ceed, atomic_add_path, &atomic_add_source));
    code << atomic_add_source;
    CeedCallBackend(CeedFree(&atomic_add_path));
    CeedCallBackend(CeedFree(&atomic_add_source));
  }

  // Load basis source files
  // TODO: Add non-tensor, AtPoints
  {
    char       *tensor_basis_kernel_source;
    const char *tensor_basis_kernel_path;

    CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h", &tensor_basis_kernel_path));
    CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Tensor Basis Kernel Source -----\n");
    CeedCallBackend(CeedLoadSourceToBuffer(ceed, tensor_basis_kernel_path, &tensor_basis_kernel_source));
    code << tensor_basis_kernel_source;
    CeedCallBackend(CeedFree(&tensor_basis_kernel_path));
    CeedCallBackend(CeedFree(&tensor_basis_kernel_source));
  }
  {
    char       *cuda_gen_template_source;
    const char *cuda_gen_template_path;

    CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/cuda/cuda-gen-templates.h", &cuda_gen_template_path));
    CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Cuda-Gen Template Source -----\n");
    CeedCallBackend(CeedLoadSourceToBuffer(ceed, cuda_gen_template_path, &cuda_gen_template_source));
    code << cuda_gen_template_source;
    CeedCallBackend(CeedFree(&cuda_gen_template_path));
    CeedCallBackend(CeedFree(&cuda_gen_template_source));
  }

  // Get QFunction source and name
  std::string qfunction_name(qf_data->qfunction_name);
  std::string operator_name;

  operator_name = "CeedKernelCudaGenOperator_" + qfunction_name;

  // Find dim, P_1d, Q_1d
  data->max_P_1d = 0;
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
    if (basis != CEED_BASIS_NONE) {
      bool is_tensor;

      // Collect dim, P_1d, and Q_1d
      CeedCallBackend(CeedBasisIsTensor(basis, &is_tensor));
      CeedCheck(is_tensor, ceed, CEED_ERROR_BACKEND, "Backend does not implement operators with non-tensor basis");
      CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
      data->max_P_1d = CeedIntMax(data->max_P_1d, P_1d);
      CeedCallBackend(CeedBasisGetDimension(basis, &dim));
      CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
    }
  }
  // Check output bases for Q_1d, dim as well
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
    if (basis != CEED_BASIS_NONE) {
      bool is_tensor;

      // Check for tensor bases
      CeedCallBackend(CeedBasisIsTensor(basis, &is_tensor));
      CeedCheck(is_tensor, ceed, CEED_ERROR_BACKEND, "Backend does not implement operators with non-tensor basis");
    }
  }
  data->dim  = dim;
  data->Q_1d = Q_1d;

  // Only use 3D collocated gradient parallelization strategy when gradient is computed
  bool use_collograd_parallelization = false;

  if (dim == 3) {
    bool                   was_grad_found = false;
    CeedBasis_Cuda_shared *basis_data;

    for (CeedInt i = 0; i < num_input_fields; i++) {
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
      if (eval_mode == CEED_EVAL_GRAD) {
        CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
        CeedCallBackend(CeedBasisGetData(basis, &basis_data));
        use_collograd_parallelization = basis_data->d_collo_grad_1d && (was_grad_found ? use_collograd_parallelization : true);
        was_grad_found                = true;
      }
    }
    for (CeedInt i = 0; i < num_output_fields; i++) {
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
      if (eval_mode == CEED_EVAL_GRAD) {
        CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
        CeedCallBackend(CeedBasisGetData(basis, &basis_data));
        use_collograd_parallelization = basis_data->d_collo_grad_1d && (was_grad_found ? use_collograd_parallelization : true);
        was_grad_found                = true;
      }
    }
  }

  // Define CEED_Q_VLA
  code << "\n#undef CEED_Q_VLA\n";
  if (dim != 3 || use_collograd_parallelization) {
    code << "#define CEED_Q_VLA 1\n\n";
  } else {
    code << "#define CEED_Q_VLA " << Q_1d << "\n\n";
  }

  // Add user QFunction source
  {
    std::string qfunction_source(qf_data->qfunction_source);

    code << qfunction_source;
  }

  // Setup
  code << "\n// -----------------------------------------------------------------------------\n";
  code << "// Operator Kernel\n";
  code << "// \n";
  code << "// d_[in,out]_i:   CeedVector device array\n";
  code << "// r_[in,out]_e_i: Element vector register\n";
  code << "// r_[in,out]_q_i: Quadrature space vector register\n";
  code << "// r_[in,out]_s_i: Quadrature space slice  vector register\n";
  code << "// \n";
  code << "// s_B_[in,out]_i: Interpolation matrix, shared memory\n";
  code << "// s_G_[in,out]_i: Gradient matrix, shared memory\n";
  code << "// -----------------------------------------------------------------------------\n";
  code << "extern \"C\" __global__ void " << operator_name
       << "(CeedInt num_elem, void* ctx, FieldsInt_Cuda indices, Fields_Cuda fields, Fields_Cuda B, Fields_Cuda G, CeedScalar *W) {\n";

  // Scratch buffers
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    if (eval_mode != CEED_EVAL_WEIGHT) {  // Skip CEED_EVAL_WEIGHT
      code << "  const CeedScalar *d_in_" << i << " = fields.inputs[" << i << "];\n";
    }
  }
  for (CeedInt i = 0; i < num_output_fields; i++) {
    code << "  CeedScalar *d_out_" << i << " = fields.outputs[" << i << "];\n";
  }

  code << "  const CeedInt dim = " << dim << ";\n";
  code << "  const CeedInt Q_1d = " << Q_1d << ";\n";

  // Shared data
  code << "  extern __shared__ CeedScalar slice[];\n";
  code << "  SharedData_Cuda data;\n";
  code << "  data.t_id_x = threadIdx.x;\n";
  code << "  data.t_id_y = threadIdx.y;\n";
  code << "  data.t_id_z = threadIdx.z;\n";
  code << "  data.t_id  = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y*blockDim.x;\n";
  code << "  data.slice = slice + data.t_id_z*T_1D" << (dim > 1 ? "*T_1D" : "") << ";\n";

  // Initialize constants, and matrices B and G
  code << "\n  // Input field constants and basis data\n";
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCall(
        CeedOperatorBuildKernelFieldData_Cuda_gen(code, data, i, op_input_fields[i], qf_input_fields[i], Q_1d, true, use_collograd_parallelization));
  }
  code << "\n  // Output field constants and basis data\n";
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCall(CeedOperatorBuildKernelFieldData_Cuda_gen(code, data, i, op_output_fields[i], qf_output_fields[i], Q_1d, false,
                                                       use_collograd_parallelization));
  }

  // Loop over all elements
  code << "\n  // Element loop\n";
  code << "  __syncthreads();\n";
  code << "  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x*blockDim.z) {\n";

  // -- Input restriction and basis
  code << "    // -- Input field restrictions and basis actions\n";
  for (CeedInt i = 0; i < num_input_fields; i++) {
    code << "    // ---- Input field " << i << "\n";

    // ---- Restriction
    CeedCallBackend(CeedOperatorBuildKernelRestriction_Cuda_gen(code, data, i, dim, op_input_fields[i], qf_input_fields[i], Q_1d, true,
                                                                use_collograd_parallelization));

    // ---- Basis action
    CeedCallBackend(
        CeedOperatorBuildKernelBasis_Cuda_gen(code, data, i, dim, op_input_fields[i], qf_input_fields[i], Q_1d, true, use_collograd_parallelization));
  }

  // -- Q function
  CeedCallBackend(CeedOperatorBuildKernelQFunction_Cuda_gen(code, data, dim, num_input_fields, op_input_fields, qf_input_fields, num_output_fields,
                                                            op_output_fields, qf_output_fields, qfunction_name, Q_1d, use_collograd_parallelization));

  // -- Output basis and restriction
  code << "\n    // -- Output field basis action and restrictions\n";
  for (CeedInt i = 0; i < num_output_fields; i++) {
    code << "    // ---- Output field " << i << "\n";

    // ---- Basis action
    CeedCallBackend(CeedOperatorBuildKernelBasis_Cuda_gen(code, data, i, dim, op_output_fields[i], qf_output_fields[i], Q_1d, false,
                                                          use_collograd_parallelization));

    // ---- Restriction
    CeedCallBackend(CeedOperatorBuildKernelRestriction_Cuda_gen(code, data, i, dim, op_output_fields[i], qf_output_fields[i], Q_1d, false,
                                                                use_collograd_parallelization));
  }

  // Close loop and function
  code << "  }\n";
  code << "}\n";
  code << "// -----------------------------------------------------------------------------\n\n";

  // View kernel for debugging
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "Generated Operator Kernels:\n");
  CeedDebug(ceed, code.str().c_str());

  CeedCallBackend(CeedCompile_Cuda(ceed, code.str().c_str(), &data->module, 1, "T_1D", CeedIntMax(Q_1d, data->max_P_1d)));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, operator_name.c_str(), &data->op));

  CeedCallBackend(CeedOperatorSetSetupDone(op));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
