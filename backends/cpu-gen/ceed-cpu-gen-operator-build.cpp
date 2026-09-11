// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#define CEED_DEBUG_COLOR 12

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>
#include <ceed/gen-tools.hpp>

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

#include "ceed-cpu-compile.h"
#include "ceed-cpu-gen.h"

#include "../ref/ceed-ref.h"

//------------------------------------------------------------------------------
// Setup fields
//------------------------------------------------------------------------------
static int CeedOperatorBuildKernelFieldData_Cpu_Gen(std::ostringstream &code, CeedOperator_Cpu_Gen *data, Tab &tab, CeedInt i,
                                                    CeedOperatorField op_field, CeedQFunctionField qf_field, bool is_input) {
  const char         *field_name;
  std::string         var_suffix = (is_input ? "_in_" : "_out_") + std::to_string(i);
  CeedElemRestriction elem_rstr;
  CeedBasis           basis;

  assert(i < CEED_FIELD_MAX);

  CeedCallBackend(CeedQFunctionFieldGetName(qf_field, &field_name));
  code << tab << "// ---- " << (is_input ? "Input" : "Output") << " Field " << i << ": " << field_name << "\n";

  // ElemRestriction data
  CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_field, &elem_rstr));
  if (elem_rstr != CEED_ELEMRESTRICTION_NONE) {
    CeedInt elem_size, num_comp;

    CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
    CeedCallBackend(CeedElemRestrictionGetNumComponents(elem_rstr, &num_comp));
    code << tab << "constexpr CeedInt elem_size" << var_suffix << " = " << elem_size << ";\n";
    code << tab << "constexpr CeedInt num_comp" << var_suffix << " = " << num_comp << ";\n";
  }
  CeedCallBackend(CeedElemRestrictionDestroy(&elem_rstr));

  // Basis data
  CeedCallBackend(CeedOperatorFieldGetBasis(op_field, &basis));
  if (basis != CEED_BASIS_NONE) {
    bool         is_tensor;
    CeedInt      dim, num_q_comp;
    CeedEvalMode eval_mode;

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_field, &eval_mode));
    CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, eval_mode, &num_q_comp));
    code << tab << "constexpr CeedInt num_q_comp" << var_suffix << " = " << num_q_comp << ";\n";
    CeedCallBackend(CeedBasisGetDimension(basis, &dim));
    code << tab << "constexpr CeedInt dim" << var_suffix << " = " << dim << ";\n";
    CeedCallBackend(CeedBasisIsTensor(basis, &is_tensor));
    if (is_tensor) {
      CeedInt P_1d;

      CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
      code << tab << "constexpr CeedInt P_1d" << var_suffix << " = " << P_1d << ";\n";
    } else {
      CeedInt P;

      CeedCallBackend(CeedBasisGetNumNodes(basis, &P));
      code << tab << "constexpr CeedInt P" << var_suffix << " = " << P << ";\n";
    }
  } else {
    code << tab << "constexpr CeedInt num_q_comp" << var_suffix << " = num_comp" << var_suffix << ";\n";
    code << tab << "constexpr CeedInt dim" << var_suffix << " = 1;\n";
    code << tab << "constexpr CeedInt P" << var_suffix << " = elem_size" << var_suffix << ";\n";
  }
  CeedCallBackend(CeedBasisDestroy(&basis));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Restriction
//------------------------------------------------------------------------------
static int CeedOperatorBuildKernelRestriction_Cpu_Gen(std::ostringstream &code, CeedOperator_Cpu_Gen *data, Tab &tab, CeedInt i,
                                                      CeedInt field_input_buffer[], CeedOperatorField op_field, CeedQFunctionField qf_field,
                                                      bool is_input, const CeedInt block_size) {
  std::string         var_suffix = (is_input ? "_in_" : "_out_") + std::to_string(i);
  CeedEvalMode        eval_mode  = CEED_EVAL_NONE;
  CeedInt             elem_size = 0, num_comp = 0;
  CeedRestrictionType rstr_type = CEED_RESTRICTION_STANDARD;
  CeedElemRestriction elem_rstr;

  assert(i < CEED_FIELD_MAX);
  assert(!is_input || field_input_buffer != NULL);

  // Label
  if (is_input) {
    const char *field_name;

    CeedCallBackend(CeedQFunctionFieldGetName(qf_field, &field_name));
    code << tab << "// ---- Input Field " << i << ": " << field_name << "\n";
  }

  // Get field data
  CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_field, &elem_rstr));
  if (elem_rstr != CEED_ELEMRESTRICTION_NONE) {
    CeedCallBackend(CeedElemRestrictionGetType(elem_rstr, &rstr_type));
    CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
    CeedCallBackend(CeedElemRestrictionGetNumComponents(elem_rstr, &num_comp));
    code << tab << "// ------ Restriction Type: " << CeedRestrictionTypes[rstr_type] << "\n";
  } else {
    code << tab << "// ------ Restriction Type: none\n";
  }
  CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_field, &eval_mode));

  // Create blockend restriction
  if (elem_rstr != CEED_ELEMRESTRICTION_NONE && block_size != 1 && (!is_input || (field_input_buffer && field_input_buffer[i] == i))) {
    CeedSize            l_size;
    CeedInt             num_elem, comp_stride;
    Ceed                ceed;
    CeedElemRestriction block_rstr = NULL;

    CeedCallBackend(CeedElemRestrictionGetCeed(elem_rstr, &ceed));
    CeedCallBackend(CeedElemRestrictionGetNumElements(elem_rstr, &num_elem));
    CeedCallBackend(CeedElemRestrictionGetLVectorSize(elem_rstr, &l_size));
    if (rstr_type != CEED_RESTRICTION_STRIDED && rstr_type != CEED_RESTRICTION_POINTS) {
      CeedCallBackend(CeedElemRestrictionGetCompStride(elem_rstr, &comp_stride));
    }
    switch (rstr_type) {
      case CEED_RESTRICTION_STANDARD: {
        const CeedInt *offsets = NULL;

        CeedCallBackend(CeedElemRestrictionGetOffsets(elem_rstr, CEED_MEM_HOST, &offsets));
        CeedCallBackend(CeedElemRestrictionCreateBlocked(ceed, num_elem, elem_size, block_size, num_comp, comp_stride, l_size, CEED_MEM_HOST,
                                                         CEED_COPY_VALUES, offsets, &block_rstr));
        CeedCallBackend(CeedElemRestrictionRestoreOffsets(elem_rstr, &offsets));
      } break;
      case CEED_RESTRICTION_ORIENTED: {
        const bool    *orients = NULL;
        const CeedInt *offsets = NULL;

        CeedCallBackend(CeedElemRestrictionGetOffsets(elem_rstr, CEED_MEM_HOST, &offsets));
        CeedCallBackend(CeedElemRestrictionGetOrientations(elem_rstr, CEED_MEM_HOST, &orients));
        CeedCallBackend(CeedElemRestrictionCreateBlockedOriented(ceed, num_elem, elem_size, block_size, num_comp, comp_stride, l_size, CEED_MEM_HOST,
                                                                 CEED_COPY_VALUES, offsets, orients, &block_rstr));
        CeedCallBackend(CeedElemRestrictionRestoreOffsets(elem_rstr, &offsets));
        CeedCallBackend(CeedElemRestrictionRestoreOrientations(elem_rstr, &orients));
      } break;
      case CEED_RESTRICTION_CURL_ORIENTED: {
        const CeedInt8 *curl_orients = NULL;
        const CeedInt  *offsets      = NULL;

        CeedCallBackend(CeedElemRestrictionGetOffsets(elem_rstr, CEED_MEM_HOST, &offsets));
        CeedCallBackend(CeedElemRestrictionGetCurlOrientations(elem_rstr, CEED_MEM_HOST, &curl_orients));
        CeedCallBackend(CeedElemRestrictionCreateBlockedCurlOriented(ceed, num_elem, elem_size, block_size, num_comp, comp_stride, l_size,
                                                                     CEED_MEM_HOST, CEED_COPY_VALUES, offsets, curl_orients, &block_rstr));
        CeedCallBackend(CeedElemRestrictionRestoreOffsets(elem_rstr, &offsets));
        CeedCallBackend(CeedElemRestrictionRestoreCurlOrientations(elem_rstr, &curl_orients));
      } break;
      case CEED_RESTRICTION_STRIDED: {
        CeedInt strides[3];

        CeedCallBackend(CeedElemRestrictionGetStrides(elem_rstr, strides));
        CeedCallBackend(CeedElemRestrictionCreateBlockedStrided(ceed, num_elem, elem_size, block_size, num_comp, l_size, strides, &block_rstr));
      } break;
      // LCOV_EXCL_START
      case CEED_RESTRICTION_POINTS:
        // Empty case - won't occur
        break;
        // LCOV_EXCL_STOP
    }
    CeedCallBackend(CeedElemRestrictionDestroy(&elem_rstr));
    if (is_input) {
      CeedCallBackend(CeedElemRestrictionReferenceCopy(block_rstr, &data->inputs_block_elem_rstr[i]));
    } else {
      CeedCallBackend(CeedElemRestrictionReferenceCopy(block_rstr, &data->outputs_block_elem_rstr[i]));
    }
    elem_rstr = block_rstr;
    CeedCallBackend(CeedDestroy(&ceed));
  }

  // Restriction
  if (is_input) {
    // Input
    if (field_input_buffer[i] != i) {
      // Restriction was already done for previous input
      std::string buffer_name = "e_vec_in_" + std::to_string(field_input_buffer[i]);

      code << tab << "CeedScalar *e_vec" << var_suffix << " = " << buffer_name << ";\n";
    } else if (eval_mode != CEED_EVAL_WEIGHT) {
      if (rstr_type == CEED_RESTRICTION_POINTS) {
        // No basis action, so space for e_vec_in_*/q_vec_in_* needs to be allocated
        code << tab << "CeedScalar e_vec" << var_suffix << "[num_comp" << var_suffix << " * max_points * block_size] = {0};\n";
      } else if (eval_mode == CEED_EVAL_NONE) {
        // No basis action, so space for e_vec_in_*/q_vec_in_* needs to be allocated
        code << tab << "CeedScalar e_vec" << var_suffix << "[num_comp" << var_suffix << " * elem_size" << var_suffix << " * block_size] = {0};\n";
      } else {
        // Otherwise we're using the scratch space
        code << tab << "CeedScalar *e_vec" << var_suffix << " = e_vec_scratch;\n";
      }
      switch (rstr_type) {
        case CEED_RESTRICTION_STRIDED: {
          bool has_backend_strides;

          CeedCallBackend(CeedElemRestrictionHasBackendStrides(elem_rstr, &has_backend_strides));
          CeedInt strides[3] = {1, elem_size, elem_size * num_comp};

          if (!has_backend_strides) {
            CeedCallBackend(CeedElemRestrictionGetStrides(elem_rstr, strides));
          }
          code << tab << "{\n";
          tab.push();
          code << tab << "constexpr CeedInt strides_0 = " << strides[0] << ", strides_1 = " << strides[1] << ", strides_2 = " << strides[2] << ";\n";
          code << tab << "\n";
          code << tab << "CeedCall(CeedElemRestriction_Apply_NoTranspose_Strided<block_size, num_comp" << var_suffix << ", elem_size" << var_suffix
               << ", num_elem, strides_0, strides_1, strides_2>(block, inputs[" << i << "].l_vec, e_vec" << var_suffix << "));\n";
          tab.pop();
          code << tab << "}\n";
          break;
        }
        case CEED_RESTRICTION_STANDARD: {
          CeedInt comp_stride;

          CeedCallBackend(CeedElemRestrictionGetCompStride(elem_rstr, &comp_stride));
          code << tab << "{\n";
          tab.push();
          code << tab << "constexpr CeedInt comp_stride = " << comp_stride << ";\n";
          code << tab << "\n";
          code << tab << "CeedCall(CeedElemRestriction_Apply_NoTranspose_Offset<block_size, num_comp" << var_suffix << ", elem_size" << var_suffix
               << ", num_elem, comp_stride>(block, inputs[" << i << "].offsets, inputs[" << i << "].l_vec, e_vec" << var_suffix << "));\n";
          tab.pop();
          code << tab << "}\n";
          break;
        }
        case CEED_RESTRICTION_ORIENTED: {
          CeedInt comp_stride;

          CeedCallBackend(CeedElemRestrictionGetCompStride(elem_rstr, &comp_stride));
          code << tab << "constexpr CeedInt comp_stride" << var_suffix << " = " << comp_stride << ";\n";
          code << tab << "\n";
          code << tab << "CeedCall(CeedElemRestriction_Apply_NoTranspose_Oriented<block_size, num_comp" << var_suffix << ", elem_size" << var_suffix
               << ", num_elem, comp_stride " << var_suffix << ">(block, inputs[" << i << "].offsets, inputs[" << i << "].orients, inputs[" << i
               << "].l_vec, e_vec" << var_suffix << "));\n";
          tab.pop();
          code << tab << "}\n";
          break;
        }
        case CEED_RESTRICTION_CURL_ORIENTED: {
          CeedInt comp_stride;

          CeedCallBackend(CeedElemRestrictionGetCompStride(elem_rstr, &comp_stride));
          code << tab << "{\n";
          tab.push();
          code << tab << "constexpr CeedInt comp_stride = " << comp_stride << ";\n";
          code << tab << "\n";
          code << tab << "CeedCall(CeedElemRestriction_Apply_NoTranspose_CurlOriented<block_size, num_comp" << var_suffix << ", elem_size"
               << var_suffix << ", num_elem, comp_stride " << var_suffix << ">(block, inputs[" << i << "].offsets, inputs[" << i
               << "].curl_orients, inputs[" << i << "].l_vec, e_vec" << var_suffix << "));\n";
          break;
        }
        case CEED_RESTRICTION_POINTS: {
          code << tab << "CeedCall(CeedElemRestriction_Apply_NoTranspose_AtPoints<block_size, num_comp>(block, inputs[" << i << "].offsets, inputs["
               << i << "].l_vec, e_vec" << var_suffix << "));\n";
          break;
        }
      }
    }
  } else {
    // Output
    switch (rstr_type) {
      case CEED_RESTRICTION_STRIDED: {
        bool has_backend_strides;

        CeedCallBackend(CeedElemRestrictionHasBackendStrides(elem_rstr, &has_backend_strides));
        CeedInt strides[3] = {1, elem_size, elem_size * num_comp};

        if (!has_backend_strides) {
          CeedCallBackend(CeedElemRestrictionGetStrides(elem_rstr, strides));
        }
        code << tab << "{\n";
        tab.push();
        code << tab << "constexpr CeedInt strides_0 = " << strides[0] << ", strides_1 = " << strides[1] << ", strides_2 = " << strides[2] << ";\n";
        code << tab << "\n";
        code << tab << "CeedCall(CeedElemRestriction_ApplyAdd_Transpose_Strided<block_size, num_comp" << var_suffix << ", elem_size" << var_suffix
             << ", num_elem, strides_0, strides_1, strides_2>(block, e_vec" << var_suffix << ", outputs[" << i << "].l_vec));\n";
        tab.pop();
        code << tab << "}\n";
        break;
      }
      case CEED_RESTRICTION_STANDARD: {
        CeedInt comp_stride;

        CeedCallBackend(CeedElemRestrictionGetCompStride(elem_rstr, &comp_stride));
        code << tab << "{\n";
        tab.push();
        code << tab << "constexpr CeedInt comp_stride = " << comp_stride << ";\n";
        code << tab << "\n";
        code << tab << "CeedCall(CeedElemRestriction_ApplyAdd_Transpose_Offset<block_size, num_comp" << var_suffix << ", elem_size" << var_suffix
             << ", num_elem, comp_stride>(block, outputs[" << i << "].offsets, e_vec" << var_suffix << ", outputs[" << i << "].l_vec));\n";
        tab.pop();
        code << tab << "}\n";
        break;
      }
      case CEED_RESTRICTION_ORIENTED: {
        CeedInt comp_stride;

        CeedCallBackend(CeedElemRestrictionGetCompStride(elem_rstr, &comp_stride));
        code << tab << "{\n";
        tab.push();
        code << tab << "constexpr CeedInt comp_stride = " << comp_stride << ";\n";
        code << tab << "\n";
        code << tab << "CeedCall(CeedElemRestriction_ApplyAdd_Transpose_Oriented<block_size, num_comp" << var_suffix << ", elem_size" << var_suffix
             << ", num_elem, comp_stride>(block, outputs[" << i << "].offsets, outputs[" << i << "].orients, e_vec" << var_suffix << ", outputs[" << i
             << "].l_vec));\n";
        tab.pop();
        code << tab << "}\n";
        break;
      }
      case CEED_RESTRICTION_CURL_ORIENTED: {
        CeedInt comp_stride;

        CeedCallBackend(CeedElemRestrictionGetCompStride(elem_rstr, &comp_stride));
        code << tab << "{\n";
        tab.push();
        code << tab << "constexpr CeedInt comp_stride = " << comp_stride << ";\n";
        code << tab << "\n";
        code << tab << "CeedCall(CeedElemRestriction_ApplyAdd_Transpose_CurlOriented<block_size, num_comp" << var_suffix << ", elem_size"
             << var_suffix << ", num_elem, comp_stride>(block, outputs[" << i << "].offsets, outputs[" << i << "].curl_orients, e_vec" << var_suffix
             << ", outputs[" << i << "].l_vec));\n";
        tab.pop();
        code << tab << "}\n";
        break;
      }
      case CEED_RESTRICTION_POINTS: {
        code << tab << "CeedCall(CeedElemRestriction_ApplyAdd_Transpose_AtPoints<block_size, num_comp>(block, outputs[" << i << "].offsets, e_vec"
             << var_suffix << ", outputs[" << i << "].l_vec));\n";
        break;
      }
    }
  }
  // Reference backend data
  if (elem_rstr != CEED_ELEMRESTRICTION_NONE) {
    if (is_input) {
      CeedElemRestriction_Ref *ref_data;

      CeedCallBackend(CeedElemRestrictionGetData(elem_rstr, &ref_data));
      data->inputs[i].offsets      = ref_data->offsets;
      data->inputs[i].orients      = ref_data->orients;
      data->inputs[i].curl_orients = ref_data->curl_orients;
    } else {
      CeedElemRestriction_Ref *ref_data;

      CeedCallBackend(CeedElemRestrictionGetData(elem_rstr, &ref_data));
      data->outputs[i].offsets      = ref_data->offsets;
      data->outputs[i].orients      = ref_data->orients;
      data->outputs[i].curl_orients = ref_data->curl_orients;
    }
  }
  CeedCallBackend(CeedElemRestrictionDestroy(&elem_rstr));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis
//------------------------------------------------------------------------------
static int CeedOperatorBuildKernelBasis_Cpu_Gen(std::ostringstream &code, CeedOperator_Cpu_Gen *data, Tab &tab, CeedInt i, CeedOperatorField op_field,
                                                CeedQFunctionField qf_field, bool is_input, bool is_at_points) {
  bool      is_tensor = true, is_collocated = true;
  CeedBasis basis;
  CeedCallBackend(CeedOperatorFieldGetBasis(op_field, &basis));
  CeedCallBackend(CeedBasisIsTensor(basis, &is_tensor));
  CeedCallBackend(CeedBasisIsCollocated(basis, &is_collocated));

  std::string         var_suffix = (is_input ? "_in_" : "_out_") + std::to_string(i);
  std::string         P_name     = (is_tensor ? "P_1d" : "P") + var_suffix;
  CeedEvalMode        eval_mode  = CEED_EVAL_NONE;
  CeedInt             dim = 0, elem_size = 0, num_comp = 0, P_1d = 0;
  CeedElemRestriction elem_rstr;

  assert(i < CEED_FIELD_MAX);

  // Label
  if (!is_input) {
    const char *field_name;

    CeedCallBackend(CeedQFunctionFieldGetName(qf_field, &field_name));
    code << tab << "// ---- Output Field " << i << ": " << field_name << "\n";
  }

  // Get field data
  CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_field, &elem_rstr));
  if (elem_rstr != CEED_ELEMRESTRICTION_NONE) {
    CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
    CeedCallBackend(CeedElemRestrictionGetNumComponents(elem_rstr, &num_comp));
  }
  CeedCallBackend(CeedElemRestrictionDestroy(&elem_rstr));
  if (basis != CEED_BASIS_NONE) {
    CeedCallBackend(CeedBasisGetDimension(basis, &dim));
    if (is_tensor) {
      CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
    } else {
      CeedCallBackend(CeedBasisGetNumNodes(basis, &P_1d));
    }
  }
  CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_field, &eval_mode));

  // Basis
  code << tab << "// ------ Evaluation Mode: " << CeedEvalModes[eval_mode] << "\n";
  if (is_input) {
    if (eval_mode == CEED_EVAL_WEIGHT) {
      // Handled separately
    } else if (eval_mode != CEED_EVAL_NONE) {
      code << tab << "CeedScalar q_vec" << var_suffix << "[num_q_comp" << var_suffix << " * num_comp" << var_suffix << " * Q * block_size] = {0};\n";
    } else {
      code << tab << "CeedScalar *q_vec" << var_suffix << " = e_vec" << var_suffix << ";\n";
    }
    if (is_tensor) {
      switch (eval_mode) {
        case CEED_EVAL_NONE:
          break;
        case CEED_EVAL_INTERP: {
          std::string name = (is_at_points ? "AtPoints_" : "Tensor_") + std::to_string(dim) + "D";

          code << tab << "CeedCall(CeedBasis_Apply_NoTranspose_Interp_" << name << "<block_size, num_comp" << var_suffix << ", " << P_name
               << ", Q_1d>(inputs[" << i << "].interp, e_vec" << var_suffix << ", q_vec" << var_suffix << "));\n";
        } break;
        case CEED_EVAL_GRAD: {
          CeedBasis_Ref *ref_data;

          CeedCallBackend(CeedBasisGetData(basis, &ref_data));
          std::string name =
              (is_at_points ? "AtPoints_" : (dim > 2 && ref_data->collo_grad_1d ? "Collo_Tensor_" : "Tensor_")) + std::to_string(dim) + "D";

          code << tab << "CeedCall(CeedBasis_Apply_NoTranspose_Grad_" << name << "<block_size, num_comp" << var_suffix << ", " << P_name
               << ", Q_1d>(inputs[" << i << "].interp, inputs[" << i << "].grad, e_vec" << var_suffix << ", q_vec" << var_suffix << "));\n";
        } break;
        case CEED_EVAL_WEIGHT: {
          if (is_at_points) {
            code << tab << "CeedScalar q_vec" << var_suffix << "[max_points * block_size] = {0};\n";
            code << tab << "CeedCall(CeedBasis_Apply_Weight_AtPoints<block_size, max_points>(q_vec" << var_suffix << "));\n";
          } else {
            std::string name = "Tensor_" + std::to_string(dim) + "D";

            code << tab << "CeedScalar q_vec" << var_suffix << "[Q * block_size] = {0};\n";
            code << tab << "CeedCall(CeedBasis_Apply_Weight_" << name << "<block_size, Q_1d>(inputs[" << i << "].weights, q_vec" << var_suffix
                 << "));\n";
          }
        } break;
        case CEED_EVAL_DIV:
        case CEED_EVAL_CURL:
          data->use_fallback = true;
          break;  // TODO: Not implemented
      }
    } else {
      if (eval_mode == CEED_EVAL_WEIGHT) {
        code << tab << "CeedScalar q_vec" << var_suffix << "[Q * block_size] = {0};\n";
        code << tab << "CeedCall(CeedBasis_Apply_Weight_Tensor_1D<block_size, Q>(inputs[" << i << "].weights, q_vec" << var_suffix << "));\n";
      } else if (eval_mode != CEED_EVAL_NONE) {
        code << tab << "CeedCall(CeedBasis_Apply_NoTranspose_NonTensor<block_size, num_comp" << var_suffix << ", num_q_comp" << var_suffix << ", "
             << P_name << ", Q>(inputs[" << i << "].";
        switch (eval_mode) {
          case CEED_EVAL_NONE:
            break;
          case CEED_EVAL_WEIGHT:
            break;
          case CEED_EVAL_INTERP:
            code << "interp";
            break;
          case CEED_EVAL_GRAD:
            code << "grad";
            break;
          case CEED_EVAL_DIV:
            code << "div";
            break;
          case CEED_EVAL_CURL:
            code << "curl";
            break;
        }
        code << ", e_vec" << var_suffix << ", q_vec" << var_suffix << "));\n";
      }
    }
  } else {
    if (eval_mode == CEED_EVAL_NONE) {
      code << tab << "CeedScalar *e_vec" << var_suffix << " = q_vec" << var_suffix << ";\n";
    } else {
      // TODO: BasisApplyAdd to reduce repeat rstrs
      code << tab << "CeedScalar *e_vec" << var_suffix << " = e_vec_scratch;\n";
    }
    if (is_tensor) {
      switch (eval_mode) {
        case CEED_EVAL_NONE:
          break;
        case CEED_EVAL_INTERP: {
          CeedBasis_Ref *ref_data;

          CeedCallBackend(CeedBasisGetData(basis, &ref_data));
          std::string name =
              (is_at_points ? "AtPoints_" : (dim > 2 && ref_data->collo_grad_1d ? "Collo_Tensor_" : "Tensor_")) + std::to_string(dim) + "D";

          code << tab << "CeedCall(CeedBasis_Apply_Transpose_Interp_" << name << "<block_size, num_comp" << var_suffix << ", " << P_name
               << ", Q_1d>(outputs[" << i << "].interp, q_vec" << var_suffix << ", e_vec" << var_suffix << "));\n";
        } break;
        case CEED_EVAL_GRAD: {
          std::string name = (is_at_points ? "AtPoints_" : "Tensor_") + std::to_string(dim) + "D";

          code << tab << "CeedCall(CeedBasis_Apply_NoTranspose_Grad_" << name << "<block_size, num_comp" << var_suffix << ", " << P_name
               << ", Q_1d>(outputs[" << i << "].interp, outputs[" << i << "].grad, q_vec" << var_suffix << ", e_vec" << var_suffix << "));\n";
        } break;
        case CEED_EVAL_WEIGHT:
        case CEED_EVAL_DIV:
        case CEED_EVAL_CURL:
          data->use_fallback = true;
          break;  // TODO: Not implemented
      }
    } else {
      if (eval_mode == CEED_EVAL_WEIGHT) {
        // Should not occur
      } else if (eval_mode != CEED_EVAL_NONE) {
        code << tab << "CeedCall(CeedBasis_Apply_Transpose_NonTensor<block_size, num_comp" << var_suffix << ", num_q_comp" << var_suffix << ", "
             << P_name << ", Q>(outputs[" << i << "].";
        switch (eval_mode) {
          case CEED_EVAL_NONE:
            break;
          case CEED_EVAL_WEIGHT:
            break;
          case CEED_EVAL_INTERP:
            code << "interp";
            break;
          case CEED_EVAL_GRAD:
            code << "grad";
            break;
          case CEED_EVAL_DIV:
            code << "div";
            break;
          case CEED_EVAL_CURL:
            code << "curl";
            break;
        }
        code << ", q_vec" << var_suffix << ", e_vec" << var_suffix << "));\n";
      }
    }
  }
  // Reference Basis data
  switch (eval_mode) {
    case CEED_EVAL_NONE:
      break;
    case CEED_EVAL_WEIGHT:
      CeedCallBackend(CeedBasisGetQWeights(basis, &data->inputs[i].weights));
      break;
    case CEED_EVAL_INTERP: {
      const CeedScalar **ptr = is_input ? &data->inputs[i].interp : &data->outputs[i].interp;

      if (is_tensor) {
        CeedCallBackend(CeedBasisGetInterp1D(basis, ptr));
      } else {
        CeedCallBackend(CeedBasisGetInterp(basis, ptr));
      }
    } break;
    case CEED_EVAL_GRAD: {
      if (is_tensor) {
        const CeedScalar **interp_ptr = is_input ? &data->inputs[i].interp : &data->outputs[i].interp;
        const CeedScalar **grad_ptr   = is_input ? &data->inputs[i].grad : &data->outputs[i].grad;
        CeedBasis_Ref     *ref_data;

        CeedCallBackend(CeedBasisGetInterp1D(basis, interp_ptr));
        CeedCallBackend(CeedBasisGetData(basis, &ref_data));
        if (dim > 2 && ref_data->collo_grad_1d) {
          *grad_ptr = ref_data->collo_grad_1d;
        } else {
          CeedCallBackend(CeedBasisGetGrad1D(basis, grad_ptr));
        }
      } else {
        const CeedScalar **ptr = is_input ? &data->inputs[i].grad : &data->outputs[i].grad;

        CeedCallBackend(CeedBasisGetGrad(basis, ptr));
      }
    } break;
    case CEED_EVAL_DIV: {
      const CeedScalar **ptr = is_input ? &data->inputs[i].div : &data->outputs[i].div;

      CeedCallBackend(CeedBasisGetDiv(basis, ptr));
    } break;
    case CEED_EVAL_CURL: {
      const CeedScalar **ptr = is_input ? &data->inputs[i].curl : &data->outputs[i].curl;

      CeedCallBackend(CeedBasisGetCurl(basis, ptr));
    } break;
  }
  CeedCallBackend(CeedBasisDestroy(&basis));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunction
//------------------------------------------------------------------------------
static int CeedOperatorBuildKernelQFunction_Cpu_Gen(std::ostringstream &code, CeedOperator_Cpu_Gen *data, Tab &tab, CeedInt num_input_fields,
                                                    CeedOperatorField *op_input_fields, CeedQFunctionField *qf_input_fields,
                                                    CeedInt num_output_fields, CeedOperatorField *op_output_fields,
                                                    CeedQFunctionField *qf_output_fields, const char *qfunction_name, bool is_at_points) {
  // Setup input array
  code << tab << "// ---- QFunction inputs\n";
  code << tab << "const CeedScalar* q_vecs_in[" << num_input_fields << "] = {\n";
  tab.push();
  for (CeedInt i = 0; i < num_input_fields; i++) {
    code << tab << "q_vec_in_" << i << ",\n";
  }
  tab.pop();
  code << tab << "};\n\n";
  // Setup input array
  code << tab << "// ---- QFunction outputs\n";
  for (CeedInt i = 0; i < num_output_fields; i++) {
    code << tab << "CeedScalar q_vec_out_" << i << "[num_q_comp_out_" << i << " * Q * block_size];\n";
  }
  code << tab << "CeedScalar* q_vecs_out[" << num_output_fields << "] = {\n";
  tab.push();
  for (CeedInt i = 0; i < num_output_fields; i++) {
    code << tab << "q_vec_out_" << i << ",\n";
  }
  tab.pop();
  code << tab << "};\n\n";
  // Call QFunction
  code << tab << "// ---- Call User QFunction\n";
  if (is_at_points) {
    code << tab << "CeedCall(" << std::string(qfunction_name) << "(ctx, num_points, q_vecs_in, q_vecs_out));\n\n";
  } else {
    code << tab << "CeedCall(" << std::string(qfunction_name) << "(ctx, Q * block_size, q_vecs_in, q_vecs_out));\n\n";
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Operator Build
//------------------------------------------------------------------------------

extern "C" int CeedOperatorBuildKernel_Cpu_Gen(CeedOperator op, bool *is_good_build) {
  const char           *qfunction_name;
  bool                  is_at_points = false;
  CeedInt               num_input_fields, num_output_fields, block_size = 1;
  Ceed                  ceed;
  CeedQFunction         qf;
  CeedQFunctionField   *qf_input_fields, *qf_output_fields;
  CeedOperatorField    *op_input_fields, *op_output_fields;
  CeedOperator_Cpu_Gen *data;
  std::ostringstream    code;
  Tab                   tab;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &data));
  {
    bool is_setup_done;

    CeedCallBackend(CeedOperatorIsSetupDone(op, &is_setup_done));
    if (is_setup_done) {
      *is_good_build = !data->use_fallback;
      return CEED_ERROR_SUCCESS;
    }
  }
  CeedCallBackend(CeedOperatorIsAtPoints(op, &is_at_points));
  if (!is_at_points) {
    Ceed_Cpu_Gen *ceed_data;

    CeedCallBackend(CeedGetData(ceed, &ceed_data));
    block_size = ceed_data->block_size;
  }
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));

  // Load utils
  code << tab << "#include <ceed/jit-source/cpu-gen/cpu-gen-utils.h>\n\n";

  // Load object source files
  code << tab << "// Ceed object templates\n";
  code << tab << "#include <ceed/jit-source/cpu-gen/cpu-gen-restriction-templates.h>\n";
  code << tab << "#include <ceed/jit-source/cpu-gen/cpu-gen-basis-templates.h>\n\n";

  code << "\n" << tab << "#undef CEED_Q_VLA\n";
  if (is_at_points) {
    // TODO: fix this
    code << tab << "#define CEED_Q_VLA 1\n\n";
  } else {
    CeedInt Q;

    CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
    code << tab << "#define CEED_Q_VLA " << Q * block_size << "\n\n";
  }

  // Add user QFunction source
  {
    const char *source_path;

    CeedCallBackend(CeedQFunctionGetSourcePath(qf, &source_path));
    CeedCheck(source_path, ceed, CEED_ERROR_UNSUPPORTED, "/cpu/self/gen backends require QFunction source code file");

    code << tab << "// User QFunction source\n";
    code << tab << "#include \"" << source_path << "\"\n\n";
  }

  // Get QFunction name
  std::string operator_name;

  CeedCallBackend(CeedQFunctionGetName(qf, &qfunction_name));
  operator_name = "CeedCpuGenOperator_" + std::string(qfunction_name);

  // Open function body
  code << tab << "// Operator function\n";
  code << tab << "extern \"C\" int " << operator_name << "(void *ctx, const InputFieldData_Cpu_Gen *inputs, OutputFieldData_Cpu_Gen *outputs) {\n";
  tab.push();

  // Get problem info
  code << tab << "// Operator constants\n";
  code << tab << "constexpr CeedInt block_size = " << block_size << ";\n";
  if (!is_at_points) {
    CeedInt Q;

    CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
    code << tab << "constexpr CeedInt Q = " << Q << ";\n";
  }
  {
    CeedInt Q_1d = -1;

    for (CeedInt i = 0; (Q_1d == -1) && (i < num_input_fields); i++) {
      bool      is_tensor;
      CeedBasis basis;

      CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
      CeedCallBackend(CeedBasisIsTensor(basis, &is_tensor));
      if (is_tensor) {
        CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
      }
      CeedCallBackend(CeedBasisDestroy(&basis));
    }
    for (CeedInt i = 0; (Q_1d == -1) && (i < num_output_fields); i++) {
      bool      is_tensor;
      CeedBasis basis;

      CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
      CeedCallBackend(CeedBasisIsTensor(basis, &is_tensor));
      if (is_tensor) {
        CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
      }
      CeedCallBackend(CeedBasisDestroy(&basis));
    }
    if (Q_1d != -1) {
      code << tab << "constexpr CeedInt Q_1d = " << Q_1d << ";\n";
    }
  }
  if (is_at_points) {
    CeedInt             max_points;
    CeedElemRestriction rstr_points = NULL;

    CeedCallBackend(CeedOperatorAtPointsGetPoints(op, &rstr_points, NULL));
    CeedCallBackend(CeedElemRestrictionGetMaxPointsInElement(rstr_points, &max_points));
    code << tab << "constexpr CeedInt max_points = " << max_points << "\n";
    CeedCallBackend(CeedElemRestrictionDestroy(&rstr_points));
  }
  {
    CeedInt num_elem;

    CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
    code << tab << "constexpr CeedInt num_elem = " << num_elem << ";\n";
    code << tab << "constexpr CeedInt num_blocks = (num_elem / block_size) + !!(num_elem % block_size);\n\n";
  }

  // Determine best input field processing order
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

  // Field info
  code << tab << "// Field constants\n";
  code << tab << "// -- Input Fields\n";
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedInt field = input_field_order[i];

    CeedCallBackend(CeedOperatorBuildKernelFieldData_Cpu_Gen(code, data, tab, field, op_input_fields[field], qf_input_fields[field], true));
  }
  code << tab << "\n";
  code << tab << "// -- Output Fields\n";
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCallBackend(CeedOperatorBuildKernelFieldData_Cpu_Gen(code, data, tab, i, op_output_fields[i], qf_output_fields[i], false));
  }
  code << "\n";

  // Compute minimum buffer space needed
  CeedInt max_rstr_buffer_size = 1;

  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedEvalMode eval_mode;

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    if (eval_mode != CEED_EVAL_NONE && eval_mode != CEED_EVAL_WEIGHT) {
      CeedInt             num_comp, elem_size;
      CeedElemRestriction elem_rstr;

      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &elem_rstr));
      CeedCallBackend(CeedElemRestrictionGetNumComponents(elem_rstr, &num_comp));
      CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
      max_rstr_buffer_size = CeedIntMax(max_rstr_buffer_size, num_comp * elem_size);
      CeedCallBackend(CeedElemRestrictionDestroy(&elem_rstr));
    }
  }
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedEvalMode eval_mode;

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    if (eval_mode != CEED_EVAL_NONE) {
      CeedInt             num_comp, elem_size;
      CeedElemRestriction elem_rstr;

      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[i], &elem_rstr));
      CeedCallBackend(CeedElemRestrictionGetNumComponents(elem_rstr, &num_comp));
      CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size);
                      max_rstr_buffer_size = CeedIntMax(max_rstr_buffer_size, num_comp * elem_size));
      CeedCallBackend(CeedElemRestrictionDestroy(&elem_rstr));
    }
  }
  code << tab << "// Scratch restriction buffer space\n";
  code << tab << "constexpr CeedInt max_e_vec_buffer_size = " << max_rstr_buffer_size << ";\n";
  code << tab << "CeedScalar e_vec_scratch[max_e_vec_buffer_size * block_size];\n\n";

  // Loop over blocks
  code << tab << "// Loop over blocks\n";
  code << tab << "for (CeedInt block = 0; block < num_blocks; block++) {\n";
  tab.push();

  // AtPoints data
  if (is_at_points) {
  }

  // Apply ElemRestrictions
  code << tab << "// -- Input ElemRestrictions and Bases\n";
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedInt field = input_field_order[i];

    CeedCallBackend(CeedOperatorBuildKernelRestriction_Cpu_Gen(code, data, tab, field, field_rstr_in_buffer, op_input_fields[field],
                                                               qf_input_fields[field], true, block_size));
    CeedCallBackend(CeedOperatorBuildKernelBasis_Cpu_Gen(code, data, tab, field, op_input_fields[field], qf_input_fields[field], true, is_at_points));
  }
  code << tab << "\n";

  // Apply QFunction
  code << tab << "// -- QFunction\n";
  CeedCallBackend(CeedOperatorBuildKernelQFunction_Cpu_Gen(code, data, tab, num_input_fields, op_input_fields, qf_input_fields, num_output_fields,
                                                           op_output_fields, qf_output_fields, qfunction_name, is_at_points));

  // Apply ElemRestrictions Transpose
  code << tab << "// -- Output Bases and ElemRestrictions\n";
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCallBackend(CeedOperatorBuildKernelBasis_Cpu_Gen(code, data, tab, i, op_output_fields[i], qf_output_fields[i], false, is_at_points));
    CeedCallBackend(CeedOperatorBuildKernelRestriction_Cpu_Gen(code, data, tab, i, NULL, op_output_fields[i], qf_output_fields[i], false,
                                                               block_size));
  }

  // Close loop over blocks
  tab.pop();
  code << tab << "}\n";

  // Return
  code << tab << "return CEED_ERROR_SUCCESS;\n";

  // Close function body
  tab.pop();
  code << tab << "}\n\n";

  // Compile
  {
    bool is_compile_good = false;

    CeedCallBackend(CeedTryCompile_Cpu(ceed, code.str().c_str(), operator_name.c_str(), &is_compile_good, &data->handle, 0));
    if (is_compile_good) {
      *is_good_build = true;
      CeedCallBackend(CeedStringAllocCopy(operator_name.c_str(), &data->op_function_name));
    } else {
      *is_good_build     = false;
      data->use_fallback = true;
    }
  }

  // Cleanup
  CeedCallBackend(CeedQFunctionDestroy(&qf));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}
