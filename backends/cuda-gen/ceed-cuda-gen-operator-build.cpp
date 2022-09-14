// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#define CEED_DEBUG_COLOR 12

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include "ceed-cuda-gen.h"
#include "../cuda/ceed-cuda-compile.h"
#include "../cuda-ref/ceed-cuda-ref.h"
#include "../cuda-shared/ceed-cuda-shared.h"

//------------------------------------------------------------------------------
// Build singe operator kernel
//------------------------------------------------------------------------------
extern "C" int CeedCudaGenOperatorBuild(CeedOperator op) {

  using std::ostringstream;
  using std::string;
  int ierr;
  bool setupdone;
  ierr = CeedOperatorIsSetupDone(op, &setupdone); CeedChkBackend(ierr);
  if (setupdone) return CEED_ERROR_SUCCESS;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  CeedOperator_Cuda_gen *data;
  ierr = CeedOperatorGetData(op, &data); CeedChkBackend(ierr);
  CeedQFunction qf;
  CeedQFunction_Cuda_gen *qf_data;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChkBackend(ierr);
  ierr = CeedQFunctionGetData(qf, &qf_data); CeedChkBackend(ierr);
  CeedSize lsize;
  CeedInt Q, P_1d = 0, Q_1d = 0, numelements, elem_size, numinputfields,
          numoutputfields, num_comp, dim = 1;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChkBackend(ierr);
  Q_1d = Q;
  ierr = CeedOperatorGetNumElements(op, &numelements); CeedChkBackend(ierr);
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &numinputfields, &opinputfields, &numoutputfields, &opoutputfields);
  CeedChkBackend(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, NULL, &qfinputfields, NULL, &qfoutputfields);
  CeedChkBackend(ierr);
  CeedEvalMode emode;
  CeedBasis basis;
  CeedBasis_Cuda_shared *basis_data;
  CeedElemRestriction Erestrict;
  CeedElemRestriction_Cuda *restr_data;

  // TODO: put in a function?
  // Check for restriction only identity operator
  bool is_identity_qf;
  ierr = CeedQFunctionIsIdentity(qf, &is_identity_qf); CeedChkBackend(ierr);
  if (is_identity_qf) {
    CeedEvalMode emodein, emodeout;
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[0], &emodein);  CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[0], &emodeout);  CeedChkBackend(ierr);
    if (emodein == CEED_EVAL_NONE && emodeout == CEED_EVAL_NONE)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "Backend does not implement restriction only identity operators");
    // LCOV_EXCL_STOP
  }

  ostringstream code;

  // TODO: put in a function?
  // Add atomicAdd function for old NVidia architectures
  struct cudaDeviceProp prop;
  Ceed_Cuda *ceed_data;
  ierr = CeedGetData(ceed, &ceed_data); CeedChkBackend(ierr); CeedChkBackend(ierr);
  ierr = cudaGetDeviceProperties(&prop, ceed_data->device_id); CeedChkBackend(ierr);
  if ((prop.major < 6) && (CEED_SCALAR_TYPE != CEED_SCALAR_FP32)){
    char *atomic_add_path, *atomic_add_source;
    ierr = CeedGetJitAbsolutePath(ceed,
                                  "ceed/jit-source/cuda/cuda-atomic-add-fallback.h",
                                  &atomic_add_path); CeedChkBackend(ierr);
    CeedDebug256(ceed, 2, "----- Loading Atomic Add Source -----\n");
    ierr = CeedLoadSourceToBuffer(ceed, atomic_add_path, &atomic_add_source);
    CeedChkBackend(ierr);
    code << atomic_add_source;
    ierr = CeedFree(&atomic_add_path); CeedChkBackend(ierr);
    ierr = CeedFree(&atomic_add_source); CeedChkBackend(ierr);
  }

  // TODO: generalize to accept different device functions?
  {
    char *tensor_basis_kernel_path, *tensor_basis_kernel_source;
    ierr = CeedGetJitAbsolutePath(ceed,
                                  "ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h",
                                  &tensor_basis_kernel_path); CeedChkBackend(ierr);
    CeedDebug256(ceed, 2, "----- Loading Tensor Basis Kernel Source -----\n");
    ierr = CeedLoadSourceToBuffer(ceed, tensor_basis_kernel_path, &tensor_basis_kernel_source);
    CeedChkBackend(ierr);
    code << tensor_basis_kernel_source;
    ierr = CeedFree(&tensor_basis_kernel_path); CeedChkBackend(ierr);
    ierr = CeedFree(&tensor_basis_kernel_source); CeedChkBackend(ierr);
  }
  {
    char *cuda_gen_template_path, *cuda_gen_template_source;
    ierr = CeedGetJitAbsolutePath(ceed,
                                  "ceed/jit-source/cuda/cuda-gen-templates.h",
                                  &cuda_gen_template_path); CeedChkBackend(ierr);
    CeedDebug256(ceed, 2, "----- Loading Cuda-Gen Template Source -----\n");
    ierr = CeedLoadSourceToBuffer(ceed, cuda_gen_template_path, &cuda_gen_template_source);
    CeedChkBackend(ierr);
    code << cuda_gen_template_source;
    ierr = CeedFree(&cuda_gen_template_path); CeedChkBackend(ierr);
    ierr = CeedFree(&cuda_gen_template_source); CeedChkBackend(ierr);
  }

  string qFunction(qf_data->qFunctionSource);
  string qFunctionName(qf_data->qFunctionName);
  string oper;
  oper = "CeedKernel_Cuda_gen_" + qFunctionName;

  // TODO: put in a function?
  // Find dim and Q_1d
  bool useCollograd = false;
  // Only use collocated gradient algorithm when we actually compute a gradient.
  if ( dim == 3 ) {
    for (CeedInt i = 0; i < numinputfields; i++) {
      ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
      if (emode == CEED_EVAL_GRAD) {
        useCollograd = true;
      }
    }
    for (CeedInt i = 0; i < numoutputfields; i++) {
      ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
      if (emode == CEED_EVAL_GRAD) {
        useCollograd = true;
      }
    }
  }
  data->maxP1d = 0;
  for (CeedInt i = 0; i < numinputfields; i++) {
    ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChkBackend(ierr);
    if (basis != CEED_BASIS_COLLOCATED) {
      ierr = CeedBasisGetData(basis, &basis_data); CeedChkBackend(ierr);
      ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
      CeedChkBackend(ierr);

      // Check for collocated gradient
      useCollograd = useCollograd && basis_data->d_collo_grad_1d; 

      // Collect dim and Q_1d
      ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
      bool isTensor;
      ierr = CeedBasisIsTensor(basis, &isTensor); CeedChkBackend(ierr); 
      if (isTensor) {
        ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d); CeedChkBackend(ierr);
        ierr = CeedBasisGetNumNodes1D(basis, &P_1d); CeedChkBackend(ierr);
        if (P_1d>data->maxP1d) data->maxP1d = P_1d;
      } else {
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_BACKEND, "Backend does not implement operators with non-tensor basis");
        // LCOV_EXCL_STOP
      }
    }
  }
  // Check output bases for Q_1d, dim as well
  //   The only input basis might be CEED_BASIS_COLLOCATED
  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis); CeedChkBackend(ierr);

    if (basis != CEED_BASIS_COLLOCATED) {
      ierr = CeedBasisGetData(basis, &basis_data); CeedChkBackend(ierr);
      ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
      CeedChkBackend(ierr);

      // Collect dim and Q_1d
      ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
      bool isTensor;
      ierr = CeedBasisIsTensor(basis, &isTensor); CeedChkBackend(ierr); 
      if (isTensor) {
        ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d); CeedChkBackend(ierr);
      } else {
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_BACKEND, "Backend does not implement operators with non-tensor basis");
        // LCOV_EXCL_STOP
      }

      // Check for collocated gradient
      useCollograd = useCollograd && basis_data->d_collo_grad_1d; 
    }
  }
  data->dim = dim;
  data->Q1d = Q_1d;

  // Define CEED_Q_VLA
  code << "\n#undef CEED_Q_VLA\n";
  if (dim != 3 || useCollograd) {
    code << "#define CEED_Q_VLA 1\n\n";
  } else {
    code << "#define CEED_Q_VLA "<<Q_1d<<"\n\n";
  }

  code << qFunction;

  // Setup
  code << "\n// -----------------------------------------------------------------------------\n";
  code << "\nextern \"C\" __global__ void "<<oper<<"(CeedInt num_elem, void* ctx, FieldsInt_Cuda indices, Fields_Cuda fields, Fields_Cuda B, Fields_Cuda G, CeedScalar* W) {\n";
  for (CeedInt i = 0; i < numinputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChkBackend(ierr);
    if (emode != CEED_EVAL_WEIGHT) { // Skip CEED_EVAL_WEIGHT
      code << "  const CeedScalar* d_u" <<i<<" = fields.inputs["<<i<<"];\n";
    }
  }

  for (CeedInt i = 0; i < numoutputfields; i++) {
    code << "  CeedScalar* d_v"<<i<<" = fields.outputs["<<i<<"];\n";
  }

  code << "  const CeedInt dim = "<<dim<<";\n";
  code << "  const CeedInt Q_1d = "<<Q_1d<<";\n";

  code << "  extern __shared__ CeedScalar slice[];\n";
  // TODO put in a function? InitSharedData_Cuda?
  code << "  SharedData_Cuda data;\n";
  code << "  data.t_id_x = threadIdx.x;\n";
  code << "  data.t_id_y = threadIdx.y;\n";
  code << "  data.t_id_z = threadIdx.z;\n";
  code << "  data.t_id  = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y*blockDim.x;\n";
  code << "  data.slice = slice+data.t_id_z*T_1D"<<(dim>1?"*T_1D":"")<<";\n";

  code << "\n  // -- Input field constants and basis data --\n";
  // TODO: Put in a function?
  //Initialize constants, and matrices B and G
  for (CeedInt i = 0; i < numinputfields; i++) {
    code << "  // ---- Input field "<<i<<" ----\n";
    // Get elem_size, emode, num_comp
    ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetElementSize(Erestrict, &elem_size);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetNumComponents(Erestrict, &num_comp);
    CeedChkBackend(ierr);

    // Set field constants
    if (emode != CEED_EVAL_WEIGHT) {
      ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChkBackend(ierr);
      if (basis != CEED_BASIS_COLLOCATED) {
        ierr = CeedBasisGetNumNodes1D(basis, &P_1d); CeedChkBackend(ierr);
        code << "  const CeedInt P_in_"<<i<<" = "<<P_1d<<";\n";
      } else {
        code << "  const CeedInt P_in_"<<i<<" = "<<Q_1d<<";\n";
      }
      code << "  const CeedInt num_comp_in_"<<i<<" = "<<num_comp<<";\n";
    }

    // Load basis data
    code << "  // EvalMode: "<<CeedEvalModes[emode]<<"\n";
    switch (emode) {
    case CEED_EVAL_NONE:
      break;
    case CEED_EVAL_INTERP:
      ierr = CeedBasisGetData(basis, &basis_data); CeedChkBackend(ierr);
      data->B.in[i] = basis_data->d_interp_1d;
      code << "  __shared__ CeedScalar s_B_in_"<<i<<"["<<P_1d*Q_1d<<"];\n";
      code << "  loadMatrix<P_in_"<<i<<",Q_1d>(data, B.inputs["<<i<<"], s_B_in_"<<i<<");\n";
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedBasisGetData(basis, &basis_data); CeedChkBackend(ierr);
      data->B.in[i] = basis_data->d_interp_1d;
      code << "  __shared__ CeedScalar s_B_in_"<<i<<"["<<P_1d*Q_1d<<"];\n";
      code << "  loadMatrix<P_in_"<<i<<",Q_1d>(data, B.inputs["<<i<<"], s_B_in_"<<i<<");\n";
      if (useCollograd) {
        data->G.in[i] = basis_data->d_collo_grad_1d;
        code << "  __shared__ CeedScalar s_G_in_"<<i<<"["<<Q_1d*Q_1d<<"];\n";
        code << "  loadMatrix<Q_1d,Q_1d>(data, G.inputs["<<i<<"], s_G_in_"<<i<<");\n";
      } else {
        data->G.in[i] = basis_data->d_grad_1d;
        code << "  __shared__ CeedScalar s_G_in_"<<i<<"["<<P_1d*Q_1d<<"];\n";
        code << "  loadMatrix<P_in_"<<i<<",Q_1d>(data, G.inputs["<<i<<"], s_G_in_"<<i<<");\n";
      }
      break;
    case CEED_EVAL_WEIGHT:
      break; // No action
    case CEED_EVAL_DIV:
      break; // TODO: Not implemented
    case CEED_EVAL_CURL:
      break; // TODO: Not implemented
    }
  }

  code << "\n  // -- Output field constants and basis data --\n";
  for (CeedInt i = 0; i < numoutputfields; i++) {
    code << "  // ---- Output field "<<i<<" ----\n";
    // Get elem_size, emode, num_comp
    ierr = CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetElementSize(Erestrict, &elem_size);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetNumComponents(Erestrict, &num_comp);
    CeedChkBackend(ierr);

    // Set field constants
    ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis); CeedChkBackend(ierr);
    if (basis != CEED_BASIS_COLLOCATED) {
      ierr = CeedBasisGetNumNodes1D(basis, &P_1d); CeedChkBackend(ierr);
      code << "  const CeedInt P_out_"<<i<<" = "<<P_1d<<";\n";
    } else {
      code << "  const CeedInt P_out_"<<i<<" = "<<Q_1d<<";\n";
    }
    code << "  const CeedInt num_comp_out_"<<i<<" = "<<num_comp<<";\n";

    // Load basis data
    code << "  // EvalMode: "<<CeedEvalModes[emode]<<"\n";
    switch (emode) {
    case CEED_EVAL_NONE:
      break; // No action
    case CEED_EVAL_INTERP:
      ierr = CeedBasisGetData(basis, &basis_data); CeedChkBackend(ierr);
      data->B.out[i] = basis_data->d_interp_1d;
      code << "  __shared__ CeedScalar s_B_out_"<<i<<"["<<P_1d*Q_1d<<"];\n";
      code << "  loadMatrix<P_out_"<<i<<",Q_1d>(data, B.outputs["<<i<<"], s_B_out_"<<i<<");\n";
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedBasisGetData(basis, &basis_data); CeedChkBackend(ierr);
      data->B.out[i] = basis_data->d_interp_1d;
      code << "  __shared__ CeedScalar s_B_out_"<<i<<"["<<P_1d*Q_1d<<"];\n";
      code << "  loadMatrix<P_out_"<<i<<",Q_1d>(data, B.outputs["<<i<<"], s_B_out_"<<i<<");\n";
      if (useCollograd) {
        data->G.out[i] = basis_data->d_collo_grad_1d;
        code << "  __shared__ CeedScalar s_G_out_"<<i<<"["<<Q_1d*Q_1d<<"];\n";
        code << "  loadMatrix<Q_1d,Q_1d>(data, G.outputs["<<i<<"], s_G_out_"<<i<<");\n";
      } else {
        data->G.out[i] = basis_data->d_grad_1d;
        code << "  __shared__ CeedScalar s_G_out_"<<i<<"["<<P_1d*Q_1d<<"];\n";
        code << "  loadMatrix<P_out_"<<i<<",Q_1d>(data, G.outputs["<<i<<"], s_G_out_"<<i<<");\n";
      }
      break;
    // LCOV_EXCL_START
    case CEED_EVAL_WEIGHT: {
      Ceed ceed;
      ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
      break; // Should not occur
    }
    case CEED_EVAL_DIV:
      break; // TODO: Not implemented
    case CEED_EVAL_CURL:
      break; // TODO: Not implemented
      // LCOV_EXCL_STOP
    }
  }
  code << "\n  // -- Element loop --\n";
  code << "  __syncthreads();\n";
  code << "  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x*blockDim.z) {\n";
  // Input basis apply if needed
  // Generate the correct eval mode code for each input
  code << "    // -- Input field restrictions and basis actions --\n";
  for (CeedInt i = 0; i < numinputfields; i++) {
    code << "    // ---- Input field "<<i<<" ----\n";
    // Get elem_size, emode, num_comp
    ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetElementSize(Erestrict, &elem_size);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetNumComponents(Erestrict, &num_comp);
    CeedChkBackend(ierr);

    // TODO: put in a function?
    // Restriction
    if (emode != CEED_EVAL_WEIGHT &&
        !((emode == CEED_EVAL_NONE) && useCollograd)) {
      code << "    CeedScalar r_u_"<<i<<"[num_comp_in_"<<i<<"*P_in_"<<i<<"];\n";
      
      bool isStrided;
      ierr = CeedElemRestrictionIsStrided(Erestrict, &isStrided); CeedChkBackend(ierr);
      if (!isStrided) {
        ierr = CeedElemRestrictionGetLVectorSize(Erestrict, &lsize);
        CeedChkBackend(ierr);
        code << "    const CeedInt lsize_in_"<<i<<" = "<<lsize<<";\n";
        CeedInt compstride;
        ierr = CeedElemRestrictionGetCompStride(Erestrict, &compstride); CeedChkBackend(ierr);
        code << "    // CompStride: "<<compstride<<"\n";
        ierr = CeedElemRestrictionGetData(Erestrict, &restr_data); CeedChkBackend(ierr);
        data->indices.in[i] = restr_data->d_ind;
        code << "    readDofsOffset"<<dim<<"d<num_comp_in_"<<i<<", "<<compstride<<", P_in_"<<i<<">(data, lsize_in_"<<i<<", elem, indices.inputs["<<i<<"], d_u"<<i<<", r_u_"<<i<<");\n";
      } else {
        bool backendstrides;
        ierr = CeedElemRestrictionHasBackendStrides(Erestrict, &backendstrides);
        CeedChkBackend(ierr);
        CeedInt num_elem;
        ierr = CeedElemRestrictionGetNumElements(Erestrict, &num_elem);
        CeedChkBackend(ierr);
        CeedInt strides[3] = {1, elem_size*num_elem, elem_size};
        if (!backendstrides) {
          ierr = CeedElemRestrictionGetStrides(Erestrict, &strides);
          CeedChkBackend(ierr);
        }
        code << "    // Strides: {"<<strides[0]<<", "<<strides[1]<<", "<<strides[2]<<"}\n";
        code << "    readDofsStrided"<<dim<<"d<num_comp_in_"<<i<<",P_in_"<<i<<","<<strides[0]<<","<<strides[1]<<","<<strides[2]<<">(data, elem, d_u"<<i<<", r_u_"<<i<<");\n";
      }
    }

    // TODO: put in a function?
    // Basis action
    code << "    // EvalMode: "<<CeedEvalModes[emode]<<"\n";
    switch (emode) {
    case CEED_EVAL_NONE:
      if (!useCollograd) {
        code << "    CeedScalar* r_t"<<i<<" = r_u_"<<i<<";\n";
      }
      break;
    case CEED_EVAL_INTERP:
      code << "    CeedScalar r_t"<<i<<"[num_comp_in_"<<i<<"*Q_1d];\n";
      code << "    Interp"<<(dim>1?"Tensor":"")<<dim<<"d<num_comp_in_"<<i<<",P_in_"<<i<<",Q_1d>(data, r_u_"<<i<<", s_B_in_"<<i<<", r_t"<<i<<");\n";
      break;
    case CEED_EVAL_GRAD:
      if (useCollograd) {
        code << "    CeedScalar r_t"<<i<<"[num_comp_in_"<<i<<"*Q_1d];\n";
        code << "    Interp"<<(dim>1?"Tensor":"")<<dim<<"d<num_comp_in_"<<i<<",P_in_"<<i<<",Q_1d>(data, r_u_"<<i<<", s_B_in_"<<i<<", r_t"<<i<<");\n";
      } else {
        code << "    CeedScalar r_t"<<i<<"[num_comp_in_"<<i<<"*dim*Q_1d];\n";
        code << "    Grad"<<(dim>1?"Tensor":"")<<dim<<"d<num_comp_in_"<<i<<",P_in_"<<i<<",Q_1d>(data, r_u_"<<i<<", s_B_in_"<<i<<", s_G_in_"<<i<<", r_t"<<i<<");\n";
      }
      break;
    case CEED_EVAL_WEIGHT:
      code << "    CeedScalar r_t"<<i<<"[Q_1d];\n";
      ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChkBackend(ierr);
      ierr = CeedBasisGetData(basis, &basis_data); CeedChkBackend(ierr);
      data->W = basis_data->d_q_weight_1d;
      code << "    Weight"<<(dim>1?"Tensor":"")<<dim<<"d<Q_1d>(data, W, r_t"<<i<<");\n";
      break; // No action
    case CEED_EVAL_DIV:
      break; // TODO: Not implemented
    case CEED_EVAL_CURL:
      break; // TODO: Not implemented
    }
  }

  // TODO: put in a function + separate colograd logic
  // Q function
  code << "\n    // -- Output field setup --\n";
  for (CeedInt i = 0; i < numoutputfields; i++) {
      code << "\n    // ---- Output field "<<i<<" ----\n";
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChkBackend(ierr);
    if (emode==CEED_EVAL_GRAD)
    {
      if (useCollograd) {
        //Accumulator for gradient slices
        code << "    CeedScalar r_tt_"<<i<<"[num_comp_out_"<<i<<"*Q_1d];\n";
        code << "    for (CeedInt i = 0; i < num_comp_out_"<<i<<"; i++) {\n";
        code << "      for (CeedInt j = 0; j < Q_1d; ++j) {\n";
        code << "        r_tt_"<<i<<"[j + i*Q_1d] = 0.0;\n";
        code << "      }\n";
        code << "    }\n";
      } else {
        code << "    CeedScalar r_tt_"<<i<<"[num_comp_out_"<<i<<"*dim*Q_1d];\n";
      }
    }
    if (emode==CEED_EVAL_NONE || emode==CEED_EVAL_INTERP)
    {
      code << "    CeedScalar r_tt_"<<i<<"[num_comp_out_"<<i<<"*Q_1d];\n";
    }
  }
  // We treat quadrature points per slice in 3d to save registers
  if (useCollograd) {
    code << "\n    // Note: Collocated Gradient\n";
    code << "#pragma unroll\n";
    code << "    for (CeedInt q = 0; q < Q_1d; q++) {\n";
    code << "      // -- Input fields --\n";
    for (CeedInt i = 0; i < numinputfields; i++) {
      code << "      // ---- Input field "<<i<<" ----\n";
      // Get elem_size, emode, num_comp
      ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
      CeedChkBackend(ierr);
      // Basis action
      code << "      // EvalMode: "<<CeedEvalModes[emode]<<"\n";
      switch (emode) {
      case CEED_EVAL_NONE:
        code << "      CeedScalar r_q_"<<i<<"[num_comp_in_"<<i<<"];\n";

        bool isStrided;
        ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict); CeedChkBackend(ierr);
        ierr = CeedElemRestrictionIsStrided(Erestrict, &isStrided); CeedChkBackend(ierr);
        if (!isStrided) {
          ierr = CeedElemRestrictionGetLVectorSize(Erestrict, &lsize);
          CeedChkBackend(ierr);
          code << "      const CeedInt lsize_in_"<<i<<" = "<<lsize<<";\n";
          CeedInt compstride;
          ierr = CeedElemRestrictionGetCompStride(Erestrict, &compstride); CeedChkBackend(ierr);
          code << "      // CompStride: "<<compstride<<"\n";
          ierr = CeedElemRestrictionGetData(Erestrict, &restr_data); CeedChkBackend(ierr);
          data->indices.in[i] = restr_data->d_ind;
          code << "      readSliceQuadsOffset"<<"3d<num_comp_in_"<<i<<", "<<compstride<<", Q_1d>(data, lsize_in_"<<i<<", elem, q, indices.inputs["<<i<<"], d_u"<<i<<", r_q_"<<i<<");\n";
        } else {
          ierr = CeedElemRestrictionGetElementSize(Erestrict, &elem_size); CeedChkBackend(ierr);
          bool backendstrides;
          ierr = CeedElemRestrictionHasBackendStrides(Erestrict, &backendstrides);
          CeedChkBackend(ierr);
          CeedInt num_elem;
          ierr = CeedElemRestrictionGetNumElements(Erestrict, &num_elem);
          CeedChkBackend(ierr);
          CeedInt strides[3] = {1, elem_size*num_elem, elem_size};
          if (!backendstrides) {
            ierr = CeedElemRestrictionGetStrides(Erestrict, &strides);
            CeedChkBackend(ierr);
          }
          code << "      // Strides: {"<<strides[0]<<", "<<strides[1]<<", "<<strides[2]<<"}\n";
          code << "      readSliceQuadsStrided"<<"3d<num_comp_in_"<<i<<",Q_1d"","<<strides[0]<<","<<strides[1]<<","<<strides[2]<<">(data, elem, q, d_u"<<i<<", r_q_"<<i<<");\n";
        }
        break;
      case CEED_EVAL_INTERP:
        code << "      CeedScalar r_q_"<<i<<"[num_comp_in_"<<i<<"];\n";
        code << "      for (CeedInt j = 0; j < num_comp_in_"<<i<<" ; ++j) {\n";
        code << "        r_q_"<<i<<"[j] = r_t"<<i<<"[q + j*Q_1d];\n";
        code << "      }\n";
        break;
      case CEED_EVAL_GRAD:
        code << "      CeedScalar r_q_"<<i<<"[num_comp_in_"<<i<<"*dim];\n";
        code << "      gradCollo3d<num_comp_in_"<<i<<",Q_1d>(data, q, r_t"<<i<<", s_G_in_"<<i<<", r_q_"<<i<<");\n";
        break;
      case CEED_EVAL_WEIGHT:
        code << "      CeedScalar r_q_"<<i<<"[1];\n";
        code << "      r_q_"<<i<<"[0] = r_t"<<i<<"[q];\n";
        break; // No action
      case CEED_EVAL_DIV:
        break; // TODO: Not implemented
      case CEED_EVAL_CURL:
        break; // TODO: Not implemented
      }
    }
    code << "\n      // -- Output fields --\n";
    for (CeedInt i = 0; i < numoutputfields; i++) {
      code << "      // ---- Output field "<<i<<" ----\n";
      ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
      CeedChkBackend(ierr);
      // Basis action
      switch (emode) {
      case CEED_EVAL_NONE:
        code << "      CeedScalar r_qq_"<<i<<"[num_comp_out_"<<i<<"];\n";
        break; // No action
      case CEED_EVAL_INTERP:
        code << "      CeedScalar r_qq_"<<i<<"[num_comp_out_"<<i<<"];\n";
        break;
      case CEED_EVAL_GRAD:
        code << "      CeedScalar r_qq_"<<i<<"[num_comp_out_"<<i<<"*dim];\n";
        break;
      case CEED_EVAL_WEIGHT:
        break; // Should not occur
      case CEED_EVAL_DIV:
        break; // TODO: Not implemented
      case CEED_EVAL_CURL:
        break; // TODO: Not implemented
      }
    }
  } else {
    code << "\n      // Note: No Collocated Gradient\n";
    code << "      // -- Input fields --\n";
    for (CeedInt i = 0; i < numinputfields; i++) {
      code << "      // ---- Input field "<<i<<" ----\n";
      code << "      CeedScalar* r_q_"<<i<<" = r_t"<<i<<";\n";
    }
    code << "      // -- Output fields --\n";
    for (CeedInt i = 0; i < numoutputfields; i++) {
      code << "      // ---- Output field "<<i<<" ----\n";
      code << "      CeedScalar* r_qq_"<<i<<" = r_tt_"<<i<<";\n";
    }
  }
  code << "\n      // -- QFunction Inputs and outputs --\n";
  code << "      CeedScalar* in["<<numinputfields<<"];\n";
  for (CeedInt i = 0; i < numinputfields; i++) {
    code << "      // ---- Input field "<<i<<" ----\n";
    code << "      in["<<i<<"] = r_q_"<<i<<";\n";
  }
  code << "      CeedScalar* out["<<numoutputfields<<"];\n";
  for (CeedInt i = 0; i < numoutputfields; i++) {
    code << "      // ---- Output field "<<i<<" ----\n";
    code << "      out["<<i<<"] = r_qq_"<<i<<";\n";
  }
  code << "\n      // -- Apply QFunction --\n";
  code << "      "<<qFunctionName<<"(ctx, ";
  if (dim != 3 || useCollograd) {
    code << "1";
  } else {
    code << "Q_1d";
  }
  code << ", in, out);\n";
  if (useCollograd) {
    code << "\n      // Note: Collocated Gradient\n";
    code << "      // -- Output fields --\n";
    for (CeedInt i = 0; i < numoutputfields; i++) {
      code << "      // ---- Output field "<<i<<" ----\n";
      ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
      CeedChkBackend(ierr);
      // Basis action
      code << "      // EvalMode: "<<CeedEvalModes[emode]<<"\n";
      switch (emode) {
      case CEED_EVAL_NONE:
        code << "      for (CeedInt j = 0; j < num_comp_out_"<<i<<" ; ++j) {\n";
        code << "        r_tt_"<<i<<"[q + j*Q_1d] = r_qq_"<<i<<"[j];\n";
        code << "      }\n";
        break; // No action
      case CEED_EVAL_INTERP:
        code << "      for (CeedInt j = 0; j < num_comp_out_"<<i<<" ; ++j) {\n";
        code << "        r_tt_"<<i<<"[q + j*Q_1d] = r_qq_"<<i<<"[j];\n";
        code << "      }\n";
        break;
      case CEED_EVAL_GRAD:
        code << "      gradColloTranspose3d<num_comp_out_"<<i<<",Q_1d>(data, q, r_qq_"<<i<<", s_G_out_"<<i<<", r_tt_"<<i<<");\n";
        break;
      case CEED_EVAL_WEIGHT:
        break; // Should not occur
      case CEED_EVAL_DIV:
        break; // TODO: Not implemented
      case CEED_EVAL_CURL:
        break; // TODO: Not implemented
      }
    }
    code << "    }\n";
  }

  // Output basis apply if needed
  // Generate the correct eval mode code for each output
  code << "\n    // -- Output field basis action and restrictions --\n";
  for (CeedInt i = 0; i < numoutputfields; i++) {
    code << "    // ---- Output field "<<i<<" ----\n";
    // Get elem_size, emode, num_comp
    ierr = CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetElementSize(Erestrict, &elem_size);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetNumComponents(Erestrict, &num_comp);
    CeedChkBackend(ierr);
    // TODO put in a function
    // Basis action
    code << "    // EvalMode: "<<CeedEvalModes[emode]<<"\n";
    switch (emode) {
    case CEED_EVAL_NONE:
      code << "    CeedScalar* r_v_"<<i<<" = r_tt_"<<i<<";\n";
      break; // No action
    case CEED_EVAL_INTERP:
      code << "    CeedScalar r_v_"<<i<<"[num_comp_out_"<<i<<"*P_out_"<<i<<"];\n";
      code << "    InterpTranspose"<<(dim>1?"Tensor":"")<<dim<<"d<num_comp_out_"<<i<<",P_out_"<<i<<",Q_1d>(data, r_tt_"<<i<<", s_B_out_"<<i<<", r_v_"<<i<<");\n";
      break;
    case CEED_EVAL_GRAD:
      code << "    CeedScalar r_v_"<<i<<"[num_comp_out_"<<i<<"*P_out_"<<i<<"];\n";
      if (useCollograd) {
        code << "    InterpTranspose"<<(dim>1?"Tensor":"")<<dim<<"d<num_comp_out_"<<i<<",P_out_"<<i<<",Q_1d>(data, r_tt_"<<i<<", s_B_out_"<<i<<", r_v_"<<i<<");\n";
      } else {
        code << "    GradTranspose"<<(dim>1?"Tensor":"")<<dim<<"d<num_comp_out_"<<i<<",P_out_"<<i<<",Q_1d>(data, r_tt_"<<i<<", s_B_out_"<<i<<", s_G_out_"<<i<<", r_v_"<<i<<");\n";
      }
      break;
    // LCOV_EXCL_START
    case CEED_EVAL_WEIGHT: {
      Ceed ceed;
      ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
      break; // Should not occur
    }
    case CEED_EVAL_DIV:
      break; // TODO: Not implemented
    case CEED_EVAL_CURL:
      break; // TODO: Not implemented
      // LCOV_EXCL_STOP
    }
    // TODO put in a function
    // Restriction
      bool isStrided;
      ierr = CeedElemRestrictionIsStrided(Erestrict, &isStrided); CeedChkBackend(ierr);
    if (!isStrided) {
      ierr = CeedElemRestrictionGetLVectorSize(Erestrict, &lsize);
      CeedChkBackend(ierr);
      code << "    const CeedInt lsize_out_"<<i<<" = "<<lsize<<";\n";
      CeedInt compstride;
      ierr = CeedElemRestrictionGetCompStride(Erestrict, &compstride); CeedChkBackend(ierr);
      code << "    // CompStride: "<<compstride<<"\n";
      ierr = CeedElemRestrictionGetData(Erestrict, &restr_data); CeedChkBackend(ierr);
      data->indices.out[i] = restr_data->d_ind;
      code << "    writeDofsOffset"<<dim<<"d<num_comp_out_"<<i<<", "<<compstride<<", P_out_"<<i<<">(data, lsize_out_"<<i<<", elem, indices.outputs["<<i<<"], r_v_"<<i<<", d_v"<<i<<");\n";
    } else {
      bool has_backend_strides;
      ierr = CeedElemRestrictionHasBackendStrides(Erestrict, &has_backend_strides);
      CeedChkBackend(ierr);
      CeedInt num_elem;
      ierr = CeedElemRestrictionGetNumElements(Erestrict, &num_elem);
      CeedChkBackend(ierr);
      CeedInt strides[3] = {1, elem_size*num_elem, elem_size};
      if (!has_backend_strides) {
        ierr = CeedElemRestrictionGetStrides(Erestrict, &strides);
        CeedChkBackend(ierr);
      }
      code << "    // Strides: {"<<strides[0]<<", "<<strides[1]<<", "<<strides[2]<<"}\n";
      code << "    writeDofsStrided"<<dim<<"d<num_comp_out_"<<i<<",P_out_"<<i<<","<<strides[0]<<","<<strides[1]<<","<<strides[2]<<">(data, elem, r_v_"<<i<<", d_v"<<i<<");\n";
    }
  }

  code << "  }\n";
  code << "}\n";
  code << "// -----------------------------------------------------------------------------\n\n";

  // View kernel for debugging
  CeedDebug256(ceed, 2, "Generated Operator Kernels:\n");
  CeedDebug(ceed, code.str().c_str());

  ierr = CeedCompileCuda(ceed, code.str().c_str(), &data->module, 1,
                         "T_1D", CeedIntMax(Q_1d, data->maxP1d));
  CeedChkBackend(ierr);
  ierr = CeedGetKernelCuda(ceed, data->module, oper.c_str(), &data->op);
  CeedChkBackend(ierr);

  ierr = CeedOperatorSetSetupDone(op); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
