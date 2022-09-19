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
#include <iostream>
#include <string>
#include <sstream>
#include "ceed-hip-gen.h"
#include "../hip-ref/ceed-hip-ref.h"
#include "../hip-shared/ceed-hip-shared.h"
#include "../hip/ceed-hip-compile.h"


//------------------------------------------------------------------------------
// Calculate the block size used for launching the operator kernel 
//------------------------------------------------------------------------------
extern "C" int BlockGridCalculate_Hip_gen(const CeedInt dim, const CeedInt nelem,
     	                                  const CeedInt P1d, const CeedInt Q1d,
				          CeedInt *block_sizes) {
  
  const CeedInt thread1d = CeedIntMax(Q1d, P1d);
  if (dim==1) {
    CeedInt elemsPerBlock = 64*thread1d > 256? 256/thread1d : 64;
    elemsPerBlock = elemsPerBlock>0?elemsPerBlock:1;
    block_sizes[0] = thread1d;
    block_sizes[1] = 1;
    block_sizes[2] = elemsPerBlock;
  } else if (dim==2) {
    const CeedInt elemsPerBlock = thread1d<4? 16 : 2;
    block_sizes[0] = thread1d;
    block_sizes[1] = thread1d;
    block_sizes[2] = elemsPerBlock;
  } else if (dim==3) {
    const CeedInt elemsPerBlock = thread1d<6? 4 : (thread1d<8? 2 : 1);
    block_sizes[0] = thread1d;
    block_sizes[1] = thread1d;
    block_sizes[2] = elemsPerBlock;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Build single operator kernel
//------------------------------------------------------------------------------
extern "C" int CeedHipGenOperatorBuild(CeedOperator op) {

  using std::ostringstream;
  using std::string;
  int ierr;
  bool setupdone;
  ierr = CeedOperatorIsSetupDone(op, &setupdone); CeedChkBackend(ierr);
  if (setupdone) return CEED_ERROR_SUCCESS;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  CeedOperator_Hip_gen *data;
  ierr = CeedOperatorGetData(op, &data); CeedChkBackend(ierr);
  CeedQFunction qf;
  CeedQFunction_Hip_gen *qf_data;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChkBackend(ierr);
  ierr = CeedQFunctionGetData(qf, &qf_data); CeedChkBackend(ierr);
  CeedSize lsize;
  CeedInt Q, P1d = 0, Q1d = 0, numelements, elemsize, numinputfields,
          numoutputfields, ncomp, dim = 1;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChkBackend(ierr);
  Q1d = Q;
  ierr = CeedOperatorGetNumElements(op, &numelements); CeedChkBackend(ierr);
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &numinputfields, &opinputfields, &numoutputfields, &opoutputfields);
  CeedChkBackend(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, NULL, &qfinputfields, NULL, &qfoutputfields);
  CeedChkBackend(ierr);
  CeedEvalMode emode;
  CeedBasis basis;
  CeedBasis_Hip_shared *basis_data;
  CeedElemRestriction Erestrict;
  CeedElemRestriction_Hip *restr_data;

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
  // TODO: generalize to accept different device functions?
  {
    char *tensor_basis_kernel_path, *tensor_basis_kernel_source;
    ierr = CeedGetJitAbsolutePath(ceed,
                                  "ceed/jit-source/hip/hip-shared-basis-tensor-templates.h",
                                  &tensor_basis_kernel_path); CeedChkBackend(ierr);
    CeedDebug256(ceed, 2, "----- Loading Tensor Basis Kernel Source -----\n");
    ierr = CeedLoadSourceToBuffer(ceed, tensor_basis_kernel_path, &tensor_basis_kernel_source);
    CeedChkBackend(ierr);
    code << tensor_basis_kernel_source;
    ierr = CeedFree(&tensor_basis_kernel_path); CeedChkBackend(ierr);
    ierr = CeedFree(&tensor_basis_kernel_source); CeedChkBackend(ierr);
  }
  {
    char *hip_gen_template_path, *hip_gen_template_source;
    ierr = CeedGetJitAbsolutePath(ceed,
                                  "ceed/jit-source/hip/hip-gen-templates.h",
                                  &hip_gen_template_path); CeedChkBackend(ierr);
    CeedDebug256(ceed, 2, "----- Loading Hip-Gen Template Source -----\n");
    ierr = CeedLoadSourceToBuffer(ceed, hip_gen_template_path, &hip_gen_template_source);
    CeedChkBackend(ierr);
    code << hip_gen_template_source;
    ierr = CeedFree(&hip_gen_template_path); CeedChkBackend(ierr);
    ierr = CeedFree(&hip_gen_template_source); CeedChkBackend(ierr);
  }

  string qFunction(qf_data->qFunctionSource);
  string qFunctionName(qf_data->qFunctionName);
  string oper;
  oper = "CeedKernel_Hip_gen_" + qFunctionName;

  // Find dim and Q1d
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

      // Collect dim and Q1d
      ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
      bool isTensor;
      ierr = CeedBasisIsTensor(basis, &isTensor); CeedChkBackend(ierr); 
      if (isTensor) {
        ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChkBackend(ierr);
        ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChkBackend(ierr);
        if (P1d>data->maxP1d) data->maxP1d = P1d;
      } else {
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_BACKEND, "Backend does not implement operators with non-tensor basis");
        // LCOV_EXCL_STOP
      }
    }
  }
  // Check output bases for Q1d, dim as well
  //   The only input basis might be CEED_BASIS_COLLOCATED
  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis); CeedChkBackend(ierr);

    if (basis != CEED_BASIS_COLLOCATED) {
      ierr = CeedBasisGetData(basis, &basis_data); CeedChkBackend(ierr);
      ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
      CeedChkBackend(ierr);

      // Collect dim and Q1d
      ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
      bool isTensor;
      ierr = CeedBasisIsTensor(basis, &isTensor); CeedChkBackend(ierr); 
      if (isTensor) {
        ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChkBackend(ierr);
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
  data->Q1d = Q1d;

  // Define CEED_Q_VLA
  if (dim != 3 || useCollograd) {
    code << "\n#define CEED_Q_VLA 1\n\n";
  } else {
    code << "\n#define CEED_Q_VLA "<<Q1d<<"\n\n";
  }

  code << qFunction;

  // Setup
  code << "\n// -----------------------------------------------------------------------------\n";
  code << "\nextern \"C\" __launch_bounds__(BLOCK_SIZE)\n";
  code << "__global__ void "<<oper<<"(CeedInt nelem, void* ctx, FieldsInt_Hip indices, Fields_Hip fields, Fields_Hip B, Fields_Hip G, CeedScalar* W) {\n";
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

  code << "  const CeedInt Dim = "<<dim<<";\n";
  code << "  const CeedInt Q1d = "<<Q1d<<";\n";

  code << "  HIP_DYNAMIC_SHARED( CeedScalar, slice)\n";
  code << "  SharedData_Hip data;\n";
  code << "  data.t_id_x = threadIdx.x;\n";
  code << "  data.t_id_y = threadIdx.y;\n";
  code << "  data.t_id_z = threadIdx.z;\n";
  code << "  data.t_id  = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y*blockDim.x;\n";
  code << "  data.slice = slice+data.t_id_z*T_1D"<<(dim>1?"*T_1D":"")<<";\n";

  code << "\n  // -- Input field constants and basis data --\n";
  //Initialize constants, and matrices B and G
  for (CeedInt i = 0; i < numinputfields; i++) {
    code << "  // ---- Input field "<<i<<" ----\n";
    // Get elemsize, emode, ncomp
    ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetNumComponents(Erestrict, &ncomp);
    CeedChkBackend(ierr);

    // Set field constants
    if (emode != CEED_EVAL_WEIGHT) {
      ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChkBackend(ierr);
      if (basis != CEED_BASIS_COLLOCATED) {
        ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChkBackend(ierr);
        code << "  const CeedInt P_in_"<<i<<" = "<<P1d<<";\n";
      } else {
        code << "  const CeedInt P_in_"<<i<<" = "<<Q1d<<";\n";
      }
      code << "  const CeedInt ncomp_in_"<<i<<" = "<<ncomp<<";\n";
    }

    // Load basis data
    code << "  // EvalMode: "<<CeedEvalModes[emode]<<"\n";
    switch (emode) {
    case CEED_EVAL_NONE:
      break;
    case CEED_EVAL_INTERP:
      ierr = CeedBasisGetData(basis, &basis_data); CeedChkBackend(ierr);
      data->B.inputs[i] = basis_data->d_interp_1d;
      code << "  __shared__ CeedScalar s_B_in_"<<i<<"["<<P1d*Q1d<<"];\n";
      code << "  loadMatrix<P_in_"<<i<<",Q1d>(data, B.inputs["<<i<<"], s_B_in_"<<i<<");\n";
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedBasisGetData(basis, &basis_data); CeedChkBackend(ierr);
      data->B.inputs[i] = basis_data->d_interp_1d;
      code << "  __shared__ CeedScalar s_B_in_"<<i<<"["<<P1d*Q1d<<"];\n";
      code << "  loadMatrix<P_in_"<<i<<",Q1d>(data, B.inputs["<<i<<"], s_B_in_"<<i<<");\n";
      if (useCollograd) {
        data->G.inputs[i] = basis_data->d_collo_grad_1d;
        code << "  __shared__ CeedScalar s_G_in_"<<i<<"["<<Q1d*Q1d<<"];\n";
        code << "  loadMatrix<Q1d,Q1d>(data, G.inputs["<<i<<"], s_G_in_"<<i<<");\n";
      } else {
        bool has_collo_grad = !!basis_data->d_collo_grad_1d;
        data->G.inputs[i] = has_collo_grad ? basis_data->d_collo_grad_1d : basis_data->d_grad_1d;
        code << "  __shared__ CeedScalar s_G_in_"<<i<<"["<<Q1d*(has_collo_grad?Q1d:P1d)<<"];\n";
        code << "  loadMatrix<"<<(has_collo_grad?"Q1d":("P_in_"+std::to_string(i)))<<",Q1d>(data, G.inputs["<<i<<"], s_G_in_"<<i<<");\n";
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
    // Get elemsize, emode, ncomp
    ierr = CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetNumComponents(Erestrict, &ncomp);
    CeedChkBackend(ierr);

    // Set field constants
    ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis); CeedChkBackend(ierr);
    if (basis != CEED_BASIS_COLLOCATED) {
      ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChkBackend(ierr);
      code << "  const CeedInt P_out_"<<i<<" = "<<P1d<<";\n";
    } else {
      code << "  const CeedInt P_out_"<<i<<" = "<<Q1d<<";\n";
    }
    code << "  const CeedInt ncomp_out_"<<i<<" = "<<ncomp<<";\n";

    // Load basis data
    code << "  // EvalMode: "<<CeedEvalModes[emode]<<"\n";
    switch (emode) {
    case CEED_EVAL_NONE:
      break; // No action
    case CEED_EVAL_INTERP:
      ierr = CeedBasisGetData(basis, &basis_data); CeedChkBackend(ierr);
      data->B.outputs[i] = basis_data->d_interp_1d;
      code << "  __shared__ CeedScalar s_B_out_"<<i<<"["<<P1d*Q1d<<"];\n";
      code << "  loadMatrix<P_out_"<<i<<",Q1d>(data, B.outputs["<<i<<"], s_B_out_"<<i<<");\n";
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedBasisGetData(basis, &basis_data); CeedChkBackend(ierr);
      data->B.outputs[i] = basis_data->d_interp_1d;
      code << "  __shared__ CeedScalar s_B_out_"<<i<<"["<<P1d*Q1d<<"];\n";
      code << "  loadMatrix<P_out_"<<i<<",Q1d>(data, B.outputs["<<i<<"], s_B_out_"<<i<<");\n";
      if (useCollograd) {
        data->G.outputs[i] = basis_data->d_collo_grad_1d;
        code << "  __shared__ CeedScalar s_G_out_"<<i<<"["<<Q1d*Q1d<<"];\n";
        code << "  loadMatrix<Q1d,Q1d>(data, G.outputs["<<i<<"], s_G_out_"<<i<<");\n";
      } else {
        bool has_collo_grad = !!basis_data->d_collo_grad_1d;
        data->G.outputs[i] = has_collo_grad ? basis_data->d_collo_grad_1d : basis_data->d_grad_1d;
        code << "  __shared__ CeedScalar s_G_out_"<<i<<"["<<Q1d*(has_collo_grad?Q1d:P1d)<<"];\n";
        code << "  loadMatrix<"<<(has_collo_grad?"Q1d":("P_out_"+std::to_string(i)))<<",Q1d>(data, G.outputs["<<i<<"], s_G_out_"<<i<<");\n";
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
  code << "  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < nelem; elem += gridDim.x*blockDim.z) {\n";
  // Input basis apply if needed
  // Generate the correct eval mode code for each input
  code << "    // -- Input field restrictions and basis actions --\n";
  for (CeedInt i = 0; i < numinputfields; i++) {
    code << "    // ---- Input field "<<i<<" ----\n";
    // Get elemsize, emode, ncomp
    ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetNumComponents(Erestrict, &ncomp);
    CeedChkBackend(ierr);

    // Restriction
    if (emode != CEED_EVAL_WEIGHT &&
        !((emode == CEED_EVAL_NONE) && useCollograd)) {
      code << "    CeedScalar r_u"<<i<<"[ncomp_in_"<<i<<"*P_in_"<<i<<"];\n";
      
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
        data->indices.inputs[i] = restr_data->d_ind;
        code << "    readDofsOffset"<<dim<<"d<ncomp_in_"<<i<<", "<<compstride<<", P_in_"<<i<<">(data, lsize_in_"<<i<<", elem, indices.inputs["<<i<<"], d_u"<<i<<", r_u"<<i<<");\n";
      } else {
        bool backendstrides;
        ierr = CeedElemRestrictionHasBackendStrides(Erestrict, &backendstrides);
        CeedChkBackend(ierr);
        CeedInt nelem;
        ierr = CeedElemRestrictionGetNumElements(Erestrict, &nelem);
        CeedChkBackend(ierr);
        CeedInt strides[3] = {1, elemsize*nelem, elemsize};
        if (!backendstrides) {
          ierr = CeedElemRestrictionGetStrides(Erestrict, &strides);
          CeedChkBackend(ierr);
        }
        code << "    // Strides: {"<<strides[0]<<", "<<strides[1]<<", "<<strides[2]<<"}\n";
        code << "    readDofsStrided"<<dim<<"d<ncomp_in_"<<i<<",P_in_"<<i<<","<<strides[0]<<","<<strides[1]<<","<<strides[2]<<">(data, elem, d_u"<<i<<", r_u"<<i<<");\n";
      }
    }

    // Basis action
    code << "    // EvalMode: "<<CeedEvalModes[emode]<<"\n";
    switch (emode) {
    case CEED_EVAL_NONE:
      if (!useCollograd) {
        code << "    CeedScalar* r_t"<<i<<" = r_u"<<i<<";\n";
      }
      break;
    case CEED_EVAL_INTERP:
      code << "    CeedScalar r_t"<<i<<"[ncomp_in_"<<i<<"*Q1d];\n";
      code << "    Interp"<<(dim>1?"Tensor":"")<<dim<<"d<ncomp_in_"<<i<<",P_in_"<<i<<",Q1d>(data, r_u"<<i<<", s_B_in_"<<i<<", r_t"<<i<<");\n";
      break;
    case CEED_EVAL_GRAD:
      if (useCollograd) {
        code << "    CeedScalar r_t"<<i<<"[ncomp_in_"<<i<<"*Q1d];\n";
        code << "    Interp"<<(dim>1?"Tensor":"")<<dim<<"d<ncomp_in_"<<i<<",P_in_"<<i<<",Q1d>(data, r_u"<<i<<", s_B_in_"<<i<<", r_t"<<i<<");\n";
      } else {
        code << "    CeedScalar r_t"<<i<<"[ncomp_in_"<<i<<"*Dim*Q1d];\n";
        code << "    Grad"<<(dim>1?"Tensor":"")<<(dim==3&&Q1d>=P1d?"Collocated":"")<<dim<<"d<ncomp_in_"<<i<<",P_in_"<<i<<",Q1d>(data, r_u"<<i<<", s_B_in_"<<i<<", s_G_in_"<<i<<", r_t"<<i<<");\n";
      }
      break;
    case CEED_EVAL_WEIGHT:
      code << "    CeedScalar r_t"<<i<<"[Q1d];\n";
      ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChkBackend(ierr);
      ierr = CeedBasisGetData(basis, &basis_data); CeedChkBackend(ierr);
      data->W = basis_data->d_q_weight_1d;
      code << "    Weight"<<(dim>1?"Tensor":"")<<dim<<"d<Q1d>(data, W, r_t"<<i<<");\n";
      break; // No action
    case CEED_EVAL_DIV:
      break; // TODO: Not implemented
    case CEED_EVAL_CURL:
      break; // TODO: Not implemented
    }
  }

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
        code << "    CeedScalar r_tt"<<i<<"[ncomp_out_"<<i<<"*Q1d];\n";
        code << "    for (CeedInt i = 0; i < ncomp_out_"<<i<<"; ++i) {\n";
        code << "      for (CeedInt j = 0; j < Q1d; ++j) {\n";
        code << "        r_tt"<<i<<"[j + i*Q1d] = 0.0;\n";
        code << "      }\n";
        code << "    }\n";
      } else {
        code << "    CeedScalar r_tt"<<i<<"[ncomp_out_"<<i<<"*Dim*Q1d];\n";
      }
    }
    if (emode==CEED_EVAL_NONE || emode==CEED_EVAL_INTERP)
    {
      code << "    CeedScalar r_tt"<<i<<"[ncomp_out_"<<i<<"*Q1d];\n";
    }
  }
  // We treat quadrature points per slice in 3d to save registers
  if (useCollograd) {
    code << "\n    // Note: Collocated Gradient\n";
    code << "#pragma unroll\n";
    code << "    for (CeedInt q=0; q<Q1d; q++) {\n";
    code << "      // -- Input fields --\n";
    for (CeedInt i = 0; i < numinputfields; i++) {
      code << "      // ---- Input field "<<i<<" ----\n";
      // Get elemsize, emode, ncomp
      ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
      CeedChkBackend(ierr);
      // Basis action
      code << "      // EvalMode: "<<CeedEvalModes[emode]<<"\n";
      switch (emode) {
      case CEED_EVAL_NONE:
        code << "      CeedScalar r_q"<<i<<"[ncomp_in_"<<i<<"];\n";

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
          data->indices.inputs[i] = restr_data->d_ind;
          code << "      readSliceQuadsOffset"<<"3d<ncomp_in_"<<i<<", "<<compstride<<", Q1d>(data, lsize_in_"<<i<<", elem, q, indices.inputs["<<i<<"], d_u"<<i<<", r_q"<<i<<");\n";
        } else {
        ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize); CeedChkBackend(ierr);
          bool backendstrides;
          ierr = CeedElemRestrictionHasBackendStrides(Erestrict, &backendstrides);
          CeedChkBackend(ierr);
          CeedInt nelem;
          ierr = CeedElemRestrictionGetNumElements(Erestrict, &nelem);
          CeedChkBackend(ierr);
          CeedInt strides[3] = {1, elemsize*nelem, elemsize};
          if (!backendstrides) {
            ierr = CeedElemRestrictionGetStrides(Erestrict, &strides);
            CeedChkBackend(ierr);
          }
          code << "      // Strides: {"<<strides[0]<<", "<<strides[1]<<", "<<strides[2]<<"}\n";
          code << "      readSliceQuadsStrided"<<"3d<ncomp_in_"<<i<<",Q1d"","<<strides[0]<<","<<strides[1]<<","<<strides[2]<<">(data, elem, q, d_u"<<i<<", r_q"<<i<<");\n";
        }
        break;
      case CEED_EVAL_INTERP:
        code << "      CeedScalar r_q"<<i<<"[ncomp_in_"<<i<<"];\n";
        code << "      for (CeedInt j = 0; j < ncomp_in_"<<i<<" ; ++j) {\n";
        code << "        r_q"<<i<<"[j] = r_t"<<i<<"[q + j*Q1d];\n";
        code << "      }\n";
        break;
      case CEED_EVAL_GRAD:
        code << "      CeedScalar r_q"<<i<<"[ncomp_in_"<<i<<"*Dim];\n";
        code << "      gradCollo3d<ncomp_in_"<<i<<",Q1d>(data, q, r_t"<<i<<", s_G_in_"<<i<<", r_q"<<i<<");\n";
        break;
      case CEED_EVAL_WEIGHT:
        code << "      CeedScalar r_q"<<i<<"[1];\n";
        code << "      r_q"<<i<<"[0] = r_t"<<i<<"[q];\n";
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
        code << "      CeedScalar r_qq"<<i<<"[ncomp_out_"<<i<<"];\n";
        break; // No action
      case CEED_EVAL_INTERP:
        code << "      CeedScalar r_qq"<<i<<"[ncomp_out_"<<i<<"];\n";
        break;
      case CEED_EVAL_GRAD:
        code << "      CeedScalar r_qq"<<i<<"[ncomp_out_"<<i<<"*Dim];\n";
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
      code << "      CeedScalar* r_q"<<i<<" = r_t"<<i<<";\n";
    }
    code << "      // -- Output fields --\n";
    for (CeedInt i = 0; i < numoutputfields; i++) {
      code << "      // ---- Output field "<<i<<" ----\n";
      code << "      CeedScalar* r_qq"<<i<<" = r_tt"<<i<<";\n";
    }
  }
  code << "\n      // -- QFunction Inputs and outputs --\n";
  code << "      CeedScalar* in["<<numinputfields<<"];\n";
  for (CeedInt i = 0; i < numinputfields; i++) {
    code << "      // ---- Input field "<<i<<" ----\n";
    code << "      in["<<i<<"] = r_q"<<i<<";\n";
  }
  code << "      CeedScalar* out["<<numoutputfields<<"];\n";
  for (CeedInt i = 0; i < numoutputfields; i++) {
    code << "      // ---- Output field "<<i<<" ----\n";
    code << "      out["<<i<<"] = r_qq"<<i<<";\n";
  }
  code << "\n      // -- Apply QFunction --\n";
  code << "      "<<qFunctionName<<"(ctx, ";
  if (dim != 3 || useCollograd) {
    code << "1";
  } else {
    code << "Q1d";
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
        code << "      for (CeedInt j = 0; j < ncomp_out_"<<i<<" ; ++j) {\n";
        code << "        r_tt"<<i<<"[q + j*Q1d] = r_qq"<<i<<"[j];\n";
        code << "      }\n";
        break; // No action
      case CEED_EVAL_INTERP:
        code << "      for (CeedInt j = 0; j < ncomp_out_"<<i<<" ; ++j) {\n";
        code << "        r_tt"<<i<<"[q + j*Q1d] = r_qq"<<i<<"[j];\n";
        code << "      }\n";
        break;
      case CEED_EVAL_GRAD:
        code << "      gradColloTranspose3d<ncomp_out_"<<i<<",Q1d>(data, q, r_qq"<<i<<", s__"<<i<<", r_tt"<<i<<");\n";
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
    // Get elemsize, emode, ncomp
    ierr = CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetNumComponents(Erestrict, &ncomp);
    CeedChkBackend(ierr);
    // Basis action
    code << "    // EvalMode: "<<CeedEvalModes[emode]<<"\n";
    switch (emode) {
    case CEED_EVAL_NONE:
      code << "    CeedScalar* r_v"<<i<<" = r_tt"<<i<<";\n";
      break; // No action
    case CEED_EVAL_INTERP:
      code << "    CeedScalar r_v"<<i<<"[ncomp_out_"<<i<<"*P_out_"<<i<<"];\n";
      code << "    InterpTranspose"<<(dim>1?"Tensor":"")<<dim<<"d<ncomp_out_"<<i<<",P_out_"<<i<<",Q1d>(data, r_tt"<<i<<", s_B_out_"<<i<<", r_v"<<i<<");\n";
      break;
    case CEED_EVAL_GRAD:
      code << "    CeedScalar r_v"<<i<<"[ncomp_out_"<<i<<"*P_out_"<<i<<"];\n";
      if (useCollograd) {
        code << "    InterpTranspose"<<(dim>1?"Tensor":"")<<dim<<"d<ncomp_out_"<<i<<",P_out_"<<i<<",Q1d>(data, r_tt"<<i<<", s_B_out_"<<i<<", r_v"<<i<<");\n";
      } else {
        code << "    GradTranspose"<<(dim>1?"Tensor":"")<<(dim==3&&Q1d>=P1d?"Collocated":"")<<dim<<"d<ncomp_out_"<<i<<",P_out_"<<i<<",Q1d>(data, r_tt"<<i<<", s_B_out_"<<i<<", s_G_out_"<<i<<", r_v"<<i<<");\n";
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
      data->indices.outputs[i] = restr_data->d_ind;
      code << "    writeDofsOffset"<<dim<<"d<ncomp_out_"<<i<<", "<<compstride<<", P_out_"<<i<<">(data, lsize_out_"<<i<<", elem, indices.outputs["<<i<<"], r_v"<<i<<", d_v"<<i<<");\n";
    } else {
      bool backendstrides;
      ierr = CeedElemRestrictionHasBackendStrides(Erestrict, &backendstrides);
      CeedChkBackend(ierr);
      CeedInt nelem;
      ierr = CeedElemRestrictionGetNumElements(Erestrict, &nelem);
      CeedChkBackend(ierr);
      CeedInt strides[3] = {1, elemsize*nelem, elemsize};
      if (!backendstrides) {
        ierr = CeedElemRestrictionGetStrides(Erestrict, &strides);
        CeedChkBackend(ierr);
      }
      code << "    // Strides: {"<<strides[0]<<", "<<strides[1]<<", "<<strides[2]<<"}\n";
      code << "    writeDofsStrided"<<dim<<"d<ncomp_out_"<<i<<",P_out_"<<i<<","<<strides[0]<<","<<strides[1]<<","<<strides[2]<<">(data, elem, r_v"<<i<<", d_v"<<i<<");\n";
    }
  }

  code << "  }\n";
  code << "}\n";
  code << "// -----------------------------------------------------------------------------\n\n";

  // View kernel for debugging
  CeedDebug256(ceed, 2, "Generated Operator Kernels:\n");
  CeedDebug(ceed, code.str().c_str());

  CeedInt block_sizes[3] = {0, 0, 0};
  ierr = BlockGridCalculate_Hip_gen(dim, numelements, data->maxP1d, Q1d, block_sizes); 
  CeedChkBackend(ierr);
  ierr = CeedCompileHip(ceed, code.str().c_str(), &data->module, 2,
                         "T_1D", block_sizes[0],
                         "BLOCK_SIZE", block_sizes[0] * block_sizes[1] * block_sizes[2]);
  CeedChkBackend(ierr);
  ierr = CeedGetKernelHip(ceed, data->module, oper.c_str(), &data->op);
  CeedChkBackend(ierr);

  ierr = CeedOperatorSetSetupDone(op); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
