// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other
// CEED contributors. All Rights Reserved. See the top-level LICENSE and NOTICE
// files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>

#include <cassert>
#include <string>
#include <sycl/sycl.hpp>

#include "../sycl/ceed-sycl-compile.hpp"
#include "ceed-sycl-ref.hpp"

//------------------------------------------------------------------------------
// Destroy operator
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Sycl(CeedOperator op) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Setup infields or outfields
//------------------------------------------------------------------------------
static int CeedOperatorSetupFields_Sycl(CeedQFunction qf, CeedOperator op, bool isinput, CeedVector *evecs, CeedVector *qvecs, CeedInt starte,
                                        CeedInt numfields, CeedInt Q, CeedInt numelements) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// CeedOperator needs to connect all the named fields (be they active or
// passive) to the named inputs and outputs of its CeedQFunction.
//------------------------------------------------------------------------------
static int CeedOperatorSetup_Sycl(CeedOperator op) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Setup Operator Inputs
//------------------------------------------------------------------------------
static inline int CeedOperatorSetupInputs_Sycl(CeedInt numinputfields, CeedQFunctionField *qfinputfields, CeedOperatorField *opinputfields,
                                               CeedVector invec, const bool skipactive, CeedScalar *edata[2 * CEED_FIELD_MAX],
                                               CeedOperator_Sycl *impl, CeedRequest *request) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(invec, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Input Basis Action
//------------------------------------------------------------------------------
static inline int CeedOperatorInputBasis_Sycl(CeedInt numelements, CeedQFunctionField *qfinputfields, CeedOperatorField *opinputfields,
                                              CeedInt numinputfields, const bool skipactive, CeedScalar *edata[2 * CEED_FIELD_MAX],
                                              CeedOperator_Sycl *impl) {
  return CeedError(NULL, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Restore Input Vectors
//------------------------------------------------------------------------------
static inline int CeedOperatorRestoreInputs_Sycl(CeedInt numinputfields, CeedQFunctionField *qfinputfields, CeedOperatorField *opinputfields,
                                                 const bool skipactive, CeedScalar *edata[2 * CEED_FIELD_MAX], CeedOperator_Sycl *impl) {
  return CeedError(NULL, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Apply and add to output
//------------------------------------------------------------------------------
static int CeedOperatorApplyAdd_Sycl(CeedOperator op, CeedVector invec, CeedVector outvec, CeedRequest *request) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Core code for assembling linear QFunction
//------------------------------------------------------------------------------
static inline int CeedOperatorLinearAssembleQFunctionCore_Sycl(CeedOperator op, bool build_objects, CeedVector *assembled, CeedElemRestriction *rstr,
                                                               CeedRequest *request) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Assemble Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunction_Sycl(CeedOperator op, CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  // Kris: Leave this like the CUDA backend for now, but we should review this later
  return CeedOperatorLinearAssembleQFunctionCore_Sycl(op, true, assembled, rstr, request);
}

//------------------------------------------------------------------------------
// Update Assembled Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunctionUpdate_Sycl(CeedOperator op, CeedVector assembled, CeedElemRestriction rstr, CeedRequest *request) {
  // Kris: Leave this like the CUDA backend for now, but we should review this later
  return CeedOperatorLinearAssembleQFunctionCore_Sycl(op, false, &assembled, &rstr, request);
}

//------------------------------------------------------------------------------
// Create point block restriction
//------------------------------------------------------------------------------
static int CreatePBRestriction(CeedElemRestriction rstr, CeedElemRestriction *pbRstr) {
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(rstr, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Assemble diagonal setup
//------------------------------------------------------------------------------
static inline int CeedOperatorAssembleDiagonalSetup_Sycl(CeedOperator op, const bool pointBlock) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Assemble diagonal common code
//------------------------------------------------------------------------------
static inline int CeedOperatorAssembleDiagonalCore_Sycl(CeedOperator op, CeedVector assembled, CeedRequest *request, const bool pointBlock) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Assemble Linear Diagonal
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleAddDiagonal_Sycl(CeedOperator op, CeedVector assembled, CeedRequest *request) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Assemble Linear Point Block Diagonal
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleAddPointBlockDiagonal_Sycl(CeedOperator op, CeedVector assembled, CeedRequest *request) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Single operator assembly setup
//------------------------------------------------------------------------------
static int CeedSingleOperatorAssembleSetup_Sycl(CeedOperator op) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Assemble matrix data for COO matrix of assembled operator.
// The sparsity pattern is set by CeedOperatorLinearAssembleSymbolic.
//
// Note that this (and other assembly routines) currently assume only one active
// input restriction/basis per operator (could have multiple basis eval modes).
// TODO: allow multiple active input restrictions/basis objects
//------------------------------------------------------------------------------
static int CeedSingleOperatorAssemble_Sycl(CeedOperator op, CeedInt offset, CeedVector values) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Create operator
//------------------------------------------------------------------------------
int CeedOperatorCreate_Sycl(CeedOperator op) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
