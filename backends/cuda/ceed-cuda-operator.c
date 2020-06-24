// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "ceed-cuda.h"
#include <string.h>

//------------------------------------------------------------------------------
// Destroy operator
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Cuda(CeedOperator op) {
  int ierr;
  CeedOperator_Cuda *impl;
  ierr = CeedOperatorGetData(op, (void *)&impl); CeedChk(ierr);

  for (CeedInt i = 0; i < impl->numein + impl->numeout; i++) {
    ierr = CeedVectorDestroy(&impl->evecs[i]); CeedChk(ierr);
  }
  ierr = CeedFree(&impl->evecs); CeedChk(ierr);
  ierr = CeedFree(&impl->edata); CeedChk(ierr);

  for (CeedInt i = 0; i < impl->numein; i++) {
    ierr = CeedVectorDestroy(&impl->qvecsin[i]); CeedChk(ierr);
  }
  ierr = CeedFree(&impl->qvecsin); CeedChk(ierr);

  for (CeedInt i = 0; i < impl->numeout; i++) {
    ierr = CeedVectorDestroy(&impl->qvecsout[i]); CeedChk(ierr);
  }
  ierr = CeedFree(&impl->qvecsout); CeedChk(ierr);

  ierr = CeedFree(&impl); CeedChk(ierr);
  return 0;
}

//------------------------------------------------------------------------------
// Setup infields or outfields
//------------------------------------------------------------------------------
static int CeedOperatorSetupFields_Cuda(CeedQFunction qf, CeedOperator op,
                                        bool inOrOut, CeedVector *evecs,
                                        CeedVector *qvecs, CeedInt starte,
                                        CeedInt numfields, CeedInt Q,
                                        CeedInt numelements) {
  CeedInt dim, ierr, size;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  CeedBasis basis;
  CeedElemRestriction Erestrict;
  CeedOperatorField *opfields;
  CeedQFunctionField *qffields;
  CeedVector fieldvec;
  bool strided;
  bool skiprestrict;

  if (inOrOut) {
    ierr = CeedOperatorGetFields(op, NULL, &opfields);
    CeedChk(ierr);
    ierr = CeedQFunctionGetFields(qf, NULL, &qffields);
    CeedChk(ierr);
  } else {
    ierr = CeedOperatorGetFields(op, &opfields, NULL);
    CeedChk(ierr);
    ierr = CeedQFunctionGetFields(qf, &qffields, NULL);
    CeedChk(ierr);
  }

  // Loop over fields
  for (CeedInt i = 0; i < numfields; i++) {
    CeedEvalMode emode;
    ierr = CeedQFunctionFieldGetEvalMode(qffields[i], &emode); CeedChk(ierr);

    strided = false;
    skiprestrict = false;
    if (emode != CEED_EVAL_WEIGHT) {
      ierr = CeedOperatorFieldGetElemRestriction(opfields[i], &Erestrict);
      CeedChk(ierr);

      // Check whether this field can skip the element restriction:
      // must be passive input, with emode NONE, and have a strided restriction with
      // CEED_STRIDES_BACKEND.

      // First, check whether the field is input or output:
      if (!inOrOut) {
        // Check for passive input:
        ierr = CeedOperatorFieldGetVector(opfields[i], &fieldvec); CeedChk(ierr);
        if (fieldvec != CEED_VECTOR_ACTIVE) {
          // Check emode
          if (emode == CEED_EVAL_NONE) {
            // Check for strided restriction
            ierr = CeedElemRestrictionIsStrided(Erestrict, &strided);
            CeedChk(ierr);
            if (strided) {
              // Check if vector is already in preferred backend ordering
              ierr = CeedElemRestrictionHasBackendStrides(Erestrict,
                     &skiprestrict); CeedChk(ierr);
            }
          }
        }
      }
      if (skiprestrict) {
        // We do not need an E-Vector, but will use the input field vector's data
        // directly in the operator application.
        evecs[i + starte] = NULL;
      } else {
        ierr = CeedElemRestrictionCreateVector(Erestrict, NULL,
                                               &evecs[i + starte]);
        CeedChk(ierr);
      }
    }

    switch (emode) {
    case CEED_EVAL_NONE:
      ierr = CeedQFunctionFieldGetSize(qffields[i], &size); CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, numelements * Q * size, &qvecs[i]);
      CeedChk(ierr);
      break;
    case CEED_EVAL_INTERP:
      ierr = CeedQFunctionFieldGetSize(qffields[i], &size); CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, numelements * Q * size, &qvecs[i]);
      CeedChk(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(opfields[i], &basis); CeedChk(ierr);
      ierr = CeedQFunctionFieldGetSize(qffields[i], &size); CeedChk(ierr);
      ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, numelements * Q * size, &qvecs[i]);
      CeedChk(ierr);
      break;
    case CEED_EVAL_WEIGHT: // Only on input fields
      ierr = CeedOperatorFieldGetBasis(opfields[i], &basis); CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, numelements * Q, &qvecs[i]); CeedChk(ierr);
      ierr = CeedBasisApply(basis, numelements, CEED_NOTRANSPOSE,
                            CEED_EVAL_WEIGHT, NULL, qvecs[i]); CeedChk(ierr);
      break;
    case CEED_EVAL_DIV:
      break; // TODO: Not implemented
    case CEED_EVAL_CURL:
      break; // TODO: Not implemented
    }
  }
  return 0;
}

//------------------------------------------------------------------------------
// CeedOperator needs to connect all the named fields (be they active or passive)
//   to the named inputs and outputs of its CeedQFunction.
//------------------------------------------------------------------------------
static int CeedOperatorSetup_Cuda(CeedOperator op) {
  int ierr;
  bool setupdone;
  ierr = CeedOperatorIsSetupDone(op, &setupdone); CeedChk(ierr);
  if (setupdone)
    return 0;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  CeedOperator_Cuda *impl;
  ierr = CeedOperatorGetData(op, (void *)&impl); CeedChk(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
  CeedInt Q, numelements, numinputfields, numoutputfields;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChk(ierr);
  ierr = CeedOperatorGetNumElements(op, &numelements); CeedChk(ierr);
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChk(ierr);
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &opinputfields, &opoutputfields);
  CeedChk(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, &qfinputfields, &qfoutputfields);
  CeedChk(ierr);

  // Allocate
  ierr = CeedCalloc(numinputfields + numoutputfields, &impl->evecs);
  CeedChk(ierr);
  ierr = CeedCalloc(numinputfields + numoutputfields, &impl->edata);
  CeedChk(ierr);

  ierr = CeedCalloc(16, &impl->qvecsin); CeedChk(ierr);
  ierr = CeedCalloc(16, &impl->qvecsout); CeedChk(ierr);

  impl->numein = numinputfields; impl->numeout = numoutputfields;

  // Set up infield and outfield evecs and qvecs
  // Infields
  ierr = CeedOperatorSetupFields_Cuda(qf, op, 0,
                                      impl->evecs, impl->qvecsin, 0,
                                      numinputfields, Q, numelements);
  CeedChk(ierr);

  // Outfields
  ierr = CeedOperatorSetupFields_Cuda(qf, op, 1,
                                      impl->evecs, impl->qvecsout,
                                      numinputfields, numoutputfields, Q,
                                      numelements); CeedChk(ierr);

  ierr = CeedOperatorSetSetupDone(op); CeedChk(ierr);
  return 0;
}

//------------------------------------------------------------------------------
// Setup Operator Inputs
//------------------------------------------------------------------------------
static inline int CeedOperatorSetupInputs_Cuda(CeedInt numinputfields,
    CeedQFunctionField *qfinputfields, CeedOperatorField *opinputfields,
    CeedVector invec, const bool skipactive, CeedOperator_Cuda *impl,
    CeedRequest *request) {
  CeedInt ierr;
  CeedEvalMode emode;
  CeedVector vec;
  CeedElemRestriction Erestrict;

  for (CeedInt i = 0; i < numinputfields; i++) {
    // Get input vector
    ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChk(ierr);
    if (vec == CEED_VECTOR_ACTIVE) {
      if (skipactive)
        continue;
      else
        vec = invec;
    }

    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChk(ierr);
    if (emode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      // Get input vector
      ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChk(ierr);
      // Get input element restriction
      ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict);
      CeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
        vec = invec;
      // Restrict, if necessary
      if (!impl->evecs[i]) {
        // No restriction for this field; read data directly from vec.
        ierr = CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE,
                                      (const CeedScalar **) &impl->edata[i]);
        CeedChk(ierr);
      } else {
        ierr = CeedElemRestrictionApply(Erestrict, CEED_NOTRANSPOSE, vec,
                                        impl->evecs[i], request); CeedChk(ierr);
        // Get evec
        ierr = CeedVectorGetArrayRead(impl->evecs[i], CEED_MEM_DEVICE,
                                      (const CeedScalar **) &impl->edata[i]);
        CeedChk(ierr);
      }
    }
  }
  return 0;
}

//------------------------------------------------------------------------------
// Input Basis Action
//------------------------------------------------------------------------------
static inline int CeedOperatorInputBasis_Cuda(CeedInt numelements,
    CeedQFunctionField *qfinputfields, CeedOperatorField *opinputfields,
    CeedInt numinputfields, const bool skipactive, CeedOperator_Cuda *impl) {
  CeedInt ierr;
  CeedInt elemsize, size;
  CeedElemRestriction Erestrict;
  CeedEvalMode emode;
  CeedBasis basis;

  for (CeedInt i=0; i<numinputfields; i++) {
    // Skip active input
    if (skipactive) {
      CeedVector vec;
      ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
        continue;
    }
    // Get elemsize, emode, size
    ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict);
    CeedChk(ierr);
    ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
    CeedChk(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChk(ierr);
    ierr = CeedQFunctionFieldGetSize(qfinputfields[i], &size); CeedChk(ierr);
    // Basis action
    switch (emode) {
    case CEED_EVAL_NONE:
      ierr = CeedVectorSetArray(impl->qvecsin[i], CEED_MEM_DEVICE,
                                CEED_USE_POINTER,
                                impl->edata[i]); CeedChk(ierr);
      break;
    case CEED_EVAL_INTERP:
      ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChk(ierr);
      ierr = CeedBasisApply(basis, numelements, CEED_NOTRANSPOSE,
                            CEED_EVAL_INTERP, impl->evecs[i],
                            impl->qvecsin[i]); CeedChk(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChk(ierr);
      ierr = CeedBasisApply(basis, numelements, CEED_NOTRANSPOSE,
                            CEED_EVAL_GRAD, impl->evecs[i],
                            impl->qvecsin[i]); CeedChk(ierr);
      break;
    case CEED_EVAL_WEIGHT:
      break; // No action
    case CEED_EVAL_DIV:
      break; // TODO: Not implemented
    case CEED_EVAL_CURL:
      break; // TODO: Not implemented
    }
  }
  return 0;
}

//------------------------------------------------------------------------------
// Restore Input Vectors
//------------------------------------------------------------------------------
static inline int CeedOperatorRestoreInputs_Cuda(CeedInt numinputfields,
    CeedQFunctionField *qfinputfields, CeedOperatorField *opinputfields,
    const bool skipactive, CeedOperator_Cuda *impl) {
  CeedInt ierr;
  CeedEvalMode emode;
  CeedVector vec;

  for (CeedInt i = 0; i < numinputfields; i++) {
    // Skip active input
    if (skipactive) {
      ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
        continue;
    }
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChk(ierr);
    if (emode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      if (!impl->evecs[i]) {  // This was a skiprestrict case
        ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChk(ierr);
        ierr = CeedVectorRestoreArrayRead(vec,
                                          (const CeedScalar **)&impl->edata[i]);
        CeedChk(ierr);
      } else {
        ierr = CeedVectorRestoreArrayRead(impl->evecs[i],
                                          (const CeedScalar **) &impl->edata[i]);
        CeedChk(ierr);
      }
    }
  }
  return 0;
}

//------------------------------------------------------------------------------
// Apply and add to output
//------------------------------------------------------------------------------
static int CeedOperatorApplyAdd_Cuda(CeedOperator op, CeedVector invec,
                                     CeedVector outvec, CeedRequest *request) {
  int ierr;
  CeedOperator_Cuda *impl;
  ierr = CeedOperatorGetData(op, (void *)&impl); CeedChk(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
  CeedInt Q, numelements, elemsize, numinputfields, numoutputfields, size;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChk(ierr);
  ierr = CeedOperatorGetNumElements(op, &numelements); CeedChk(ierr);
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChk(ierr);
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &opinputfields, &opoutputfields);
  CeedChk(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, &qfinputfields, &qfoutputfields);
  CeedChk(ierr);
  CeedEvalMode emode;
  CeedVector vec;
  CeedBasis basis;
  CeedElemRestriction Erestrict;

  // Setup
  ierr = CeedOperatorSetup_Cuda(op); CeedChk(ierr);

  // Input Evecs and Restriction
  ierr = CeedOperatorSetupInputs_Cuda(numinputfields, qfinputfields,
                                      opinputfields, invec, false, impl,
                                      request); CeedChk(ierr);

  // Input basis apply if needed
  ierr = CeedOperatorInputBasis_Cuda(numelements, qfinputfields, opinputfields,
                                     numinputfields, false, impl);
  CeedChk(ierr);

  // Output pointers, as necessary
  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChk(ierr);
    if (emode == CEED_EVAL_NONE) {
      // Set the output Q-Vector to use the E-Vector data directly.
      ierr = CeedVectorGetArray(impl->evecs[i + impl->numein], CEED_MEM_DEVICE,
                                &impl->edata[i + numinputfields]); CeedChk(ierr);
      ierr = CeedVectorSetArray(impl->qvecsout[i], CEED_MEM_DEVICE,
                                CEED_USE_POINTER,
                                impl->edata[i + numinputfields]);
      CeedChk(ierr);
    }
  }

  // Q function
  ierr = CeedQFunctionApply(qf, numelements * Q, impl->qvecsin, impl->qvecsout);
  CeedChk(ierr);

  // Output basis apply if needed
  for (CeedInt i = 0; i < numoutputfields; i++) {
    // Get elemsize, emode, size
    ierr = CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict);
    CeedChk(ierr);
    ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
    CeedChk(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChk(ierr);
    ierr = CeedQFunctionFieldGetSize(qfoutputfields[i], &size); CeedChk(ierr);
    // Basis action
    switch (emode) {
    case CEED_EVAL_NONE:
      break;
    case CEED_EVAL_INTERP:
      ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis);
      CeedChk(ierr);
      ierr = CeedBasisApply(basis, numelements, CEED_TRANSPOSE,
                            CEED_EVAL_INTERP, impl->qvecsout[i],
                            impl->evecs[i + impl->numein]); CeedChk(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis);
      CeedChk(ierr);
      ierr = CeedBasisApply(basis, numelements, CEED_TRANSPOSE,
                            CEED_EVAL_GRAD, impl->qvecsout[i],
                            impl->evecs[i + impl->numein]); CeedChk(ierr);
      break;
    case CEED_EVAL_WEIGHT: {
      Ceed ceed;
      ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
      return CeedError(ceed, 1,
                       "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
      break; // Should not occur
    }
    case CEED_EVAL_DIV:
      break; // TODO: Not implemented
    case CEED_EVAL_CURL:
      break; // TODO: Not implemented
    }
  }

  // Output restriction
  for (CeedInt i = 0; i < numoutputfields; i++) {
    // Restore evec
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChk(ierr);
    if (emode == CEED_EVAL_NONE) {
      ierr = CeedVectorRestoreArray(impl->evecs[i+impl->numein],
                                    &impl->edata[i + numinputfields]);
      CeedChk(ierr);
    }
    // Get output vector
    ierr = CeedOperatorFieldGetVector(opoutputfields[i], &vec); CeedChk(ierr);
    // Restrict
    ierr = CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict);
    CeedChk(ierr);
    // Active
    if (vec == CEED_VECTOR_ACTIVE)
      vec = outvec;

    ierr = CeedElemRestrictionApply(Erestrict, CEED_TRANSPOSE,
                                    impl->evecs[i + impl->numein], vec,
                                    request); CeedChk(ierr);
  }

  // Restore input arrays
  ierr = CeedOperatorRestoreInputs_Cuda(numinputfields, qfinputfields,
                                        opinputfields, false, impl);
  CeedChk(ierr);
  return 0;
}

//------------------------------------------------------------------------------
// Assemble Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunction_Cuda(CeedOperator op,
    CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  int ierr;
  CeedOperator_Cuda *impl;
  ierr = CeedOperatorGetData(op, (void *)&impl); CeedChk(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
  CeedInt Q, numelements, numinputfields, numoutputfields, size;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChk(ierr);
  ierr = CeedOperatorGetNumElements(op, &numelements); CeedChk(ierr);
  ierr= CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChk(ierr);
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &opinputfields, &opoutputfields);
  CeedChk(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, &qfinputfields, &qfoutputfields);
  CeedChk(ierr);
  CeedVector vec;
  CeedInt numactivein = 0, numactiveout = 0;
  CeedVector *activein = NULL;
  CeedScalar *a, *tmp;
  Ceed ceed, ceedparent;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  ierr = CeedGetOperatorFallbackParentCeed(ceed, &ceedparent); CeedChk(ierr);
  ceedparent = ceedparent ? ceedparent : ceed;

  // Setup
  ierr = CeedOperatorSetup_Cuda(op); CeedChk(ierr);

  // Check for identity
  bool identityqf;
  ierr = CeedQFunctionIsIdentity(qf, &identityqf); CeedChk(ierr);
  if (identityqf)
    // LCOV_EXCL_START
    return CeedError(ceed, 1, "Assembling identity QFunctions not supported");
  // LCOV_EXCL_STOP

  // Input Evecs and Restriction
  ierr = CeedOperatorSetupInputs_Cuda(numinputfields, qfinputfields,
                                      opinputfields, NULL, true, impl, request);
  CeedChk(ierr);

  // Count number of active input fields
  for (CeedInt i=0; i<numinputfields; i++) {
    // Get input vector
    ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChk(ierr);
    // Check if active input
    if (vec == CEED_VECTOR_ACTIVE) {
      ierr = CeedQFunctionFieldGetSize(qfinputfields[i], &size); CeedChk(ierr);
      ierr = CeedVectorSetValue(impl->qvecsin[i], 0.0); CeedChk(ierr);
      ierr = CeedVectorGetArray(impl->qvecsin[i], CEED_MEM_DEVICE, &tmp);
      CeedChk(ierr);
      ierr = CeedRealloc(numactivein + size, &activein); CeedChk(ierr);
      for (CeedInt field=0; field<size; field++) {
        ierr = CeedVectorCreate(ceed, Q*numelements,
                                &activein[numactivein+field]); CeedChk(ierr);
        ierr = CeedVectorSetArray(activein[numactivein+field], CEED_MEM_DEVICE,
                                  CEED_USE_POINTER, &tmp[field*Q*numelements]);
        CeedChk(ierr);
      }
      numactivein += size;
      ierr = CeedVectorRestoreArray(impl->qvecsin[i], &tmp); CeedChk(ierr);
    }
  }

  // Count number of active output fields
  for (CeedInt i=0; i<numoutputfields; i++) {
    // Get output vector
    ierr = CeedOperatorFieldGetVector(opoutputfields[i], &vec); CeedChk(ierr);
    // Check if active output
    if (vec == CEED_VECTOR_ACTIVE) {
      ierr = CeedQFunctionFieldGetSize(qfoutputfields[i], &size); CeedChk(ierr);
      numactiveout += size;
    }
  }

  // Check sizes
  if (!numactivein || !numactiveout)
    // LCOV_EXCL_START
    return CeedError(ceed, 1, "Cannot assemble QFunction without active inputs "
                     "and outputs");
  // LCOV_EXCL_STOP

  // Create output restriction
  CeedInt strides[3] = {1, numelements*Q, Q}; /* *NOPAD* */
  ierr = CeedElemRestrictionCreateStrided(ceedparent, numelements, Q,
                                          numactivein*numactiveout,
                                          numactivein*numactiveout*numelements*Q,
                                          strides, rstr); CeedChk(ierr);
  // Create assembled vector
  ierr = CeedVectorCreate(ceedparent, numelements*Q*numactivein*numactiveout,
                          assembled); CeedChk(ierr);
  ierr = CeedVectorSetValue(*assembled, 0.0); CeedChk(ierr);
  ierr = CeedVectorGetArray(*assembled, CEED_MEM_DEVICE, &a); CeedChk(ierr);

  // Input basis apply
  ierr = CeedOperatorInputBasis_Cuda(numelements, qfinputfields, opinputfields,
                                     numinputfields, true, impl);
  CeedChk(ierr);

  // Assemble QFunction
  for (CeedInt in=0; in<numactivein; in++) {
    // Set Inputs
    ierr = CeedVectorSetValue(activein[in], 1.0); CeedChk(ierr);
    if (numactivein > 1) {
      ierr = CeedVectorSetValue(activein[(in+numactivein-1)%numactivein],
                                0.0); CeedChk(ierr);
    }
    // Set Outputs
    for (CeedInt out=0; out<numoutputfields; out++) {
      // Get output vector
      ierr = CeedOperatorFieldGetVector(opoutputfields[out], &vec);
      CeedChk(ierr);
      // Check if active output
      if (vec == CEED_VECTOR_ACTIVE) {
        CeedVectorSetArray(impl->qvecsout[out], CEED_MEM_DEVICE,
                           CEED_USE_POINTER, a); CeedChk(ierr);
        ierr = CeedQFunctionFieldGetSize(qfoutputfields[out], &size);
        CeedChk(ierr);
        a += size*Q*numelements; // Advance the pointer by the size of the output
      }
    }
    // Apply QFunction
    ierr = CeedQFunctionApply(qf, Q*numelements, impl->qvecsin, impl->qvecsout);
    CeedChk(ierr);
  }


  // Un-set output Qvecs to prevent accidental overwrite of Assembled
  for (CeedInt out=0; out<numoutputfields; out++) {
    // Get output vector
    ierr = CeedOperatorFieldGetVector(opoutputfields[out], &vec);
    CeedChk(ierr);
    // Check if active output
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedVectorSetArray(impl->qvecsout[out], CEED_MEM_HOST, CEED_COPY_VALUES,
                         NULL); CeedChk(ierr);
    }
  }

  // Restore input arrays
  ierr = CeedOperatorRestoreInputs_Cuda(numinputfields, qfinputfields,
                                        opinputfields, true, impl);
  CeedChk(ierr);

  // Restore output
  ierr = CeedVectorRestoreArray(*assembled, &a); CeedChk(ierr);

  // Cleanup
  for (CeedInt i=0; i<numactivein; i++) {
    ierr = CeedVectorDestroy(&activein[i]); CeedChk(ierr);
  }
  ierr = CeedFree(&activein); CeedChk(ierr);

  return 0;
}

//------------------------------------------------------------------------------
// Assemble linear diagonal not supported
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleDiagonal_Cuda(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  return CeedError(ceed, 1,
                   "Backend does not implement Operator diagonal assembly");
}

//------------------------------------------------------------------------------
// Assemble linear point block diagonal not supported
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssemblePointBlockDiagonal_Cuda(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  return CeedError(ceed, 1,
                   "Backend does not implement Operator point block diagonal assembly");
}

//------------------------------------------------------------------------------
// Create FDM element inverse not supported
//------------------------------------------------------------------------------
static int CeedOperatorCreateFDMElementInverse_Cuda(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  return CeedError(ceed, 1, "Backend does not implement FDM inverse creation");
}

//------------------------------------------------------------------------------
// Create operator
//------------------------------------------------------------------------------
int CeedOperatorCreate_Cuda(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  CeedOperator_Cuda *impl;

  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  ierr = CeedOperatorSetData(op, (void *)&impl); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunction",
                                CeedOperatorLinearAssembleQFunction_Cuda);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleDiagonal",
                                CeedOperatorLinearAssembleDiagonal_Cuda);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op,
                                "LinearAssemblePointBlockDiagonal",
                                CeedOperatorLinearAssemblePointBlockDiagonal_Cuda);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "CreateFDMElementInverse",
                                CeedOperatorCreateFDMElementInverse_Cuda);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd",
                                CeedOperatorApplyAdd_Cuda); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "Destroy",
                                CeedOperatorDestroy_Cuda); CeedChk(ierr);
  return 0;
}
//------------------------------------------------------------------------------
