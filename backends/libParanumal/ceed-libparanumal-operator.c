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

#include <ceed-impl.h>
#include "ceed-cuda.h"

#define BUFSIZ 16

typedef struct {
  //libCeed stuff
  CeedVector  *evecs;   /// E-vectors needed to apply operator (input followed by outputs)
  CeedScalar **edata;
  CeedScalar **qdata; /// Inputs followed by outputs
  CeedScalar **qdata_alloc; /// Allocated quadrature data arrays (to be freed by us)
  CeedScalar **indata;
  CeedScalar **outdata;
  CeedInt    numein;
  CeedInt    numeout;
  CeedInt    numqin;
  CeedInt    numqout;
  //libParanumal stuff
  mesh_t mesh;
  char[BUFSIZ] fileName;
  char[BUFSIZ] kernelName;
  occa::properties kernelInfo;
  occa::kernel kernel;
} CeedOperator_libparanumal;

static int CeedOperatorDestroy_libparanumal(CeedOperator op) {
  //TODO Destroy the CeedOperator_libparanumal?
  int ierr;
  CeedOperator_libparanumal *impl;
  ierr = CeedOperatorGetData(op, (void*)&impl); CeedChk(ierr);

  for (CeedInt i=0; i<impl->numein+impl->numeout; i++) {
    ierr = CeedVectorDestroy(&impl->evecs[i]); CeedChk(ierr);
  }
  ierr = CeedFree(&impl->evecs); CeedChk(ierr);
  ierr = CeedFree(&impl->edata); CeedChk(ierr);

  for (CeedInt i=0; i<impl->numqin+impl->numqout; i++) {
    ierr = CeedFree(&impl->qdata_alloc[i]); CeedChk(ierr);
  }
  ierr = CeedFree(&impl->qdata_alloc); CeedChk(ierr);
  ierr = CeedFree(&impl->qdata); CeedChk(ierr);

  ierr = CeedFree(&impl->indata); CeedChk(ierr);
  ierr = CeedFree(&impl->outdata); CeedChk(ierr);

  //Free mesh?
  occaFree(impl->kernelInfo);
  occaFree(impl->kernel);

  ierr = CeedFree(&op->data); CeedChk(ierr);
  return 1;
}

static int CeedOperatorSetupFields_libparanumal(CeedOperator op,
                                        CeedQFunctionField qfields[16],
                                        CeedOperatorField ofields[16],
                                        CeedVector *evecs, CeedScalar **qdata, CeedScalar **qdata_alloc,
                                        CeedScalar **indata,
                                        const CeedInt starti,
                                        CeedInt startq,
                                        const CeedInt numfields,
                                        const CeedInt Q) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
  CeedQFunction_libparanumal *qf_data;//FIXME What do we expect?!
  ierr = CeedQFunctionGetData(qf, (void*)&qf_data); CeedChk(ierr);
  CeedInt ncomp;
  CeedInt dim = 1;
  CeedInt iq=startq;
  CeedBasis basis;
  CeedElemRestriction Erestrict;

  // Loop over fields
  for (CeedInt i=0; i<numfields; i++) {
    CeedEvalMode emode;
    ierr = CeedQFunctionFieldGetEvalMode(qfields[i], &emode); CeedChk(ierr);
    if (emode != CEED_EVAL_WEIGHT) {
      ierr = CeedOperatorFieldGetElemRestriction(ofields[i], &Erestrict); CeedChk(ierr);
      ierr = CeedElemRestrictionCreateVector(Erestrict, NULL, &evecs[i+starti]); CeedChk(ierr);
    } else {
      //TODO something missing?
    }
    switch(emode) {
    case CEED_EVAL_NONE:
      break; // No action
    case CEED_EVAL_INTERP:
      ierr = CeedOperatorFieldGetBasis(ofields[i], &basis); CeedChk(ierr);
      ierr = CeedQFunctionFieldGetNumComponents(qfields[i], &ncomp); CeedChk(ierr);
      ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
      ierr = CeedMalloc(Q*ncomp, &qdata_alloc[iq]); CeedChk(ierr);
      qdata[i + starti] = qdata_alloc[iq];
      iq++;
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(ofields[i], &basis); CeedChk(ierr);
      ierr = CeedQFunctionFieldGetNumComponents(qfields[i], &ncomp); CeedChk(ierr);
      ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
      ierr = CeedMalloc(Q*ncomp*dim, &qdata_alloc[iq]); CeedChk(ierr);
      qdata[i + starti] = qdata_alloc[iq];
      iq++;
      break;
    case CEED_EVAL_WEIGHT: // Only on input fields
      ierr = CeedOperatorFieldGetBasis(ofields[i], &basis); CeedChk(ierr);
      ierr = CeedMalloc(Q, &qdata_alloc[iq]); CeedChk(ierr);
      ierr = CeedBasisApply(basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT,
                            NULL, qdata_alloc[iq]); CeedChk(ierr);
      assert(starti==0);
      ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
      qdata[i + starti] = qdata_alloc[iq];
      indata[i] = qdata[i];
      break;
    case CEED_EVAL_DIV: break; // TODO Not implemented
    case CEED_EVAL_CURL: break; // TODO Not implemented
    }
    qf_data->dim = dim;
  }
  return 0;
}

static int CeedOperatorSetup_libparanumal(CeedOperator op) {
  int ierr;
  CeedOperator_libparanumal *impl;
  ierr = CeedOperatorGetData(op, (void*)&impl); CeedChk(ierr);
  //TODO
  return 1;
}

static int CeedOperatorApply_libparanumal(CeedOperator op, CeedVector invec,
                                 CeedVector outvec, CeedRequest *request) {
  int ierr;
  CeedOperator_libparanumal *impl;
  ierr = CeedOperatorGetData(op, (void*)&impl); CeedChk(ierr);
	//Check if the operator Kernel is instanciated, otherwise creates it (jit)
  // Is it what goes in OperatorSetup?
  if(impl->kernel==NULL){
    impl->mesh;//TODO
    impl->fileName;//TODO
    impl->kernelName;//TODO
    impl->kernelInfo;//TODO
    impl->kernel = mesh->device.buildKernel(fileName, kernelName, kernelInfo);
  }
	//Apply the operator
  mesh_t *mesh = impl->mesh;
  impl->kernel(mesh->Nelements, 
                mesh->o_vgeo, 
                mesh->o_Dmatrices,
                (occa::memory)invec->data->d_array, 
                (occa::memory)outvec->data->d_array);
	return 1;
}

int CeedOperatorCreate_libparanumal(CeedOperator op) {
  int ierr;
  CeedOperator_libparanumal *impl;

  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  op->data = impl;
  op->Destroy = CeedOperatorDestroy_libparanumal;
  op->Apply = CeedOperatorApply_libparanumal;
  return 0;
}
