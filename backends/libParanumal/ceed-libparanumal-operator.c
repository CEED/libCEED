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

typedef struct {
  mesh_t mesh;
  char[BUFSIZ] fileName;
  char[BUFSIZ] kernelName;
  occa::properties kernelInfo;
  occa::kernel kernel;
} CeedOperator_libparanumal;

static int CeedOperatorSetup_libparanumal(CeedOperator op) {

}

static int CeedOperatorApply_libparanumal(CeedOperator op, CeedVector invec,
                                 CeedVector outvec, CeedRequest *request) {
  int ierr;
  CeedOperator_libparanumal *impl;
  ierr = CeedOperatorGetData(op, (void*)&impl); CeedChk(ierr);
	//Check if the operator Kernel is instanciated, otherwise creates it (jit)
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
