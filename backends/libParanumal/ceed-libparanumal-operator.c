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

#include "ceed-libparanumal.h"
#include <ceed-occa.h>

int CeedOperatorDestroy_libparanumal(CeedOperator op) {
  int ierr;
  CeedOperator_libparanumal *impl;
  ierr = CeedOperatorGetData(op, (void*)&impl); CeedChk(ierr);

  occaFree(impl->kernelInfo);
  occaFree(impl->kernel);

  ierr = CeedFree(&op->data); CeedChk(ierr);
  return 0;
}

static int CeedOperatorSetup_libparanumal(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  CeedOperator_libparanumal *impl;
  ierr = CeedOperatorGetData(op, (void*)&impl); CeedChk(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &opinputfields, &opoutputfields);
  CeedChk(ierr);
  // CeedQFunctionField *qfinputfields, *qfoutputfields;
  // ierr = CeedQFunctionGetFields(qf, &qfinputfields, &qfoutputfields);
  // CeedChk(ierr);
  impl->kernelInfo = occaCreateProperties();
  occaPropertiesSet(impl->kernelInfo, "defines/dlong", occaString("long"));
  occaPropertiesSet(impl->kernelInfo, "defines/dfloat", occaString("double"));
  occaPropertiesSet(impl->kernelInfo, "defines/pfloat", occaString("float"));
  if(!strcmp(qf->spec, "elliptic")) {
    CeedBasis basis = opinputfields[3]->basis;//Basis for q
    occaPropertiesSet(impl->kernelInfo, "defines/p_Np"    , occaInt(basis->P1d));
    occaPropertiesSet(impl->kernelInfo, "defines/p_Nq"    , occaInt(basis->Q1d));
    const CeedInt dim = basis->dim;
    occaPropertiesSet(impl->kernelInfo, "defines/p_dim"   , occaInt(dim));
    occaPropertiesSet(impl->kernelInfo, "defines/p_Nverts", occaInt(2*dim));//FIXME:assumes Hex topology
    //FIXME: I have no idea what those are// Might depend on the dim
    occaPropertiesSet(impl->kernelInfo, "defines/p_Nggeo", occaInt(7));
    occaPropertiesSet(impl->kernelInfo, "defines/p_G00ID", occaInt(0));
    occaPropertiesSet(impl->kernelInfo, "defines/p_G01ID", occaInt(1));
    occaPropertiesSet(impl->kernelInfo, "defines/p_G02ID", occaInt(2));
    occaPropertiesSet(impl->kernelInfo, "defines/p_G11ID", occaInt(3));
    occaPropertiesSet(impl->kernelInfo, "defines/p_G12ID", occaInt(4));
    occaPropertiesSet(impl->kernelInfo, "defines/p_G22ID", occaInt(5));
    occaPropertiesSet(impl->kernelInfo, "defines/p_GWJID", occaInt(6));
    Ceed_Occa *ceed_data;
    Ceed ceeddelegate;
    ierr = CeedGetDelegate(ceed, &ceeddelegate); CeedChk(ierr);
    ierr = CeedGetData(ceeddelegate, (void *)&ceed_data);
    const occaDevice dev = ceed_data->device;
    //TODO check the topology, store it somewhere? assume Hex?
    if (dim==1){
      //TODO is there a 1d kernel?
      impl->kernel = occaDeviceBuildKernel(dev, LIBP_OKLPATH"/ellipticAxHex3D.okl",
                                         "ellipticAxHex3D", impl->kernelInfo);
    }else if (dim==2){
      //TODO put the 2d kernel for Quad
      impl->kernel = occaDeviceBuildKernel(dev, LIBP_OKLPATH"/ellipticAxHex3D.okl",
                                         "ellipticAxHex3D", impl->kernelInfo);
    }else if (dim==3){
      impl->kernel = occaDeviceBuildKernel(dev, LIBP_OKLPATH"/ellipticAxHex3D.okl",
                                         "ellipticAxHex3D", impl->kernelInfo);
    }
  } else {
    Ceed ceed;
    CeedOperatorGetCeed(op, &ceed);
    CeedError(ceed,1," [libParanumal] Unrecognized operator. ");
  }
  impl->setupDone = true;
  return 0;
}

static int CeedOperatorApply_libparanumal(CeedOperator op, CeedVector invec,
    CeedVector outvec, CeedRequest *request) {
  int ierr;
  CeedOperator_libparanumal *impl;
  ierr = CeedOperatorGetData(op, (void*)&impl); CeedChk(ierr);
  //Finish to Setup the operrator if needed
  if (!impl->setupDone) CeedOperatorSetup_libparanumal(op);
  //Apply the operator
  CeedOperatorField *inputfields, *outputfields;
  CeedOperatorGetFields(op, &inputfields, &outputfields);
  if (!strcmp(op->qf->spec, "elliptic")) {
    CeedInt nelem   = op->numelements;
    double lambda   = 1;
    occaMemory ggeo = ((CeedVector_Occa*)inputfields[0]->vec->data)->d_array;
    occaMemory D    = ((CeedBasis_Occa*)inputfields[3]->basis->data)->grad1d;
    occaMemory S    = ((CeedVector_Occa*)inputfields[1]->vec->data)->d_array;
    occaMemory MM   = ((CeedVector_Occa*)inputfields[2]->vec->data)->d_array;
    occaMemory q    = ((CeedVector_Occa*)inputfields[3]->vec->data)->d_array;
    occaMemory Aq   = ((CeedVector_Occa*)outputfields[0]->vec->data)->d_array;
    occaKernelRun(impl->kernel, nelem, ggeo, D, S, MM, lambda, q, Aq);
  } else {
    Ceed ceed;
    CeedOperatorGetCeed(op, &ceed);
    CeedError(ceed,1," [libParanumal] Unrecognized operator. ");
  }
  return 0;
}

int CeedOperatorCreate_libparanumal(CeedOperator op) {
  int ierr;
  Ceed ceed;

  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  CeedOperator_libparanumal *impl;

  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  ierr = CeedOperatorSetData(op, (void*)&impl); CeedChk(ierr);

  impl->setupDone = false;

  ierr = CeedSetBackendFunction(ceed, "Operator", op, "Apply",
                                CeedOperatorApply_libparanumal); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "Destroy",
                                CeedOperatorDestroy_libparanumal); CeedChk(ierr);
  return 0;
}
