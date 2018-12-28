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
    occaPropertiesSet(impl->kernelInfo, "defines/p_Np"    , occaInt(basis->P));
    occaPropertiesSet(impl->kernelInfo, "defines/p_Nq"    , occaInt(basis->Q1d));
    const CeedInt dim = basis->dim;
    occaPropertiesSet(impl->kernelInfo, "defines/p_dim"   , occaInt(dim));
    occaPropertiesSet(impl->kernelInfo, "defines/p_Nverts",
                      occaInt(CeedIntPow(2, dim)));//FIXME:assumes Hex topology
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
    if (dim==1) {
      //TODO is there a 1d kernel?
      impl->kernel = occaDeviceBuildKernel(dev, LIBP_OKLPATH"/ellipticAxHex3D.okl",
                                           "ellipticAxHex3D", impl->kernelInfo);
    } else if (dim==2) {
      //TODO put the 2d kernel for Quad
      impl->kernel = occaDeviceBuildKernel(dev, LIBP_OKLPATH"/ellipticAxHex3D.okl",
                                           "ellipticAxHex3D", impl->kernelInfo);
    } else if (dim==3) {
      impl->kernel = occaDeviceBuildKernel(dev, LIBP_OKLPATH"/ellipticAxHex3D.okl",
                                           "ellipticAxHex3D", impl->kernelInfo);
    }
    CeedElemRestriction Erestrict;
    ierr = CeedOperatorFieldGetElemRestriction(opinputfields[2], &Erestrict);
    CeedChk(ierr);
    ierr = CeedElemRestrictionCreateVector(Erestrict, NULL, &impl->evecIn);
    CeedChk(ierr);
    ierr = CeedOperatorFieldGetElemRestriction(opinputfields[3], &Erestrict);
    CeedChk(ierr);
    ierr = CeedElemRestrictionCreateVector(Erestrict, NULL, &impl->evecOut);
    CeedChk(ierr);
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
    double lambda   = 0.0;
    occaMemory ggeo = ((CeedVector_Occa*)inputfields[0]->vec->data)->d_array;
    occaMemory D    = ((CeedBasis_Occa*)inputfields[3]->basis->data)->grad1d;
    occaMemory S    = ((CeedVector_Occa*)inputfields[1]->vec->data)->d_array;//unused by the kernel?
    occaMemory MM   = ((CeedVector_Occa*)inputfields[2]->vec->data)->d_array;//unused by the kernel?
    // Restrict
    ierr = CeedOperatorFieldGetElemRestriction(inputfields[2], &Erestrict);
    CeedChk(ierr);
    CeedTransposeMode lmode;
    ierr = CeedOperatorFieldGetLMode(inputfields[2], &lmode); CeedChk(ierr);
    ierr = CeedElemRestrictionApply(Erestrict, CEED_NOTRANSPOSE,
                                    lmode, invec, evecIn,
                                    request); CeedChk(ierr);
    occaMemory q    = ((CeedVector_Occa*)invec->data)->d_array;
    occaMemory Aq   = ((CeedVector_Occa*)impl->evecOut->data)->d_array;
    // Apply Basis + QFunction
    occaKernelRun(impl->kernel, nelem, ggeo, D, S, MM, lambda, q, Aq);
    // Prolong
    ierr = CeedOperatorFieldGetElemRestriction(outputfields[0], &Erestrict);
    CeedChk(ierr);
    ierr = CeedOperatorFieldGetLMode(outputfields[0], &lmode); CeedChk(ierr);
    ierr = CeedElemRestrictionApply(Erestrict, CEED_TRANSPOSE,
                                    lmode, impl->evecOut, outvec,
                                    request); CeedChk(ierr);
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

  if (op->qf->spec)
  {
    CeedOperator_libparanumal *impl;

    ierr = CeedCalloc(1, &impl); CeedChk(ierr);
    ierr = CeedOperatorSetData(op, (void*)&impl); CeedChk(ierr);

    impl->setupDone = false;

    ierr = CeedSetBackendFunction(ceed, "Operator", op, "Apply",
                                  CeedOperatorApply_libparanumal); CeedChk(ierr);
    ierr = CeedSetBackendFunction(ceed, "Operator", op, "Destroy",
                                  CeedOperatorDestroy_libparanumal); CeedChk(ierr);
  } else {
    Ceed delegate;
    ierr = CeedGetDelegate(ceed, &delegate); CeedChk(ierr);
    delegate->OperatorCreate(op);
  }
  return 0;
}
