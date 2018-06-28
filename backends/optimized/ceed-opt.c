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
#include <string.h>
#include "ceed-opt.h"

static int CeedDestroy_Opt(Ceed ceed) {
  int ierr;
  Ceed_Opt *impl = ceed->data;
  Ceed ceedref = impl->ceedref;
  ierr = CeedFree(&ceedref); CeedChk(ierr);
  ierr = CeedFree(&impl); CeedChk(ierr);
  return 0;
}

static int CeedInit_Opt(const char *resource, Ceed ceed) {
  if (strcmp(resource, "/cpu/self")
      && strcmp(resource, "/cpu/self/opt"))
    return CeedError(ceed, 1, "Opt backend cannot use resource: %s", resource);

  int ierr;
  Ceed_Opt *impl;
  Ceed ceedref;

  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  CeedInit("/cpu/self/ref", &ceedref);
  ceed->data = impl;
  impl->ceedref = ceedref;

  ceed->VecCreate = CeedVectorCreate_Opt;
  ceed->BasisCreateTensorH1 = CeedBasisCreateTensorH1_Opt;
  ceed->ElemRestrictionCreate = CeedElemRestrictionCreate_Opt;
  ceed->QFunctionCreate = CeedQFunctionCreate_Opt;
  ceed->OperatorCreate = CeedOperatorCreate_Opt;
  ceed->Destroy = CeedDestroy_Opt;

  return 0;
}

__attribute__((constructor))
static void Register(void) {
  CeedRegister("/cpu/self/opt", CeedInit_Opt, 10);
}
