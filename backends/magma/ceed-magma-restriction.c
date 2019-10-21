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

#include "ceed-magma.h"

static int CeedElemRestrictionApply_Magma(CeedElemRestriction r,
                                        CeedTransposeMode tmode,
                                        CeedTransposeMode lmode, CeedVector u,
                                        CeedVector v, CeedRequest *request) {
//***
// Implementation here 
//***
//
}

int CeedElemRestrictionApplyBlock_Magma(CeedElemRestriction r,
                                       CeedInt block, CeedTransposeMode tmode,
                                       CeedTransposeMode lmode, CeedVector u,
                                       CeedVector v, CeedRequest *request) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
    // LCOV_EXCL_START
  return CeedError(ceed, 1, "Backend does not implement blocked restrictions");
  // LCOV_EXCL_STOP
}

static int CeedElemRestrictionDestroy_Magma(CeedElemRestriction r) {
  int ierr;
  CeedElemRestriction_Magma *impl;
  ierr = CeedElemRestrictionGetData(r, (void *)&impl); CeedChk(ierr);

  ierr = CeedFree(&impl->indices_allocated); CeedChk(ierr);
  ierr = CeedFree(&impl); CeedChk(ierr);
  return 0;
}

int CeedElemRestrictionCreate_Magma(CeedMemType mtype, CeedCopyMode cmode,
                                  const CeedInt *indices, CeedElemRestriction r) {
  int ierr;
  CeedElemRestriction_Magma *impl;
  CeedInt elemsize, nelem;
  ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChk(ierr);
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);

//***
// Do more setup stuff...
// ***


  ierr = CeedElemRestrictionSetData(r, (void *)&impl); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Apply",
                                CeedElemRestrictionApply_Magma); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "ApplyBlock",
                                CeedElemRestrictionApplyBlock_Magma);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Destroy",
                                CeedElemRestrictionDestroy_Magma); CeedChk(ierr);
  return 0;
}

int CeedElemRestrictionCreateBlocked_Magma(const CeedMemType mtype,
    const CeedCopyMode cmode,
    const CeedInt *indices,
    const CeedElemRestriction r) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
    // LCOV_EXCL_START
  return CeedError(ceed, 1, "Backend does not implement blocked restrictions");
  // LCOV_EXCL_STOP
}
