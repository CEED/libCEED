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
#include "ceed-tmpl.h"

int CeedBasisCreateTensorH1_Tmpl(Ceed ceed, CeedInt dim, CeedInt P1d,
                                 CeedInt Q1d, const CeedScalar *interp1d,
                                 const CeedScalar *grad1d,
                                 const CeedScalar *qref1d,
                                 const CeedScalar *qweight1d,
                                 CeedBasis basis) {
  int ierr;
  Ceed_Tmpl *impl = ceed->data;
  Ceed ceedref = impl->ceedref;
  ierr = ceedref->BasisCreateTensorH1(ceed, dim, P1d, Q1d, interp1d,
                                      grad1d, qref1d, qweight1d, basis);
  CeedChk(ierr);

  return 0;
}

int CeedBasisCreateH1_Tmpl(Ceed ceed, CeedElemTopology topo, CeedInt dim,
                           CeedInt ndof, CeedInt nqpts,
                           const CeedScalar *interp,
                           const CeedScalar *grad,
                           const CeedScalar *qref,
                           const CeedScalar *qweight,
                           CeedBasis basis) {
  int ierr;
  Ceed_Tmpl *impl = ceed->data;
  Ceed ceedref = impl->ceedref;
  ierr = ceedref->BasisCreateH1(ceed, topo, dim, ndof, nqpts, interp,
                                grad, qref, qweight, basis);
  CeedChk(ierr);

  return 0;
}
