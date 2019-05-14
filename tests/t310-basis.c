// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
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

/// @file
/// Test creation and destruction of a 2D Simplex non-tensor H1 basis
/// \test Test creation and distruction of a 2D Simplex non-tensor H1 basis
#include <ceed.h>
#include "t310-basis.h"

int main(int argc, char **argv) {
  Ceed ceed;
  const CeedInt P = 6, Q = 4, dim = 2;
  CeedBasis b;
  CeedScalar qref[dim*Q], qweight[Q];
  CeedScalar interp[P*Q], grad[dim*P*Q];

  buildmats(qref, qweight, interp, grad);

  CeedInit(argv[1], &ceed);
  CeedBasisCreateH1(ceed, CEED_TRIANGLE, 1, P, Q, interp, grad, qref, qweight,
                    &b);
  CeedBasisView(b, stdout);

  CeedBasisDestroy(&b);
  CeedDestroy(&ceed);
  return 0;
}
