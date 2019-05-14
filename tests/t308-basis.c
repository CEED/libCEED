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
/// Test that topological and geometric dimensions of basis match
/// \test Test that topological and geometric dimensions of basis match
#include <ceed.h>
#include <math.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedBasis b;
  CeedVector U, V;
  CeedInt Q = 8, P = 2, ncomp = 1, dim = 3,
          len = pow((double)(Q), dim);

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, len,   &U);
  CeedVectorCreate(ceed, len+1, &V);

  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncomp, P, Q, CEED_GAUSS, &b);

  CeedBasisApply(b, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, U, V);

  CeedBasisDestroy(&b);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedDestroy(&ceed);
  return 0;
}
