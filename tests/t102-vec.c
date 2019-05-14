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
/// Test CeedVectorGetArrayRead state counter
/// \test Test CeedVectorGetArrayRead state counter
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x;
  CeedInt n;
  const CeedScalar *a, *b;

  CeedInit(argv[1], &ceed);
  n = 10;
  CeedVectorCreate(ceed, n, &x);
  // Two read accesses should not generate an error
  CeedVectorGetArrayRead(x, CEED_MEM_HOST, &a);
  CeedVectorGetArrayRead(x, CEED_MEM_HOST, &b);

  CeedVectorRestoreArrayRead(x, &a);
  CeedVectorRestoreArrayRead(x, &b);

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
}
