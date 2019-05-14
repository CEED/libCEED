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
/// Test CeedVector readers counter
/// \test Test CeedVector readers counter
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x;
  CeedInt n = 10;
  CeedScalar a[10];
  const CeedScalar *b;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, n, &x);
  for (CeedInt i=0; i<n; i++) a[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, a);

  CeedVectorGetArrayRead(x, CEED_MEM_HOST, &b);
  for (CeedInt i=0; i<n; i++) {
    if (b[i] != 10+i)
      printf("Error reading array b[%d] = %f",i,(double)b[i]);
  }

  // Try to set vector again (should fail)
  for (CeedInt i=0; i<n; i++) a[i] = 20 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);

  CeedVectorRestoreArrayRead(x, &b);

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
}
