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
/// Test CeedVectorSetValue
/// \test Test CeedVectorSetValue
#include <ceed.h>

static int CheckValues(Ceed ceed, CeedVector x, CeedScalar value) {
  const CeedScalar *b;
  CeedInt n;
  CeedVectorGetLength(x, &n);
  CeedVectorGetArrayRead(x, CEED_MEM_HOST, &b);
  for (CeedInt i=0; i<n; i++) {
    if (b[i] != value)
      printf("Error reading array b[%d] = %f",i,
             (double)b[i]);
  }
  CeedVectorRestoreArrayRead(x, &b);
  return 0;
}

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x;
  CeedInt n;
  CeedScalar a[10];
  const CeedScalar *b;

  CeedInit(argv[1], &ceed);
  n = 10;
  CeedVectorCreate(ceed, n, &x);
  for (CeedInt i=0; i<n; i++) a[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);
  CeedVectorGetArrayRead(x, CEED_MEM_HOST, &b);
  for (CeedInt i=0; i<n; i++) {
    if (b[i] != 10+i)
      printf("Error reading array b[%d] = %f",i,
             (double)b[i]);
  }
  CeedVectorRestoreArrayRead(x, &b);

  CeedVectorSetValue(x, 3.0);
  CheckValues(ceed, x, 3.0);
  CeedVectorDestroy(&x);

  CeedVectorCreate(ceed, n, &x);
  CeedVectorSetValue(x, 5.0); // Set value before setting or getting the array
  CheckValues(ceed, x, 5.0);
  CeedVectorDestroy(&x);

  CeedDestroy(&ceed);
  return 0;
}
