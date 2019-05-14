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
/// Test setting one vector from array of another vector
/// \test Test setting one vector from array of another vector
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector X, Y;
  CeedInt n;
  CeedScalar a[10];
  CeedScalar *x;
  const CeedScalar *y;

  CeedInit(argv[1], &ceed);
  n = 10;
  CeedVectorCreate(ceed, n, &X);
  CeedVectorCreate(ceed, n, &Y);

  for (CeedInt i=0; i<n; i++) a[i] = 10 + i;
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, a);

  CeedVectorGetArray(X, CEED_MEM_HOST, &x);
  CeedVectorSetArray(Y, CEED_MEM_HOST, CEED_COPY_VALUES, x);
  CeedVectorRestoreArray(X, &x);

  CeedVectorGetArrayRead(Y, CEED_MEM_HOST, &y);
  for (CeedInt i=0; i<n; i++) {
    if (y[i] != 10+i)
      printf("Error reading array y[%d] = %f",i,(double)y[i]);
  }
  CeedVectorRestoreArrayRead(Y, &y);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&Y);
  CeedDestroy(&ceed);
  return 0;
}
