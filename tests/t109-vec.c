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
/// Test creation, setting, reading, restoring, and destroying of a vector using CEED_MEM_DEVICE
/// \test Test creation, setting, reading, restoring, and destroying of a vector
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x;
  CeedVector y;
  CeedInt n;
  CeedScalar a[10];
  const CeedScalar *b, *c;

  CeedInit(argv[1], &ceed);
  n = 10;
  CeedVectorCreate(ceed, n, &x);
  CeedVectorCreate(ceed, n, &y);
  for (CeedInt i=0; i<n; i++) a[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);
  CeedVectorGetArrayRead(x, CEED_MEM_DEVICE, &b);
  CeedVectorSetArray(y, CEED_MEM_DEVICE, CEED_COPY_VALUES, (CeedScalar *)b);
  CeedVectorRestoreArrayRead(x, &b);
  CeedVectorGetArrayRead(y, CEED_MEM_HOST, &c);
  for (CeedInt i=0; i<n; i++) {
    if (c[i] != 10+i)
      printf("Error reading array c[%d] = %f",i,(double)c[i]);
  }
  CeedVectorRestoreArrayRead(y, &c);
  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedDestroy(&ceed);
  return 0;
}
