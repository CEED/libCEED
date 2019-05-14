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
/// Test creation, use, and destruction of a blocked element restriction
/// \test Test creation, use, and destruction of a blocked element restriction
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x, y;
  CeedInt ne = 8;
  CeedInt blksize = 5;
  CeedInt ind[2*ne];
  CeedScalar a[ne+1];
  CeedElemRestriction r;

  CeedInit(argv[1], &ceed);
  CeedVectorCreate(ceed, ne+1, &x);
  for (CeedInt i=0; i<ne+1; i++) a[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);

  for (CeedInt i=0; i<ne; i++) {
    ind[2*i+0] = i;
    ind[2*i+1] = i+1;
  }
  CeedElemRestrictionCreateBlocked(ceed, ne, 2, blksize, ne+1, 1, CEED_MEM_HOST,
                                   CEED_USE_POINTER, ind, &r);
  CeedVectorCreate(ceed, 2*blksize*2, &y);
  CeedVectorSetValue(y, 0); // Allocates array

  // NoTranspose
  CeedElemRestrictionApply(r, CEED_NOTRANSPOSE, CEED_NOTRANSPOSE, x, y,
                           CEED_REQUEST_IMMEDIATE);
  CeedVectorView(y, "%12.8f", stdout);

  // Transpose
  CeedVectorGetArray(x, CEED_MEM_HOST, (CeedScalar **)&a);
  for (CeedInt i=0; i<ne+1; i++) a[i] = 0;
  CeedVectorRestoreArray(x, (CeedScalar **)&a);
  CeedElemRestrictionApply(r, CEED_TRANSPOSE, CEED_NOTRANSPOSE, y, x,
                           CEED_REQUEST_IMMEDIATE);
  CeedVectorView(x, "%12.8f", stdout);

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedElemRestrictionDestroy(&r);
  CeedDestroy(&ceed);
  return 0;
}
