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
/// Test polynomial interpolation in 1D
/// \test Test polynomial interpolation in 1D
#include <ceed.h>
#include <math.h>

#define ALEN(a) (sizeof(a) / sizeof((a)[0]))

static CeedScalar PolyEval(CeedScalar x, CeedInt n, const CeedScalar *p) {
  CeedScalar y = p[n-1];
  for (CeedInt i=n-2; i>=0; i--) y = y*x + p[i];
  return y;
}

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector X, Xq, U, Uq;
  CeedBasis bxl, bul, bxg, bug;
  CeedInt Q = 6;
  const CeedScalar p[6] = {1, 2, 3, 4, 5, 6}; // 1 + 2x + 3x^2 + ...
  const CeedScalar *xq, *uuq;
  CeedScalar x[2], uq[Q];

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, 2, &X);
  CeedVectorCreate(ceed, Q, &Xq);
  CeedVectorSetValue(Xq, 0);
  CeedVectorCreate(ceed, Q, &U);
  CeedVectorSetValue(U, 0);
  CeedVectorCreate(ceed, Q, &Uq);

  CeedBasisCreateTensorH1Lagrange(ceed, 1,  1, 2, Q, CEED_GAUSS_LOBATTO, &bxl);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, Q, Q, CEED_GAUSS_LOBATTO, &bul);

  for (int i = 0; i < 2; i++) x[i] = CeedIntPow(-1, i+1);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, (CeedScalar *)&x);

  CeedBasisApply(bxl, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, X, Xq);

  CeedVectorGetArrayRead(Xq, CEED_MEM_HOST, &xq);
  for (CeedInt i=0; i<Q; i++) uq[i] = PolyEval(xq[i], ALEN(p), p);
  CeedVectorRestoreArrayRead(Xq, &xq);
  CeedVectorSetArray(Uq, CEED_MEM_HOST, CEED_USE_POINTER, (CeedScalar *)&uq);

  // This operation is the identity because the quadrature is collocated
  CeedBasisApply(bul, 1, CEED_TRANSPOSE, CEED_EVAL_INTERP, Uq, U);

  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q, CEED_GAUSS, &bxg);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, Q, Q, CEED_GAUSS, &bug);

  CeedBasisApply(bxg, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, X, Xq);
  CeedBasisApply(bug, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, U, Uq);

  CeedVectorGetArrayRead(Xq, CEED_MEM_HOST, &xq);
  CeedVectorGetArrayRead(Uq, CEED_MEM_HOST, &uuq);
  for (CeedInt i=0; i<Q; i++) {
    CeedScalar px = PolyEval(xq[i], ALEN(p), p);
    if ((fabs(uuq[i] - px) > 1e-14)) {
      printf("%f != %f=p(%f)\n", uuq[i], px, xq[i]);
    }
  }
  CeedVectorRestoreArrayRead(Xq, &xq);
  CeedVectorRestoreArrayRead(Uq, &uuq);

  CeedVectorDestroy(&X);
  CeedVectorDestroy(&Xq);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&Uq);
  CeedBasisDestroy(&bxl);
  CeedBasisDestroy(&bul);
  CeedBasisDestroy(&bxg);
  CeedBasisDestroy(&bug);
  CeedDestroy(&ceed);
  return 0;
}
