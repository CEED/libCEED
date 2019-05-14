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
/// Test Collocated Grad calculated matches basis with Lobatto points
/// \test Test Collocated Grad calculated matches basis with Lobatto points
#include <ceed.h>
#include <ceed-backend.h>
#include <math.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedInt P=4, Q=4;
  CeedScalar colograd1d[Q*Q], colograd1d2[(P+2)*(P+2)];
  CeedBasis b;

  CeedInit(argv[1], &ceed);

  // Already collocated, GetCollocatedGrad will return grad1d
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, P, Q, CEED_GAUSS_LOBATTO, &b);
  CeedBasisGetCollocatedGrad(b, colograd1d);

  CeedBasisView(b, stdout);
  for (int i=0; i<Q; i++) {
    fprintf(stdout, "%12s[%d]:", "colograd1d", i);
    for (int j=0; j<Q; j++) {
      if (fabs(colograd1d[j+Q*i]) <= 1E-14) colograd1d[j+Q*i] = 0;
      fprintf(stdout, "\t% 12.8f", colograd1d[j+Q*i]);
    }
    fprintf(stdout, "\n");
  }
  CeedBasisDestroy(&b);

  // Q = P, not already collocated
  CeedBasisCreateTensorH1Lagrange(ceed, 1,  1, P, Q, CEED_GAUSS, &b);
  CeedBasisGetCollocatedGrad(b, colograd1d);

  CeedBasisView(b, stdout);
  for (int i=0; i<Q; i++) {
    fprintf(stdout, "%12s[%d]:", "colograd1d", i);
    for (int j=0; j<Q; j++) {
      if (fabs(colograd1d[j+Q*i]) <= 1E-14) colograd1d[j+Q*i] = 0;
      fprintf(stdout, "\t% 12.8f", colograd1d[j+Q*i]);
    }
    fprintf(stdout, "\n");
  }
  CeedBasisDestroy(&b);

  // Q = P + 2, not already collocated
  CeedBasisCreateTensorH1Lagrange(ceed, 1,  1, P, P+2, CEED_GAUSS, &b);
  CeedBasisGetCollocatedGrad(b, colograd1d2);

  CeedBasisView(b, stdout);
  for (int i=0; i<P+2; i++) {
    fprintf(stdout, "%12s[%d]:", "colograd1d", i);
    for (int j=0; j<P+2; j++) {
      if (fabs(colograd1d2[j+(P+2)*i]) <= 1E-14) colograd1d2[j+(P+2)*i] = 0;
      fprintf(stdout, "\t% 12.8f", colograd1d2[j+(P+2)*i]);
    }
    fprintf(stdout, "\n");
  }
  CeedBasisDestroy(&b);

  CeedDestroy(&ceed);
  return 0;
}
