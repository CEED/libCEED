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
/// Test QR Factorization
/// \test Test QR Factorization
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedScalar qr[12] = {1, -1, 4, 1, 4, -2, 1, 4, 2, 1, -1, 0};
  CeedScalar tau[3];

  CeedInit(argv[1], &ceed);

  CeedQRFactorization(qr, tau, 4, 3);
  for (int i=0; i<12; i++) {
    if (qr[i] <= 1E-14 && qr[i] >= -1E-14) qr[i] = 0;
    fprintf(stdout, "%12.8f\n", qr[i]);
  }
  for (int i=0; i<3; i++) {
    if (tau[i] <= 1E-14 && qr[i] >= -1E-14) tau[i] = 0;
    fprintf(stdout, "%12.8f\n", tau[i]);
  }
  CeedDestroy(&ceed);
  return 0;
}
