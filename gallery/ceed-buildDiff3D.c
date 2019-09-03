// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
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

#ifndef __CUDACC__
#  include "ceed-backend.h"
#  include <string.h>
#endif

/**
  @brief Ceed QFunction for building the geometric data for the 3D diff operator
**/
CEED_QFUNCTION(buildDiff3D)(void *ctx, const CeedInt Q,
                            const CeedScalar *const *in, CeedScalar *const *out) {
  // At every quadrature point, compute qw/det(J).adj(J).adj(J)^T and store
  // the symmetric part of the result.

  // in[0] is Jacobians with shape [3, nc=3, Q]
  // in[1] is quadrature weights, size (Q)
  const CeedScalar *J = in[0], *qw = in[1];

  // out[0] is qdata, size (Q)
  CeedScalar *qd = out[0];

  // Quadrature point loop
  for (CeedInt i=0; i<Q; i++) {
    // J: 0 3 6   qd: 0 1 2
    //    1 4 7       1 3 4
    //    2 5 8       2 4 5
    const CeedScalar J11 = J[i+Q*0];
    const CeedScalar J21 = J[i+Q*1];
    const CeedScalar J31 = J[i+Q*2];
    const CeedScalar J12 = J[i+Q*3];
    const CeedScalar J22 = J[i+Q*4];
    const CeedScalar J32 = J[i+Q*5];
    const CeedScalar J13 = J[i+Q*6];
    const CeedScalar J23 = J[i+Q*7];
    const CeedScalar J33 = J[i+Q*8];
    const CeedScalar A11 = J22*J33 - J23*J32;
    const CeedScalar A12 = J13*J32 - J12*J33;
    const CeedScalar A13 = J12*J23 - J13*J22;
    const CeedScalar A21 = J23*J31 - J21*J33;
    const CeedScalar A22 = J11*J33 - J13*J31;
    const CeedScalar A23 = J13*J21 - J11*J23;
    const CeedScalar A31 = J21*J32 - J22*J31;
    const CeedScalar A32 = J12*J31 - J11*J32;
    const CeedScalar A33 = J11*J22 - J12*J21;
    const CeedScalar w = qw[i] / (J11*A11 + J21*A12 + J31*A13);
    qd[i+Q*0] = w * (A11*A11 + A12*A12 + A13*A13);
    qd[i+Q*1] = w * (A11*A21 + A12*A22 + A13*A23);
    qd[i+Q*2] = w * (A11*A31 + A12*A32 + A13*A33);
    qd[i+Q*3] = w * (A21*A21 + A22*A22 + A23*A23);
    qd[i+Q*4] = w * (A21*A31 + A22*A32 + A23*A33);
    qd[i+Q*5] = w * (A31*A31 + A32*A32 + A33*A33);
  }

  return 0;
}

/**
  @brief Set fields for Ceed QFunction building the geometric data for the 3D
           diff operator
**/
static int CeedQFunctionInit_BuildDiff3D(Ceed ceed, const char *name,
    CeedQFunction qf) {
  int ierr;

  // Check QFunction name
  if (strcmp(name, "buildDiff3D"))
    return CeedError(ceed, 1, "QFunction does not mach name: %s", name);

  // Add QFunction fields
  ierr = CeedQFunctionAddInput(qf, "dx", 3*3, CEED_EVAL_GRAD); CeedChk(ierr);
  ierr = CeedQFunctionAddInput(qf, "weights", 1, CEED_EVAL_WEIGHT);
  CeedChk(ierr);
  ierr = CeedQFunctionAddOutput(qf, "qdata", 1, CEED_EVAL_NONE); CeedChk(ierr);

  return 0;
}

/**
  @brief Register Ceed QFunction for building the geometric data for the 3D diff
           operator
**/
__attribute__((constructor))
static void Register(void) {
  CeedQFunctionRegister("buildDiff3D", buildDiff3D_loc, 1, buildDiff3D,
                        CeedQFunctionInit_BuildDiff3D);
}
