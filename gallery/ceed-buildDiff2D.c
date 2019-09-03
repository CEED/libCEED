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
  @brief Ceed QFunction for building the geometric data for the 2D diff operator
**/
CEED_QFUNCTION(buildDiff2D)(void *ctx, const CeedInt Q,
                            const CeedScalar *const *in, CeedScalar *const *out) {
  // At every quadrature point, compute qw/det(J).adj(J).adj(J)^T and store
  // the symmetric part of the result.

  // in[0] is Jacobians with shape [2, nc=2, Q]
  // in[1] is quadrature weights, size (Q)
  const CeedScalar *J = in[0], *qw = in[1];

  // out[0] is qdata, size (Q)
  CeedScalar *qd = out[0];

  // Quadrature point loop
  for (CeedInt i=0; i<Q; i++) {
    // J: 0 2   qd: 0 1   adj(J):  J22 -J12
    //    1 3       1 2           -J21  J11
    const CeedScalar J11 = J[i+Q*0];
    const CeedScalar J21 = J[i+Q*1];
    const CeedScalar J12 = J[i+Q*2];
    const CeedScalar J22 = J[i+Q*3];
    const CeedScalar w = qw[i] / (J11*J22 - J21*J12);
    qd[i+Q*0] =   w * (J12*J12 + J22*J22);
    qd[i+Q*1] = - w * (J11*J12 + J21*J22);
    qd[i+Q*2] =   w * (J11*J11 + J21*J21);
  }

  return 0;
}

/**
  @brief Set fields for Ceed QFunction building the geometric data for the 2D
           diff operator
**/
static int CeedQFunctionInit_BuildDiff2D(Ceed ceed, const char *name,
    CeedQFunction qf) {
  int ierr;

  // Check QFunction name
  if (strcmp(name, "buildDiff2D"))
    return CeedError(ceed, 1, "QFunction does not mach name: %s", name);

  // Add QFunction fields
  ierr = CeedQFunctionAddInput(qf, "dx", 2*2, CEED_EVAL_GRAD); CeedChk(ierr);
  ierr = CeedQFunctionAddInput(qf, "weights", 1, CEED_EVAL_WEIGHT);
  CeedChk(ierr);
  ierr = CeedQFunctionAddOutput(qf, "qdata", 1, CEED_EVAL_NONE); CeedChk(ierr);

  return 0;
}

/**
  @brief Register Ceed QFunction for building the geometric data for the 2D diff
           operator
**/
__attribute__((constructor))
static void Register(void) {
  CeedQFunctionRegister("buildDiff2D", buildDiff2D_loc, 1, buildDiff2D,
                        CeedQFunctionInit_BuildDiff2D);
}
