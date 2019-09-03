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
  @brief Ceed QFunction for building the geometric data for the 1D diff operator
**/
CEED_QFUNCTION(buildDiff1D)(void *ctx, const CeedInt Q,
                            const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0] is gradient u, size (Q)
  // in[1] is quadrature data, size (Q)
  const CeedScalar *du = in[0], *qd = in[1];

  // out[0] is output to multiply against gradient v, size (Q)
  CeedScalar *dv = out[0];

  // Quadrature point loop
  for (CeedInt i=0; i<Q; i++) {
    dv[i] = du[i] * qd[i];
  }

  return 0;
}

/**
  @brief Set fields for Ceed QFunction building the geometric data for the 1D
           diff operator
**/
static int CeedQFunctionInit_ApplyDiff1D(Ceed ceed, const char *name,
    CeedQFunction qf) {
  int ierr;

  // Check QFunction name
  if (strcmp(name, "buildDiff1D"))
    return CeedError(ceed, 1, "QFunction does not mach name: %s", name);

  // Add QFunction fields
  ierr = CeedQFunctionAddInput(qf, "du", 1, CEED_EVAL_GRAD); CeedChk(ierr);
  ierr = CeedQFunctionAddInput(qf, "qdata", 1*(1+1)/2, CEED_EVAL_NONE);
  CeedChk(ierr);
  ierr = CeedQFunctionAddOutput(qf, "dv", 1, CEED_EVAL_GRAD); CeedChk(ierr);

  return 0;
}

/**
  @brief Register Ceed QFunction for building the geometric data for the 1D diff
           operator
**/
__attribute__((constructor))
static void Register(void) {
  CeedQFunctionRegister("buildDiff1D", buildDiff1D_loc, 1, buildDiff1D,
                        CeedQFunctionInit_ApplyDiff1D);
}
