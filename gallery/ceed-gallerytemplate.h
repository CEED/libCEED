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

/**
 This file is not compiled into libCEED. This file provides a template to
   build additional gallery QFunctions. Copy this file and the registeration/
   initalization .c file to a new folder in this directory and modify.
**/

/**
  @brief New Ceed QFunction
**/
CEED_QFUNCTION(GalleryTemplate)(void *ctx, const CeedInt Q,
                                const CeedScalar *const *in,
                                CeedScalar *const *out) {
  // in[0] is u, size (Q)
  // in[1] is quadrature data, size (Q)
  const CeedScalar *u = in[0], *qd = in[1];
  // out[0] is v, size (Q)
  CeedScalar *v = out[0];

  // Quadrature point loop
  for (CeedInt i=0; i<Q; i++) {
    v[i] = u[i] * qd[i];
  }

  return 0;
}
