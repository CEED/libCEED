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

static void buildmats(CeedScalar *q_ref, CeedScalar *q_weight,
                      CeedScalar *interp,
                      CeedScalar *grad) {
  CeedInt P = 6, Q = 4;

  q_ref[0] = 0.2;
  q_ref[1] = 0.6;
  q_ref[2] = 1./3.;
  q_ref[3] = 0.2;
  q_ref[4] = 0.2;
  q_ref[5] = 0.2;
  q_ref[6] = 1./3.;
  q_ref[7] = 0.6;
  q_weight[0] = 25./96.;
  q_weight[1] = 25./96.;
  q_weight[2] = -27./96.;
  q_weight[3] = 25./96.;

  // Loop over quadrature points
  for (int i=0; i<Q; i++) {
    CeedScalar x1 = q_ref[0*Q+i], x2 = q_ref[1*Q+i];
    // Interp
    interp[i*P+0] =  2.*(x1+x2-1.)*(x1+x2-1./2.);
    interp[i*P+1] = -4.*x1*(x1+x2-1.);
    interp[i*P+2] =  2.*x1*(x1-1./2.);
    interp[i*P+3] = -4.*x2*(x1+x2-1.);
    interp[i*P+4] =  4.*x1*x2;
    interp[i*P+5] =  2.*x2*(x2-1./2.);
    // Grad
    grad[(i+0)*P+0] =  2.*(1.*(x1+x2-1./2.)+(x1+x2-1.)*1.);
    grad[(i+Q)*P+0] =  2.*(1.*(x1+x2-1./2.)+(x1+x2-1.)*1.);
    grad[(i+0)*P+1] = -4.*(1.*(x1+x2-1.)+x1*1.);
    grad[(i+Q)*P+1] = -4.*(x1*1.);
    grad[(i+0)*P+2] =  2.*(1.*(x1-1./2.)+x1*1.);
    grad[(i+Q)*P+2] =  2.*0.;
    grad[(i+0)*P+3] = -4.*(x2*1.);
    grad[(i+Q)*P+3] = -4.*(1.*(x1+x2-1.)+x2*1.);
    grad[(i+0)*P+4] =  4.*(1.*x2);
    grad[(i+Q)*P+4] =  4.*(x1*1.);
    grad[(i+0)*P+5] =  2.*0.;
    grad[(i+Q)*P+5] =  2.*(1.*(x2-1./2.)+x2*1.);
  }
}
