#ifndef Hdiv_hex_h
#define Hdiv_hex_h
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

// To see how the nodal basis is constructed visit:
// https://github.com/rezgarshakeri/H-div-Tests
int NodalHdivBasisHex(CeedScalar *x, CeedScalar *Bx, CeedScalar *By,
                      CeedScalar *Bz) {

  Bx[ 0] = 0.0625*x[0]*x[0] - 0.0625 ;
  By[ 0] = -0.0625*x[0]*x[1]*x[1] + 0.0625*x[0] + 0.0625*x[1]*x[1] - 0.0625 ;
  Bz[ 0] = 0.125*x[0]*x[1]*x[2] - 0.125*x[0]*x[1] - 0.125*x[0]*x[2] + 0.125*x[0] -
           0.125*x[1]*x[2] + 0.125*x[1] + 0.125*x[2] - 0.125 ;
  Bx[ 1] = 0.0625 - 0.0625*x[0]*x[0] ;
  By[ 1] = 0.0625*x[0]*x[1]*x[1] - 0.0625*x[0] + 0.0625*x[1]*x[1] - 0.0625 ;
  Bz[ 1] = -0.125*x[0]*x[1]*x[2] + 0.125*x[0]*x[1] + 0.125*x[0]*x[2] - 0.125*x[0]
           - 0.125*x[1]*x[2] + 0.125*x[1] + 0.125*x[2] - 0.125 ;
  Bx[ 2] = 0.0625*x[0]*x[0] - 0.0625 ;
  By[ 2] = 0.0625*x[0]*x[1]*x[1] - 0.0625*x[0] - 0.0625*x[1]*x[1] + 0.0625 ;
  Bz[ 2] = -0.125*x[0]*x[1]*x[2] + 0.125*x[0]*x[1] - 0.125*x[0]*x[2] + 0.125*x[0]
           + 0.125*x[1]*x[2] - 0.125*x[1] + 0.125*x[2] - 0.125 ;
  Bx[ 3] = 0.0625 - 0.0625*x[0]*x[0] ;
  By[ 3] = -0.0625*x[0]*x[1]*x[1] + 0.0625*x[0] - 0.0625*x[1]*x[1] + 0.0625 ;
  Bz[ 3] = 0.125*x[0]*x[1]*x[2] - 0.125*x[0]*x[1] + 0.125*x[0]*x[2] - 0.125*x[0] +
           0.125*x[1]*x[2] - 0.125*x[1] + 0.125*x[2] - 0.125 ;
  Bx[ 4] = 0.0625*x[0]*x[0] - 0.0625 ;
  By[ 4] = -0.0625*x[0]*x[1]*x[1] + 0.0625*x[0] + 0.0625*x[1]*x[1] - 0.0625 ;
  Bz[ 4] = 0.125*x[0]*x[1]*x[2] + 0.125*x[0]*x[1] - 0.125*x[0]*x[2] - 0.125*x[0] -
           0.125*x[1]*x[2] - 0.125*x[1] + 0.125*x[2] + 0.125 ;
  Bx[ 5] = 0.0625 - 0.0625*x[0]*x[0] ;
  By[ 5] = 0.0625*x[0]*x[1]*x[1] - 0.0625*x[0] + 0.0625*x[1]*x[1] - 0.0625 ;
  Bz[ 5] = -0.125*x[0]*x[1]*x[2] - 0.125*x[0]*x[1] + 0.125*x[0]*x[2] + 0.125*x[0]
           - 0.125*x[1]*x[2] - 0.125*x[1] + 0.125*x[2] + 0.125 ;
  Bx[ 6] = 0.0625*x[0]*x[0] - 0.0625 ;
  By[ 6] = 0.0625*x[0]*x[1]*x[1] - 0.0625*x[0] - 0.0625*x[1]*x[1] + 0.0625 ;
  Bz[ 6] = -0.125*x[0]*x[1]*x[2] - 0.125*x[0]*x[1] - 0.125*x[0]*x[2] - 0.125*x[0]
           + 0.125*x[1]*x[2] + 0.125*x[1] + 0.125*x[2] + 0.125 ;
  Bx[ 7] = 0.0625 - 0.0625*x[0]*x[0] ;
  By[ 7] = -0.0625*x[0]*x[1]*x[1] + 0.0625*x[0] - 0.0625*x[1]*x[1] + 0.0625 ;
  Bz[ 7] = 0.125*x[0]*x[1]*x[2] + 0.125*x[0]*x[1] + 0.125*x[0]*x[2] + 0.125*x[0] +
           0.125*x[1]*x[2] + 0.125*x[1] + 0.125*x[2] + 0.125 ;
  Bx[ 8] = 0.0625*x[0]*x[0]*x[2] - 0.0625*x[0]*x[0] - 0.0625*x[2] + 0.0625 ;
  By[ 8] = -0.125*x[0]*x[1]*x[2] + 0.125*x[0]*x[1] + 0.125*x[0]*x[2] - 0.125*x[0]
           - 0.125*x[1]*x[2] + 0.125*x[1] + 0.125*x[2] - 0.125 ;
  Bz[ 8] = 0.0625*x[2]*x[2] - 0.0625 ;
  Bx[ 9] = -0.0625*x[0]*x[0]*x[2] + 0.0625*x[0]*x[0] + 0.0625*x[2] - 0.0625 ;
  By[ 9] = 0.125*x[0]*x[1]*x[2] - 0.125*x[0]*x[1] - 0.125*x[0]*x[2] + 0.125*x[0] -
           0.125*x[1]*x[2] + 0.125*x[1] + 0.125*x[2] - 0.125 ;
  Bz[ 9] = 0.0625*x[2]*x[2] - 0.0625 ;
  Bx[10] = -0.0625*x[0]*x[0]*x[2] - 0.0625*x[0]*x[0] + 0.0625*x[2] + 0.0625 ;
  By[10] = 0.125*x[0]*x[1]*x[2] + 0.125*x[0]*x[1] - 0.125*x[0]*x[2] - 0.125*x[0] +
           0.125*x[1]*x[2] + 0.125*x[1] - 0.125*x[2] - 0.125 ;
  Bz[10] = 0.0625 - 0.0625*x[2]*x[2] ;
  Bx[11] = 0.0625*x[0]*x[0]*x[2] + 0.0625*x[0]*x[0] - 0.0625*x[2] - 0.0625 ;
  By[11] = -0.125*x[0]*x[1]*x[2] - 0.125*x[0]*x[1] + 0.125*x[0]*x[2] + 0.125*x[0]
           + 0.125*x[1]*x[2] + 0.125*x[1] - 0.125*x[2] - 0.125 ;
  Bz[11] = 0.0625 - 0.0625*x[2]*x[2] ;
  Bx[12] = 0.0625*x[0]*x[0]*x[2] - 0.0625*x[0]*x[0] - 0.0625*x[2] + 0.0625 ;
  By[12] = -0.125*x[0]*x[1]*x[2] + 0.125*x[0]*x[1] - 0.125*x[0]*x[2] + 0.125*x[0]
           - 0.125*x[1]*x[2] + 0.125*x[1] - 0.125*x[2] + 0.125 ;
  Bz[12] = 0.0625*x[2]*x[2] - 0.0625 ;
  Bx[13] = -0.0625*x[0]*x[0]*x[2] + 0.0625*x[0]*x[0] + 0.0625*x[2] - 0.0625 ;
  By[13] = 0.125*x[0]*x[1]*x[2] - 0.125*x[0]*x[1] + 0.125*x[0]*x[2] - 0.125*x[0] -
           0.125*x[1]*x[2] + 0.125*x[1] - 0.125*x[2] + 0.125 ;
  Bz[13] = 0.0625*x[2]*x[2] - 0.0625 ;
  Bx[14] = -0.0625*x[0]*x[0]*x[2] - 0.0625*x[0]*x[0] + 0.0625*x[2] + 0.0625 ;
  By[14] = 0.125*x[0]*x[1]*x[2] + 0.125*x[0]*x[1] + 0.125*x[0]*x[2] + 0.125*x[0] +
           0.125*x[1]*x[2] + 0.125*x[1] + 0.125*x[2] + 0.125 ;
  Bz[14] = 0.0625 - 0.0625*x[2]*x[2] ;
  Bx[15] = 0.0625*x[0]*x[0]*x[2] + 0.0625*x[0]*x[0] - 0.0625*x[2] - 0.0625 ;
  By[15] = -0.125*x[0]*x[1]*x[2] - 0.125*x[0]*x[1] - 0.125*x[0]*x[2] - 0.125*x[0]
           + 0.125*x[1]*x[2] + 0.125*x[1] + 0.125*x[2] + 0.125 ;
  Bz[15] = 0.0625 - 0.0625*x[2]*x[2] ;
  Bx[16] = 0.125*x[0]*x[1]*x[2] - 0.125*x[0]*x[1] - 0.125*x[0]*x[2] + 0.125*x[0] +
           0.125*x[1]*x[2] - 0.125*x[1] - 0.125*x[2] + 0.125 ;
  By[16] = 0.0625*x[1]*x[1] - 0.0625 ;
  Bz[16] = -0.0625*x[1]*x[2]*x[2] + 0.0625*x[1] + 0.0625*x[2]*x[2] - 0.0625 ;
  Bx[17] = -0.125*x[0]*x[1]*x[2] + 0.125*x[0]*x[1] - 0.125*x[0]*x[2] + 0.125*x[0]
           - 0.125*x[1]*x[2] + 0.125*x[1] - 0.125*x[2] + 0.125 ;
  By[17] = 0.0625 - 0.0625*x[1]*x[1] ;
  Bz[17] = 0.0625*x[1]*x[2]*x[2] - 0.0625*x[1] + 0.0625*x[2]*x[2] - 0.0625 ;
  Bx[18] = -0.125*x[0]*x[1]*x[2] - 0.125*x[0]*x[1] + 0.125*x[0]*x[2] + 0.125*x[0]
           - 0.125*x[1]*x[2] - 0.125*x[1] + 0.125*x[2] + 0.125 ;
  By[18] = 0.0625*x[1]*x[1] - 0.0625 ;
  Bz[18] = 0.0625*x[1]*x[2]*x[2] - 0.0625*x[1] - 0.0625*x[2]*x[2] + 0.0625 ;
  Bx[19] = 0.125*x[0]*x[1]*x[2] + 0.125*x[0]*x[1] + 0.125*x[0]*x[2] + 0.125*x[0] +
           0.125*x[1]*x[2] + 0.125*x[1] + 0.125*x[2] + 0.125 ;
  By[19] = 0.0625 - 0.0625*x[1]*x[1] ;
  Bz[19] = -0.0625*x[1]*x[2]*x[2] + 0.0625*x[1] - 0.0625*x[2]*x[2] + 0.0625 ;
  Bx[20] = 0.125*x[0]*x[1]*x[2] - 0.125*x[0]*x[1] - 0.125*x[0]*x[2] + 0.125*x[0] -
           0.125*x[1]*x[2] + 0.125*x[1] + 0.125*x[2] - 0.125 ;
  By[20] = 0.0625*x[1]*x[1] - 0.0625 ;
  Bz[20] = -0.0625*x[1]*x[2]*x[2] + 0.0625*x[1] + 0.0625*x[2]*x[2] - 0.0625 ;
  Bx[21] = -0.125*x[0]*x[1]*x[2] + 0.125*x[0]*x[1] - 0.125*x[0]*x[2] + 0.125*x[0]
           + 0.125*x[1]*x[2] - 0.125*x[1] + 0.125*x[2] - 0.125 ;
  By[21] = 0.0625 - 0.0625*x[1]*x[1] ;
  Bz[21] = 0.0625*x[1]*x[2]*x[2] - 0.0625*x[1] + 0.0625*x[2]*x[2] - 0.0625 ;
  Bx[22] = -0.125*x[0]*x[1]*x[2] - 0.125*x[0]*x[1] + 0.125*x[0]*x[2] + 0.125*x[0]
           + 0.125*x[1]*x[2] + 0.125*x[1] - 0.125*x[2] - 0.125 ;
  By[22] = 0.0625*x[1]*x[1] - 0.0625 ;
  Bz[22] = 0.0625*x[1]*x[2]*x[2] - 0.0625*x[1] - 0.0625*x[2]*x[2] + 0.0625 ;
  Bx[23] = 0.125*x[0]*x[1]*x[2] + 0.125*x[0]*x[1] + 0.125*x[0]*x[2] + 0.125*x[0] -
           0.125*x[1]*x[2] - 0.125*x[1] - 0.125*x[2] - 0.125 ;
  By[23] = 0.0625 - 0.0625*x[1]*x[1] ;
  Bz[23] = -0.0625*x[1]*x[2]*x[2] + 0.0625*x[1] - 0.0625*x[2]*x[2] + 0.0625 ;
  return 0;
}
static void HdivBasisHex(CeedInt Q, CeedScalar *q_ref, CeedScalar *q_weights,
                         CeedScalar *interp, CeedScalar *div, CeedQuadMode quad_mode) {

  // Get 1D quadrature on [-1,1]
  CeedScalar q_ref_1d[Q], q_weight_1d[Q];
  switch (quad_mode) {
  case CEED_GAUSS:
    CeedGaussQuadrature(Q, q_ref_1d, q_weight_1d);
    break;
  case CEED_GAUSS_LOBATTO:
    CeedLobattoQuadrature(Q, q_ref_1d, q_weight_1d);
    break;
  }

  // Divergence operator; Divergence of nodal basis for ref element
  CeedScalar D = 0.125;
  // Loop over quadrature points
  CeedScalar Bx[24], By[24], Bz[24];
  CeedScalar x[3];
  for (CeedInt k=0; k<Q; k++) {
    for (CeedInt i=0; i<Q; i++) {
      for (CeedInt j=0; j<Q; j++) {
        CeedInt k1 = Q*Q*k+Q*i+j;
        q_ref[k1 + 0*Q*Q] = q_ref_1d[j];
        q_ref[k1 + 1*Q*Q] = q_ref_1d[i];
        q_ref[k1 + 2*Q*Q] = q_ref_1d[k];
        q_weights[k1] = q_weight_1d[j]*q_weight_1d[i]*q_weight_1d[k];
        x[0] = q_ref_1d[j];
        x[1] = q_ref_1d[i];
        x[2] = q_ref_1d[k];
        NodalHdivBasisHex(x, Bx, By, Bz);
        for (CeedInt d=0; d<24; d++) {
          interp[k1*24+d] = Bx[d];
          interp[k1*24+d+24*Q*Q*Q] = By[d];
          interp[k1*24+d+48*Q*Q*Q] = Bz[d];
          div[k1*24+d] = D;
        }
      }
    }
  }
}

#endif // Hdiv_hex_h