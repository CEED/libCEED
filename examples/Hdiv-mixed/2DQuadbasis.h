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

// Hdiv basis for quadrilateral element in 2D
// Local numbering is as follow (each edge has 2 vector dof)
//     b5     b7
//    2---------3
//  b4|         |b6
//    |         |
//  b0|         |b2
//    0---------1
//     b1     b3
// Bx[0-->7] = b0_x-->b7_x, By[0-->7] = b0_y-->b7_y
int HdivBasisQuad(CeedScalar *xhat, CeedScalar *Bx, CeedScalar *By) {
  Bx[0] = (-xhat[0]*xhat[1] + xhat[0] + xhat[1] - 1)*0.25;
  By[0] = (xhat[1]*xhat[1] - 1)*0.125;
  Bx[1] = (xhat[0]*xhat[0] - 1)*0.125;
  By[1] = (-xhat[0]*xhat[1] + xhat[0] + xhat[1] - 1)*0.25;
  Bx[2] = (-xhat[0]*xhat[1] + xhat[0] - xhat[1] + 1)*0.25;
  By[2] = (xhat[1]*xhat[1] - 1)*0.125;
  Bx[3] = (-xhat[0]*xhat[0] + 1)*0.125;
  By[3] = (xhat[0]*xhat[1] - xhat[0] + xhat[1] - 1)*0.25;
  Bx[4] = (xhat[0]*xhat[1] + xhat[0] - xhat[1] - 1)*0.25;
  By[4] = (-xhat[1]*xhat[1] + 1)*0.125;
  Bx[5] = (xhat[0]*xhat[0] - 1)*0.125;
  By[5] = (-xhat[0]*xhat[1] - xhat[0] + xhat[1] + 1)*0.25;
  Bx[6] = (xhat[0]*xhat[1] + xhat[0] + xhat[1] + 1)*0.25;
  By[6] = (-xhat[1]*xhat[1] + 1)*0.125;
  Bx[7] = (-xhat[0]*xhat[0] + 1)*0.125;
  By[7] = (xhat[0]*xhat[1] + xhat[0] + xhat[1] + 1)*0.25;
  return 0;
}

static void buildmats(CeedInt Q1d, CeedScalar *q_ref, CeedScalar *q_weights,
                      CeedScalar *interp, CeedScalar *div) {

  // Get 1D quadrature on [-1,1]
  CeedScalar q_ref_1d[Q1d], q_weight_1d[Q1d];
  CeedGaussQuadrature(Q1d, q_ref_1d, q_weight_1d);

  // Divergence operator; Divergence of nodal basis for ref element
  CeedScalar D[8] = {0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25};
  // Loop over quadrature points
  CeedScalar Bx[8], By[8];
  CeedScalar xhat[2];

  for (CeedInt i=0; i<Q1d; i++) {
    for (CeedInt j=0; j<Q1d; j++) {
      CeedInt k1 = Q1d*i+j;
      q_ref[k1] = q_ref_1d[j];
      q_ref[k1 + Q1d*Q1d] = q_ref_1d[i];
      q_weights[k1] = q_weight_1d[j]*q_weight_1d[i];
      xhat[0] = q_ref_1d[j];
      xhat[1] = q_ref_1d[i];
      HdivBasisQuad(xhat, Bx, By);
      for (CeedInt k=0; k<8; k++) {
        interp[k1*8+k] = Bx[k];
        interp[k1*8+k+8*Q1d*Q1d] = By[k];
        div[k1*8+k] = D[k];
      }
    }
  }
}


