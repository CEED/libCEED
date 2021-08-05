// -----------------------------------------------------------------------------
// Nodal Basis (B=[Bx;By]), xhat is in reference element [-1,1]^2
// -----------------------------------------------------------------------------
//   B = [b1,b2,b3,b4,b5,b6,b7,b8],  size(2x8),
//    local numbering is as follow (each edge has 2 dof)
//     b6     b8
//    3---------4
//  b5|         |b7
//    |         |
//  b1|         |b3
//    1---------2
//     b2     b4
// Sience nodal basis are vector, we have 16 componenets
// For example B[0] = b1_x, B[1] = b1_y, and so on

CEED_QFUNCTION_HELPER int HdivBasisQuad(CeedScalar *xhat, CeedScalar *B) {
  B[ 0] = (-xhat[0]*xhat[1] + xhat[0] + xhat[1] - 1)*0.25;
  B[ 1] = (xhat[1]*xhat[1] - 1)*0.125;
  B[ 2] = (xhat[0]*xhat[0] - 1)*0.125;
  B[ 3] = (-xhat[0]*xhat[1] + xhat[0] + xhat[1] - 1)*0.25;
  B[ 4] = (-xhat[0]*xhat[1] + xhat[0] - xhat[1] + 1)*0.25;
  B[ 5] = (xhat[1]*xhat[1] - 1)*0.125;
  B[ 6] = (-xhat[0]*xhat[0] + 1)*0.125;
  B[ 7] = (xhat[0]*xhat[1] - xhat[0] + xhat[1] - 1)*0.25;
  B[ 8] = (xhat[0]*xhat[1] + xhat[0] - xhat[1] - 1)*0.25;
  B[ 9] = (-xhat[1]*xhat[1] + 1)*0.125;
  B[10] = (xhat[0]*xhat[0] - 1)*0.125;
  B[11] = (-xhat[0]*xhat[1] - xhat[0] + xhat[1] + 1)*0.25;
  B[12] = (xhat[0]*xhat[1] + xhat[0] + xhat[1] + 1)*0.25;
  B[13] = (-xhat[1]*xhat[1] + 1)*0.125;
  B[14] = (-xhat[0]*xhat[0] + 1)*0.125;
  B[15] = (xhat[0]*xhat[1] + xhat[0] + xhat[1] + 1)*0.25;
  return 0;
};

CEED_QFUNCTION(setup)(void *ctx, const CeedInt Q,
                      const CeedScalar *const *in,
                      CeedScalar *const *out) {

  const CeedScalar *J = in[0];

  CeedScalar *u = out[0];

  CeedInt Q1d = (int)sqrt(Q);
  //============ Compute Basis in Quadrature points =================
  CeedScalar Bq[16], B[16*Q1d*Q1d], q_ref_1d[Q1d], q_weight_1d[Q1d];
  CeedScalar xhat[2];

  CeedGaussQuadrature(Q1d, q_ref_1d, q_weight_1d);
  for (CeedInt i=0; i<Q1d; i++) {
    for (CeedInt j=0; j<Q1d; j++) {
      xhat[0] = q_ref_1d[j];
      xhat[1] = q_ref_1d[i];
      HdivBasisQuad(xhat, Bq);
      for (CeedInt k=0; k<16; k++) {
        CeedInt k1 = Q1d*i+j;
        B[k1+k*Q1d*Q1d] = Bq[k];
      }
    }
  }
  // Quadrature point loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    const CeedScalar J11 = J[i+Q*0];
    const CeedScalar J21 = J[i+Q*1];
    const CeedScalar J12 = J[i+Q*2];
    const CeedScalar J22 = J[i+Q*3];
    const CeedScalar detJ = (J11*J22 - J21*J12);

    u[i+Q*0 ] = (J11 * B[i+Q*0 ] + J12 * B[i+Q*1 ]) / detJ;
    u[i+Q*1 ] = (J21 * B[i+Q*0 ] + J22 * B[i+Q*1 ]) / detJ;
    u[i+Q*2 ] = (J11 * B[i+Q*2 ] + J12 * B[i+Q*3 ]) / detJ;
    u[i+Q*3 ] = (J21 * B[i+Q*2 ] + J22 * B[i+Q*3 ]) / detJ;
    u[i+Q*4 ] = (J11 * B[i+Q*4 ] + J12 * B[i+Q*5 ]) / detJ;
    u[i+Q*5 ] = (J21 * B[i+Q*4 ] + J22 * B[i+Q*5 ]) / detJ;
    u[i+Q*6 ] = (J11 * B[i+Q*6 ] + J12 * B[i+Q*7 ]) / detJ;
    u[i+Q*7 ] = (J21 * B[i+Q*6 ] + J22 * B[i+Q*7 ]) / detJ;
    u[i+Q*8 ] = (J11 * B[i+Q*8 ] + J12 * B[i+Q*9 ]) / detJ;
    u[i+Q*9 ] = (J21 * B[i+Q*8 ] + J22 * B[i+Q*9 ]) / detJ;
    u[i+Q*10] = (J11 * B[i+Q*10] + J12 * B[i+Q*11]) / detJ;
    u[i+Q*11] = (J21 * B[i+Q*10] + J22 * B[i+Q*11]) / detJ;
    u[i+Q*12] = (J11 * B[i+Q*12] + J12 * B[i+Q*13]) / detJ;
    u[i+Q*13] = (J21 * B[i+Q*12] + J22 * B[i+Q*13]) / detJ;
    u[i+Q*14] = (J11 * B[i+Q*14] + J12 * B[i+Q*15]) / detJ;
    u[i+Q*15] = (J21 * B[i+Q*14] + J22 * B[i+Q*15]) / detJ;
    //printf("%12.8f\n",u[i+Q*0 ]);
  }
  return 0;
}
