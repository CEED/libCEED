// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Helper functions for computing stabilization terms of a newtonian simulation


#ifndef stabilization_h
#define stabilization_h

// *****************************************************************************
// Helper function for computing the stabilization terms
// *****************************************************************************
CEED_QFUNCTION_HELPER void Stabilization(NewtonianIdealGasContext gas, State s,
    CeedScalar dY[5], const CeedScalar x[3], CeedScalar stab[5][3]) {
  const CeedScalar dx_i[3] = {0};
  StateConservative dF[3];
  State ds = StateFromY_fwd(gas, s, dY, x, dx_i);
  FluxInviscid_fwd(gas, s, ds, dF);
  for (CeedInt j=0; j<3; j++) {
    CeedScalar dF_j[5];
    UnpackState_U(dF[j], dF_j);
    for (CeedInt k=0; k<5; k++)
      stab[k][j] += dF_j[k];
  }
}

// *****************************************************************************
// Helper function for computing the variation in primitive variables,
//   given Tau_d
// *****************************************************************************
CEED_QFUNCTION_HELPER void dYFromTau(CeedScalar Y[5], CeedScalar Tau_d[3],
                                     CeedScalar dY[5]) {
  dY[0] = Tau_d[0] * Y[0];
  dY[1] = Tau_d[1] * Y[1];
  dY[2] = Tau_d[1] * Y[2];
  dY[3] = Tau_d[1] * Y[3];
  dY[4] = Tau_d[2] * Y[4];
}

// *****************************************************************************
// Helper function for computing Tau elements (stabilization constant)
//   Model from:
//     PHASTA
//
//   Tau[i] = itau=0 which is diagonal-Shakib (3 values still but not spatial)
//
// Where NOT UPDATED YET
// *****************************************************************************
CEED_QFUNCTION_HELPER void Tau_diagPrim(NewtonianIdealGasContext gas, State s,
                                        const CeedScalar dXdx[3][3],
                                        const CeedScalar dt, CeedScalar Tau_d[3]) {
  // Context
  const CeedScalar Ctau_t = gas->Ctau_t;
  const CeedScalar Ctau_v = gas->Ctau_v;
  const CeedScalar Ctau_C = gas->Ctau_C;
  const CeedScalar Ctau_M = gas->Ctau_M;
  const CeedScalar Ctau_E = gas->Ctau_E;
  const CeedScalar cv = gas->cv;
  const CeedScalar mu = gas->mu;
  const CeedScalar u[3] = {s.Y.velocity[0], s.Y.velocity[1], s.Y.velocity[2]};
  const CeedScalar rho = s.U.density;

  CeedScalar gijd[6];
  CeedScalar tau;
  CeedScalar dts;
  CeedScalar fact;

  //*INDENT-OFF*
  gijd[0] =   dXdx[0][0] * dXdx[0][0]
            + dXdx[1][0] * dXdx[1][0]
            + dXdx[2][0] * dXdx[2][0];

  gijd[1] =   dXdx[0][0] * dXdx[0][1]
            + dXdx[1][0] * dXdx[1][1]
            + dXdx[2][0] * dXdx[2][1];

  gijd[2] =   dXdx[0][1] * dXdx[0][1]
            + dXdx[1][1] * dXdx[1][1]
            + dXdx[2][1] * dXdx[2][1];

  gijd[3] =   dXdx[0][0] * dXdx[0][2]
            + dXdx[1][0] * dXdx[1][2]
            + dXdx[2][0] * dXdx[2][2];

  gijd[4] =   dXdx[0][1] * dXdx[0][2]
            + dXdx[1][1] * dXdx[1][2]
            + dXdx[2][1] * dXdx[2][2];

  gijd[5] =   dXdx[0][2] * dXdx[0][2]
            + dXdx[1][2] * dXdx[1][2]
            + dXdx[2][2] * dXdx[2][2];
  //*INDENT-ON*

  dts = Ctau_t / dt ;

  tau = rho*rho*((4. * dts * dts)
                 + u[0] * ( u[0] * gijd[0] + 2. * ( u[1] * gijd[1] + u[2] * gijd[3]))
                 + u[1] * ( u[1] * gijd[2] + 2. *   u[2] * gijd[4])
                 + u[2] *   u[2] * gijd[5])
        + Ctau_v* mu * mu *
        (gijd[0]*gijd[0] + gijd[2]*gijd[2] + gijd[5]*gijd[5] +
         + 2. * (gijd[1]*gijd[1] + gijd[3]*gijd[3] + gijd[4]*gijd[4]));

  fact = sqrt(tau);

  Tau_d[0] = Ctau_C * fact / (rho*(gijd[0] + gijd[2] + gijd[5]))*0.125;

  Tau_d[1] = Ctau_M / fact;
  Tau_d[2] = Ctau_E / ( fact * cv );

  // consider putting back the way I initially had it  Ctau_E * Tau_d[1] /cv
  //  to avoid a division if the compiler is smart enough to see that cv IS
  // a constant that it could invert once for all elements
  // but in that case energy tau is scaled by the product of Ctau_E * Ctau_M
  // OR we could absorb cv into Ctau_E but this puts more burden on user to
  // know how to change constants with a change of fluid or units.  Same for
  // Ctau_v * mu * mu IF AND ONLY IF we don't add viscosity law =f(T)
}

// *****************************************************************************
// Helper function for computing flux Jacobian of Primitive variables
// *****************************************************************************
CEED_QFUNCTION_HELPER void computeFluxJacobian_NSp(CeedScalar dF[3][5][5],
    const CeedScalar rho, const CeedScalar u[3], const CeedScalar E,
    const CeedScalar Rd, const CeedScalar cv) {
  CeedScalar u_sq = u[0]*u[0] + u[1]*u[1] + u[2]*u[2]; // Velocity square
  // TODO Add in gravity's contribution

  CeedScalar T    = ( E / rho - u_sq / 2. ) / cv;
  CeedScalar drdT = -rho / T;
  CeedScalar drdP = 1. / ( Rd * T);
  CeedScalar etot =  E / rho ;
  CeedScalar e2p  = drdP * etot + 1. ;
  CeedScalar e3p  = ( E  + rho * Rd * T );
  CeedScalar e4p  = drdT * etot + rho * cv ;

  for (CeedInt i=0; i<3; i++) { // Jacobian matrices for 3 directions
    for (CeedInt j=0; j<3; j++) { // j counts F^{m_j}
      //   [row][col] of A_i
      dF[i][j+1][0] = drdP * u[i] * u[j] + ((i==j) ? 1. : 0.); // F^{{m_j} wrt p
      for (CeedInt k=0; k<3; k++) { // k counts the wrt vel_k
        dF[i][0][k+1]   =  ((i==k) ? rho  : 0.);   // F^c wrt u_k
        dF[i][j+1][k+1] = (((j==k) ? u[i] : 0.) +  // F^m_j wrt u_k
                           ((i==k) ? u[j] : 0.) ) * rho;
        dF[i][4][k+1]   = rho * u[i] * u[k]
                          + ((i==k) ? e3p  : 0.) ; // F^e wrt u_k
      }
      dF[i][j+1][4] = drdT * u[i] * u[j]; // F^{m_j} wrt T
    }
    dF[i][4][0] = u[i] * e2p; // F^e wrt p
    dF[i][4][4] = u[i] * e4p; // F^e wrt T
    dF[i][0][0] = u[i] * drdP; // F^c wrt p
    dF[i][0][4] = u[i] * drdT; // F^c wrt T
  }
}

// *****************************************************************************

#endif // stabilization_h
