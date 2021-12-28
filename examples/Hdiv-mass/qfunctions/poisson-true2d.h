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
/// Compute true solution of the H(div) example using PETSc

#ifndef TRUE_H
#define TRUE_H

#include <math.h>

// -----------------------------------------------------------------------------
// Compuet true solution
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupTrueSoln2D)(void *ctx, const CeedInt Q,
                                const CeedScalar *const *in,
                                CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*coords) = in[0],
                   (*dxdX)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[1];
  // Outputs
  CeedScalar (*true_soln_Hdiv) = out[0];
  // Quadrature Point Loop
  printf("True solution projected into H(div) space;Qfunction poisson-true2d.h\n");
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Setup, J = dx/dX
    const CeedScalar J[2][2] = {{dxdX[0][0][i], dxdX[1][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i]}};
    CeedScalar x = coords[i+0*Q], y = coords[i+1*Q];
    CeedScalar ue[2] = {x-y, x+y};
    CeedScalar nl[2] = {-J[1][1],J[0][1]};
    CeedScalar nr[2] = {J[1][1],-J[0][1]};
    CeedScalar nb[2] = {J[1][0],-J[0][0]};
    CeedScalar nt[2] = {-J[1][0],J[0][0]};
    CeedScalar ue_x, ue_y;
    if (i == 0){ // node 1
      ue_x = ue[0]*nl[0]+ue[1]*nl[1];
      ue_y = ue[0]*nb[0]+ue[1]*nb[1];
    }
    else if (i == 1){ // node 2
      ue_x = ue[0]*nr[0]+ue[1]*nr[1];
      ue_y = ue[0]*nb[0]+ue[1]*nb[1];
    }
    else if (i == 2){ // node 3
      ue_x = ue[0]*nl[0]+ue[1]*nl[1];
      ue_y = ue[0]*nt[0]+ue[1]*nt[1];
    }
    else if (i == 3){ // node 4
      ue_x = ue[0]*nr[0]+ue[1]*nr[1];
      ue_y = ue[0]*nt[0]+ue[1]*nt[1];
    }
    printf("ux %f\n",ue_x);
    printf("uy %f\n",ue_y);

    // True solution
    true_soln_Hdiv[i+0*Q] = ue_x;
    true_soln_Hdiv[i+1*Q] = ue_y;

  } // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif // End ERROR_H
