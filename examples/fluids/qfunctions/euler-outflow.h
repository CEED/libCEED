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
/// Euler outflow BC

#ifndef euler_outflow_h
#define euler_outflow_h

#include <math.h>

#ifndef euler_context_struct
#define euler_context_struct
typedef struct EulerContext_ *EulerContext;
struct EulerContext_ {
  CeedScalar center[3];
  CeedScalar curr_time;
  CeedScalar vortex_strength;
  CeedScalar mean_velocity[3];
  int euler_test;
  bool implicit;
};
#endif

// *****************************************************************************
// This QFunction sets the outflow boundary conditions for
//   the traveling vortex problem.
//
//  Outflow BCs:
//    The validity of the weak form of the governing equations is
//      extended to the outflow.
//
// *****************************************************************************
CEED_QFUNCTION(Euler_Outflow)(void *ctx, CeedInt Q,
                              const CeedScalar *const *in,
                              CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*q_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*
  EulerContext context = (EulerContext)ctx;
  const bool implicit       = context->implicit;
  CeedScalar *mean_velocity = context->mean_velocity;

  const CeedScalar gamma = 1.4;

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Interp in
    const CeedScalar rho      =  q[0][i];
    const CeedScalar u[3]     = {q[1][i] / rho,
                                 q[2][i] / rho,
                                 q[3][i] / rho
                                };
    const CeedScalar E        =  q[4][i];

    // -- Interp-to-Interp q_data
    // For explicit mode, the surface integral is on the RHS of ODE q_dot = f(q).
    // For implicit mode, it gets pulled to the LHS of implicit ODE/DAE g(q_dot, q).
    // We can effect this by swapping the sign on this weight
    const CeedScalar wdetJb     =   (implicit ? -1. : 1.) * q_data_sur[0][i];
    // ---- Normal vectors
    const CeedScalar norm[3]    =   {q_data_sur[1][i],
                                     q_data_sur[2][i],
                                     q_data_sur[3][i]
                                    };

    // face_normal = Normal vector of the face
    const CeedScalar face_normal = norm[0]*mean_velocity[0] +
                                   norm[1]*mean_velocity[1] +
                                   norm[2]*mean_velocity[2];
    // The Physics
    // Zero v so all future terms can safely sum into it
    for (int j=0; j<5; j++) v[j][i] = 0;

    // Implementing in/outflow BCs
    if (face_normal > 0) { // outflow
      const CeedScalar E_kinetic = (u[0]*u[0] + u[1]*u[1]) / 2.;
      const CeedScalar P         = (E - E_kinetic * rho) * (gamma - 1.); // pressure
      const CeedScalar u_normal  = norm[0]*u[0] + norm[1]*u[1] +
                                   norm[2]*u[2]; // Normal velocity
      // The Physics
      // -- Density
      v[0][i] -= wdetJb * rho * u_normal;

      // -- Momentum
      for (int j=0; j<3; j++)
        v[j+1][i] -= wdetJb *(rho * u_normal * u[j] + norm[j] * P);

      // -- Total Energy Density
      v[4][i] -= wdetJb * u_normal * (E + P);
    }
  } // End Quadrature Point Loop
  return 0;
}

// *****************************************************************************

#endif // euler_outflow_h
