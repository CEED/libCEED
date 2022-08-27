// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef blasius_exact_h
#define blasius_exact_h

#include "utils.h"

typedef struct BlasiusContext_ *BlasiusContext;
struct BlasiusContext_ {
  bool       implicit; // !< Using implicit timesteping or not
  bool       weakT;    // !< flag to set Temperature weakly at inflow
  CeedScalar delta0;   // !< Boundary layer height at inflow
  CeedScalar Uinf;     // !< Velocity at boundary layer edge
  CeedScalar Tinf;     // !< Temperature at boundary layer edge
  CeedScalar T_wall;   // !< Temperature at the wall
  CeedScalar P0;       // !< Pressure at outflow
  CeedScalar theta0;   // !< Temperature at inflow
  CeedScalar x_inflow; // !< Location of inflow in x
  CeedScalar n_cheb;   // !< Number of Chebyshev terms
  CeedScalar *X;       // !< Chebyshev polynomial coordinate vector
  CeedScalar eta_max;  // !< Maximum eta in the domain
  CeedScalar *Tf_cheb; // !< Chebyshev coefficient for f
  CeedScalar *Th_cheb; // !< Chebyshev coefficient for h
  struct NewtonianIdealGasContext_ newtonian_ctx;
};

// *****************************************************************************
//   Helper function to evaluate Chebyshev polynomials with a set of coefficients
//   with all their derivatives represented as a recurrence table
// *****************************************************************************
CEED_QFUNCTION_HELPER void ChebyshevEval(int N, const double *Tf, double x,
    double eta_max, double *f) {
  double dX_deta   = 2 / eta_max;
  double table[4][3] = {
    // Chebyshev polynomials T_0, T_1, T_2 of the first kind in (-1,1)
    {1, x, 2*x *x - 1}, {0, 1, 4*x}, {0, 0, 4}, {0, 0, 0}
  };
  for (int i=0; i<4; i++) {
    // i-th derivative of f
    f[i] = table[i][0] * Tf[0] + table[i][1] * Tf[1] + table[i][2] * Tf[2];
  }
  for (int i=3; i<N; i++) {
    // T_n(x) = 2xT_{n-1}(x) - T_{n-2}(x)
    table[0][i%3] = 2 * x * table[0][(i-1) % 3] - table[0][(i-2)%3];
    // Differentiate Chebyshev polynomials with the recurrence relation
    for (int j=1; j<4; j++) {
      // T'_{n}(x)/n = 2T_{n-1}(x) + T'_{n-2}(x)/n-2
      table[j][i%3] = i * (2 * table[j-1][(i-1) % 3] + table[j][(i-2)%3] / (i-2));
    }
    for (int j=0; j<4; j++) {
      f[j] += table[j][i%3] * Tf[i];
    }
  }
  for (int i=1; i<4; i++) {
    // Transform derivatives from Chebyshev [-1, 1] to [0, eta_max].
    for (int j=0; j<i; j++) f[i] *= dX_deta;
  }
}

// *****************************************************************************
PetscErrorCode CompressibleBlasiusResidual(SNES snes, Vec X, Vec R, void *ctx) {
  const BlasiusContext blasius = (BlasiusContext)ctx;
  const PetscScalar *Tf, *Th;  // Chebyshev coefficients
  PetscScalar       *r, f[4], h[4];
  PetscInt          N = blasius->n_cheb;
  PetscScalar Ma = Mach(&blasius->newtonian_ctx, blasius->Tinf, blasius->Uinf),
              Pr = Prandtl(&blasius->newtonian_ctx),
              gamma = HeatCapacityRatio(&blasius->newtonian_ctx);
  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(X, &Tf));
  Th = Tf + N;
  PetscCall(VecGetArray(R, &r));

  // Left boundary conditions f = f' = 0
  ChebyshevEval(N, Tf, -1., blasius->eta_max, f);
  r[0] = f[0];
  r[1] = f[1];

  // f - right end boundary condition
  ChebyshevEval(N, Tf, 1., blasius->eta_max, f);
  r[2] = f[1]  - 1.;

  for (int i=0; i<N-3; i++) {
    ChebyshevEval(N, Tf, blasius->X[i], blasius->eta_max, f);
    r[3+i] = 2*f[3] + f[2] * f[0];
    ChebyshevEval(N-1, Th, blasius->X[i], blasius->eta_max, h);
    r[N+2+i] = h[2] + Pr * f[0] * h[1] +
               Pr * (gamma - 1) * PetscSqr(Ma * f[2]);
  }

  // h - left end boundary condition
  ChebyshevEval(N-1, Th, -1., blasius->eta_max, h);
  r[N] = h[0] - blasius->T_wall / blasius->Tinf;

  // h - right end boundary condition
  ChebyshevEval(N-1, Th, 1., blasius->eta_max, h);
  r[N+1] = h[0] - 1.;

  // Restore vectors
  PetscCall(VecRestoreArrayRead(X, &Tf));
  PetscCall(VecRestoreArray(R, &r));
  PetscFunctionReturn(0);
}

// *****************************************************************************
PetscErrorCode ComputeChebyshevCoefficients(BlasiusContext blasius) {
  SNES      snes;
  Vec       sol, res;
  PetscReal *w;
  PetscInt  N = blasius->n_cheb;
  const PetscScalar *cheb_coefs;
  PetscFunctionBegin;
  PetscCall(PetscMalloc2(N-3, &blasius->X, N-3, &w));
  PetscCall(PetscDTGaussQuadrature(N-3, -1., 1., blasius->X, w));
  PetscCall(SNESCreate(PETSC_COMM_SELF, &snes));
  PetscCall(VecCreate(PETSC_COMM_SELF, &sol));
  PetscCall(VecSetSizes(sol, PETSC_DECIDE, 2*N-1));
  PetscCall(VecSetFromOptions(sol));
  PetscCall(VecDuplicate(sol, &res));
  PetscCall(SNESSetFunction(snes, res, CompressibleBlasiusResidual, blasius));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(SNESSolve(snes, NULL, sol));
  PetscCall(VecGetArrayRead(sol, &cheb_coefs));
  for (int i=0; i<N; i++) blasius->Tf_cheb[i] = cheb_coefs[i];
  for (int i=0; i<N-1; i++) blasius->Th_cheb[i] = cheb_coefs[i+N];
  PetscCall(PetscFree2(blasius->X, w));
  PetscCall(VecDestroy(&sol));
  PetscCall(VecDestroy(&res));
  PetscCall(SNESDestroy(&snes));
  PetscFunctionReturn(0);
}

// *****************************************************************************
#endif // blasius_exact_h
