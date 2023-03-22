// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Element anisotropy tensor, as defined in 'Invariant data-driven subgrid stress modeling in the strain-rate eigenframe for large eddy simulation'
/// Prakash et al. 2022

#ifndef grid_anisotropy_tensor_h
#define grid_anisotropy_tensor_h

#include <ceed.h>

#include "utils.h"
#include "utils_eigensolver_jacobi.h"

// @brief Get Anisotropy tensor from xi_{i,j}
// @details A_ij = \Delta_{ij} / ||\Delta_ij||, \Delta_ij = (xi_{i,j})^(-1/2)
CEED_QFUNCTION_HELPER void AnisotropyTensor(const CeedScalar km_g_ij[6], CeedScalar A_ij[3][3], CeedScalar *delta, const CeedInt n_sweeps) {
  CeedScalar evals[3], evecs[3][3], evals_evecs[3][3] = {{0.}}, g_ij[3][3];
  CeedInt    work_vector[2];

  // Invert square root of metric tensor to get \Delta_ij
  KMUnpack(km_g_ij, g_ij);
  Diagonalize3(g_ij, evals, evecs, work_vector, SORT_DECREASING_EVALS, true, n_sweeps);
  for (int i = 0; i < 3; i++) evals[i] = 1 / sqrt(evals[i]);
  MatDiag3(evecs, evals, CEED_NOTRANSPOSE, evals_evecs);
  MatMat3(evecs, evals_evecs, CEED_TRANSPOSE, CEED_NOTRANSPOSE, A_ij);  // A_ij = E^T D E

  // Scale by delta to get anisotropy tensor
  *delta = sqrt(Dot3(evals, evals));
  ScaleN((CeedScalar *)A_ij, 1 / *delta, 9);
  // NOTE Need 2 factor to get physical element size (rather than projected onto [-1,1]^dim)
  // Should attempt to auto-determine this from the quadrature point coordinates in reference space
  *delta *= 2;
}

// @brief RHS for L^2 projection of anisotropic tensor and it's Frobenius norm
CEED_QFUNCTION(AnisotropyTensorProjection)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  CeedScalar(*v)[CEED_Q_VLA]            = (CeedScalar(*)[CEED_Q_VLA])out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar wdetJ      = q_data[0][i];
    const CeedScalar dXdx[3][3] = {
        {q_data[1][i], q_data[2][i], q_data[3][i]},
        {q_data[4][i], q_data[5][i], q_data[6][i]},
        {q_data[7][i], q_data[8][i], q_data[9][i]}
    };

    CeedScalar km_g_ij[6] = {0.}, A_ij[3][3] = {{0.}}, km_A_ij[6], delta;
    KMMetricTensor(dXdx, km_g_ij);
    AnisotropyTensor(km_g_ij, A_ij, &delta, 15);
    KMPack(A_ij, km_A_ij);

    for (CeedInt j = 0; j < 6; j++) v[j][i] = wdetJ * km_A_ij[j];
    v[6][i] = wdetJ * delta;
  }
  return 0;
}

#endif /* ifndef grid_anisotropy_tensor_h */
