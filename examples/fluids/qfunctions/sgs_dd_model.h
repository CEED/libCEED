// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Structs and helper functions for data-driven subgrid-stress modeling
/// See 'Invariant data-driven subgrid stress modeling in the strain-rate eigenframe for large eddy simulation' 2022 and 'S-frame discrepancy
/// correction models for data-informed Reynolds stress closure' 2022

#ifndef sgs_dd_model_h
#define sgs_dd_model_h

#include <ceed.h>

#include "newtonian_state.h"
#include "utils.h"
#include "utils_eigensolver_jacobi.h"

typedef struct SGS_DD_ModelContext_ *SGS_DDModelContext;
struct SGS_DD_ModelContext_ {
  CeedInt    num_inputs, num_outputs;
  CeedInt    num_layers;
  CeedInt    num_neurons;
  CeedScalar alpha;

  struct {
    size_t bias1, bias2;
    size_t weight1, weight2;
    size_t out_scaling;
  } offsets;
  size_t     total_bytes;
  CeedScalar data[1];
};

CEED_QFUNCTION_HELPER void LeakyReLU(CeedScalar *x, const CeedScalar alpha, const CeedInt N) {
  for (CeedInt i = 0; i < N; i++) x[i] *= (x[i] < 0 ? alpha : 1.);
}

CEED_QFUNCTION_HELPER void DataDrivenInference(const CeedScalar *inputs, CeedScalar *outputs, SGS_DDModelContext sgsdd_ctx) {
  const CeedInt     num_neurons = sgsdd_ctx->num_neurons;
  const CeedInt     num_inputs  = sgsdd_ctx->num_inputs;
  const CeedInt     num_outputs = sgsdd_ctx->num_outputs;
  const CeedScalar  alpha       = sgsdd_ctx->alpha;
  const CeedScalar *bias1       = &sgsdd_ctx->data[sgsdd_ctx->offsets.bias1];
  const CeedScalar *bias2       = &sgsdd_ctx->data[sgsdd_ctx->offsets.bias2];
  const CeedScalar *weight1     = &sgsdd_ctx->data[sgsdd_ctx->offsets.weight1];
  const CeedScalar *weight2     = &sgsdd_ctx->data[sgsdd_ctx->offsets.weight2];
  CeedScalar        V[20]       = {0.};

  CopyN(bias1, V, num_neurons);
  MatVecNM(weight1, inputs, num_neurons, num_inputs, NO_TRANSPOSE, V);
  LeakyReLU(V, alpha, num_neurons);
  CopyN(bias2, outputs, num_outputs);
  MatVecNM(weight2, V, num_outputs, num_neurons, NO_TRANSPOSE, outputs);
}

CEED_QFUNCTION_HELPER CeedScalar VelocityGradientMagnitude(const CeedScalar strain_sframe[3], const CeedScalar vorticity_sframe[3]) {
  return sqrt(Dot3(strain_sframe, strain_sframe) + 0.5 * Dot3(vorticity_sframe, vorticity_sframe));
};

// @brief Denormalize outputs using min-max (de-)normalization
CEED_QFUNCTION_HELPER void DenormalizeDDOutputs(CeedScalar output[6], const CeedScalar new_bounds[6][2], const CeedScalar old_bounds[6][2]) {
  CeedScalar bounds_ratio;
  for (int i = 0; i < 6; i++) {
    bounds_ratio = (new_bounds[i][1] - new_bounds[i][0]) / (old_bounds[i][1] - old_bounds[i][0]);
    output[i]    = bounds_ratio * (output[i] - old_bounds[i][1]) + new_bounds[i][1];
  }
}

// @brief Change the order of basis vectors so that they align with vector and obey right-hand rule
// @details The e_1 and e_3 basis vectors are the closest aligned to the vector. The e_2 is set via  e_3 x e_1
// The basis vectors are assumed to form the rows of the basis matrix.
CEED_QFUNCTION_HELPER void OrientBasisWithVector(CeedScalar basis[3][3], const CeedScalar vector[3]) {
  CeedScalar alignment[3] = {0.}, cross[3];

  MatVec3(basis, vector, NO_TRANSPOSE, alignment);

  if (alignment[0] < 0) ScaleN(basis[0], -1, 3);
  if (alignment[2] < 0) ScaleN(basis[2], -1, 3);

  Cross3(basis[2], basis[0], cross);
  CeedScalar basis_1_orientation = Dot3(cross, basis[1]);
  if (basis_1_orientation < 0) ScaleN(basis[1], -1, 3);
}

// @brief Get Anisotropy tensor from xi_{i,j}
// @details A_ij = \Delta_{ij} / ||\Delta_ij||, \Delta_ij = (xi_{i,j})^(-1/2)
CEED_QFUNCTION_HELPER void AnisotropyTensor(const CeedScalar km_g_ij[6], CeedScalar A_ij[3][3], CeedScalar *delta, const CeedInt n_sweeps) {
  CeedScalar evals[3], evecs[3][3], evals_evecs[3][3] = {{0.}}, g_ij[3][3];
  CeedInt    work_vector[2];

  // Invert square root of metric tensor to get \Delta_ij
  KMUnpack(km_g_ij, g_ij);
  Diagonalize3(g_ij, evals, evecs, work_vector, SORT_DECREASING_EVALS, true, n_sweeps);
  for (int i = 0; i < 3; i++) evals[i] = 1 / sqrt(evals[i]);
  MatDiag3(evecs, evals, NO_TRANSPOSE, evals_evecs);
  MatMat3(evecs, evals_evecs, TRANSPOSE, NO_TRANSPOSE, A_ij);  // A_ij = E^T D E

  // Scale by delta to get anisotropy tensor
  *delta = sqrt(Dot3(evals, evals));
  ScaleN((CeedScalar *)A_ij, 1 / *delta, 9);
  // NOTE Need 2 factor to get physical element size (rather than projected onto [-1,1]^dim)
  *delta *= 2;
}

CEED_QFUNCTION_HELPER void ComputeSGS_DDAnisotropic(const CeedScalar grad_velo_aniso[3][3], const CeedScalar km_A_ij[6], const CeedScalar delta,
                                                    const CeedScalar viscosity, CeedScalar kmsgs_stress[6], SGS_DDModelContext sgsdd_ctx) {
  CeedScalar strain_sframe[3] = {0.}, vorticity_sframe[3] = {0.}, eigenvectors[3][3];
  CeedScalar A_ij[3][3] = {{0.}}, grad_velo_iso[3][3] = {{0.}};

  // -- Unpack anisotropy tensor
  KMUnpack(km_A_ij, A_ij);

  // -- Transform physical, anisotropic velocity gradient to isotropic
  MatMat3(grad_velo_aniso, A_ij, NO_TRANSPOSE, NO_TRANSPOSE, grad_velo_iso);

  {  // -- Get Eigenframe
    CeedScalar kmstrain_iso[6], strain_iso[3][3];
    CeedInt    work_vector[3] = {0};
    KMStrainRate(grad_velo_iso, kmstrain_iso);
    KMUnpack(kmstrain_iso, strain_iso);
    Diagonalize3(strain_iso, strain_sframe, eigenvectors, work_vector, SORT_DECREASING_EVALS, true, 5);
  }

  {  // -- Get vorticity in S-frame
    CeedScalar rotation_iso[3][3];
    RotationRate(grad_velo_iso, rotation_iso);
    CeedScalar vorticity_iso[3] = {-2 * rotation_iso[1][2], 2 * rotation_iso[0][2], -2 * rotation_iso[0][1]};
    OrientBasisWithVector(eigenvectors, vorticity_iso);
    MatVec3(eigenvectors, vorticity_iso, NO_TRANSPOSE, vorticity_sframe);
  }

  // -- Setup DD model inputs
  const CeedScalar grad_velo_magnitude = VelocityGradientMagnitude(strain_sframe, vorticity_sframe);
  CeedScalar inputs[6] = {strain_sframe[0], strain_sframe[1], strain_sframe[2], vorticity_sframe[0], vorticity_sframe[1], viscosity / Square(delta)};
  ScaleN(inputs, 1 / grad_velo_magnitude, 6);

  CeedScalar sgs_sframe_sym[6] = {0.};
  DataDrivenInference(inputs, sgs_sframe_sym, sgsdd_ctx);

  CeedScalar old_bounds[6][2] = {{0}};
  for (int j = 0; j < 6; j++) old_bounds[j][1] = 1;
  const CeedScalar(*new_bounds)[2] = (const CeedScalar(*)[2]) & sgsdd_ctx->data[sgsdd_ctx->offsets.out_scaling];
  DenormalizeDDOutputs(sgs_sframe_sym, new_bounds, old_bounds);

  // Re-dimensionalize sgs_stress
  ScaleN(sgs_sframe_sym, Square(delta) * Square(grad_velo_magnitude), 6);

  CeedScalar sgs_stress[3][3] = {{0.}};
  {  // Rotate SGS Stress back to physical frame, SGS_physical = E^T SGS_sframe E
    CeedScalar       Evec_sgs[3][3]   = {{0.}};
    const CeedScalar sgs_sframe[3][3] = {
        {sgs_sframe_sym[0], sgs_sframe_sym[3], sgs_sframe_sym[4]},
        {sgs_sframe_sym[3], sgs_sframe_sym[1], sgs_sframe_sym[5]},
        {sgs_sframe_sym[4], sgs_sframe_sym[5], sgs_sframe_sym[2]},
    };
    MatMat3(eigenvectors, sgs_sframe, TRANSPOSE, NO_TRANSPOSE, Evec_sgs);
    MatMat3(Evec_sgs, eigenvectors, NO_TRANSPOSE, NO_TRANSPOSE, sgs_stress);
  }

  KMPack(sgs_stress, kmsgs_stress);
}

#endif  // sgs_dd_model_h
