// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Structs and helper functions for data-driven subgrid-stress modeling
/// See 'Invariant data-driven subgrid stress modeling in the strain-rate eigenframe for large eddy simulation' 2022 and 'S-frame discrepancy
/// correction models for data-informed Reynolds stress closure' 2022

#ifndef sgs_dd_utils_h
#define sgs_dd_utils_h

#include <ceed.h>

#include "newtonian_state.h"
#include "newtonian_types.h"
#include "utils.h"
#include "utils_eigensolver_jacobi.h"

// @brief Calculate Frobenius norm of velocity gradient from eigenframe quantities
CEED_QFUNCTION_HELPER CeedScalar VelocityGradientMagnitude(const CeedScalar strain_sframe[3], const CeedScalar vorticity_sframe[3]) {
  return sqrt(Dot3(strain_sframe, strain_sframe) + 0.5 * Dot3(vorticity_sframe, vorticity_sframe));
};

// @brief Change the order of basis vectors so that they align with vector and obey right-hand rule
// @details The e_1 and e_3 basis vectors are the closest aligned to the vector. The e_2 is set via  e_3 x e_1
// The basis vectors are assumed to form the rows of the basis matrix.
CEED_QFUNCTION_HELPER void OrientBasisWithVector(CeedScalar basis[3][3], const CeedScalar vector[3]) {
  CeedScalar alignment[3] = {0.}, cross[3];

  MatVec3(basis, vector, CEED_NOTRANSPOSE, alignment);

  if (alignment[0] < 0) ScaleN(basis[0], -1, 3);
  if (alignment[2] < 0) ScaleN(basis[2], -1, 3);

  Cross3(basis[2], basis[0], cross);
  CeedScalar basis_1_orientation = Dot3(cross, basis[1]);
  if (basis_1_orientation < 0) ScaleN(basis[1], -1, 3);
}

// @brief Denormalize outputs using min-max (de-)normalization
CEED_QFUNCTION_HELPER void DenormalizeDDOutputs(CeedScalar output[6], const CeedScalar new_bounds[6][2], const CeedScalar old_bounds[6][2]) {
  CeedScalar bounds_ratio;
  for (int i = 0; i < 6; i++) {
    bounds_ratio = (new_bounds[i][1] - new_bounds[i][0]) / (old_bounds[i][1] - old_bounds[i][0]);
    output[i]    = bounds_ratio * (output[i] - old_bounds[i][1]) + new_bounds[i][1];
  }
}

/**
 * @brief Compute model inputs for anisotropic data-driven model
 *
 * @param[in]  grad_velo_aniso     Gradient of velocity in physical (anisotropic) coordinates
 * @param[in]  km_A_ij             Anisotropy tensor, in Kelvin-Mandel notation
 * @param[in]  delta               Length used to create anisotropy tensor
 * @param[in]  viscosity           Kinematic viscosity
 * @param[out] eigenvectors        Eigenvectors of the (anisotropic) velocity gradient
 * @param[out] inputs              Data-driven model inputs
 * @param[out] grad_velo_magnitude Frobenius norm of the velocity gradient
 */
CEED_QFUNCTION_HELPER void ComputeSgsDDInputs(const CeedScalar grad_velo_aniso[3][3], const CeedScalar km_A_ij[6], const CeedScalar delta,
                                              const CeedScalar viscosity, CeedScalar eigenvectors[3][3], CeedScalar inputs[6],
                                              CeedScalar *grad_velo_magnitude) {
  CeedScalar strain_sframe[3] = {0.}, vorticity_sframe[3] = {0.};
  CeedScalar A_ij[3][3] = {{0.}}, grad_velo_iso[3][3] = {{0.}};

  // -- Transform physical, anisotropic velocity gradient to isotropic
  KMUnpack(km_A_ij, A_ij);
  MatMat3(grad_velo_aniso, A_ij, CEED_NOTRANSPOSE, CEED_NOTRANSPOSE, grad_velo_iso);

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
    MatVec3(eigenvectors, vorticity_iso, CEED_NOTRANSPOSE, vorticity_sframe);
  }

  // -- Calculate DD model inputs
  *grad_velo_magnitude = VelocityGradientMagnitude(strain_sframe, vorticity_sframe);
  inputs[0]            = strain_sframe[0];
  inputs[1]            = strain_sframe[1];
  inputs[2]            = strain_sframe[2];
  inputs[3]            = vorticity_sframe[0];
  inputs[4]            = vorticity_sframe[1];
  inputs[5]            = viscosity / Square(delta);
  ScaleN(inputs, 1 / (*grad_velo_magnitude + CEED_EPSILON), 6);
}

/**
 * @brief Compute the physical SGS stresses from the neural-network output
 *
 * @param[in,out] outputs             Outputs from the neural-network
 * @param[in]     delta               Length used to create anisotropy tensor
 * @param[in]     eigenvectors        Eigenvectors of the (anisotropic) velocity gradient
 * @param[in]     new_bounds          Bounds used for min-max de-normalization
 * @param[in]     grad_velo_magnitude Magnitude of the velocity gradient
 * @param[out]    kmsgs_stress        Physical SGS stresses in Kelvin-Mandel notation
 */
CEED_QFUNCTION_HELPER void ComputeSgsDDOutputs(CeedScalar outputs[6], const CeedScalar delta, const CeedScalar eigenvectors[3][3],
                                               const CeedScalar new_bounds[6][2], const CeedScalar grad_velo_magnitude, CeedScalar kmsgs_stress[6]) {
  CeedScalar old_bounds[6][2] = {{0}};
  for (int j = 0; j < 6; j++) old_bounds[j][1] = 1;
  DenormalizeDDOutputs(outputs, new_bounds, old_bounds);

  // Re-dimensionalize sgs_stress
  ScaleN(outputs, Square(delta) * Square(grad_velo_magnitude), 6);

  CeedScalar sgs_stress[3][3] = {{0.}};
  {  // Rotate SGS Stress back to physical frame, SGS_physical = E^T SGS_sframe E
    CeedScalar       Evec_sgs[3][3]   = {{0.}};
    const CeedScalar sgs_sframe[3][3] = {
        {outputs[0], outputs[3], outputs[4]},
        {outputs[3], outputs[1], outputs[5]},
        {outputs[4], outputs[5], outputs[2]},
    };
    MatMat3(eigenvectors, sgs_sframe, CEED_TRANSPOSE, CEED_NOTRANSPOSE, Evec_sgs);
    MatMat3(Evec_sgs, eigenvectors, CEED_NOTRANSPOSE, CEED_NOTRANSPOSE, sgs_stress);
  }

  KMPack(sgs_stress, kmsgs_stress);
}

#endif  // sgs_dd_utils_h
