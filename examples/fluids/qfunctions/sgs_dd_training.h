// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Structs and helper functions for training data-driven subgrid-stress models
/// See 'Invariant data-driven subgrid stress modeling in the strain-rate eigenframe for large eddy simulation' 2022 and 'S-frame discrepancy
/// correction models for data-informed Reynolds stress closure' 2022

#ifndef sgs_dd_training_h
#define sgs_dd_training_h

#include <ceed.h>

#include "newtonian_state.h"
#include "newtonian_types.h"
#include "sgs_dd_utils.h"
#include "utils.h"
#include "utils_eigensolver_jacobi.h"

typedef struct SGS_DD_TrainingContext_ *SGS_DDTrainingContext;
struct SGS_DD_TrainingContext_ {
  struct NewtonianIdealGasContext_ gas;
};

// @brief Calculate model inputs for anisotropic data-driven model at nodes
CEED_QFUNCTION_HELPER int ComputeSGS_DDAnisotropicInputsNodal(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out,
                                                              StateFromQi_t StateFromQi) {
  const CeedScalar(*q)[CEED_Q_VLA]            = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*x)[CEED_Q_VLA]            = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*grad_velo)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[2];
  const CeedScalar(*A_ij_delta)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[3];
  const CeedScalar(*inv_multiplicity)         = (const CeedScalar(*))in[4];
  CeedScalar(*v)[CEED_Q_VLA]                  = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const SGS_DDTrainingContext    sgsdd_ctx = (SGS_DDTrainingContext)ctx;
  const NewtonianIdealGasContext gas       = &sgsdd_ctx->gas;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5]                 = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const CeedScalar x_i[3]                = {x[0][i], x[1][i], x[2][i]};
    const CeedScalar grad_velo_aniso[3][3] = {
        {grad_velo[0][0][i], grad_velo[0][1][i], grad_velo[0][2][i]},
        {grad_velo[1][0][i], grad_velo[1][1][i], grad_velo[1][2][i]},
        {grad_velo[2][0][i], grad_velo[2][1][i], grad_velo[2][2][i]}
    };
    const CeedScalar km_A_ij[6] = {A_ij_delta[0][i], A_ij_delta[1][i], A_ij_delta[2][i], A_ij_delta[3][i], A_ij_delta[4][i], A_ij_delta[5][i]};
    const CeedScalar delta      = A_ij_delta[6][i];
    const State      s          = StateFromQi(gas, qi, x_i);
    CeedScalar       inputs[6];
    CeedScalar       eigenvectors[3][3], grad_velo_magnitude;  // dummy variables, don't actually use them

    ComputeSGS_DDAnisotropicInputs(grad_velo_aniso, km_A_ij, delta, gas->mu / s.U.density, eigenvectors, inputs, &grad_velo_magnitude);

    for (int j = 0; j < 6; j++) v[j][i] = inv_multiplicity[i] * inputs[j];
  }
  return 0;
}

CEED_QFUNCTION(ComputeSGS_DDAnisotropicInputsNodal_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ComputeSGS_DDAnisotropicInputsNodal(ctx, Q, in, out, StateFromY);
}

#endif  // sgs_dd_training_h
