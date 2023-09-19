// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Structs and helper functions to evaluate data-driven subgrid-stress modeling
/// See 'Invariant data-driven subgrid stress modeling in the strain-rate eigenframe for large eddy simulation' 2022 and 'S-frame discrepancy
/// correction models for data-informed Reynolds stress closure' 2022

#ifndef sgs_dd_model_h
#define sgs_dd_model_h

#include <ceed.h>

#include "newtonian_state.h"
#include "newtonian_types.h"
#include "sgs_dd_utils.h"
#include "utils.h"
#include "utils_eigensolver_jacobi.h"

typedef struct SgsDDContext_ *SgsDDContext;
struct SgsDDContext_ {
  CeedInt    num_inputs, num_outputs;
  CeedInt    num_layers;
  CeedInt    num_neurons;
  CeedScalar alpha;

  struct NewtonianIdealGasContext_ gas;
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

CEED_QFUNCTION_HELPER void DataDrivenInference(const CeedScalar *inputs, CeedScalar *outputs, SgsDDContext sgsdd_ctx) {
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
  MatVecNM(weight1, inputs, num_neurons, num_inputs, CEED_NOTRANSPOSE, V);
  LeakyReLU(V, alpha, num_neurons);
  CopyN(bias2, outputs, num_outputs);
  MatVecNM(weight2, V, num_outputs, num_neurons, CEED_NOTRANSPOSE, outputs);
}

CEED_QFUNCTION_HELPER void ComputeSgsDD_Fused(const CeedScalar grad_velo_aniso[3][3], const CeedScalar km_A_ij[6], const CeedScalar delta,
                                              const CeedScalar viscosity, CeedScalar kmsgs_stress[6], SgsDDContext sgsdd_ctx) {
  CeedScalar inputs[6], grad_velo_magnitude, eigenvectors[3][3], sgs_sframe_sym[6] = {0.}, new_bounds[6][2];
  // Copying new_bounds because Sycl online compiler doesn't like direct casting the pointer
  CopyN(&sgsdd_ctx->data[sgsdd_ctx->offsets.out_scaling], (CeedScalar *)new_bounds, 12);

  ComputeSgsDDInputs(grad_velo_aniso, km_A_ij, delta, viscosity, eigenvectors, inputs, &grad_velo_magnitude);
  DataDrivenInference(inputs, sgs_sframe_sym, sgsdd_ctx);
  ComputeSgsDDOutputs(sgs_sframe_sym, delta, eigenvectors, new_bounds, grad_velo_magnitude, kmsgs_stress);
}

// @brief Calculate subgrid stress at nodes using anisotropic data-driven model
CEED_QFUNCTION_HELPER int ComputeSgsDDNodal_Fused(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out,
                                                  StateVariable state_var) {
  const CeedScalar(*q)[CEED_Q_VLA]            = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*grad_velo)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[2];
  const CeedScalar(*A_ij_delta)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[3];
  const CeedScalar(*inv_multiplicity)         = (const CeedScalar(*))in[4];
  CeedScalar(*v)[CEED_Q_VLA]                  = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const SgsDDContext             sgsdd_ctx = (SgsDDContext)ctx;
  const NewtonianIdealGasContext gas       = &sgsdd_ctx->gas;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5]                 = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const CeedScalar grad_velo_aniso[3][3] = {
        {grad_velo[0][0][i], grad_velo[0][1][i], grad_velo[0][2][i]},
        {grad_velo[1][0][i], grad_velo[1][1][i], grad_velo[1][2][i]},
        {grad_velo[2][0][i], grad_velo[2][1][i], grad_velo[2][2][i]}
    };
    const CeedScalar km_A_ij[6] = {A_ij_delta[0][i], A_ij_delta[1][i], A_ij_delta[2][i], A_ij_delta[3][i], A_ij_delta[4][i], A_ij_delta[5][i]};
    const CeedScalar delta      = A_ij_delta[6][i];
    const State      s          = StateFromQ(gas, qi, state_var);
    CeedScalar       km_sgs[6];

    ComputeSgsDD_Fused(grad_velo_aniso, km_A_ij, delta, gas->mu / s.U.density, km_sgs, sgsdd_ctx);

    for (int j = 0; j < 6; j++) v[j][i] = inv_multiplicity[i] * km_sgs[j];
  }
  return 0;
}

CEED_QFUNCTION(ComputeSgsDDNodal_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ComputeSgsDDNodal_Fused(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

CEED_QFUNCTION(ComputeSgsDDNodal_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ComputeSgsDDNodal_Fused(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

// @brief Calculate inputs to anisotropic data-driven model
CEED_QFUNCTION_HELPER int ComputeSgsDDNodal_Sequential_Inputs(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out,
                                                              StateVariable state_var) {
  const CeedScalar(*q)[CEED_Q_VLA]            = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*grad_velo)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[1];
  const CeedScalar(*A_ij_delta)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  const CeedScalar(*inv_multiplicity)         = (const CeedScalar(*))in[3];
  CeedScalar(*eigenvectors_stored)            = out[0];
  CeedScalar(*model_inputs)[CEED_Q_VLA]       = (CeedScalar(*)[CEED_Q_VLA])out[1];

  const SgsDDModelContext        sgsdd_ctx = (SgsDDModelContext)ctx;
  const NewtonianIdealGasContext gas       = &sgsdd_ctx->gas;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5]                 = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const CeedScalar grad_velo_aniso[3][3] = {
        {grad_velo[0][0][i], grad_velo[0][1][i], grad_velo[0][2][i]},
        {grad_velo[1][0][i], grad_velo[1][1][i], grad_velo[1][2][i]},
        {grad_velo[2][0][i], grad_velo[2][1][i], grad_velo[2][2][i]}
    };
    const CeedScalar km_A_ij[6] = {A_ij_delta[0][i], A_ij_delta[1][i], A_ij_delta[2][i], A_ij_delta[3][i], A_ij_delta[4][i], A_ij_delta[5][i]};
    const CeedScalar delta      = A_ij_delta[6][i];
    const State      s          = StateFromQ(gas, qi, state_var);

    CeedScalar model_inputs_i[6], grad_velo_magnitude, eigenvectors[3][3];
    ComputeSgsDDInputs(grad_velo_aniso, km_A_ij, delta, gas->mu / s.U.density, eigenvectors, model_inputs_i, &grad_velo_magnitude);

    ScaleN(model_inputs_i, inv_multiplicity[i], 6);
    StoredValuesPack(Q, i, 0, 6, model_inputs_i, (CeedScalar *)model_inputs);
    StoredValuesPack(Q, i, 0, 9, (const CeedScalar *)eigenvectors, eigenvectors_stored);
  }
  return CEED_ERROR_SUCCESS;
}

CEED_QFUNCTION(ComputeSgsDDNodal_Sequential_Inputs_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ComputeSgsDDNodal_Sequential_Inputs(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

CEED_QFUNCTION(ComputeSgsDDNodal_Sequential_Inputs_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ComputeSgsDDNodal_Sequential_Inputs(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

// @brief Calculates SGS from outputs of anisotropic data-driven model
CEED_QFUNCTION_HELPER int ComputeSgsDDNodal_Sequential_Outputs(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out,
                                                               StateVariable state_var) {
  const CeedScalar(*model_outputs)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*grad_velo)[3][CEED_Q_VLA]  = (const CeedScalar(*)[3][CEED_Q_VLA])in[1];
  const CeedScalar(*A_ij_delta)[CEED_Q_VLA]    = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  const CeedScalar(*inv_multiplicity)          = (const CeedScalar(*))in[3];
  const CeedScalar(*eigenvectors_stored)       = in[4];
  CeedScalar(*kmsgs_stress)[CEED_Q_VLA]        = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const SgsDDModelContext sgsdd_ctx = (SgsDDModelContext)ctx;
  CeedScalar              new_bounds[6][2];
  CopyN(&sgsdd_ctx->data[sgsdd_ctx->offsets.out_scaling], (CeedScalar *)new_bounds, 12);

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar       model_outputs_i[6]    = {model_outputs[0][i], model_outputs[1][i], model_outputs[2][i],
                                              model_outputs[3][i], model_outputs[4][i], model_outputs[5][i]};
    const CeedScalar grad_velo_aniso[3][3] = {
        {grad_velo[0][0][i], grad_velo[0][1][i], grad_velo[0][2][i]},
        {grad_velo[1][0][i], grad_velo[1][1][i], grad_velo[1][2][i]},
        {grad_velo[2][0][i], grad_velo[2][1][i], grad_velo[2][2][i]}
    };
    const CeedScalar delta = A_ij_delta[6][i];

    StoredValuesUnpack(Q, i, 0, 6, model_outputs, model_outputs_i);
    CeedScalar grad_velo_magnitude, eigenvectors[3][3], kmsgs_stress_i[6];
    StoredValuesUnpack(Q, i, 0, 9, eigenvectors_stored, (CeedScalar *)eigenvectors);
    grad_velo_magnitude = sqrt(DotN((CeedScalar *)grad_velo_aniso, (CeedScalar *)grad_velo_aniso, 9));
    ComputeSgsDDOutputs(model_outputs_i, delta, eigenvectors, new_bounds, grad_velo_magnitude, kmsgs_stress_i);

    for (int j = 0; j < 6; j++) kmsgs_stress[j][i] = inv_multiplicity[i] * kmsgs_stress_i[j];
  }
  return CEED_ERROR_SUCCESS;
}

// @brief Adds subgrid stress to residual (during IFunction evaluation)
CEED_QFUNCTION_HELPER int FluxSubgridStress(const StatePrimitive Y, const CeedScalar km_sgs[6], CeedScalar Flux[5][3]) {
  CeedScalar sgs[3][3];

  KMUnpack(km_sgs, sgs);
  for (CeedInt j = 0; j < 3; j++) {
    Flux[0][j] = 0.;
    for (CeedInt k = 0; k < 3; k++) Flux[k + 1][j] = sgs[k][j];
    Flux[4][j] = Y.velocity[0] * sgs[0][j] + Y.velocity[1] * sgs[1][j] + Y.velocity[2] * sgs[2][j];
  }
  return 0;
}

CEED_QFUNCTION_HELPER int IFunction_NodalSgs(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var) {
  const CeedScalar(*q)[CEED_Q_VLA]      = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data)             = in[1];
  const CeedScalar(*km_sgs)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  CeedScalar(*Grad_v)[5][CEED_Q_VLA]    = (CeedScalar(*)[5][CEED_Q_VLA])out[0];

  NewtonianIdealGasContext gas = (NewtonianIdealGasContext)ctx;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5] = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const State      s     = StateFromQ(gas, qi, state_var);

    CeedScalar wdetJ, dXdx[3][3];
    QdataUnpack_3D(Q, i, q_data, &wdetJ, dXdx);

    CeedScalar       Flux[5][3];
    const CeedScalar km_sgs_i[6] = {km_sgs[0][i], km_sgs[1][i], km_sgs[2][i], km_sgs[3][i], km_sgs[4][i], km_sgs[5][i]};
    FluxSubgridStress(s.Y, km_sgs_i, Flux);

    for (CeedInt k = 0; k < 3; k++) {
      for (CeedInt j = 0; j < 5; j++) {
        Grad_v[k][j][i] = -wdetJ * (dXdx[k][0] * Flux[j][0] + dXdx[k][1] * Flux[j][1] + dXdx[k][2] * Flux[j][2]);
      }
    }
  }
  return 0;
}

CEED_QFUNCTION(IFunction_NodalSgs_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return IFunction_NodalSgs(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(IFunction_NodalSgs_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return IFunction_NodalSgs(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

#endif  // sgs_dd_model_h
