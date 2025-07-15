// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/types.h>

/// A structure used to pass additional data to f_build_mass
struct BuildContext {
  CeedInt dim, space_dim;
};



//CEED_QFUNCTION(add_num)(uint32_t x);
//extern "C" uint32_t __device__ add_num(uint32_t x);

// GOAL: make it so that you can CEED_QFUNCTION_RUST(build_mass, "path-to-rs.rs") to a rust-defined qfunction
// Can't use CEED_QFUNCTION macro bc static
// in ceed-cuda-compile.ccp, make it so that it detects if the file is a .rs

//CEED_QFUNCTION(build_mass)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out){return 0;};
/*extern "C" uint32_t __device__ build_mass_rs(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out);*/
/*CEED_QFUNCTION(build_mass)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
    return build_mass_rs(ctx, Q, in, out);
}*/
CEED_QFUNCTION_RUST(build_mass)


/// libCEED Q-function for building quadrature data for a mass operator
//CEED_QFUNCTION(build_mass)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0] is Jacobians with shape [dim, dim, Q]
  // in[1] is quadrature weights with shape [1, Q]

  //uint32_t num = 3;
  //volatile uint32_t var = (uint32_t) add_num(num);
  //return 0;

  //return build_mass_rs(ctx, Q, in, out);
  /*const CeedScalar    *w          = in[1];
  CeedScalar          *q_data     = out[0];
  struct BuildContext *build_data = (struct BuildContext *)ctx;



  switch (build_data->dim + 10 * build_data->space_dim) {
    case 11: {
      const CeedScalar(*J)[1][CEED_Q_VLA] = (const CeedScalar(*)[1][CEED_Q_VLA])in[0];

      // Quadrature Point Loop
      CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) { q_data[i] = J[0][0][i] * w[i]; }  // End of Quadrature Point Loop
    } break;
    case 22: {
      const CeedScalar(*J)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[0];

      // Quadrature Point Loop
      CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
        q_data[i] = (J[0][0][i] * J[1][1][i] - J[0][1][i] * J[1][0][i]) * w[i];
      }  // End of Quadrature Point Loop
    } break;
    case 33: {
      const CeedScalar(*J)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0];

      // Quadrature Point Loop
      CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
        q_data[i] =
            (J[0][0][i] * (J[1][1][i] * J[2][2][i] - J[1][2][i] * J[2][1][i]) - J[0][1][i] * (J[1][0][i] * J[2][2][i] - J[1][2][i] * J[2][0][i]) +
             J[0][2][i] * (J[1][0][i] * J[2][1][i] - J[1][1][i] * J[2][0][i])) *
            w[i];
      }  // End of Quadrature Point Loop
    } break;
  }
  return CEED_ERROR_SUCCESS;*/
//}

/// libCEED Q-function for applying a mass operator
CEED_QFUNCTION(apply_mass)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0], out[0] are solution variables with shape [1, Q]
  // in[1] is quadrature data with shape [1, Q]
  const CeedScalar *u = in[0], *q_data = in[1];
  CeedScalar       *v = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) { v[i] = q_data[i] * u[i]; }  // End of Quadrature Point Loop
  return CEED_ERROR_SUCCESS;
}
