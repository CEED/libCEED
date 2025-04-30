#ifndef ex3_volume_h
#define ex3_volume_h

#include <ceed.h>

/// A structure used to pass additional data to build_mass_diff
struct BuildContext {
  CeedInt dim, space_dim;
};

/// libCEED Q-function for building quadrature data for a mass + diffusion operator
CEED_QFUNCTION(build_mass_diff)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0] is Jacobians with shape [dim, dim, Q]
  // in[1] is quadrature weights, size (Q)
  const CeedScalar *w = in[1];
  CeedScalar(*q_data)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  struct BuildContext *build_data = (struct BuildContext *)ctx;

  // At every quadrature point, compute w/det(J).adj(J).adj(J)^T and store
  // the symmetric part of the result.
  switch (build_data->dim + 10 * build_data->space_dim) {
    case 11: {  // dim = 1, space_dim = 1
      const CeedScalar(*J)[1][CEED_Q_VLA] = (const CeedScalar(*)[1][CEED_Q_VLA])in[0];

      CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
        // Mass
        q_data[0][i] = w[i] * J[0][0][i];

        // Diffusion
        q_data[1][i] = w[i] / J[0][0][i];
      }
    } break;
    case 22: {  // dim = 2, space_dim = 2
      const CeedScalar(*J)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[0];

      CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
        // J: 0 2   q_data: 0 2   adj(J):  J22 -J12
        //    1 3           2 1           -J10  J00
        const CeedScalar J00 = J[0][0][i];
        const CeedScalar J10 = J[0][1][i];
        const CeedScalar J01 = J[1][0][i];
        const CeedScalar J11 = J[1][1][i];
        const CeedScalar qw = w[i] / (J00 * J11 - J10 * J01);

        // Mass
        q_data[0][i] = w[i] * (J00 * J11 - J10 * J01);

        // Diffusion
        q_data[1][i] = qw * (J01 * J01 + J11 * J11);
        q_data[2][i] = qw * (J00 * J00 + J10 * J10);
        q_data[3][i] = -qw * (J00 * J01 + J10 * J11);
      }
    } break;
    case 33: {  // dim = 3, space_dim = 3
      const CeedScalar(*J)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0];

      CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
        // Compute the adjoint
        CeedScalar A[3][3];
        for (CeedInt j = 0; j < 3; j++) {
          for (CeedInt k = 0; k < 3; k++) {
            A[k][j] = J[(k + 1) % 3][(j + 1) % 3][i] * J[(k + 2) % 3][(j + 2) % 3][i] -
                     J[(k + 2) % 3][(j + 1) % 3][i] * J[(k + 1) % 3][(j + 2) % 3][i];
          }
        }

        // Compute quadrature weight / det(J)
        const CeedScalar qw = w[i] / (J[0][0][i] * A[0][0] + J[0][1][i] * A[0][1] + J[0][2][i] * A[0][2]);

        // Mass
        q_data[0][i] = w[i] * (J[0][0][i] * A[0][0] + J[0][1][i] * A[0][1] + J[0][2][i] * A[0][2]);

        // Diffusion
        // Stored in Voigt convention
        // 1 6 5
        // 6 2 4
        // 5 4 3
        q_data[1][i] = qw * (A[0][0] * A[0][0] + A[0][1] * A[0][1] + A[0][2] * A[0][2]);
        q_data[2][i] = qw * (A[1][0] * A[1][0] + A[1][1] * A[1][1] + A[1][2] * A[1][2]);
        q_data[3][i] = qw * (A[2][0] * A[2][0] + A[2][1] * A[2][1] + A[2][2] * A[2][2]);
        q_data[4][i] = qw * (A[1][0] * A[2][0] + A[1][1] * A[2][1] + A[1][2] * A[2][2]);
        q_data[5][i] = qw * (A[0][0] * A[2][0] + A[0][1] * A[2][1] + A[0][2] * A[2][2]);
        q_data[6][i] = qw * (A[0][0] * A[1][0] + A[0][1] * A[1][1] + A[0][2] * A[1][2]);
      }
    } break;
  }
  return CEED_ERROR_SUCCESS;
}

/// libCEED Q-function for applying a mass + diffusion operator
CEED_QFUNCTION(apply_mass_diff)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  struct BuildContext *build_data = (struct BuildContext *)ctx;
  // in[0], out[0] solution values with shape [1, 1, Q]
  // in[1], out[1] solution gradients with shape [dim, 1, Q]
  // in[2] is quadrature data with shape [num_components, Q]
  const CeedScalar(*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];

  switch (build_data->dim) {
    case 1: {
      const CeedScalar *u = in[0], *ug = in[1];
      CeedScalar *v = out[0], *vg = out[1];

      CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
        // Mass
        v[i] = q_data[0][i] * u[i];

        // Diffusion
        vg[i] = q_data[1][i] * ug[i];
      }
    } break;
    case 2: {
      const CeedScalar *u = in[0];
      const CeedScalar(*ug)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
      CeedScalar *v = out[0];
      CeedScalar(*vg)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[1];

      CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
        // Mass
        v[i] = q_data[0][i] * u[i];

        // Diffusion
        // Read q_data (dXdxdXdx_T symmetric matrix)
        // Stored in Voigt convention
        // 1 3
        // 3 2
        const CeedScalar dXdxdXdx_T[2][2] = {
            {q_data[1][i], q_data[3][i]},
            {q_data[3][i], q_data[2][i]}
        };

        // j = direction of vg
        for (int j = 0; j < 2; j++) {
          vg[j][i] = (ug[0][i] * dXdxdXdx_T[0][j] + ug[1][i] * dXdxdXdx_T[1][j]);
        }
      }
    } break;
    case 3: {
      const CeedScalar *u = in[0];
      const CeedScalar(*ug)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
      CeedScalar *v = out[0];
      CeedScalar(*vg)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[1];

      CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
        // Mass
        v[i] = q_data[0][i] * u[i];

        // Diffusion
        // Read q_data (dXdxdXdx_T symmetric matrix)
        // Stored in Voigt convention
        // 1 6 5
        // 6 2 4
        // 5 4 3
        const CeedScalar dXdxdXdx_T[3][3] = {
            {q_data[1][i], q_data[6][i], q_data[5][i]},
            {q_data[6][i], q_data[2][i], q_data[4][i]},
            {q_data[5][i], q_data[4][i], q_data[3][i]}
        };

        // j = direction of vg
        for (int j = 0; j < 3; j++) {
          vg[j][i] = (ug[0][i] * dXdxdXdx_T[0][j] + ug[1][i] * dXdxdXdx_T[1][j] + ug[2][i] * dXdxdXdx_T[2][j]);
        }
      }
    } break;
  }
  return CEED_ERROR_SUCCESS;
}

#endif 