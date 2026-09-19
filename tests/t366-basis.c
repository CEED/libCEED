/// @file
/// Test tensor basis apply against a reference contraction for centro-symmetric and general 1D matrices
/// \test Test tensor basis apply against a reference contraction for centro-symmetric and general 1D matrices
#include <ceed.h>
#include <math.h>
#include <stdio.h>

// Kinds of 1D matrix to build
typedef enum {
  MATRIX_SYMMETRIC     = 0,  // t[i][j] = t[Q-1-i][P-1-j]
  MATRIX_ANTISYMMETRIC = 1,  // t[i][j] = -t[Q-1-i][P-1-j]
  MATRIX_QUADRANT_ONLY = 2,  // centro-symmetric only over the top-left quadrant pairs
  MATRIX_GENERAL       = 3,  // no centro-symmetry at all
} MatrixKind;

static CeedScalar Entry(CeedInt i, CeedInt j) { return cos(0.7 * (i + 1) + 0.3 * (j + 1)); }

static void BuildMatrix(MatrixKind kind, CeedInt num_rows, CeedInt num_cols, CeedScalar *matrix) {
  for (CeedInt i = 0; i < num_rows; i++) {
    for (CeedInt j = 0; j < num_cols; j++) matrix[i * num_cols + j] = Entry(i, j);
  }
  switch (kind) {
    case MATRIX_SYMMETRIC:
    case MATRIX_ANTISYMMETRIC: {
      const CeedScalar sign = kind == MATRIX_SYMMETRIC ? 1.0 : -1.0;

      // Mirror the top half of the rows onto the bottom half, over every column
      for (CeedInt i = 0; i < (num_rows + 1) / 2; i++) {
        for (CeedInt j = 0; j < num_cols; j++) matrix[(num_rows - 1 - i) * num_cols + (num_cols - 1 - j)] = sign * matrix[i * num_cols + j];
      }
      break;
    }
    case MATRIX_QUADRANT_ONLY:
      // Mirror only the top-left quadrant, which leaves the top-right and bottom-left halves unrelated
      for (CeedInt i = 0; i < (num_rows + 1) / 2; i++) {
        for (CeedInt j = 0; j < (num_cols + 1) / 2; j++) matrix[(num_rows - 1 - i) * num_cols + (num_cols - 1 - j)] = matrix[i * num_cols + j];
      }
      break;
    case MATRIX_GENERAL:
      break;
  }
}

// Reference contraction, contracting on the middle index
static void ContractReference(CeedInt A, CeedInt B, CeedInt C, CeedInt J, const CeedScalar *t, CeedTransposeMode t_mode, const CeedScalar *u,
                              CeedScalar *v) {
  for (CeedInt a = 0; a < A; a++) {
    for (CeedInt j = 0; j < J; j++) {
      for (CeedInt c = 0; c < C; c++) {
        CeedScalar sum = 0.0;

        for (CeedInt b = 0; b < B; b++) sum += (t_mode == CEED_TRANSPOSE ? t[b * J + j] : t[j * B + b]) * u[(a * B + b) * C + c];
        v[(a * J + j) * C + c] = sum;
      }
    }
  }
}

int main(int argc, char **argv) {
  Ceed             ceed;
  const CeedInt    num_comp = 2, num_elem = 3;
  const CeedScalar tol = CEED_SCALAR_TYPE == CEED_SCALAR_FP32 ? 1.e-4 : 1.e-11;

  CeedInit(argv[1], &ceed);

  for (CeedInt dim = 1; dim <= 3; dim++) {
    for (CeedInt p = 2; p <= 6; p++) {
      for (CeedInt q = 2; q <= 6; q++) {
        for (CeedInt kind = MATRIX_SYMMETRIC; kind <= MATRIX_GENERAL; kind++) {
          CeedBasis  basis;
          CeedScalar interp_1d[q * p], grad_1d[q * p], q_ref_1d[q], q_weight_1d[q];
          CeedInt    p_dim = CeedIntPow(p, dim), q_dim = CeedIntPow(q, dim);

          BuildMatrix((MatrixKind)kind, q, p, interp_1d);
          // Pair a symmetric interp with an antisymmetric grad, as a Lagrange basis on symmetric nodes does
          BuildMatrix(kind == MATRIX_SYMMETRIC ? MATRIX_ANTISYMMETRIC : (MatrixKind)kind, q, p, grad_1d);
          for (CeedInt i = 0; i < q; i++) {
            q_ref_1d[i]    = -1.0 + 2.0 * i / (q - 1);
            q_weight_1d[i] = 2.0 / q;
          }
          CeedBasisCreateTensorH1(ceed, dim, num_comp, p, q, interp_1d, grad_1d, q_ref_1d, q_weight_1d, &basis);
          // Exercise the even-odd path explicitly so this test does not depend on the default
          CeedBasisSetUseEvenOdd(basis, true);

          for (CeedInt t_mode = 0; t_mode < 2; t_mode++) {
            const CeedTransposeMode mode   = t_mode ? CEED_TRANSPOSE : CEED_NOTRANSPOSE;
            const CeedInt           num_in = t_mode ? q_dim : p_dim, num_out = t_mode ? p_dim : q_dim;
            const CeedInt           B = t_mode ? q : p, J = t_mode ? p : q;
            CeedVector              u, v;
            CeedScalar              u_array[num_comp * num_in * num_elem];
            CeedScalar              reference[2][num_comp * (p_dim > q_dim ? p_dim : q_dim) * num_elem];

            for (CeedInt i = 0; i < num_comp * num_in * num_elem; i++) u_array[i] = cos(0.13 * i + 0.5);
            CeedVectorCreate(ceed, num_comp * num_in * num_elem, &u);
            CeedVectorSetArray(u, CEED_MEM_HOST, CEED_COPY_VALUES, u_array);
            CeedVectorCreate(ceed, num_comp * num_out * num_elem, &v);
            CeedVectorSetValue(v, 0.0);

            CeedBasisApply(basis, num_elem, mode, CEED_EVAL_INTERP, u, v);

            // Reference: one contraction per dimension
            {
              const CeedScalar *in  = u_array;
              CeedInt           pre = num_comp * CeedIntPow(B, dim - 1), post = num_elem;

              for (CeedInt d = 0; d < dim; d++) {
                ContractReference(pre, B, post, J, interp_1d, mode, in, reference[d % 2]);
                in = reference[d % 2];
                pre /= B;
                post *= J;
              }
              {
                const CeedScalar *v_array;
                const CeedScalar *expected = reference[(dim - 1) % 2];

                CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
                for (CeedInt i = 0; i < num_comp * num_out * num_elem; i++) {
                  if (fabs(v_array[i] - expected[i]) > tol * (fabs(expected[i]) + 1.0)) {
                    printf("[%" CeedInt_FMT ", p %" CeedInt_FMT ", q %" CeedInt_FMT ", kind %" CeedInt_FMT ", %s] %f != %f\n", dim, p, q, kind,
                           t_mode ? "transpose" : "notranspose", v_array[i], expected[i]);
                    break;
                  }
                }
                CeedVectorRestoreArrayRead(v, &v_array);
              }
            }
            CeedVectorDestroy(&u);
            CeedVectorDestroy(&v);
          }
          CeedBasisDestroy(&basis);
        }
      }
    }
  }
  CeedDestroy(&ceed);
  return 0;
}
