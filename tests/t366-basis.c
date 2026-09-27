/// @file
/// Test tensor basis apply against a reference contraction
/// \test Test tensor basis apply against a reference contraction
#include <ceed.h>
#include <math.h>
#include <stdio.h>

typedef enum {
  MATRIX_SYMMETRIC     = 0,
  MATRIX_ANTISYMMETRIC = 1,
  MATRIX_QUADRANT_ONLY = 2,  // symmetric in the top-left quadrant only
  MATRIX_GENERAL       = 3,
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

      // Mirror over every column
      for (CeedInt i = 0; i < (num_rows + 1) / 2; i++) {
        for (CeedInt j = 0; j < num_cols; j++) matrix[(num_rows - 1 - i) * num_cols + (num_cols - 1 - j)] = sign * matrix[i * num_cols + j];
      }
      break;
    }
    case MATRIX_QUADRANT_ONLY:
      // Mirror the top-left quadrant only
      for (CeedInt i = 0; i < (num_rows + 1) / 2; i++) {
        for (CeedInt j = 0; j < (num_cols + 1) / 2; j++) matrix[(num_rows - 1 - i) * num_cols + (num_cols - 1 - j)] = matrix[i * num_cols + j];
      }
      break;
    case MATRIX_GENERAL:
      break;
  }
}

// Contract on the middle index
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

  // Orders either side of CEED_EVEN_ODD_MIN_DIM
  const CeedInt orders[] = {4, 5, 10, 11, 12};

  for (CeedInt dim = 1; dim <= 3; dim++) {
    for (unsigned pi = 0; pi < sizeof(orders) / sizeof(orders[0]); pi++) {
      const CeedInt p = orders[pi];

      if (dim == 3 && p > 5) continue;
      for (CeedInt q = p - 1; q <= p + 1; q++) {
        if (q < 2) continue;
        for (CeedInt kind = MATRIX_SYMMETRIC; kind <= MATRIX_GENERAL; kind++) {
          CeedBasis  basis;
          CeedScalar interp_1d[q * p], grad_1d[q * p], q_ref_1d[q], q_weight_1d[q];
          CeedInt    p_dim = CeedIntPow(p, dim), q_dim = CeedIntPow(q, dim);

          BuildMatrix((MatrixKind)kind, q, p, interp_1d);
          // Lagrange bases pair a symmetric interp with an antisymmetric grad
          BuildMatrix(kind == MATRIX_SYMMETRIC ? MATRIX_ANTISYMMETRIC : (MatrixKind)kind, q, p, grad_1d);
          for (CeedInt i = 0; i < q; i++) {
            q_ref_1d[i]    = -1.0 + 2.0 * i / (q - 1);
            q_weight_1d[i] = 2.0 / q;
          }
          CeedBasisCreateTensorH1(ceed, dim, num_comp, p, q, interp_1d, grad_1d, q_ref_1d, q_weight_1d, &basis);
          // Do not depend on the size based default
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

            // One contraction per dimension
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
