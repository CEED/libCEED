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

// Build matrix matching symmetry pattern
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

int main(int argc, char **argv) {
  Ceed          ceed;
  const CeedInt num_comp = 2, num_elem = 8;

  CeedInit(argv[1], &ceed);

  {
    CeedMemType type;

    // Only CPU backends use this test
    CeedGetPreferredMemType(ceed, &type);
    if (type != CEED_MEM_HOST) return 0;
  }

  // Orders either side of CEED_EVEN_ODD_MIN_DIM
  const CeedInt num_orders = 2;
  const CeedInt orders[2]  = {5, 11};

  for (CeedInt dim = 1; dim <= 2; dim++) {
    for (CeedInt i = 0; i < num_orders; i++) {
      const CeedInt p = orders[i];

      for (CeedInt q = p - 1; q <= p + 1; q++) {
        for (CeedInt kind = MATRIX_SYMMETRIC; kind <= MATRIX_GENERAL; kind++) {
          CeedBasis  basis;
          CeedScalar interp_1d[q * p], grad_1d[q * p], q_ref_1d[q], q_weight_1d[q];

          BuildMatrix((MatrixKind)kind, q, p, interp_1d);
          // Note: Lagrange bases pair a symmetric interp with an antisymmetric grad
          BuildMatrix(kind == MATRIX_SYMMETRIC ? MATRIX_ANTISYMMETRIC : (MatrixKind)kind, q, p, grad_1d);
          CeedBasisCreateTensorH1(ceed, dim, num_comp, p, q, interp_1d, grad_1d, q_ref_1d, q_weight_1d, &basis);

          for (CeedInt mode = 0; mode < 2; mode++) {
            const CeedTransposeMode t_mode = mode == 0 ? CEED_TRANSPOSE : CEED_NOTRANSPOSE;
            const CeedInt           num_u  = CeedIntPow(t_mode == CEED_NOTRANSPOSE ? p : q, dim) * (t_mode == CEED_NOTRANSPOSE ? 1 : dim),
                                    num_v  = CeedIntPow(t_mode == CEED_NOTRANSPOSE ? q : p, dim) * (t_mode == CEED_NOTRANSPOSE ? dim : 1);
            CeedVector              u, v_with_split, v_without_split;

            // Set work arrays
            CeedVectorCreate(ceed, num_comp * num_u * num_elem, &u);
            {
              CeedScalar *u_array;

              CeedVectorGetArrayWrite(u, CEED_MEM_HOST, &u_array);
              for (CeedInt i = 0; i < num_comp * num_u * num_elem; i++) u_array[i] = cos(0.13 * i + 0.5);
              CeedVectorRestoreArray(u, &u_array);
            }
            CeedVectorCreate(ceed, num_comp * num_v * num_elem, &v_with_split);
            CeedVectorSetValue(v_with_split, 0.0);
            CeedVectorCreate(ceed, num_comp * num_v * num_elem, &v_without_split);
            CeedVectorSetValue(v_without_split, 0.0);

            // Check Interp

            // -- Force use of even-odd split
            CeedSetContractUseEvenOdd(ceed, true);
            CeedBasisApply(basis, num_elem, t_mode, CEED_EVAL_INTERP, u, v_with_split);

            // -- Force no use of even-odd split
            CeedSetContractUseEvenOdd(ceed, false);
            CeedBasisApply(basis, num_elem, t_mode, CEED_EVAL_INTERP, u, v_without_split);

            // -- Ensure split and non-split match
            {
              const CeedScalar *v_with_array;
              const CeedScalar *v_without_array;

              CeedVectorGetArrayRead(v_with_split, CEED_MEM_HOST, &v_with_array);
              CeedVectorGetArrayRead(v_without_split, CEED_MEM_HOST, &v_without_array);
              for (CeedInt j = 0; j < num_comp * (num_v / dim) * num_elem; j++) {
                if (fabs(v_with_array[j] - v_without_array[j]) > 100 * CEED_EPSILON * (fabs(v_without_array[j]) + 1.0)) {
                  printf("[%" CeedInt_FMT ", p %" CeedInt_FMT ", q %" CeedInt_FMT ", kind %" CeedInt_FMT ", interp %s] %f != %f\n", dim, p, q, kind,
                         t_mode ? "transpose" : "notranspose", v_with_array[j], v_without_array[j]);
                }
              }
              CeedVectorRestoreArrayRead(v_with_split, &v_with_array);
              CeedVectorRestoreArrayRead(v_without_split, &v_without_array);
            }

            // Check Grad

            // -- Force use of even-odd split
            CeedSetContractUseEvenOdd(ceed, true);
            CeedBasisApply(basis, num_elem, t_mode, CEED_EVAL_GRAD, u, v_with_split);

            // -- Force no use of even-odd split
            CeedSetContractUseEvenOdd(ceed, false);
            CeedBasisApply(basis, num_elem, t_mode, CEED_EVAL_GRAD, u, v_without_split);

            // -- Ensure split and non-split match
            {
              const CeedScalar *v_with_array;
              const CeedScalar *v_without_array;

              CeedVectorGetArrayRead(v_with_split, CEED_MEM_HOST, &v_with_array);
              CeedVectorGetArrayRead(v_without_split, CEED_MEM_HOST, &v_without_array);
              for (CeedInt j = 0; j < num_comp * num_v * num_elem; j++) {
                if (fabs(v_with_array[j] - v_without_array[j]) > 100 * CEED_EPSILON * (fabs(v_without_array[j]) + 1.0)) {
                  printf("[%" CeedInt_FMT ", p %" CeedInt_FMT ", q %" CeedInt_FMT ", kind %" CeedInt_FMT ", grad %s] %f != %f\n", dim, p, q, kind,
                         t_mode ? "transpose" : "notranspose", v_with_array[j], v_without_array[j]);
                }
              }
              CeedVectorRestoreArrayRead(v_with_split, &v_with_array);
              CeedVectorRestoreArrayRead(v_without_split, &v_without_array);
            }
            // Cleanup
            CeedVectorDestroy(&u);
            CeedVectorDestroy(&v_with_split);
            CeedVectorDestroy(&v_without_split);
          }
          CeedBasisDestroy(&basis);
        }
      }
    }
  }
  CeedDestroy(&ceed);
  return 0;
}
