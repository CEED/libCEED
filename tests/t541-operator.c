/// @file
/// Test creation and use of FDM element inverse
/// \test Test creation and use of FDM element inverse
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "t541-operator.h"

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_x, elem_restriction_u, elem_restriction_q_data;
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_setup_diff, qf_apply;
  CeedOperator        op_setup_diff, op_apply, op_inverse;
  CeedVector          q_data_diff, x, u, v, w;
  CeedInt             num_elem = 1, p = 4, q = 5, dim = 2;
  CeedInt             num_dofs = p * p, num_qpts = num_elem * q * q, q_data_size = dim * (dim + 1) / 2;

  CeedInit(argv[1], &ceed);

  // Test skipped if using single precision
  if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Test not implemented in single precision");

  // Vectors
  CeedVectorCreate(ceed, dim * num_elem * (2 * 2), &x);
  {
    CeedScalar x_array[dim * num_elem * (2 * 2)];

    for (CeedInt i = 0; i < 2; i++) {
      for (CeedInt j = 0; j < 2; j++) {
        x_array[i + j * 2 + 0 * 4] = i;
        x_array[i + j * 2 + 1 * 4] = j;
      }
    }
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  CeedVectorCreate(ceed, num_dofs, &u);
  CeedVectorCreate(ceed, num_dofs, &v);
  CeedVectorCreate(ceed, num_dofs, &w);
  CeedVectorCreate(ceed, q_data_size * num_qpts, &q_data_diff);

  // Restrictions
  CeedInt strides_x[3] = {1, 2 * 2, 2 * 2 * dim};
  CeedElemRestrictionCreateStrided(ceed, num_elem, 2 * 2, dim, dim * num_elem * 2 * 2, strides_x, &elem_restriction_x);

  CeedInt strides_u[3] = {1, p * p, p * p};
  CeedElemRestrictionCreateStrided(ceed, num_elem, p * p, 1, num_dofs, strides_u, &elem_restriction_u);

  CeedInt strides_q_data[3] = {1, q * q, q_data_size * q * q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, q * q, q_data_size, num_qpts * q_data_size, strides_q_data, &elem_restriction_q_data);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, p, q, CEED_GAUSS, &basis_u);

  // QFunction - setup diff
  CeedQFunctionCreateInterior(ceed, 1, setup_diff, setup_diff_loc, &qf_setup_diff);
  CeedQFunctionAddInput(qf_setup_diff, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup_diff, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup_diff, "q data", q_data_size, CEED_EVAL_NONE);

  // Operator - setup diff
  CeedOperatorCreate(ceed, qf_setup_diff, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_diff);
  CeedOperatorSetField(op_setup_diff, "dx", elem_restriction_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_diff, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_diff, "q data", elem_restriction_q_data, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Apply Setup Operator
  CeedOperatorApply(op_setup_diff, x, q_data_diff, CEED_REQUEST_IMMEDIATE);

  // QFunction - apply
  CeedQFunctionCreateInterior(ceed, 1, apply, apply_loc, &qf_apply);
  CeedQFunctionAddInput(qf_apply, "u", dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_apply, "q data diff", q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_apply, "v", dim, CEED_EVAL_GRAD);

  // Operator - apply
  CeedOperatorCreate(ceed, qf_apply, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_apply);
  CeedOperatorSetField(op_apply, "u", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "q data diff", elem_restriction_q_data, CEED_BASIS_COLLOCATED, q_data_diff);
  CeedOperatorSetField(op_apply, "v", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);

  // Create FDM element inverse
  CeedOperatorCreateFDMElementInverse(op_apply, &op_inverse, CEED_REQUEST_IMMEDIATE);

  // Create Schur complement for element corners
  CeedScalar S[16];
  for (CeedInt i = 0; i < 4; i++) {
    CeedScalar *u_array;

    CeedVectorSetValue(u, 0.0);
    CeedVectorGetArray(u, CEED_MEM_HOST, &u_array);
    switch (i) {
      case 0:
        u_array[0] = 1.0;
        break;
      case 1:
        u_array[p - 1] = 1.0;
        break;
      case 2:
        u_array[p * p - p] = 1.0;
        break;
      case 3:
        u_array[p * p - 1] = 1.0;
        break;
    }
    CeedVectorRestoreArray(u, &u_array);

    CeedOperatorApply(op_inverse, u, v, CEED_REQUEST_IMMEDIATE);

    const CeedScalar *v_array;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    S[0 * 4 + i] = -v_array[0];
    S[1 * 4 + i] = -v_array[p - 1];
    S[2 * 4 + i] = -v_array[p * p - p];
    S[3 * 4 + i] = -v_array[p * p - 1];
    CeedVectorRestoreArrayRead(v, &v_array);
  }
  CeedScalar S_inv[16];
  {
    CeedScalar det;
    S_inv[0] = S[5] * S[10] * S[15] - S[5] * S[11] * S[14] - S[9] * S[6] * S[15] + S[9] * S[7] * S[14] + S[13] * S[6] * S[11] - S[13] * S[7] * S[10];

    S_inv[4] = -S[4] * S[10] * S[15] + S[4] * S[11] * S[14] + S[8] * S[6] * S[15] - S[8] * S[7] * S[14] - S[12] * S[6] * S[11] + S[12] * S[7] * S[10];

    S_inv[8] = S[4] * S[9] * S[15] - S[4] * S[11] * S[13] - S[8] * S[5] * S[15] + S[8] * S[7] * S[13] + S[12] * S[5] * S[11] - S[12] * S[7] * S[9];

    S_inv[12] = -S[4] * S[9] * S[14] + S[4] * S[10] * S[13] + S[8] * S[5] * S[14] - S[8] * S[6] * S[13] - S[12] * S[5] * S[10] + S[12] * S[6] * S[9];

    S_inv[1] = -S[1] * S[10] * S[15] + S[1] * S[11] * S[14] + S[9] * S[2] * S[15] - S[9] * S[3] * S[14] - S[13] * S[2] * S[11] + S[13] * S[3] * S[10];

    S_inv[5] = S[0] * S[10] * S[15] - S[0] * S[11] * S[14] - S[8] * S[2] * S[15] + S[8] * S[3] * S[14] + S[12] * S[2] * S[11] - S[12] * S[3] * S[10];

    S_inv[9] = -S[0] * S[9] * S[15] + S[0] * S[11] * S[13] + S[8] * S[1] * S[15] - S[8] * S[3] * S[13] - S[12] * S[1] * S[11] + S[12] * S[3] * S[9];

    S_inv[13] = S[0] * S[9] * S[14] - S[0] * S[10] * S[13] - S[8] * S[1] * S[14] + S[8] * S[2] * S[13] + S[12] * S[1] * S[10] - S[12] * S[2] * S[9];

    S_inv[2] = S[1] * S[6] * S[15] - S[1] * S[7] * S[14] - S[5] * S[2] * S[15] + S[5] * S[3] * S[14] + S[13] * S[2] * S[7] - S[13] * S[3] * S[6];

    S_inv[6] = -S[0] * S[6] * S[15] + S[0] * S[7] * S[14] + S[4] * S[2] * S[15] - S[4] * S[3] * S[14] - S[12] * S[2] * S[7] + S[12] * S[3] * S[6];

    S_inv[10] = S[0] * S[5] * S[15] - S[0] * S[7] * S[13] - S[4] * S[1] * S[15] + S[4] * S[3] * S[13] + S[12] * S[1] * S[7] - S[12] * S[3] * S[5];

    S_inv[14] = -S[0] * S[5] * S[14] + S[0] * S[6] * S[13] + S[4] * S[1] * S[14] - S[4] * S[2] * S[13] - S[12] * S[1] * S[6] + S[12] * S[2] * S[5];

    S_inv[3] = -S[1] * S[6] * S[11] + S[1] * S[7] * S[10] + S[5] * S[2] * S[11] - S[5] * S[3] * S[10] - S[9] * S[2] * S[7] + S[9] * S[3] * S[6];

    S_inv[7] = S[0] * S[6] * S[11] - S[0] * S[7] * S[10] - S[4] * S[2] * S[11] + S[4] * S[3] * S[10] + S[8] * S[2] * S[7] - S[8] * S[3] * S[6];

    S_inv[11] = -S[0] * S[5] * S[11] + S[0] * S[7] * S[9] + S[4] * S[1] * S[11] - S[4] * S[3] * S[9] - S[8] * S[1] * S[7] + S[8] * S[3] * S[5];

    S_inv[15] = S[0] * S[5] * S[10] - S[0] * S[6] * S[9] - S[4] * S[1] * S[10] + S[4] * S[2] * S[9] + S[8] * S[1] * S[6] - S[8] * S[2] * S[5];

    det = 1 / (S[0] * S_inv[0] + S[1] * S_inv[4] + S[2] * S_inv[8] + S[3] * S_inv[12]);

    for (CeedInt i = 0; i < 16; i++) S_inv[i] *= det;
  }

  // Set initial values
  {
    CeedScalar  nodes[p];
    CeedScalar *u_array;

    CeedLobattoQuadrature(p, nodes, NULL);
    CeedVectorGetArray(u, CEED_MEM_HOST, &u_array);
    for (CeedInt i = 0; i < p; i++) {
      for (CeedInt j = 0; j < p; j++) u_array[i * p + j] = -(nodes[i] - 1.0) * (nodes[i] + 1.0) - (nodes[j] - 1.0) * (nodes[j] + 1.0);
    }
    CeedVectorRestoreArray(u, &u_array);
  }

  // Apply original operator
  CeedOperatorApply(op_apply, u, v, CEED_REQUEST_IMMEDIATE);

  // Apply FDM element inverse
  {
    // -- Zero corners
    CeedScalar *v_array;

    CeedVectorGetArray(v, CEED_MEM_HOST, &v_array);
    v_array[0]         = 0.0;
    v_array[p - 1]     = 0.0;
    v_array[p * p - p] = 0.0;
    v_array[p * p - 1] = 0.0;
    CeedVectorRestoreArray(v, &v_array);

    // -- Apply FDM inverse to interior
    CeedOperatorApply(op_inverse, v, w, CEED_REQUEST_IMMEDIATE);

    // -- Pick off corners
    const CeedScalar *w_array;
    CeedScalar        w_Pi[4];

    CeedVectorGetArrayRead(w, CEED_MEM_HOST, &w_array);
    w_Pi[0] = w_array[0];
    w_Pi[1] = w_array[p - 1];
    w_Pi[2] = w_array[p * p - p];
    w_Pi[3] = w_array[p * p - 1];
    CeedVectorRestoreArrayRead(w, &w_array);

    // -- Apply inverse of Schur complement
    CeedScalar v_Pi[4];
    for (CeedInt i = 0; i < 4; i++) {
      CeedScalar sum = 0.0;
      for (CeedInt j = 0; j < 4; j++) {
        sum += w_Pi[j] * S_inv[i * 4 + j];
      }
      v_Pi[i] = sum;
    }

    // -- Set corners
    CeedVectorGetArray(v, CEED_MEM_HOST, &v_array);
    v_array[0]         = v_Pi[0];
    v_array[p - 1]     = v_Pi[1];
    v_array[p * p - p] = v_Pi[2];
    v_array[p * p - 1] = v_Pi[3];
    CeedVectorRestoreArray(v, &v_array);

    // -- Apply full FDM inverse again
    CeedOperatorApply(op_inverse, v, w, CEED_REQUEST_IMMEDIATE);
  }

  // Check output
  {
    const CeedScalar *u_array, *w_array;
    CeedVectorGetArrayRead(u, CEED_MEM_HOST, &u_array);
    CeedVectorGetArrayRead(w, CEED_MEM_HOST, &w_array);
    for (CeedInt i = 0; i < p; i++) {
      for (CeedInt j = 0; j < p; j++) {
        if (fabs(u_array[i * p + j] - w_array[i * p + j]) > 2e-3) {
          // LCOV_EXCL_START
          printf("[%" CeedInt_FMT ", %" CeedInt_FMT "] Error in inverse: %e != %e\n", i, j, w_array[i * p + j], u_array[i * p + j]);
          // LCOV_EXCL_STOP
        }
      }
    }
    CeedVectorRestoreArrayRead(u, &u_array);
    CeedVectorRestoreArrayRead(w, &w_array);
  }

  // Cleanup
  CeedVectorDestroy(&x);
  CeedVectorDestroy(&q_data_diff);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedVectorDestroy(&w);
  CeedElemRestrictionDestroy(&elem_restriction_u);
  CeedElemRestrictionDestroy(&elem_restriction_x);
  CeedElemRestrictionDestroy(&elem_restriction_q_data);
  CeedBasisDestroy(&basis_x);
  CeedBasisDestroy(&basis_u);
  CeedQFunctionDestroy(&qf_setup_diff);
  CeedQFunctionDestroy(&qf_apply);
  CeedOperatorDestroy(&op_setup_diff);
  CeedOperatorDestroy(&op_apply);
  CeedOperatorDestroy(&op_inverse);
  CeedDestroy(&ceed);
  return 0;
}
