/// @file
/// Test creation and use of FDM element inverse
/// \test Test creation and use of FDM element inverse
#include "t541-operator.h"

#include <ceed.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restr_x_i, elem_restr_u_i, elem_restr_qd_i;
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_setup_diff, qf_apply;
  CeedOperator        op_setup_diff, op_apply, op_inv;
  CeedVector          q_data_diff, X, U, V, W;
  CeedInt             num_elem = 1, P = 4, Q = 5, dim = 2;
  CeedInt             num_dofs = P * P, num_qpts = num_elem * Q * Q, q_data_size = dim * (dim + 1) / 2;
  CeedScalar          x[dim * num_elem * (2 * 2)];

  CeedInit(argv[1], &ceed);

  // Test skipped if using single precision
  if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Test not implemented in single precision");

  // DoF Coordinates
  for (CeedInt i = 0; i < 2; i++) {
    for (CeedInt j = 0; j < 2; j++) {
      x[i + j * 2 + 0 * 4] = i;
      x[i + j * 2 + 1 * 4] = j;
    }
  }
  CeedVectorCreate(ceed, dim * num_elem * (2 * 2), &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);

  // Qdata Vector
  CeedVectorCreate(ceed, q_data_size * num_qpts, &q_data_diff);

  // Element Setup

  // Restrictions
  CeedInt strides_x[3] = {1, 2 * 2, 2 * 2 * dim};
  CeedElemRestrictionCreateStrided(ceed, num_elem, 2 * 2, dim, dim * num_elem * 2 * 2, strides_x, &elem_restr_x_i);

  CeedInt strides_u[3] = {1, P * P, P * P};
  CeedElemRestrictionCreateStrided(ceed, num_elem, P * P, 1, num_dofs, strides_u, &elem_restr_u_i);

  CeedInt strides_qd[3] = {1, Q * Q, q_data_size * Q * Q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, Q * Q, q_data_size, num_qpts * q_data_size, strides_qd, &elem_restr_qd_i);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, Q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, P, Q, CEED_GAUSS, &basis_u);

  // QFunction - setup diff
  CeedQFunctionCreateInterior(ceed, 1, setup_diff, setup_diff_loc, &qf_setup_diff);
  CeedQFunctionAddInput(qf_setup_diff, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup_diff, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup_diff, "qdata", q_data_size, CEED_EVAL_NONE);

  // Operator - setup diff
  CeedOperatorCreate(ceed, qf_setup_diff, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_diff);
  CeedOperatorSetField(op_setup_diff, "dx", elem_restr_x_i, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_diff, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_diff, "qdata", elem_restr_qd_i, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Apply Setup Operator
  CeedOperatorApply(op_setup_diff, X, q_data_diff, CEED_REQUEST_IMMEDIATE);

  // QFunction - apply
  CeedQFunctionCreateInterior(ceed, 1, apply, apply_loc, &qf_apply);
  CeedQFunctionAddInput(qf_apply, "u", dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_apply, "qdata_diff", q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_apply, "v", dim, CEED_EVAL_GRAD);

  // Operator - apply
  CeedOperatorCreate(ceed, qf_apply, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_apply);
  CeedOperatorSetField(op_apply, "u", elem_restr_u_i, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "qdata_diff", elem_restr_qd_i, CEED_BASIS_COLLOCATED, q_data_diff);
  CeedOperatorSetField(op_apply, "v", elem_restr_u_i, basis_u, CEED_VECTOR_ACTIVE);

  // Create FDM element inverse
  CeedOperatorCreateFDMElementInverse(op_apply, &op_inv, CEED_REQUEST_IMMEDIATE);

  // Create vectors
  CeedVectorCreate(ceed, num_dofs, &U);
  CeedVectorCreate(ceed, num_dofs, &V);
  CeedVectorCreate(ceed, num_dofs, &W);

  // Create Schur complement for element corners
  CeedScalar S[16];
  for (CeedInt i = 0; i < 4; i++) {
    CeedScalar *u;
    CeedVectorSetValue(U, 0.0);
    CeedVectorGetArray(U, CEED_MEM_HOST, &u);
    switch (i) {
      case 0:
        u[0] = 1.0;
        break;
      case 1:
        u[P - 1] = 1.0;
        break;
      case 2:
        u[P * P - P] = 1.0;
        break;
      case 3:
        u[P * P - 1] = 1.0;
        break;
    }
    CeedVectorRestoreArray(U, &u);

    CeedOperatorApply(op_inv, U, V, CEED_REQUEST_IMMEDIATE);

    const CeedScalar *v;
    CeedVectorGetArrayRead(V, CEED_MEM_HOST, &v);
    S[0 * 4 + i] = -v[0];
    S[1 * 4 + i] = -v[P - 1];
    S[2 * 4 + i] = -v[P * P - P];
    S[3 * 4 + i] = -v[P * P - 1];
    CeedVectorRestoreArrayRead(V, &v);
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
    CeedScalar nodes[P];
    CeedLobattoQuadrature(P, nodes, NULL);
    CeedScalar *u;
    CeedVectorGetArray(U, CEED_MEM_HOST, &u);
    for (CeedInt i = 0; i < P; i++) {
      for (CeedInt j = 0; j < P; j++) u[i * P + j] = -(nodes[i] - 1.0) * (nodes[i] + 1.0) - (nodes[j] - 1.0) * (nodes[j] + 1.0);
    }
    CeedVectorRestoreArray(U, &u);
  }

  // Apply original operator
  CeedOperatorApply(op_apply, U, V, CEED_REQUEST_IMMEDIATE);

  // Apply FDM element inverse
  {
    // -- Zero corners
    CeedScalar *v;
    CeedVectorGetArray(V, CEED_MEM_HOST, &v);
    v[0]         = 0.0;
    v[P - 1]     = 0.0;
    v[P * P - P] = 0.0;
    v[P * P - 1] = 0.0;
    CeedVectorRestoreArray(V, &v);

    // -- Apply FDM inverse to interior
    CeedOperatorApply(op_inv, V, W, CEED_REQUEST_IMMEDIATE);

    // -- Pick off corners
    const CeedScalar *w;
    CeedScalar        w_Pi[4];
    CeedVectorGetArrayRead(W, CEED_MEM_HOST, &w);
    w_Pi[0] = w[0];
    w_Pi[1] = w[P - 1];
    w_Pi[2] = w[P * P - P];
    w_Pi[3] = w[P * P - 1];
    CeedVectorRestoreArrayRead(W, &w);

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
    CeedVectorGetArray(V, CEED_MEM_HOST, &v);
    v[0]         = v_Pi[0];
    v[P - 1]     = v_Pi[1];
    v[P * P - P] = v_Pi[2];
    v[P * P - 1] = v_Pi[3];
    CeedVectorRestoreArray(V, &v);

    // -- Apply full FDM inverse again
    CeedOperatorApply(op_inv, V, W, CEED_REQUEST_IMMEDIATE);
  }

  // Check output
  {
    const CeedScalar *u, *w;
    CeedVectorGetArrayRead(U, CEED_MEM_HOST, &u);
    CeedVectorGetArrayRead(W, CEED_MEM_HOST, &w);
    for (CeedInt i = 0; i < P; i++) {
      for (CeedInt j = 0; j < P; j++) {
        if (fabs(u[i * P + j] - w[i * P + j]) > 2e-3) {
          // LCOV_EXCL_START
          printf("[%" CeedInt_FMT ", %" CeedInt_FMT "] Error in inverse: %e != %e\n", i, j, w[i * P + j], u[i * P + j]);
          // LCOV_EXCL_STOP
        }
      }
    }
    CeedVectorRestoreArrayRead(U, &u);
    CeedVectorRestoreArrayRead(W, &w);
  }

  // Cleanup
  CeedQFunctionDestroy(&qf_setup_diff);
  CeedQFunctionDestroy(&qf_apply);
  CeedOperatorDestroy(&op_setup_diff);
  CeedOperatorDestroy(&op_apply);
  CeedOperatorDestroy(&op_inv);
  CeedElemRestrictionDestroy(&elem_restr_u_i);
  CeedElemRestrictionDestroy(&elem_restr_x_i);
  CeedElemRestrictionDestroy(&elem_restr_qd_i);
  CeedBasisDestroy(&basis_x);
  CeedBasisDestroy(&basis_u);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&q_data_diff);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedVectorDestroy(&W);
  CeedDestroy(&ceed);
  return 0;
}
