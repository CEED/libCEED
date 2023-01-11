/// @file
/// Test collocated grad in multiple dimensions
/// \test Test collocated grad in multiple dimensions
#include <ceed.h>
#include <math.h>

static CeedScalar Eval(CeedInt dim, const CeedScalar x[]) {
  CeedScalar result = tanh(x[0] + 0.1);
  if (dim > 1) result += atan(x[1] + 0.2);
  if (dim > 2) result += exp(-(x[2] + 0.3) * (x[2] + 0.3));
  return result;
}

static CeedScalar GetTolerance(CeedScalarType scalar_type, int dim) {
  CeedScalar tol;
  if (scalar_type == CEED_SCALAR_FP32) {
    if (dim == 3) tol = 1.e-3;
    else tol = 1.e-4;
  } else {
    tol = 1.e-11;
  }
  return tol;
}

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);

  for (CeedInt dim = 1; dim <= 3; dim++) {
    CeedVector        X, X_q, U, U_q, ones, grad_T_ones;
    CeedBasis         basis_x_lobatto, basis_u_gauss;
    CeedInt           P = 8, Q = 8, P_dim = CeedIntPow(P, dim), Qdim_ = CeedIntPow(Q, dim), X_dim = CeedIntPow(2, dim);
    CeedScalar        x[X_dim * dim], u[P_dim];
    const CeedScalar *x_q, *u_q, *grad_t_ones_array;
    CeedScalar        sum_1 = 0, sum_2 = 0;

    for (CeedInt d = 0; d < dim; d++) {
      for (CeedInt i = 0; i < X_dim; i++) x[d * X_dim + i] = (i % CeedIntPow(2, dim - d)) / CeedIntPow(2, dim - d - 1) ? 1 : -1;
    }

    CeedVectorCreate(ceed, X_dim * dim, &X);
    CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);
    CeedVectorCreate(ceed, P_dim * dim, &X_q);
    CeedVectorSetValue(X_q, 0);
    CeedVectorCreate(ceed, P_dim, &U);
    CeedVectorCreate(ceed, Qdim_ * dim, &U_q);
    CeedVectorSetValue(U_q, 0);
    CeedVectorCreate(ceed, Qdim_ * dim, &ones);
    CeedVectorSetValue(ones, 1);
    CeedVectorCreate(ceed, P_dim, &grad_T_ones);
    CeedVectorSetValue(grad_T_ones, 0);

    // Get function values at quadrature points
    CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, P, CEED_GAUSS_LOBATTO, &basis_x_lobatto);
    CeedBasisApply(basis_x_lobatto, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, X, X_q);

    CeedVectorGetArrayRead(X_q, CEED_MEM_HOST, &x_q);
    for (CeedInt i = 0; i < P_dim; i++) {
      CeedScalar xx[dim];
      for (CeedInt d = 0; d < dim; d++) xx[d] = x_q[d * P_dim + i];
      u[i] = Eval(dim, xx);
    }
    CeedVectorRestoreArrayRead(X_q, &x_q);
    CeedVectorSetArray(U, CEED_MEM_HOST, CEED_USE_POINTER, u);

    // Calculate G u at quadrature points, G' * 1 at dofs
    CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, P, Q, CEED_GAUSS_LOBATTO, &basis_u_gauss);
    CeedBasisApply(basis_u_gauss, 1, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, U, U_q);
    CeedBasisApply(basis_u_gauss, 1, CEED_TRANSPOSE, CEED_EVAL_GRAD, ones, grad_T_ones);

    // Check if 1' * G * u = u' * (G' * 1)
    CeedVectorGetArrayRead(grad_T_ones, CEED_MEM_HOST, &grad_t_ones_array);
    CeedVectorGetArrayRead(U_q, CEED_MEM_HOST, &u_q);
    for (CeedInt i = 0; i < P_dim; i++) sum_1 += grad_t_ones_array[i] * u[i];
    for (CeedInt i = 0; i < dim * Qdim_; i++) sum_2 += u_q[i];
    CeedVectorRestoreArrayRead(grad_T_ones, &grad_t_ones_array);
    CeedVectorRestoreArrayRead(U_q, &u_q);
    CeedScalar tol = GetTolerance(CEED_SCALAR_TYPE, dim);
    if (fabs(sum_1 - sum_2) > tol) printf("[%" CeedInt_FMT "] %f != %f\n", dim, sum_1, sum_2);

    CeedVectorDestroy(&X);
    CeedVectorDestroy(&X_q);
    CeedVectorDestroy(&U);
    CeedVectorDestroy(&U_q);
    CeedVectorDestroy(&ones);
    CeedVectorDestroy(&grad_T_ones);
    CeedBasisDestroy(&basis_x_lobatto);
    CeedBasisDestroy(&basis_u_gauss);
  }
  CeedDestroy(&ceed);
  return 0;
}
