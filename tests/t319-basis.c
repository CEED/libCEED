/// @file
/// Test projection interp and grad in multiple dimensions
/// \test Test projection interp and grad in multiple dimensions
#include <ceed.h>
#include <math.h>

static CeedScalar Eval(CeedInt dim, const CeedScalar x[]) {
  CeedScalar result = (x[0] + 0.1) * (x[0] + 0.1);
  if (dim > 1) result += (x[1] + 0.2) * (x[1] + 0.2);
  if (dim > 2) result += -(x[2] + 0.3) * (x[2] + 0.3);
  return result;
}

static CeedScalar EvalGrad(CeedInt dim, const CeedScalar x[]) {
  switch (dim) {
    case 0:
      return 2 * x[0] + 0.2;
    case 1:
      return 2 * x[1] + 0.4;
    default:
      return -2 * x[2] - 0.6;
  }
}

static CeedScalar GetTolerance(CeedScalarType scalar_type, int dim) {
  CeedScalar tol;
  if (scalar_type == CEED_SCALAR_FP32) {
    if (dim == 3) tol = 1.e-4;
    else tol = 1.e-5;
  } else {
    tol = 1.e-11;
  }
  return tol;
}

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);

  for (CeedInt dim = 1; dim <= 3; dim++) {
    CeedVector        X_corners, X_from, X_to, U_from, U_to, dU_to;
    CeedBasis         basis_x, basis_from, basis_to, basis_project;
    CeedInt           P_from = 5, P_to = 6, Q = 7, X_dim = CeedIntPow(2, dim), P_from_dim = CeedIntPow(P_from, dim), P_to_dim = CeedIntPow(P_to, dim);
    CeedScalar        x[X_dim * dim], u_from[P_from_dim];
    const CeedScalar *u_to, *du_to, *x_from, *x_to;

    for (CeedInt d = 0; d < dim; d++) {
      for (CeedInt i = 0; i < X_dim; i++) x[X_dim * d + i] = (i % CeedIntPow(2, dim - d)) / CeedIntPow(2, dim - d - 1) ? 1 : -1;
    }

    CeedVectorCreate(ceed, X_dim * dim, &X_corners);
    CeedVectorSetArray(X_corners, CEED_MEM_HOST, CEED_USE_POINTER, x);
    CeedVectorCreate(ceed, P_from_dim * dim, &X_from);
    CeedVectorCreate(ceed, P_to_dim * dim, &X_to);
    CeedVectorCreate(ceed, P_from_dim, &U_from);
    CeedVectorSetValue(U_from, 0);
    CeedVectorCreate(ceed, P_to_dim, &U_to);
    CeedVectorSetValue(U_to, 0);
    CeedVectorCreate(ceed, P_to_dim * dim, &dU_to);
    CeedVectorSetValue(dU_to, 0);

    // Get nodal coordinates
    CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, P_from, CEED_GAUSS_LOBATTO, &basis_x);
    CeedBasisApply(basis_x, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, X_corners, X_from);
    CeedBasisDestroy(&basis_x);
    CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, P_to, CEED_GAUSS_LOBATTO, &basis_x);
    CeedBasisApply(basis_x, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, X_corners, X_to);
    CeedBasisDestroy(&basis_x);

    // Create U and projection bases
    CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, P_from, Q, CEED_GAUSS, &basis_from);
    CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, P_to, Q, CEED_GAUSS, &basis_to);
    CeedBasisCreateProjection(basis_from, basis_to, &basis_project);

    // Setup coarse solution
    CeedVectorGetArrayRead(X_from, CEED_MEM_HOST, &x_from);
    for (CeedInt i = 0; i < P_from_dim; i++) {
      CeedScalar xx[dim];
      for (CeedInt d = 0; d < dim; d++) xx[d] = x_from[P_from_dim * d + i];
      u_from[i] = Eval(dim, xx);
    }
    CeedVectorRestoreArrayRead(X_from, &x_from);
    CeedVectorSetArray(U_from, CEED_MEM_HOST, CEED_USE_POINTER, u_from);

    // Project to fine basis
    CeedBasisApply(basis_project, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, U_from, U_to);

    // Check solution
    CeedVectorGetArrayRead(U_to, CEED_MEM_HOST, &u_to);
    CeedVectorGetArrayRead(X_to, CEED_MEM_HOST, &x_to);
    CeedScalar tol = GetTolerance(CEED_SCALAR_TYPE, dim);
    for (CeedInt i = 0; i < P_to_dim; i++) {
      CeedScalar xx[dim];
      for (CeedInt d = 0; d < dim; d++) xx[d] = x_to[d * P_to_dim + i];
      const CeedScalar u = Eval(dim, xx);
      if (fabs(u - u_to[i]) > tol) printf("[%" CeedInt_FMT ", %" CeedInt_FMT "] %f != %f\n", dim, i, u_to[i], u);
    }
    CeedVectorRestoreArrayRead(X_to, &x_to);
    CeedVectorRestoreArrayRead(U_to, &u_to);

    // Project and take gradient
    CeedBasisApply(basis_project, 1, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, U_from, dU_to);

    // Check solution
    CeedVectorGetArrayRead(dU_to, CEED_MEM_HOST, &du_to);
    CeedVectorGetArrayRead(X_to, CEED_MEM_HOST, &x_to);
    for (CeedInt i = 0; i < P_to_dim; i++) {
      CeedScalar xx[dim];
      for (CeedInt d = 0; d < dim; d++) xx[d] = x_to[P_to_dim * d + i];
      for (CeedInt d = 0; d < dim; d++) {
        const CeedScalar du = EvalGrad(d, xx);
        if (fabs(du - du_to[P_to_dim * (dim - 1 - d) + i]) > tol) {
          // LCOV_EXCL_START
          printf("[%" CeedInt_FMT ", %" CeedInt_FMT ", %" CeedInt_FMT "] %f != %f\n", dim, i, d, du_to[P_to_dim * (dim - 1 - d) + i], du);
          // LCOV_EXCL_STOP
        }
      }
    }
    CeedVectorRestoreArrayRead(X_to, &x_to);
    CeedVectorRestoreArrayRead(dU_to, &du_to);

    CeedVectorDestroy(&X_corners);
    CeedVectorDestroy(&X_from);
    CeedVectorDestroy(&X_to);
    CeedVectorDestroy(&U_from);
    CeedVectorDestroy(&U_to);
    CeedVectorDestroy(&dU_to);
    CeedBasisDestroy(&basis_from);
    CeedBasisDestroy(&basis_to);
    CeedBasisDestroy(&basis_project);
  }
  CeedDestroy(&ceed);
  return 0;
}
