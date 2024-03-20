/// @file
/// Test grad in multiple dimensions
/// \test Test grad in multiple dimensions
#include <ceed.h>
#include <math.h>
#include <stdio.h>

static CeedScalar Eval(CeedInt dim, const CeedScalar x[]) {
  CeedScalar result = tanh(x[0] + 0.1);
  if (dim > 1) result += atan(x[1] + 0.2);
  if (dim > 2) result += exp(-(x[2] + 0.3) * (x[2] + 0.3));
  return result;
}

static CeedScalar GetTolerance(CeedScalarType scalar_type, int dim) {
  CeedScalar tol;
  if (scalar_type == CEED_SCALAR_FP32) {
    if (dim == 3) tol = 0.05;
    else tol = 1.e-3;
  } else {
    tol = 1.e-10;
  }
  return tol;
}

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);

  for (CeedInt dim = 1; dim <= 3; dim++) {
    CeedVector x, x_q, u, u_q, ones, v;
    CeedBasis  basis_x_lobatto, basis_u_gauss;
    CeedInt    p = 8, q = 10, p_dim = CeedIntPow(p, dim), q_dim = CeedIntPow(q, dim), x_dim = CeedIntPow(2, dim);
    CeedScalar sum_1 = 0, sum_2 = 0;

    CeedVectorCreate(ceed, x_dim * dim, &x);
    {
      CeedScalar x_array[x_dim * dim];

      for (CeedInt d = 0; d < dim; d++) {
        for (CeedInt i = 0; i < x_dim; i++) x_array[d * x_dim + i] = (i % CeedIntPow(2, d + 1)) / CeedIntPow(2, d) ? 1 : -1;
      }
      CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
    }
    CeedVectorCreate(ceed, p_dim * dim, &x_q);
    CeedVectorSetValue(x_q, 0);
    CeedVectorCreate(ceed, p_dim, &u);
    CeedVectorCreate(ceed, q_dim * dim, &u_q);
    CeedVectorSetValue(u_q, 0);
    CeedVectorCreate(ceed, q_dim * dim, &ones);
    CeedVectorSetValue(ones, 1);
    CeedVectorCreate(ceed, p_dim, &v);
    CeedVectorSetValue(v, 0);

    // Get function values at quadrature points
    CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, p, CEED_GAUSS_LOBATTO, &basis_x_lobatto);
    CeedBasisApply(basis_x_lobatto, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x, x_q);

    {
      const CeedScalar *x_q_array;
      CeedScalar        u_array[p_dim];

      CeedVectorGetArrayRead(x_q, CEED_MEM_HOST, &x_q_array);
      for (CeedInt i = 0; i < p_dim; i++) {
        CeedScalar coord[dim];

        for (CeedInt d = 0; d < dim; d++) coord[d] = x_q_array[d * p_dim + i];
        u_array[i] = Eval(dim, coord);
      }
      CeedVectorRestoreArrayRead(x_q, &x_q_array);
      CeedVectorSetArray(u, CEED_MEM_HOST, CEED_COPY_VALUES, u_array);
    }

    // Calculate G u at quadrature points, G' * 1 at dofs
    CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, p, q, CEED_GAUSS, &basis_u_gauss);
    CeedBasisApply(basis_u_gauss, 1, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, u, u_q);
    CeedBasisApply(basis_u_gauss, 1, CEED_TRANSPOSE, CEED_EVAL_GRAD, ones, v);

    // Check if 1' * G * u = u' * (G' * 1)
    {
      const CeedScalar *v_array, *u_array, *u_q_array;

      CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
      CeedVectorGetArrayRead(u, CEED_MEM_HOST, &u_array);
      CeedVectorGetArrayRead(u_q, CEED_MEM_HOST, &u_q_array);
      for (CeedInt i = 0; i < p_dim; i++) sum_1 += v_array[i] * u_array[i];
      for (CeedInt i = 0; i < dim * q_dim; i++) sum_2 += u_q_array[i];
      CeedVectorRestoreArrayRead(v, &v_array);
      CeedVectorRestoreArrayRead(u, &u_array);
      CeedVectorRestoreArrayRead(u_q, &u_q_array);
    }
    {
      CeedScalarType scalar_type;

      CeedGetScalarType(&scalar_type);

      CeedScalar tol = GetTolerance(scalar_type, dim);

      if (fabs(sum_1 - sum_2) > tol) printf("[%" CeedInt_FMT "] %0.12f != %0.12f\n", dim, sum_1, sum_2);
    }

    CeedVectorDestroy(&x);
    CeedVectorDestroy(&x_q);
    CeedVectorDestroy(&u);
    CeedVectorDestroy(&u_q);
    CeedVectorDestroy(&ones);
    CeedVectorDestroy(&v);
    CeedBasisDestroy(&basis_x_lobatto);
    CeedBasisDestroy(&basis_u_gauss);
  }
  CeedDestroy(&ceed);
  return 0;
}
