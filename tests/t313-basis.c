/// @file
/// Test interpolation in multiple dimensions
/// \test Test interpolation in multiple dimensions
#include <ceed.h>
#include <math.h>

static CeedScalar Eval(CeedInt dim, const CeedScalar x[]) {
  CeedScalar result = 1, center = 0.1;
  for (CeedInt d = 0; d < dim; d++) {
    result *= tanh(x[d] - center);
    center += 0.1;
  }
  return result;
}

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);
  for (CeedInt dim = 1; dim <= 3; dim++) {
    CeedVector x, x_q, u, u_q;
    CeedBasis  basis_x_lobatto, basis_u_lobatto, basis_x_gauss, basis_u_gauss;
    CeedInt    q = 10, q_dim = CeedIntPow(q, dim), x_dim = CeedIntPow(2, dim);

    CeedVectorCreate(ceed, x_dim * dim, &x);
    {
      CeedScalar x_array[x_dim * dim];

      for (CeedInt d = 0; d < dim; d++) {
        for (CeedInt i = 0; i < x_dim; i++) x_array[d * x_dim + i] = (i % CeedIntPow(2, dim - d)) / CeedIntPow(2, dim - d - 1) ? 1 : -1;
      }
      CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
    }
    CeedVectorCreate(ceed, q_dim * dim, &x_q);
    CeedVectorSetValue(x_q, 0);
    CeedVectorCreate(ceed, q_dim, &u);
    CeedVectorSetValue(u, 0);
    CeedVectorCreate(ceed, q_dim, &u_q);

    CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, q, CEED_GAUSS_LOBATTO, &basis_x_lobatto);
    CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, q, q, CEED_GAUSS_LOBATTO, &basis_u_lobatto);

    CeedBasisApply(basis_x_lobatto, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x, x_q);

    {
      const CeedScalar *x_q_array;
      CeedScalar        u_q_array[q_dim];

      CeedVectorGetArrayRead(x_q, CEED_MEM_HOST, &x_q_array);
      for (CeedInt i = 0; i < q_dim; i++) {
        CeedScalar coord[dim];
        for (CeedInt d = 0; d < dim; d++) coord[d] = x_q_array[d * q_dim + i];
        u_q_array[i] = Eval(dim, coord);
      }
      CeedVectorRestoreArrayRead(x_q, &x_q_array);
      CeedVectorSetArray(u_q, CEED_MEM_HOST, CEED_COPY_VALUES, u_q_array);
    }

    // This operation is the identity because the quadrature is collocated
    CeedBasisApply(basis_u_lobatto, 1, CEED_TRANSPOSE, CEED_EVAL_INTERP, u_q, u);

    CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, q, CEED_GAUSS, &basis_x_gauss);
    CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, q, q, CEED_GAUSS, &basis_u_gauss);

    CeedBasisApply(basis_x_gauss, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x, x_q);
    CeedBasisApply(basis_u_gauss, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, u, u_q);

    {
      const CeedScalar *x_q_array, *u_array;

      CeedVectorGetArrayRead(x_q, CEED_MEM_HOST, &x_q_array);
      CeedVectorGetArrayRead(u_q, CEED_MEM_HOST, &u_array);
      for (CeedInt i = 0; i < q_dim; i++) {
        CeedScalar coord[dim];
        for (CeedInt d = 0; d < dim; d++) coord[d] = x_q_array[d * q_dim + i];
        CeedScalar fx = Eval(dim, coord);
        if (fabs(u_array[i] - fx) > 1E-4) {
          // LCOV_EXCL_START
          printf("[%" CeedInt_FMT "] %f != %f=f(%f", dim, u_array[i], fx, coord[0]);
          for (CeedInt d = 1; d < dim; d++) printf(",%f", coord[d]);
          puts(")");
          // LCOV_EXCL_STOP
        }
      }
      CeedVectorRestoreArrayRead(x_q, &x_q_array);
      CeedVectorRestoreArrayRead(u_q, &u_array);
    }

    CeedVectorDestroy(&x);
    CeedVectorDestroy(&x_q);
    CeedVectorDestroy(&u);
    CeedVectorDestroy(&u_q);
    CeedBasisDestroy(&basis_x_lobatto);
    CeedBasisDestroy(&basis_u_lobatto);
    CeedBasisDestroy(&basis_x_gauss);
    CeedBasisDestroy(&basis_u_gauss);
  }
  CeedDestroy(&ceed);
  return 0;
}
