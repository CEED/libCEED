/// @file
/// Test projection interp and grad in multiple dimensions
/// \test Test projection interp and grad in multiple dimensions
#include <ceed.h>
#include <math.h>
#include <stdio.h>

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
    CeedVector x_corners, x_from, x_to, u_from, u_to, du_to;
    CeedBasis  basis_x, basis_from, basis_to, basis_project;
    CeedInt    p_from = 5, p_to = 6, q = 7, x_dim = CeedIntPow(2, dim), p_from_dim = CeedIntPow(p_from, dim), p_to_dim = CeedIntPow(p_to, dim);
    CeedScalar tol;

    {
      CeedScalarType scalar_type;

      CeedGetScalarType(&scalar_type);
      tol = GetTolerance(scalar_type, dim);
    }
    CeedVectorCreate(ceed, x_dim * dim, &x_corners);
    {
      CeedScalar x_array[x_dim * dim];

      for (CeedInt d = 0; d < dim; d++) {
        for (CeedInt i = 0; i < x_dim; i++) x_array[x_dim * d + i] = (i % CeedIntPow(2, d + 1)) / CeedIntPow(2, d) ? 1 : -1;
      }
      CeedVectorSetArray(x_corners, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
    }
    CeedVectorCreate(ceed, p_from_dim * dim, &x_from);
    CeedVectorCreate(ceed, p_to_dim * dim, &x_to);
    CeedVectorCreate(ceed, p_from_dim, &u_from);
    CeedVectorSetValue(u_from, 0);
    CeedVectorCreate(ceed, p_to_dim, &u_to);
    CeedVectorSetValue(u_to, 0);
    CeedVectorCreate(ceed, p_to_dim * dim, &du_to);
    CeedVectorSetValue(du_to, 0);

    // Get nodal coordinates
    CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, p_from, CEED_GAUSS_LOBATTO, &basis_x);
    CeedBasisApply(basis_x, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_corners, x_from);
    CeedBasisDestroy(&basis_x);
    CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, p_to, CEED_GAUSS_LOBATTO, &basis_x);
    CeedBasisApply(basis_x, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_corners, x_to);
    CeedBasisDestroy(&basis_x);

    // Create U and projection bases
    CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, p_from, q, CEED_GAUSS, &basis_from);
    CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, p_to, q, CEED_GAUSS, &basis_to);
    CeedBasisCreateProjection(basis_from, basis_to, &basis_project);

    // Setup coarse solution
    {
      const CeedScalar *x_array;
      CeedScalar        u_array[p_from_dim];

      CeedVectorGetArrayRead(x_from, CEED_MEM_HOST, &x_array);
      for (CeedInt i = 0; i < p_from_dim; i++) {
        CeedScalar coord[dim];
        for (CeedInt d = 0; d < dim; d++) coord[d] = x_array[p_from_dim * d + i];
        u_array[i] = Eval(dim, coord);
      }
      CeedVectorRestoreArrayRead(x_from, &x_array);
      CeedVectorSetArray(u_from, CEED_MEM_HOST, CEED_COPY_VALUES, u_array);
    }

    // Project to fine basis
    CeedBasisApply(basis_project, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, u_from, u_to);

    // Check solution
    {
      const CeedScalar *x_array, *u_array;

      CeedVectorGetArrayRead(x_to, CEED_MEM_HOST, &x_array);
      CeedVectorGetArrayRead(u_to, CEED_MEM_HOST, &u_array);
      for (CeedInt i = 0; i < p_to_dim; i++) {
        CeedScalar coord[dim];
        for (CeedInt d = 0; d < dim; d++) coord[d] = x_array[d * p_to_dim + i];
        const CeedScalar u = Eval(dim, coord);
        if (fabs(u - u_array[i]) > tol) printf("[%" CeedInt_FMT ", %" CeedInt_FMT "] %f != %f\n", dim, i, u_array[i], u);
      }
      CeedVectorRestoreArrayRead(x_to, &x_array);
      CeedVectorRestoreArrayRead(u_to, &u_array);
    }

    // Project and take gradient
    CeedBasisApply(basis_project, 1, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, u_from, du_to);

    // Check solution
    {
      const CeedScalar *x_array, *du_array;

      CeedVectorGetArrayRead(x_to, CEED_MEM_HOST, &x_array);
      CeedVectorGetArrayRead(du_to, CEED_MEM_HOST, &du_array);
      for (CeedInt i = 0; i < p_to_dim; i++) {
        CeedScalar coord[dim];

        for (CeedInt d = 0; d < dim; d++) coord[d] = x_array[p_to_dim * d + i];
        for (CeedInt d = 0; d < dim; d++) {
          const CeedScalar du = EvalGrad(d, coord);

          if (fabs(du - du_array[p_to_dim * d + i]) > tol) {
            // LCOV_EXCL_START
            printf("[%" CeedInt_FMT ", %" CeedInt_FMT ", %" CeedInt_FMT "] %f != %f\n", dim, i, d, du_array[p_to_dim * (dim - 1 - d) + i], du);
            // LCOV_EXCL_STOP
          }
        }
      }
      CeedVectorRestoreArrayRead(x_to, &x_array);
      CeedVectorRestoreArrayRead(du_to, &du_array);
    }

    CeedVectorDestroy(&x_corners);
    CeedVectorDestroy(&x_from);
    CeedVectorDestroy(&x_to);
    CeedVectorDestroy(&u_from);
    CeedVectorDestroy(&u_to);
    CeedVectorDestroy(&du_to);
    CeedBasisDestroy(&basis_from);
    CeedBasisDestroy(&basis_to);
    CeedBasisDestroy(&basis_project);
  }
  CeedDestroy(&ceed);
  return 0;
}
