/// @file
/// Test grad in multiple dimensions
/// \test Test grad in multiple dimensions
#include <ceed.h>
#include <math.h>

static CeedScalar Eval(CeedInt dim, const CeedScalar x[]) {
  CeedScalar result = tanh(x[0] + 0.1);
  if (dim > 1) result += atan(x[1] + 0.2);
  if (dim > 2) result += exp(-(x[2] + 0.3)*(x[2] + 0.3));
  return result;
}

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);

  for (CeedInt dim=1; dim<=3; dim++) {
    CeedVector X, Xq, U, Uq, Ones, Gtposeones;
    CeedBasis bxl, bug;
    CeedInt P = 8, Q = 10, Pdim = CeedIntPow(P, dim), Qdim = CeedIntPow(Q, dim),
            Xdim = CeedIntPow(2, dim);
    CeedScalar x[Xdim*dim], u[Pdim];
    const CeedScalar *xq, *uq, *gtposeones;
    CeedScalar sum1 = 0, sum2 = 0;

    for (CeedInt d=0; d<dim; d++)
      for (CeedInt i=0; i<Xdim; i++)
        x[d*Xdim + i] = (i % CeedIntPow(2, dim-d)) /
                        CeedIntPow(2, dim-d-1) ? 1 : -1;

    CeedVectorCreate(ceed, Xdim*dim, &X);
    CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);
    CeedVectorCreate(ceed, Pdim*dim, &Xq);
    CeedVectorSetValue(Xq, 0);
    CeedVectorCreate(ceed, Pdim, &U);
    CeedVectorCreate(ceed, Qdim*dim, &Uq);
    CeedVectorSetValue(Uq, 0);
    CeedVectorCreate(ceed, Qdim*dim, &Ones);
    CeedVectorSetValue(Ones, 1);
    CeedVectorCreate(ceed, Pdim, &Gtposeones);
    CeedVectorSetValue(Gtposeones, 0);

    // Get function values at quadrature points
    CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, P,
                                    CEED_GAUSS_LOBATTO, &bxl);
    CeedBasisApply(bxl, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, X, Xq);

    CeedVectorGetArrayRead(Xq, CEED_MEM_HOST, &xq);
    for (CeedInt i=0; i<Pdim; i++) {
      CeedScalar xx[dim];
      for (CeedInt d=0; d<dim; d++)
        xx[d] = xq[d*Pdim + i];
      u[i] = Eval(dim, xx);
    }
    CeedVectorRestoreArrayRead(Xq, &xq);
    CeedVectorSetArray(U, CEED_MEM_HOST, CEED_USE_POINTER, u);

    // Calculate G u at quadrature points, G' * 1 at dofs
    CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, P, Q, CEED_GAUSS, &bug);
    CeedBasisApply(bug, 1, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, U, Uq);
    CeedBasisApply(bug, 1, CEED_TRANSPOSE, CEED_EVAL_GRAD, Ones, Gtposeones);

    // Check if 1' * G * u = u' * (G' * 1)
    CeedVectorGetArrayRead(Gtposeones, CEED_MEM_HOST, &gtposeones);
    CeedVectorGetArrayRead(Uq, CEED_MEM_HOST, &uq);
    for (CeedInt i=0; i<Pdim; i++)
      sum1 += gtposeones[i]*u[i];
    for (CeedInt i=0; i<dim*Qdim; i++)
      sum2 += uq[i];
    CeedVectorRestoreArrayRead(Gtposeones, &gtposeones);
    CeedVectorRestoreArrayRead(Uq, &uq);
    if (fabs(sum1 - sum2) > 1e-10)
      // LCOV_EXCL_START
      printf("[%d] %f != %f\n", dim, sum1, sum2);
    // LCOV_EXCL_STOP

    CeedVectorDestroy(&X);
    CeedVectorDestroy(&Xq);
    CeedVectorDestroy(&U);
    CeedVectorDestroy(&Uq);
    CeedVectorDestroy(&Ones);
    CeedVectorDestroy(&Gtposeones);
    CeedBasisDestroy(&bxl);
    CeedBasisDestroy(&bug);
  }
  CeedDestroy(&ceed);
  return 0;
}
