/// @file
/// Test interpolation in multiple dimensions
/// \test Test interpolation in multiple dimensions
#include <ceed.h>
#include <math.h>

static CeedScalar Eval(CeedInt dim, const CeedScalar x[]) {
  CeedScalar result = 1, center = 0.1;
  for (CeedInt d=0; d<dim; d++) {
    result *= tanh(x[d] - center);
    center += 0.1;
  }
  return result;
}

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);
  for (CeedInt dim=1; dim<=3; dim++) {
    CeedVector X, Xq, U, Uq;
    CeedBasis bxl, bul, bxg, bug;
    CeedInt Q = 10, Qdim = CeedIntPow(Q, dim), Xdim = CeedIntPow(2, dim);
    CeedScalar x[Xdim*dim];
    const CeedScalar *xq, *u;
    CeedScalar uq[Qdim];

    for (CeedInt d=0; d<dim; d++) {
      for (CeedInt i=0; i<Xdim; i++) {
        x[d*Xdim + i] = (i % CeedIntPow(2, dim-d)) / CeedIntPow(2, dim-d-1) ? 1 : -1;
      }
    }

    CeedVectorCreate(ceed, Xdim*dim, &X);
    CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, (CeedScalar *)&x);
    CeedVectorCreate(ceed, Qdim*dim, &Xq);
    CeedVectorSetValue(Xq, 0);
    CeedVectorCreate(ceed, Qdim, &U);
    CeedVectorSetValue(U, 0);
    CeedVectorCreate(ceed, Qdim, &Uq);

    CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, Q, CEED_GAUSS_LOBATTO, &bxl);
    CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, Q, Q, CEED_GAUSS_LOBATTO, &bul);

    CeedBasisApply(bxl, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, X, Xq);

    CeedVectorGetArrayRead(Xq, CEED_MEM_HOST, &xq);
    for (CeedInt i=0; i<Qdim; i++) {
      CeedScalar xx[dim];
      for (CeedInt d=0; d<dim; d++) xx[d] = xq[d*Qdim + i];
      uq[i] = Eval(dim, xx);
    }
    CeedVectorRestoreArrayRead(Xq, &xq);
    CeedVectorSetArray(Uq, CEED_MEM_HOST, CEED_USE_POINTER, (CeedScalar *)&uq);

    // This operation is the identity because the quadrature is collocated
    CeedBasisApply(bul, 1, CEED_TRANSPOSE, CEED_EVAL_INTERP, Uq, U);

    CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, Q, CEED_GAUSS, &bxg);
    CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, Q, Q, CEED_GAUSS, &bug);

    CeedBasisApply(bxg, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, X, Xq);
    CeedBasisApply(bug, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, U, Uq);

    CeedVectorGetArrayRead(Xq, CEED_MEM_HOST, &xq);
    CeedVectorGetArrayRead(Uq, CEED_MEM_HOST, &u);
    for (CeedInt i=0; i<Qdim; i++) {
      CeedScalar xx[dim];
      for (CeedInt d=0; d<dim; d++) xx[d] = xq[d*Qdim + i];
      CeedScalar fx = Eval(dim, xx);
      if (fabs(u[i] - fx) > 1e-4) {
        // LCOV_EXCL_START
        printf("[%d] %f != %f=f(%f", dim, u[i], fx, xx[0]);
        for (CeedInt d=1; d<dim; d++) printf(",%f", xx[d]);
        puts(")");
        // LCOV_EXCL_STOP
      }
    }
    CeedVectorRestoreArrayRead(Xq, &xq);
    CeedVectorRestoreArrayRead(Uq, &u);

    CeedVectorDestroy(&X);
    CeedVectorDestroy(&Xq);
    CeedVectorDestroy(&U);
    CeedVectorDestroy(&Uq);
    CeedBasisDestroy(&bxl);
    CeedBasisDestroy(&bul);
    CeedBasisDestroy(&bxg);
    CeedBasisDestroy(&bug);
  }
  CeedDestroy(&ceed);
  return 0;
}
