// Test interpolation in multiple dimensions
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

  CeedInit("/cpu/self", &ceed);
  for (CeedInt dim=1; dim<=3; dim++) {
    CeedBasis bxl, bul, bxg, bug;
    CeedInt Q = 10, Qdim = CeedPowInt(Q, dim), Xdim = CeedPowInt(2, dim);
    CeedScalar x[Xdim*dim];
    CeedScalar xq[Qdim*dim], uq[Qdim], u[Qdim];

    for (CeedInt d=0; d<dim; d++) {
      for (CeedInt i=0; i<Xdim; i++) {
        x[d*Xdim + i] = (i % CeedPowInt(2, dim-d)) / CeedPowInt(2, dim-d-1) ? 1 : -1;
      }
    }
    CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 1, Q, CEED_GAUSS_LOBATTO, &bxl);
    CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, Q-1, Q, CEED_GAUSS_LOBATTO, &bul);
    CeedBasisApply(bxl, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x, xq);
    for (CeedInt i=0; i<Qdim; i++) {
      CeedScalar xx[dim];
      for (CeedInt d=0; d<dim; d++) xx[d] = xq[d*Qdim + i];
      uq[i] = Eval(dim, xx);
    }

    // This operation is the identity because the quadrature is collocated
    CeedBasisApply(bul, CEED_TRANSPOSE, CEED_EVAL_INTERP, uq, u);

    CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 1, Q, CEED_GAUSS, &bxg);
    CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, Q-1, Q, CEED_GAUSS, &bug);
    CeedBasisApply(bxg, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x, xq);
    CeedBasisApply(bug, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, u, uq);
    for (CeedInt i=0; i<Qdim; i++) {
      CeedScalar xx[dim];
      for (CeedInt d=0; d<dim; d++) xx[d] = xq[d*Qdim + i];
      CeedScalar fx = Eval(dim, xx);
      if (!(fabs(uq[i] - fx) < 1e-4)) {
        printf("[%d] %f != %f=f(%f", dim, uq[i], fx, xx[0]);
        for (CeedInt d=1; d<dim; d++) printf(",%f", xx[d]);
        puts(")");
      }
    }

    CeedBasisDestroy(&bxl);
    CeedBasisDestroy(&bul);
    CeedBasisDestroy(&bxg);
    CeedBasisDestroy(&bug);
  }
  CeedDestroy(&ceed);
  return 0;
}
