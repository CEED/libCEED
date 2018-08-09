// Test grad in multiple dimensions
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
    CeedBasis bxl, bug;
    CeedInt P = 8, Q = 10, Pdim = CeedPowInt(P, dim), Qdim = CeedPowInt(Q, dim),
            Xdim = CeedPowInt(2, dim);
    CeedScalar x[Xdim*dim], ones[dim*Qdim], gtposeones[Pdim];
    CeedScalar xq[Pdim*dim], uq[dim*Qdim], u[Pdim], sum1 = 0, sum2 = 0;

    for (CeedInt i=0; i<dim*Qdim; i++) ones[i] = 1;
    for (CeedInt d=0; d<dim; d++) {
      for (CeedInt i=0; i<Xdim; i++) {
        x[d*Xdim + i] = (i % CeedPowInt(2, dim-d)) / CeedPowInt(2, dim-d-1) ? 1 : -1;
      }
    }

    // Get function values at quadrature points
    CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, P, CEED_GAUSS_LOBATTO, &bxl);
    CeedBasisApply(bxl, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x, xq);
    for (CeedInt i=0; i<Pdim; i++) {
      CeedScalar xx[dim];
      for (CeedInt d=0; d<dim; d++) xx[d] = xq[d*Pdim + i];
      u[i] = Eval(dim, xx);
    }

    // Calculate G u at quadrature points, G' * 1 at dofs
    CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, P, Q, CEED_GAUSS, &bug);
    CeedBasisApply(bug, 1, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, u, uq);
    CeedBasisApply(bug, 1, CEED_TRANSPOSE, CEED_EVAL_GRAD, ones, gtposeones);

    // Check if 1' * G * u = u' * (G' * 1)
    for (CeedInt i=0; i<Pdim; i++) {
      sum1 += gtposeones[i]*u[i];
    }
    for (CeedInt i=0; i<dim*Qdim; i++) {
      sum2 += uq[i];
    }
    if (fabs(sum1 - sum2) > 1e-10) {
      printf("[%d] %f != %f\n", dim, sum1, sum2);
    }

    CeedBasisDestroy(&bxl);
    CeedBasisDestroy(&bug);
  }
  CeedDestroy(&ceed);
  return 0;
}
