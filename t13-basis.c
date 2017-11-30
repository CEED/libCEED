// Test interpolation in multiple dimensions
#include <feme.h>
#include <math.h>

static FemeScalar Eval(FemeInt dim, const FemeScalar x[]) {
  FemeScalar result = 1, center = 0.1;
  for (FemeInt d=0; d<dim; d++) {
    result *= tanh(x[d] - center);
    center += 0.1;
  }
  return result;
}

int main(int argc, char **argv) {
  Feme feme;

  FemeInit("/cpu/self", &feme);
  for (FemeInt dim=1; dim<=3; dim++) {
    FemeBasis bxl, bul, bxg, bug;
    FemeInt Q = 10, Qdim = FemePowInt(Q, dim);
    FemeScalar x[dim][FemePowInt(2, dim)];
    FemeScalar xq[dim][Qdim], uq[Qdim], u[Qdim];

    for (FemeInt d=0; d<dim; d++) {
      for (FemeInt i=0; i<FemePowInt(2, dim); i++) {
        x[d][i] = (i % FemePowInt(2, dim-d)) / FemePowInt(2, dim-d-1) ? 1 : -1;
      }
    }
    FemeBasisCreateTensorH1Lagrange(feme, dim, 1, 1, Q, FEME_GAUSS_LOBATTO, &bxl);
    FemeBasisCreateTensorH1Lagrange(feme, dim, 1, Q-1, Q, FEME_GAUSS_LOBATTO, &bul);
    for (FemeInt d=0; d<dim; d++) {
      FemeBasisApply(bxl, FEME_NOTRANSPOSE, FEME_EVAL_INTERP, x[d], xq[d]);
    }
    for (FemeInt i=0; i<Qdim; i++) {
      FemeScalar xx[dim];
      for (FemeInt d=0; d<dim; d++) xx[d] = xq[d][i];
      uq[i] = Eval(dim, xx);
    }

    FemeBasisApply(bul, FEME_TRANSPOSE, FEME_EVAL_INTERP, uq, u); // Should be identity

    FemeBasisCreateTensorH1Lagrange(feme, dim, 1, 1, Q, FEME_GAUSS, &bxg);
    FemeBasisCreateTensorH1Lagrange(feme, dim, 1, Q-1, Q, FEME_GAUSS, &bug);
    for (FemeInt d=0; d<dim; d++) {
      FemeBasisApply(bxg, FEME_NOTRANSPOSE, FEME_EVAL_INTERP, x[d], xq[d]);
    }
    FemeBasisApply(bug, FEME_NOTRANSPOSE, FEME_EVAL_INTERP, u, uq);
    for (FemeInt i=0; i<Qdim; i++) {
      FemeScalar xx[dim];
      for (FemeInt d=0; d<dim; d++) xx[d] = xq[d][i];
      FemeScalar fx = Eval(dim, xx);
      if (fabs(uq[i] - fx) > 1e-4) {
        printf("[%d] %f != %f=f(%f", dim, uq[i], fx, xx[0]);
        for (FemeInt d=1; d<dim; d++) printf(",%f", xx[d]);
        puts(")");
      }
    }

    FemeBasisDestroy(&bxl);
    FemeBasisDestroy(&bul);
    FemeBasisDestroy(&bxg);
    FemeBasisDestroy(&bug);
  }
  FemeDestroy(&feme);
  return 0;
}
