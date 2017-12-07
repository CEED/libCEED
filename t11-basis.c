// Test square Gauss Lobatto interp1d is identity
#include <ceed.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char **argv)
{
   Ceed ceed;
   CeedBasis b;
   int i, dim = 2, P1d = 3, Q1d = 4, len = (int)(pow((double)(Q1d), dim) + 0.4);
   CeedScalar u[len], v[len];

   CeedInit("/cpu/self", &ceed);
   for (i = 0; i < len; i++)
   {
      u[i] = 1.0;
   }
   CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, P1d, Q1d, CEED_GAUSS_LOBATTO, &b);
   CeedBasisApply(b, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, u, v);
   for (i = 0; i < len; i++)
   {
      if (fabs(v[i] - 1.) > 1e-15) { printf("v[%d] = %f != 1.\n", i, v[i]); }
   }
   CeedBasisDestroy(&b);
   CeedDestroy(&ceed);
   return 0;
}
