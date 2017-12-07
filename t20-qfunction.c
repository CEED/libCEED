// Test qfunction evaluation
#include <ceed.h>

static int setup(void *ctx, void *qdata, CeedInt Q, const CeedScalar *const *u,
                 CeedScalar *const *v)
{
   CeedScalar *w = qdata;
   for (CeedInt i=0; i<Q; i++)
   {
      w[i] = 1.0;
   }
   return 0;
}

static int mass(void *ctx, void *qdata, CeedInt Q, const CeedScalar *const *u,
                CeedScalar *const *v)
{
   const CeedScalar *w = qdata;
   for (CeedInt i=0; i<Q; i++)
   {
      v[0][i] = w[i] * u[0][i];
   }
   return 0;
}

int main(int argc, char **argv)
{
   Ceed ceed;
   CeedQFunction qf_setup, qf_mass;

   CeedInit("/cpu/self", &ceed);
   CeedQFunctionCreateInterior(ceed, 1, 1, sizeof(CeedScalar), CEED_EVAL_NONE,
                               CEED_EVAL_NONE, setup, __FILE__ ":setup", &qf_setup);
   CeedQFunctionCreateInterior(ceed, 1, 1, sizeof(CeedScalar), CEED_EVAL_INTERP,
                               CEED_EVAL_INTERP, mass, __FILE__ ":mass", &qf_mass);
   CeedQFunctionDestroy(&qf_setup);
   CeedQFunctionDestroy(&qf_mass);
   CeedDestroy(&ceed);
   return 0;
}
