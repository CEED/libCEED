/// @file
/// Test creation, copying, and destruction for QFunction and QFunctionContext
/// \test Test creation, copying, and destruction for QFunction and QFunctionContext
#include <ceed.h>

#include "t400-qfunction.h"

int main(int argc, char **argv) {
  Ceed                 ceed;
  CeedQFunction        qf, qf_2;
  CeedQFunctionContext ctx, ctx_2;

  CeedInit(argv[1], &ceed);

  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf);
  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_2);

  CeedQFunctionReferenceCopy(qf, &qf_2);  // This destroys the previous qf_2
  if (qf != qf_2) printf("Error copying CeedQFunction reference\n");

  CeedQFunctionContextCreate(ceed, &ctx);
  CeedQFunctionContextCreate(ceed, &ctx_2);

  CeedQFunctionContextReferenceCopy(ctx, &ctx_2);
  if (ctx != ctx_2) printf("Error copying CeedQFunctionContext reference\n");

  CeedQFunctionDestroy(&qf);
  CeedQFunctionDestroy(&qf_2);
  CeedQFunctionContextDestroy(&ctx);
  CeedQFunctionContextDestroy(&ctx_2);
  CeedDestroy(&ceed);
  return 0;
}
