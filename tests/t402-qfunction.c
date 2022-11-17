/// @file
/// Test viewing of QFunction
/// \test Test viewing of QFunction
#include <ceed.h>

#include "t400-qfunction.h"

int main(int argc, char **argv) {
  Ceed                 ceed;
  CeedQFunction        qf_setup, qf_mass;
  CeedQFunctionContext ctx;

  CeedInit(argv[1], &ceed);

  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "w", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup, "qdata", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "qdata", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", 1, CEED_EVAL_INTERP);

  CeedQFunctionView(qf_setup, stdout);
  CeedQFunctionView(qf_mass, stdout);

  CeedQFunctionContextCreate(ceed, &ctx);
  if (CEED_SCALAR_TYPE == CEED_SCALAR_FP64) {
    CeedScalar ctxData[5] = {1, 2, 3, 4, 5};
    CeedQFunctionContextSetData(ctx, CEED_MEM_HOST, CEED_COPY_VALUES, sizeof(ctxData), &ctxData);
  } else {  // Make context twice as long so the size is the same in output
    CeedScalar ctxData[10] = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
    CeedQFunctionContextSetData(ctx, CEED_MEM_HOST, CEED_COPY_VALUES, sizeof(ctxData), &ctxData);
  }
  CeedQFunctionContextView(ctx, stdout);

  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedQFunctionContextDestroy(&ctx);
  CeedDestroy(&ceed);
  return 0;
}
