/// @file
/// Test read access for QFunctionContext data
/// \test Test read access for QFunctionContext data
#include <ceed.h>
#include <math.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed                 ceed;
  CeedQFunctionContext ctx;
  CeedScalar           ctx_data[5] = {1, 2, 3, 4, 5}, *ctx_data_copy;

  CeedInit(argv[1], &ceed);

  CeedQFunctionContextCreate(ceed, &ctx);
  CeedQFunctionContextSetData(ctx, CEED_MEM_HOST, CEED_COPY_VALUES, sizeof(ctx_data), &ctx_data);

  // Get data access
  CeedQFunctionContextGetDataRead(ctx, CEED_MEM_HOST, &ctx_data_copy);
  if (ctx_data_copy[4] != 5) {
    // LCOV_EXCL_START
    printf("error reading data: %f != 5.0\n", ctx_data_copy[4]);
    // LCOV_EXCL_STOP
  }
  CeedQFunctionContextRestoreDataRead(ctx, &ctx_data_copy);

  // Check access protection - should error
  CeedQFunctionContextGetDataRead(ctx, CEED_MEM_HOST, &ctx_data_copy);
  CeedQFunctionContextGetData(ctx, CEED_MEM_HOST, &ctx_data_copy);
  CeedQFunctionContextRestoreData(ctx, &ctx_data_copy);
  CeedQFunctionContextRestoreDataRead(ctx, &ctx_data_copy);

  CeedQFunctionContextDestroy(&ctx);
  CeedDestroy(&ceed);
  return 0;
}
