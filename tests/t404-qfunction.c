/// @file
/// Test creation, setting, and taking data for QFunctionContext
/// \test Test creation, setting, and taking data for QFunctionContext
#include <ceed.h>
#include <math.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed                 ceed;
  CeedQFunctionContext ctx;
  CeedScalar           ctx_data[5] = {1, 2, 3, 4, 5}, *ctx_data_copy;

  CeedInit(argv[1], &ceed);

  // Set borrowed pointer
  CeedQFunctionContextCreate(ceed, &ctx);
  CeedQFunctionContextSetData(ctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(ctx_data), &ctx_data);

  // Update borrowed pointer
  CeedQFunctionContextGetData(ctx, CEED_MEM_HOST, &ctx_data_copy);
  ctx_data_copy[4] = 6;
  CeedQFunctionContextRestoreData(ctx, &ctx_data_copy);
  if (ctx_data[4] != 6) printf("error modifying data: %f != 6.0\n", ctx_data[4]);

  // Take back borrowed pointer
  CeedQFunctionContextTakeData(ctx, CEED_MEM_HOST, &ctx_data_copy);
  if (ctx_data_copy[4] != 6) printf("error accessing borrowed data: %f != 6.0\n", ctx_data_copy[4]);

  // Set copied data
  ctx_data[4] = 6;
  CeedQFunctionContextSetData(ctx, CEED_MEM_HOST, CEED_COPY_VALUES, sizeof(ctx_data), &ctx_data);

  // Check copied data
  CeedQFunctionContextGetData(ctx, CEED_MEM_HOST, &ctx_data_copy);
  if (ctx_data_copy[4] != 6) printf("error accessing copied data: %f != 6.0\n", ctx_data_copy[4]);
  CeedQFunctionContextRestoreData(ctx, &ctx_data_copy);

  CeedQFunctionContextDestroy(&ctx);
  CeedDestroy(&ceed);
  return 0;
}
