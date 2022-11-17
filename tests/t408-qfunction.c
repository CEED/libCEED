/// @file
/// Test read access for QFunctionContext data
/// \test Test read access for QFunctionContext data
#include <ceed.h>
#include <math.h>

int main(int argc, char **argv) {
  Ceed                 ceed;
  CeedQFunctionContext ctx;
  CeedScalar           ctxData[5] = {1, 2, 3, 4, 5}, *ctxDataCopy;

  CeedInit(argv[1], &ceed);

  CeedQFunctionContextCreate(ceed, &ctx);
  CeedQFunctionContextSetData(ctx, CEED_MEM_HOST, CEED_COPY_VALUES, sizeof(ctxData), &ctxData);

  // Get data access
  CeedQFunctionContextGetDataRead(ctx, CEED_MEM_HOST, &ctxDataCopy);
  if (ctxDataCopy[4] != 5)
    // LCOV_EXCL_START
    printf("error reading data: %f != 5.0\n", ctxDataCopy[4]);
  // LCOV_EXCL_STOP
  CeedQFunctionContextRestoreDataRead(ctx, &ctxDataCopy);

  // Check access protection - should error
  CeedQFunctionContextGetDataRead(ctx, CEED_MEM_HOST, &ctxDataCopy);
  CeedQFunctionContextGetData(ctx, CEED_MEM_HOST, &ctxDataCopy);
  CeedQFunctionContextRestoreData(ctx, &ctxDataCopy);
  CeedQFunctionContextRestoreDataRead(ctx, &ctxDataCopy);

  CeedQFunctionContextDestroy(&ctx);
  CeedDestroy(&ceed);
  return 0;
}
